"""
Synthetic flare component for signal-injection studies.

Defines :class:`Flare`, a compact spatial + temporal cluster of cosmic-ray
events used to overlay a localised signal on an isotropic background.

Parameters carried on the instance:

- ``n_flare`` — number of flare events to inject (``>= 1``),
- ``duration`` — temporal length of the flare; the flare's start time is
  drawn uniformly in ``[t0, tf - duration]`` at generation time,
- ``centre`` — sky position ``[RA_deg, Dec_deg]`` of the flare,
- ``exposure_model`` — the same :class:`~spacetimecorr.exposure.ExposureModel`
  used for the background, providing the Bernoulli detection thinning and
  directional-exposure evaluation.

The spatial spread (``sigma``, Rayleigh radius of the Gaussian cluster in
the tangent plane) is supplied at generation time to
:meth:`Flare.generate_in_window`, not at construction.

The class is constructed independently of an
:class:`~spacetimecorr.event_sample.EventSample`. Use
:meth:`Flare.generate_in_window` to populate the flare's own arrays,
and :meth:`EventSample.inject_flare` to overlay it on a sample.
"""

from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.units import Quantity
from astropy.coordinates import SkyCoord

from .skywindow import SkyWindow
from .exposure import ExposureModel


class Flare:
    """
    Synthetic flare component (compact spatial + temporal cluster of events).

    A ``Flare`` is constructed independently of an :class:`EventSample`. It
    stores the parameters needed to generate a compact set of events in time
    and sky coordinates within a fixed observation interval ``[t0, tf]``,
    and exposes high-level helpers to populate them
    (:meth:`generate_in_window`) or to inject them into an existing sample
    (via :meth:`EventSample.inject_flare`).

    Parameters
    ----------
    n_flare : int
        Number of flare events to generate. Must be ``>= 1``: zero-event
        flares are forbidden by construction (filter at the caller, e.g.
        ``if n > 0: Flare(n, ...)``).
    duration : astropy.units.Quantity
        Flare duration with time units (e.g. ``30 * u.day``). Must be
        positive and not exceed ``tf - t0``.
    t0, tf : astropy.time.Time
        Start and end of the observation interval. Must satisfy ``tf > t0``.
    centre : array-like of shape (2,)
        Central sky position of the flare ``[RA_deg, Dec_deg]``.
    exposure_model : ExposureModel
        Directional exposure model used for thinning and exposure
        evaluation.
    rng : numpy.random.Generator
        Random stream used to draw flare coordinates and times.

    Notes
    -----
    Generated arrays (``ra``, ``dec``, ``time``, ``exposure``) are
    initialised to ``None`` and populated by the ``generate_*`` /
    ``compute_directional_exposure`` methods, or in one go by
    :meth:`generate_in_window`.
    """

    def __init__(
        self,
        n_flare: int,
        duration: Quantity,
        t0: Time,
        tf: Time,
        centre: np.ndarray,
        exposure_model: "ExposureModel",
        rng: np.random.Generator,
    ):
        
        if not isinstance(n_flare, (int, np.integer)) or isinstance(n_flare, bool):
            raise TypeError("n_flare must be a positive integer.")
        if n_flare < 1:
            # Zero-event flares carry no signal and force every downstream
            # consumer to handle a degenerate case (NaN exposures, empty
            # arrays, ambiguous `has_flare` semantics, etc.). They should be
            # filtered out by the caller, e.g. `if n > 0: Flare(n, ...)`.
            raise ValueError("n_flare must be >= 1; do not construct a Flare with 0 events.")

        if not isinstance(duration, u.Quantity):
            raise TypeError(
                "duration must be an astropy.units.Quantity with time units "
                "(e.g., 10*u.s)."
            )
        if not duration.unit.is_equivalent(u.s):
            raise ValueError("duration must have time units (equivalent to seconds).")
        if duration <= 0 * u.s:
            raise ValueError("duration must be > 0.")

        if not isinstance(t0, Time):
            raise TypeError("t0 must be an astropy.time.Time object.")
        if not isinstance(tf, Time):
            raise TypeError("tf must be an astropy.time.Time object.")
        if tf <= t0:
            raise ValueError("tf must be later than t0.")

        centre = np.asarray(centre, dtype=float)
        if centre.shape != (2,):
            raise ValueError("centre must be a length-2 array: [RA, Dec] in degrees.")
        ra_c, dec_c = centre
        if not (0.0 <= ra_c < 360.0):
            raise ValueError("RA must be in [0, 360).")
        if not (-90.0 <= dec_c <= 90.0):
            raise ValueError("Dec must be in [-90, 90].")
        
        if not isinstance(exposure_model, ExposureModel):
            raise TypeError("exposure must be an instance of ExposureModel.")

        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be a numpy.random.Generator. "
                "Obtain one from RNGManager.get(name) and pass it here."
            )
        
        self._T_obs_sec = (tf - t0).to_value(u.s)
        duration_sec = duration.to_value(u.s)

        if duration_sec > self._T_obs_sec:
            raise ValueError("flare duration cannot exceed the observation interval.")

        self.rng = rng

        self.n_flare = int(n_flare)
        self.duration = duration.to(u.s)         # Quantity, preserved unit
        self.duration_sec = float(duration_sec)  # cached float for hot paths

        self.t0 = t0
        self.tf = tf
        self.centre = centre
        self.exposure_model = exposure_model

        self.spatial_profile: str | None = None
        self.time_profile: str | None = None

        # Generated data
        self.ra: np.ndarray | None = None
        self.dec: np.ndarray | None = None
        self.time: Time | None = None
        self.exposure: np.ndarray | None = None

    # -------------------------------------------------------------------------
    # State-check properties
    # -------------------------------------------------------------------------

    @property
    def has_coordinates(self) -> bool:
        """Return True if flare coordinates have been generated."""
        return self.ra is not None and self.dec is not None

    # -------------------------------------------------------------------------
    # Low-level sampling / evaluation methods
    # -------------------------------------------------------------------------

    def _draw_flare_start(self, max_tries: int = 1000) -> Time:
        """
        Draw a flare start time uniformly in ``[t0, tf - duration]``, keeping
        only intervals whose centre is inside the FoV at the start *or* the
        end of the flare.

        Parameters
        ----------
        max_tries : int
            Maximum number of rejection draws before giving up.

        Notes
        -----
        When ``duration == tf - t0`` (the spatial-only regime exercised by
        the sensitivity study) ``latest_start`` is exactly 0 and the start
        time collapses to ``t0`` deterministically. In that regime every draw
        would reproduce the identical full-window interval, so the FoV
        rejection is skipped and ``t0`` is returned directly.

        Raises
        ------
        RuntimeError
            If no interval with a visible endpoint is found within
            ``max_tries`` draws (e.g. a centre that never rises above the
            acceptance cut for this observatory).
        """
        latest_start = self._T_obs_sec - self.duration_sec

        if latest_start == 0.0:
            return self.t0

        for _ in range(max_tries):
            start_offset_sec = self.rng.uniform(0.0, latest_start)
            endpoints = self.t0 + TimeDelta(
                np.array([start_offset_sec, start_offset_sec + self.duration_sec]),
                format="sec",
            )
            if np.any(self._check_time_in_FoV(endpoints)):
                return endpoints[0]

        raise RuntimeError(
            f"Could not place the flare centre {self.centre} inside the FoV at "
            f"either endpoint within {max_tries} draws; the direction may never "
            "rise above the acceptance cut for this observatory."
        )

    def _check_time_in_FoV(self, time: Time) -> np.ndarray:
        """
        Boolean mask, ``True`` where the flare centre is inside the FoV at
        ``time``.

        The centre is considered visible when the observatory's instantaneous
        geometric acceptance toward it is non-zero, i.e. its zenith angle is
        below ``theta_max``.

        Parameters
        ----------
        time : astropy.time.Time
            Scalar or array of evaluation times.

        Returns
        -------
        numpy.ndarray
            Boolean array matching the shape of ``time``.
        """
        acceptance = self.exposure_model.instantaneous_acceptance(
            t=time, centre=self.centre
        )
        return np.asarray(acceptance) > 0.0

    def _sample_gaussian_cluster(
        self,
        n_flare: int,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample `n_flare` equatorial coordinates from a Gaussian cluster
        on the sphere around `self.centre`.

        Parameters
        ----------
        n_flare : int
            Number of coordinates to draw.
        sigma : float
            Width of the cluster in degrees.

        Returns
        -------
        ra, dec : tuple[np.ndarray, np.ndarray]
            Arrays of right ascension and declination in degrees.
        """

        center = SkyCoord(
            ra=self.centre[0] * u.deg,
            dec=self.centre[1] * u.deg,
            frame="icrs",
        )

        local_theta = self.rng.rayleigh(scale=sigma, size=n_flare) * u.deg
        local_azimuth = self.rng.uniform(0.0, 2.0 * np.pi, size=n_flare) * u.rad

        event_coords = center.directional_offset_by(local_azimuth, local_theta)

        ra = event_coords.ra.deg.astype(float, copy=False)
        dec = event_coords.dec.deg.astype(float, copy=False)
        return ra, dec
    
    def _sample_uniform_times(self, n_flare: int, start: Time | None = None) -> Time:
        """
        Sample `n_flare` times uniformly inside one flare interval.

        If `start` is not given, a flare start is drawn uniformly in
        [self.t0, self.tf - self.duration_sec].
        """
        if start is None:
            start = self._draw_flare_start()

        # Draw time offsets inside the flare duration
        offsets_sec = self.rng.uniform(0.0, self.duration_sec, size=n_flare)

        # Convert offsets into absolute event times
        return start + TimeDelta(offsets_sec, format="sec")
    
    def _evaluate_directional_exposure(
        self,
        time: Time,
        direction: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate directional exposure at the provided times and direction.

        Parameters
        ----------
        time : astropy.time.Time
            Event times.
        direction : np.ndarray
            Sky direction [RA, Dec] in degrees.

        Returns
        -------
        np.ndarray
            Directional exposure values.
        """

        return np.asarray(
            self.exposure_model.cumulative_directional_exposure(time, direction),
            dtype=float,
        )
    
    # -------------------------------------------------------------------------
    # Public population methods
    # -------------------------------------------------------------------------

    def compute_directional_exposure(self, direction: np.ndarray) -> None:
        """
        Compute directional exposure for the generated flare times at a given direction.

        Parameters
        ----------
        direction : np.ndarray
            Sky direction [RA, Dec] in degrees at which the exposure is evaluated.
        """
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (2,):
            raise ValueError("direction must be a length-2 array: [RA, Dec] in degrees.")

        if self.time is None:
            raise ValueError("Flare times are not set; populate `self.time` first.")

        self.exposure = self._evaluate_directional_exposure(self.time, direction)

    # -------------------------------------------------------------------------
    # High-level realization method
    # -------------------------------------------------------------------------

    def generate_in_window(
        self,
        window: SkyWindow,
        sigma: float,
        efficiency = None,
    ) -> Flare:
        """
        Generate a flare realisation inside a sky window and store it on ``self``.

        The procedure is:

        1. draw a single flare start time uniformly in ``[t0, tf - duration]``,
        2. iterate by batches:
           a. sample spatial candidates from a Gaussian cluster around
              ``self.centre`` with width ``sigma``,
           b. keep only candidates inside ``window``,
           c. draw uniform candidate times within the flare interval,
           d. apply Bernoulli detection thinning via
              :meth:`ExposureModel.acceptance_mask`,
        3. accumulate accepted events until exactly ``self.n_flare`` are
           reached, then trim,
        4. compute directional exposure values for the kept times via
           :meth:`compute_directional_exposure` evaluated at ``window.centre``.

        Parameters
        ----------
        window : SkyWindow
            Sky window used for spatial selection and as the reference
            direction for exposure evaluation.
        sigma : float
            Standard deviation (in degrees) of the Gaussian spatial profile
            on the sphere. The implementation uses a small-angle
            approximation (Rayleigh radius in the tangent plane); errors
            grow as ``sigma^2 / 24`` and are negligible for ``sigma``
            of a few degrees.
        efficiency : callable or None, optional
            Optional time-dependent efficiency in ``[0, 1]`` forwarded to
            the exposure model.

        Raises
        ------
        TypeError
            If ``window`` is not a :class:`SkyWindow`.
        ValueError
            If ``sigma <= 0``.
        RuntimeError
            If the rejection loop cannot reach ``self.n_flare`` accepted
            events within ``1000 * self.n_flare`` candidate draws (e.g.
            very small window combined with very low acceptance).

        Notes
        -----
        Sets ``self.ra``, ``self.dec``, ``self.time``, ``self.exposure``
        and tags ``self.spatial_profile = "gaussian_spherical"``,
        ``self.time_profile = "uniform_thinned"``.

        Directional exposure for each generated event is evaluated at
        ``window.centre`` rather than at the event's own ``(ra, dec)``.
        This matches the convention used by
        :meth:`EventSample.assign_directional_exposure` for in-window
        samples: every event inside the cap is treated as having the
        same directional exposure as the cap centre, which is a good
        approximation for the small radii relevant to this analysis.
        """

        if not isinstance(window, SkyWindow):
            raise TypeError("window must be an instance of SkyWindow.")
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")

        target = self.n_flare

        ra_acc: list[np.ndarray] = []
        dec_acc: list[np.ndarray] = []
        time_acc: list[Time] = []

        n_kept = 0
        n_drawn = 0
        max_draws = 1000 * target

        # 1. Fix the Flare start time for this realization
        # All candidate events for this flare instance happen in [start, start + duration]
        flare_start = self._draw_flare_start()

        while n_kept < target:
            remaining = target - n_kept
            current_batch = max(200, 10 * remaining) # Avoid very low values of current_batch
            n_drawn += current_batch

            if n_drawn > max_draws:
                raise RuntimeError(
                    "Could not generate enough events inside the window before "
                    "reaching max_draws = 1000 * self.n_flare."
                )

            # --- Step 1: Spatial Sampling ---
            ra_batch, dec_batch = self._sample_gaussian_cluster(current_batch, sigma)
            spatial_mask = window.contains(ra_batch, dec_batch)
            
            # Filter batch to only those in window
            if not np.any(spatial_mask):
                continue

            ra_cand = ra_batch[spatial_mask]
            dec_cand = dec_batch[spatial_mask]

            # --- Step 2: Temporal Sampling + Exposure Thinning ---
            times_cand = self._sample_uniform_times(ra_cand.size, start=flare_start)

            # --- Step 3: Exposure Thinning ---
            _, detection_mask = self.exposure_model.detect_times(
                times_cand,
                window.centre,
                efficiency=efficiency,
                return_mask=True,
            )

            if not np.any(detection_mask):
                continue

            ra_acc.append(ra_cand[detection_mask])
            dec_acc.append(dec_cand[detection_mask])
            time_acc.append(times_cand[detection_mask])
            n_kept += int(np.count_nonzero(detection_mask))


        # Clean up and slice to exact target
        self.ra = np.concatenate(ra_acc)[:target]
        self.dec = np.concatenate(dec_acc)[:target]
        self.time = Time(
            np.concatenate([t.jd for t in time_acc])[:target],
            format="jd",
            scale=flare_start.scale,
        )
        
        self.spatial_profile = "gaussian_spherical"
        self.time_profile = "uniform_thinned"

        # Exposure attached to the final accepted events
        self.compute_directional_exposure(window.centre)

    @property
    def flare_type(self) -> str:
        """
        Compact string label describing the flare model.

        Format ``"{spatial_profile}-{time_profile}"`` once both have been
        set (e.g. ``"gaussian_spherical-uniform_thinned"``); returns
        ``"undefined_flare"`` if either profile is missing.
        """
        if self.spatial_profile is None or self.time_profile is None:
            return "undefined_flare"

        return f"{self.spatial_profile}-{self.time_profile}"