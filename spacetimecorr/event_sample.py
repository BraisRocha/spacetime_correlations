"""
Event-level data container and sampling factories.

Defines :class:`EventSample`, the main container for a set of cosmic-ray
events used throughout the pipeline. Each sample stores per-event arrival
directions (RA/Dec, in degrees), optional directional-exposure values,
and the exposure-weighted full-sky population (``n_total``) the sample
was derived from. Per-event arrival times are also carried but are not
consumed by the current analysis: the Lambda estimator operates on
directional-exposure spacings, not on times. The time arrays are kept
for future extensions.

End-user code constructs samples through the factory classmethods rather
than ``__init__``:

- :meth:`EventSample.full_sky` — isotropic background over the full sky.
- :meth:`EventSample.in_window` — Poisson-drawn events restricted to a
  :class:`~spacetimecorr.skywindow.SkyWindow`, with exposure weighting
  threaded through.

Flare injection (:meth:`EventSample.inject_flare`) and exposure
assignment (``assign_*``) operate in place on an existing sample.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time
from astropy.coordinates import EarthLocation

if TYPE_CHECKING:
    from .skywindow import SkyWindow
    from .exposure import ExposureModel
    from .flare import Flare


class EventSample:
    """Generate and store samples of events in equatorial coordinates.

    Parameters
    ----------
    n_sample : int
        Number of events the sample will hold (length of ``ra``/``dec``).
        Must be non-negative.
    n_total : int
        Equivalent full-sky event count this sample is derived from.
        For full-sky samples ``n_total == n_sample``. For per-window
        samples ``n_total`` is the parent full-sky population whose
        exposure-weighted fraction inside the window produced
        ``expected_n`` and, by Poisson draw, ``n_sample``.
    t0 : astropy.time.Time
        Observation start time.
    tf : astropy.time.Time
        Observation end time (must be later than ``t0``).
    rng : numpy.random.Generator
        Random generator used for reproducible sampling
        (e.g. obtained from ``RNGManager.get(name)``).

    Notes
    -----
    - ``__init__`` is the low-level constructor; it allocates state but
      does **not** draw coordinates. End-user code should normally use
      one of the factory classmethods, :meth:`full_sky` or
      :meth:`in_window`, which return a fully-formed sample with
      coordinates and ``expected_n`` set.
    - All event coordinates are stored in degrees.
    - Optional state (exposure values, flare bookkeeping) is set lazily
      by the corresponding ``assign_*`` / ``inject_flare`` methods and
      inspected via the ``has_*`` properties.
    """

    # -------------------------------------------------------------------------
    # Construction and basic initialization
    # -------------------------------------------------------------------------

    @staticmethod
    def _healpy():
        return importlib.import_module("healpy")

    @staticmethod
    def _pyplot():
        return importlib.import_module("matplotlib.pyplot")

    def __init__(
        self,
        n_sample: int,
        n_total: int,
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
    ):
        # ---- Input validation ------------------------------------------------
        if not isinstance(n_sample, int) or isinstance(n_sampleO, bool):
            raise TypeError("n_sample must be an integer.")
        if n_sample < 0:
            raise ValueError("n_sample must be non-negative.")

        if not isinstance(n_total, int) or isinstance(n_total, bool):
            raise TypeError("n_total must be an integer.")
        if n_total < 0:
            raise ValueError("n_total must be non-negative.")

        if not isinstance(t0, Time) or not isinstance(tf, Time):
            raise TypeError("t0 and tf must be astropy.time.Time objects.")

        if tf <= t0:
            raise ValueError("tf must be strictly later than t0.")

        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be a numpy.random.Generator. "
                "Obtain one from RNGManager.get(name) and pass it here."
            )

        # ---- Core configuration ----------------------------------------------
        self.rng = rng
        self.n_sample = int(n_sample)
        self.n_total = int(n_total)
        self.expected_n: float | None = None
        self.t0 = t0
        self.tf = tf

        # ---- Sample metadata / state labels ----------------------------------
        self.spatial_type: str | None = None
        self.exposure_type: str | None = None
        self.flare_type: str | None = None

        # ---- Generation context (set by per-window factory) ------------------
        self.window: "SkyWindow | None" = None
        self.exposure_model: "ExposureModel | None" = None

        # ---- Event coordinates (stored in degrees) ---------------------------
        self.ra: np.ndarray | None = None
        self.dec: np.ndarray | None = None

        # ---- Exposure-related attributes -------------------------------------
        self.expected_exposure_rate: float | None = None
        self.exposure: np.ndarray | None = None

        # ---- Flare bookkeeping -----------------------------------------------
        self.flare_mask: np.ndarray | None = None

    # -------------------------------------------------------------------------
    # Public factory classmethods
    # -------------------------------------------------------------------------

    @classmethod
    def full_sky(
        cls,
        n_total: int,
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
    ) -> "EventSample":
        """
        Build a full-sky isotropic sample of ``n_total`` events.

        ``n_sample == n_total`` and ``expected_n == n_total``. The sample
        carries no window and no exposure model.

        Parameters
        ----------
        n_total : int
            Number of events in the full-sky sample (also the expected
            count under the uniform isotropic model).
        t0, tf : astropy.time.Time
            Observation interval.
        rng : numpy.random.Generator
            Random generator used for the isotropic draw.

        Returns
        -------
        EventSample
            Sample with ``ra``, ``dec``, ``expected_n`` and
            ``spatial_type='full_sky'`` set.
        """
        obj = cls(
            n_sample=int(n_total),
            n_total=int(n_total),
            t0=t0,
            tf=tf,
            rng=rng,
        )
        obj._assign_full_sky_coordinates()
        obj.expected_n = float(n_total)
        return obj

    @classmethod
    def in_window(
        cls,
        window: "SkyWindow",
        n_total: int,
        exposure_model: "ExposureModel",
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
    ) -> "EventSample":
        """
        Build a per-window sample drawn directly inside ``window``.

        The construction is:

        1. ``expected_n = window.expected_n_in_window(n_total, exposure_model)``
           — the exposure-weighted expected count inside the window for an
           equivalent full-sky population of size ``n_total``.
        2. ``n_sample = rng.poisson(expected_n)`` — the realised event count.
        3. Coordinates are drawn uniformly in solid angle inside the cap.

        The window and the exposure model are stored on the returned
        instance (``self.window``, ``self.exposure_model``) so downstream
        code can reference them without re-passing.

        Parameters
        ----------
        window : SkyWindow
            Spherical-cap window defining the sampling region and the
            expected-count weighting.
        n_total : int
            Equivalent full-sky population (parent count) used to compute
            ``expected_n`` via the window's exposure-weighted fraction.
        exposure_model : ExposureModel
            Used to weight the expected count by the relative directional
            exposure at the window centre.
        t0, tf : astropy.time.Time
            Observation interval.
        rng : numpy.random.Generator
            Random generator used for the Poisson draw and the spatial
            sampling.

        Returns
        -------
        EventSample
            Fully-formed sample with ``ra``, ``dec``, ``expected_n``,
            ``window``, ``exposure_model`` and
            ``spatial_type='window'`` set.

        Raises
        ------
        ValueError
            If the Poisson draw yields ``n_sample == 0``; the downstream
            pipeline is not meaningful for zero-event samples (mirrors
            the zero-event Flare and the ``>=2`` events requirement of
            the Lambda estimator).
        """
        expected_n = float(window.expected_n_in_window(n_total, exposure_model))
        n_sample = int(rng.poisson(expected_n))

        if n_sample == 0:
            raise ValueError(
                f"Per-window Poisson draw yielded n_sample=0 "
                f"(expected_n={expected_n:.3g}). Downstream estimators "
                f"require at least 2 events; aborting construction."
            )

        obj = cls(
            n_sample=n_sample,
            n_total=int(n_total),
            t0=t0,
            tf=tf,
            rng=rng,
        )
        obj._assign_window_coordinates(window)
        obj.expected_n = expected_n
        obj.window = window
        obj.exposure_model = exposure_model
        return obj

    @classmethod
    def _from_arrays(
        cls,
        ra: np.ndarray,
        dec: np.ndarray,
        n_total: int,
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
        *,
        spatial_type: str | None = None,
        expected_n: float | None = None,
        window: "SkyWindow | None" = None,
        exposure_model: "ExposureModel | None" = None,
        exposure: np.ndarray | None = None,
        exposure_type: str | None = None,
        expected_exposure_rate: float | None = None,
        flare_mask: np.ndarray | None = None,
        flare_type: str | None = None,
    ) -> "EventSample":
        """
        Create an EventSample from existing arrays without drawing new coordinates.

        Optional metadata and event-level attributes can also be attached, such as
        exposure values and flare bookkeeping.
        """

        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)

        if ra.shape != dec.shape:
            raise ValueError(
                f"ra and dec must have the same shape, got {ra.shape} vs {dec.shape}."
            )
        if ra.ndim != 1:
            raise ValueError(f"ra and dec must be 1D arrays, got ndim={ra.ndim}.")

        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)
            if exposure.shape != ra.shape:
                raise ValueError(
                    f"exposure must have the same shape as ra/dec, "
                    f"got {exposure.shape} vs {ra.shape}."
                )

        if flare_mask is not None:
            flare_mask = np.asarray(flare_mask, dtype=bool)
            if flare_mask.shape != ra.shape:
                raise ValueError(
                    f"flare_mask must have same shape as ra/dec, "
                    f"got {flare_mask.shape} vs {ra.shape}."
                )

        obj = cls(
            n_sample=int(ra.size),
            n_total=int(n_total),
            t0=t0,
            tf=tf,
            rng=rng,
        )

        # Coordinates
        obj.ra = ra
        obj.dec = dec
        obj.spatial_type = spatial_type

        # Expected counts
        if expected_n is not None:
            obj.expected_n = float(expected_n)

        # Generation context
        obj.window = window
        obj.exposure_model = exposure_model

        # Exposure-related attributes
        obj.exposure = exposure
        obj.exposure_type = exposure_type
        obj.expected_exposure_rate = expected_exposure_rate

        # Flare bookkeeping
        obj.flare_mask = flare_mask
        obj.flare_type = flare_type

        return obj

    # -------------------------------------------------------------------------
    # Basic derived properties and state checks
    # -------------------------------------------------------------------------

    @property
    def T_obs(self) -> Quantity:
        """Observation duration as an astropy Quantity."""
        return (self.tf - self.t0).to(u.s)

    @property
    def expected_temporal_rate(self) -> float:
        """Expected rate of events per unit of time."""
        if self.expected_n is None:
            raise RuntimeError(
                "expected_n is not set; cannot compute expected_temporal_rate. "
                "Assign expected_n via the factory / generation routine first."
            )
        return float(self.expected_n / self.T_obs.to(u.s).value)

    @property
    def has_coordinates(self) -> bool:
        """Return True if coordinates have been assigned."""
        return self.ra is not None and self.dec is not None

    @property
    def has_exposure(self) -> bool:
        """Return True if the exposure array has been allocated (structural check only).

        A True value does not imply all entries are finite — after
        ``inject_flare()`` the array exists but background slots remain NaN
        until a subsequent ``assign_directional_exposure()`` is called.
        Finite-value validity is enforced separately by ``lambda_estimator``.
        """
        return self.exposure is not None

    @property
    def has_flare(self) -> bool:
        """Return True if flare events have been identified in the sample."""
        return self.flare_mask is not None and np.any(self.flare_mask)

    # -------------------------------------------------------------------------
    # Core sampling and low-level data manipulation
    # -------------------------------------------------------------------------

    def _generate_full_sky_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw ``self.n_sample`` isotropic coordinates over the whole sphere.

        Right ascension is uniform in ``[0, 360)``.
        Declination is distributed so that ``sin(dec)`` is uniform in
        ``[-1, 1]`` (isotropic on the sphere). Coordinates are returned in
        degrees.
        """
        ra = self.rng.uniform(0.0, 360.0, size=self.n_sample)
        u_rand = self.rng.uniform(-1.0, 1.0, size=self.n_sample)
        dec = np.degrees(np.arcsin(u_rand))

        return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)

    def _generate_window_coordinates(
        self,
        window: "SkyWindow",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw ``self.n_sample`` isotropic coordinates within ``window``.

        Delegates to :meth:`SkyWindow.sample_uniform`, which draws uniformly
        in solid angle over the spherical cap (i.e. isotropic on the cap).
        Coordinates are returned in degrees.
        """
        return window.sample_uniform(self.n_sample, self.rng)

    def _assign_full_sky_coordinates(self) -> None:
        """
        Generate isotropic full-sky coordinates and store them on ``self``.

        Sets ``self.ra``, ``self.dec`` and tags ``self.spatial_type`` as
        ``"full_sky"``. Private helper invoked by :meth:`full_sky`.
        """
        ra, dec = self._generate_full_sky_coordinates()
        self.ra = ra
        self.dec = dec
        self.spatial_type = "full_sky"

    def _assign_window_coordinates(self, window: "SkyWindow") -> None:
        """
        Generate isotropic coordinates within ``window`` and store them on ``self``.

        Sets ``self.ra``, ``self.dec`` and tags ``self.spatial_type`` as
        ``"window"``. Private helper invoked by :meth:`in_window`.

        Parameters
        ----------
        window : SkyWindow
            Spherical-cap window defining the sampling region.
        """
        ra, dec = self._generate_window_coordinates(window)
        self.ra = ra
        self.dec = dec
        self.spatial_type = "window"

    # -------------------------------------------------------------------------
    # Public selection and exposure methods
    # -------------------------------------------------------------------------

    def _subset(self, mask: np.ndarray) -> "EventSample":
        """
        Return a new ``EventSample`` containing only events where ``mask`` is True.

        All optional per-event arrays present on ``self`` (``exposure``,
        ``flare_mask``) are sliced consistently. Sample-level metadata
        (``spatial_type``, ``exposure_type``, ``flare_type``,
        ``expected_n``, ``expected_exposure_rate``) is propagated as-is.
        Note that ``expected_n`` is *not* rescaled by the subset fraction
        and should be set explicitly by the caller when its meaning changes
        (e.g. by :meth:`select_subsample`).
        """

        if self.ra is None or self.dec is None:
            raise ValueError("ra and dec are not available.")

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.ra.shape:
            raise ValueError(f"Mask must have shape {self.ra.shape}, got {mask.shape}.")

        exposure = None
        if self.exposure is not None:
            exposure = self.exposure[mask]

        flare_mask = None
        if self.flare_mask is not None:
            flare_mask = self.flare_mask[mask]

        return EventSample._from_arrays(
            ra=self.ra[mask],
            dec=self.dec[mask],
            n_total=self.n_total,
            t0=self.t0,
            tf=self.tf,
            rng=self.rng,
            spatial_type=self.spatial_type,
            expected_n=self.expected_n,
            window=self.window,
            exposure_model=self.exposure_model,
            exposure=exposure,
            exposure_type=self.exposure_type,
            expected_exposure_rate=self.expected_exposure_rate,
            flare_mask=flare_mask,
            flare_type=self.flare_type,
        )

    def select_subsample(
        self,
        window: SkyWindow,
    ) -> EventSample:
        """
        Return a new ``EventSample`` containing only events inside ``window``.

        The returned sample carries an updated ``expected_n`` set to
        ``window.expected_n_in_window(self.n_total)`` (i.e. the expected
        number of events inside the window under the *uniform full-sky*
        assumption built into :class:`SkyWindow`).

        Parameters
        ----------
        window : SkyWindow
            Spherical-cap window used to define the subset.

        Returns
        -------
        EventSample
            New sample with sliced ``ra``, ``dec``, and any optional
            per-event arrays.

        Raises
        ------
        ValueError
            If coordinates have not been generated, or if no event lies
            inside the window.
        """
        if not self.has_coordinates:
            raise ValueError("ra and dec are not available.")

        mask = window.contains(self.ra, self.dec)

        if not np.any(mask):
            raise ValueError("No events found inside the sky window.")

        subsample = self._subset(mask)
        subsample.expected_n = window.expected_n_in_window(self.n_total)
        subsample.window = window

        return subsample

    def generate_directional_exposure(
        self,
        window: "SkyWindow",
        exposure_model: "ExposureModel",
    ) -> tuple[np.ndarray, np.ndarray, float, str]:
        """
        Generate sampled cumulative directional exposure values for this EventSample,
        using the window definition as the reference direction.

        Intended workflow
        -----------------
        1) Start from a full dataset (parent sample) spanning [t0, tf].
        2) Apply a sky-window selection to build a subsample:
            `subsample = parent.select_subsample(window)`
        3) Call this method on the subsample to generate per-event epsilon values.

        Important notes
        ---------------
        - This method is designed for *window-selected subsamples*. It assumes that the
        sample was obtained via `select_subsample(...)` and does not validate that
        the events lie inside `window`.
        - No event times are generated or required. The method samples values directly
        in cumulative exposure space.
        - If flare events are present, exposure values are generated only for the
        non-flare events.
        - This method does not modify the sample.

        Parameters
        ----------
        window : SkyWindow
            The sky region that defined this subsample.
        exposure_model : ExposureModel
            Model providing the exposure-space sampling machinery.

        Returns
        -------
        eps : np.ndarray
            Sampled exposure values for the selected events.
        target_mask : np.ndarray
            Boolean mask indicating which events the sampled values correspond to.
        expected_exposure_rate : float
            Expected rate used in the exposure sampling.
        method : str
            Identifier of the sampling method used.
        """

        max_exposure = exposure_model.max_directional_exposure(window.centre)
        expected_exposure_rate = self.expected_n/ max_exposure

        if self.has_flare:
            isotropy_mask = ~self.flare_mask
        else:
            isotropy_mask = np.ones(self.n_sample, dtype=bool)

        n_target = int(np.count_nonzero(isotropy_mask))

        eps, method = exposure_model.sample_directional_exposure(
            n_events=n_target,
            expected_exposure_rate=expected_exposure_rate,
            max_dir_exposure=max_exposure,
        )

        eps = np.asarray(eps, dtype=float)

        return eps, isotropy_mask, expected_exposure_rate, str(method)

    def assign_directional_exposure(
        self,
        window: "SkyWindow",
        exposure_model: "ExposureModel",
    ) -> None:
        """
        Generate and assign cumulative directional exposure values to this EventSample.

        This method is intended for *window-selected subsamples* and acts as an
        in-place wrapper around `generate_directional_exposure(...)`.

        Important notes
        ---------------
        - If flare events are present, exposure values are generated only for the
        non-flare events, and existing flare exposure values are preserved.
        - If the sample does not yet contain an exposure array, it is initialized with
        `np.nan` values and then populated at the relevant positions.
        - This method modifies the current sample in-place.

        Parameters
        ----------
        window : SkyWindow
            The sky region that defined this subsample.
        exposure_model : ExposureModel
            Model providing the exposure-space sampling machinery.
        """
        eps, target_mask, expected_exposure_rate, method = (
            self.generate_directional_exposure(window, exposure_model)
        )

        if self.exposure is None:
            self.exposure = np.full(self.n_sample, np.nan, dtype=float)

        self.exposure[target_mask] = eps
        self.expected_exposure_rate = expected_exposure_rate
        self.exposure_type = method

    # -------------------------------------------------------------------------
    # Public flare manipulation
    # -------------------------------------------------------------------------

    def inject_flare(self, flare: "Flare", *, mode: str) -> None:
        """
        Inject a fully-generated flare into the current sample, in place.

        Two modes are supported and the caller must pick one explicitly
        (the keyword-only ``mode`` argument has no default).  Both modes
        place the flare events at the **tail** of the sample arrays
        (indices ``[-n_flare:]``) and set ``flare_mask`` ``True`` there;
        they differ only in how many existing background events are
        removed to make room.

        ``mode="overdensity"``
            Appends ``n_flare`` flare events to the sample after
            removing a Poisson-distributed number of existing
            background events.  Net count grows.  Reproduces the
            legacy "full-sky → carve window" pipeline in the
            per-window pipeline: a flare would overwrite
            ``n_flare`` random slots in a hypothetical full-sky parent
            of size ``n_total``; on average ``p * n_flare`` of those
            slots happened to lie inside the window (where
            ``p = expected_n / n_total``), and those events are no
            longer in the in-window sample::

                n_removed ~ Poisson(p * n_flare)        # clipped at n_sample
                n_sample_after = n_sample_before - n_removed + n_flare

            Tests both spatial and temporal anisotropy: the window
            count goes up *and* the flare events cluster in time.

        ``mode="no_overdensity"``
            Removes exactly ``n_flare`` random background events and
            appends the ``n_flare`` flare events.  Net count
            preserved.  This is the semantics of the legacy full-sky
            pipeline before any window cut: the flare replaces
            ``n_flare`` events of the parent ``n_total``-event sample
            without changing the sample size.  Useful for testing
            temporal-only signals (no spatial overdensity at the
            window-count level)::

                n_removed      = n_flare
                n_sample_after = n_sample_before

        Parameters
        ----------
        flare : Flare
            Flare with ``ra``, ``dec`` and ``exposure`` already
            populated (typically via :meth:`Flare.generate_in_window`).
        mode : {"overdensity", "no_overdensity"}
            Required keyword.  Selects the injection semantics
            described above.  There is no default — the caller must
            pick one so it is always obvious what kind of injection
            took place.

        Raises
        ------
        TypeError
            If ``flare`` is not a :class:`Flare` instance.
        RuntimeError
            If the sample already contains an injected flare.
        ValueError
            - If ``mode`` is not one of the two accepted strings.
            - If coordinates have not been assigned.
            - If the flare has not been fully generated.
            - In ``"overdensity"`` mode: if ``expected_n`` is unset,
              ``n_total <= 0``, or ``n_flare > n_total``.
            - In ``"no_overdensity"`` mode: if ``n_flare > n_sample``.

        Notes
        -----
        - If ``self.exposure`` is ``None`` it is allocated as ``NaN``
          for the surviving background slots and filled with
          ``flare.exposure`` at the appended tail.  The caller is
          responsible for a subsequent
          :meth:`assign_directional_exposure` to populate the
          background slots.
        - If ``self.exposure`` was already populated, surviving
          background entries keep their exposure values and the flare
          exposure is appended unchanged.
        - The Poisson model for ``n_removed`` in ``"overdensity"``
          mode is the small-``p`` / large-``n_flare`` approximation of
          the underlying Hypergeometric draw; accurate when the
          window covers a small fraction of the sky, which is the
          regime of interest.
        """
        from .flare import Flare

        # ---- Common validation ------------------------------------------
        if not isinstance(flare, Flare):
            raise TypeError("flare must be an instance of Flare.")

        if mode not in ("overdensity", "no_overdensity"):
            raise ValueError(
                f"mode must be 'overdensity' or 'no_overdensity'; got {mode!r}."
            )

        if self.has_flare:
            raise RuntimeError("This sample already contains an injected flare.")

        if not self.has_coordinates:
            raise ValueError("Sample coordinates are not available.")

        if flare.ra is None or flare.dec is None or flare.exposure is None:
            raise ValueError(
                "Flare is not fully generated. "
                "Coordinates and exposure must be computed before injection."
            )

        # ---- Mode-specific n_removed ------------------------------------
        if mode == "overdensity":
            if self.expected_n is None:
                raise ValueError(
                    "Sample expected_n is not set; cannot determine the "
                    "background-overlap probability used for overdensity-mode "
                    "flare thinning."
                )
            if self.n_total <= 0:
                raise ValueError(
                    "Sample n_total must be > 0 for flare injection."
                )
            if flare.n_flare > self.n_total:
                raise ValueError(
                    f"In overdensity mode, n_flare ({flare.n_flare}) cannot "
                    f"exceed n_total ({self.n_total}): the flare is drawn "
                    f"from a hypothetical full-sky sample of that size."
                )
            p_in_window = float(self.expected_n) / float(self.n_total)
            mu_removed = p_in_window * flare.n_flare
            n_removed = int(self.rng.poisson(mu_removed))
            n_removed = min(n_removed, self.n_sample)
        else:  # mode == "no_overdensity"
            if flare.n_flare > self.n_sample:
                raise ValueError(
                    f"In no_overdensity mode, n_flare ({flare.n_flare}) "
                    f"cannot exceed n_sample ({self.n_sample}): there are "
                    f"not enough slots to replace."
                )
            n_removed = flare.n_flare

        # ---- Common removal + tail append -------------------------------
        keep_mask = np.ones(self.n_sample, dtype=bool)
        if n_removed > 0:
            remove_idx = self.rng.choice(
                self.n_sample, size=n_removed, replace=False,
            )
            keep_mask[remove_idx] = False

        new_ra = np.concatenate([self.ra[keep_mask], flare.ra])
        new_dec = np.concatenate([self.dec[keep_mask], flare.dec])

        if self.exposure is None:
            new_exposure = np.full(new_ra.size, np.nan, dtype=float)
            new_exposure[-flare.n_flare:] = flare.exposure
        else:
            new_exposure = np.concatenate(
                [self.exposure[keep_mask], flare.exposure]
            )

        new_flare_mask = np.zeros(new_ra.size, dtype=bool)
        new_flare_mask[-flare.n_flare:] = True

        self.ra = new_ra
        self.dec = new_dec
        self.exposure = new_exposure
        self.flare_mask = new_flare_mask
        self.flare_type = flare.flare_type
        self.n_sample = int(new_ra.size)

    # -------------------------------------------------------------------------
    # Public skymap / visualization interface
    # -------------------------------------------------------------------------

    def get_healpix_skymap(
        self,
        nside: int = 32,
        *,
        mask_fov: bool = False,
        location: EarthLocation | None = None,
        zenith_max: u.Quantity | None = None,
    ) -> np.ndarray:
        """
        Build a HEALPix counts map from the sample event coordinates.

        Parameters
        ----------
        nside : int, optional
            HEALPix NSIDE parameter. Must be a valid HEALPix value.
        mask_fov : bool, optional
            If True, mask pixels outside the declination band visible from
            the observatory defined by ``location`` and ``zenith_max``.
        location : astropy.coordinates.EarthLocation, optional
            Observatory location. Required if ``mask_fov=True``.
        zenith_max : astropy.units.Quantity, optional
            Maximum zenith angle. Required if ``mask_fov=True``.

        Returns
        -------
        skymap : numpy.ndarray or numpy.ma.MaskedArray
            HEALPix map of event counts. If ``mask_fov=True``, a masked map
            is returned.

        Notes
        -----
        This method assumes ``self.ra`` and ``self.dec`` are stored in degrees.
        """
        if self.ra is None or self.dec is None:
            raise ValueError(
                "ra and dec are not available. "
                "Sample coordinates before building the skymap."
            )

        hp = self._healpy()

        if not hp.isnsideok(nside):
            raise ValueError("nside must be a valid HEALPix NSIDE value.")

        ra_deg = np.asarray(self.ra, dtype=float)
        dec_deg = np.asarray(self.dec, dtype=float)

        if ra_deg.shape != dec_deg.shape:
            raise ValueError("ra and dec must have the same shape.")

        valid = np.isfinite(ra_deg) & np.isfinite(dec_deg)
        ra_deg = ra_deg[valid]
        dec_deg = dec_deg[valid]

        ra_rad = np.deg2rad(np.mod(ra_deg, 360.0))
        dec_rad = np.deg2rad(dec_deg)

        npix = hp.nside2npix(nside)
        skymap = np.zeros(npix, dtype=float)

        theta = 0.5 * np.pi - dec_rad
        phi = ra_rad

        pix = hp.ang2pix(nside, theta, phi)
        np.add.at(skymap, pix, 1)

        if mask_fov:
            if location is None or zenith_max is None:
                raise ValueError(
                    "location and zenith_max must be provided when mask_fov=True."
                )
            skymap = self._mask_skymap_outside_fov(
                skymap=skymap,
                nside=nside,
                location=location,
                zenith_max=zenith_max,
            )

        return skymap

    def plot_skymap(
        self,
        nside: int = 32,
        *,
        mask_fov: bool = False,
        location: EarthLocation | None = None,
        zenith_max: u.Quantity | None = None,
        title: str = "Sky map",
        cmap: str = "magma",
        output_file: str | None = None,
        astronomical: bool = True,
        show: bool = True,
        xticks_deg: np.ndarray | list | None = None,
        yticks_deg: np.ndarray | list | None = None,
    ):
        """
        Plot the event sample as a HEALPix-binned sky map in Hammer projection.

        Parameters
        ----------
        nside : int, optional
            HEALPix resolution parameter.
        mask_fov : bool, optional
            If True, mask pixels outside the observatory declination band
            defined by ``location`` and ``zenith_max``.
        location : astropy.coordinates.EarthLocation, optional
            Required when ``mask_fov=True``.
        zenith_max : astropy.units.Quantity, optional
            Required when ``mask_fov=True``.
        title : str, optional
            Figure title.
        cmap : str, optional
            Matplotlib colormap name.
        output_file : str or None, optional
            If given, the figure is saved at this path with ``dpi=300``.
        astronomical : bool, optional
            If True (default), display the sky in the astronomical
            convention (RA increases to the left).
        show : bool, optional
            If True, call ``plt.show()`` after drawing.
        xticks_deg, yticks_deg : array-like or None, optional
            Override the default tick locations (in degrees) on the
            longitude/latitude axes.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Created figure and axes objects.
        """
        skymap = self.get_healpix_skymap(
            nside=nside,
            mask_fov=mask_fov,
            location=location,
            zenith_max=zenith_max,
        )
        plt = self._pyplot()

        lon_edges, lat_edges, image = self._healpix_to_lonlat_image(
            skymap=skymap,
            nside=nside,
            astronomical=astronomical,
        )

        fig, ax = plt.subplots(
            figsize=(8, 4.8),
            subplot_kw={"projection": "hammer"},
        )

        vmin = np.nanmin(image)
        vmax = np.nanmax(image)

        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            image,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title)
        ax.grid(True, alpha=0.6)
        ax.set_facecolor("lightgrey")

        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08)
        cbar.set_label("Number of events")

        if xticks_deg is None:
            xticks_deg = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
        else:
            xticks_deg = np.asarray(xticks_deg, dtype=float)
        ax.set_xticks(np.deg2rad(xticks_deg))

        if astronomical:
            ax.set_xticklabels([f"{(-x) % 360:.0f}°" for x in xticks_deg])
            ax.set_xlabel(r"Right ascension $\alpha$")
        else:
            ax.set_xticklabels([f"{x:.0f}°" for x in xticks_deg])
            ax.set_xlabel("Longitude")

        if yticks_deg is None:
            yticks_deg = np.array([-60, -30, 0, 30, 60])
        else:
            yticks_deg = np.asarray(yticks_deg, dtype=float)
        ax.set_yticks(np.deg2rad(yticks_deg))
        ax.set_yticklabels([f"{y:.0f}°" for y in yticks_deg])
        ax.set_ylabel(r"Declination $\delta$")

        fig.tight_layout()

        if output_file is not None:
            fig.savefig(output_file, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax

    # -------------------------------------------------------------------------
    # Internal skymap helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _visible_declination_band(
        location: EarthLocation,
        zenith_max: u.Quantity,
    ) -> tuple[float, float]:
        """
        Compute the accessible declination band for an observatory.

        Returns
        -------
        dec_min, dec_max : tuple of float
            Minimum and maximum visible declinations in radians.
        """
        if not isinstance(location, EarthLocation):
            raise TypeError("location must be an astropy.coordinates.EarthLocation.")

        if not isinstance(zenith_max, u.Quantity):
            raise TypeError("zenith_max must be an astropy.units.Quantity.")

        zenith_max_rad = zenith_max.to_value(u.rad)
        lat_rad = location.lat.to_value(u.rad)

        dec_min = max(-0.5 * np.pi, lat_rad - zenith_max_rad)
        dec_max = min(+0.5 * np.pi, lat_rad + zenith_max_rad)

        return dec_min, dec_max

    @classmethod
    def _mask_skymap_outside_fov(
        cls,
        skymap: np.ndarray,
        nside: int,
        *,
        location: EarthLocation,
        zenith_max: u.Quantity,
    ):
        """
        Mask HEALPix pixels outside the observatory declination band.
        """
        dec_min, dec_max = cls._visible_declination_band(location, zenith_max)
        hp = cls._healpy()

        masked = np.array(skymap, copy=True)
        npix = hp.nside2npix(nside)
        ipix = np.arange(npix)

        theta, _phi = hp.pix2ang(nside, ipix)
        dec = 0.5 * np.pi - theta

        outside = (dec < dec_min) | (dec > dec_max)
        masked[outside] = hp.UNSEEN

        return hp.ma(masked)

    @staticmethod
    def _healpix_to_lonlat_image(
        skymap,
        nside: int,
        *,
        nx: int = 361,
        ny: int = 181,
        astronomical: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample a HEALPix map onto a regular lon/lat grid for pcolormesh.

        Parameters
        ----------
        skymap : numpy.ndarray or numpy.ma.MaskedArray
            Input HEALPix map.
        nside : int
            HEALPix NSIDE parameter.
        nx, ny : int, optional
            Number of longitude/latitude grid edges.
        astronomical : bool, optional
            If True, use the standard astronomical convention in which
            right ascension increases to the left.

        Returns
        -------
        lon_edges, lat_edges, image : tuple of numpy.ndarray
            Grid edges in radians and the gridded image.
        """
        hp = EventSample._healpy()
        lon_edges = np.linspace(-np.pi, np.pi, nx)
        lat_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, ny)

        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

        lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)

        if astronomical:
            phi = np.mod(-lon2d, 2.0 * np.pi)
        else:
            phi = np.mod(lon2d, 2.0 * np.pi)

        theta = 0.5 * np.pi - lat2d
        pix = hp.ang2pix(nside, theta, phi)

        image = np.asarray(skymap[pix], dtype=float)

        if np.ma.isMaskedArray(skymap):
            image = np.where(np.isfinite(image), image, np.nan)

        return lon_edges, lat_edges, image
    