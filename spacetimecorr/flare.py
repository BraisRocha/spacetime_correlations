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
    Generate a synthetic flare component.

    A flare is defined independently of an EventSample. It stores the
    parameters needed to generate a compact set of events in time and sky
    coordinates within a fixed observation interval [t0, tf].

    Parameters
    ----------
    n_events : int
        Number of flare events to generate (non-negative).
    duration : astropy.units.Quantity
        Flare duration as a time Quantity. Must be > 0.
    t0 : astropy.time.Time
        Start of the observation interval.
    tf : astropy.time.Time
        End of the observation interval.
    centre : np.ndarray
        Central sky position of the flare, as [RA, Dec] in degrees.
    exposure : ExposureModel
        Exposure model associated with the observation.
    rng : numpy.random.Generator
        Random generator stream used to generate flare coordinates and times.
    """

    def __init__(
        self,
        n_events: int,
        duration: Quantity,
        t0: Time,
        tf: Time,
        centre: np.ndarray,
        exposure: "ExposureModel",
        rng: np.random.Generator,
    ):
        
        if not isinstance(n_events, int) or isinstance(n_events, bool):
            raise TypeError("n_events must be a non-negative integer.")
        if n_events < 0:
            raise ValueError("n_events must be >= 0.")

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
        
        if not isinstance(exposure, ExposureModel):
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

        self.n_events = int(n_events)
        self.duration = float(duration_sec)

        self.t0 = t0
        self.tf = tf
        self.centre = centre
        self.exposure = exposure

        self.spatial_profile: str | None = None
        self.time_profile: str | None = None

        # Generated data
        self.RA: np.ndarray | None = None
        self.Dec: np.ndarray | None = None
        self.time: Time | None = None
        self.dir_exposure: np.ndarray | None = None

    # -------------------------------------------------------------------------
    # Low-level sampling / evaluation methods
    # -------------------------------------------------------------------------

    def _draw_flare_start(self) -> Time:
        """
        Draw the flare start time uniformly in [t0, tf - duration].
        """
        latest_start = self._T_obs_sec - self.duration
        start_offset_sec = self.rng.uniform(0.0, latest_start)
        return self.t0 + TimeDelta(start_offset_sec, format="sec")


    def _sample_gaussian_cluster(
        self,
        n_events: int,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample `n_events` equatorial coordinates from a Gaussian cluster
        on the sphere around `self.centre`.

        Parameters
        ----------
        n_events : int
            Number of coordinates to draw.
        sigma : float
            Width of the cluster in degrees.

        Returns
        -------
        RA, Dec : tuple[np.ndarray, np.ndarray]
            Arrays of right ascension and declination in degrees.
        """

        center = SkyCoord(
            ra=self.centre[0] * u.deg,
            dec=self.centre[1] * u.deg,
            frame="icrs",
        )

        local_theta = self.rng.rayleigh(scale=sigma, size=n_events) * u.deg
        local_azimuth = self.rng.uniform(0.0, 2.0 * np.pi, size=n_events) * u.rad

        event_coords = center.directional_offset_by(local_azimuth, local_theta)

        RA = event_coords.ra.deg.astype(float, copy=False)
        Dec = event_coords.dec.deg.astype(float, copy=False)
        return RA, Dec
    
    def _sample_uniform_times(self, n_events: int, start: Time | None = None) -> Time:
        """
        Sample `n_events` times uniformly inside one flare interval.

        If `start` is not given, a flare start is drawn uniformly in
        [self.t0, self.tf - self.duration].
        """
        if start is None:
            start = self._draw_flare_start()

        # Draw time offsets inside the flare duration
        offsets_sec = self.rng.uniform(0.0, self.duration, size=n_events)

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
            self.exposure.cumulative_directional_exposure(time, direction),
            dtype=float,
        )
    
    # -------------------------------------------------------------------------
    # Public population methods
    # -------------------------------------------------------------------------

    def generate_gaussian_cluster(self, sigma: float) -> None:
        """
        Generate equatorial coordinates for this flare from a Gaussian cluster
        on the sphere.

        Parameters
        ----------
        sigma : float
            Width of the cluster in degrees.
        """
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")

        RA, Dec = self._sample_gaussian_cluster(self.n_events, sigma)
        self.RA = RA
        self.Dec = Dec
        self.spatial_profile = "gaussian_spherical"

    @property
    def is_populated(self) -> bool:
        """Return True if flare coordinates have been generated."""
        return self.RA is not None and self.Dec is not None
    
    def generate_uniform_times(self) -> None:
        """
        Generate and store `self.n_events` flare times with a uniform profile.
        """
        self.time = self._sample_uniform_times(self.n_events)
        self.time_profile = "uniform"

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
            raise ValueError("Flare times are not set. Call generate_uniform_times() first.")

        self.dir_exposure = self._evaluate_directional_exposure(self.time, direction)

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------

    @classmethod
    def _from_arrays(
        cls,
        RA: np.ndarray,
        Dec: np.ndarray,
        duration: Quantity,
        t0: Time,
        tf: Time,
        centre: np.ndarray,
        exposure: ExposureModel,
        rng: np.random.Generator,
        time: Time | None = None,
        dir_exposure: np.ndarray | None = None,
        spatial_profile: str | None = None,
        time_profile: str | None = None,
    ) -> Flare:
        """
        Create a Flare from existing arrays without generating new random values.
        """

        RA = np.asarray(RA, dtype=float)
        Dec = np.asarray(Dec, dtype=float)

        if RA.shape != Dec.shape:
            raise ValueError(f"RA and Dec must have the same shape, got {RA.shape} vs {Dec.shape}.")
        if RA.ndim != 1:
            raise ValueError(f"RA and Dec must be 1D arrays, got ndim={RA.ndim}.")

        if time is not None and np.size(time) != RA.size:
            raise ValueError(f"time must have size {RA.size}, got {np.size(time)}.")

        if dir_exposure is not None:
            dir_exposure = np.asarray(dir_exposure, dtype=float)
            if dir_exposure.shape != RA.shape:
                raise ValueError(
                    f"dir_exposure must have shape {RA.shape}, got {dir_exposure.shape}."
                )
            
        obj = cls(
            n_events=int(RA.size),
            duration=duration,
            t0=t0,
            tf=tf,
            centre=centre,
            exposure=exposure,
            rng=rng,
        )

        obj.RA = RA.copy()
        obj.Dec = Dec.copy()
        obj.time = time
        obj.dir_exposure = None if dir_exposure is None else dir_exposure.copy()
        obj.spatial_profile = spatial_profile
        obj.time_profile = time_profile

        return obj
    
    def _subset(self, mask: np.ndarray) -> Flare:
        """
        Return a new Flare containing only events where mask is True.
        """
        if self.RA is None or self.Dec is None:
            raise ValueError("RA/Dec are not set in the flare.")

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.RA.shape:
            raise ValueError(f"Mask must have shape {self.RA.shape}, got {mask.shape}.")

        sub_time = None if self.time is None else self.time[mask]
        sub_dir_exposure = None if self.dir_exposure is None else self.dir_exposure[mask]

        return Flare._from_arrays(
            RA=self.RA[mask],
            Dec=self.Dec[mask],
            duration=self.duration * u.s,
            t0=self.t0,
            tf=self.tf,
            centre=self.centre,
            exposure=self.exposure,
            rng=self.rng,
            time=sub_time,
            dir_exposure=sub_dir_exposure,
            spatial_profile=self.spatial_profile,
            time_profile=self.time_profile,
        )
    
    def select_subsample(self, window: SkyWindow) -> Flare:
        """
        Return a new Flare containing only the events within the sky window.
        """
        if self.RA is None or self.Dec is None:
            raise ValueError("RA/Dec are not set in the flare.")

        mask = window.contains(self.RA, self.Dec)
        return self._subset(mask)
    
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
        Generate a flare realization inside a sky window and store it in `self`.

        This method:
        1. draws spatial candidates from the Gaussian profile,
        2. keeps only those inside the sky window,
        3. assigns times within a single flare interval,
        4. applies exposure thinning,
        5. stores exactly `self.n_events` accepted events.

        Parameters
        ----------
        window : SkyWindow
            Sky window used for spatial selection and exposure evaluation.
        sigma : float
            Standard deviation of the Gaussian spatial profile in degrees.
        efficiency : optional
            Optional efficiency parameter passed to the exposure model.
        """

        if not isinstance(window, SkyWindow):
            raise TypeError("window must be an instance of SkyWindow.")
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")

        target = self.n_events

        if target == 0:
            self.RA = np.empty(0, dtype=float)
            self.Dec = np.empty(0, dtype=float)
            self.time = self.t0 + TimeDelta(np.empty(0), format="sec")
            self.dir_exposure = np.empty(0, dtype=float)
            self.spatial_profile = "gaussian_spherical"
            self.time_profile = "uniform_thinned"
            return
        
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
                    "reaching max_draws = 1000 * self.n_events."
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
            detection_mask = self.exposure.acceptance_mask(
                times_cand, 
                window.centre,
                efficiency=efficiency
            )

            if not np.any(detection_mask):
                continue

            ra_acc.append(ra_cand[detection_mask])
            dec_acc.append(dec_cand[detection_mask])
            time_acc.append(times_cand[detection_mask])
            n_kept += int(np.count_nonzero(detection_mask))


        # Clean up and slice to exact target
        self.RA = np.concatenate(ra_acc)[:target]
        self.Dec = np.concatenate(dec_acc)[:target]
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
        Return a string describing the flare model.
        """
        if self.spatial_profile is None or self.time_profile is None:
            return "undefined_flare"

        return f"{self.spatial_profile}-{self.time_profile}"