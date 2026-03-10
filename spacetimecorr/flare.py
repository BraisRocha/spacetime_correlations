from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.units import Quantity

from .skywindow import SkyWindow
from .event_sample import EventSample
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
    ) -> None:
        
        if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events <= 0:
            raise TypeError("n_events must be a positive integer.")

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
        
        # Observation span in seconds
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

    def generate_gaussian_cluster(self, sigma: float) -> None:
        """
        Generate equatorial coordinates from a 2D Gaussian cluster.

        Parameters
        ----------
        sigma : float
            Standard deviation of the Gaussian in degrees.
        """
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")

        ra_dec = self.rng.multivariate_normal(
            mean=self.centre,
            cov=np.eye(2) * sigma**2,
            size=self.n_events,
        )

        ra = np.mod(ra_dec[:, 0], 360.0)
        dec = np.clip(ra_dec[:, 1], -90.0, 90.0)

        self.RA = ra.astype(float, copy=False)
        self.Dec = dec.astype(float, copy=False)
        self.spatial_profile = "gaussian_equatorial"

    @property
    def is_populated(self) -> bool:
        """Return True if flare coordinates have been generated."""
        return self.RA is not None and self.Dec is not None

    def generate_uniform_times(self) -> None:
        """
        Generate flare event times uniformly within a randomly placed flare interval.

        The flare interval has fixed length ``self.duration`` and is placed uniformly
        within the observation window ``[self.t0, self.tf]``. Event times are then
        sampled uniformly inside that flare interval.
        """

        latest_start = self._T_obs_sec - self.duration

        # Choose flare start uniformly in [t0, tf - duration]
        start_offset_sec = self.rng.uniform(0.0, latest_start)
        start = self.t0 + TimeDelta(start_offset_sec, format="sec")

        # Sample event times uniformly in [start, start + duration]
        offsets_sec = self.rng.uniform(0.0, self.duration, size=self.n_events)
        offsets_sec.sort()  # optional, but often convenient
        self.time = start + TimeDelta(offsets_sec, format="sec")

        self.time_profile = "uniform"

    def compute_directional_exposure(self, direction: np.ndarray) -> None:
        """
        Compute directional exposure for the generated flare times at a given direction.

        Parameters
        ----------
        direction : np.ndarray
            Sky direction [RA, Dec] in degrees at which the exposure is evaluated.
        """

        if self.time is None:
            raise ValueError("Flare times are not set. Call generate_uniform_times() first.")

        direction = np.asarray(direction, dtype=float)
        if direction.shape != (2,):
            raise ValueError("direction must be a length-2 array: [RA, Dec] in degrees.")

        self.dir_exposure = self.exposure.to_directional_exposure(self.time, direction)

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
        obj.dir_exposure = None if dir_exposure is None else np.asarray(dir_exposure, dtype=float).copy()
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
    
    def generate_in_window(self, window: SkyWindow, sigma: float) -> Flare:
        """
        Generate a flare realization and return the subset of events inside a sky window.

        This is a high-level convenience method that:
        1. generates flare coordinates from a Gaussian spatial profile,
        2. generates flare event times from a uniform temporal profile,
        3. computes directional exposure at the window centre,
        4. applies the window selection.

        Parameters
        ----------
        window : SkyWindow
            Sky window used both for exposure evaluation and event selection.
        sigma : float
            Standard deviation of the Gaussian spatial profile in degrees.

        Returns
        -------
        Flare
            A new Flare containing only the generated events that fall inside
            the given sky window.
        """
        if not isinstance(window, SkyWindow):
            raise TypeError("window must be an instance of SkyWindow.")

        self.generate_gaussian_cluster(sigma=sigma)
        self.generate_uniform_times()
        self.compute_directional_exposure(direction=window.centre)

        return self.select_subsample(window=window)
    
    @property
    def flare_type(self) -> str:
        """
        Return a string describing the flare model.
        """
        if self.spatial_profile is None or self.time_profile is None:
            return "undefined_flare"

        return f"{self.spatial_profile}-{self.time_profile}"