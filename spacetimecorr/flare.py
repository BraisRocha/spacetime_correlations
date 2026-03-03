from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.units import Quantity

from .skywindow import SkyWindow
from .event_sample import EventSample
from .exposure import ExposureModel


class FlareInjector:
    """Generate a flare component: clustered RA/Dec and event times within a short duration.

    Parameters
    ----------
    n_events : int
        Number of flare events to generate (non-negative).
    duration : astropy.units.Quantity
        Flare duration as a time Quantity (e.g., 10*u.s, 5*u.min). Must be > 0.
    window : SkyWindow
        Sky region where the flare is injected (centre used as cluster mean).
    parent_sample : EventSample
        Parent sample defining the observation time interval [t0, tf].
    exposure : ExposureModel
        Exposure model associated with the observation (used later for ε(t) mapping).
    rng : numpy.random.Generator
        Random generator stream used to generate flare coordinates and times.
    """

    def __init__(
        self,
        n_events: int,
        duration: Quantity,
        window: SkyWindow,
        parent_sample: EventSample,
        exposure: ExposureModel,
        rng: np.random.Generator,
    ) -> None:

        if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events < 0:
            raise TypeError("n_events must be a non-negative integer.")

        if not isinstance(duration, u.Quantity):
            raise TypeError("duration must be an astropy.units.Quantity with time units (e.g., 10*u.s).")
        if not duration.unit.is_equivalent(u.s):
            raise ValueError("duration must have time units (equivalent to seconds).")
        if duration <= 0 * u.s:
            raise ValueError("duration must be > 0.")
        duration_sec = float(duration.to_value(u.s))

        if not isinstance(window, SkyWindow):
            raise TypeError("window must be an instance of SkyWindow.")
        if not isinstance(parent_sample, EventSample):
            raise TypeError("parent_sample must be an instance of EventSample.")
        if not isinstance(exposure, ExposureModel):
            raise TypeError("exposure must be an instance of ExposureModel.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be a numpy.random.Generator. "
                "Obtain one from RNGManager.get(name) and pass it here."
            )

        # Ensure flare fits in the observation window
        T_obs_sec = float(parent_sample.T_obs.to_value(u.s))
        if duration_sec > T_obs_sec:
            raise ValueError("duration cannot exceed the parent sample observation duration.")

        self.parent_sample = parent_sample
        self.window = window
        self.exposure = exposure
        self.rng = rng

        self.n_events = int(n_events)
        self.duration = duration_sec  # stored internally as float seconds

        self.spatial_type: str | None = None

        # Assigned later
        self.RA: np.ndarray | None = None
        self.Dec: np.ndarray | None = None
        self.time: Time | None = None                 # astropy Time array
        self.dir_exposure: np.ndarray | None = None   # optional, computed later

    def generate_gaussian_cluster(self) -> None:
        """Generate a Gaussian cluster around window centre with std = window.radius (degrees)."""
        ra_dec = self.rng.multivariate_normal(
            mean=self.window.centre,
            cov=np.eye(2) * (self.window.radius ** 2),
            size=self.n_events,
        )
        ra = ra_dec[:, 0]
        dec = ra_dec[:, 1]

        # RA is periodic; wrap to [0, 360)
        ra = np.mod(ra, 360.0)

        # Dec is not periodic; clip to [-90, 90] (simple safeguard)
        dec = np.clip(dec, -90.0, 90.0)

        # Store
        self.RA = ra.astype(float, copy=False)
        self.Dec = dec.astype(float, copy=False)
        self.spatial_type = "equatorial"

    def generate_uniform_flare_times(self) -> None:
        """Generate flare times uniformly within a random start time and given duration."""
        T_obs_sec = float(self.parent_sample.T_obs.to_value(u.s))
        latest_start = T_obs_sec - self.duration
        if latest_start < 0:
            raise ValueError("duration cannot exceed the parent sample observation duration.")

        # Choose a flare start within [t0, tf - duration]
        start_offset_sec = self.rng.uniform(0.0, latest_start)
        start = self.parent_sample.t0 + TimeDelta(start_offset_sec, format="sec")

        # Sample event times within [start, start + duration]
        offsets = self.rng.uniform(0.0, self.duration, size=self.n_events)
        self.time = start + TimeDelta(offsets, format="sec")

        # Convert times to cumulative directional exposure using the model
        self.dir_exposure = self.exposure.to_directional_exposure(self.time, self.window.centre)
    
    def inject_flare_into_sample(self):

        # Inject the flare events at random indices to avoid biases
        idx = self.rng.choice(self.parent_sample.n_events, size=self.n_events, replace=False)

        ra = self.parent_sample.RA[idx]=self.RA
        dec = self.parent_sample.Dec[idx]=self.Dec
        dir_exp =
    
        



"""    
    # Convert to cumulative directional exposure using the model
    exp = self.exposure_model.to_directional_exposure(times, centre=self.window.centre)

    # Inject into random indices
    idx = self.rng.choice(self.n_events, size=n_flare, replace=False)
    self.RA[idx] = points[:, 0]
    self.Dec[idx] = points[:, 1]
    self.dir_exposure[idx] = exp
"""