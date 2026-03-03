from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.units import Quantity

if TYPE_CHECKING:
    from .skywindow import SkyWindow
    from .exposure import ExposureModel

class EventSample:
    """Generate and store samples of events in equatorial coordinates.

    Parameters
    ----------
    n_events : int
        Number of events to generate (non-negative).
    t0 : astropy.time.Time
        Observation start time.
    tf : astropy.time.Time
        Observation end time (must be > t0).
    rng : numpy.random.Generator
        Reproducible random generator stream (e.g., from RNGManager.get(name)).

    Notes
    -----
    - Coordinates are not generated until you call `sample_equatorial_coordinates()`.
    - Use `subset(mask)` to build a new EventSample containing only selected events.
    - Use `add_directional_exposure_for_window()` after building a subsample to 
    add directional exposure to the selected events.
    """

    def __init__(
        self,
        n_events: int,
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
    ):
        if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events < 0:
            raise TypeError("n_events must be a non-negative integer.")
        if not isinstance(t0, Time) or not isinstance(tf, Time):
            raise TypeError("t0 and tf must be astropy.time.Time objects.")
        if tf <= t0:
            raise ValueError("tf must be strictly later than t0.")
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be a numpy.random.Generator. "
                "Obtain one from RNGManager.get(name) and pass it here."
            )

        self.rng = rng

        self.n_events = int(n_events)
        self.t0 = t0
        self.tf = tf

        self.spatial_type: str | None = None

        # Assigned later
        self.RA: np.ndarray | None = None
        self.Dec: np.ndarray | None = None

    @property
    def T_obs(self) -> Quantity:
        """Observation duration as an astropy Quantity."""
        return (self.tf - self.t0).to(u.s)
    
    @property 
    def exp_rate_time(self) -> float:
        """Expected rate of events per unit of time."""
        return float(self.n_events/self.T_obs.to(u.s).value) 

    def is_populated(self) -> bool:
        """Return True if RA/Dec have been generated/assigned."""
        return self.RA is not None and self.Dec is not None

    def sample_equatorial_coordinates(self) -> None:
        """Simulate an isotropic distribution on the sphere in equatorial coordinates.

        RA is uniform in [0, 360).
        Dec is distributed such that sin(Dec) is uniform in [-1, 1] (isotropic on the sphere).
        """
        RA = self.rng.uniform(0.0, 360.0, size=self.n_events)
        uu = self.rng.uniform(-1.0, 1.0, size=self.n_events)  # uu = sin(Dec)
        Dec = np.degrees(np.arcsin(uu))

        self.RA = np.asarray(RA, dtype=float)
        self.Dec = np.asarray(Dec, dtype=float)
        self.spatial_type = "equatorial"

    @classmethod
    def from_arrays(
        cls,
        RA: np.ndarray,
        Dec: np.ndarray,
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
        spatial_type: str | None = "equatorial",
    ) -> "EventSample":
        """Create an EventSample from existing RA/Dec arrays (no new random draws)."""
        RA = np.asarray(RA, dtype=float)
        Dec = np.asarray(Dec, dtype=float)

        if RA.shape != Dec.shape:
            raise ValueError(f"RA and Dec must have the same shape, got {RA.shape} vs {Dec.shape}.")
        if RA.ndim != 1:
            raise ValueError(f"RA and Dec must be 1D arrays, got ndim={RA.ndim}.")

        obj = cls(n_events=int(RA.size), t0=t0, tf=tf, rng=rng)
        obj.RA = RA
        obj.Dec = Dec
        obj.spatial_type = spatial_type
        return obj

    def subset(self, mask: np.ndarray) -> "EventSample":
        """Return a new EventSample containing only events where mask is True."""
        if not self.is_populated():
            raise ValueError("RA/Dec are not set. Call sample_equatorial_coordinates() first.")

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.RA.shape:
            raise ValueError(f"Mask must have shape {self.RA.shape}, got {mask.shape}.")

        return EventSample.from_arrays(
            RA=self.RA[mask],
            Dec=self.Dec[mask],
            t0=self.t0,
            tf=self.tf,
            rng=self.rng,
            spatial_type=self.spatial_type,
        )

    def add_directional_exposure_for_window(
        self: "EventSample",
        exposure_model: "ExposureModel",
        window: "SkyWindow",
        parent_sample: "EventSample"
    ) -> None:
        """
        Attach sampled cumulative directional exposure values to this EventSample,
        using the *window definition* as the reference direction and the *parent sample*
        to set the overall event-rate normalization.

        Intended workflow
        -----------------
        1) Start from a full dataset (parent sample) spanning [t0, tf].
        2) Apply a sky-window selection to build a subsample:
            sub = parent.subset(window.contains(parent))
        3) Call this method on the subsample `sub` to generate per-event epsilon values:
            sub.add_directional_exposure_for_window(exposure_model, window, parent)

        Important notes
        ---------------
        - This method is designed for *window-selected subsamples*.
        It does not validate that the events actually lie inside `window`; it assumes
        you already applied the window cut via `subset(...)`.
        - No event times are generated or required. The method samples values directly
        in cumulative exposure space.

        Parameters
        ----------
        exposure_model : ExposureModel
            Provides the exposure-space sampling machinery and the mapping needed to
            compute the maximum cumulative directional exposure for the window centre.
        window : SkyWindow
            The sky region that defined this subsample. Its `centre` sets the reference
            direction used to compute max exposure, and is stored for provenance.
        parent_sample : EventSample
            The full (pre-window) dataset used to compute the expected rate as
            `exp_rate_exposure = parent_sample.n_events / max_dir_exposure`.

        Returns
        -------
        None
            The EventSample is modified in place. The following attributes are set
            (or overwritten):

            - self.dir_exposure
            - self.dir_exposure_method
        """

        # maximum cumulative directional exposure over [t0, tf] for this centre
        max_dir_exposure = exposure_model.max_directional_exposure(window.centre)

        # exposure-space event rate normalization from the parent sample
        exp_rate_exposure = parent_sample.n_events / max_dir_exposure

        # sample exposure values for this subsample's number of events
        eps, method = exposure_model.sample_directional_exposure(
            n_events=self.n_events,
            exp_rate_exposure=exp_rate_exposure,
            max_dir_exposure=max_dir_exposure,
        )

        eps = np.asarray(eps, dtype=float)

        # --- store results ---
        self.dir_exposure = eps
        self.dir_exposure_method = str(method)

    def has_exposure(self) -> bool:
        """Return True if directional exposure has been generated/assigned."""
        return getattr(self, "dir_exposure", None) is not None
