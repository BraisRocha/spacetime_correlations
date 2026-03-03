from .event_sample import EventSample
from .observatory import Observatory
from .skywindow import SkyWindow
from .exposure import ExposureModel

import numpy as np
import scipy.stats as scp
from __future__ import annotations

class WindowAnalysis:
    """Perform statistical analysis on events selected by a SkyWindow.

    Responsibilities (after refactor):
    - Build and hold the selected subsample (EventSample)
    - Ask ExposureModel for exposure mapping/sampling
    - Compute Lambda estimator / p-value
    - Optionally inject a flare (kept as your current approach)
    """

    def __init__(self, 
                 parent_sample: "EventSample", 
                 observatory: "Observatory", 
                 window: "SkyWindow", 
                 exposure_model: "ExposureModel"):
        
        if not parent_sample.is_populated():
            raise RuntimeError("Parent sample has no coordinates. Call sample_equatorial_coordinates() first.")
        if exposure_model is None:
            raise ValueError(
                "An ExposureModel must be provided. "
                "If you want uniform exposure, explicitly pass UniformExposureModel.")

        self.parent_sample = parent_sample
        self.observatory = observatory
        self.window = window
        self.exposure_model = exposure_model

        # Selection + uniform expected count (handled by SkyWindow)
        self.sample, self.mu_expected = self.window.select(parent_sample)

        # Convenience aliases (so old methods keep working for now)
        self.RA = self.sample.RA
        self.Dec = self.sample.Dec
        self.spatial_type = self.sample.spatial_type
        self.n_events = self.sample.n_events
        self.T_obs = self.sample.T_obs

        # Exposure scale (exposure is cumulative in your model, so exposure(T_obs) is max)
        self.max_dir_exposure = self.exposure_model.max_directional_exposure(
            T_obs=self.T_obs,
            centre=self.window.centre,
        )

        # Be explicit: rate per exposure unit
        self.exp_rate_exposure = self.mu_expected / self.max_dir_exposure

        # Later assigned
        self.dir_exposure = None
        self.directional_exposure_type = None

    # ---- Exposure sampling now delegated to ExposureModel ----
    def sample_directional_exposure(self, factor: int = 30) -> None:
        """Generate a sample of cumulative directional exposures for the selected events."""
        self.dir_exposure, self.directional_exposure_type = self.exposure_model.sample_directional_exposure(
            n_events=self.n_events,
            exp_rate=self.exp_rate_exposure,
            max_dir_exposure=self.max_dir_exposure,
            factor=factor)
        

    # ---- Statistics stays here ----
    def Lambda_estimator(self):
        """Compute Lambda estimator and p-value (gamma survival function)."""
        if self.dir_exposure is None:
            raise RuntimeError("Directional exposure not set. Call sample_directional_exposure() first.")

        delta_exp = np.diff(np.sort(self.dir_exposure))
        Lambda = -np.sum(np.log(1.0 - np.exp(-delta_exp * self.exp_rate_exposure)))
        p_value = scp.gamma.sf(Lambda, a=self.n_events - 1, scale=1.0)

        return Lambda, p_value

    # ---- Flare injection stays here (uses ExposureModel for exposures) ----
    def generate_flare(self, n_flare: int, t_flare: float) -> None:
        """Inject a flare: replace n_flare events with clustered RA/Dec and exposure times."""
        if self.dir_exposure is None:
            raise RuntimeError("Directional exposure not set. Call get_directional_exposure() first.")
        if n_flare > self.n_events:
            raise ValueError("Not enough events in the selected sample to inject that many flare events.")
        if t_flare <= 0 or t_flare > self.T_obs:
            raise ValueError("t_flare must be in (0, T_obs].")

        # Gaussian cluster around window centre
        points = self.rng.multivariate_normal(
            mean=self.window.centre,
            cov=np.eye(2) * self.window.radius**2,
            size=n_flare,
        )

        # Flare times inside [0, T_obs]
        t0 = self.rng.uniform(0.0, self.T_obs - t_flare)
        times = self.rng.uniform(t0, t0 + t_flare, size=n_flare)

        # Convert to cumulative directional exposure using the model
        exp = self.exposure_model.to_directional_exposure(times, centre=self.window.centre)

        # Inject into random indices
        idx = self.rng.choice(self.n_events, size=n_flare, replace=False)
        self.RA[idx] = points[:, 0]
        self.Dec[idx] = points[:, 1]
        self.dir_exposure[idx] = exp
