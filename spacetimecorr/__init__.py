from .observatory import Observatory
from .skywindow import SkyWindow
from .exposure import ExposureModel
from .event_sample import EventSample
from .flare import Flare
from .statistics import (
    lambda_estimator,
    theoretical_lambda_estimator,
    spatial_estimator,
    tau_method,
)
from .rng import RNGManager

__all__ = [
    "Observatory",
    "SkyWindow",
    "ExposureModel",
    "EventSample",
    "Flare",
    "RNGManager",
    "lambda_estimator",
    "theoretical_lambda_estimator",
    "spatial_estimator",
    "tau_method",
]

"""
Recommended
-----------
    Import spacetimecorr package as `stc` in your script.
"""
