from .observatory import Observatory
from .skywindow import SkyWindow
from .exposure import ExposureModel
from .event_sample import EventSample
#from .flare import FlareCatalog
from .statistics import (
    lambda_estimator,
    theoretical_lambda_estimator,
)
from .rng import RNGManager

__all__ = [
    "Observatory",
    "SkyWindow",
    "ExposureModel",
    "EventSample",
    "FlareCatalog",
    "WindowAnalysis",
    "RNGManager",
    "lambda_estimator"
]

"""If classes inside spacetimecorr package are called I think
a good name could be 'import spacetimecorr as stc'
"""