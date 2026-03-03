from .observatory import Observatory
from .skywindow import SkyWindow
from .exposure import ExposureModel
from .event_sample import EventSample
#from .flare import FlareCatalog
#from .analysis import WindowAnalysis
from .rng import RNGManager

__all__ = [
    "Observatory",
    "SkyWindow",
    "ExposureModel",
    "EventSample",
    "FlareCatalog",
    "WindowAnalysis",
    "RNGManager"
]

"""If classes inside spacetimecorr package are called I think
a good name could be 'import spacetimecorr as stc'
"""