from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

if TYPE_CHECKING:
    from ..event_sample import EventSample
def plot_events_vs_exposure(
    sample: "EventSample", 
    save_path: Path,
    *,
    invert:bool = False) -> None:
    """
    Plot cumulative number of events vs directional exposure.
    """
    if not sample.has_exposure:
        raise RuntimeError("Sample has no exposure values in it.")
    
    x = np.asarray(sample.dir_exposure)
    y = np.arange(1, sample.n_events + 1)

    fig, ax = plt.subplots()

    if invert:
        ax.plot(y, x, marker="o")
    else:
        ax.plot(x, y, marker="o")

    ax.set_xlabel("Directional exposure")
    ax.set_ylabel("Number of events")

    fig.tight_layout()

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)



