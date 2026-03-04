from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.stats as scp

if TYPE_CHECKING:
    from ..event_sample import EventSample

def plot_events_vs_exposure(
    sample: "EventSample", 
    save_path: Path
) -> None:
    """
    Plot cumulative number of events vs directional exposure.
    """
    if not sample.has_exposure:
        raise RuntimeError("Sample has no exposure values in it.")
    
    x = np.asarray(sample.dir_exposure)
    y = np.arange(1, sample.n_events + 1)

    fig, ax = plt.subplots()

    ax.plot(x, y, marker="o")

    ax.set_xlabel("Directional exposure")
    ax.set_ylabel("Number of events")

    fig.tight_layout()

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_exponential_exposure_diffs(
    sample: "EventSample", 
    save_path: Path, 
    *, 
    compare_with: np.ndarray = None
    ) -> None:
    """
    Histogram of exposure increments Delta E = E[i] - E[i-1] from sample.dir_exposure.

    If events are sampled as a Poisson process in exposure,
    the increments often follow an exponential-like distribution.

    Parameters
    ----------
    sample : EventSample
        Must have attribute `dir_exposure` as a 1D iterable of exposure values.
    save_path : Path
        Where to save the figure.
    compare_with : array-like, optional
        Another distribution of exposure differences to compare against.
    """
    exposure = np.asarray(sample.dir_exposure, dtype=float)

    if exposure.size < 2:
        raise ValueError("Need at least 2 exposure values to compute differences.")

    # Sort to ensure consecutive differences make sense even if exposure isn't ordered
    exposure_sorted = np.sort(exposure)

    diffs = np.diff(exposure_sorted)

    # Guardrails: diffs should be non-negative (cumulative exposure)
    if np.any(diffs < 0):
        raise ValueError("Found negative exposure increments after sorting; check exposure definition.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diffs, bins="fd", histtype="step", density=True, label="sample")

    if compare_with is not None:
        compare_with = np.asarray(compare_with, dtype=float)
        ax.hist(compare_with, bins="fd", histtype="step", density=True, label="comparison")

    ax.set_xlabel(r"$\Delta \epsilon(t)$")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of directional exposure differences")

    fig.tight_layout()

    Path(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)





