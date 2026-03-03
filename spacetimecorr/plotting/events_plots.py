from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

if TYPE_CHECKING:
    from ..event_sample import EventSample

def plot_plain(
    sample: "EventSample",
    save_path: str | Path
) -> None:
    """
    Plain RA–Dec scatter plot.

    Parameters
    ----------
    sample : EventSample
        Must contain RA and Dec attributes.
    save_path : str | Path
        Output file path.
    """

    if not sample.is_populated():
        raise RuntimeError("Sample has no coordinates in it.")

    # Ensure numpy arrays

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        sample.RA,
        sample.Dec,
        s=5,
        alpha=0.6,
        edgecolors="none",
    )

    ax.set_xlabel("Right Ascension [degrees]")
    ax.set_xlim(0,360)
    ax.set_ylabel("Declination [degrees]")
    ax.set_ylim(-90,90)

    ax.set_title("Plain RA–Dec projection")

    fig.tight_layout()

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_hammer(
    sample: "EventSample", 
    save_path: str | Path
) -> None:

    if not sample.is_populated():
        raise RuntimeError("Sample has no coordinates in it.")

    # Convert to radians for Hammer
    ra = np.deg2rad(sample.RA)
    dec = np.deg2rad(sample.Dec)

    # Wrap RA into [-π, π]
    lon = (ra + np.pi) % (2 * np.pi) - np.pi
    lat = dec

    fig, ax = plt.subplots(figsize=(9, 5), subplot_kw={"projection": "hammer"})

    ax.scatter(lon, lat, s=2, alpha=0.6)
    ax.grid(True)

    ax.set_title("Hammer projection (Equatorial coordinates)")

     # ----- RA ticks 0 → 360 -----
    ra_ticks_deg = np.arange(0, 361, 60)

    # Convert ticks to projection coordinates
    ra_ticks_rad = np.deg2rad(ra_ticks_deg)
    ra_ticks_wrapped = (ra_ticks_rad + np.pi) % (2 * np.pi) - np.pi
    ra_ticks_wrapped = -ra_ticks_wrapped  # match inversion

    ax.set_xticks(ra_ticks_wrapped)
    ax.set_xticklabels([f"{int(t)}°" for t in ra_ticks_deg])

    # ----- Dec ticks -----
    dec_ticks_deg = np.arange(-60, 91, 30)
    ax.set_yticks(np.deg2rad(dec_ticks_deg))
    ax.set_yticklabels([f"{int(t)}°" for t in dec_ticks_deg])

    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_hammer_heatmap(sample: "EventSample", save_path: str | Path) -> None:
    """
    Plot sky positions from a sample as a heatmap on a Hammer projection.

    Parameters
    ----------
    sample : EventSample
        Must contain `RA` and `Dec` attributes in degrees.
        RA expected in [0, 360], Dec in [-90, 90].
    save_path : str or Path
        Output file path.
    """
    ra = np.asarray(sample.RA)
    dec = np.asarray(sample.Dec)

    # 1° bins
    ra_edges = np.linspace(0, 360, 361)
    dec_edges = np.linspace(-90, 90, 181)

    H, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])
    H_plot = np.log10(H + 1)

    proj = ccrs.Hammer(central_longitude=0)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=proj)
    ax.set_global()

    mesh = ax.pcolormesh(
        ra_edges,
        dec_edges,
        H_plot,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    # Astronomy convention: RA increases to the left
    ax.invert_xaxis()

    # Gridlines (also handles "ticks" for non-rectangular projections)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        color="gray",
        alpha=0.4,
        linestyle="--",
        xlocs=np.arange(0, 361, 60),
        ylocs=np.arange(-60, 61, 30),
    )

    # Cartopy draws lon/lat labels; we only want bottom labels typically
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False  # hide Dec labels on the side if you prefer
    gl.bottom_labels = True

    # Colorbar + title
    cbar = fig.colorbar(mesh, ax=ax, pad=0.05, shrink=0.85)
    cbar.set_label(r"$\log_{10}(\mathrm{count}+1)$")
    ax.set_title("Hammer Projection Sky Map (RA 0–360)")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)