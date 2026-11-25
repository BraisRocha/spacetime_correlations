"""
Sky maps of a blind search on the mock samples over the HEALPix pixels.

Reads a run of ``run_blind_search_mock.py`` (``results.npz`` and
``metadata.json`` of every dataset) with the H0 Monte Carlo p-values of
its pixels (``pvalues_mc.npz``, from
``h0_mc_distribution/mc_pvalues_blind_search.py``), and draws for every
dataset ``skymap.png``: a Hammer-projected map with every tested pixel
coloured by ``-log10`` of its p-value. All the datasets share the colour
scale, from 0 to ``-log10 p`` at ``SIGMA_MAX``, so that the maps compare;
pixels beyond it saturate. A grey cap marks the declination band above the
field of view; pixels not tested (outside the field of view, or with fewer
than 2 events) are left white.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import healpy as hp
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scp
from tqdm import tqdm

# ----------------------------------------------------------------------
# Global style: load the matplotlibrc of the plots/ folder
# ----------------------------------------------------------------------
RC_FILE = Path(__file__).resolve().parents[2] / "plots" / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)

FIGSIZE = (3.6, 2.6)
SIGMA_MAX = 4.0         # top of the colour scale, as a one-sided significance


def sci_label(n: int) -> str:
    """LaTeX scientific notation for a count, e.g. ``5\\times10^{5}``."""
    exponent = int(np.floor(np.log10(n)))
    mantissa = n / 10**exponent

    if np.isclose(mantissa, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:.1f}\times10^{{{exponent}}}"


# ----------------------------------------------------------------------
# Sky-map helpers
# ----------------------------------------------------------------------

def _draw_fov_cap(ax, dec_max_deg: float, *, color: str = "0.85") -> None:
    """Shade the out-of-FoV declination cap (``dec > dec_max``)."""
    lon = np.linspace(-np.pi, np.pi, 400)
    ax.fill_between(
        lon, np.deg2rad(dec_max_deg), 0.5 * np.pi,
        color=color, linewidth=0.0, zorder=0,
    )


def _style_axes(ax) -> None:
    """Apply the shared graticule, ticks and labels to a Hammer panel.

    Same ticks as ``plot_targeted_search.py``. The graticule and
    the longitude labels lie over the coloured pixels, so they are white.
    """
    xticks_deg = np.array([-150, -90, -30, 30, 90, 150])
    yticks_deg = np.array([-60, -30, 0, 30, 60])

    # Projected longitude is -RA, so the label of each tick is -lon.
    ax.set_xticks(np.deg2rad(xticks_deg))
    ax.set_xticklabels([rf"${-v:d}^\circ$" for v in xticks_deg])
    ax.set_yticks(np.deg2rad(yticks_deg))
    ax.set_yticklabels([rf"${v:d}^\circ$" for v in yticks_deg])

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\delta$")
    ax.grid(True, color="white", linewidth=0.4, alpha=0.8)

    # The longitude labels sit on the equator, inside the map, where the
    # pixels would hide them. Redraw them as high-zorder white text, slightly
    # larger, with a thin black contour, and hide the originals. A thicker
    # contour swallows the thin serif strokes.
    ax.figure.canvas.draw()
    for label in ax.get_xticklabels():
        text = label.get_text()
        if not text:
            continue
        ax.text(
            *label.get_position(), text,
            transform=label.get_transform(),
            ha=label.get_ha(), va=label.get_va(),
            fontsize=label.get_fontsize() + 1, color="white",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=0.25, foreground="black")],
        )
        label.set_visible(False)


def _pixel_image(
    values: np.ndarray,
    pixels: np.ndarray,
    nside: int,
    n_lon: int = 1440,
    n_lat: int = 720,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample per-pixel values on a regular (longitude, latitude) grid.

    Each grid cell takes the value of the HEALPix pixel containing its
    centre, so the pixels can be drawn with ``pcolormesh`` on the Hammer
    axes. Cells in pixels with no value (not in ``pixels``, or NaN) are NaN.

    Returns the longitude and latitude cell edges (rad) and the
    ``(n_lat, n_lon)`` image.
    """
    lon_edges = np.linspace(-np.pi, np.pi, n_lon + 1)
    lat_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat + 1)
    lon, lat = np.meshgrid(
        0.5 * (lon_edges[1:] + lon_edges[:-1]),
        0.5 * (lat_edges[1:] + lat_edges[:-1]),
    )

    ra = (-np.degrees(lon)) % 360.0          # projected longitude is -RA
    cell_pixel = hp.ang2pix(nside, ra, np.degrees(lat), lonlat=True)

    full_map = np.full(hp.nside2npix(nside), np.nan)
    full_map[pixels] = values
    return lon_edges, lat_edges, full_map[cell_pixel]


def plot_skymap(dataset_dir: Path) -> None:
    """``-log10(p)`` sky map of the MC p-values of one dataset."""
    data = np.load(dataset_dir / "results.npz")
    pvalues = np.load(dataset_dir / "pvalues_mc.npz")["pvalues_mc"]
    with open(dataset_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    nside = int(metadata["nside"])
    latitude = float(metadata["latitude_pa_deg"])
    dec_max_fov = min(90.0, latitude + float(metadata["theta_max_deg"]))

    # -log10(p): the bulk of the pixels sits near 0, at the dark end of the
    # colormap, and small p-values light up.
    neglog = -np.log10(pvalues)
    cmap = mpl.colormaps["magma"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(-np.log10(scp.norm.sf(SIGMA_MAX))))

    fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={"projection": "hammer"})
    _draw_fov_cap(ax, dec_max_fov)
    lon_edges, lat_edges, image = _pixel_image(neglog, data["pixels"], nside)
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, np.ma.masked_invalid(image),
        cmap=cmap, norm=norm, shading="flat", rasterized=True, zorder=1,
    )
    _style_axes(ax)

    cbar = fig.colorbar(
        mesh, ax=ax, orientation="horizontal", location="bottom",
        pad=0.12, fraction=0.05, aspect=30, extend="max",
    )
    cbar.set_label(r"$-\log_{10} p$", labelpad=1)
    cbar.outline.set_linewidth(mpl.rcParams["axes.linewidth"])
    cbar.ax.tick_params(direction="out")

    dataset = metadata["dataset"].replace("_", r"\_")
    T_obs_yr = float(metadata["T_obs_days"]) / 365.25
    ax.set_title(
        rf"{dataset}, $N_{{\rm total}} = {sci_label(int(metadata['n_total']))}$, "
        rf"$T_{{\rm obs}} = {T_obs_yr:.1f}\,\mathrm{{yr}}$"
    )

    fig.savefig(dataset_dir / "skymap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(run_dir: Path) -> None:
    """Sky map of every dataset of a blind-search run with MC p-values."""
    datasets = sorted(
        (d for d in run_dir.glob("dataset_*") if (d / "pvalues_mc.npz").exists()),
        key=lambda d: int(re.findall(r"\d+", d.name)[-1]),
    )
    for d in tqdm(datasets, desc="Sky maps", disable=None):
        plot_skymap(d)
    print(f"Saved the sky maps of {len(datasets)} datasets in {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sky maps of the MC p-values of a blind-search run of the mock samples."
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="run_blind_search_mock.py run, with pvalues_mc.npz in its dataset_* folders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().run_dir)
