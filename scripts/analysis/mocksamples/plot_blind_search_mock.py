"""
Plots for a blind search on a mock data sample over a HEALPix grid.

Reads a single run produced by ``run_blind_search_mock.py`` (``results.npz``
plus ``metadata.json``) and makes two figures:

- ``significances.png``: the distribution of the per-pixel significances
  of the Lambda and Poisson tests, one on top of the other, as saved by the
  run (one-sided, from ``pvalue_to_sigma``). The standard normal expected
  for a sample with no signal is drawn for reference.
- ``skymap.png``: two Hammer-projected panels (Poisson, Lambda) with every
  tested pixel coloured by ``-log10`` of its raw p-value. A grey cap marks
  the declination band above the field of view; pixels that were not
  tested (on the FoV edge, or with fewer than 2 events) are left white.

Pixels not tested carry NaN and are left out of both. Pixels whose p-value
is exactly 1 have a significance of -inf and are left out of the histogram.
"""
from __future__ import annotations

import json
from pathlib import Path

import healpy as hp
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scp
import spacetimecorr as stc
# ----------------------------------------------------------------------
# Global style: load the matplotlibrc of the plots/ folder
# ----------------------------------------------------------------------
RC_FILE = Path(__file__).resolve().parents[2] / "plots" / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)

# Sky-map figure geometry, as in plot_targeted_search.py: JCAP full-width
# figure (~7 in), height from a fixed aspect.
FIG_WIDTH_IN = 7.0
FIG_ASPECT = 2.1 / 7.2        # height / width
FIGSIZE = (FIG_WIDTH_IN, FIG_WIDTH_IN * FIG_ASPECT)


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


def _draw_panel(
    fig,
    ax,
    values: np.ndarray,
    pixels: np.ndarray,
    nside: int,
    *,
    title: str,
    dec_max_deg: float,
    cmap,
    norm,
) -> None:
    """Draw one ``-log10(p)`` sky panel with its own horizontal colour bar."""
    _draw_fov_cap(ax, dec_max_deg)

    lon_edges, lat_edges, image = _pixel_image(values, pixels, nside)
    mesh = ax.pcolormesh(
        lon_edges, lat_edges, np.ma.masked_invalid(image),
        cmap=cmap, norm=norm, shading="flat", rasterized=True, zorder=1,
    )

    _style_axes(ax)
    ax.set_title(title)

    cbar = fig.colorbar(
        mesh, ax=ax, orientation="horizontal", location="bottom",
        pad=0.12, fraction=0.05, aspect=30,
    )
    cbar.set_label(r"$-\log_{10} p$", labelpad=1)
    cbar.outline.set_linewidth(mpl.rcParams["axes.linewidth"])
    cbar.ax.tick_params(direction="out")


def plot_skymap(data, metadata: dict, results_dir: Path) -> None:
    """Two-panel (Poisson, Lambda) ``-log10(p)`` sky map of one run."""
    nside = int(metadata["nside"])
    pixels = data["pixels"]

    latitude = float(metadata["latitude_pa_deg"])
    dec_max_fov = min(90.0, latitude + float(metadata["theta_max_deg"]))

    # Raw (unpenalised) p-values as -log10(p): the bulk of the pixels sits
    # near 0, at the dark end of the colormap, and small p-values light up.
    # A p-value of exactly 0 (underflow) is floored at the smallest float.
    # Shared scale from 0 to the 99th percentile across both panels, so the
    # larger values reach the bright end; the top 1% saturate.
    tiny = np.finfo(float).tiny
    neglog_poisson = -np.log10(np.maximum(data["pvalues_poisson"], tiny))
    neglog_lambda = -np.log10(np.maximum(data["pvalues_lambda"], tiny))

    cmap = mpl.colormaps["magma"]
    vmax = float(np.nanpercentile(np.concatenate([neglog_poisson, neglog_lambda]), 99))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=FIGSIZE,
        subplot_kw={"projection": "hammer"},
    )
    fig.subplots_adjust(wspace=0.20, top=0.86)

    _draw_panel(
        fig, ax0, neglog_poisson, pixels, nside,
        title="Poisson", dec_max_deg=dec_max_fov, cmap=cmap, norm=norm,
    )
    _draw_panel(
        fig, ax1, neglog_lambda, pixels, nside,
        title=r"$\Lambda$", dec_max_deg=dec_max_fov, cmap=cmap, norm=norm,
    )

    dataset = metadata["dataset"].replace("_", r"\_")
    T_obs_yr = float(metadata["T_obs_days"]) / 365.25
    fig.suptitle(
        rf"Mock sample {dataset}, "
        rf"$N_{{\rm total}} = {sci_label(int(metadata['n_total']))}$, "
        rf"$T_{{\rm obs}} = {T_obs_yr:.1f}\,\mathrm{{yr}}$",
        y=1.0,
    )

    fig.savefig(results_dir / "skymap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pvalues(data, metadata: dict, results_dir: Path) -> None:
    """Histogram of the raw Lambda and Poisson p-values of one run.

    With no signal the p-values are uniform on [0, 1], drawn for reference.
    """
    p_lambda = data["pvalues_lambda"][np.isfinite(data["pvalues_lambda"])]
    p_poisson = data["pvalues_poisson"][np.isfinite(data["pvalues_poisson"])]

    bins = np.linspace(0.0, 1.0, 41)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.hist(p_lambda, bins=bins, density=False, histtype="step",
            linewidth=1.2, label=r"$\Lambda$")
    ax.hist(p_poisson, bins=bins, density=False, histtype="step",
            linewidth=1.2, label="Poisson")
    #ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               #label=r"$\mathcal{U}(0, 1)$")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$p$-value")
    ax.set_ylabel("Pixels")
    ax.set_yscale("log")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(results_dir / "pvalues.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(results_dir: str | Path) -> None:
    """
    Load a saved blind-search run on a mock sample and create the plots.
    """
    results_dir = Path(results_dir)

    results_path = results_dir / "results.npz"
    metadata_path = results_dir / "metadata.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Could not find results file: {results_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata file: {metadata_path}")

    data = np.load(results_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # ------------------------------------------------------------------
    # Load outputs and metadata
    # ------------------------------------------------------------------
    # Finite significances only: NaN for pixels not tested, -inf for
    # p-values of exactly 1.
    sigma_lambda = data["sigma_lambda"][np.isfinite(data["sigma_lambda"])]
    sigma_poisson = data["sigma_poisson"][np.isfinite(data["sigma_poisson"])]

    nside = metadata["nside"]
    n_tested = metadata["n_windows_tested"]
    n_total = metadata["n_total"]

    # ------------------------------------------------------------------
    # Significance plot
    # ------------------------------------------------------------------
    # One-sided Gaussian-equivalent significances; with no signal both
    # follow a standard normal. Symmetric range wide enough for every value.
    sigma_lim = max(4.0, np.ceil(np.max(np.abs(np.r_[sigma_lambda, sigma_poisson]))))
    bins = np.arange(-sigma_lim, sigma_lim + 1e-9, 0.25)
    x = np.linspace(-sigma_lim, sigma_lim, 400)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.hist(sigma_lambda, bins=bins, density=True, histtype="step",
            linewidth=1.2, label=r"$\Lambda$")
    ax.hist(sigma_poisson, bins=bins, density=True, histtype="step",
            linewidth=1.2, label="Poisson")
    ax.plot(x, scp.norm.pdf(x), color="black", linestyle="--", linewidth=0.8,
            label=r"$\mathcal{N}(0, 1)$")

    ax.set_xlim(-sigma_lim, sigma_lim)
    ax.set_xlabel(r"Significance [$\sigma$]")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend(loc="upper left")

    info_text = (
        rf"$N_{{\rm side}} = {nside}$" "\n"
        rf"$N_{{\rm pix}} = {n_tested}$" "\n"
        rf"$N_{{\rm total}} = {sci_label(n_total)}$"
    )
    ax.text(
        0.98,
        0.97,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(results_dir / "significances.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # p-value distribution
    # ------------------------------------------------------------------
    plot_pvalues(data, metadata, results_dir)

    # ------------------------------------------------------------------
    # p-value sky map
    # ------------------------------------------------------------------
    plot_skymap(data, metadata, results_dir)


if __name__ == "__main__":
    # The project root, found from this file's own location
    # (scripts/analysis/mocksamples/ is three levels below it), so the path
    # works on any machine.
    project_root = Path(__file__).resolve().parents[3]

    # Change this to the run you want to plot
    run_dir = project_root / "output" / "analysis" / "blind_search_mock"
    run_id = "dataset_50_20261015_150812"
    main(run_dir / run_id)
