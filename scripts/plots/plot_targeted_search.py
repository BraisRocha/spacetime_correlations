"""
Significance sky map for a targeted multi-flare search.

Two Hammer-projected panels show the ``n_f`` search windows of a
:func:`run_targeted_search` run as circles of the search radius, each
colour-coded by its one-sided Gaussian-equivalent significance ``sigma``:

    left panel  - Poisson counting test,
    right panel - Lambda test.

Each panel title reports the Fisher-combined significance of its p-values,
computed with :func:`spacetimecorr.fisher_sigma`: Fisher's statistic
``X = -2 sum_j ln p_j ~ chi2_{2 n_f}`` under the null of independent,
uniform ``p_j``; its one-sided Gaussian-equivalent significance is
evaluated in log space for tail accuracy.

A grey cap marks the declination band above the observatory field of view
(``dec > latitude + theta_max``), which the analysis never probes.

Reads a single run produced by ``run_targeted_search.py`` (``results.npz``
plus ``metadata.json``).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from spacetimecorr.statistics import fisher_sigma, pvalue_to_sigma

# Zenith-angle cut defining the field of view (deg). Matches the
# ExposureModel default / Auger SD standard cut used by the analysis.
THETA_MAX_DEG = 60.0

# Lower limit of the significance colour scale (sigma). The upper limit is
# set per run from the largest per-window significance across both panels.
SIGMA_MIN = 0.0

# Figure geometry for the paper. The width is fixed by the journal column and
# the height follows a chosen aspect, so the two-panel layout (and the
# data-coordinate window circles) scales as a whole. Font and line sizes are
# NOT scaled here -- they stay at the journal values set in matplotlibrc_test.
# JCAP double-column / full-width figure: ~170-180 mm = 6.7-7.1 in.
FIG_WIDTH_IN = 7.0
FIG_ASPECT = 2.1 / 7.2        # height / width
FIGSIZE = (FIG_WIDTH_IN, FIG_WIDTH_IN * FIG_ASPECT)

# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
RC_FILE = Path(__file__).resolve().parent / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def _load(run_dir: Path) -> tuple[dict, dict]:
    """Return ``(results, metadata)`` for a single targeted-search run.

    ``results`` is the ``results.npz`` archive (per-window arrays);
    ``metadata`` is the parsed ``metadata.json`` (scalar parameters).
    """
    with np.load(run_dir / "results.npz") as npz:
        results = {k: npz[k] for k in npz.files}
    with (run_dir / "metadata.json").open() as fh:
        metadata = json.load(fh)
    return results, metadata

# ------------------------------------------------------------------
# Projection helpers
# ------------------------------------------------------------------

def _ra_to_x(ra_deg: np.ndarray) -> np.ndarray:
    """Map RA (deg) to the Hammer x-coordinate (rad), astronomical sense.

    RA increases to the left: a source at ``RA`` is placed at the projected
    longitude ``-RA`` wrapped to ``[-180, 180)``, matching the signed tick
    labels on the x-axis.
    """
    ra_wrapped = ((np.asarray(ra_deg, dtype=float) + 180.0) % 360.0) - 180.0
    return -np.deg2rad(ra_wrapped)


def _draw_fov_cap(ax, dec_max_deg: float, *, color: str = "0.85") -> None:
    """Shade the out-of-FoV declination cap (``dec > dec_max``)."""
    lon = np.linspace(-np.pi, np.pi, 400)
    ax.fill_between(
        lon, np.deg2rad(dec_max_deg), 0.5 * np.pi,
        color=color, linewidth=0.0, zorder=0,
    )


def _style_axes(ax) -> None:
    """Apply the shared graticule, ticks and labels to a Hammer panel."""
    xticks_deg = np.array([-150, -90, -30, 30, 90, 150])
    yticks_deg = np.array([-60, -30, 0, 30, 60])

    ax.set_xticks(np.deg2rad(xticks_deg))
    ax.set_xticklabels([rf"${v:d}^\circ$" for v in xticks_deg])
    ax.set_yticks(np.deg2rad(yticks_deg))
    ax.set_yticklabels([rf"${v:d}^\circ$" for v in yticks_deg])

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\delta$")
    ax.grid(True, color="0.7", linewidth=0.4, alpha=0.8)
    ax.set_axisbelow(True)

    # The window circles are drawn above the axis, so the inline longitude
    # labels (they sit on the equator) get hidden where a circle lands on
    # them. A per-label zorder cannot lift a tick label above a sibling
    # collection, so redraw the longitude labels as standalone, high-zorder
    # text with a thin white halo and hide the originals. The latitude labels
    # sit outside the map and are left untouched.
    ax.figure.canvas.draw()
    for label in ax.get_xticklabels():
        text = label.get_text()
        if not text:
            continue
        ax.text(
            *label.get_position(), text,
            transform=label.get_transform(),
            ha=label.get_ha(), va=label.get_va(),
            fontsize=label.get_fontsize(), color=label.get_color(),
            zorder=5,
            path_effects=[pe.withStroke(linewidth=0.6, foreground="white")],
        )
        label.set_visible(False)


# ------------------------------------------------------------------
# Combined significance (Fisher's method)
# ------------------------------------------------------------------

def combined_fisher_sigma(pvalues: np.ndarray) -> float:
    """Fisher-combined significance of a set of per-window p-values.

    Non-finite entries (windows that failed and were stored as NaN) are
    dropped; the rest are combined via
    :func:`spacetimecorr.statistics.fisher_sigma`, which returns the
    one-sided Gaussian-equivalent significance of Fisher's statistic.
    """
    finite = np.isfinite(pvalues)
    return float(fisher_sigma(pvalues[finite]))


# ------------------------------------------------------------------
# Panel
# ------------------------------------------------------------------

def _draw_panel(
    fig,
    ax,
    ra: np.ndarray,
    dec: np.ndarray,
    pvalues: np.ndarray,
    *,
    title: str,
    dec_max_deg: float,
    search_radius_deg: float,
    cmap,
    norm,
) -> None:
    """Draw one significance sky panel with its own horizontal colour bar.

    Each window is a :class:`~matplotlib.patches.Circle` of the search
    radius drawn in data coordinates, so it tracks the Hammer projection and
    rescales automatically with the axes (no pixel/dpi conversion needed).
    """
    _draw_fov_cap(ax, dec_max_deg)

    finite = np.isfinite(pvalues)
    sigma = pvalue_to_sigma(pvalues[finite])

    size_factor = 4

    radius_rad = np.deg2rad(size_factor*search_radius_deg)
    circles = [
        Circle((x, y), radius=radius_rad)
        for x, y in zip(_ra_to_x(ra[finite]), np.deg2rad(dec[finite]))
    ]
    windows = PatchCollection(circles, cmap=cmap, norm=norm, zorder=3)
    windows.set_array(sigma)
    windows.set_edgecolor("0.15")
    windows.set_linewidth(mpl.rcParams["axes.linewidth"])
    ax.add_collection(windows, autolim=False)

    _style_axes(ax)
    ax.set_title(title)

    cbar = fig.colorbar(
        windows, ax=ax, orientation="horizontal", location="bottom",
        pad=0.12, fraction=0.05, aspect=30,
    )
    cbar.set_label(r"$\sigma$", labelpad=1)
    cbar.outline.set_linewidth(mpl.rcParams["axes.linewidth"])
    cbar.ax.tick_params(direction="out")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(run_dir: str | Path, output_dir: str | Path) -> None:
    """Build the two-panel significance sky map for one targeted-search run."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results, metadata = _load(run_dir)

    centres = results["centres"]            # (N, 2) -> [RA_deg, Dec_deg]
    ra, dec = centres[:, 0], centres[:, 1]
    p_poisson = results["pvalues_poisson"]
    p_lambda = results["pvalues_lambda"]

    latitude = float(metadata["latitude_pa_deg"])
    dec_max_fov = min(90.0, latitude + THETA_MAX_DEG)
    search_radius_deg = float(metadata["search_radius_deg"])

    fisher_sig_poisson = combined_fisher_sigma(p_poisson)
    fisher_sig_lambda = combined_fisher_sigma(p_lambda)

    # ------------------------------------------------------------------
    # Colour scale (shared): floor at SIGMA_MIN, top at the largest
    # per-window significance across both panels.
    # ------------------------------------------------------------------
    cmap = mpl.colormaps["magma"]
    all_pvalues = np.concatenate([p_poisson, p_lambda])
    sigma_max = float(np.max(pvalue_to_sigma(all_pvalues[np.isfinite(all_pvalues)])))
    norm = mpl.colors.Normalize(vmin=SIGMA_MIN, vmax=sigma_max)

    # ------------------------------------------------------------------
    # Figure: two Hammer panels, each over its own horizontal colour bar
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=FIGSIZE,
        subplot_kw={"projection": "hammer"},
    )
    fig.subplots_adjust(wspace=0.20, top=0.86)

    _draw_panel(
        fig, ax0, ra, dec, p_poisson,
        title=rf"Poisson, $\sigma_{{\rm Fisher}} = {fisher_sig_poisson:.1f}$",
        dec_max_deg=dec_max_fov, search_radius_deg=search_radius_deg,
        cmap=cmap, norm=norm,
    )
    _draw_panel(
        fig, ax1, ra, dec, p_lambda,
        title=rf"$\Lambda$, $\sigma_{{\rm Fisher}} = {fisher_sig_lambda:.1f}$",
        dec_max_deg=dec_max_fov, search_radius_deg=search_radius_deg,
        cmap=cmap, norm=norm,
    )

    # ------------------------------------------------------------------
    # Suptitle from metadata (duration varies per source -> show range;
    # intensity is shared across flares -> single S/N).
    # ------------------------------------------------------------------
    n_flares = int(metadata["number_of_flares"])
    dur_lo_h, dur_hi_h = metadata["flare_duration_range_hours"]
    dur_hi_d = dur_hi_h / 24.0
    snr = float(metadata["flare_intensity"])
    fig.suptitle(
        rf"$n_f = {n_flares}$, "
        rf"$\Delta t_{{\rm flare}} \in [{dur_lo_h:g}\,\mathrm{{h}},\,"
        rf"{dur_hi_d:g}\,\mathrm{{d}}]$, "
        rf"$SNR = {snr:.2f}$",
        y=1.0,
    )

    out_path = output_dir / "targeted_search.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Edit to point at a run_target_search.py output directory.
    run_dir = Path(
        "/home/brais/PhD/dev/stc_project/output/scripts/targeted_search/"
        "20260617_111354_seed42"
    )
    output_dir = run_dir / "figures"
    main(run_dir=run_dir, output_dir=output_dir)
