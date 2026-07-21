"""
Paper-quality 2D sensitivity map on the (flare duration, flare intensity)
grid from run_grid_p50.py.

Color encodes the Gaussian-equivalent significance (in sigma) of the
median (``PERCENTILE``-th percentile) p-value across the per-cell
simulation ensemble.

Each grid cell stores, for both the Poisson and Lambda tests, the full
per-simulation p-value distribution (assembled from the per-job pickles
by ``merge_grid_pvalues``). For every cell this script takes the median
p-value over the simulation axis and converts it to a one-sided
Gaussian-equivalent significance. Because both test p-values are
monotonic in their underlying statistic, the median p-value equals the
p-value of the median statistic, so this reproduces the legacy
"p50 of the statistic" map while keeping the full distribution available.

Left panel  - Poisson counting test.
Right panel - Lambda test.
"""
from __future__ import annotations

import json
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from spacetimecorr.statistics import pvalue_to_sigma

# Percentile of the per-cell p-value distribution used for the map.
PERCENTILE = 50.0

# Gaussian-equivalent significance contours (sigma) overlaid on each map.
THRESHOLD_LEVELS = (1.0, 2.0, 3.0, 5.0)

# Reference flare durations drawn as dashed vertical guides.
REF_LABELS = ["hour", "day", "week", "mth", "year"]
REF_DAYS = [1.0 / 24.0, 1.0, 7.0, 30.0, 365.0]

# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
RC_FILE = Path(__file__).resolve().parent / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)


# ==================================================================
# Data loading
# ==================================================================

def _sigma_with_nan(p: np.ndarray) -> np.ndarray:
    """Convert p-values to sigma, preserving NaN for missing cells.

    ``pvalue_to_sigma`` rejects non-finite inputs, but merged grids may have
    cells with no data (all-NaN over the simulation axis). Those stay NaN here
    and render blank; only finite cells are passed to ``pvalue_to_sigma``.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if finite.any():
        out[finite] = pvalue_to_sigma(p[finite])
    return out


def _load_merged_pvalues(path: Path) -> tuple:
    """Load a merged ``(durations, intensities, pvalues)`` pickle.

    ``pvalues`` has shape ``(n_durations, n_intensities, n_simulations)``.
    """
    with path.open("rb") as fh:
        durations, intensities, pvalues = pickle.load(fh)
    durations = np.asarray(durations, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    # Map any non-finite p-value (e.g. an undefined Lambda sample stored as
    # +/-inf) to NaN so the per-cell ``nanpercentile`` simply drops those
    # samples instead of letting them poison the quantile.
    pvalues[~np.isfinite(pvalues)] = np.nan
    return durations, intensities, pvalues


def _read_expected_n_and_tobs(data_dir: Path) -> tuple[float, float]:
    """Read ``expected_n`` and ``T_obs_days`` from any per-job metadata file.

    Both quantities are identical across grid cells (they depend only on
    the window, exposure and ``n_total``), so the first metadata file found
    is sufficient.
    """
    meta_files = sorted(data_dir.glob("metadata_job*.json"))
    if not meta_files:
        meta_files = sorted(data_dir.glob("metadata*.json"))
    if not meta_files:
        raise FileNotFoundError(
            f"No metadata_job*.json files found in {data_dir}"
        )
    with meta_files[0].open() as fh:
        meta = json.load(fh)
    return float(meta["expected_n"]), float(meta["time"]["T_obs_days"])


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges for ``pcolormesh`` from cell centres.

    Interior edges are the midpoints between centres; the outer edges are
    extended by half a cell.
    """
    c = np.asarray(centres, dtype=float)
    mids = 0.5 * (c[:-1] + c[1:])
    return np.concatenate((
        [c[0] - (mids[0] - c[0])],
        mids,
        [c[-1] + (c[-1] - mids[-1])],
    ))


def _load(run_dir: Path, percentile: float = PERCENTILE) -> tuple:
    """Return significance grids and axes needed for plotting.

    Reads the merged Lambda and Poisson p-value pickles, takes the
    ``percentile``-th percentile of the p-value distribution in each cell
    (over the simulation axis), and converts it to sigma.

    Returns
    -------
    sig_lam_grid   : (n_intensities, n_durations) significance of Lambda
    sig_poi_grid   : (n_intensities, n_durations) significance of Poisson
    intensities_snr: (n_intensities,) flare intensity as signal-to-noise ratio
    x_log          : (n_durations,) log10(duration_days / T_obs_days)
    expected_n     : float, expected background events in window
    T_obs_years    : float
    """
    data_dir = run_dir / "data" if (run_dir / "data").exists() else run_dir

    durations, intensities, pvals_lam = _load_merged_pvalues(
        data_dir / "pvalues_lambda_merged.pkl"
    )
    durations_poi, intensities_poi, pvals_poi = _load_merged_pvalues(
        data_dir / "pvalues_poisson_merged.pkl"
    )

    if not (np.array_equal(durations, durations_poi)
            and np.array_equal(intensities, intensities_poi)):
        raise ValueError(
            "Lambda and Poisson merged pickles have mismatched grid axes."
        )

    expected_n, T_obs_days = _read_expected_n_and_tobs(data_dir)
    T_obs_years = T_obs_days / 365.25

    # Median p-value per cell, then sigma. Arrays are (n_dur, n_int);
    # transpose to (n_int, n_dur) for pcolormesh (y=intensity, x=duration).
    # Cells with no data (all-NaN over the simulation axis) stay NaN and
    # render blank; ``pvalue_to_sigma`` only sees the finite cells.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_p_lam = np.nanpercentile(pvals_lam, percentile, axis=2)
        median_p_poi = np.nanpercentile(pvals_poi, percentile, axis=2)
    sig_lam_grid = _sigma_with_nan(median_p_lam).T
    sig_poi_grid = _sigma_with_nan(median_p_poi).T

    n_missing = int(np.isnan(sig_lam_grid).sum())
    if n_missing:
        warnings.warn(
            f"{n_missing} grid cells have no data and will render blank."
        )

    x_log = np.log10(durations / T_obs_days)
    intensities_snr = intensities

    return (
        sig_lam_grid, sig_poi_grid, intensities_snr, x_log,
        expected_n, T_obs_years,
    )


# ==================================================================
# Plot helpers
# ==================================================================

def _draw_threshold_step(ax, Z, x_edges, y_edges, level, label=None,
                         label_dx=0.0, label_dy=0.97,
                         **kwargs):
    """Draw the boundary between cells with Z >= level and cells with Z < level.

    Only interior edges separating above/below cells are drawn (the outer
    frame is never traced).  If ``label`` is given, place it at the midpoint
    of the longest straight (collinear, contiguous) run of boundary segments,
    nudged just off the line into the above-threshold region.

    Parameters
    ----------
    label_dx, label_dy : float
        Manual nudge of the label in x and y, as a fraction of the full
        x-axis and y-axis range respectively. Positive is rightward / upward.
        Applied on top of the default placement, so both can move the label
        freely in either direction, e.g. slid sideways when it crowds another
        element.
    """
    above = Z >= level
    ny, nx = above.shape

    # Each segment is (orient, above_side, (p0, p1)) where:
    #   orient     : "h" (horizontal edge) or "v" (vertical edge)
    #   above_side : True if the above-threshold cell lies in the +y direction
    #                (for "h") or in the +x direction (for "v")
    segs = []

    for i in range(ny):
        for j in range(nx - 1):
            if above[i, j] != above[i, j + 1]:
                x = x_edges[j + 1]
                p0 = (x, y_edges[i])
                p1 = (x, y_edges[i + 1])
                segs.append(("v", bool(above[i, j + 1]), (p0, p1)))

    for i in range(ny - 1):
        for j in range(nx):
            if above[i, j] != above[i + 1, j]:
                y = y_edges[i + 1]
                p0 = (x_edges[j], y)
                p1 = (x_edges[j + 1], y)
                segs.append(("h", bool(above[i + 1, j]), (p0, p1)))

    if not segs:
        return

    lc = LineCollection([s[2] for s in segs], **kwargs)
    ax.add_collection(lc)

    if label is None:
        return

    # --- Find the longest straight run of segments. ---
    # Group by (orient, fixed coord): horizontals share their y, verticals
    # share their x. Within each group, sort by varying coord and walk
    # through accumulating contiguous runs (consecutive segments touching
    # end-to-end with the same above_side; a flip in above_side marks a
    # T-junction and breaks the run).
    groups = defaultdict(list)
    for orient, above_side, ((x0, y0), (x1, y1)) in segs:
        if orient == "h":
            groups[("h", y0)].append((min(x0, x1), max(x0, x1), above_side))
        else:
            groups[("v", x0)].append((min(y0, y1), max(y0, y1), above_side))

    # best: (orient, fixed, lo, hi, above_side, count)
    best = None
    for (orient, fixed), items in groups.items():
        items.sort()
        lo, hi, side = items[0]
        count = 1
        for nxt_lo, nxt_hi, nxt_side in items[1:]:
            if nxt_lo == hi and nxt_side == side:
                hi = nxt_hi
                count += 1
            else:
                if best is None or count > best[5]:
                    best = (orient, fixed, lo, hi, side, count)
                lo, hi, side, count = nxt_lo, nxt_hi, nxt_side, 1
        if best is None or count > best[5]:
            best = (orient, fixed, lo, hi, side, count)

    orient, fixed, lo, hi, above_side, _ = best
    mid = 0.5 * (lo + hi)
    sign = 1.0 if above_side else -1.0
    x_range = x_edges[-1] - x_edges[0]
    y_range = y_edges[-1] - y_edges[0]

    # Default placement: sit at the run midpoint, pushed just off the line
    # (perpendicular to the run) into the above-threshold region.
    if orient == "h":
        cx, cy = mid, fixed + sign * y_range
    else:
        cx, cy = fixed + sign * x_range, mid

    # Manual nudge, available in both x and y regardless of orientation.
    cx += -label_dx * x_range
    cy += -label_dy * y_range

    ax.text(cx, cy, label, color="white", 
            fontsize=8.5,
            ha="center", va="center",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=0.2, foreground="grey")])


def _draw_thresholds(ax, Z, x_edges, y_edges, label_dx=0.0, label_dy=0.97):
    """Overlay all ``THRESHOLD_LEVELS`` significance contours on ``ax``."""
    for level in THRESHOLD_LEVELS:
        _draw_threshold_step(
            ax, Z, x_edges, y_edges, level=level,
            colors="white", linewidths=0.5, linestyles="solid",
            label=rf"${level:g}\,\sigma$",
            label_dx=label_dx, label_dy=label_dy,
        )


def _draw_reference_lines(ax, ref_x):
    """Draw dashed vertical guides at the reference flare durations."""
    for label, xr in zip(REF_LABELS, ref_x):
        ax.axvline(xr, color="0.6", linewidth=0.6, linestyle="--")
        ax.text(
            xr, 1.01, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6, color="0.5",
        )


def _style_colorbar(cbar, label):
    """Apply the shared colorbar label and tick/outline styling."""
    cbar.set_label(label)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(direction="out")


def _add_suptitle(fig, expected_n, T_obs_years):
    """Add the shared ``mu`` / ``T_obs`` figure title."""
    fig.suptitle(
        rf"$\mu = {expected_n:.1f}\,$events, "
        rf"$T_{{\rm obs}} = {int(T_obs_years)}\,$years"
    )


# ==================================================================
# Figures
# ==================================================================

def _plot_ratio(
    sig_lam_grid: np.ndarray,
    sig_poi_grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ref_x: list[float],
    expected_n: float,
    T_obs_years: float,
    output_dir: Path,
    mark_unity: bool = True,
    unity_tol: float = 0.1,
) -> None:
    """Save a separate figure with sigma_Lambda / sigma_Poisson.

    Parameters
    ----------
    mark_unity : bool
        If True, overlay markers on cells where |ratio - 1| < ``unity_tol``.
    unity_tol : float
        Tolerance around 1 used to flag "ratio ~ 1" cells.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sig_lam_grid / sig_poi_grid

    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    mesh = ax.pcolormesh(x_edges, y_edges, ratio,
                         rasterized=True, shading="flat")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    if mark_unity:
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        XX, YY = np.meshgrid(x_centers, y_centers)
        near_one = np.isfinite(ratio) & (np.abs(ratio - 1.0) < unity_tol)
        ax.scatter(
            XX[near_one], YY[near_one],
            s=4, marker="o",
            facecolor="white", edgecolor="black", linewidth=0.3,
            zorder=5,
            label=rf"$|\sigma_\Lambda/\sigma_{{\rm Poisson}} - 1| < {unity_tol:g}$",
        )
        ax.legend(loc="best", fontsize=5, framealpha=0.9)
    ax.set_xlabel(r"$\log_{10}(\Delta t_{\rm flare}/10\,{\rm years})$")
    ax.set_ylabel(r"SNR")

    _draw_reference_lines(ax, ref_x)

    cbar = fig.colorbar(mesh, ax=ax)
    _style_colorbar(cbar, r"$\sigma_\Lambda / \sigma_{\rm Poisson}$")

    _add_suptitle(fig, expected_n, T_obs_years)
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.2, top=0.82)

    out_path = output_dir / "grid_p50_ratio.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_lambda_only(
    sig_lam_grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ref_x: list[float],
    expected_n: float,
    T_obs_years: float,
    cmap,
    norm,
    output_dir: Path,
) -> None:
    """Save a standalone single-panel figure with only the Lambda map.

    Identical in content and style to the right panel of the combined
    ``grid_p50.png`` figure, but rendered as a (1, 1) figure.
    """
    fig, ax = plt.subplots(figsize=(3.2, 2.5))

    mesh = ax.pcolormesh(x_edges, y_edges, sig_lam_grid, cmap=cmap, norm=norm,
                         rasterized=True, shading="flat")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    _draw_thresholds(ax, sig_lam_grid, x_edges, y_edges)
    ax.set_title(r"$\Lambda$", pad=8)
    ax.set_xlabel(r"$\log_{10}(\Delta t_{\rm flare}/10\,{\rm years})$")
    ax.set_ylabel(r"SNR")

    _draw_reference_lines(ax, ref_x)

    cbar = fig.colorbar(mesh, ax=ax)
    _style_colorbar(cbar, r"significance of median $(\sigma)$")

    _add_suptitle(fig, expected_n, T_obs_years)
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.2, top=0.82)

    out_path = output_dir / "grid_p50_lambda.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(
    run_dir: str | Path,
    output_dir: str | Path,
    percentile: float = PERCENTILE,
    plot_ratio: bool = True,
    plot_lambda_only: bool = True,
    ratio_mark_unity: bool = True,
    ratio_unity_tol: float = 0.1,
) -> None:
    """Build the 2D significance figure from a single grid_p50 run.

    Parameters
    ----------
    percentile : float
        Percentile of the per-cell p-value distribution to map (default 50).
    plot_ratio : bool
        If True, also save a separate sigma_Lambda / sigma_Poisson figure.
    plot_lambda_only : bool
        If True, also save a standalone single-panel Lambda figure.
    ratio_mark_unity : bool
        If True, overlay markers on cells where |ratio - 1| < ``ratio_unity_tol``.
    ratio_unity_tol : float
        Tolerance around 1 used to flag "ratio ~ 1" cells.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (sig_lam_grid, sig_poi_grid, intensities_snr, x_log,
     expected_n, T_obs_years) = _load(run_dir, percentile=percentile)

    x_edges = _cell_edges(x_log)
    y_edges = _cell_edges(intensities_snr)

    # Reference vertical lines (positions depend on the run's T_obs).
    T_obs_days = T_obs_years * 365.25
    ref_x = [np.log10(d / T_obs_days) for d in REF_DAYS]

    # Color scale: shared between panels (ignore blank/missing cells).
    vmax = float(np.nanmax([np.nanmax(sig_lam_grid), np.nanmax(sig_poi_grid)]))
    vmax = max(vmax, 3.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "magma_trimmed",
        mpl.colormaps["magma"](np.linspace(0.05, 0.95, 256)),
    )
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    # ------------------------------------------------------------------
    # Combined Poisson + Lambda figure with a shared colorbar.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(5, 2.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.08)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0, sharex=ax0)
    ax1.tick_params(labelleft=False)
    cax = fig.add_subplot(gs[2])

    for ax, Z, title in zip(
        [ax0, ax1],
        [sig_poi_grid, sig_lam_grid],
        ["Poisson", r"$\Lambda$"],
    ):
        ax.pcolormesh(x_edges, y_edges, Z, cmap=cmap, norm=norm,
                      rasterized=True, shading="flat")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])

        if ax == ax0:
            _draw_thresholds(ax0, Z, x_edges, y_edges, label_dx= -0.04)
        elif ax == ax1:
            _draw_thresholds(ax1, Z, x_edges, y_edges)

        ax.set_title(title, pad=8)
        ax.set_xlabel(r"$\log_{10}(\Delta t_{\rm flare}/10\,{\rm years})$")

        _draw_reference_lines(ax, ref_x)

    ax0.set_ylabel(r"SNR")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    _style_colorbar(cbar, r"significance of median $(\sigma)$")

    _add_suptitle(fig, expected_n, T_obs_years)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.83)

    out_path = output_dir / "grid_p50.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    if plot_lambda_only:
        _plot_lambda_only(
            sig_lam_grid=sig_lam_grid,
            x_edges=x_edges,
            y_edges=y_edges,
            ref_x=ref_x,
            expected_n=expected_n,
            T_obs_years=T_obs_years,
            cmap=cmap,
            norm=norm,
            output_dir=output_dir,
        )

    if plot_ratio:
        _plot_ratio(
            sig_lam_grid=sig_lam_grid,
            sig_poi_grid=sig_poi_grid,
            x_edges=x_edges,
            y_edges=y_edges,
            ref_x=ref_x,
            expected_n=expected_n,
            T_obs_years=T_obs_years,
            output_dir=output_dir,
            mark_unity=ratio_mark_unity,
            unity_tol=ratio_unity_tol,
        )


if __name__ == "__main__":
    run_dir = Path(
        "/lustre/Auger/brais.rocha/spacetime_correlations/output/scripts/"
        "grid_p50/20260525_153127"
    )
    output_dir = run_dir / "figures"
    main(run_dir=run_dir, output_dir=output_dir)
