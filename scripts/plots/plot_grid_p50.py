"""
Paper-quality 2D sensitivity map on the (flare duration, flare intensity)
grid from run_grid_p50.py.

Color encodes the Gaussian-equivalent significance (in sigma) of the
median (50th percentile) test statistic stored per grid cell.

Left panel  - Poisson counting test: significance of n_sample_window_p50
              under N ~ Poisson(expected_n).
Right panel - Lambda test: significance of lambda_flare_p50 under the
              marginal Lambda distribution with background expected_n.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scp

from spacetimecorr.statistics import (
    lambda_marginal_sigma,
    pvalue_to_sigma,
)

# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
RC_FILE = Path(__file__).resolve().parent / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)


# ------------------------------------------------------------------
# Significance computations
# ------------------------------------------------------------------

def _sigma_poisson(n_p50: np.ndarray, expected_n: float) -> np.ndarray:
    """Significance (sigma) of observing >= n_p50 under Poisson(expected_n)."""
    p = scp.poisson.sf(np.asarray(n_p50) - 1, expected_n)
    return pvalue_to_sigma(p)


def _sigma_lambda(lambda_p50: np.ndarray, expected_n: float) -> np.ndarray:
    """Significance (sigma) of lambda_p50 under marginal Lambda(expected_n)."""
    flat = np.asarray(lambda_p50, dtype=float).ravel()
    out = np.array([lambda_marginal_sigma(float(x), expected_n) for x in flat])
    return out.reshape(np.asarray(lambda_p50).shape)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def _load(run_dir: Path) -> tuple:
    """Return grids and axes needed for plotting.

    Returns
    -------
    lam_grid       : (n_intensities, n_durations) median Lambda
    n_grid         : (n_intensities, n_durations) median n_sample in window
    intensities_pct: (n_intensities,) flare intensity in percent
    x_log          : (n_durations,) log10(duration_days / T_obs_days)
    expected_n     : float, expected background events in window
    T_obs_years    : float
    """
    data_dir = run_dir / "data" if (run_dir / "data").exists() else run_dir
    data = np.load(data_dir / "results_merged.npz")

    with (data_dir / "metadata_job0.json").open() as fh:
        meta = json.load(fh)
    expected_n = float(meta["expected_n"])
    T_obs_days = float(meta["time"]["T_obs_days"])
    T_obs_years = T_obs_days / 365.25

    durations = np.array(sorted(set(data["flare_duration_days"])))
    intensities = np.array(sorted(set(data["flare_intensity"])))

    nd, nf = len(durations), len(intensities)
    lam_grid = np.empty((nf, nd))
    n_grid = np.empty((nf, nd))

    for idx in range(len(data["job_id"])):
        d = data["flare_duration_days"][idx]
        f = data["flare_intensity"][idx]
        j = np.searchsorted(durations, d)
        i = np.searchsorted(intensities, f)
        lam_grid[i, j] = data["lambda_flare_p50"][idx]
        n_grid[i, j] = data["n_sample_window_p50"][idx]

    x_log = np.log10(durations / T_obs_days)
    intensities_pct = intensities * 100.0

    return lam_grid, n_grid, intensities_pct, x_log, expected_n, T_obs_years


# ------------------------------------------------------------------
# Threshold boundary (cell-aligned step line)
# ------------------------------------------------------------------

def _draw_threshold_step(ax, Z, x_edges, y_edges, level, label=None,
                         label_fontsize=7, **kwargs):
    """Draw the boundary between cells with Z >= level and cells with Z < level.

    Only interior edges separating above/below cells are drawn (the outer
    frame is never traced).  If ``label`` is given, place it near the
    middle of the longest contiguous run of segments.
    """
    above = Z >= level
    ny, nx = above.shape
    segments = []

    for i in range(ny):
        for j in range(nx - 1):
            if above[i, j] != above[i, j + 1]:
                x = x_edges[j + 1]
                segments.append([(x, y_edges[i]), (x, y_edges[i + 1])])

    for i in range(ny - 1):
        for j in range(nx):
            if above[i, j] != above[i + 1, j]:
                y = y_edges[i + 1]
                segments.append([(x_edges[j], y), (x_edges[j + 1], y)])

    if not segments:
        return

    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, **kwargs)
    ax.add_collection(lc)

    if label is not None:
        (x0, y0), (x1, y1) = segments[len(segments) // 2]
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        # Nudge the label up by a fraction of the y-axis range
        ymin, ymax = y_edges[0], y_edges[-1]
        cy += 0.05 * (ymax - ymin)
        ax.text(cx, cy, label, color="white",
                fontsize=label_fontsize, ha="center", va="center",
                zorder=5)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(run_dir: str | Path, output_dir: str | Path) -> None:
    """Build the 2D significance figure from a single grid_p50 run."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lam_grid, n_grid, intensities_pct, x_log, expected_n, T_obs_years = (
        _load(run_dir)
    )

    sig_lam_grid = _sigma_lambda(lam_grid, expected_n)
    sig_poi_grid = _sigma_poisson(n_grid, expected_n)

    # Cell edges for pcolormesh: midpoints between centres, with
    # outer edges extended by half a cell.
    def _edges(centres: np.ndarray) -> np.ndarray:
        c = np.asarray(centres, dtype=float)
        mids = 0.5 * (c[:-1] + c[1:])
        return np.concatenate(([c[0] - (mids[0] - c[0])],
                               mids,
                               [c[-1] + (c[-1] - mids[-1])]))

    x_edges = _edges(x_log)
    y_edges = _edges(intensities_pct)

    # Color scale: shared between panels
    vmax = float(np.nanmax([sig_lam_grid.max(), sig_poi_grid.max()]))
    vmax = max(vmax, 3.0)

    # ------------------------------------------------------------------
    # Reference vertical lines
    # ------------------------------------------------------------------
    T_obs_days = T_obs_years * 365.25
    ref_labels = ["1 day", "1 week", "1 mth", "1 year"]
    ref_days   = [1.0,     7.0,      30.0,     365.0  ]
    ref_x      = [np.log10(d / T_obs_days) for d in ref_days]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "magma_trimmed",
        mpl.colormaps["magma"](np.linspace(0.05, 0.95, 256)),
    )
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(5.5, 2.5))
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

        _draw_threshold_step(ax, Z, x_edges, y_edges, level=3.0,
                             colors="white", linewidths=1.0,
                             linestyles="dashed",
                             label=r"$3\sigma$")
        _draw_threshold_step(ax, Z, x_edges, y_edges, level=5.0,
                             colors="white", linewidths=1.0,
                             linestyles="solid",
                             label=r"$5\sigma$")
        _draw_threshold_step(ax, Z, x_edges, y_edges, level=1.0,
                             colors="white", linewidths=1.0,
                             linestyles="solid",
                             label=r"$1\sigma$")
        _draw_threshold_step(ax, Z, x_edges, y_edges, level=2.0,
                             colors="white", linewidths=1.0,
                             linestyles="solid",
                             label=r"$2\sigma$")
        ax.set_title(title)
        ax.set_xlabel(r"$\log_{10}(\Delta t_{\rm flare}/10\,{\rm years})$")

        for label, xr in zip(ref_labels, ref_x):
            ax.axvline(xr, color="0.6", linewidth=0.6, linestyle="--")
            ax.text(
                xr, 1.01, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5, color="0.5",
            )

    ax0.set_ylabel(r"$f\,(\%)$")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"significance of median $(\sigma)$")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(direction="out")

    fig.suptitle(
        rf"$\mu = {expected_n:.0f}$ events, "
        rf"$T_{{\rm obs}} = {int(T_obs_years)}\,$years"
    )

    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.82)

    out_path = output_dir / "grid_p50.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_dir = Path(
        "/lustre/Auger/brais.rocha/spacetime_correlations/output/scripts/"
        "grid_p50/20260508_153734"
    )
    output_dir = run_dir / "figures"
    main(run_dir=run_dir, output_dir=output_dir)
