"""
Paper-quality plots for the T_obs scan.

Combines two ``run_scan_intensity.py`` runs (one per observation
duration, same flare duration) into a single two-panel figure of
Lambda distributions vs flare intensity. Each run contributes one
panel; the isotropy null is pooled across both runs to maximise
statistics.

The two panels share flare duration and ``expected_n``; what differs
is ``T_obs``.  The trials penalty (a shorter ``T_obs`` divides any
realistic observation campaign into N independent windows, so any
p-value reported for one window must be multiplied by N) is folded
into the 3-sigma / 5-sigma threshold lines:

    sigma_corrected(panel) = norm.isf(norm.sf(sigma) / trials_factor),
    trials_factor          = T_obs_longest / T_obs_panel.

The longest-T_obs panel has ``trials_factor = 1`` (no penalty); the
shorter-T_obs panel(s) get higher thresholds.
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import scipy.stats as scp

from spacetimecorr.statistics import lambda_marginal_isigma


# ----------------------------------------------------------------------
# Global style: load the matplotlibrc next to this script
# ----------------------------------------------------------------------
RC_FILE = Path(__file__).resolve().parent / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)


# ----------------------------------------------------------------------
# Histogram helpers
# ----------------------------------------------------------------------
def _hist_as_line(values: np.ndarray, bins: np.ndarray):
    """
    Bin ``values`` and return ``(centres, density, density_error)``.

    The returned arrays are suitable for ``ax.errorbar`` and produce the
    line-through-bin-centres look used in the reference figure. Errors
    are Poisson on the bin counts, propagated to the density.
    """
    counts, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    n = counts.sum()

    if n == 0:
        density = np.zeros_like(counts, dtype=float)
        density_err = np.zeros_like(density)
    else:
        density = counts / (n * widths)
        density_err = np.sqrt(counts) / (n * widths)

    return centres, density, density_err


def _draw_lambda_panel(
    ax,
    lambda_iso: np.ndarray,
    lambda_flare_per_intensity: np.ndarray,
    intensities: np.ndarray,
    bins: np.ndarray,
    cmap,
    norm,
    panel_title: str,
) -> None:
    """
    Draw one panel: isotropy null + flare distributions vs flare intensity.

    ``lambda_flare_per_intensity`` is expected to have shape
    ``(n_intensities, n_sim)``.
    """
    # Isotropy (gray dashed, no markers)
    c, d, _ = _hist_as_line(lambda_iso, bins)
    nz = d > 0
    ax.plot(
        c[nz], d[nz],
        linestyle="--", linewidth=0.8, color="0.55",
    )

    # One coloured solid-with-markers line per flare intensity
    for f, lam_f in zip(intensities, lambda_flare_per_intensity):
        c, d, e = _hist_as_line(lam_f, bins)
        nz = d > 0
        ax.errorbar(
            c[nz], d[nz], yerr=e[nz],
            linestyle="-", linewidth=0.5,
            marker="o", markersize=1.8,
            color=cmap(norm(f)),
            elinewidth=0.3, capsize=0,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\Lambda$")
    ax.set_title(panel_title)

    # Two-entry legend: dashed iso + representative solid+marker
    handles = [
        Line2D([0], [0], linestyle="--", linewidth=0.8, color="0.55",
               label="Isotropy"),
        Line2D([0], [0], linestyle="-", linewidth=0.5, color="black",
               marker="o", markersize=1.8, label="Iso. BG + flare"),
    ]
    ax.legend(handles=handles, loc="upper right")


# ----------------------------------------------------------------------
# Run loading
# ----------------------------------------------------------------------
def _load_run(run_dir: Path) -> dict:
    """
    Load ``results.npz`` and ``metadata.json`` from a single run directory.

    Returns a dict with keys: ``lambda_bkg``, ``lambda_flare``,
    ``intensities``, ``duration_days``, ``expected_n``, ``T_obs_days``.
    """
    run_dir = Path(run_dir)
    results = np.load(run_dir / "results.npz")
    with (run_dir / "metadata.json").open() as fh:
        meta = json.load(fh)

    return {
        "lambda_bkg": results["lambda_bkg"],
        "lambda_flare": results["lambda_flare"],
        "intensities": np.asarray(meta["flare"]["intensity"], dtype=float),
        "duration_days": float(meta["flare"]["duration_days"]),
        "expected_n": float(meta["expected_n"]),
        "T_obs_days": float(meta["time"]["T_obs_days"]),
    }


def _format_T_obs(days: float) -> str:
    """Return a LaTeX-friendly label for an observation interval in days."""
    years = days / 365.25
    if np.isclose(years, 1.0, atol=0.05):
        return r"$T_{\rm obs} = 1\,$year"
    if np.isclose(years, round(years), atol=0.05) and years >= 1.0:
        return rf"$T_{{\rm obs}} = {int(round(years))}\,$years"
    if days >= 30.0:
        months = days / 30.0
        return rf"$T_{{\rm obs}} \approx {months:g}\,$months"
    return rf"$T_{{\rm obs}} = {days:g}\,$days"


def _format_duration(days: float) -> str:
    """Return a LaTeX-friendly label for a flare duration in days."""
    if np.isclose(days, 1.0):
        return r"\Delta t_{\rm flare} = 1\,\mathrm{day}"
    if np.isclose(days, 30.0):
        return r"\Delta t_{\rm flare} = 1\,\mathrm{month}"
    if days < 1.0:
        return rf"\Delta t_{{\rm flare}} = {days:g}\,\mathrm{{day}}"
    return rf"\Delta t_{{\rm flare}} = {days:g}\,\mathrm{{days}}"


# ----------------------------------------------------------------------
# Trials-corrected sigma threshold
# ----------------------------------------------------------------------
def _trials_corrected_lambda_threshold(
    sigma_target: float,
    expected_n: float,
    trials_factor: float,
) -> float:
    """
    Lambda value that, after trials-correcting its marginal p-value by
    ``trials_factor``, gives a Gaussian-equivalent significance of
    ``sigma_target``.

    Concretely::

        p_corr = trials_factor * p_uncorr  ->  p_uncorr = p_target / trials_factor
        sigma_uncorr = norm.isf(p_target / trials_factor)
        Lambda = lambda_marginal_isigma(sigma_uncorr, expected_n)
    """
    p_target = scp.norm.sf(sigma_target)
    p_uncorr = p_target / trials_factor
    sigma_uncorr = scp.norm.isf(p_uncorr)
    return lambda_marginal_isigma(sigma_uncorr, expected_n)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(run_dirs: list[str | Path], output_dir: str | Path) -> None:
    """
    Build the T_obs-scan figure from two MC runs.

    ``run_dirs`` must contain exactly two paths, each pointing to a
    ``run_scan_intensity.py`` output directory.  The two runs must
    share the flare duration and ``expected_n`` (i.e. the same window
    geometry) but differ in ``T_obs``.  The two panels of the
    resulting figure correspond to the two runs, in the order given.

    The 3-sigma / 5-sigma threshold lines drawn in each panel include
    a trials-penalty correction: the longest-T_obs panel is treated as
    the reference (no penalty); shorter-T_obs panels get
    ``trials_factor = T_obs_longest / T_obs_panel``.
    """
    if len(run_dirs) != 2:
        raise ValueError(
            f"Expected 2 run directories, got {len(run_dirs)}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load runs and check consistency
    # ------------------------------------------------------------------
    runs = [_load_run(Path(d)) for d in run_dirs]

    intensities = runs[0]["intensities"]
    for r in runs[1:]:
        if not np.allclose(r["intensities"], intensities):
            raise ValueError(
                "Runs have different flare-intensity grids: "
                f"{intensities} vs {r['intensities']}."
            )

    expected_n = runs[0]["expected_n"]
    duration_days = runs[0]["duration_days"]
    for r in runs[1:]:
        if not np.isclose(r["expected_n"], expected_n):
            raise ValueError(
                f"Runs have different expected_n: {expected_n} vs {r['expected_n']}."
            )
        if not np.isclose(r["duration_days"], duration_days):
            raise ValueError(
                "Runs have different flare durations: "
                f"{duration_days} vs {r['duration_days']} days."
            )

    # Pool the isotropy null across both runs (independent draws of the
    # same distribution -> more statistics for the dashed line).
    lambda_bkg = np.concatenate([r["lambda_bkg"] for r in runs])

    # Trials factors: longest T_obs is the reference (factor 1.0); each
    # panel's factor scales the others up.
    T_obs_max = max(r["T_obs_days"] for r in runs)
    trials_factors = [T_obs_max / r["T_obs_days"] for r in runs]

    # ------------------------------------------------------------------
    # Lambda distributions vs flare intensity (panels = T_obs values)
    # ------------------------------------------------------------------
    bins = np.linspace(0, 350, 70)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "plasma_trimmed",
        mpl.colormaps["magma"](np.linspace(0.1, 0.95, 256))
    )
    norm = mpl.colors.Normalize(vmin=intensities.min(), vmax=intensities.max())

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(5.5, 2.5))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.05],  # last column = colorbar
        wspace=0.08
    )

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0, sharex=ax0)
    ax1.tick_params(labelleft=False)  # Remove labels from the right subplot
    cax = fig.add_subplot(gs[2])  # dedicated colorbar axis

    axes = [ax0, ax1]

    for ax, run in zip(axes, runs):
        _draw_lambda_panel(
            ax,
            lambda_iso=lambda_bkg,
            lambda_flare_per_intensity=run["lambda_flare"],
            intensities=intensities,
            bins=bins,
            cmap=cmap,
            norm=norm,
            panel_title=_format_T_obs(run["T_obs_days"]),
        )
        ax.set_ylim(1e-5, 1e0)

    axes[0].set_ylabel("Prob. density")

    # ------------------------------------------------------------------
    # 3-sigma / 5-sigma threshold lines, trials-corrected per panel
    # ------------------------------------------------------------------
    # Uncorrected reference (trials_factor = 1): same for every panel, depends
    # only on the marginal Lambda PDF at this expected_n.
    lam_3_uncorr = _trials_corrected_lambda_threshold(3.0, expected_n, 1.0)
    lam_5_uncorr = _trials_corrected_lambda_threshold(5.0, expected_n, 1.0)
    print(
        f"Lambda thresholds (mu = expected_n = {expected_n:.2f}):\n"
        f"  uncorrected (single-trial reference): "
        f"Lambda(3 sigma) = {lam_3_uncorr:8.3f}, "
        f"Lambda(5 sigma) = {lam_5_uncorr:8.3f}"
    )
    for ax, factor, run in zip(axes, trials_factors, runs):
        lam_3 = _trials_corrected_lambda_threshold(3.0, expected_n, factor)
        lam_5 = _trials_corrected_lambda_threshold(5.0, expected_n, factor)
        T_obs_years = run["T_obs_days"] / 365.25
        print(
            f"  T_obs = {T_obs_years:6.2f} yr "
            f"(trials_factor = {factor:5.2f}): "
            f"Lambda(3 sigma) = {lam_3:8.3f} "
            f"(+{lam_3 - lam_3_uncorr:5.3f}), "
            f"Lambda(5 sigma) = {lam_5:8.3f} "
            f"(+{lam_5 - lam_5_uncorr:5.3f})"
        )
        ax.axvline(lam_3, color="0.3", linewidth=0.8, linestyle=":")
        ax.axvline(lam_5, color="0.3", linewidth=0.8, linestyle="-.")
        trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(lam_3, 0.97, r"$3\sigma$", transform=trans,
                va="top", ha="right", fontsize=6, color="0.3")
        ax.text(lam_5, 0.97, r"$5\sigma$", transform=trans,
                va="top", ha="right", fontsize=6, color="0.3")

    # ------------------------------------------------------------------
    # Discrete colorbar over the intensity grid (displayed in percent).
    # ------------------------------------------------------------------
    intensities_pct = intensities * 100.0
    step_pct = (
        float(np.median(np.diff(intensities_pct)))
        if len(intensities_pct) > 1 else 10.0
    )
    bounds_pct = np.concatenate([
        intensities_pct - step_pct / 2.0,
        [intensities_pct[-1] + step_pct / 2.0],
    ])
    from matplotlib.colors import BoundaryNorm

    cbar_norm = BoundaryNorm(bounds_pct, ncolors=cmap.N, clip=True)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=cbar_norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        cax=cax,
        boundaries=bounds_pct,
        ticks=intensities_pct,
        spacing="proportional",
    )
    cbar.ax.minorticks_off()
    cbar.set_label(r"$f\,(\%)$")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(direction="out")

    fig.suptitle(
        rf"$\mu = {expected_n:.1f}$ events, "
        rf"${_format_duration(duration_days)}$"
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.85,
        wspace=0.05
    )
    fig.savefig(output_dir / "lambda_vs_T_obs.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Edit these paths to point at the two runs you want to combine
    base = Path(
        "/lustre/Auger/brais.rocha/spacetime_correlations/output/scripts/scan_intensity"
    )
    run_dirs = [
        base / "20260519_095850_seed42",  # T_obs = 10 years (reference)
        base / "20260519_100251_seed42",  # T_obs = 1 year
    ]
    output_dir = base / "figures"
    main(run_dirs=run_dirs, output_dir=output_dir)
