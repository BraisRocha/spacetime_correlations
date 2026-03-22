from pathlib import Path

import numpy as np
import scipy.stats as scp
import astropy.units as u
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from spacetimecorr import ExposureModel
from spacetimecorr import Observatory
from spacetimecorr import RNGManager

# ------------------------------------------------------------------
# Output directory
# ------------------------------------------------------------------

def build_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    outdir_acceptance = project_root / "output" / "diagnostics" / "exposure"/ "acceptance"
    outdir_acceptance.mkdir(parents=True, exist_ok=True)
    outdir_sampling = project_root / "output" / "diagnostics" / "exposure" / "sampling"
    outdir_sampling.mkdir(parents=True, exist_ok=True)
    return outdir_acceptance, outdir_sampling

# ------------------------------------------------------------------
# Summary text
# ------------------------------------------------------------------

def exposure_acceptance_summary_text(
    exposure: ExposureModel,
    centre: np.ndarray,
    candidate_times: Time,
    accepted_times: Time,
    accepted_exposure: np.ndarray,
    probs: np.ndarray,
    mask: np.ndarray,
    max_rows: int = 10,
) -> str:
    """Return a human-readable diagnostic summary for one exposure test."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPOSURE ACCEPTANCE DIAGNOSTIC SUMMARY")
    lines.append("=" * 70)

    lines.append(f"t0                       : {exposure.t0.isot}")
    lines.append(f"tf                       : {exposure.tf.isot}")
    lines.append(f"centre [RA, Dec] deg     : {np.asarray(centre, dtype=float)}")
    lines.append(f"candidate events         : {len(candidate_times)}")
    lines.append(f"accepted events          : {len(accepted_times)}")
    lines.append(f"acceptance fraction      : {len(accepted_times) / len(candidate_times):.6f}" if len(candidate_times) > 0 else "acceptance fraction      : nan")

    max_exp = float(exposure.max_directional_exposure(centre))
    lines.append(f"max directional exposure : {max_exp:.6g}")

    lines.append("")
    lines.append("Acceptance diagnostics:")
    if len(probs) > 0:
        lines.append(f"  min p_det              : {np.min(probs):.6g}")
        lines.append(f"  max p_det              : {np.max(probs):.6g}")
        lines.append(f"  mean p_det             : {np.mean(probs):.6g}")
        lines.append(f"  kept by mask           : {np.count_nonzero(mask)} / {len(mask)}")
    else:
        lines.append("  no candidate times provided")

    lines.append("")
    lines.append("Accepted exposure diagnostics:")
    if len(accepted_exposure) > 0:
        lines.append(f"  min epsilon            : {np.min(accepted_exposure):.6g}")
        lines.append(f"  max epsilon            : {np.max(accepted_exposure):.6g}")
        lines.append(f"  mean epsilon           : {np.mean(accepted_exposure):.6g}")
        lines.append(f"  monotonic after sort?  : {bool(np.all(np.diff(np.sort(accepted_exposure)) >= -1e-12))}")
    else:
        lines.append("  no accepted events")

    nshow = min(max_rows, len(candidate_times))
    lines.append("")
    lines.append(f"First {nshow} candidate events:")
    lines.append(" idx | time | p_det | kept")
    lines.append("-" * 70)
    for i in range(nshow):
        lines.append(
            f"{i:4d} | "
            f"{candidate_times[i].isot} | "
            f"{probs[i]:.6g} | "
            f"{bool(mask[i])}"
        )

    nshow_acc = min(max_rows, len(accepted_times))
    lines.append("")
    lines.append(f"First {nshow_acc} accepted events:")
    lines.append(" idx | time | cumulative directional exposure")
    lines.append("-" * 70)
    for i in range(nshow_acc):
        lines.append(
            f"{i:4d} | "
            f"{accepted_times[i].isot} | "
            f"{accepted_exposure[i]:.6g}"
        )

    return "\n".join(lines)

def exposure_sampling_summary_text(
    sample_exposure: np.ndarray,
    exposure_gaps: np.ndarray,
    exp_rate_exposure: float,
    max_dir_exposure: float,
    method_name: str,
    max_rows: int = 10,
) -> str:
    """Return a human-readable diagnostic summary for one exposure-space sampling test."""
    sample_exposure = np.asarray(sample_exposure, dtype=float)
    exposure_gaps = np.asarray(exposure_gaps, dtype=float)

    lines = []
    lines.append("=" * 70)
    lines.append("EXPOSURE SAMPLING DIAGNOSTIC SUMMARY")
    lines.append("=" * 70)

    lines.append(f"method_name              : {method_name}")
    lines.append(f"exp_rate_exposure        : {exp_rate_exposure:.6g}")
    lines.append(f"max_dir_exposure         : {max_dir_exposure:.6g}")
    lines.append(f"sampled exposure count   : {len(sample_exposure)}")
    lines.append(
        f"expected mean gap        : {1.0 / exp_rate_exposure:.6g}"
        if exp_rate_exposure > 0
        else "expected mean gap        : nan"
    )

    lines.append("")
    lines.append("Sampled exposure diagnostics:")
    if len(sample_exposure) > 0:
        lines.append(f"  min sampled exposure   : {np.min(sample_exposure):.6g}")
        lines.append(f"  max sampled exposure   : {np.max(sample_exposure):.6g}")
        lines.append(f"  mean sampled exposure  : {np.mean(sample_exposure):.6g}")
        lines.append(
            f"  within [0, max]?       : "
            f"{bool(np.all((sample_exposure >= 0.0) & (sample_exposure <= max_dir_exposure)))}"
        )
        lines.append(
            f"  monotonic after sort?  : "
            f"{bool(np.all(np.diff(np.sort(sample_exposure)) >= -1e-12))}"
        )
    else:
        lines.append("  no sampled exposures")

    lines.append("")
    lines.append("Gap diagnostics:")
    if len(exposure_gaps) > 0:
        lines.append(f"  min gap                : {np.min(exposure_gaps):.6g}")
        lines.append(f"  max gap                : {np.max(exposure_gaps):.6g}")
        lines.append(f"  mean gap               : {np.mean(exposure_gaps):.6g}")
        lines.append(f"  median gap             : {np.median(exposure_gaps):.6g}")
        lines.append(
            f"  non-negative gaps?     : {bool(np.all(exposure_gaps >= -1e-12))}"
        )
    else:
        lines.append("  no gaps available")

    nshow = min(max_rows, len(sample_exposure))
    lines.append("")
    lines.append(f"First {nshow} sampled exposures:")
    lines.append(" idx | sampled exposure")
    lines.append("-" * 70)
    for i in range(nshow):
        lines.append(f"{i:4d} | {sample_exposure[i]:.6g}")

    nshow_gap = min(max_rows, len(exposure_gaps))
    lines.append("")
    lines.append(f"First {nshow_gap} exposure gaps:")
    lines.append(" idx | gap")
    lines.append("-" * 70)
    for i in range(nshow_gap):
        lines.append(f"{i:4d} | {exposure_gaps[i]:.6g}")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Save arrays
# ------------------------------------------------------------------

def save_exposure_acceptance_arrays(
    outdir: Path,
    stem: str,
    centre: np.ndarray,
    candidate_times: Time,
    accepted_times: Time,
    probs: np.ndarray,
    mask: np.ndarray,
    accepted_exposure: np.ndarray,
    grid_times: Time,
    grid_acceptance: np.ndarray,
    grid_exposure: np.ndarray,
) -> Path:
    """Save exposure acceptance diagnostic arrays."""
    path = outdir / f"{stem}_acceptance_arrays.npz"

    np.savez_compressed(
        path,
        centre=np.asarray(centre, dtype=float),
        candidate_time_isot=np.array(candidate_times.isot),
        candidate_time_jd=candidate_times.jd,
        accepted_time_isot=np.array(accepted_times.isot),
        accepted_time_jd=accepted_times.jd,
        p_det=probs,
        mask=mask,
        accepted_exposure=accepted_exposure,
        grid_time_isot=np.array(grid_times.isot),
        grid_time_jd=grid_times.jd,
        grid_acceptance=grid_acceptance,
        grid_exposure=grid_exposure,
    )
    return path

def save_exposure_sampling_arrays(
    outdir: Path,
    stem: str,
    sample_exposure: np.ndarray,
    exposure_gaps: np.ndarray,
    exp_rate_exposure: float,
    max_dir_exposure: float,
    method_name: str,
) -> Path:
    """Save exposure-space sampling diagnostic arrays."""
    path = outdir / f"{stem}_sampling_arrays.npz"

    np.savez_compressed(
        path,
        sample_exposure=sample_exposure,
        exposure_gaps=exposure_gaps,
        exp_rate_exposure=exp_rate_exposure,
        max_dir_exposure=max_dir_exposure,
        method_name=method_name,
    )
    return path


# ------------------------------------------------------------------
# Save plots
# ------------------------------------------------------------------

def save_exposure_acceptance_plots(
    outdir: Path,
    centre: np.ndarray,
    candidate_times: Time,
    accepted_times: Time,
    accepted_exposure: np.ndarray,
    grid_times: Time,
    grid_acceptance: np.ndarray,
    grid_exposure: np.ndarray,
) -> list[Path]:
    """Save quick-look diagnostic plots for the exposure acceptance model."""
    saved = []

    grid_offsets = (grid_times - grid_times[0]).to_value(u.h)

    # Instantaneous acceptance curve
    plt.figure(figsize=(7, 4))
    plt.plot(grid_offsets, grid_acceptance)
    plt.xlabel("Time offset from t0 [h]")
    plt.ylabel("Instantaneous acceptance")
    plt.title(f"[RA={centre[0]}, Dec={centre[1]}] Instantaneous acceptance vs Time")
    plt.tight_layout()
    p = outdir / f"acceptance_vs_time.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    # Cumulative directional exposure curve
    plt.figure(figsize=(7, 4))
    plt.plot(grid_offsets, grid_exposure)
    plt.xlabel("Time offset from t0 [h]")
    plt.ylabel("Cumulative directional exposure")
    plt.title("Cumulative directional exposure vs time")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    plt.tight_layout()
    p = outdir / f"exposure_vs_time.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    # Accepted events directional exposure vs time
    plt.plot(grid_offsets, grid_exposure, lw=2, label="cumulative exposure")
    if len(accepted_times) > 0:
        acc_offsets = (accepted_times - t0).to_value(u.h)
        acc_exposure = np.interp(acc_offsets, grid_offsets, grid_exposure)
        x = np.asarray(acc_offsets)
        y = np.asarray(acc_exposure)
        if len(x) > 1 and np.ptp(x) > 0:
            z = gaussian_kde(x)(x)
        else:
            z = np.ones_like(x)
        idx = np.argsort(z)
        x, y, z = x[idx], y[idx], z[idx]
        sc = plt.scatter(x, y, c=z, s=24, alpha=0.9, zorder=3, label="accepted events")
        plt.colorbar(sc, label="Accepted events density")
    plt.xlabel("Time offset from t0 [h]")
    plt.ylabel("Cumulative directional exposure")
    plt.title("Cumulative directional exposure vs time")
    plt.legend()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    plt.tight_layout()
    p = outdir / f"accepted_exposure_vs_time.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    # Candidate and accepted times
    cand_offsets = (candidate_times - candidate_times.min()).to_value(u.h)
    plt.figure(figsize=(7, 4))
    bins = np.histogram_bin_edges(cand_offsets, bins=min(30, max(5, len(cand_offsets) // 3)))
    plt.hist(cand_offsets, bins=bins, alpha=0.6, label="candidate", edgecolor="black", linewidth=0.8)
    if len(accepted_times) > 0:
        acc_offsets = (accepted_times - candidate_times.min()).to_value(u.h)
        plt.hist(acc_offsets, bins=bins, alpha=0.8, label="accepted", edgecolor="black", linewidth=0.8)
    plt.xlabel("Time offset from first candidate [h]")
    plt.ylabel("Counts")
    plt.title("Candidate vs accepted event times")
    plt.legend()
    plt.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    plt.tight_layout()
    p = outdir / f"candidate_vs_accepted_time_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    # Accepted exposure histogram
    if len(accepted_exposure) > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(accepted_exposure, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
        plt.xlabel("Cumulative directional exposure")
        plt.ylabel("Counts")
        plt.title("Accepted events directional exposure")
        plt.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        plt.tight_layout()
        p = outdir / f"dir_exposure_hist.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(p)

    return saved

def save_exposure_sampling_plots(
    outdir: Path,
    sample_exposure: np.ndarray,
    exp_rate_exposure: float,
) -> list[Path]:
    """Save quick-look diagnostic plots for the exposure sampling model."""
    saved = []

    if len(sample_exposure) == 0:
        return saved
    
    # Histogram of sampled exposure values
    plt.figure(figsize=(7, 4))
    plt.hist(sample_exposure, bins="fd", alpha=0.8, edgecolor="black", linewidth=0.8)
    plt.xlabel("Sampled directional exposure")
    plt.ylabel("Counts")
    plt.title("Sample from sample_directional_exposure")
    plt.tight_layout()
    p = outdir / f"dir_exposure_hist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    # Histogram of directional exposure gaps
    exposure_gaps = np.diff(np.sort(sample_exposure))

    plt.figure(figsize=(7, 4))
    plt.hist(exposure_gaps, bins=min(30, max(5, len(exposure_gaps) // 3)), density=True, alpha=0.8, label=r"$\Delta\varepsilon$", edgecolor = "black", linewidth=0.8)

    xmax = max(
        np.max(exposure_gaps),
        np.percentile(exposure_gaps, 99.5),
        5.0 / exp_rate_exposure,
    )
    x = np.linspace(0.0, xmax, 400)
    pdf = scp.expon(scale=1.0 / exp_rate_exposure).pdf(x)
    plt.plot(x, pdf,label=f"Exponential (rate={exp_rate_exposure:.4g})")
    plt.xlabel("Consecutive directional exposure differences")
    plt.ylabel(r"$log_{10}$(Density)")
    plt.yscale("log")
    plt.title("Exposure gaps vs exponential law")
    plt.legend()
    plt.tight_layout()
    p = outdir / f"exposure_gaps_dist.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(p)

    return saved

# ------------------------------------------------------------------
# Main diagnostic runner
# ------------------------------------------------------------------

def run_exposure_diagnostic(
    exposure: ExposureModel,
    centre: np.ndarray,
    n_candidates: int = 200,
    grid_size: int = 2000,
    efficiency=None,
    max_rows: int = 12,
    stem: str = "exposure",
) -> None:
    """
    Generate one exposure-model diagnostic, print summary, and save outputs.
    """

    # ------------------------------------------------------------------
    # Exposure Acceptance Diagnostic
    # ------------------------------------------------------------------

    outdir_acceptance, outdir_sampling = build_output_dir()

    # Candidate times sampled uniformly in [t0, tf]
    total_sec = (exposure.tf - exposure.t0).to_value(u.s)
    offsets = np.sort(exposure.rng.uniform(0.0, total_sec, size=n_candidates))
    candidate_times = exposure.t0 + TimeDelta(offsets, format="sec")

    accepted_times, mask, probs, accepted_exposure = exposure.detect_times(
        candidate_times,
        centre=centre,
        efficiency=efficiency,
        return_mask=True,
        return_prob=True,
        return_exposure=True,
    )

    # Dense grid for smooth diagnostic curves
    grid_offsets = np.linspace(0.0, total_sec, grid_size)
    grid_times = exposure.t0 + TimeDelta(grid_offsets, format="sec")
    grid_acceptance = np.asarray(exposure.instantaneous_acceptance(grid_times, centre), dtype=float)
    grid_exposure = np.asarray(exposure.cumulative_directional_exposure(grid_times, centre), dtype=float)

    summary = exposure_acceptance_summary_text(
        exposure=exposure,
        centre=centre,
        candidate_times=candidate_times,
        accepted_times=accepted_times,
        accepted_exposure=accepted_exposure,
        probs=probs,
        mask=mask,
        max_rows=max_rows,
    )

    summary_path = outdir_acceptance / f"{stem}_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    arrays_path = save_exposure_acceptance_arrays(
        outdir=outdir_acceptance,
        stem=stem,
        centre=centre,
        candidate_times=candidate_times,
        accepted_times=accepted_times,
        probs=probs,
        mask=mask,
        accepted_exposure=accepted_exposure,
        grid_times=grid_times,
        grid_acceptance=grid_acceptance,
        grid_exposure=grid_exposure,
    )

    plot_paths = save_exposure_acceptance_plots(
        outdir=outdir_acceptance,
        centre=centre,
        candidate_times=candidate_times,
        accepted_times=accepted_times,
        accepted_exposure=accepted_exposure,
        grid_times=grid_times,
        grid_acceptance=grid_acceptance,
        grid_exposure=grid_exposure,
    )

    print("\nSaved acceptance diagnostic files:")
    print(f"  Summary : {summary_path}")
    print(f"  Arrays  : {arrays_path}")
    for p in plot_paths:
        print(f"  Plot    : {p}")

    # ------------------------------------------------------------------
    # Exposure Sampling Diagnostic
    # ------------------------------------------------------------------

    max_dir_exposure = float(exposure.max_directional_exposure(centre))
    exp_rate_exposure = float(n_candidates) / max_dir_exposure

    sample_exposure, method_name = exposure.sample_directional_exposure(
        n_events=n_candidates,
        exp_rate_exposure=exp_rate_exposure,
        max_dir_exposure=max_dir_exposure,
    )

    exposure_gaps = np.diff(np.sort(sample_exposure))

    summary = exposure_sampling_summary_text(
        sample_exposure=sample_exposure,
        exposure_gaps=exposure_gaps,
        exp_rate_exposure=exp_rate_exposure,
        max_dir_exposure=max_dir_exposure,
        method_name=method_name,
    )

    summary_path = outdir_sampling / f"{stem}_sampling_summary.txt"
    summary_path.write_text(summary)

    arrays_path = save_exposure_sampling_arrays(
        outdir=outdir_sampling,
        stem=stem,
        sample_exposure=sample_exposure,
        exposure_gaps=exposure_gaps,
        exp_rate_exposure=exp_rate_exposure,
        max_dir_exposure=max_dir_exposure,
        method_name=method_name,
    )

    plot_paths = save_exposure_sampling_plots(
        outdir=outdir_sampling,
        sample_exposure=sample_exposure,
        exp_rate_exposure=exp_rate_exposure,
    )

    print("\nSaved sampling diagnostic files:")
    print(f"  Summary : {summary_path}")
    print(f"  Arrays  : {arrays_path}")
    for p in plot_paths:
        print(f"  Plot    : {p}")

# ------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------

if __name__ == "__main__":

    rng_manager = RNGManager(seed=42)
    rng_exposure = rng_manager.get("exposure")

    # Observation interval
    t0 = Time("2025-01-01T00:00:00")
    tf = Time("2025-01-07T00:00:00")

    centre = np.array([30.0, 0.0])

    # Pierre Auger Observatory coordinates
    latitude_pa = -35.15
    longitude_pa = -69.15
    altitude_pa = 1425

    observatory = Observatory(
        latitude=latitude_pa,
        longitude=longitude_pa,
        altitude=altitude_pa,
    )
    exposure_model = ExposureModel(
        observatory=observatory,
        t0=t0,
        tf=tf,
        rng=rng_exposure,
    )

    run_exposure_diagnostic(
        exposure=exposure_model,
        centre=centre,
        n_candidates=100000,
        grid_size=4000,
        efficiency=None,
        max_rows=12,
        stem="exposure",
    )