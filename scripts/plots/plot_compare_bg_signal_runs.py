"""
Compare bg-vs-signal Monte Carlo runs against each other.

Loads two (or more) run directories produced by run_compare_bg_signal.py
and overlays their Lambda, n_sample and p-value distributions on the
same axes. Background distributions across runs are expected to coincide
(isotropy is the same); the flare distributions are what should differ.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spacetimecorr as stc


def _load_run(run_dir: Path) -> dict:
    """Read results.npz + metadata.json from a run directory."""
    data = np.load(run_dir / "results.npz")
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return {
        "lambda_bkg": data["lambda_bkg"],
        "lambda_flare": data["lambda_flare"],
        "pvalues_bkg": data["pvalues_bkg"],
        "pvalues_flare": data["pvalues_flare"],
        "n_sample_bkg": data["n_sample_bkg"],
        "n_sample_flare": data["n_sample_flare"],
        "expected_n": meta["expected_n"],
        "mu_flare": meta["mu_flare"],
        "T_obs_days": meta["T_obs_days"],
        "flare_duration_days": meta["flare_duration_days"],
        "n_sim": meta["n_simulations_requested"],
    }


def main(run_dirs: list[str | Path],
         output_dir: str | Path,
         labels: list[str] | None = None) -> None:
    """
    Compare the bkg/flare distributions across the given run directories.

    Parameters
    ----------
    run_dirs : list of paths
        Run directories, each containing results.npz and metadata.json.
    output_dir : path
        Where to write the comparison plots.
    labels : list of str, optional
        Display labels for each run. Defaults to the run directory name.
    """
    run_dirs = [Path(p) for p in run_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = [p.name for p in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError("labels must have the same length as run_dirs")

    runs = [_load_run(p) for p in run_dirs]

    # Sanity: expected_n must match for the Poisson p-value comparison
    # to be meaningful (same null hypothesis across runs).
    mus = [r["expected_n"] for r in runs]
    if not np.allclose(mus, mus[0]):
        raise ValueError(
            f"expected_n differs across runs ({mus}); p-value comparison "
            "would mix different null hypotheses."
        )
    expected_n = mus[0]

    # ------------------------------------------------------------------
    # Lambda comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    # Shared bkg curve: bkg distributions should coincide across runs.
    # Plot each run's bkg in a single neutral style so overlap is visible.
    for r, lbl in zip(runs, labels):
        ax.hist(r["lambda_bkg"], bins="sqrt", density=True,
                histtype="step", linewidth=1.2, color="0.5",
                label=f"bkg ({lbl})" if r is runs[0] else None)

    for r, lbl in zip(runs, labels):
        ax.hist(r["lambda_flare"], bins="sqrt", density=True,
                histtype="step", linewidth=1.5,
                label=fr"flare ({lbl}, $\mu_f={r['mu_flare']:.2f}$)")

    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(r"$\Lambda$ distribution comparison")

    fig.tight_layout()
    fig.savefig(output_dir / "lambda_compare.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # n_sample comparison
    # ------------------------------------------------------------------
    all_n = np.concatenate(
        [r["n_sample_bkg"] for r in runs] + [r["n_sample_flare"] for r in runs]
    )
    n_edges = np.arange(int(all_n.min()) - 0.5, int(all_n.max()) + 1.5, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, lbl in zip(runs, labels):
        ax.hist(r["n_sample_bkg"], bins=n_edges, density=True,
                histtype="step", linewidth=1.2, color="0.5",
                label=f"bkg ({lbl})" if r is runs[0] else None)
    for r, lbl in zip(runs, labels):
        ax.hist(r["n_sample_flare"], bins=n_edges, density=True,
                histtype="step", linewidth=1.5,
                label=fr"flare ({lbl}, $\mu_f={r['mu_flare']:.2f}$)")
    ax.axvline(expected_n, color="black", linestyle="--",
               linewidth=1.0, label=fr"$\mu={expected_n:.2f}$")

    ax.set_xlabel("Number of events in window")
    ax.set_ylabel("Density")
    ax.set_title("Number of events comparison")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "n_sample_compare.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # p-value comparison
    # ------------------------------------------------------------------
    p_edges = np.linspace(0.0, 1.0, 25)

    fig, ax = plt.subplots(figsize=(8, 5))
    for r, lbl in zip(runs, labels):
        ax.hist(r["pvalues_bkg"], bins=p_edges, density=True,
                histtype="step", linewidth=1.2, color="0.5",
                label=fr"$\Lambda$ bkg ({lbl})" if r is runs[0] else None)
    for r, lbl in zip(runs, labels):
        ax.hist(r["pvalues_flare"], bins=p_edges, density=True,
                histtype="step", linewidth=1.5,
                label=fr"$\Lambda$ flare ({lbl})")

    # Poisson p-values, derived from n_sample
    for r, lbl in zip(runs, labels):
        p_bkg = stc.poisson_mid_p_value(r["n_sample_bkg"], expected_n)
        ax.hist(p_bkg, bins=p_edges, density=True,
                histtype="step", linewidth=1.2, color="0.5", linestyle="--",
                label=f"n bkg ({lbl})" if r is runs[0] else None)
    for r, lbl in zip(runs, labels):
        p_flare = stc.poisson_mid_p_value(r["n_sample_flare"], expected_n)
        ax.hist(p_flare, bins=p_edges, density=True,
                histtype="step", linewidth=1.5, linestyle="--",
                label=f"n flare ({lbl})")

    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title("p-value distribution comparison")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "pvalues_compare.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    scripts_out = Path(
        "/lustre/Auger/brais.rocha/spacetime_correlations/output/scripts"
    )
    runs_base = scripts_out / "compare_bg_signal"
    run_dirs = [
        runs_base / "20260525_114456_seed42",
        runs_base / "20260525_130216_seed42",
    ]
    output_dir = scripts_out / "compare_bg_signal_runs"
    main(run_dirs=run_dirs, output_dir=output_dir)
