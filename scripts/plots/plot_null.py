"""
Load saved null-hypothesis MC outputs and make plots.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spacetimecorr as stc

import scipy.stats as scp
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

def main(results_dir: str | Path) -> None:
    """
    Load a saved Monte Carlo run and create the plots.
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

    # data from `results.npz`
    lambda_mc = data["lambda_mc"]
    n_sample = data["n_sample_window"]

    p_values_conditional = data["p_values_conditional"]
    p_values_marginal = data["p_values_marginal"]
    p_values_spatial = data["p_values_spatial"]

    # metadata from `metadata.json`
    expected_n = metadata["expected_n"]
    n_sim = metadata["n_simulations_successful"]
    T_obs_days = metadata["T_obs_days"]

    # Scientific notation for the plots
    def sci_label(n: int) -> str:
        exponent = int(np.floor(np.log10(n)))
        mantissa = n / 10**exponent

        if np.isclose(mantissa, 1.0):
            return rf"10^{{{exponent}}}"
        return rf"{mantissa:.1f}\times10^{{{exponent}}}"
    
    # ------------------------------------------------------------------
    # f_Lambda (lambda | n) Heatmap
    # ------------------------------------------------------------------
    
    stc.plot_lambda_joint_heatmap(int(expected_n), results_dir)

    # ------------------------------------------------------------------
    # Lambda estimator plot
    # ------------------------------------------------------------------

    # Quantiles and associated tail probabilities
    q = np.quantile(lambda_mc, [0.05, 0.5])
    q_p_value = stc.lambda_marginal_sf(q, expected_n)

    # Build theoretical conditional pdf and numerical cdf
    x_max = np.max(lambda_mc) * 1.2
    x_full = np.linspace(0.0, x_max, num=int(1e4))

    pdf_full = stc.lambda_marginal_pdf(x_full, expected_n)
    # Atom at 0
    p0 = np.exp(-expected_n) * (1.0 + expected_n)
    print(f"p0 = {p0:.3e}")
    # Conditional pdf on x > 0
    pdf_cond_full = pdf_full / (1.0 - p0)

    # Numerical conditional cdf
    cdf_vals = cumulative_trapezoid(pdf_cond_full, x_full, initial=0.0)
    cdf_vals /= cdf_vals[-1]  # protect against tiny numerical drift

    cdf_func = interp1d(
        x_full,
        cdf_vals,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 1.0),
    )

    # KS test
    D, p_value = scp.kstest(lambda_mc, cdf_func)
    print(f"KS statistic = {D:.6f}")
    print(f"KS p-value   = {p_value:.6g}")

    # Plot
    x_plot = np.linspace(
        np.min(lambda_mc),
        np.max(lambda_mc),
        num=int(1e4)
    )
    pdf_plot = stc.lambda_marginal_pdf(x_plot, expected_n)
    pdf_cond_plot = pdf_plot / (1.0 - p0)

    # Sample lambda from the marginal distribution
    lambda_marginal = stc.lambda_marginal_rvs(expected_n, size=len(lambda_mc))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(lambda_mc, bins="sqrt", density=True, histtype="step",
        linewidth=1.5, label="MC",)

    #ax.hist(lambda_marginal, bins="sqrt", density=True, histtype="step",
        #linewidth=1.5, label=fr"$f_{{\Lambda}}(x\mid\mu)$ rvs",)

    ax.plot(x_plot, pdf_cond_plot, linewidth=2.0, linestyle="-", 
        label=fr"$f(\Lambda\mid\mu)$ pdf",)
    
    """
    # Quantile markers
    for qi, pi in zip(q, q_p_value):
        ax.axvline(qi, linestyle="--", linewidth=1.5)
        ax.text(
            qi,
            ax.get_ylim()[1] * 0.5,
            f"q={qi:.2f}\np={pi:.4f}",
            rotation=90,
            va="center",
            ha="right",
        )
    """
    ax.set_xlabel("Lambda estimator")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Lambda estimator")
    #ax.set_yscale("log")
    ax.legend()

    # Info box
    info_text = (
        rf"$N_{{\rm sim}} = {sci_label(n_sim)}$" "\n"
        rf"$\mu = {expected_n:.2f}$" "\n"
        rf"$T_{{\rm obs}} = {T_obs_days}\,\mathrm{{d}}$" "\n"
        rf"$D_{{\rm KS}} = {D:.4f}$" "\n"
        rf"$p_{{\rm KS}} = {p_value:.3g}$"
    )

    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(results_dir / "lambda.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Cumulative Density Function
    # ------------------------------------------------------------------

    x_data = np.sort(lambda_mc)
    ecdf = np.arange(1, len(x_data) + 1) / len(x_data)

    plt.figure(figsize=(8, 5))

    # Empirical CDF (step-like)
    plt.step(x_data, ecdf, where="post", linewidth=1, label="ECDF")

    # Model CDF (smooth)
    plt.plot(x_full, cdf_vals, linewidth=1.0, label="Model CDF")

    plt.xlabel(r"$\Lambda$")
    plt.ylabel("CDF")
    plt.title("ECDF vs Model CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "cdf.png", dpi=300, bbox_inches="tight")
    plt.close()


    # ------------------------------------------------------------------
    # number of events inside window plot
    # ------------------------------------------------------------------


    fig, ax = plt.subplots(figsize=(8, 5)) # Create bins of width 1 centered on integers
    bin_edges = np.arange(min(n_sample) - 0.5, max(n_sample) + 1.5, 1)
    ax.hist(n_sample, 
            bins=bin_edges, 
            density=False, histtype="step", linewidth=1.5, label="MC")
    ax.axvline(expected_n, color="black", linestyle="--", linewidth=1.5, label=fr"$\mu={expected_n:.2f}$")

    k = np.arange(min(n_sample), max(n_sample) + 1)
    pmf = scp.poisson.pmf(k, expected_n)
    N = len(n_sample)
    expected_counts = N * pmf

    step_y = np.r_[expected_counts, expected_counts[-1]]
    ax.step(
        bin_edges,
        step_y,
        where="post",
        linewidth=2.0,
        linestyle="-",
        label="Poisson",
    )

    ax.set_xlabel("Number of events")
    ax.set_ylabel("Counts")
    ax.set_title("Number of events inside window")
    #ax.set_yscale("log")
    ax.legend()

    info_text = (
        rf"$N_{{\rm sim}} = {sci_label(n_sim)}$" "\n"
        rf"$T_{{\rm obs}} = {T_obs_days}\,\mathrm{{d}}$"
    )

    ax.text(
        0.98,
        0.98,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(results_dir/ "n_sample.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # p-value plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p_values_conditional, bins=30, density=True, histtype="step",
            linewidth=1.5, label=fr"$f_{{\Lambda}}(x|n,\mu)$")
    ax.hist(p_values_marginal, bins=30, density=True, histtype="step",
            linewidth=1.5, label=fr"$f_{{\Lambda}}(x|\mu)$")
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1.5,
           label="Uniform")

    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of p-values")
    #ax.set_yscale("log")
    #ax.set_xscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/ "p_values.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    # Change this path to the run you want to plot
    run_dir = Path("/lustre/Auger/brais.rocha/spacetime_correlations/output/scripts/null")
    sim_id = "20260518_164126_seed42"
    main(run_dir/sim_id)