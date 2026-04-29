"""
Load saved isotropy Monte Carlo outputs and make plots.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import spacetimecorr as stc

import scipy.stats as scp

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
    lambda_bkg = data["lambda_bkg"]
    lambda_ST = data["lambda_ST"]
    lambda_T = data["lambda_T"]
    lambda_S = data["lambda_S"]

    n_events_bkg = data["n_events_bkg"]
    n_events_ST = data ["n_events_ST"]
    n_events_T = data ["n_events_T"]
    n_events_S = data ["n_events_S"]

    p_values_bkg = data["p_values_bkg"]
    p_values_ST = data["p_values_ST"]
    p_values_T = data["p_values_T"]
    p_values_S = data["p_values_S"]

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
    # Lambda estimator plot
    # ------------------------------------------------------------------

    # Quantiles and their p-values
    q_bkg = np.quantile(lambda_bkg, [0.05, 0.5])
    q_ST = np.quantile(lambda_ST, [0.05, 0.5])
    q_T = np.quantile(lambda_T, [0.05, 0.5])
    q_S = np.quantile(lambda_S, [0.05, 0.5])

    qp_bkg = stc.lambda_marginal_sf(q_bkg, expected_n)
    qp_ST = stc.lambda_marginal_sf(q_ST, expected_n)
    qp_T = stc.lambda_marginal_sf(q_T, expected_n)
    qp_S = stc.lambda_marginal_sf(q_S, expected_n)

    fig, ax = plt.subplots(figsize=(8, 5))

    _, _, patches_bkg = ax.hist(
        lambda_bkg, bins="sqrt", density=True, histtype="step",
        linewidth=1.5, label="Isotropy"
    )
    _, _, patches_ST = ax.hist(
        lambda_ST, bins="sqrt", density=True, histtype="step",
        linewidth=1.5, label="Space-Time"
    )
    _, _, patches_T = ax.hist(
        lambda_T, bins="sqrt", density=True, histtype="step",
        linewidth=1.5, label="Time Only"
    )
    _, _, patches_S = ax.hist(
        lambda_S, bins="sqrt", density=True, histtype="step",
        linewidth=1.5, label="Space Only"
    )

    quantiles_dict = {
        "ST": {
            "q": q_ST,
            "p": qp_ST,
            "color": patches_ST[0].get_edgecolor(),
        },
        "T": {
            "q": q_T,
            "p": qp_T,
            "color": patches_T[0].get_edgecolor(),
        },
        "S": {
            "q": q_S,
            "p": qp_S,
            "color": patches_S[0].get_edgecolor(),
        },
        # "bkg": {
        #     "q": q_bkg,
        #     "p": qp_bkg,
        #     "color": patches_bkg[0].get_edgecolor(),
        # },
    }

    def plot_quantile_lines(ax, quantiles_dict):
        # Fixed vertical position (in axis coordinates, not data)
        y_text = 0.6  # 60% up the axis

        for _, values in quantiles_dict.items():
            q_vals = values["q"]
            p_vals = values["p"]
            color = values["color"]

            for qi, pi in zip(q_vals, p_vals):
                # Vertical line in histogram color
                ax.axvline(qi, linestyle="--", linewidth=1.5, color=color)

                # Text in black, aligned to same height
                ax.text(
                    qi,
                    y_text,
                    f"p={pi:.4e}",
                    rotation=90,
                    va="center",
                    ha="right",
                    color="black",
                    transform=ax.get_xaxis_transform(),  # <- KEY LINE
                )

    #plot_quantile_lines(ax, quantiles_dict)

    ax.set_xlabel(r"$\Lambda$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    #ax.set_title("Sensitivity Comparison")
    ax.set_yscale("log")
    ax.legend(loc="center right", bbox_to_anchor=(1,0.72))

    info_text = (
        rf"$N_{{\rm sim}} = {sci_label(n_sim)}$" "\n"
        rf"$\mu_{{window}} = {expected_n:.2f}$" "\n"
        rf"$T_{{\rm obs}} = {int(T_obs_days/365)}\,\mathrm{{years}}$"
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
    # number of events inside window plot
    # ------------------------------------------------------------------

    # Compute global min/max across all samples
    n_min = min(
        np.min(n_events_bkg),
        np.min(n_events_ST),
        np.min(n_events_T),
        np.min(n_events_S),
    )

    n_max = max(
        np.max(n_events_bkg),
        np.max(n_events_ST),
        np.max(n_events_T),
        np.max(n_events_S),
    )

    # Integer-centered bins of width 1
    bin_edges = np.arange(n_min - 0.5, n_max + 1.5, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(n_events_bkg, 
            bins=bin_edges, 
            density=False, alpha=0.7, linewidth=1.5, label="Isotropy")
    ax.axvline(expected_n, color="black", linestyle="--", linewidth=1.5, label=fr"$\mu={expected_n:.2f}$")
    ax.hist(n_events_ST, 
            bins=bin_edges, 
            density=False, alpha=0.7, linewidth=1.5, label="Space-Time")
    ax.hist(n_events_T, 
            bins=bin_edges, 
            density=False, histtype="step", linewidth=2, label="Time Only")
    ax.hist(n_events_S, 
            bins=bin_edges, 
            density=False, histtype="step", linewidth=2, label="Space Only")

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
    fig.savefig(results_dir/ "n_events.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # p-value plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p_values_ST, bins=30, density=True, histtype="step",
            linewidth=1.5, label=fr"Space-Time")
    ax.hist(p_values_T, bins=30, density=True, histtype="step",
            linewidth=1.5, label=fr"Time Only")
    ax.hist(p_values_S, bins=30, density=True, histtype="step",
            linewidth=1.5, label=fr"Space Only")
    ax.axhline(y=1, color="black", linestyle="--", linewidth=1.5,
           label="Uniform")
    
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of p-values")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/ "p_values.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # sigma plot
    # ------------------------------------------------------------------

    sigma_bkg = stc.pvalue_to_sigma(p_values_bkg)
    sigma_ST = stc.pvalue_to_sigma(p_values_ST)
    sigma_T = stc.pvalue_to_sigma(p_values_T)
    sigma_S = stc.pvalue_to_sigma(p_values_S)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sigma_bkg, bins=30, density=True, histtype="step",
            linewidth=1.5, label=fr"Isotropy")
    ax.hist(sigma_ST, bins=30, density=True, histtype="step",
            linewidth=1.5, label=fr"Space-Time")
    ax.hist(sigma_T, bins=30, density=True, histtype="step",
            linewidth=1.5, label=fr"Time Only")
    ax.hist(sigma_S, bins=30, density=True, histtype="step",
            linewidth=1.5, label=fr"Space Only")
    
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel("Density")
    ax.set_title(r"Histogram of $\sigma$")
    #ax.set_yscale("log")
    #ax.set_xscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/ "sigma.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    # Change this path to the run you want to plot
    run_dir = Path("/home/brais_rocha/Work/dev/stc_project/output/scripts/sensitivity_study")
    sim_id = "20260428_212445_seed42"
    main(run_dir/sim_id)