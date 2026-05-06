"""
Load saved flare-injection Monte Carlo outputs and make plots.
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

    # data from `results.npz`
    lambda_bkg = data["lambda_bkg"]
    lambda_flare = data["lambda_flare"]

    pvalues_bkg = data["pvalues_bkg"]
    pvalues_flare = data["pvalues_flare"]

    delta_exposure_bkg = data["delta_exposure_bkg"]
    delta_exposure_flare = data["delta_exposure_flare"]

    n_events_bkg = data["n_events_bkg"]
    n_events_flare = data["n_events_flare"]

    # data from `metadata.json`
    mu_window = metadata["mu_window"]
    n_sim = metadata["n_simulations_requested"]
    expected_exposure_rate = metadata["expected_exposure_rate"]
    T_obs_days = metadata["T_obs_days"]
    flare_duration_days = metadata["flare_duration_days"]
    mu_flare = metadata["mu_flare"]


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
    stc.plot_lambda_joint_heatmap(int(mu_window), results_dir)

    # Quantiles of the distribution
    quantiles = [0.1, 0.5]
    q_flare = np.quantile(lambda_flare, quantiles)
    qp_flare = stc.lambda_marginal_sf(q_flare, mu_window)
    qsigma_flare = stc.pvalue_to_sigma(qp_flare)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histograms and grab the color of the flare one
    ax.hist(lambda_bkg, bins="sqrt", density=True,
            histtype="step", linewidth=1.5, label="Isotropy")

    _, _, flare_patches = ax.hist(
        lambda_flare, bins="sqrt", density=True,
        histtype="step", linewidth=1.5, label="Isotropy+Flare"
    )

    flare_color = flare_patches[0].get_edgecolor()

    # Add vertical lines and annotations
    for q, si, qlvl in zip(q_flare, qsigma_flare, quantiles):
        ax.axvline(q, linestyle="--", linewidth=1.5, color=flare_color)

        ax.text(
            q,
            ax.get_ylim()[1] * 0.6,
            fr"$q_{{{int(qlvl*100)}}}$" + "\n" + fr"$\sigma={si:.2f}$",
            rotation=90,
            va="center",
            ha="right",
            color="black"
        )


    ax.set_xlabel(r"$\Lambda$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    #ax.set_title("Histogram of Lambda estimator")
    ax.set_yscale("log")
    ax.legend(loc="center right", bbox_to_anchor=(1,0.7))

    info_text = (
        rf"$N_{{\rm sim}} = {sci_label(n_sim)}$" "\n"
        rf"$\mu_{{\rm window}} = {mu_window:.2f}$" "\n"
        rf"$\mu_{{\rm flare}} = {mu_flare:.2f}$" "\n"
        rf"$T_{{\rm obs}} = {int(T_obs_days/365)}\,\mathrm{{years}}$" "\n"
        rf"$\Delta t_{{\rm flare}} = {int(flare_duration_days/30)}\,\mathrm{{month}}$"
        
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
    fig.savefig(results_dir/ "lambda.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


    # ------------------------------------------------------------------
    # number of events inside window plot
    # ------------------------------------------------------------------


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(n_events_bkg, 
            bins=np.arange(min(n_events_bkg) - 0.5, max(n_events_bkg) + 1.5, 1), # Create bins of width 1 centered on integers
            density=False, histtype="step", linewidth=1.5, label="Isotropy")
    ax.hist(n_events_flare, 
            bins=np.arange(min(n_events_flare) - 0.5, max(n_events_flare) + 1.5, 1),
            density=False, histtype="step", linewidth=1.5, label="Flare")
    ax.axvline(mu_window, color="black", linestyle="--", linewidth=1.5, label="Expected n")

    ax.set_xlabel("Number of events")
    ax.set_ylabel("Counts")
    ax.set_title("Number of events inside window")
    #ax.set_yscale("log")
    ax.legend(loc='center left')

    info_text = (
        rf"$N_{{\rm sim}} = {sci_label(n_sim)}$" "\n"
        rf"$\mu_{{\rm window}} = {mu_window:.2f}$" "\n"
        rf"$\mu_{{\rm flare}} = {mu_flare:.2f}$" "\n"
        rf"$T_{{\rm obs}} = {T_obs_days}\,\mathrm{{d}}$" "\n"
        rf"$\Delta t_{{\rm flare}} = {flare_duration_days}\,\mathrm{{d}}$"
        
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
    ax.hist(pvalues_bkg, bins="sqrt", density=True, histtype="step", linewidth=1.5, label="Isotropy")
    ax.hist(pvalues_flare, bins="sqrt", density=True, histtype="step", linewidth=1.5, label="Flare")

    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of p-values")
    ax.set_yscale("log")
    #ax.set_xscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/ "p_values.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Delta-exposure plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_exposure_bkg, bins="fd", density=False, histtype="step", linewidth=1.5, label="Isotropy")
    ax.hist(delta_exposure_flare, bins="fd", density=False, histtype="step", linewidth=1.5, label="Isotropy+Flare")

    ax.set_xlabel(r"$\Delta\varepsilon$")
    ax.set_ylabel("Counts")
    #ax.set_title(r"Histogram of $\Delta$ exposure")
    ax.set_xlim(-250,20000)
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/ "delta_exp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # log(Delta-exposure plot)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_exposure_bkg, bins="fd", density=False, histtype="step", linewidth=1.5, label="Isotropy")
    ax.hist(delta_exposure_flare, bins="fd", density=False, histtype="step", linewidth=1.5, label="Flare")

    ax.set_xlabel(r"$\Delta$ exposure")
    ax.set_ylabel("Count")
    ax.set_title(r"Histogram of $\Delta$ exposure")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/ "log_delta_exp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    # Change this path to the run you want to plot
    run_dir = Path("/home/brais/PhD/dev/stc_project/output/scripts/flare_injection")
    sim_id = "20260505_162754_seed42"
    main(run_dir/sim_id)