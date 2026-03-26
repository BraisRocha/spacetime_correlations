"""
Load saved flare-injection Monte Carlo outputs and make plots.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    p_values_bkg = data["p_values_bkg"]

    lambda_flare = data["lambda_flare"]
    p_values_flare = data["p_values_flare"]

    delta_exposure_bkg = data["delta_exposure_bkg"]
    delta_exposure_flare = data["delta_exposure_flare"]

    spatial_p_values = data["spatial_p_values"]

    # data from `metadata.json`
    mu_window = metadata["mu_window"]
    n_sim = metadata["n_simulations_requested"]
    exp_rate_exposure = metadata["exp_rate_exposure"]
    T_obs_days = metadata["T_obs_days"]
    flare_duration_days = metadata["flare_duration_days"]
    mu_flare = metadata["mu_flare"]

    # ------------------------------------------------------------------
    # Lambda estimator plot
    # ------------------------------------------------------------------
    def sci_label(n: int) -> str:
        exponent = int(np.floor(np.log10(n)))
        mantissa = n / 10**exponent

        if np.isclose(mantissa, 1.0):
            return rf"10^{{{exponent}}}"
        return rf"{mantissa:.1f}\times10^{{{exponent}}}"


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lambda_bkg, bins="sqrt", density=True, histtype="step", linewidth=1.5, label="Isotropy")
    ax.hist(lambda_flare, bins="sqrt", density=True, histtype="step", linewidth=1.5, label="Flare")

    ax.set_xlabel("Lambda estimator")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Lambda estimator")
    ax.set_yscale("log")
    ax.legend()

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
    fig.savefig(results_dir/ "lambda.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # p-value plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p_values_bkg, bins="sqrt", density=True, histtype="step", linewidth=1.5, label="Isotropy")
    ax.hist(p_values_flare, bins="sqrt", density=True, histtype="step", linewidth=1.5, label="Flare")
    ax.hist(spatial_p_values, bins="sqrt", density=True, histtype="step", linewidth=1.5, label="Spatial")

    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of p-values")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/ "p_values.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Delta-exposure plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_exposure_bkg, bins="fd", density=False, histtype="step", linewidth=1.5, label="Isotropy")
    ax.hist(delta_exposure_flare, bins="fd", density=False, histtype="step", linewidth=1.5, label="Flare")

    ax.set_xlabel(r"$\Delta$ exposure")
    ax.set_ylabel("Count")
    ax.set_title(r"Histogram of $\Delta$ exposure")
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

    # ------------------------------------------------------------------
    # Delta-exposure*Expected-rate plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_exposure_bkg*exp_rate_exposure, bins="fd", density=False, histtype="step", linewidth=1.5, label="Isotropy")
    ax.hist(delta_exposure_flare*exp_rate_exposure, bins="fd", density=False, histtype="step", linewidth=1.5, label="Flare")

    ax.set_xlabel(r"$\Delta$ exposure $\times$ $\Gamma$")
    ax.set_ylabel("Count")
    ax.set_title(r"Histogram of $\Delta$ exposure $\times$ $\Gamma$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/ "norm_delta_exp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # log(Delta-exposure*Expected-rate) plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_exposure_bkg*exp_rate_exposure, bins="fd", density=False, histtype="step", linewidth=1.5, label="Isotropy")
    ax.hist(delta_exposure_flare*exp_rate_exposure, bins="fd", density=False, histtype="step", linewidth=1.5, label="Flare")

    ax.set_xlabel(r"$\Delta$ exposure $\times$ $\Gamma$")
    ax.set_ylabel("Count")
    ax.set_title(r"Histogram of $\Delta$ exposure $\times$ $\Gamma$")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir/ "log_norm_delta_exp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


    print(f"Saved plots to {results_dir}")


if __name__ == "__main__":
    # Change this path to the run you want to plot
    run_dir = Path("/home/brais_rocha/Work/dev/stc_project/output/scripts/flare_injection")
    sim_id = "20260325_175932_seed42"
    main(run_dir/sim_id)