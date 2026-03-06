from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_lambda_estimator(
    lambda_estimators: Mapping[str, np.ndarray],
    save_path: str | Path,
) -> None:
    """
    Plot and save one or more Lambda estimator distributions.

    Parameters
    ----------
    lambda_estimators
        Mapping from label to array of Lambda estimator values.
    save_path
        Path where the figure will be saved.
    """

    if len(lambda_estimators) == 0:
        raise ValueError("At least one Lambda estimator array must be provided.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, lambda_est in lambda_estimators.items():
        values = np.asarray(lambda_est, dtype=float)

        ax.hist(
            values,
            bins="fd",
            density=True,
            histtype="step",
            linewidth=1.5,
            label=label,
        )

    ax.set_xlabel("Lambda estimator")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Lambda estimator")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

def plot_p_value(
    p_values: Mapping[str, np.ndarray],
    save_path: str | Path,
) -> None:
    """
    Plot and save one or more p-value distributions.

    Parameters
    ----------
    p_values
        Mapping from label to array of p-values.
    save_path
        Path where the figure will be saved.
    """

    if len(p_values) == 0:
        raise ValueError("At least one p-value array must be provided.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, p_value in p_values.items():
        values = np.asarray(p_value, dtype=float)

        ax.hist(
            values,
            bins="fd",
            density=True,
            histtype="step",
            linewidth=1.5,
            label=label,
        )

    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of p-values")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)