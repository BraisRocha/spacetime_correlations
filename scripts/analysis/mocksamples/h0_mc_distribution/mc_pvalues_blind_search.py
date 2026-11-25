"""
H0 p-values of the pixels of a blind-search run of the mock samples
(``run_blind_search_mock.py``), from the H0 Monte Carlo distribution of the
suppressed Lambda of an h0_mc_distribution run (``h0_pipeline.py``) and its
tail fits (``fit_h0_tails.py``).

The blind search and the Monte Carlo share their pixels, so every pixel is
compared with the band of its own ring. The p-value is
``P(Lambda >= Lambda_obs)`` under H0 in that band:

- the empirical ``(count + 1) / (N + 1)``, with ``count`` the MC values at
  least as large as ``Lambda_obs``, when ``count >= MIN_COUNT``;
- otherwise, the fitted tail of the band,
  ``0.1 Q(1/b, ((Lambda_obs - q) / lambda)^b)``.

Every dataset is also summarised by Fisher's statistic of all its pixels,
``X = -2 sum log p``, over ``k`` pixels. The pixels of a sky share its
number of events, so ``X`` does not follow ``chi2_2k``: the global p-value
is taken from the isotropic skies of the Monte Carlo (``skies.npz``), every
pixel of every sky with its p-value from the same bands,

    p_global = (number of MC skies with X/2k at least as large + 1) / (S + 1)

Output, in every dataset folder: ``pvalues_mc.npz`` (the MC p-value of
every pixel, NaN where not tested) and ``pvalues_mc.png`` (histogram of the
MC p-values, with the global p-value). In the run folder:
``pvalues_mc.gif``, ``pvalues_mc_summary.json``, ``pvalues_mc_ranking.json``,
``fisher_X.png`` (``X`` of the datasets against the MC skies) and
``pglobal_log10.png`` (``log10 p_global`` of the datasets
against a uniform ``p_global``).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps
import scipy.stats as scp
from PIL import Image

# The import also loads the matplotlibrc of the plots/ folder.
import fit_h0_bands  # noqa: F401

MIN_COUNT = 100         # least MC values above Lambda_obs to use the empirical p
TAIL_QUANTILE = 0.90    # as in fit_h0_tails.py
P_FLOOR = 1e-300        # floor of the p-values in log p
GIF_SECONDS = 0.5


# ----------------------------------------------------------------------
# H0 reference
# ----------------------------------------------------------------------

def load_reference(mc_run_dir: Path) -> dict:
    """Per ring: the sorted suppressed Lambda values of its band and its
    fitted tail ``(q, b, lambda)``."""
    with np.load(mc_run_dir / "tail_fit" / "tail_fits.npz") as f:
        tails = {int(r): (q, b, lam) for r, q, b, lam in zip(f["ring"], f["q"], f["b"], f["lambda"])}

    ref = {}
    for path in sorted((mc_run_dir / "bands").glob("band_ring*.npz")):
        with np.load(path) as f:
            ring = int(f["ring"])
            lambdas = f["lambdas"]
        ref[ring] = (np.sort(lambdas[np.isfinite(lambdas)]), tails[ring])
    return ref


def band_pvalue(values: np.ndarray, tail: tuple, x: np.ndarray) -> np.ndarray:
    """``P(Lambda >= x)`` under H0 in a band of sorted ``values`` with the
    fitted ``tail``: empirical where at least ``MIN_COUNT`` values reach
    ``x``, the fitted tail beyond."""
    count = values.size - np.searchsorted(values, x, side="left")
    p = (count + 1.0) / (values.size + 1.0)
    beyond = count < MIN_COUNT
    if beyond.any():
        q, b, lam = tail
        p[beyond] = (1.0 - TAIL_QUANTILE) * sps.gammaincc(1.0 / b, ((x[beyond] - q) / lam) ** b)
    return np.maximum(p, P_FLOOR)


def pixel_pvalues(ref: dict, rings: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """H0 p-value of every pixel, from the band of its ring."""
    p = np.empty(lambdas.shape)
    for ring in np.unique(rings):
        sel = rings == ring
        p[sel] = band_pvalue(*ref[int(ring)], lambdas[sel])
    return p


def fisher(p: np.ndarray) -> dict:
    """Fisher's ``X = -2 sum log p`` and ``log10 P(chi2_2k >= X)``.

    When ``P`` underflows, its log comes from the Wilson-Hilferty normal
    approximation of ``chi2_2k``, accurate for the ``2k ~ 1e4`` here.
    """
    x = float(-2.0 * np.sum(np.log(np.maximum(p, P_FLOOR))))
    k = int(p.size)
    dof = 2 * k
    log_p = scp.chi2.logsf(x, dof)
    if not np.isfinite(log_p):
        z = ((x / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / np.sqrt(2.0 / (9.0 * dof))
        log_p = scp.norm.logsf(z)
    return {"X": x, "k": k, "X_over_2k": x / dof, "log10_p_chi2": float(log_p / np.log(10.0))}


def mc_sky_fisher(mc_run_dir: Path, ref: dict) -> np.ndarray:
    """``X/2k`` of every isotropic sky of the MC (``skies.npz``), with the
    p-value of every pixel from the band of its ring."""
    with np.load(mc_run_dir / "skies.npz") as f:
        rings = f["rings"]
        lambdas = f["lambdas"]
    x = np.zeros(lambdas.shape[0])
    k = np.zeros(lambdas.shape[0])
    for ring in np.unique(rings):
        values = lambdas[:, rings == ring].astype(float)
        tested = np.isfinite(values)
        y = np.zeros_like(values)
        y[tested] = -2.0 * np.log(band_pvalue(*ref[int(ring)], values[tested]))
        x += y.sum(axis=1)
        k += tested.sum(axis=1)
    return x / (2.0 * k)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_pvalues(p: np.ndarray, p_global: float, *, title: str, output_path: Path) -> None:
    """Histogram of the MC p-values of the pixels against a uniform
    distribution, with the global (Fisher) p-value."""
    bins = np.linspace(0.0, 1.0, 41)
    uniform = p.size / (bins.size - 1)
    counts = np.histogram(p, bins=bins)[0]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.hist(p, bins=bins, histtype="step", linewidth=1.2, color="C0")
    ax.axhline(uniform, color="black", linestyle="--", linewidth=0.8, label="uniform")
    ax.text(0.04, 0.95, rf"$p_{{\rm global}} = {p_global:.2g}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.set_ylim(0.0, 1.25 * max(counts.max(), uniform))
    ax.set_xlabel(r"$p$-value")
    ax.set_ylabel("counts")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fisher_x(summary: list[dict], mc_x_over_2k: np.ndarray, output_path: Path) -> None:
    """Histogram of Fisher's ``X`` of the datasets against the ``X`` of the
    MC skies, ``X/2k`` scaled to the median number ``k`` of pixels."""
    x = np.array([row["fisher"]["X"] for row in summary])
    k = int(np.median([row["n_windows"] for row in summary]))
    mc_x = 2 * k * mc_x_over_2k
    bins = np.linspace(min(x.min(), mc_x.min()), max(x.max(), mc_x.max()), 31)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.hist(x, bins=bins, histtype="step", linewidth=1.2, color="C0", label="datasets")
    mc_counts = np.histogram(mc_x, bins=bins)[0]
    ax.stairs(x.size * mc_counts / mc_x.size, bins, color="C1", linewidth=1.0,
              label=f"isotropic MC skies ({mc_x.size})")
    # Headroom above the histograms for the legend.
    ax.set_ylim(0.0, 1.3 * max(np.histogram(x, bins=bins)[0].max(), x.size * mc_counts.max() / mc_x.size))
    ax.set_xlabel(r"$X = -2 \sum \ln p$")
    ax.set_ylabel("counts")
    ax.set_title("Fisher statistic of the datasets")
    ax.legend(loc="upper left", fontsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_log_pglobal(summary: list[dict], output_path: Path) -> None:
    """Histogram of ``log10 p_global`` of the datasets against the counts
    expected for a uniform ``p_global``, ``P(log10 p < y) = 10^y``."""
    y = np.log10([row["p_global"] for row in summary])
    bins = np.linspace(min(np.floor(y.min()), -2.0), 0.0, 21)
    expected = y.size * np.diff(10.0 ** bins)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.hist(y, bins=bins, histtype="step", linewidth=1.2, color="C0", label="datasets")
    ax.stairs(expected, bins, color="black", linestyle="--", linewidth=0.8,
              label=r"uniform $p_{\rm global}$")
    ax.set_xlabel(r"$\log_{10} p_{\rm global}$")
    ax.set_ylabel("counts")
    ax.set_title(r"Global $p$-value of the datasets")
    ax.legend(loc="upper left", fontsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_ranking(summary: list[dict], output_path: Path) -> None:
    """The datasets from the smallest ``p_global`` to the largest."""
    ranking = [
        {
            "rank": rank,
            "dataset": row["dataset"],
            "p_global": row["p_global"],
            "sigma_global": float(scp.norm.isf(row["p_global"])),
            "X_over_2k": row["fisher"]["X_over_2k"],
            "min_p_window": row["min_p_mc"],
        }
        for rank, row in enumerate(sorted(summary, key=lambda r: r["p_global"]), start=1)
    ]
    with open(output_path, "w") as f:
        json.dump(ranking, f, indent=2)


def make_gif(paths: list[Path], output_path: Path) -> None:
    """GIF of the plots, centred on a common white canvas."""
    frames = [Image.open(p).convert("RGB") for p in paths]
    width = max(im.size[0] for im in frames)
    height = max(im.size[1] for im in frames)
    canvases = []
    for im in frames:
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(im, ((width - im.size[0]) // 2, (height - im.size[1]) // 2))
        canvases.append(canvas)
    canvases[0].save(output_path, save_all=True, append_images=canvases[1:],
                     duration=int(1000 * GIF_SECONDS), loop=0)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(mc_run_dir: Path, blind_search_run_dir: Path) -> None:
    ref = load_reference(mc_run_dir)
    mc_x = mc_sky_fisher(mc_run_dir, ref)
    print(f"{mc_x.size} MC skies: X/2k = {mc_x.mean():.4f} +- {mc_x.std():.4f}")

    datasets = sorted(
        (d for d in blind_search_run_dir.glob("dataset_*") if (d / "results.npz").exists()),
        key=lambda d: int(re.findall(r"\d+", d.name)[-1]),
    )
    summary, plots = [], []
    for d in datasets:
        with np.load(d / "results.npz") as f:
            lambdas, rings = f["lambda_array"], f["rings"]
        tested = np.isfinite(lambdas)

        p_mc = np.full(lambdas.shape, np.nan)
        p_mc[tested] = pixel_pvalues(ref, rings[tested], lambdas[tested])
        np.savez(d / "pvalues_mc.npz", pvalues_mc=p_mc)

        f_mc = fisher(p_mc[tested])
        p_global = float((np.sum(mc_x >= f_mc["X_over_2k"]) + 1) / (mc_x.size + 1))
        plot_pvalues(p_mc[tested], p_global, title=d.name.replace("_", r"\_"),
                     output_path=d / "pvalues_mc.png")
        plots.append(d / "pvalues_mc.png")
        summary.append({
            "dataset": d.name, "n_windows": int(tested.sum()),
            "min_p_mc": float(np.nanmin(p_mc)),
            "p_global": p_global,
            "fisher": f_mc,
        })
        print(f"{d.name}: p_global = {p_global:.3g}; min p = {summary[-1]['min_p_mc']:.2e}")

    make_gif(plots, blind_search_run_dir / "pvalues_mc.gif")
    with open(blind_search_run_dir / "pvalues_mc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_ranking(summary, blind_search_run_dir / "pvalues_mc_ranking.json")
    plot_fisher_x(summary, mc_x, blind_search_run_dir / "fisher_X.png")
    plot_log_pglobal(summary, blind_search_run_dir / "pglobal_log10.png")

    print("\nMost significant datasets (p_global):")
    for row in sorted(summary, key=lambda r: r["p_global"])[:10]:
        print(f"  {row['dataset']:>11s}: p_global = {row['p_global']:.3g}, "
              f"X/2k = {row['fisher']['X_over_2k']:.4f}, min p = {row['min_p_mc']:.2e}")
    print("\nMost significant datasets (smallest MC p-value of a pixel):")
    for row in sorted(summary, key=lambda r: r["min_p_mc"])[:10]:
        print(f"  {row['dataset']:>11s}: min p = {row['min_p_mc']:.2e}, "
              f"p_global = {row['p_global']:.3g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H0 MC p-values of the pixels of a blind-search run of the mock samples."
    )
    parser.add_argument("mc_run_dir", type=Path,
                        help="h0_mc_distribution run, with bands/, tail_fit/ and skies.npz.")
    parser.add_argument("blind_search_run_dir", type=Path,
                        help="run_blind_search_mock.py run, with the dataset_* folders.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.mc_run_dir, args.blind_search_run_dir)
