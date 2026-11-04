"""
H0 p-values of the windows of a blind-search run of the mock samples
(``run_blind_search_mock.py``, exponential CDF), from the H0 Monte Carlo
distribution of Lambda of an h0_mc_distribution run.

For every window, the p-value is ``P(Lambda >= Lambda_obs)`` under H0 in
each of the two MC bands whose ``mu`` bracket the ``mu`` of the window,
interpolated linearly in ``mu`` on ``log p`` (outside the range of the
bands, the nearest band). In each band, ``P`` is:

- the empirical ``(count + 1) / (N + 1)``, with ``count`` the MC values at
  least as large as ``Lambda_obs``, when ``count >= MIN_COUNT``;
- otherwise, the fitted tail of the band (``fit_h0_tails.py``),
  ``0.1 Q(1/b, ((Lambda_obs - q) / lambda)^b)``.

Every dataset is also summarised by Fisher's statistic of all its windows,
``X = -2 sum log p``, and its global p-value ``p_global``. By default,
``p_global = P(chi2_2k >= X)``, which assumes independent p-values: the
windows of a sky share its fixed number of events (and may overlap), so
``X`` does not follow ``chi2_2k``. With ``--calibrate``, ``p_global`` is
instead the fraction of the complete isotropic skies of the MC (the skies
present in every band) with ``X/2k`` at least as large, ``(count + 1) /
(S + 1)``. The MC windows are circular caps centred on the nside=32 pixels,
which overlap and leave gaps, so this is only meant for blind searches on
the nside=32 grid, and it is conservative for pixel windows.

Output, in every dataset folder: ``pvalues_mc.npz`` (the MC p-value of
every window, NaN where not tested) and ``pvalues_mc.png`` (histogram of
the MC p-values, with the global p-value). In the run folder: ``pvalues_mc.gif``,
``pvalues_mc_summary.json``, ``fisher_X.png`` (``X`` of the datasets against
``chi2_2k``, and against the MC skies with ``--calibrate``) and ``pglobal_log10.png`` (``log10 p_global`` of the datasets
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
    """Per band, sorted by ``mu``: the ring, ``mu``, the sorted exponential
    Lambda values and the fitted tail ``(q, b, lambda)``."""
    with np.load(mc_run_dir / "tail_fit_exp" / "tail_fits.npz") as f:
        tails = {int(r): (q, b, lam) for r, q, b, lam in zip(f["ring"], f["q"], f["b"], f["lambda"])}

    bands = []
    for path in sorted((mc_run_dir / "bands").glob("band_ring*.npz")):
        with np.load(path) as f:
            ring = int(f["ring"])
            lambdas = f["lambdas_exp"]
            tested = np.isfinite(lambdas)
            mu = float(np.broadcast_to(f["mu"], lambdas.shape)[tested].mean())
        bands.append((mu, np.sort(lambdas[tested]), tails[ring], ring))
    bands.sort(key=lambda band: band[0])
    return {
        "mu": np.array([band[0] for band in bands]),
        "values": [band[1] for band in bands],
        "tails": [band[2] for band in bands],
        "rings": [band[3] for band in bands],
    }


def band_pvalue(ref: dict, k: int, x: np.ndarray) -> np.ndarray:
    """``P(Lambda >= x)`` under H0 in band ``k``: empirical where at least
    ``MIN_COUNT`` MC values reach ``x``, the fitted tail beyond."""
    values = ref["values"][k]
    count = values.size - np.searchsorted(values, x, side="left")
    p = (count + 1.0) / (values.size + 1.0)
    beyond = count < MIN_COUNT
    if beyond.any():
        q, b, lam = ref["tails"][k]
        p[beyond] = (1.0 - TAIL_QUANTILE) * sps.gammaincc(1.0 / b, ((x[beyond] - q) / lam) ** b)
    return np.maximum(p, P_FLOOR)


def mc_pvalues(ref: dict, lambdas: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """H0 p-value of every window, interpolated on ``log p`` linearly in
    ``mu`` between the two bands that bracket it."""
    mu_b = ref["mu"]
    hi = np.clip(np.searchsorted(mu_b, mu), 1, mu_b.size - 1)
    lo = hi - 1
    w = np.clip((mu - mu_b[lo]) / (mu_b[hi] - mu_b[lo]), 0.0, 1.0)
    log_p = np.empty_like(lambdas)
    for k in np.unique(lo):
        sel = lo == k
        log_p[sel] = (
            (1.0 - w[sel]) * np.log(band_pvalue(ref, k, lambdas[sel]))
            + w[sel] * np.log(band_pvalue(ref, k + 1, lambdas[sel]))
        )
    return np.exp(log_p)


def fisher(p: np.ndarray) -> dict:
    """Fisher's ``X = -2 sum log p`` and ``log10 P(chi2_2k >= X)``.

    When ``P`` underflows, its log comes from the Wilson-Hilferty normal
    approximation of ``chi2_2k``, accurate for the ``2k ~ 1e5`` here.
    """
    x = float(-2.0 * np.sum(np.log(np.maximum(p, P_FLOOR))))
    k = int(p.size)
    dof = 2 * k
    log_p = scp.chi2.logsf(x, dof)
    if not np.isfinite(log_p):
        z = ((x / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / np.sqrt(2.0 / (9.0 * dof))
        log_p = scp.norm.logsf(z)
    return {"X": x, "k": k, "X_over_2k": x / dof, "log10_p": float(log_p / np.log(10.0))}


def mc_sky_fisher(mc_run_dir: Path, ref: dict) -> np.ndarray:
    """``X/2k`` of every complete isotropic sky of the MC, the skies present
    in every band, with the p-value of every cap from its own band."""
    band_index = {ring: k for k, ring in enumerate(ref["rings"])}
    paths = sorted((mc_run_dir / "bands").glob("band_ring*.npz"))
    # Every band holds the skies 0, 1, ...: the complete ones are those
    # below the smallest number of skies of a band.
    n_complete = min(np.load(p)["skies"].size for p in paths)
    x = np.zeros(n_complete)
    k = np.zeros(n_complete)
    for path in paths:
        with np.load(path) as f:
            ring = int(f["ring"])
            skies = f["skies"]
            lambdas = f["lambdas_exp"]
        rows = skies < n_complete
        values = lambdas[rows]
        tested = np.isfinite(values)
        y = np.zeros_like(values)
        y[tested] = -2.0 * np.log(band_pvalue(ref, band_index[ring], values[tested]))
        x[skies[rows]] += y.sum(axis=1)
        k[skies[rows]] += tested.sum(axis=1)
    return x / (2.0 * k)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_pvalues(p: np.ndarray, p_global: float, *, title: str, output_path: Path) -> None:
    """Histogram of the MC p-values of the windows against a uniform
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


def plot_fisher_x(summary: list[dict], output_path: Path, mc_x_over_2k: np.ndarray | None = None) -> None:
    """Histogram of Fisher's ``X`` of the datasets against the density of
    ``chi2_2k``, with ``k`` the median number of windows, and against the
    ``X`` of the MC skies (``X/2k`` scaled to the same ``k``) when given."""
    x = np.array([row["fisher_mc"]["X"] for row in summary])
    k = int(np.median([row["n_windows"] for row in summary]))
    lo, hi = scp.chi2.ppf([1e-4, 1.0 - 1e-4], 2 * k)
    bins = np.linspace(min(lo, x.min()), max(hi, x.max()), 31)
    width = bins[1] - bins[0]
    grid = np.linspace(bins[0], bins[-1], 500)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.hist(x, bins=bins, histtype="step", linewidth=1.2, color="C0", label="datasets")
    if mc_x_over_2k is not None:
        mc_counts = np.histogram(2 * k * mc_x_over_2k, bins=bins)[0]
        ax.stairs(x.size * mc_counts / mc_x_over_2k.size, bins, color="C1", linewidth=1.0,
                  label=f"isotropic MC skies ({mc_x_over_2k.size})")
    ax.plot(grid, x.size * width * scp.chi2.pdf(grid, 2 * k), color="black", linestyle="--",
            linewidth=0.8, label=rf"$\chi^2_{{2k}}$, $k = {k}$")
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
            "X_over_2k": row["fisher_mc"]["X_over_2k"],
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

def main(mc_run_dir: Path, blind_search_run_dir: Path, calibrate: bool) -> None:
    ref = load_reference(mc_run_dir)
    mc_x = mc_sky_fisher(mc_run_dir, ref) if calibrate else None
    if calibrate:
        print(f"{mc_x.size} complete MC skies: X/2k = {mc_x.mean():.4f} +- {mc_x.std():.4f}")
    datasets = sorted(
        (d for d in blind_search_run_dir.glob("dataset_*") if (d / "results.npz").exists()),
        key=lambda d: int(re.findall(r"\d+", d.name)[-1]),
    )
    summary, plots = [], []
    for d in datasets:
        with np.load(d / "results.npz") as f:
            lambdas, mu, p_old = f["lambda_array"], f["expected_n"], f["pvalues_lambda"]
        tested = np.isfinite(lambdas) & np.isfinite(mu)

        p_mc = np.full(lambdas.shape, np.nan)
        p_mc[tested] = mc_pvalues(ref, lambdas[tested], mu[tested])
        np.savez(d / "pvalues_mc.npz", pvalues_mc=p_mc)

        f_old, f_mc = fisher(p_old[tested]), fisher(p_mc[tested])
        if calibrate:
            p_global = float((np.sum(mc_x >= f_mc["X_over_2k"]) + 1) / (mc_x.size + 1))
        else:
            p_global = float(10.0 ** f_mc["log10_p"])
        plot_pvalues(p_mc[tested], p_global, title=d.name.replace("_", r"\_"),
                     output_path=d / "pvalues_mc.png")
        plots.append(d / "pvalues_mc.png")
        summary.append({
            "dataset": d.name, "n_windows": int(tested.sum()),
            "min_p_mc": float(np.nanmin(p_mc)),
            "p_global": p_global,
            "p_global_method": "mc_skies" if calibrate else "chi2",
            "fisher_mc": f_mc, "fisher_old": f_old,
        })
        print(f"{d.name}: p_global = {p_global:.3g}; min p = {summary[-1]['min_p_mc']:.2e}")

    make_gif(plots, blind_search_run_dir / "pvalues_mc.gif")
    with open(blind_search_run_dir / "pvalues_mc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_ranking(summary, blind_search_run_dir / "pvalues_mc_ranking.json")
    plot_fisher_x(summary, blind_search_run_dir / "fisher_X.png", mc_x)
    plot_log_pglobal(summary, blind_search_run_dir / "pglobal_log10.png")

    print("\nMost significant datasets (p_global):")
    for row in sorted(summary, key=lambda r: r["p_global"])[:10]:
        print(f"  {row['dataset']:>11s}: p_global = {row['p_global']:.3g}, "
              f"X/2k = {row['fisher_mc']['X_over_2k']:.4f}, min p = {row['min_p_mc']:.2e}")
    print("\nMost significant datasets (smallest MC p-value of a window):")
    for row in sorted(summary, key=lambda r: r["min_p_mc"])[:10]:
        print(f"  {row['dataset']:>11s}: min p = {row['min_p_mc']:.2e}, "
              f"p_global = {row['p_global']:.3g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H0 MC p-values of the windows of a blind-search run of the mock samples."
    )
    parser.add_argument("mc_run_dir", type=Path,
                        help="h0_mc_distribution run, with bands/ and tail_fit_exp/.")
    parser.add_argument("blind_search_run_dir", type=Path,
                        help="run_blind_search_mock.py run, with the dataset_* folders.")
    parser.add_argument("--calibrate", action="store_true",
                        help="p_global from the complete MC skies instead of chi2_2k "
                             "(only for blind searches on the nside=32 grid).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.mc_run_dir, args.blind_search_run_dir, args.calibrate)
