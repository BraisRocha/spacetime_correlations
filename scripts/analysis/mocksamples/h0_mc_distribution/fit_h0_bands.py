"""
Fit of the Lambda distribution of every declination band of an
h0_mc_distribution run (the output of ``h0_pipeline.py merge``).

For every band, the Lambda values with the suppressed CDF are fitted to a
Gamma distribution of shape ``k = mu - 1`` in the variable
``y = (Lambda - a) / b``::

    f(Lambda) = (1/b) y^(k-1) exp(-y) / Gamma(k),    y = (Lambda - a) / b

with ``mu`` the mean expected count of the band, and ``a`` (bias) and
``b`` (width) free. The fit is unbinned maximum likelihood: ``a`` and ``b``
maximise ``sum log f(Lambda_i)``. Their covariance is the inverse of the
Hessian of ``-sum log f`` at the maximum.

The goodness of fit is Pearson's chi2 of a histogram of the values against
the counts expected from the fitted ``f``, with the bins at the ends merged
until every one expects at least ``MIN_EXPECTED`` values, and
``dof = n_bins - 1 - 2``.

Input, the ``band_ring*.npz`` of ``<run_dir>/bands/``. Output, in
``<run_dir>/fits/``: one plot per band (histogram, fitted ``f``
and residuals), ``fit_vs_dec.png`` (``a``, ``b`` and ``chi2 / dof`` against
the declination of the band) and the fitted values in ``fits.npz`` and
``fits.json``.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sco
import scipy.stats as scp

# ----------------------------------------------------------------------
# Global style: load the matplotlibrc of the plots/ folder
# ----------------------------------------------------------------------
RC_FILE = Path(__file__).resolve().parents[3] / "plots" / "matplotlibrc_test"
if RC_FILE.exists():
    mpl.rc_file(RC_FILE, use_default_template=False)

N_BINS = 60             # histogram bins over the range of the values, as in h0_pipeline.py
MIN_EXPECTED = 5.0      # least expected count of a bin of the chi2


# ----------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------

def negative_log_likelihood(params: np.ndarray, x: np.ndarray, k: float) -> float:
    """``-sum log f(x_i)``, infinite when some ``x_i <= a``, outside the
    support of ``f``."""
    a, b = params
    if b <= 0.0 or np.any(x <= a):
        return np.inf
    return -float(np.sum(scp.gamma.logpdf(x, k, loc=a, scale=b)))


def gradient(params: np.ndarray, x: np.ndarray, k: float) -> np.ndarray:
    """Gradient of :func:`negative_log_likelihood`. With ``y = (x - a)/b``
    and ``log f = -log b + (k - 1) log y - y - log Gamma(k)``::

        d(-log f)/da = (k - 1)/(b y) - 1/b
        d(-log f)/db = (k - y)/b
    """
    a, b = params
    y = (x - a) / b
    return np.array([
        np.sum((k - 1.0) / (b * y) - 1.0 / b),
        np.sum((k - y) / b),
    ])


def hessian(params: np.ndarray, x: np.ndarray, k: float) -> np.ndarray:
    """Hessian of :func:`negative_log_likelihood`::

        d2/da2  = (k - 1)/(b^2 y^2)
        d2/db2  = (2y - k)/b^2
        d2/dadb = 1/b^2
    """
    a, b = params
    y = (x - a) / b
    h_aa = np.sum((k - 1.0) / (b * y) ** 2)
    h_bb = np.sum(2.0 * y - k) / b**2
    h_ab = x.size / b**2
    return np.array([[h_aa, h_ab], [h_ab, h_bb]])


def fit_band(x: np.ndarray, k: float) -> dict:
    """Maximum-likelihood ``a`` and ``b`` of the values ``x`` for the shape
    ``k``, with their uncertainties and correlation."""
    # Start where the mean and variance of f match those of the values:
    # mean = a + b k, variance = b^2 k. A start outside the support is
    # moved just below the smallest value.
    b0 = x.std() / np.sqrt(k)
    a0 = min(x.mean() - b0 * k, x.min() - b0)

    result = sco.minimize(
        negative_log_likelihood, x0=[a0, b0], args=(x, k),
        jac=gradient, hess=hessian, method="trust-exact",
    )
    a, b = result.x
    covariance = np.linalg.inv(hessian(result.x, x, k))
    sigma_a, sigma_b = np.sqrt(np.diag(covariance))
    return {
        "a": float(a),
        "b": float(b),
        "sigma_a": float(sigma_a),
        "sigma_b": float(sigma_b),
        "corr_ab": float(covariance[0, 1] / (sigma_a * sigma_b)),
        "converged": bool(result.success),
    }


# ----------------------------------------------------------------------
# Goodness of fit
# ----------------------------------------------------------------------

def chi2_test(x: np.ndarray, k: float, a: float, b: float) -> dict:
    """Pearson's chi2 of the histogram of ``x`` against the fitted ``f``.

    The bins are ``N_BINS`` of equal width over the range of the values;
    the two outer bins extend to the ends of the support, so that the
    expected counts add up to ``x.size``. Bins expecting fewer than
    ``MIN_EXPECTED`` values are merged with their neighbours, from each end
    towards the centre.
    """
    edges = np.linspace(x.min(), x.max(), N_BINS + 1)
    observed = np.histogram(x, bins=edges)[0].astype(float)
    cdf = scp.gamma.cdf(edges, k, loc=a, scale=b)
    cdf[0], cdf[-1] = 0.0, 1.0
    expected = x.size * np.diff(cdf)

    # Merge from the left: every bin is closed once it expects enough, and
    # a short last bin joins the one before it.
    merged_o, merged_e, merged_edges = [], [], [edges[0]]
    o_acc = e_acc = 0.0
    for o, e, right in zip(observed, expected, edges[1:]):
        o_acc += o
        e_acc += e
        if e_acc >= MIN_EXPECTED:
            merged_o.append(o_acc)
            merged_e.append(e_acc)
            merged_edges.append(right)
            o_acc = e_acc = 0.0
    if e_acc > 0.0:
        merged_o[-1] += o_acc
        merged_e[-1] += e_acc
        merged_edges[-1] = edges[-1]

    observed, expected = np.array(merged_o), np.array(merged_e)
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    dof = observed.size - 1 - 2
    return {
        "chi2": chi2,
        "dof": int(dof),
        "pvalue": float(scp.chi2.sf(chi2, dof)),
        "edges": np.array(merged_edges),
        "observed": observed,
        "expected": expected,
    }


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_band_fit(
    x: np.ndarray, k: float, fit: dict, test: dict, *, title: str, output_path: Path,
) -> None:
    """Histogram of the values with the fitted ``f``, and the residuals
    ``(O - E)/sqrt(E)`` of the bins of the chi2 below."""
    a, b = fit["a"], fit["b"]
    bins = np.linspace(x.min(), x.max(), N_BINS + 1)
    width = bins[1] - bins[0]
    grid = np.linspace(bins[0], bins[-1], 1000)

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(3.4, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    counts = np.histogram(x, bins=bins)[0]
    ax.hist(x, bins=bins, histtype="step", linewidth=1.2, color="C0", label="suppressed CDF")
    curve = x.size * width * scp.gamma.pdf(grid, k, loc=a, scale=b)
    ax.plot(grid, curve, color="black", linestyle="--", linewidth=0.8,
            label=r"$f(x) = \frac{x^{\mu-2}\,e^{-x}}{b\,\Gamma(\mu-1)}$, "
                  r"$x = \frac{\Lambda - a}{b}$")
    ax.text(
        0.04, 0.95,
        rf"$a = {a:.3f} \pm {fit['sigma_a']:.3f}$" "\n"
        rf"$b = {b:.4f} \pm {fit['sigma_b']:.4f}$" "\n"
        rf"$\rho_{{ab}} = {fit['corr_ab']:.3f}$" "\n"
        rf"$\chi^2/\mathrm{{dof}} = {test['chi2'] / test['dof']:.2f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=6,
    )
    # Headroom above the data for the legend and the text.
    ax.set_ylim(0.0, 1.45 * max(counts.max(), curve.max()))
    ax.set_ylabel("counts")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=6)

    edges = test["edges"]
    residuals = (test["observed"] - test["expected"]) / np.sqrt(test["expected"])
    centres = 0.5 * (edges[1:] + edges[:-1])
    ax_res.axhline(0.0, color="black", linewidth=0.6)
    ax_res.errorbar(centres, residuals, yerr=1.0, fmt="o", markersize=2,
                    elinewidth=0.6, capsize=0)
    ax_res.set_xlim(bins[0], bins[-1])
    ax_res.set_xlabel(r"$\Lambda$")
    ax_res.set_ylabel(r"$(O - E)/\sqrt{E}$", fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vs_dec(fits: dict, output_path: Path) -> None:
    """``a``, ``b`` and ``chi2 / dof`` of every band against its declination."""
    dec = fits["dec"]
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.6), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    axes[0].errorbar(dec, fits["a"], yerr=fits["sigma_a"], fmt="o-", markersize=1.2,
                     linewidth=0.6, elinewidth=0.8, capsize=0)
    axes[0].set_ylabel(r"$a$")
    axes[0].set_title(r"Fit of the $\Lambda$ distribution per declination band")

    axes[1].errorbar(dec, fits["b"], yerr=fits["sigma_b"], fmt="o-", markersize=1.2,
                     linewidth=0.6, elinewidth=0.8, capsize=0)
    axes[1].set_ylabel(r"$b$")

    axes[2].plot(dec, fits["chi2"] / fits["dof"], "o-", markersize=1.2, linewidth=0.6)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel(r"$\chi^2/\mathrm{dof}$")
    axes[2].set_xlabel(r"Dec [deg]")

    for ax in axes:
        ax.grid(True, linewidth=0.3, alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(run_dir: Path) -> None:
    """Fit every band of ``run_dir/bands/`` and save the plots and the
    values to ``run_dir/fits/``."""
    output_dir = run_dir / "fits"
    rows = []
    for path in sorted((run_dir / "bands").glob("band_ring*.npz")):
        with np.load(path) as f:
            ring = int(f["ring"])
            dec = float(f["dec"])
            lambdas = f["lambdas"]
            mu = f["mu"]

        tested = np.isfinite(lambdas)
        x = lambdas[tested]
        if x.size < 2:
            print(f"Ring {ring}: {x.size} Lambda values, not fitted")
            continue
        # Mean expected count of the band, over the (sky, cap) with a value.
        mu_band = float(np.broadcast_to(mu, lambdas.shape)[tested].mean())
        k = mu_band - 1.0

        fit = fit_band(x, k)
        test = chi2_test(x, k, fit["a"], fit["b"])
        if not fit["converged"]:
            print(f"Ring {ring}: the fit did not converge")

        plot_band_fit(
            x, k, fit, test,
            title=rf"ring {ring}, Dec $= {dec:.1f}^\circ$, $\mu = {mu_band:.2f}$",
            output_path=output_dir / f"band_ring{ring:03d}_fit.png",
        )
        rows.append({
            "ring": ring, "dec": dec, "mu": mu_band, "n_lambda": int(x.size),
            **fit, "chi2": test["chi2"], "dof": test["dof"], "pvalue": test["pvalue"],
        })
        print(
            f"Ring {ring:3d} (Dec {dec:6.1f}): a = {fit['a']:.3f} +- {fit['sigma_a']:.3f}, "
            f"b = {fit['b']:.4f} +- {fit['sigma_b']:.4f}, corr = {fit['corr_ab']:.3f}, "
            f"chi2/dof = {test['chi2'] / test['dof']:.2f}"
        )

    fits = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez(output_dir / "fits.npz", **fits)
    with open(output_dir / "fits.json", "w") as f:
        json.dump(rows, f, indent=2)
    plot_vs_dec(fits, output_dir / "fit_vs_dec.png")
    print(f"Saved the fits of {len(rows)} bands to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the Lambda distribution of every band of an h0_mc_distribution run."
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Output folder of h0_pipeline.py merge, with the band_ring*.npz files in bands/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().run_dir)
