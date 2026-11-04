"""
Fit of the upper tail of the Lambda distribution of every declination band
of an h0_mc_distribution run (the output of ``h0_pipeline.py merge``).

For every band, the Lambda values (with the suppressed or the exponential
CDF, see ``ESTIMATORS``) above the ``TAIL_QUANTILE`` quantile ``q`` of the
band are fitted to the density::

    g(Lambda) = exp(-(Lambda - q)^b / c) / I,    Lambda > q

with ``b > 0`` and ``c > 0`` free, and ``I`` the integral of the numerator
over ``(q, inf)``::

    I = (c^(1/b) / b) Gamma(1/b)

The tail starts at ``q``: a free origin ``a < q`` in place of ``q`` is not
determined by the tail values alone. The fit is unbinned maximum
likelihood: ``b`` and ``c`` maximise ``sum log g(Lambda_i)``. Their
covariance is the inverse of the Hessian of ``-sum log g`` at the maximum,
by finite differences.

The goodness of fit is Pearson's chi2 of the histogram of the tail values
against the counts expected from the fitted ``g``, with the bins merged
until every one expects at least ``MIN_EXPECTED`` values, and
``dof = n_bins - 1 - 2``.

Input, the ``band_ring*.npz`` of ``<run_dir>/bands/``. Output, in
``<run_dir>/tail_fit/`` (suppressed) or ``<run_dir>/tail_fit_exp/``
(exponential): one plot per band (histogram, fitted ``g`` and
residuals, log scale), ``tail_fit_vs_dec.png`` (``a = q``, ``b``, the
width ``lambda = c^(1/b)`` and ``chi2 / dof`` against the declination of
the band) and the fitted values
in ``tail_fits.npz`` and ``tail_fits.json``.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sco
import scipy.special as sps
import scipy.stats as scp

# Same binning and chi2 merging as the fit of the whole distribution; the
# import also loads the matplotlibrc of the plots/ folder.
from fit_h0_bands import MIN_EXPECTED, N_BINS

TAIL_QUANTILE = 0.90    # the tail is the values above this quantile of the band
N_PARAMS = 2

# Lambda estimator -> (key of the band npz, legend label, output folder).
ESTIMATORS = {
    "suppressed": ("lambdas", "suppressed CDF", "tail_fit"),
    "exponential": ("lambdas_exp", "exponential CDF", "tail_fit_exp"),
}


# ----------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------

def log_norm(b: float, c: float) -> float:
    """``log I``, the log of the integral of ``exp(-t^b / c)`` over
    ``(0, inf)``."""
    return np.log(c) / b - np.log(b) + sps.gammaln(1.0 / b)


def survival(t: np.ndarray, b: float, c: float) -> np.ndarray:
    """The integral of ``g`` over ``(q + t, inf)``."""
    return sps.gammaincc(1.0 / b, t ** b / c)


def negative_log_likelihood(params: np.ndarray, t: np.ndarray) -> float:
    """``-sum log g`` of the excesses ``t = Lambda - q``, infinite outside
    the domain of the parameters."""
    b, c = params
    if b <= 0.0 or c <= 0.0:
        return np.inf
    return float(np.sum(t ** b) / c + t.size * log_norm(b, c))


def numerical_hessian(fun, p: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    """Hessian of ``fun`` at ``p`` by central differences, with steps
    ``rel_step`` relative to each parameter."""
    h = rel_step * np.maximum(np.abs(p), 1.0)
    n = p.size
    hess = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            hess[i, j] = hess[j, i] = (
                fun(p + ei + ej) - fun(p + ei - ej) - fun(p - ei + ej) + fun(p - ei - ej)
            ) / (4.0 * h[i] * h[j])
    return hess


def fit_tail(x: np.ndarray) -> dict:
    """Maximum likelihood ``b`` and ``c`` of the values of ``x`` above its
    ``TAIL_QUANTILE`` quantile ``q``, their errors and correlation.

    The minimisation runs over ``(log b, log c)``, so that both stay
    positive. It starts from the exponential tail: ``b = 1`` and ``c`` the
    mean excess over ``q``.
    """
    q = float(np.quantile(x, TAIL_QUANTILE))
    t = x[x > q] - q

    def objective(p):
        return negative_log_likelihood(np.exp(p), t)

    start = np.log([1.0, t.mean()])
    result = sco.minimize(
        objective, start, method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000},
    )
    params = np.exp(result.x)

    cov = np.linalg.inv(numerical_hessian(lambda p: negative_log_likelihood(p, t), params))
    sigma = np.sqrt(np.diag(cov))

    # The width lambda = c^(1/b), in units of Lambda, and its error and
    # correlation with b propagated from the covariance of (b, c).
    b, c = params
    lam = c ** (1.0 / b)
    grad = np.array([-lam * np.log(c) / b**2, lam / (b * c)])
    sigma_lam = float(np.sqrt(grad @ cov @ grad))
    return {
        "q": q, "n_tail": int(t.size),
        "b": float(b), "c": float(c),
        "sigma_b": float(sigma[0]), "sigma_c": float(sigma[1]),
        "corr_bc": float(cov[0, 1] / (sigma[0] * sigma[1])),
        "lambda": float(lam), "sigma_lambda": sigma_lam,
        "corr_blambda": float((cov[0] @ grad) / (sigma[0] * sigma_lam)),
        "converged": bool(result.success),
    }


# ----------------------------------------------------------------------
# Goodness of fit
# ----------------------------------------------------------------------

def tail_edges(x: np.ndarray, q: float) -> np.ndarray:
    """The edges of the ``N_BINS`` equal bins over the range of ``x`` above
    ``q``, with ``q`` as the first edge."""
    edges = np.linspace(x.min(), x.max(), N_BINS + 1)
    return np.concatenate([[q], edges[edges > q]])


def chi2_test(x: np.ndarray, fit: dict) -> dict:
    """Pearson's chi2 of the histogram of the tail values against the
    fitted ``g``.

    The bins are those of the histogram of all the values above ``q`` (see
    ``tail_edges``); the last one extends to infinity, so that the expected
    counts add up to the number of tail values. Bins expecting fewer than
    ``MIN_EXPECTED`` values are merged with their neighbours, from left to
    right.
    """
    b, c, q = fit["b"], fit["c"], fit["q"]
    tail = x[x > q]
    edges = tail_edges(x, q)
    observed = np.histogram(tail, bins=edges)[0].astype(float)
    cdf = 1.0 - survival(edges - q, b, c)
    cdf[0], cdf[-1] = 0.0, 1.0
    expected = tail.size * np.diff(cdf)

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
    dof = observed.size - 1 - N_PARAMS
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

def plot_tail_fit(
    x: np.ndarray, fit: dict, test: dict, *, label: str, title: str, output_path: Path,
) -> None:
    """Histogram of all the values with the fitted ``g`` above ``q``, in
    log scale, and the residuals ``(O - E)/sqrt(E)`` of the bins of the
    chi2 below."""
    b, c, q, n_tail = fit["b"], fit["c"], fit["q"], fit["n_tail"]
    bins = np.linspace(x.min(), x.max(), N_BINS + 1)
    width = bins[1] - bins[0]
    grid = np.linspace(q, x.max(), 500)
    curve = n_tail * width * np.exp(-(grid - q) ** b / c - log_norm(b, c))
    q_label = rf"\Lambda_{{{100 * TAIL_QUANTILE:.0f}}}"

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(3.4, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    counts = np.histogram(x, bins=bins)[0]
    ax.hist(x, bins=bins, histtype="step", linewidth=1.2, color="C0", label=label)
    ax.plot(grid, curve, color="black", linestyle="--", linewidth=0.8,
            label=rf"$f(x) \propto e^{{-((x - {q_label}) / \lambda)^b}}$")
    ax.axvline(q, color="grey", linestyle=":", linewidth=0.8, label=rf"${q_label}$")
    ax.text(
        0.04, 0.95,
        rf"${q_label} = {q:.2f}$" "\n"
        rf"$b = {b:.4f} \pm {fit['sigma_b']:.4f}$" "\n"
        rf"$\lambda = {fit['lambda']:.3f} \pm {fit['sigma_lambda']:.3f}$" "\n"
        rf"$\rho_{{b\lambda}} = {fit['corr_blambda']:.3f}$" "\n"
        rf"$\chi^2/\mathrm{{dof}} = {test['chi2'] / test['dof']:.2f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=6,
    )
    # Headroom above the data for the legend and the text.
    ax.set_yscale("log")
    ax.set_ylim(0.5, 60.0 * counts.max())
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
    """``a = q``, ``b``, ``lambda = c^(1/b)`` and ``chi2 / dof`` of every
    band against its declination."""
    dec = fits["dec"]
    fig, axes = plt.subplots(4, 1, figsize=(3.4, 6.0), sharex=True,
                             gridspec_kw={"hspace": 0.08})

    # a is not fitted: it is the quantile q where the tail starts.
    axes[0].plot(dec, fits["q"], "o-", markersize=1.2, linewidth=0.6)
    axes[0].set_ylabel(rf"$a = \Lambda_{{{100 * TAIL_QUANTILE:.0f}}}$")
    axes[0].set_title(r"Fit of the upper tail of the $\Lambda$ distribution per declination band")

    for ax, name, label in zip(axes[1:3], ("b", "lambda"), (r"$b$", r"$\lambda = c^{1/b}$")):
        ax.errorbar(dec, fits[name], yerr=fits[f"sigma_{name}"], fmt="o-", markersize=1.2,
                    linewidth=0.6, elinewidth=0.8, capsize=0)
        ax.set_ylabel(label)

    axes[3].plot(dec, fits["chi2"] / fits["dof"], "o-", markersize=1.2, linewidth=0.6)
    axes[3].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel(r"$\chi^2/\mathrm{dof}$")
    axes[3].set_xlabel(r"Dec [deg]")

    for ax in axes:
        ax.grid(True, linewidth=0.3, alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(run_dir: Path, estimator: str) -> None:
    """Fit the tail of the ``estimator`` Lambda of every band of
    ``run_dir/bands/`` and save the plots and the values to its output
    folder in ``run_dir``."""
    key, label, folder = ESTIMATORS[estimator]
    output_dir = run_dir / folder
    rows = []
    for path in sorted((run_dir / "bands").glob("band_ring*.npz")):
        with np.load(path) as f:
            ring = int(f["ring"])
            dec = float(f["dec"])
            lambdas = f[key]
            mu = f["mu"]

        tested = np.isfinite(lambdas)
        x = lambdas[tested]
        mu_band = float(np.broadcast_to(mu, lambdas.shape)[tested].mean())

        fit = fit_tail(x)
        test = chi2_test(x, fit)
        if not fit["converged"]:
            print(f"Ring {ring}: the fit did not converge")

        plot_tail_fit(
            x, fit, test, label=label,
            title=rf"ring {ring}, Dec $= {dec:.1f}^\circ$, $\mu = {mu_band:.2f}$",
            output_path=output_dir / f"band_ring{ring:03d}_tail_fit.png",
        )
        rows.append({
            "ring": ring, "dec": dec, "mu": mu_band, "n_lambda": int(x.size),
            **fit, "chi2": test["chi2"], "dof": test["dof"], "pvalue": test["pvalue"],
        })
        print(
            f"Ring {ring:3d} (Dec {dec:6.1f}): q = {fit['q']:.2f}, "
            f"b = {fit['b']:.4f} +- {fit['sigma_b']:.4f}, c = {fit['c']:.3f} +- {fit['sigma_c']:.3f}, "
            f"corr = {fit['corr_bc']:.3f}, chi2/dof = {test['chi2'] / test['dof']:.2f}"
        )

    fits = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez(output_dir / "tail_fits.npz", **fits)
    with open(output_dir / "tail_fits.json", "w") as f:
        json.dump(rows, f, indent=2)
    plot_vs_dec(fits, output_dir / "tail_fit_vs_dec.png")
    print(f"Saved the tail fits of {len(rows)} bands to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the upper tail of the Lambda distribution of every band of an "
                    "h0_mc_distribution run."
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Output folder of h0_pipeline.py merge, with the band_ring*.npz files in bands/.",
    )
    parser.add_argument(
        "--estimator", choices=sorted(ESTIMATORS), default="suppressed",
        help="Lambda whose tail is fitted (default: suppressed).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.run_dir, args.estimator)
