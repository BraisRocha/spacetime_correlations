from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.stats as scp
import scipy.special as scs
import os

from .event_sample import EventSample


def _pyplot():
    """Lazy matplotlib.pyplot import (avoids requiring a display at import time)."""
    import matplotlib.pyplot as plt
    return plt

# --------------------------------------------------------------------------------
# Lambda Conditional PDF | $f_{Lambda}(x|n) = Gamma(n-1,1)$
# --------------------------------------------------------------------------------

def lambda_conditional_pdf(x, n_events):
    """
    Probability density of ``Lambda`` given ``n`` events::

        Lambda | n  ~  Gamma(shape = n - 1, scale = 1).

    Parameters
    ----------
    x : float or array-like
        Evaluation point(s).
    n_events : int or array-like of int
        Number of events conditioning the distribution. Must be ``>= 2``.

    Returns
    -------
    float or numpy.ndarray
        PDF value(s) at ``x``.
    """
    x = np.asarray(x)
    n_events = np.asarray(n_events)

    shape = n_events - 1
    return scp.gamma.pdf(x, a=shape, loc=0.0, scale=1.0)

def lambda_conditional_cdf(x, n_events):
    """
    Cumulative distribution function of ``Lambda | n ~ Gamma(n-1, 1)``.

    Parameters and returns mirror :func:`lambda_conditional_pdf`.
    """
    x = np.asarray(x)
    n_events = np.asarray(n_events)

    shape = n_events - 1
    return scp.gamma.cdf(x, a=shape, loc=0.0, scale=1.0)

def lambda_conditional_sf(x, n_events):
    """
    Upper-tail probability ``P(Lambda >= x | n)`` for ``Lambda | n ~ Gamma(n-1, 1)``.

    This is the conditional p-value associated with the observed Lambda
    value at fixed event count ``n``.
    """
    x = np.asarray(x)
    n_events = np.asarray(n_events)

    shape = n_events - 1
    return scp.gamma.sf(x, a=shape, loc=0.0, scale=1.0)

def lambda_conditional_logsf(x, n_events):
    """
    Logarithm of the conditional upper-tail probability,
    ``log P(Lambda >= x | n)``.

    Useful when the survival function underflows (very large ``x``).
    """
    x = np.asarray(x)
    n_events = np.asarray(n_events)

    shape = n_events - 1
    return scp.gamma.logsf(x, a=shape, loc=0.0, scale=1.0)

def lambda_conditional_sigma(x, n_events):
    """
    One-sided Gaussian-equivalent significance corresponding
    to the upper-tail p-value.
    """
    x = np.asarray(x)
    n_events = np.asarray(n_events)

    p = lambda_conditional_sf(x, n_events)
    return scp.norm.isf(p)

def lambda_conditional_pvalue(x, n_events):
    """Alias for :func:`lambda_conditional_sf`, the conditional upper-tail p-value."""
    x = np.asarray(x)
    n_events = np.asarray(n_events)

    return lambda_conditional_sf(x, n_events)

def lambda_conditional_pvalue_and_sigma(x, n_events):
    """
    Return both the conditional p-value ``P(Lambda >= x | n)`` and its
    one-sided Gaussian-equivalent significance.
    """
    x = np.asarray(x)
    n_events = np.asarray(n_events)

    p = lambda_conditional_sf(x, n_events)
    z = scp.norm.isf(p)
    return p, z

def lambda_conditional_rvs(n_events, size=1, random_state=None):
    """
    Draw random samples from ``Lambda | n ~ Gamma(n-1, 1)``.

    Parameters
    ----------
    n_events : int or array-like of int
        Conditioning event count(s); must be ``>= 2``.
    size : int or tuple of int, optional
        Sample size, forwarded to ``scipy.stats.gamma.rvs``.
    random_state : int, numpy.random.Generator, or None, optional
        Random state forwarded to ``scipy.stats``.
    """
    shape = n_events - 1
    return scp.gamma.rvs(
        a=shape,
        loc=0.0,
        scale=1.0,
        size=size,
        random_state=random_state,
    )

# --------------------------------------------------------------------------------
# Lambda Marginal PDF | $f_{Lambda}(x) = e^{-mu-x}frac{mu}{x}I_{2}(2sqrt(mu*x))$
# --------------------------------------------------------------------------------

def lambda_marginal_pdf(x, mu):
    """
    Continuous part of the marginal PDF of Lambda for x > 0.

    f_Lambda(x) = exp(-mu - x) * (mu / x) * I_2(2 * sqrt(mu * x))

    Notes
    -----
    This is the non-atomic part only. It is not normalized over 
    (0, inf); there is leftover point mass at x = 0 with 
    P(x=0) = exp(-mu) * (1 + mu).
    """

    x = np.asarray(x, dtype=float)

    pdf = np.zeros_like(x, dtype=float)
    mask = x > 0

    xm = x[mask]
    pdf[mask] = np.exp(-mu - xm) * (mu / xm) * scs.iv(2, 2.0 * np.sqrt(mu * xm))

    return pdf if pdf.ndim > 0 else float(pdf)

def lambda_marginal_logsf(x, mu, nmax=None):
    """
    Log survival function of the marginal Lambda distribution.

    log SF(x) = log P(Lambda >= x)
              = log[ exp(-mu) * sum_{n=2}^infty mu^n / n! * Q(n-1, x) ]

    where Q(n-1, x) is the regularized upper incomplete gamma function.

    Parameters
    ----------
    x : float or array-like
        Evaluation point(s).
    mu : float
        Poisson mean parameter; must be > 0.
    nmax : int or None, optional
        Truncation index for the infinite sum. If None, a conservative
        automatic value is used.

    Returns
    -------
    float or ndarray
        log survival function at x.
    """

    if mu <= 0:
        raise ValueError("mu must be > 0")

    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("This implementation expects x > 0")

    if nmax is None:
        # Conservative Poisson-tail truncation
        nmax = max(50, int(np.ceil(mu + 10.0 * np.sqrt(mu + 1.0) + 50.0)))

    n = np.arange(2, nmax + 1, dtype=float)
    a = n - 1.0  # Gamma shapes

    # log Poisson weights: log[e^{-mu} mu^n / n!]
    logw = -mu + n * np.log(mu) - scs.gammaln(n + 1.0)

    # Vectorised over x: gamma.logsf broadcasts x[..., None] against a[None, :].
    x_arr = np.atleast_1d(x)
    logQ = scp.gamma.logsf(x_arr[:, None], a=a[None, :], loc=0.0, scale=1.0)
    out = scs.logsumexp(logw[None, :] + logQ, axis=1)

    if np.ndim(x) == 0:
        return float(out[0])
    return out.reshape(np.shape(x))

def lambda_marginal_sf(x, mu, nmax=None):
    """
    Survival function SF(x) = P(Lambda >= x)
    computed as exp(logsf) for numerical consistency.
    """
    logsf = lambda_marginal_logsf(x, mu, nmax=nmax)
    return np.exp(logsf)

def lambda_marginal_sigma(x, mu, nmax=None):
    """
    One-sided Gaussian-equivalent significance corresponding
    to the upper-tail p-value for the marginal Lambda distribution.
    """
    p = lambda_marginal_sf(x, mu, nmax=nmax)
    return scp.norm.isf(p)

def lambda_marginal_pvalue(x, mu, nmax=None):
    """Alias for :func:`lambda_marginal_sf`, the marginal upper-tail p-value."""
    return lambda_marginal_sf(x, mu, nmax=nmax)

def lambda_marginal_pvalue_and_sigma(x, mu, nmax=None):
    """
    Return both the marginal p-value ``P(Lambda >= x)`` (with ``N ~ Poisson(mu)``,
    truncated to ``N >= 2``) and its one-sided Gaussian-equivalent significance.
    """
    p = lambda_marginal_sf(x, mu, nmax=nmax)
    z = scp.norm.isf(p)
    return p, z

def lambda_marginal_rvs(mu, size=1, random_state=None):
    """
    Random samples from the marginal Lambda distribution.

    Construction:
      N ~ Poisson(mu)
      Lambda | N=n ~ Gamma(n-1, 1), for n >= 2

    Samples with N < 2 are rejected and resampled, so this generates
    from the marginal distribution corresponding to the series starting
    at n=2.
    """
    if mu <= 0:
        raise ValueError("mu must be > 0")

    rng = np.random.default_rng(random_state)

    size_tuple = np.atleast_1d(size)
    total = int(np.prod(size_tuple))

    out = np.empty(total, dtype=float)
    filled = 0

    while filled < total:
        need = total - filled

        n = rng.poisson(mu, size=need)
        valid = n >= 2
        nv = n[valid]

        if nv.size == 0:
            continue

        draws = rng.gamma(shape=nv - 1.0, scale=1.0)
        k = draws.size
        out[filled:filled + k] = draws
        filled += k

    if np.isscalar(size):
        return out[0] if size == 1 else out
    return out.reshape(size)

# --------------------------------------------------------------------------------
# Anisotropy estimators
# --------------------------------------------------------------------------------

def lambda_estimator(sample: EventSample) -> float:
    """
    Compute the Lambda test statistic and its p-value.

    The statistic is computed from the spacings of the sorted directional exposure
    values. The p-value is obtained from the Gamma survival function.

    Parameters
    ----------
    sample:
        EventSample instance with directional exposure already computed.

    Returns
    -------
    lambda_stat:
        Lambda estimator
    """

    if not sample.has_exposure:
        raise RuntimeError(
            "Directional exposure not set. Call a method to generate it first."
        )

    if sample.n_events < 2:
        raise ValueError("Need at least 2 events to compute Delta exposure.")

    # Reject incomplete exposure arrays before doing anything else: NaN
    # comparisons silently evaluate to False, so a partly-filled array
    # would slip through the duplicate check below and produce a NaN
    # statistic. This commonly happens when `inject_flare` is called and
    # the caller forgets the follow-up `assign_directional_exposure`.
    exposure = np.asarray(sample.exposure, dtype=float)
    if not np.all(np.isfinite(exposure)):
        raise ValueError(
            "lambda_estimator: sample.exposure contains non-finite values. "
            "Did you call inject_flare() without a subsequent "
            "assign_directional_exposure() to fill the background slots?"
        )

    # Spacings of sorted exposure values
    delta_exp = np.diff(np.sort(exposure))

    # Duplicate exposure values produce delta_exp == 0, which makes the
    # log term diverge. The spacings model assumes a continuous exposure
    # distribution, so ties indicate a quantisation issue upstream rather
    # than something we should silently regularise away.
    if np.any(delta_exp <= 0.0):
        raise ValueError(
            "lambda_estimator: duplicate (or non-increasing) exposure values "
            "detected. The spacings model requires strictly increasing exposure "
            "values; check the upstream sampling for quantisation."
        )

    # Computation of the Lambda estimator
    lambda_stat = float(-np.sum(np.log(1.0 - np.exp(-delta_exp * sample.expected_exposure_rate))))

    return lambda_stat

def spatial_estimator(n_events, mu) -> float|np.ndarray:
    """
    Purely spatial correlation estimator: Poisson upper-tail probability.

    For an observed event count ``n_events`` in a sky window with
    expected count ``mu`` under the null (e.g. isotropy under uniform
    full-sky exposure), this returns

        P(N >= n_events | mu)  =  Q(n_events, mu),

    where ``N ~ Poisson(mu)``. Smaller values indicate a stronger
    (positive) excess.

    Parameters
    ----------
    n_events : int or array-like of int
        Observed event count(s).
    mu : float or array-like
        Expected count(s) under the null.

    Returns
    -------
    float or numpy.ndarray
        Upper-tail probability/probabilities; broadcasts over inputs.
    """
    n_events = np.asarray(n_events, dtype=float)
    mu = np.asarray(mu, dtype=float)

    return scp.poisson.sf(n_events - 1, mu)

# --------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------

def pvalue_to_sigma(p):
    """
    Convert one-sided p-values to Gaussian-equivalent significances.

    Parameters
    ----------
    p : float or array-like
        One-sided p-value(s), expected in the interval [0, 1].

    Returns
    -------
    sigma : float or ndarray
        Gaussian-equivalent significance/significances.
    """
    p = np.asarray(p, dtype=float)

    # Avoid infinities at exactly 0 or 1
    eps = np.finfo(float).tiny
    p = np.clip(p, eps, 1.0 - eps)

    return scp.norm.isf(p)

def sigma_to_pvalue(sigma):
    """
    Convert one-sided Gaussian-equivalent significances to p-values.

    Parameters
    ----------
    sigma : float or array-like
        Gaussian-equivalent significance/significances.

    Returns
    -------
    p : float or ndarray
        One-sided p-value(s).
    """
    sigma = np.asarray(sigma, dtype=float)
    return scp.norm.sf(sigma)

def plot_lambda_joint_heatmap(
    mu,
    output_dir,
    nsigma_n=4.0,
    lambda_half_width=70,
    n_lambda=500,
    filename="lambda_joint_heatmap.png",
    label_fontsize=15,
    tick_fontsize=13,
    title_fontsize=16,
    legend_fontsize=12,
    log_range=30,
):
    """
    Plot a heatmap over (n, Lambda) for the truncated model

        N ~ Poisson(mu), conditioned on N >= 2
        Lambda | N=n ~ Gamma(n-1, 1)

    The heatmap color shows

        P(N=n | N>=2) * f(Lambda | n)

    Parameters
    ----------
    mu : float
        Poisson mean parameter.
    output_dir : str
        Directory where the figure will be saved.
    nsigma_n : float, optional
        Half-width of the n-axis in units of sqrt(mu).
    lambda_half_width : float, optional
        Half-width of the Lambda axis around a reference median.
    n_lambda : int, optional
        Number of Lambda grid points.
    filename : str, optional
        Name of the output file.
    label_fontsize, tick_fontsize, title_fontsize, legend_fontsize : int
        Font sizes.
    log_range : float, optional
        Number of log10 units below the maximum to keep in the color scale.

    Returns
    -------
    lambda_center : float
        Reference Lambda value used to center the x-axis.
    """

    if mu <= 0:
        raise ValueError("mu must be > 0")

    # Representative central n
    n_ref = max(2, int(round(mu)))

    # n-range chosen from Poisson spread
    n_half_width = max(8, int(np.ceil(nsigma_n * np.sqrt(mu + 1.0))))
    n_min = max(2, n_ref - n_half_width)
    n_max = n_ref + n_half_width
    n_values = np.arange(n_min, n_max + 1)

    # Reference Lambda center: median for representative n
    lambda_center = scp.gamma.ppf(0.5, a=n_ref - 1, scale=1.0)
    lam_min = max(1e-8, lambda_center - lambda_half_width)
    lam_max = lambda_center + lambda_half_width
    lambda_grid = np.linspace(lam_min, lam_max, n_lambda)

    # Truncated Poisson weights: P(N=n | N>=2)
    trunc_norm = 1.0 - np.exp(-mu) * (1.0 + mu)
    weights = scp.poisson.pmf(n_values, mu) / trunc_norm

    # Compute heatmap: P(N=n | N>=2) * f(Lambda | n)
    Z = np.zeros((len(n_values), len(lambda_grid)))
    for i, n in enumerate(n_values):
        pdf_vals = scp.gamma.pdf(lambda_grid, a=n - 1, scale=1.0)
        Z[i, :] = weights[i] * pdf_vals

    # Log scaling
    Z_log = np.log10(np.maximum(Z, 1e-300))
    zmax = np.max(Z_log)
    zmin = zmax - log_range

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        Z_log,
        aspect="auto",
        origin="lower",
        extent=[lam_min, lam_max, n_min, n_max],
        vmin=zmin,
        vmax=zmax,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        r"$\log_{10}\!\left[P(N=n\mid N\geq 2)\,f(\Lambda\mid n)\right]$",
        fontsize=label_fontsize,
    )
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # Guide lines
    ax.axhline(mu, linestyle="--", linewidth=1.5, color="white",
               label=fr"$\mu = {mu:.2f}$")
    ax.axvline(lambda_center, linestyle="--", linewidth=1.5, color="red",
               label=fr"reference median $\Lambda \approx {lambda_center:.2f}$")

    ax.set_xlabel(r"$\Lambda$", fontsize=label_fontsize)
    ax.set_ylabel(r"Number of events $n$", fontsize=label_fontsize)
    ax.set_title(
        r"Heatmap of $P(N=n\mid N\geq 2)\,f(\Lambda\mid n)$",
        fontsize=title_fontsize,
    )

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return lambda_center

# --------------------------------------------------------------------------------
# Empirical (Monte-Carlo based) p-values
# --------------------------------------------------------------------------------

def empirical_p_values(null_estimators: np.ndarray, estimators: np.ndarray) -> np.ndarray:
    """
    Compute empirical one-sided p-values from a null distribution.

    The p-value for each estimator x is defined as the fraction of null
    simulations with estimator values smaller than or equal to x, i.e.

        p(x) = #{null <= x} / N_null

    This convention assumes that more negative estimator values are more
    extreme.

    Parameters
    ----------
    null_estimators : np.ndarray
        Array of estimator values obtained under the null hypothesis
        (e.g. isotropy).
    estimators : np.ndarray
        Array of estimator values for which p-values are to be computed.
        This can be the same array as ``null_estimators`` or another sample
        (e.g. isotropy+flare).

    Returns
    -------
    np.ndarray
        Empirical p-values for ``estimators``.
    """
    null_estimators = np.asarray(null_estimators, dtype=float)
    estimators = np.asarray(estimators, dtype=float)

    if null_estimators.ndim != 1:
        raise ValueError("null_estimators must be a 1D array.")
    if estimators.ndim != 1:
        raise ValueError("estimators must be a 1D array.")
    if len(null_estimators) == 0:
        raise ValueError("null_estimators must not be empty.")

    null_sorted = np.sort(null_estimators)

    # Number of null values <= each estimator
    counts = np.searchsorted(null_sorted, estimators, side="right")

    return counts / len(null_sorted)

