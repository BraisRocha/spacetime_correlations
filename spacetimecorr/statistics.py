from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.stats as scp

from .event_sample import EventSample


def lambda_estimator(sample: EventSample) -> Tuple[float, float]:
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
    (lambda_stat, p_value):
        Lambda statistic and the corresponding p-value.
    """

    if not sample.has_exposure:
        raise RuntimeError(
            "Directional exposure not set. Call a method to generate it first."
        )

    if sample.n_events < 2:
        raise ValueError("Need at least 2 events to compute Delta exposure.")

    # Spacings of sorted exposure values
    delta_exp = np.diff(np.sort(sample.dir_exposure))

    # Computation of the Lambda estimator
    lambda_stat = float(-np.sum(np.log(1.0 - np.exp(-delta_exp * sample.exp_rate_exposure))))

    # Gamma survival function
    p_value = float(scp.gamma.sf(lambda_stat, a=sample.n_events - 1, scale=1.0))

    return lambda_stat, p_value

def spatial_estimator(sample: EventSample) -> float:
    """
    Compute a purely spatial correlation estimator.

    The estimator is defined as the Poisson tail probability

        P(N >= n_obs | mu),

    where
        n_obs = sample.n_events
        mu = sample.expected_counts
    """

    return scp.poisson.sf(sample.n_events - 1, sample.expected_counts)

def theoretical_lambda_estimator(
    sample: EventSample
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate theoretical Lambda statistics and corresponding p-values.

    Parameters
    ----------
    sample
        EventSample instance used to determine the Gamma distribution parameters.
    n_simulations
        Number of simulated Lambda values to generate.

    Returns
    -------
    lambda_stat : np.ndarray
        Simulated Lambda statistics.
    p_values : np.ndarray
        Corresponding p-values computed from the Gamma survival function.
    """

    shape = sample.n_events - 1

    lambda_stat = scp.gamma.rvs(a=shape, scale=1.0, size=1)
    p_values = scp.gamma.sf(lambda_stat, a=shape, scale=1.0)

    return lambda_stat, p_values

import numpy as np
import scipy.special as scs


def tau_log_likelihood(sample, n_bins: int = 10) -> float:
    """
    Compute a Poisson-binned log-likelihood statistic from consecutive
    directional-exposure differences.

    The method sorts ``sample.dir_exposure``, computes the consecutive gaps,

        Delta_i = eps[i+1] - eps[i],

    and bins them into ``n_bins`` disjoint intervals between 0 and the maximum
    observed gap.

    For each bin [a, b), the expected number of counts under the null
    hypothesis is approximated as

        lambda_k = (n_events - 1) * [exp(-Gamma * a) - exp(-Gamma * b)],

    where ``Gamma = sample.exp_rate_exposure``.

    The returned statistic is

        lnL = sum_k ln P(c_k | lambda_k),

    where ``P(c_k | lambda_k)`` is the Poisson probability of observing
    ``c_k`` counts in bin ``k``.

    Parameters
    ----------
    sample : EventSample
        Sample containing ``dir_exposure`` and ``exp_rate_exposure``.
    n_bins : int, default=10
        Number of bins used for the gap histogram.

    Returns
    -------
    float
        Poisson-binned log-likelihood statistic.
    """
    eps = np.sort(np.asarray(sample.dir_exposure, dtype=float))
    gamma = float(sample.exp_rate_exposure)

    if eps.ndim != 1:
        raise ValueError("sample.dir_exposure must be a 1D array.")
    if len(eps) < 2:
        raise ValueError("At least two events are required to define gaps.")
    if gamma < 0:
        raise ValueError("sample.exp_rate_exposure must be >= 0.")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    gaps = np.diff(eps)
    n_gaps = len(gaps)

    gmax = gaps.max()

    if gmax <= 0:
        # Degenerate case: all gaps are zero
        return 0.0

    bin_edges = np.linspace(0.0, gmax, n_bins + 1)

    counts, _ = np.histogram(gaps, bins=bin_edges)

    left = bin_edges[:-1]
    right = bin_edges[1:]

    lambda_k = n_gaps * (np.exp(-gamma * left) - np.exp(-gamma * right))

    lnP_k = np.zeros_like(lambda_k, dtype=float)

    positive = lambda_k > 0.0
    lnP_k[positive] = (
        counts[positive] * np.log(lambda_k[positive])
        - lambda_k[positive]
        - scs.gammaln(counts[positive] + 1)
    )

    impossible = (~positive) & (counts > 0)
    if np.any(impossible):
        return -np.inf

    return float(np.sum(lnP_k))

import numpy as np

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
