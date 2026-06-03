"""Tests for the Lambda statistics module and ``lambda_estimator`` pipeline."""

import numpy as np
import pytest
import scipy.stats as scp

import spacetimecorr as stc
from spacetimecorr.statistics import (
    empirical_p_values,
    lambda_conditional_cdf,
    lambda_conditional_logsf,
    lambda_conditional_pdf,
    lambda_conditional_rvs,
    lambda_conditional_sf,
    lambda_conditional_sigma,
    lambda_estimator,
    lambda_marginal_isigma,
    lambda_marginal_logsf,
    lambda_marginal_pdf,
    lambda_marginal_rvs,
    lambda_marginal_sf,
    lambda_marginal_sigma,
    pvalue_to_sigma,
    sigma_to_pvalue,
)


# -------------------------------------------------------------------------
# Conditional Lambda: Gamma(n-1, 1)
# -------------------------------------------------------------------------


@pytest.mark.parametrize("n_events", [2, 5, 10, 50])
def test_conditional_pdf_matches_scipy_gamma(n_events):
    x = np.linspace(0.1, 50.0, 10)
    expected = scp.gamma.pdf(x, a=n_events - 1, scale=1.0)
    np.testing.assert_allclose(lambda_conditional_pdf(x, n_events), expected)


@pytest.mark.parametrize("n_events", [2, 5, 10])
def test_conditional_cdf_plus_sf_is_one(n_events):
    x = np.linspace(0.5, 20.0, 8)
    total = lambda_conditional_cdf(x, n_events) + lambda_conditional_sf(x, n_events)
    np.testing.assert_allclose(total, 1.0, atol=1e-12)


@pytest.mark.parametrize("n_events", [2, 5, 10])
def test_conditional_logsf_exp_matches_sf(n_events):
    x = np.linspace(0.5, 20.0, 8)
    np.testing.assert_allclose(
        np.exp(lambda_conditional_logsf(x, n_events)),
        lambda_conditional_sf(x, n_events),
        rtol=1e-12,
    )


def test_conditional_sigma_monotonic_in_x():
    n_events = 10
    x = np.linspace(1.0, 50.0, 20)
    sigmas = lambda_conditional_sigma(x, n_events)
    assert np.all(np.diff(sigmas) >= -1e-9)


def test_conditional_rvs_shape():
    samples = lambda_conditional_rvs(n_events=5, size=200, random_state=0)
    assert np.asarray(samples).shape == (200,)


def test_conditional_rvs_nonneg():
    samples = lambda_conditional_rvs(n_events=5, size=200, random_state=0)
    assert np.all(np.asarray(samples) >= 0.0)


def test_conditional_rvs_mean_close_to_gamma_mean():
    # E[Gamma(n-1, 1)] = n - 1
    samples = lambda_conditional_rvs(n_events=20, size=20_000, random_state=0)
    assert float(np.mean(samples)) == pytest.approx(19.0, rel=0.05)


# -------------------------------------------------------------------------
# Marginal Lambda
# -------------------------------------------------------------------------


def test_marginal_pdf_zero_at_zero():
    # Scalar input: lambda_marginal_pdf returns a Python float.
    val = lambda_marginal_pdf(0.0, mu=5.0)
    assert val == 0.0


def test_marginal_pdf_nonneg():
    x = np.linspace(0.1, 30.0, 20)
    assert np.all(lambda_marginal_pdf(x, mu=5.0) >= 0.0)


def test_marginal_sf_in_unit_interval():
    x = np.linspace(0.5, 40.0, 12)
    sf = lambda_marginal_sf(x, mu=5.0)
    assert np.all((sf >= 0.0) & (sf <= 1.0))


def test_marginal_sf_matches_exp_logsf():
    x = np.linspace(0.5, 40.0, 12)
    np.testing.assert_allclose(
        lambda_marginal_sf(x, mu=5.0),
        np.exp(lambda_marginal_logsf(x, mu=5.0)),
        rtol=1e-12,
    )


def test_marginal_sf_monotonic_decreasing():
    x = np.linspace(0.5, 40.0, 12)
    sf = lambda_marginal_sf(x, mu=5.0)
    assert np.all(np.diff(sf) <= 1e-12)


def test_marginal_sigma_monotonic_increasing_in_x():
    x = np.linspace(0.5, 40.0, 12)
    sigmas = lambda_marginal_sigma(x, mu=5.0)
    assert np.all(np.diff(sigmas) >= -1e-9)


def test_marginal_isigma_round_trip():
    sigma_target = 2.0
    x = lambda_marginal_isigma(sigma_target, mu=5.0)
    sigma_back = lambda_marginal_sigma(x, mu=5.0)
    assert float(sigma_back) == pytest.approx(sigma_target, abs=1e-4)


def test_marginal_logsf_requires_positive_x():
    with pytest.raises(ValueError):
        lambda_marginal_logsf(np.array([0.0]), mu=5.0)


def test_marginal_logsf_requires_positive_mu():
    with pytest.raises(ValueError):
        lambda_marginal_logsf(np.array([1.0]), mu=0.0)


@pytest.mark.parametrize("mu", [3.0, 10.0, 50.0, 200.0])
@pytest.mark.parametrize("sigma_target", [1.0, 3.0, 5.0, 7.0])
def test_marginal_logsf_truncation_converged(mu, sigma_target):
    """The truncation index in lambda_marginal_logsf must be large enough
    that extending it barely changes the result, even for the small tail
    probabilities (high sigma) that matter for sensitivity studies.

    Criterion: compare results using the default truncation index and a
    larger truncation index. Stability is assessed in log space via

        |logsf_b - logsf_a|

    which approximates the relative change in sf for small differences.
    We require convergence at the 1e-10 level.
    """
    x = lambda_marginal_isigma(sigma_target, mu)
    default_nmax = max(50, int(np.ceil(mu + 10.0 * np.sqrt(mu + 1.0) + 50.0)))

    logsf_a = lambda_marginal_logsf(x, mu, nmax=default_nmax)
    logsf_b = lambda_marginal_logsf(x, mu, nmax=default_nmax + 10)

    assert abs(logsf_b - logsf_a) < 1e-10


def test_marginal_rvs_shape():
    samples = lambda_marginal_rvs(mu=5.0, size=200, random_state=0)
    assert np.asarray(samples).shape == (200,)


def test_marginal_rvs_nonneg():
    samples = lambda_marginal_rvs(mu=5.0, size=200, random_state=0)
    assert np.all(np.asarray(samples) >= 0.0)


# -------------------------------------------------------------------------
# p-value <-> sigma utilities
# -------------------------------------------------------------------------


def test_pvalue_to_sigma_round_trip():
    sigmas = np.linspace(0.1, 5.0, 10)
    pvals = sigma_to_pvalue(sigmas)
    np.testing.assert_allclose(pvalue_to_sigma(pvals), sigmas, rtol=1e-9)


def test_pvalue_to_sigma_clips_at_p_zero():
    # The clip protects against p = 0 (otherwise scipy returns +inf).
    # NOTE: the symmetric protection at p = 1 currently does NOT trigger,
    # because `1.0 - np.finfo(float).tiny == 1.0` in float64.  We assert
    # only the protected side here; the p=1 edge case is a separate concern.
    out = pvalue_to_sigma(np.array([0.0]))
    assert np.isfinite(out[0])


# -------------------------------------------------------------------------
# empirical_p_values
# -------------------------------------------------------------------------


def test_empirical_p_values_in_unit_interval():
    rng = np.random.default_rng(0)
    null = rng.normal(size=1000)
    sample = rng.normal(size=200)
    pvals = empirical_p_values(null, sample)
    assert np.all((pvals >= 0.0) & (pvals <= 1.0))


def test_empirical_p_values_uniformity_on_null():
    """When estimators come from the same distribution as the null, p ~ Uniform."""
    rng = np.random.default_rng(0)
    null = rng.normal(size=5000)
    sample = rng.normal(size=5000)
    pvals = empirical_p_values(null, sample)
    assert float(np.mean(pvals)) == pytest.approx(0.5, abs=0.02)


def test_empirical_p_values_empty_null_raises():
    with pytest.raises(ValueError):
        empirical_p_values(np.array([]), np.array([0.1]))


def test_empirical_p_values_non_1d_null_raises():
    with pytest.raises(ValueError):
        empirical_p_values(np.zeros((5, 2)), np.array([0.1]))


# -------------------------------------------------------------------------
# lambda_estimator (end-to-end)
# -------------------------------------------------------------------------


def test_lambda_estimator_returns_finite(window, exposure_model, t0, tf, rng):
    s = stc.EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.assign_directional_exposure(window, exposure_model)
    assert np.isfinite(lambda_estimator(s))


def test_lambda_estimator_raises_without_exposure(window, exposure_model, t0, tf, rng):
    s = stc.EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    with pytest.raises(RuntimeError):
        lambda_estimator(s)


def test_lambda_estimator_raises_with_nan_exposure(window, exposure_model, t0, tf, rng):
    s = stc.EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.assign_directional_exposure(window, exposure_model)
    s.exposure[0] = np.nan
    with pytest.raises(ValueError):
        lambda_estimator(s)


def test_lambda_estimator_raises_with_duplicate_exposure(window, exposure_model, t0, tf, rng):
    s = stc.EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.assign_directional_exposure(window, exposure_model)
    s.exposure[1] = s.exposure[0]  # force a tie
    with pytest.raises(ValueError):
        lambda_estimator(s)


def test_lambda_estimator_raises_with_too_few_events(t0, tf, rng):
    s = stc.EventSample.full_sky(n_total=1, t0=t0, tf=tf, rng=rng)
    s.exposure = np.array([1.0])  # bypass assign to test n_sample guard
    s.expected_exposure_rate = 1.0
    with pytest.raises(ValueError):
        lambda_estimator(s)
