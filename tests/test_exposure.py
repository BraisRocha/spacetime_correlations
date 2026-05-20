"""Tests for ``ExposureModel`` — construction, invariants, and sampling."""

import numpy as np
import pytest
from astropy.time import Time, TimeDelta
import astropy.units as u

from spacetimecorr import ExposureModel, Observatory


# -------------------------------------------------------------------------
# Construction / validation
# -------------------------------------------------------------------------


def test_construction_stores_observatory(observatory, t0, tf, rng):
    model = ExposureModel(observatory=observatory, t0=t0, tf=tf, rng=rng)
    assert model.observatory is observatory


def test_construction_stores_t0(observatory, t0, tf, rng):
    model = ExposureModel(observatory=observatory, t0=t0, tf=tf, rng=rng)
    assert model.t0 is t0


def test_construction_stores_tf(observatory, t0, tf, rng):
    model = ExposureModel(observatory=observatory, t0=t0, tf=tf, rng=rng)
    assert model.tf is tf


def test_construction_default_theta_max(observatory, t0, tf, rng):
    model = ExposureModel(observatory=observatory, t0=t0, tf=tf, rng=rng)
    assert model.theta_max_deg == pytest.approx(60.0)


def test_invalid_observatory_raises(t0, tf, rng):
    with pytest.raises(TypeError):
        ExposureModel(observatory="not-an-observatory", t0=t0, tf=tf, rng=rng)


def test_invalid_t0_raises(observatory, tf, rng):
    with pytest.raises(TypeError):
        ExposureModel(observatory=observatory, t0="2020-01-01", tf=tf, rng=rng)


def test_tf_before_t0_raises(observatory, rng):
    t0 = Time("2021-01-01")
    tf = Time("2020-01-01")
    with pytest.raises(ValueError):
        ExposureModel(observatory=observatory, t0=t0, tf=tf, rng=rng)


def test_invalid_rng_raises(observatory, t0, tf):
    with pytest.raises(TypeError):
        ExposureModel(observatory=observatory, t0=t0, tf=tf, rng=42)


@pytest.mark.parametrize("bad_theta", [0.0, -1.0, 91.0, 200.0])
def test_invalid_theta_max_raises(observatory, t0, tf, rng, bad_theta):
    with pytest.raises(ValueError):
        ExposureModel(
            observatory=observatory, t0=t0, tf=tf, rng=rng,
            theta_max_deg=bad_theta,
        )


# -------------------------------------------------------------------------
# Instantaneous acceptance
# -------------------------------------------------------------------------


def test_instantaneous_acceptance_in_zero_or_above_cut(exposure_model, t0):
    centre = np.array([180.0, -30.0])
    a = exposure_model.instantaneous_acceptance(t0, centre)
    cos_cut = np.cos(np.deg2rad(exposure_model.theta_max_deg))
    assert (a == 0.0) or (a >= cos_cut)


def test_instantaneous_acceptance_array_shape(exposure_model, t0, tf):
    t = t0 + TimeDelta(np.linspace(0, (tf - t0).to_value(u.s), 50), format="sec")
    a = exposure_model.instantaneous_acceptance(t, np.array([180.0, -30.0]))
    assert a.shape == (50,)


def test_instantaneous_acceptance_out_of_interval_raises(exposure_model, t0):
    with pytest.raises(ValueError):
        exposure_model.instantaneous_acceptance(t0 - TimeDelta(1.0, format="sec"),
                                                np.array([0.0, 0.0]))


# -------------------------------------------------------------------------
# Cumulative directional exposure
# -------------------------------------------------------------------------


def test_cumulative_directional_exposure_zero_at_t0(exposure_model, t0):
    eps = exposure_model.cumulative_directional_exposure(t0, np.array([180.0, -30.0]))
    assert float(eps) == pytest.approx(0.0)


def test_cumulative_directional_exposure_max_at_tf(exposure_model, tf):
    centre = np.array([180.0, -30.0])
    eps = exposure_model.cumulative_directional_exposure(tf, centre)
    max_eps = exposure_model.max_directional_exposure(centre)
    assert float(eps) == pytest.approx(max_eps, rel=1e-9)


def test_cumulative_directional_exposure_monotonic(exposure_model, t0, tf):
    centre = np.array([180.0, -30.0])
    total_sec = (tf - t0).to_value(u.s)
    grid = t0 + TimeDelta(np.linspace(0.0, total_sec, 200), format="sec")
    eps = exposure_model.cumulative_directional_exposure(grid, centre)
    assert np.all(np.diff(np.asarray(eps)) >= -1e-9)


def test_cumulative_directional_exposure_nonneg(exposure_model, t0, tf):
    centre = np.array([180.0, -30.0])
    total_sec = (tf - t0).to_value(u.s)
    grid = t0 + TimeDelta(np.linspace(0.0, total_sec, 100), format="sec")
    eps = exposure_model.cumulative_directional_exposure(grid, centre)
    assert np.all(np.asarray(eps) >= -1e-12)


def test_cumulative_directional_exposure_zero_for_always_invisible(exposure_model):
    # From Auger (lat ~ -35°) with theta_max = 60°, the north celestial pole
    # is always below the horizon → cumulative exposure must stay at 0.
    invisible = np.array([0.0, 80.0])  # dec = +80°, never reaches zenith from Auger
    eps = exposure_model.max_directional_exposure(invisible)
    assert eps == pytest.approx(0.0)


# -------------------------------------------------------------------------
# Relative directional exposure (Sommers)
# -------------------------------------------------------------------------


def test_relative_exposure_zero_for_always_invisible(exposure_model):
    omega = exposure_model.relative_exposure(np.array([0.0, 80.0]))
    assert omega == pytest.approx(0.0)


def test_relative_exposure_nonneg_for_visible(exposure_model):
    omega = exposure_model.relative_exposure(np.array([180.0, -30.0]))
    assert omega >= 0.0


def test_relative_exposure_independent_of_ra(exposure_model):
    omega1 = exposure_model.relative_exposure(np.array([0.0, -30.0]))
    omega2 = exposure_model.relative_exposure(np.array([180.0, -30.0]))
    assert omega1 == pytest.approx(omega2)


# -------------------------------------------------------------------------
# Acceptance mask / detect_times
# -------------------------------------------------------------------------


def test_acceptance_mask_returns_bool_array(exposure_model, t0, tf):
    times = t0 + TimeDelta(
        np.linspace(0.0, (tf - t0).to_value(u.s), 200), format="sec"
    )
    mask = exposure_model.acceptance_mask(times, np.array([180.0, -30.0]))
    assert mask.dtype == bool and mask.shape == (200,)


def test_acceptance_mask_efficiency_zero_rejects_all(exposure_model, t0, tf):
    times = t0 + TimeDelta(
        np.linspace(0.0, (tf - t0).to_value(u.s), 200), format="sec"
    )
    mask = exposure_model.acceptance_mask(
        times, np.array([180.0, -30.0]), efficiency=lambda t: np.zeros(len(t)),
    )
    assert not mask.any()


def test_detect_times_only_returns_accepted_times(exposure_model, t0, tf):
    times = t0 + TimeDelta(
        np.linspace(0.0, (tf - t0).to_value(u.s), 100), format="sec"
    )
    accepted = exposure_model.detect_times(times, np.array([180.0, -30.0]))
    assert len(accepted) <= len(times)


def test_detect_times_with_return_mask(exposure_model, t0, tf):
    times = t0 + TimeDelta(
        np.linspace(0.0, (tf - t0).to_value(u.s), 50), format="sec"
    )
    accepted, mask = exposure_model.detect_times(
        times, np.array([180.0, -30.0]), return_mask=True,
    )
    assert mask.shape == (50,) and len(accepted) == mask.sum()


# -------------------------------------------------------------------------
# sample_directional_exposure
# -------------------------------------------------------------------------


def test_sample_directional_exposure_length(exposure_model):
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_directional_exposure(centre)
    rate = 1000.0 / max_eps
    sample, _ = exposure_model.sample_directional_exposure(
        n_events=1000, expected_exposure_rate=rate, max_dir_exposure=max_eps,
    )
    assert sample.size == 1000


def test_sample_directional_exposure_sorted(exposure_model):
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_directional_exposure(centre)
    rate = 1000.0 / max_eps
    sample, _ = exposure_model.sample_directional_exposure(
        n_events=1000, expected_exposure_rate=rate, max_dir_exposure=max_eps,
    )
    assert np.all(np.diff(sample) >= 0.0)


def test_sample_directional_exposure_within_bounds(exposure_model):
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_directional_exposure(centre)
    rate = 500.0 / max_eps
    sample, _ = exposure_model.sample_directional_exposure(
        n_events=500, expected_exposure_rate=rate, max_dir_exposure=max_eps,
    )
    assert np.all((sample >= 0.0) & (sample <= max_eps))


def test_sample_directional_exposure_method_label(exposure_model):
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_directional_exposure(centre)
    rate = 100.0 / max_eps
    _, method = exposure_model.sample_directional_exposure(
        n_events=100, expected_exposure_rate=rate, max_dir_exposure=max_eps,
    )
    assert method == "free_maximum_exposure_method"


def test_sample_directional_exposure_invalid_rate_raises(exposure_model):
    with pytest.raises(ValueError):
        exposure_model.sample_directional_exposure(
            n_events=10, expected_exposure_rate=0.0, max_dir_exposure=1.0,
        )


def test_sample_directional_exposure_nonpositive_n_raises(exposure_model):
    for n in (-1, 0):
        with pytest.raises(ValueError):
            exposure_model.sample_directional_exposure(
                n_events=n, expected_exposure_rate=1.0, max_dir_exposure=1.0,
            )


def test_sample_directional_exposure_max_zero_raises(exposure_model):
    with pytest.raises(ValueError):
        exposure_model.sample_directional_exposure(
            n_events=10, expected_exposure_rate=1.0, max_dir_exposure=0.0,
        )


def test_sample_directional_exposure_mean_gap(exposure_model):
    """Mean exposure gap is ≈ 1 / expected_exposure_rate (Poisson-process intervals)."""
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_directional_exposure(centre)
    n = 5000
    rate = n / max_eps
    sample, _ = exposure_model.sample_directional_exposure(
        n_events=n, expected_exposure_rate=rate, max_dir_exposure=max_eps,
    )
    mean_gap = float(np.mean(np.diff(sample)))
    assert mean_gap == pytest.approx(1.0 / rate, rel=0.05)
