"""Tests for ``EventSample`` — bare constructor, factories, and contracts."""

import numpy as np
import pytest
import scipy.stats as scp
from astropy.time import Time

from spacetimecorr import EventSample, Flare, SkyWindow
from spacetimecorr.statistics import lambda_estimator, lambda_marginal_sf


# -------------------------------------------------------------------------
# Bare constructor validation
# -------------------------------------------------------------------------


def test_init_with_valid_inputs(t0, tf, rng):
    s = EventSample(n_sample=10, n_total=100, t0=t0, tf=tf, rng=rng)
    assert s.n_sample == 10 and s.n_total == 100


def test_init_negative_n_sample_raises(t0, tf, rng):
    with pytest.raises(ValueError):
        EventSample(n_sample=-1, n_total=10, t0=t0, tf=tf, rng=rng)


def test_init_negative_n_total_raises(t0, tf, rng):
    with pytest.raises(ValueError):
        EventSample(n_sample=10, n_total=-1, t0=t0, tf=tf, rng=rng)


def test_init_non_integer_n_sample_raises(t0, tf, rng):
    with pytest.raises(TypeError):
        EventSample(n_sample=10.5, n_total=10, t0=t0, tf=tf, rng=rng)


def test_init_invalid_t0_raises(tf, rng):
    with pytest.raises(TypeError):
        EventSample(n_sample=10, n_total=10, t0="2020-01-01", tf=tf, rng=rng)


def test_init_tf_before_t0_raises(rng):
    with pytest.raises(ValueError):
        EventSample(
            n_sample=10, n_total=10,
            t0=Time("2021-01-01"), tf=Time("2020-01-01"),
            rng=rng,
        )


def test_init_invalid_rng_raises(t0, tf):
    with pytest.raises(TypeError):
        EventSample(n_sample=10, n_total=10, t0=t0, tf=tf, rng=42)


def test_init_expected_n_is_none(t0, tf, rng):
    s = EventSample(n_sample=10, n_total=10, t0=t0, tf=tf, rng=rng)
    assert s.expected_n is None


def test_init_window_is_none(t0, tf, rng):
    s = EventSample(n_sample=10, n_total=10, t0=t0, tf=tf, rng=rng)
    assert s.window is None


def test_init_exposure_model_is_none(t0, tf, rng):
    s = EventSample(n_sample=10, n_total=10, t0=t0, tf=tf, rng=rng)
    assert s.exposure_model is None


def test_init_has_coordinates_false(t0, tf, rng):
    s = EventSample(n_sample=10, n_total=10, t0=t0, tf=tf, rng=rng)
    assert not s.has_coordinates


# -------------------------------------------------------------------------
# full_sky factory
# -------------------------------------------------------------------------


def test_full_sky_n_sample_equals_n_total(t0, tf, rng):
    s = EventSample.full_sky(n_total=500, t0=t0, tf=tf, rng=rng)
    assert s.n_sample == 500


def test_full_sky_expected_n_equals_n_total(t0, tf, rng):
    s = EventSample.full_sky(n_total=500, t0=t0, tf=tf, rng=rng)
    assert s.expected_n == pytest.approx(500.0)


def test_full_sky_spatial_type(t0, tf, rng):
    s = EventSample.full_sky(n_total=500, t0=t0, tf=tf, rng=rng)
    assert s.spatial_type == "full_sky"


def test_full_sky_window_is_none(t0, tf, rng):
    s = EventSample.full_sky(n_total=500, t0=t0, tf=tf, rng=rng)
    assert s.window is None


def test_full_sky_exposure_model_is_none(t0, tf, rng):
    s = EventSample.full_sky(n_total=500, t0=t0, tf=tf, rng=rng)
    assert s.exposure_model is None


def test_full_sky_ra_shape(t0, tf, rng):
    s = EventSample.full_sky(n_total=500, t0=t0, tf=tf, rng=rng)
    assert s.ra.shape == (500,)


def test_full_sky_dec_shape(t0, tf, rng):
    s = EventSample.full_sky(n_total=500, t0=t0, tf=tf, rng=rng)
    assert s.dec.shape == (500,)


def test_full_sky_ra_in_range(t0, tf, rng):
    s = EventSample.full_sky(n_total=2000, t0=t0, tf=tf, rng=rng)
    assert np.all((s.ra >= 0.0) & (s.ra < 360.0))


def test_full_sky_dec_in_range(t0, tf, rng):
    s = EventSample.full_sky(n_total=2000, t0=t0, tf=tf, rng=rng)
    assert np.all((s.dec >= -90.0) & (s.dec <= 90.0))


def test_full_sky_coordinates_finite(t0, tf, rng):
    s = EventSample.full_sky(n_total=2000, t0=t0, tf=tf, rng=rng)
    assert np.all(np.isfinite(s.ra)) and np.all(np.isfinite(s.dec))


def test_full_sky_has_coordinates_true(t0, tf, rng):
    s = EventSample.full_sky(n_total=10, t0=t0, tf=tf, rng=rng)
    assert s.has_coordinates


def test_full_sky_has_exposure_false(t0, tf, rng):
    s = EventSample.full_sky(n_total=10, t0=t0, tf=tf, rng=rng)
    assert not s.has_exposure


def test_full_sky_has_flare_false(t0, tf, rng):
    s = EventSample.full_sky(n_total=10, t0=t0, tf=tf, rng=rng)
    assert not s.has_flare


def test_full_sky_isotropy_mean_sin_dec(t0, tf, rng):
    """For isotropic full-sky sampling, mean(sin Dec) ≈ 0."""
    s = EventSample.full_sky(n_total=50_000, t0=t0, tf=tf, rng=rng)
    mean_sin = float(np.mean(np.sin(np.deg2rad(s.dec))))
    assert mean_sin == pytest.approx(0.0, abs=0.02)


def test_full_sky_isotropy_mean_ra(t0, tf, rng):
    """For isotropic full-sky sampling, mean(RA) ≈ 180."""
    s = EventSample.full_sky(n_total=50_000, t0=t0, tf=tf, rng=rng)
    assert float(np.mean(s.ra)) == pytest.approx(180.0, abs=3.0)


# -------------------------------------------------------------------------
# in_window factory
# -------------------------------------------------------------------------


def test_in_window_expected_n_matches_window_formula(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    assert s.expected_n == pytest.approx(
        window.expected_n_in_window(10_000, exposure_model)
    )


def test_in_window_stores_window(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    assert s.window is window


def test_in_window_stores_exposure_model(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    assert s.exposure_model is exposure_model


def test_in_window_spatial_type(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    assert s.spatial_type == "window"


def test_in_window_all_inside(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    assert np.all(window.contains(s.ra, s.dec))


def test_in_window_ra_dec_same_length(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    assert s.ra.shape == s.dec.shape == (s.n_sample,)


def test_in_window_n_total_preserved(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    assert s.n_total == 10_000


def test_in_window_zero_poisson_draw_raises(exposure_model, t0, tf):
    """An invisible-cap construction must trigger the n_sample==0 guard."""
    invisible = SkyWindow(centre=[180.0, 89.0], radius=0.05)
    with pytest.raises(ValueError):
        EventSample.in_window(
            window=invisible, n_total=1, exposure_model=exposure_model,
            t0=t0, tf=tf, rng=np.random.default_rng(0),
        )


# -------------------------------------------------------------------------
# expected_temporal_rate
# -------------------------------------------------------------------------


def test_expected_temporal_rate_raises_when_expected_n_unset(t0, tf, rng):
    s = EventSample(n_sample=10, n_total=10, t0=t0, tf=tf, rng=rng)
    with pytest.raises(RuntimeError):
        s.expected_temporal_rate


def test_expected_temporal_rate_after_full_sky(t0, tf, rng):
    import astropy.units as u
    s = EventSample.full_sky(n_total=365, t0=t0, tf=tf, rng=rng)
    assert s.expected_temporal_rate == pytest.approx(
        s.expected_n / s.T_obs.to(u.s).value
    )


# -------------------------------------------------------------------------
# select_subsample
# -------------------------------------------------------------------------


def test_select_subsample_all_events_inside(window, t0, tf, rng):
    parent = EventSample.full_sky(n_total=20_000, t0=t0, tf=tf, rng=rng)
    sub = parent.select_subsample(window)
    assert np.all(window.contains(sub.ra, sub.dec))


def test_select_subsample_n_sample_matches_array_length(window, t0, tf, rng):
    parent = EventSample.full_sky(n_total=20_000, t0=t0, tf=tf, rng=rng)
    sub = parent.select_subsample(window)
    assert sub.n_sample == len(sub.ra)


def test_select_subsample_sets_expected_n(window, t0, tf, rng):
    parent = EventSample.full_sky(n_total=20_000, t0=t0, tf=tf, rng=rng)
    sub = parent.select_subsample(window)
    assert sub.expected_n == pytest.approx(window.expected_n_in_window(20_000))


def test_select_subsample_sets_window(window, t0, tf, rng):
    parent = EventSample.full_sky(n_total=20_000, t0=t0, tf=tf, rng=rng)
    sub = parent.select_subsample(window)
    assert sub.window is window


def test_select_subsample_propagates_n_total(window, t0, tf, rng):
    parent = EventSample.full_sky(n_total=20_000, t0=t0, tf=tf, rng=rng)
    sub = parent.select_subsample(window)
    assert sub.n_total == 20_000


def test_select_subsample_no_coordinates_raises(window, t0, tf, rng):
    parent = EventSample(n_sample=10, n_total=10, t0=t0, tf=tf, rng=rng)
    with pytest.raises(ValueError):
        parent.select_subsample(window)


def test_select_subsample_no_events_inside_raises(exposure_model, t0, tf, rng):
    parent = EventSample.full_sky(n_total=10, t0=t0, tf=tf, rng=rng)
    invisible = SkyWindow(centre=[180.0, 89.999], radius=0.001)
    # Almost certainly no random events land in such a tiny cap.
    with pytest.raises(ValueError):
        parent.select_subsample(invisible)


# -------------------------------------------------------------------------
# assign_directional_exposure
# -------------------------------------------------------------------------


def test_assign_directional_exposure_sets_exposure(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.assign_directional_exposure(window, exposure_model)
    assert s.has_exposure


def test_assign_directional_exposure_sets_exposure_type(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.assign_directional_exposure(window, exposure_model)
    assert s.exposure_type == "free_maximum_exposure_method"


def test_assign_directional_exposure_sets_rate(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.assign_directional_exposure(window, exposure_model)
    assert s.expected_exposure_rate is not None
    assert s.expected_exposure_rate > 0.0


def test_assign_directional_exposure_array_length(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=10_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.assign_directional_exposure(window, exposure_model)
    assert len(s.exposure) == s.n_sample


# -------------------------------------------------------------------------
# inject_flare
# -------------------------------------------------------------------------


@pytest.fixture
def generated_flare(window, exposure_model, t0, tf, rng_manager):
    import astropy.units as u
    flare = Flare(
        n_flare=20,
        duration=1.0 * u.day,
        t0=t0,
        tf=tf,
        centre=window.centre,
        exposure_model=exposure_model,
        rng=rng_manager.get("flare"),
    )
    flare.generate_in_window(window=window, sigma=2.0)
    return flare


def test_inject_flare_grows_n_sample(window, exposure_model, t0, tf, rng, generated_flare):
    """n_sample = n_before - n_removed + n_flare; with small window and small
    flare, n_removed is typically 0 or 1 so n_sample almost always grows."""
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    n_before = s.n_sample
    s.inject_flare(generated_flare, mode="overdensity")
    # n_sample_after = n_before - n_removed + n_flare with n_removed in
    # [0, n_before]: bounded between n_flare and n_before + n_flare.
    assert generated_flare.n_flare <= s.n_sample <= n_before + generated_flare.n_flare


def test_inject_flare_array_lengths_consistent(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="overdensity")
    assert s.n_sample == len(s.ra) == len(s.dec) == len(s.exposure) == len(s.flare_mask)


def test_inject_flare_appends_at_tail(window, exposure_model, t0, tf, rng, generated_flare):
    """The new semantics put the flare events at the end of the arrays."""
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="overdensity")
    tail = slice(-generated_flare.n_flare, None)
    np.testing.assert_array_equal(s.ra[tail], generated_flare.ra)
    np.testing.assert_array_equal(s.dec[tail], generated_flare.dec)
    np.testing.assert_array_equal(s.exposure[tail], generated_flare.exposure)
    assert s.flare_mask[tail].all() and not s.flare_mask[:-generated_flare.n_flare].any()


def test_inject_flare_overdensity_n_flare_greater_than_n_sample_ok(
    window, exposure_model, t0, tf, rng_manager,
):
    """In overdensity mode, n_flare may exceed n_sample (the array grows by
    appending the tail) as long as it does not exceed n_total."""
    import astropy.units as u
    # n_sample ~ Poisson(expected_n) is typically much smaller than n_total.
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng_manager.get("sample"),
    )
    flare = Flare(
        n_flare=s.n_sample + 50, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng_manager.get("flare"),
    )
    flare.generate_in_window(window=window, sigma=2.0)
    s.inject_flare(flare, mode="overdensity")  # must not raise
    assert s.n_sample >= flare.n_flare


def test_inject_flare_overdensity_n_flare_exceeds_n_total_raises(
    window, exposure_model, t0, tf, rng_manager,
):
    """In overdensity mode, n_flare > n_total is incoherent (the flare is
    notionally drawn from a hypothetical full-sky sample of size n_total),
    and must raise."""
    import astropy.units as u
    s = EventSample.full_sky(n_total=5, t0=t0, tf=tf, rng=rng_manager.get("sample"))
    big_flare = Flare(
        n_flare=10, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng_manager.get("flare"),
    )
    big_flare.generate_in_window(window=window, sigma=2.0)
    with pytest.raises(ValueError):
        s.inject_flare(big_flare, mode="overdensity")


def test_inject_flare_requires_expected_n(window, exposure_model, t0, tf, rng_manager, generated_flare):
    """The Poisson thinning needs ``expected_n / n_total``; a bare-constructor
    sample with coordinates but no expected_n must raise."""
    s = EventSample(n_sample=200, n_total=20_000, t0=t0, tf=tf, rng=rng_manager.get("sample"))
    s.ra = np.zeros(s.n_sample)
    s.dec = np.zeros(s.n_sample)
    assert s.expected_n is None
    with pytest.raises(ValueError):
        s.inject_flare(generated_flare, mode="overdensity")


def test_inject_flare_mean_n_removed_matches_poisson(window, exposure_model, t0, tf, rng_manager):
    """Average over many trials: E[n_removed] = p_in_window * n_flare."""
    import astropy.units as u

    n_total = 100_000
    n_flare = 200
    expected_n = window.expected_n_in_window(n_total, exposure_model)
    p_in_window = expected_n / n_total
    mu_removed_expected = p_in_window * n_flare

    n_trials = 500
    n_removed_samples = np.empty(n_trials, dtype=int)
    sample_rng = rng_manager.get("sample")
    flare_rng = rng_manager.get("flare")

    for i in range(n_trials):
        s = EventSample.in_window(
            window=window, n_total=n_total, exposure_model=exposure_model,
            t0=t0, tf=tf, rng=sample_rng,
        )
        n_before = s.n_sample
        flare = Flare(
            n_flare=n_flare, duration=1.0 * u.day,
            t0=t0, tf=tf, centre=window.centre,
            exposure_model=exposure_model, rng=flare_rng,
        )
        flare.generate_in_window(window=window, sigma=2.0)
        s.inject_flare(flare, mode="overdensity")
        n_removed_samples[i] = n_before - (s.n_sample - n_flare)

    mean_n_removed = float(np.mean(n_removed_samples))
    # Tolerance ~ 3 * std(Poisson) / sqrt(n_trials)
    tol = 3.0 * np.sqrt(mu_removed_expected) / np.sqrt(n_trials)
    assert abs(mean_n_removed - mu_removed_expected) < max(tol, 0.5)


def test_inject_flare_mask_count(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="overdensity")
    assert int(np.count_nonzero(s.flare_mask)) == generated_flare.n_flare


def test_inject_flare_sets_has_flare(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="overdensity")
    assert s.has_flare


def test_inject_flare_sets_flare_type(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="overdensity")
    assert s.flare_type == generated_flare.flare_type


def test_inject_flare_overwrites_flare_slots(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="overdensity")
    np.testing.assert_array_equal(
        np.sort(s.ra[s.flare_mask]),
        np.sort(generated_flare.ra),
    )


def test_inject_flare_double_injection_raises(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="overdensity")
    with pytest.raises(RuntimeError):
        s.inject_flare(generated_flare, mode="overdensity")


def test_inject_flare_not_a_flare_raises(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=200, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    with pytest.raises(TypeError):
        s.inject_flare("not a flare", mode="overdensity")


def test_inject_flare_no_coordinates_raises(t0, tf, rng, generated_flare):
    s = EventSample(n_sample=200, n_total=200, t0=t0, tf=tf, rng=rng)
    with pytest.raises(ValueError):
        s.inject_flare(generated_flare, mode="overdensity")


# -------------------------------------------------------------------------
# inject_flare — mode dispatch
# -------------------------------------------------------------------------


def test_inject_flare_mode_is_required(window, exposure_model, t0, tf, rng, generated_flare):
    """Omitting the keyword-only ``mode`` argument must raise."""
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    with pytest.raises(TypeError):
        s.inject_flare(generated_flare)


def test_inject_flare_invalid_mode_raises(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    with pytest.raises(ValueError):
        s.inject_flare(generated_flare, mode="bogus")


# -------------------------------------------------------------------------
# inject_flare — no_overdensity mode
# -------------------------------------------------------------------------


def test_inject_flare_no_overdensity_preserves_n_sample(
    window, exposure_model, t0, tf, rng, generated_flare,
):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    n_before = s.n_sample
    s.inject_flare(generated_flare, mode="no_overdensity")
    assert s.n_sample == n_before


def test_inject_flare_no_overdensity_array_lengths_consistent(
    window, exposure_model, t0, tf, rng, generated_flare,
):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="no_overdensity")
    assert s.n_sample == len(s.ra) == len(s.dec) == len(s.exposure) == len(s.flare_mask)


def test_inject_flare_no_overdensity_mask_count(
    window, exposure_model, t0, tf, rng, generated_flare,
):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="no_overdensity")
    assert int(np.count_nonzero(s.flare_mask)) == generated_flare.n_flare


def test_inject_flare_no_overdensity_appends_at_tail(
    window, exposure_model, t0, tf, rng, generated_flare,
):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare, mode="no_overdensity")
    tail = slice(-generated_flare.n_flare, None)
    np.testing.assert_array_equal(s.ra[tail], generated_flare.ra)
    np.testing.assert_array_equal(s.dec[tail], generated_flare.dec)
    np.testing.assert_array_equal(s.exposure[tail], generated_flare.exposure)


def test_inject_flare_no_overdensity_does_not_require_expected_n(
    window, exposure_model, t0, tf, rng_manager, generated_flare,
):
    """``no_overdensity`` does not consult ``expected_n`` / ``n_total``;
    a bare-constructor sample with coordinates but no expected_n must
    accept the injection."""
    s = EventSample(
        n_sample=200, n_total=20_000, t0=t0, tf=tf, rng=rng_manager.get("sample"),
    )
    # Manually populate coordinates / exposure so we bypass the in_window
    # factory (which would set expected_n).
    s.ra = np.zeros(s.n_sample)
    s.dec = np.zeros(s.n_sample)
    s.exposure = np.zeros(s.n_sample)
    assert s.expected_n is None
    s.inject_flare(generated_flare, mode="no_overdensity")  # must not raise
    assert s.n_sample == 200


def test_inject_flare_no_overdensity_n_flare_exceeds_n_sample_raises(
    window, exposure_model, t0, tf, rng_manager,
):
    import astropy.units as u
    s = EventSample.full_sky(n_total=5, t0=t0, tf=tf, rng=rng_manager.get("sample"))
    big_flare = Flare(
        n_flare=10, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng_manager.get("flare"),
    )
    big_flare.generate_in_window(window=window, sigma=2.0)
    with pytest.raises(ValueError):
        s.inject_flare(big_flare, mode="no_overdensity")


# -------------------------------------------------------------------------
# Statistical validation under H0 (background-only)
# -------------------------------------------------------------------------


@pytest.mark.parametrize("dec_centre", [-50.0, -30.0, -10.0])
def test_in_window_realised_n_matches_expected_across_declinations(
    dec_centre, exposure_model, t0, tf, seed,
):
    """Realised ``n_sample`` averaged over trials must match
    ``SkyWindow.expected_n_in_window`` to Poisson tolerance.

    Closes the loop between the analytic exposure-weighted expectation and
    the actual Poisson draw performed by ``EventSample.in_window``.  The
    existing ``test_in_window_expected_n_matches_window_formula`` only
    checks that the stored ``expected_n`` attribute equals the formula, so
    a regression that broke the sampler (e.g. forgetting ``omega``,
    applying it twice, or using the wrong declination) would slip through.
    Sweeping the window centre across declinations exercises the range
    over which ``omega(delta_centre)`` actually varies.
    """
    n_trials = 200
    n_total = 20_000
    window = SkyWindow(centre=[180.0, dec_centre], radius=15.0)
    expected = window.expected_n_in_window(n_total, exposure_model)

    rng = np.random.default_rng(seed)
    counts = np.empty(n_trials, dtype=int)
    for i in range(n_trials):
        s = EventSample.in_window(
            window=window, n_total=n_total, exposure_model=exposure_model,
            t0=t0, tf=tf, rng=rng,
        )
        counts[i] = s.n_sample

    # SE of the mean for N Poisson(expected) trials is sqrt(expected/N).
    # 4-sigma is a loose regression guard.
    se = np.sqrt(expected / n_trials)
    assert abs(counts.mean() - expected) < 4.0 * se


def test_in_window_pvalue_uniform_under_h0(
    window, exposure_model, t0, tf, seed,
):
    """Background-only realisations must yield p-values uniform on [0, 1].

    For each trial: draw a per-window background sample, assign
    directional exposure, compute the Lambda statistic, convert to a
    p-value through the marginal Lambda survival function evaluated at
    ``mu = expected_n``.  Under H0 the resulting p-values should be
    ``Uniform(0, 1)``; a one-sample KS test guards against bias in any
    stage of the pipeline (sampler, exposure assignment, or reference
    distribution).
    """
    n_trials = 200
    n_total = 20_000

    rng = np.random.default_rng(seed)
    lambdas = np.empty(n_trials)
    for i in range(n_trials):
        s = EventSample.in_window(
            window=window, n_total=n_total, exposure_model=exposure_model,
            t0=t0, tf=tf, rng=rng,
        )
        s.assign_directional_exposure(window, exposure_model)
        lambdas[i] = lambda_estimator(s)

    pvals = lambda_marginal_sf(lambdas, mu=float(s.expected_n))
    ks_p = float(scp.kstest(pvals, "uniform").pvalue)
    assert ks_p > 0.01
