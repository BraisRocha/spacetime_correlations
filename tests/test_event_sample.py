"""Tests for ``EventSample`` — bare constructor, factories, and contracts."""

import numpy as np
import pytest
from astropy.time import Time

from spacetimecorr import EventSample, Flare, SkyWindow


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


def test_inject_flare_preserves_n_sample(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    n_before = s.n_sample
    s.inject_flare(generated_flare)
    assert s.n_sample == n_before


def test_inject_flare_mask_count(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare)
    assert int(np.count_nonzero(s.flare_mask)) == generated_flare.n_flare


def test_inject_flare_sets_has_flare(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare)
    assert s.has_flare


def test_inject_flare_sets_flare_type(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare)
    assert s.flare_type == generated_flare.flare_type


def test_inject_flare_overwrites_flare_slots(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare)
    np.testing.assert_array_equal(
        np.sort(s.ra[s.flare_mask]),
        np.sort(generated_flare.ra),
    )


def test_inject_flare_double_injection_raises(window, exposure_model, t0, tf, rng, generated_flare):
    s = EventSample.in_window(
        window=window, n_total=20_000, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    s.inject_flare(generated_flare)
    with pytest.raises(RuntimeError):
        s.inject_flare(generated_flare)


def test_inject_flare_too_large_raises(window, exposure_model, t0, tf, rng_manager):
    import astropy.units as u
    s = EventSample.full_sky(n_total=5, t0=t0, tf=tf, rng=rng_manager.get("sample"))
    big_flare = Flare(
        n_flare=10, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng_manager.get("flare"),
    )
    big_flare.generate_in_window(window=window, sigma=2.0)
    with pytest.raises(ValueError):
        s.inject_flare(big_flare)


def test_inject_flare_not_a_flare_raises(window, exposure_model, t0, tf, rng):
    s = EventSample.in_window(
        window=window, n_total=200, exposure_model=exposure_model,
        t0=t0, tf=tf, rng=rng,
    )
    with pytest.raises(TypeError):
        s.inject_flare("not a flare")


def test_inject_flare_no_coordinates_raises(t0, tf, rng, generated_flare):
    s = EventSample(n_sample=200, n_total=200, t0=t0, tf=tf, rng=rng)
    with pytest.raises(ValueError):
        s.inject_flare(generated_flare)
