"""Tests for ``Flare`` — construction, validation, and ``generate_in_window``."""

import numpy as np
import pytest
import astropy.units as u
from astropy.time import Time

from spacetimecorr import Flare, SkyWindow


# -------------------------------------------------------------------------
# Construction / validation
# -------------------------------------------------------------------------


def test_construction_with_valid_inputs(window, exposure_model, t0, tf, rng):
    flare = Flare(
        n_flare=10, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng,
    )
    assert flare.n_flare == 10


def test_zero_n_flare_raises(window, exposure_model, t0, tf, rng):
    with pytest.raises(ValueError):
        Flare(
            n_flare=0, duration=1.0 * u.day,
            t0=t0, tf=tf, centre=window.centre,
            exposure_model=exposure_model, rng=rng,
        )


def test_negative_n_flare_raises(window, exposure_model, t0, tf, rng):
    with pytest.raises(ValueError):
        Flare(
            n_flare=-5, duration=1.0 * u.day,
            t0=t0, tf=tf, centre=window.centre,
            exposure_model=exposure_model, rng=rng,
        )


def test_non_integer_n_flare_raises(window, exposure_model, t0, tf, rng):
    with pytest.raises(TypeError):
        Flare(
            n_flare=2.5, duration=1.0 * u.day,
            t0=t0, tf=tf, centre=window.centre,
            exposure_model=exposure_model, rng=rng,
        )


def test_non_quantity_duration_raises(window, exposure_model, t0, tf, rng):
    with pytest.raises(TypeError):
        Flare(
            n_flare=10, duration=86400.0,  # raw float
            t0=t0, tf=tf, centre=window.centre,
            exposure_model=exposure_model, rng=rng,
        )


def test_non_time_duration_units_raises(window, exposure_model, t0, tf, rng):
    with pytest.raises(ValueError):
        Flare(
            n_flare=10, duration=1.0 * u.m,
            t0=t0, tf=tf, centre=window.centre,
            exposure_model=exposure_model, rng=rng,
        )


def test_zero_duration_raises(window, exposure_model, t0, tf, rng):
    with pytest.raises(ValueError):
        Flare(
            n_flare=10, duration=0.0 * u.day,
            t0=t0, tf=tf, centre=window.centre,
            exposure_model=exposure_model, rng=rng,
        )


def test_duration_longer_than_observation_raises(window, exposure_model, rng):
    t0 = Time("2020-01-01")
    tf = Time("2020-01-02")
    with pytest.raises(ValueError):
        Flare(
            n_flare=10, duration=10.0 * u.day,
            t0=t0, tf=tf, centre=window.centre,
            exposure_model=exposure_model, rng=rng,
        )


def test_tf_before_t0_raises(window, exposure_model, rng):
    with pytest.raises(ValueError):
        Flare(
            n_flare=10, duration=1.0 * u.day,
            t0=Time("2021-01-01"), tf=Time("2020-01-01"),
            centre=window.centre, exposure_model=exposure_model, rng=rng,
        )


def test_invalid_centre_shape_raises(exposure_model, t0, tf, rng):
    with pytest.raises(ValueError):
        Flare(
            n_flare=10, duration=1.0 * u.day,
            t0=t0, tf=tf, centre=np.array([0.0, 0.0, 0.0]),
            exposure_model=exposure_model, rng=rng,
        )


@pytest.mark.parametrize("bad_ra", [-1.0, 360.0])
def test_invalid_centre_ra_raises(exposure_model, t0, tf, rng, bad_ra):
    with pytest.raises(ValueError):
        Flare(
            n_flare=10, duration=1.0 * u.day,
            t0=t0, tf=tf, centre=np.array([bad_ra, 0.0]),
            exposure_model=exposure_model, rng=rng,
        )


def test_invalid_exposure_model_raises(window, t0, tf, rng):
    with pytest.raises(TypeError):
        Flare(
            n_flare=10, duration=1.0 * u.day,
            t0=t0, tf=tf, centre=window.centre,
            exposure_model="not-a-model", rng=rng,
        )


def test_invalid_rng_raises(window, exposure_model, t0, tf):
    with pytest.raises(TypeError):
        Flare(
            n_flare=10, duration=1.0 * u.day,
            t0=t0, tf=tf, centre=window.centre,
            exposure_model=exposure_model, rng=42,
        )


def test_has_coordinates_false_before_generate(window, exposure_model, t0, tf, rng):
    flare = Flare(
        n_flare=10, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng,
    )
    assert not flare.has_coordinates


def test_flare_type_undefined_before_generate(window, exposure_model, t0, tf, rng):
    flare = Flare(
        n_flare=10, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng,
    )
    assert flare.flare_type == "undefined_flare"


# -------------------------------------------------------------------------
# generate_in_window
# -------------------------------------------------------------------------


@pytest.fixture
def generated_flare(window, exposure_model, t0, tf, rng_manager):
    flare = Flare(
        n_flare=30, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng_manager.get("flare"),
    )
    flare.generate_in_window(window=window, sigma=2.0)
    return flare


def test_generate_ra_length(generated_flare):
    assert len(generated_flare.ra) == generated_flare.n_flare


def test_generate_dec_length(generated_flare):
    assert len(generated_flare.dec) == generated_flare.n_flare


def test_generate_time_length(generated_flare):
    assert len(generated_flare.time) == generated_flare.n_flare


def test_generate_exposure_length(generated_flare):
    assert len(generated_flare.exposure) == generated_flare.n_flare


def test_generate_all_inside_window(generated_flare, window):
    assert np.all(window.contains(generated_flare.ra, generated_flare.dec))


def test_generate_times_inside_observation(generated_flare):
    assert np.all(generated_flare.time >= generated_flare.t0)
    assert np.all(generated_flare.time <= generated_flare.tf)


def test_generate_time_span_lte_duration(generated_flare):
    span = (generated_flare.time.max() - generated_flare.time.min()).to_value("sec")
    assert span <= generated_flare.duration + 1e-6


def test_generate_has_coordinates_true(generated_flare):
    assert generated_flare.has_coordinates


def test_generate_sets_spatial_profile(generated_flare):
    assert generated_flare.spatial_profile == "gaussian_spherical"


def test_generate_sets_time_profile(generated_flare):
    assert generated_flare.time_profile == "uniform_thinned"


def test_generate_flare_type_format(generated_flare):
    assert generated_flare.flare_type == "gaussian_spherical-uniform_thinned"


def test_generate_exposure_finite(generated_flare):
    assert np.all(np.isfinite(generated_flare.exposure))


def test_generate_exposure_nonneg(generated_flare):
    assert np.all(generated_flare.exposure >= 0.0)


# -------------------------------------------------------------------------
# generate_in_window validation
# -------------------------------------------------------------------------


def test_generate_in_window_invalid_window_raises(window, exposure_model, t0, tf, rng):
    flare = Flare(
        n_flare=10, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng,
    )
    with pytest.raises(TypeError):
        flare.generate_in_window(window="not-a-window", sigma=1.0)


def test_generate_in_window_nonpositive_sigma_raises(window, exposure_model, t0, tf, rng):
    flare = Flare(
        n_flare=10, duration=1.0 * u.day,
        t0=t0, tf=tf, centre=window.centre,
        exposure_model=exposure_model, rng=rng,
    )
    with pytest.raises(ValueError):
        flare.generate_in_window(window=window, sigma=0.0)
