"""Geometry, validation, and analytic-identity tests for ``SkyWindow``."""

import numpy as np
import pytest

from spacetimecorr import SkyWindow


# -------------------------------------------------------------------------
# Construction / validation
# -------------------------------------------------------------------------


def test_construction_with_valid_inputs():
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    assert win.radius == pytest.approx(15.0)


def test_centre_is_stored_as_array():
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    np.testing.assert_array_almost_equal(np.asarray(win.centre), [180.0, -30.0])


@pytest.mark.parametrize("bad_ra", [-1.0, 360.0, 400.0])
def test_ra_out_of_range_raises(bad_ra):
    with pytest.raises(ValueError):
        SkyWindow(centre=[bad_ra, 0.0], radius=10.0)


@pytest.mark.parametrize("bad_dec", [-90.5, 91.0])
def test_dec_out_of_range_raises(bad_dec):
    with pytest.raises(ValueError):
        SkyWindow(centre=[0.0, bad_dec], radius=10.0)


@pytest.mark.parametrize("bad_radius", [0.0, -1.0, 180.5])
def test_radius_out_of_range_raises(bad_radius):
    with pytest.raises(ValueError):
        SkyWindow(centre=[0.0, 0.0], radius=bad_radius)


def test_centre_shape_mismatch_raises():
    with pytest.raises(TypeError):
        SkyWindow(centre=[0.0, 0.0, 0.0], radius=10.0)


# -------------------------------------------------------------------------
# Sky fraction
# -------------------------------------------------------------------------


@pytest.mark.parametrize("radius_deg", [0.5, 1.0, 10.0, 45.0, 90.0, 179.0])
def test_sky_fraction_matches_spherical_cap_formula(radius_deg):
    win = SkyWindow(centre=[0.0, 0.0], radius=radius_deg)
    expected = 0.5 * (1.0 - np.cos(np.deg2rad(radius_deg)))
    assert win.sky_fraction == pytest.approx(expected)


def test_full_sphere_sky_fraction_is_one():
    win = SkyWindow(centre=[0.0, 0.0], radius=180.0)
    assert win.sky_fraction == pytest.approx(1.0)


@pytest.mark.parametrize("radius_deg", [1.0, 30.0, 90.0])
def test_sky_fraction_in_unit_interval(radius_deg):
    win = SkyWindow(centre=[180.0, -30.0], radius=radius_deg)
    assert 0.0 < win.sky_fraction <= 1.0


# -------------------------------------------------------------------------
# Contains
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "centre",
    [[0.0, 0.0], [180.0, -30.0], [359.9, 89.0], [10.0, -89.0]],
)
def test_centre_is_contained(centre):
    win = SkyWindow(centre=centre, radius=5.0)
    inside = win.contains(np.array([centre[0]]), np.array([centre[1]]))
    assert bool(inside[0])


def test_antipode_not_contained_for_small_radius():
    win = SkyWindow(centre=[0.0, 0.0], radius=10.0)
    inside = win.contains(np.array([180.0]), np.array([0.0]))
    assert not bool(inside[0])


def test_contains_returns_bool_array_of_same_shape():
    win = SkyWindow(centre=[0.0, 0.0], radius=10.0)
    ra = np.array([0.0, 5.0, 30.0])
    dec = np.array([0.0, 0.0, 0.0])
    mask = win.contains(ra, dec)
    assert mask.shape == ra.shape
    assert mask.dtype == bool


def test_contains_shape_mismatch_raises():
    win = SkyWindow(centre=[0.0, 0.0], radius=10.0)
    with pytest.raises(ValueError):
        win.contains(np.array([0.0, 1.0]), np.array([0.0]))


# -------------------------------------------------------------------------
# sample_uniform
# -------------------------------------------------------------------------


def test_sample_uniform_returns_correct_count(rng):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    ra, dec = win.sample_uniform(100, rng)
    assert ra.shape == (100,)


def test_sample_uniform_dec_shape_matches(rng):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    ra, dec = win.sample_uniform(100, rng)
    assert dec.shape == ra.shape


def test_sample_uniform_zero_returns_empty(rng):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    ra, dec = win.sample_uniform(0, rng)
    assert ra.size == 0


def test_sample_uniform_all_inside_window(rng):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    ra, dec = win.sample_uniform(2000, rng)
    assert np.all(win.contains(ra, dec))


def test_sample_uniform_ra_in_valid_range(rng):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    ra, _ = win.sample_uniform(1000, rng)
    assert np.all((ra >= 0.0) & (ra < 360.0))


def test_sample_uniform_dec_in_valid_range(rng):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    _, dec = win.sample_uniform(1000, rng)
    assert np.all((dec >= -90.0) & (dec <= 90.0))


def test_sample_uniform_finite(rng):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    ra, dec = win.sample_uniform(1000, rng)
    assert np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))


# -------------------------------------------------------------------------
# expected_n_in_window
# -------------------------------------------------------------------------


def test_expected_n_in_window_unweighted_matches_formula():
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    result = win.expected_n_in_window(n_events=10_000)
    assert result == pytest.approx(10_000 * win.sky_fraction)


def test_expected_n_in_window_with_exposure_model_includes_omega(exposure_model):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    omega = exposure_model.relative_exposure(win.centre)
    result = win.expected_n_in_window(n_events=10_000, exposure_model=exposure_model)
    assert result == pytest.approx(10_000 * win.sky_fraction * omega)


def test_expected_n_in_window_scales_linearly_in_n_events():
    win = SkyWindow(centre=[0.0, 0.0], radius=15.0)
    a = win.expected_n_in_window(n_events=100)
    b = win.expected_n_in_window(n_events=300)
    assert b == pytest.approx(3.0 * a)
