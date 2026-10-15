"""Geometry, validation, and analytic-identity tests for ``SkyWindow``."""

import numpy as np
import pytest

from spacetimecorr import SkyWindow, HealpixWindow, SkyGrid, HealpixGrid


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


def test_expected_n_in_window_with_exposure_model_uses_exposure_share(exposure_model):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    # The weight is the window's share of the observatory's total exposure.
    share = exposure_model.relative_window_exposure(win)
    result = win.expected_n_in_window(n_events=10_000, exposure_model=exposure_model)
    assert result == pytest.approx(10_000 * share)


def test_expected_n_in_window_matches_exposure_ratio(exposure_model):
    win = SkyWindow(centre=[180.0, -30.0], radius=15.0)
    ratio = (
        exposure_model.cumul_exposure(exposure_model.tf, win)
        / exposure_model.total_exposure()
    )
    result = win.expected_n_in_window(n_events=10_000, exposure_model=exposure_model)
    assert result == pytest.approx(10_000 * ratio, rel=1e-12)


def test_expected_n_in_window_scales_linearly_in_n_events():
    win = SkyWindow(centre=[0.0, 0.0], radius=15.0)
    a = win.expected_n_in_window(n_events=100)
    b = win.expected_n_in_window(n_events=300)
    assert b == pytest.approx(3.0 * a)


# -------------------------------------------------------------------------
# SkyGrid.expected_n_in_window
# -------------------------------------------------------------------------


GRID_CENTRES = np.array([[180.0, -30.0], [10.0, -60.0], [200.0, 50.0]])


def test_grid_expected_n_unweighted_matches_per_window():
    grid = SkyGrid(centres=GRID_CENTRES, radii=5.0)
    expected = [
        SkyWindow(centre=c, radius=5.0).expected_n_in_window(10_000)
        for c in GRID_CENTRES
    ]
    assert grid.expected_n_in_window(10_000) == pytest.approx(expected)


def test_grid_expected_n_weighted_matches_per_window(exposure_model):
    grid = SkyGrid(centres=GRID_CENTRES, radii=5.0)
    expected = [
        SkyWindow(centre=c, radius=5.0).expected_n_in_window(10_000, exposure_model)
        for c in GRID_CENTRES
    ]
    result = grid.expected_n_in_window(10_000, exposure_model)
    assert result == pytest.approx(expected)


def test_grid_expected_n_honours_per_window_radii(exposure_model):
    radii = np.array([1.0, 5.0, 10.0])
    grid = SkyGrid(centres=GRID_CENTRES, radii=radii)
    expected = [
        SkyWindow(centre=c, radius=r).expected_n_in_window(10_000, exposure_model)
        for c, r in zip(GRID_CENTRES, radii)
    ]
    assert grid.expected_n_in_window(10_000, exposure_model) == pytest.approx(expected)


def test_grid_expected_n_zero_for_never_visible_window(exposure_model):
    # dec = +50 never enters the field of view from Auger with theta_max = 60.
    grid = SkyGrid(centres=GRID_CENTRES, radii=5.0)
    result = grid.expected_n_in_window(10_000, exposure_model)
    assert result[2] == 0.0
    assert np.all(result[:2] > 0.0)


# -------------------------------------------------------------------------
# HealpixWindow
# -------------------------------------------------------------------------


def test_healpix_window_centre_and_solid_angle():
    hp = pytest.importorskip("healpy")
    win = HealpixWindow(nside=32, ipix=5000)
    ra, dec = hp.pix2ang(32, 5000, lonlat=True)
    assert win.centre == pytest.approx([ra, dec])
    assert win.solid_angle == pytest.approx(4.0 * np.pi / (12 * 32**2))


@pytest.mark.parametrize("nside, ipix", [(0, 0), (32, -1), (32, 12 * 32**2)])
def test_healpix_window_invalid_inputs_raise(nside, ipix):
    with pytest.raises(ValueError):
        HealpixWindow(nside=nside, ipix=ipix)


def test_healpix_window_equality_by_nside_and_ipix():
    assert HealpixWindow(32, 7) == HealpixWindow(32, 7)
    assert HealpixWindow(32, 7) != HealpixWindow(32, 8)


def test_healpix_window_contains_its_centre():
    win = HealpixWindow(nside=32, ipix=5000)
    assert win.contains(np.array([win.centre[0]]), np.array([win.centre[1]]))[0]


def test_healpix_windows_partition_the_sky(rng):
    """Every direction lies in exactly one pixel."""
    nside = 2
    ra = rng.uniform(0.0, 360.0, 5000)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 5000)))
    n_containing = sum(
        HealpixWindow(nside, i).contains(ra, dec).astype(int)
        for i in range(12 * nside**2)
    )
    assert np.all(n_containing == 1)


def test_healpix_window_contains_shape_mismatch_raises():
    with pytest.raises(ValueError):
        HealpixWindow(32, 0).contains(np.zeros(3), np.zeros(4))


def test_healpix_expected_n_unweighted_sums_to_n_events():
    nside = 4
    total = sum(
        HealpixWindow(nside, i).expected_n_in_window(10_000)
        for i in range(12 * nside**2)
    )
    assert total == pytest.approx(10_000)


def test_healpix_expected_n_with_exposure_model_uses_exposure_share(exposure_model):
    win = HealpixWindow(nside=32, ipix=5000)
    share = exposure_model.relative_window_exposure(win)
    result = win.expected_n_in_window(10_000, exposure_model)
    assert result == pytest.approx(10_000 * share)


def test_healpix_expected_n_weighted_sums_to_n_events(exposure_model):
    """Pixels do not overlap, so their exposure shares add up to the whole
    sky, up to the error of evaluating the exposure at each pixel centre."""
    nside = 8
    total = sum(
        HealpixWindow(nside, i).expected_n_in_window(1.0, exposure_model)
        for i in range(12 * nside**2)
    )
    assert total == pytest.approx(1.0, abs=2e-3)


# -------------------------------------------------------------------------
# HealpixGrid
# -------------------------------------------------------------------------


def test_healpix_grid_default_nside_is_32(exposure_model):
    assert HealpixGrid(exposure_model).nside == 32


def test_healpix_grid_invalid_nside_raises(exposure_model):
    with pytest.raises(ValueError):
        HealpixGrid(exposure_model, nside=0)


def test_healpix_grid_keeps_exactly_the_pixels_with_centre_in_fov(exposure_model):
    hp = pytest.importorskip("healpy")
    nside = 16
    grid = HealpixGrid(exposure_model, nside=nside)
    _, dec = hp.pix2ang(nside, np.arange(12 * nside**2), lonlat=True)
    dropped = np.setdiff1d(np.arange(12 * nside**2), grid.pixels)
    assert np.all(exposure_model.spatial_exposure(dec[grid.pixels]) > 0.0)
    assert np.all(exposure_model.spatial_exposure(dec[dropped]) == 0.0)
    # Auger with theta_max = 60: the FoV ends at dec = -35.15 + 60.
    assert grid.centres[:, 1].max() < -35.15 + 60.0


def test_healpix_grid_indexing_builds_matching_window(exposure_model):
    grid = HealpixGrid(exposure_model, nside=16)
    window = grid[10]
    assert isinstance(window, HealpixWindow)
    assert window.ipix == grid.pixels[10]
    assert window.centre == pytest.approx(grid.centres[10])
    assert window.solid_angle == pytest.approx(grid.solid_angle)


def test_healpix_grid_iterates_over_all_windows(exposure_model):
    grid = HealpixGrid(exposure_model, nside=8)
    assert [w.ipix for w in grid] == list(grid.pixels)


def test_healpix_grid_pixel_of_agrees_with_window_contains(exposure_model, rng):
    grid = HealpixGrid(exposure_model, nside=8)
    ra = rng.uniform(0.0, 360.0, 2000)
    dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 2000)))
    pix = grid.pixel_of(ra, dec)
    for window in grid:
        assert np.array_equal(pix == window.ipix, window.contains(ra, dec))


def test_healpix_grid_expected_n_matches_per_window(exposure_model):
    grid = HealpixGrid(exposure_model, nside=8)
    expected = [w.expected_n_in_window(10_000, exposure_model) for w in grid]
    assert grid.expected_n_in_window(10_000, exposure_model) == pytest.approx(expected)


def test_healpix_grid_expected_n_sums_to_n_events(exposure_model):
    """Dropped pixels have zero exposure at their centre, so the grid holds
    the whole-sky total, up to the error of the centre approximation."""
    grid = HealpixGrid(exposure_model, nside=8)
    total = grid.expected_n_in_window(1.0, exposure_model).sum()
    assert total == pytest.approx(1.0, abs=2e-3)
