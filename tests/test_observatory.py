"""Input-validation and basic-property tests for ``Observatory``."""

import numpy as np
import pytest
from astropy.coordinates import EarthLocation

from spacetimecorr import Observatory


def test_construction_with_valid_inputs():
    obs = Observatory(latitude=-35.15, longitude=-69.15, altitude=1425.0)
    assert obs.latitude == pytest.approx(-35.15)


def test_construction_stores_longitude():
    obs = Observatory(latitude=0.0, longitude=10.0, altitude=0.0)
    assert obs.longitude == pytest.approx(10.0)


def test_construction_stores_altitude():
    obs = Observatory(latitude=0.0, longitude=0.0, altitude=2500.0)
    assert obs.altitude == pytest.approx(2500.0)


def test_construction_caches_earth_location():
    obs = Observatory(latitude=-35.15, longitude=-69.15, altitude=1425.0)
    assert isinstance(obs.location, EarthLocation)


@pytest.mark.parametrize("bad_lat", [-90.5, 91.0, 180.0, -200.0])
def test_latitude_out_of_range_raises(bad_lat):
    with pytest.raises(ValueError):
        Observatory(latitude=bad_lat, longitude=0.0, altitude=0.0)


@pytest.mark.parametrize("bad_lon", [-181.0, 181.0, 360.0])
def test_longitude_out_of_range_raises(bad_lon):
    with pytest.raises(ValueError):
        Observatory(latitude=0.0, longitude=bad_lon, altitude=0.0)


def test_negative_altitude_raises():
    with pytest.raises(ValueError):
        Observatory(latitude=0.0, longitude=0.0, altitude=-1.0)


@pytest.mark.parametrize("bad_lat", [True, "0", None, np.array([0.0])])
def test_non_numeric_latitude_raises(bad_lat):
    with pytest.raises(TypeError):
        Observatory(latitude=bad_lat, longitude=0.0, altitude=0.0)


@pytest.mark.parametrize("bad_alt", [True, "0", None])
def test_non_numeric_altitude_raises(bad_alt):
    with pytest.raises(TypeError):
        Observatory(latitude=0.0, longitude=0.0, altitude=bad_alt)


def test_latitude_boundary_minus90_allowed():
    obs = Observatory(latitude=-90.0, longitude=0.0, altitude=0.0)
    assert obs.latitude == -90.0


def test_latitude_boundary_plus90_allowed():
    obs = Observatory(latitude=90.0, longitude=0.0, altitude=0.0)
    assert obs.latitude == 90.0


def test_zero_altitude_allowed():
    obs = Observatory(latitude=0.0, longitude=0.0, altitude=0.0)
    assert obs.altitude == 0.0
