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
# Detection probability
# -------------------------------------------------------------------------


def test_detection_probability_in_zero_or_above_cut(exposure_model, t0):
    centre = np.array([180.0, -30.0])
    a = exposure_model.detection_probability(t0, centre)
    cos_cut = np.cos(np.deg2rad(exposure_model.theta_max_deg))
    assert (a == 0.0) or (a >= cos_cut)


def test_detection_probability_array_shape(exposure_model, t0, tf):
    t = t0 + TimeDelta(np.linspace(0, (tf - t0).to_value(u.s), 50), format="sec")
    a = exposure_model.detection_probability(t, np.array([180.0, -30.0]))
    assert a.shape == (50,)


def test_detection_probability_out_of_interval_raises(exposure_model, t0):
    with pytest.raises(ValueError):
        exposure_model.detection_probability(t0 - TimeDelta(1.0, format="sec"),
                                                np.array([0.0, 0.0]))


# -------------------------------------------------------------------------
# Cumulative directional exposure
# -------------------------------------------------------------------------


def test_norm_cumul_exposure_zero_at_t0(exposure_model, t0):
    eps = exposure_model.norm_cumul_exposure(t0, np.array([180.0, -30.0]))
    assert float(eps) == pytest.approx(0.0)


def test_norm_cumul_exposure_max_at_tf(exposure_model, tf):
    centre = np.array([180.0, -30.0])
    eps = exposure_model.norm_cumul_exposure(tf, centre)
    max_eps = exposure_model.max_norm_cumul_exposure(centre)
    assert float(eps) == pytest.approx(max_eps, rel=1e-9)


def test_norm_cumul_exposure_monotonic(exposure_model, t0, tf):
    centre = np.array([180.0, -30.0])
    total_sec = (tf - t0).to_value(u.s)
    grid = t0 + TimeDelta(np.linspace(0.0, total_sec, 200), format="sec")
    eps = exposure_model.norm_cumul_exposure(grid, centre)
    assert np.all(np.diff(np.asarray(eps)) >= -1e-9)


def test_norm_cumul_exposure_nonneg(exposure_model, t0, tf):
    centre = np.array([180.0, -30.0])
    total_sec = (tf - t0).to_value(u.s)
    grid = t0 + TimeDelta(np.linspace(0.0, total_sec, 100), format="sec")
    eps = exposure_model.norm_cumul_exposure(grid, centre)
    assert np.all(np.asarray(eps) >= -1e-12)


def test_norm_cumul_exposure_zero_for_always_invisible(exposure_model):
    # From Auger (lat ~ -35°) with theta_max = 60°, the north celestial pole
    # is always below the horizon → cumulative exposure must stay at 0.
    invisible = np.array([0.0, 80.0])  # dec = +80°, never reaches zenith from Auger
    eps = exposure_model.max_norm_cumul_exposure(invisible)
    assert eps == pytest.approx(0.0)


# -------------------------------------------------------------------------
# detect_times
# -------------------------------------------------------------------------


def test_detect_times_efficiency_zero_rejects_all(exposure_model, t0, tf):
    times = t0 + TimeDelta(
        np.linspace(0.0, (tf - t0).to_value(u.s), 200), format="sec"
    )
    _, mask = exposure_model.detect_times(
        times,
        np.array([180.0, -30.0]),
        efficiency=lambda t: np.zeros(len(t)),
        return_mask=True,
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
    assert mask.dtype == bool
    assert mask.shape == (50,) and len(accepted) == mask.sum()


# -------------------------------------------------------------------------
# spatial_exposure / detect_directions
# -------------------------------------------------------------------------


@pytest.mark.parametrize("dec", [-90.0, -60.0, -35.15, 0.0, 20.0])
def test_spatial_exposure_matches_max_norm_cumul_exposure(exposure_model, dec):
    """Same quantity, computed per direction through the time integral.
    They differ only by the leftover partial sidereal day."""
    assert float(exposure_model.spatial_exposure(dec)) == pytest.approx(
        exposure_model.max_norm_cumul_exposure(np.array([180.0, dec])),
        rel=1e-3,
    )


def test_spatial_exposure_sky_integral(exposure_model):
    """∫_sky spatial_exposure dOmega = pi sin²(theta_max)."""
    dec = np.linspace(-90.0, 90.0, 200_001)
    integrand = exposure_model.spatial_exposure(dec) * np.cos(np.deg2rad(dec))
    integral = 2.0 * np.pi * np.trapezoid(integrand, np.deg2rad(dec))
    assert integral == pytest.approx(np.pi * np.sin(np.deg2rad(60.0)) ** 2, rel=1e-6)


def test_spatial_exposure_zero_outside_fov(exposure_model):
    assert float(exposure_model.spatial_exposure(40.0)) == 0.0


def test_spatial_exposure_invalid_dec_raises(exposure_model):
    with pytest.raises(ValueError):
        exposure_model.spatial_exposure(91.0)


def test_detect_directions_rejects_all_outside_fov(exposure_model):
    ra = np.full(1000, 180.0)
    dec = np.full(1000, 40.0)
    ra_acc, dec_acc = exposure_model.detect_directions(ra, dec)
    assert ra_acc.size == dec_acc.size == 0


def test_detect_directions_keeps_all_at_maximum(exposure_model):
    """At the Auger site the maximum sits at the south pole, where the
    acceptance probability is exactly 1."""
    ra = np.full(1000, 180.0)
    dec = np.full(1000, -90.0)
    ra_acc, _ = exposure_model.detect_directions(ra, dec)
    assert ra_acc.size == 1000


# -------------------------------------------------------------------------
# sample_iso_cumul_exposure
# -------------------------------------------------------------------------


def test_sample_iso_cumul_exposure_length(exposure_model):
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_norm_cumul_exposure(centre)
    rate = 1000.0 / max_eps
    sample, _ = exposure_model.sample_iso_cumul_exposure(
        n_events=1000, expected_exposure_rate=rate,
    )
    assert sample.size == 1000


def test_sample_iso_cumul_exposure_sorted(exposure_model):
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_norm_cumul_exposure(centre)
    rate = 1000.0 / max_eps
    sample, _ = exposure_model.sample_iso_cumul_exposure(
        n_events=1000, expected_exposure_rate=rate,
    )
    assert np.all(np.diff(sample) >= 0.0)


def test_sample_iso_cumul_exposure_starts_at_zero(exposure_model):
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_norm_cumul_exposure(centre)
    rate = 500.0 / max_eps
    sample, _ = exposure_model.sample_iso_cumul_exposure(
        n_events=500, expected_exposure_rate=rate,
    )
    assert sample[0] == 0.0
    assert np.all(sample >= 0.0)


def test_sample_iso_cumul_exposure_method_label(exposure_model):
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_norm_cumul_exposure(centre)
    rate = 100.0 / max_eps
    _, method = exposure_model.sample_iso_cumul_exposure(
        n_events=100, expected_exposure_rate=rate,
    )
    assert method == "exponential_delta_exposure_method"


def test_sample_iso_cumul_exposure_invalid_rate_raises(exposure_model):
    with pytest.raises(ValueError):
        exposure_model.sample_iso_cumul_exposure(
            n_events=10, expected_exposure_rate=0.0,
        )


def test_sample_iso_cumul_exposure_nonpositive_n_raises(exposure_model):
    for n in (-1, 0):
        with pytest.raises(ValueError):
            exposure_model.sample_iso_cumul_exposure(
                n_events=n, expected_exposure_rate=1.0,
            )


def test_sample_iso_cumul_exposure_single_event(exposure_model):
    """n_events=1 is a valid edge case: no gaps to draw, sample == [0.0]."""
    sample, _ = exposure_model.sample_iso_cumul_exposure(
        n_events=1, expected_exposure_rate=1.0,
    )
    assert sample.shape == (1,)
    assert sample[0] == 0.0


def test_sample_iso_cumul_exposure_mean_gap(exposure_model):
    """Mean exposure gap is ≈ 1 / expected_exposure_rate (Poisson-process intervals)."""
    centre = np.array([180.0, -30.0])
    max_eps = exposure_model.max_norm_cumul_exposure(centre)
    n = 5000
    rate = n / max_eps
    sample, _ = exposure_model.sample_iso_cumul_exposure(
        n_events=n, expected_exposure_rate=rate,
    )
    mean_gap = float(np.mean(np.diff(sample)))
    assert mean_gap == pytest.approx(1.0 / rate, rel=0.05)


# -------------------------------------------------------------------------
# Active area A(t)
# -------------------------------------------------------------------------


def _area_steps(t0, tf, step_sec, values):
    """Piecewise-constant A(t): steps of ``step_sec`` from t0, values from
    ``values(n)``, with the last step starting at or after tf (data up to tf)."""
    n = int(np.ceil((tf - t0).to_value("sec") / step_sec)) + 1
    starts = t0 + TimeDelta(np.arange(n) * step_sec, format="sec")
    return starts, values(n)


def _with_area(observatory, area):
    """The same observatory, with ``area`` (a constant or A(t) steps)."""
    return Observatory(
        latitude=observatory.latitude, longitude=observatory.longitude,
        altitude=observatory.altitude, area=area,
    )


CENTRES = [np.array([180.0, -30.0]), np.array([40.0, 10.0]), np.array([300.0, -80.0])]


def test_active_area_full_reproduces_closed_form(observatory, t0, tf, rng, exposure_model):
    steps = _area_steps(t0, tf, 3600.0, lambda n: np.full(n, observatory.area))
    model = ExposureModel(_with_area(observatory, steps), t0, tf, rng)
    times = t0 + TimeDelta(np.linspace(0.0, (tf - t0).to_value("sec"), 7), format="sec")
    for centre in CENTRES:
        assert model.norm_cumul_exposure(times, centre) == pytest.approx(
            exposure_model.norm_cumul_exposure(times, centre), rel=1e-9, abs=1e-12,
        )
    assert model.total_exposure() == pytest.approx(exposure_model.total_exposure())


def test_active_area_half_halves_exposure_but_not_window_share(observatory, t0, tf, rng, exposure_model):
    """Half the area throughout: physical exposures halve; A_max halves too,
    so the normalised exposure and the window share do not change."""
    from spacetimecorr import SkyWindow

    steps = _area_steps(t0, tf, 3600.0, lambda n: np.full(n, 0.5 * observatory.area))
    model = ExposureModel(_with_area(observatory, steps), t0, tf, rng)
    centre = CENTRES[0]
    window = SkyWindow(centre=centre, radius=3.0)
    assert model.total_exposure() == pytest.approx(0.5 * exposure_model.total_exposure())
    assert model.cumul_exposure(tf, window) == pytest.approx(
        0.5 * exposure_model.cumul_exposure(tf, window)
    )
    assert model.max_norm_cumul_exposure(centre) == pytest.approx(
        exposure_model.max_norm_cumul_exposure(centre)
    )
    assert model.relative_window_exposure(window) == pytest.approx(
        exposure_model.relative_window_exposure(window)
    )


def test_active_area_zero_stops_the_exposure(observatory, t0, tf, rng):
    """Full area in the first half of the run, none in the second."""
    half = 0.5 * (tf - t0).to_value("sec")
    starts = t0 + TimeDelta([0.0, half, 2.0 * half], format="sec")
    model = ExposureModel(
        _with_area(observatory, (starts, np.array([observatory.area, 0.0, 0.0]))), t0, tf, rng,
    )
    centre = CENTRES[0]
    later = t0 + TimeDelta(np.linspace(half, 2.0 * half, 5), format="sec")
    eps = model.norm_cumul_exposure(later, centre)
    assert np.all(eps == pytest.approx(eps[0], rel=1e-12))
    assert eps[0] > 0.0


def test_active_area_matches_numerical_integral(observatory, rng):
    """eps_norm(tf) against a brute-force sum of p_det(t), which includes A(t)/A_max."""
    t0 = Time("2020-01-01T00:00:00")
    tf = t0 + 3 * u.day
    # 72 hourly steps over the run, plus one starting at tf (data up to tf).
    values = np.random.default_rng(1).uniform(0.0, observatory.area, size=73)
    starts = t0 + TimeDelta(np.arange(73) * 3600.0, format="sec")
    model = ExposureModel(_with_area(observatory, (starts, values)), t0, tf, rng)

    dt = 10.0                                            # s
    mid = (np.arange(int(3 * 86400 / dt)) + 0.5) * dt
    times = t0 + TimeDelta(mid, format="sec")
    area_ratio = values[(mid // 3600).astype(int)] / values.max()   # A_max = max(A)
    geometric = ExposureModel(observatory, t0, tf, rng)             # constant area: factor 1
    for centre in CENTRES[:2]:
        p_geo = geometric.detection_probability(times, centre)
        assert model.detection_probability(times, centre) == pytest.approx(p_geo * area_ratio)
        expected = np.sum(p_geo * area_ratio) * dt / (tf - t0).to_value("sec")
        assert model.max_norm_cumul_exposure(centre) == pytest.approx(expected, rel=1e-4)


def test_active_area_starting_after_t0_raises(observatory, t0, tf, rng):
    starts = t0 + TimeDelta([60.0, 120.0], format="sec")
    with pytest.raises(ValueError):
        ExposureModel(_with_area(observatory, (starts, np.array([1.0, 1.0]))), t0, tf, rng)


def test_active_area_ending_before_tf_raises(observatory, t0, tf, rng):
    """The last start time is the last minute of data: a run past it raises."""
    starts = t0 + TimeDelta([0.0, 60.0], format="sec")
    with pytest.raises(ValueError):
        ExposureModel(_with_area(observatory, (starts, np.array([1.0, 1.0]))), t0, tf, rng)
