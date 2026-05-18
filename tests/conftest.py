"""
Shared fixtures for the spacetimecorr test suite.

Everything that more than one test file would otherwise rebuild lives here:
- a fixed seed and RNG helpers,
- the observation interval ``t0`` / ``tf``,
- a representative Pierre Auger ``Observatory``,
- an ``ExposureModel`` and a ``SkyWindow`` configured for that observatory.

Fixtures are function-scoped so each test gets a freshly-seeded RNG.
"""

import numpy as np
import pytest
from astropy.time import Time

from spacetimecorr import (
    ExposureModel,
    Observatory,
    RNGManager,
    SkyWindow,
)


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def rng_manager(seed: int) -> RNGManager:
    return RNGManager(seed=seed)


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture
def t0() -> Time:
    return Time("2020-01-01T00:00:00")


@pytest.fixture
def tf() -> Time:
    return Time("2021-01-01T00:00:00")


@pytest.fixture
def observatory() -> Observatory:
    """Pierre Auger Observatory."""
    return Observatory(latitude=-35.15, longitude=-69.15, altitude=1425.0)


@pytest.fixture
def exposure_model(
    observatory: Observatory,
    t0: Time,
    tf: Time,
    rng_manager: RNGManager,
) -> ExposureModel:
    return ExposureModel(
        observatory=observatory,
        t0=t0,
        tf=tf,
        rng=rng_manager.get("exposure"),
        theta_max_deg=60.0,
    )


@pytest.fixture
def window() -> SkyWindow:
    """Visible-sky cap centred on a declination Auger can see well."""
    return SkyWindow(centre=[180.0, -30.0], radius=15.0)


@pytest.fixture
def small_window() -> SkyWindow:
    """Small cap, useful for compact-flare tests."""
    return SkyWindow(centre=[180.0, -30.0], radius=3.0)
