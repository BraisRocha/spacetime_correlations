"""
Isotropic skies from a mock sample.

:func:`isotropic_sky` draws the whole sky as a Poisson process in the
whole-sky cumulative exposure (:func:`exposure_time_table`): an event at
the start of the run, then exponential gaps until the end of the run. The
number of events of a sky comes out of the draw. The arrival times follow
from the exposure (:func:`exposure_to_time`); the zenith angles are drawn
from those of the sample, the azimuths uniform in [0, 360) deg, and both
are converted to ICRS at the new arrival times.

:func:`scramble` is the earlier procedure: the arrival times and the zenith
angles of the sample permuted independently, with the matrices of
:func:`local_to_icrs_matrices` computed once for the arrival times of the
sample.
"""

import astropy.units as u
import numpy as np

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time


def zenith_angles(
    times: Time, ra: np.ndarray, dec: np.ndarray, location: EarthLocation,
) -> np.ndarray:
    """Zenith angle (deg) of every event at its arrival time, from ICRS
    ``(ra, dec)`` (deg) to ``AltAz`` at ``location``."""
    local = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").transform_to(
        AltAz(obstime=times, location=location)
    )
    return 90.0 - local.alt.deg


def local_to_icrs_matrices(times: Time, location: EarthLocation) -> np.ndarray:
    """Matrix ``R(t)`` of every time, shape ``(N, 3, 3)``, that takes a
    local unit vector at ``location`` to ICRS::

        v_icrs = R(t) v_local,   v_local = (sin theta cos az, sin theta sin az, cos theta)

    with the azimuth ``az`` from north towards east. The columns of ``R(t)``
    are the ICRS unit vectors of north and east on the horizon and of the
    zenith, from astropy ``AltAz`` at ``t``.

    ``R(t)`` holds the rotations of the transformation (Earth rotation,
    precession, nutation, polar motion) exactly. Aberration, up to ~20
    arcsec, depends on the direction and is exact only for the three
    columns.
    """
    # Three directions against all the times, broadcast to shape (3, N).
    columns = SkyCoord(
        alt=np.array([0.0, 0.0, 90.0])[:, None] * u.deg,
        az=np.array([0.0, 90.0, 0.0])[:, None] * u.deg,
        frame=AltAz(obstime=times, location=location),
    ).transform_to("icrs")
    xyz = columns.cartesian.xyz.value          # (3 components, 3 columns, N)
    return np.transpose(xyz, (2, 0, 1))


def exposure_time_table(model) -> tuple[np.ndarray, np.ndarray]:
    """Whole-sky cumulative exposure at the edges of the steps of the active
    area of ``model`` (an ``stc.ExposureModel``)::

        u(t) = int_{t0}^{t} A(t') dt' / int_{t0}^{tf} A(t') dt'

    The whole-sky exposure is ``pi sin^2(theta_max) A(t)`` per unit time,
    so the constant cancels. ``u`` grows linearly on every step and is flat
    on the steps with ``A = 0``.

    Returns the edges, in seconds since ``t0``, and ``u`` at each.
    """
    edges = model._area_edges_sec
    cumul = np.concatenate(([0.0], np.cumsum(model._area_km2 * np.diff(edges))))
    return edges, cumul / cumul[-1]


def exposure_to_time(
    exposure: np.ndarray, edges: np.ndarray, cumul: np.ndarray,
) -> np.ndarray:
    """Seconds since ``t0`` at which ``u(t)`` of :func:`exposure_time_table`
    reaches every value of ``exposure``, in ``[0, 1)``.

    The step holding a value is the last edge with ``u <= exposure``, which
    is never a step with ``A = 0``, and ``u`` is inverted linearly in it.
    Linear interpolation over the edges would spread the flat steps over
    their neighbours instead.
    """
    k = np.clip(np.searchsorted(cumul, exposure, side="right") - 1, 0, cumul.size - 2)
    return edges[k] + (exposure - cumul[k]) / (cumul[k + 1] - cumul[k]) * (edges[k + 1] - edges[k])


def isotropic_sky(
    lam: float, theta: np.ndarray, edges: np.ndarray, cumul: np.ndarray,
    t0: Time, location: EarthLocation, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One isotropic sky, drawn as a Poisson process of ``lam`` events per
    unit of the whole-sky cumulative exposure ``u`` in ``[0, 1]``
    (:func:`exposure_time_table`).

    An event at ``u = 0``, as the run starts with an event, then gaps
    ``~ Exp(lam)`` until ``u`` passes 1; the gap that passes 1 is dropped.
    The number of events ``n`` is ``1 + Poisson(lam)``.

    Every event gets the zenith angle of a random event of the sample
    (``theta``, deg): ``n`` distinct ones when ``n`` is not above the size
    of the sample, otherwise all of them and ``n - theta.size`` random
    repeats. Azimuths are uniform in [0, 360) deg, and the local directions
    are converted to ICRS at the arrival times.

    Returns the arrival times in seconds since ``t0``, sorted, and the RA
    and Dec (deg) of every event.
    """
    # Drawn in chunks a little above the expected count; a chunk more is
    # drawn when they fall short of the end of the run.
    chunk = int(lam + 8.0 * np.sqrt(lam)) + 1
    positions = np.cumsum(rng.exponential(1.0 / lam, chunk))
    while positions[-1] < 1.0:
        positions = np.concatenate(
            (positions, positions[-1] + np.cumsum(rng.exponential(1.0 / lam, chunk)))
        )
    positions = np.concatenate(([0.0], positions[positions < 1.0]))
    n = positions.size
    seconds = exposure_to_time(positions, edges, cumul)

    if n <= theta.size:
        picked = rng.choice(theta.size, size=n, replace=False)
    else:
        picked = rng.permutation(np.concatenate(
            (np.arange(theta.size), rng.integers(theta.size, size=n - theta.size))
        ))
    new_theta = np.radians(theta[picked])
    azimuth = np.radians(rng.uniform(0.0, 360.0, n))

    v_local = np.column_stack((
        np.sin(new_theta) * np.cos(azimuth),
        np.sin(new_theta) * np.sin(azimuth),
        np.cos(new_theta),
    ))
    matrices = local_to_icrs_matrices(t0 + seconds * u.s, location)
    x, y, z = np.einsum("nij,nj->in", matrices, v_local)

    # As in scramble: R(t) v is unit only to ~1e-4, so the norm is divided out.
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    return seconds, ra, dec


def scramble(
    theta: np.ndarray, matrices: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One isotropic sky: the arrival times and ``theta`` (deg) of the
    sample permuted independently, azimuths uniform in [0, 360) deg,
    converted to ICRS with ``matrices`` from :func:`local_to_icrs_matrices`.

    Returns, for every new event, the index of its arrival time in the
    sample, its RA and its Dec (deg).
    """
    order = rng.permutation(theta.size)
    new_theta = np.radians(theta[rng.permutation(theta.size)])
    azimuth = np.radians(rng.uniform(0.0, 360.0, theta.size))

    v_local = np.column_stack((
        np.sin(new_theta) * np.cos(azimuth),
        np.sin(new_theta) * np.sin(azimuth),
        np.cos(new_theta),
    ))
    x, y, z = np.einsum("nij,nj->in", matrices[order], v_local)

    # The columns of R(t) carry their own aberration, so R(t) v is unit
    # only to ~1e-4: the norm is divided out.
    ra = np.degrees(np.arctan2(y, x)) % 360.0
    dec = np.degrees(np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)))
    return order, ra, dec
