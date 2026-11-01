"""
Isotropic skies from a mock sample, by scrambling its events in local
coordinates.

Every event is converted from equatorial (ICRS) to local coordinates at
its arrival time (astropy ``AltAz`` at the observatory). The arrival times
and the zenith angles are then permuted independently, and every azimuth
is drawn uniform in [0, 360) deg. Converting back to ICRS gives a sky with
the time and zenith-angle distributions of the sample, which carry its
exposure, and no anisotropy.

The conversion back to ICRS uses one matrix per arrival time, from
:func:`local_to_icrs_matrices`, computed once for all the skies, since
every sky reuses the arrival times of the sample.
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
