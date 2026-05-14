from __future__ import annotations

from typing import TYPE_CHECKING

from .event_sample import EventSample

import numpy as np
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .exposure import ExposureModel

@dataclass(frozen=True, slots=True)
class SkyWindow:
    """A circular window (spherical cap) on the celestial sphere.

    Parameters
    ----------
    centre : array-like of shape (2,)
        [RA_deg, Dec_deg] in degrees.
        RA must be in [0, 360), Dec in [-90, 90].
    radius : float
        Angular radius in degrees, in (0, 180].

    Notes
    -----
    This class is *geometry-only*. It provides selection masks, spherical-cap
    sky fraction, and uniform sampling within the cap.  Expected-count
    computations are available as a convenience method but are only meaningful
    under the stated exposure assumptions.
    """

    centre: np.ndarray  # shape (2,) -> [RA_deg, Dec_deg]
    radius: float       # degrees

    # Cached private attributes (set in __post_init__)
    _center_vec: np.ndarray = field(init=False, repr=False, compare=False)
    _cos_radius: float      = field(init=False, repr=False, compare=False)
    _sky_fraction: float    = field(init=False, repr=False, compare=False)

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        # --- coerce + validate centre ---
        c = np.asarray(self.centre, dtype=float)
        if c.size != 2:
            raise TypeError("centre must be array-like with 2 elements: [RA_deg, Dec_deg].")
        c = c.reshape(2,)
        ra, dec = float(c[0]), float(c[1])

        if not (0.0 <= ra < 360.0):
            raise ValueError("RA must be in [0, 360).")
        if not (-90.0 <= dec <= 90.0):
            raise ValueError("Dec must be in [-90, 90].")

        r = float(self.radius)
        if not (0.0 < r <= 180.0):
            raise ValueError("Radius must be in (0, 180].")

        # write back coerced values into frozen dataclass
        object.__setattr__(self, "centre", c)
        object.__setattr__(self, "radius", r)

        # --- cache constants (radians, vectors, cos cut) ---
        ra_c_rad, dec_c_rad = np.deg2rad(c)
        radius_rad = np.deg2rad(r)

        center_vec = np.array(
            [
                np.cos(dec_c_rad) * np.cos(ra_c_rad),
                np.cos(dec_c_rad) * np.sin(ra_c_rad),
                np.sin(dec_c_rad),
            ],
            dtype=float,
        )

        object.__setattr__(self, "_center_vec", center_vec)
        object.__setattr__(self, "_cos_radius", float(np.cos(radius_rad)))
        object.__setattr__(self, "_sky_fraction", float((1.0 - np.cos(radius_rad)) / 2.0))

    # -------------------------------------------------------------------------
    # Basic properties
    # -------------------------------------------------------------------------

    @property
    def sky_fraction(self) -> float:
        """Fraction of the full sky covered by this window (spherical cap)."""
        return self._sky_fraction

    # -------------------------------------------------------------------------
    # Geometric selection
    # -------------------------------------------------------------------------

    def contains(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """Return boolean mask selecting coordinates inside the window.

        Parameters
        ----------
        ra : np.ndarray
            Right ascension values in degrees.
        dec : np.ndarray
            Declination values in degrees.

        Returns
        -------
        np.ndarray of bool
            True for coordinates within the angular radius of the window centre.
        """
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)

        if ra.shape != dec.shape:
            raise ValueError(
                f"ra and dec must have the same shape, got {ra.shape} vs {dec.shape}."
            )

        ra_rad  = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

        event_vecs = np.column_stack(
            (
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            )
        )

        dots = event_vecs @ self._center_vec
        dots = np.clip(dots, -1.0, 1.0)

        return dots >= self._cos_radius

    # -------------------------------------------------------------------------
    # Uniform sampling
    # -------------------------------------------------------------------------

    def sample_uniform(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample ``n`` points distributed uniformly in solid angle within the window.

        The method samples in a local frame with the cap centre at the north
        pole and then rotates to the true centre direction.  In the local
        frame, ``cos(theta_local)`` is drawn uniformly in
        ``[cos(radius), 1]`` (which gives uniform solid-angle coverage) and
        the azimuthal angle ``phi`` is drawn uniformly in ``[0, 2π)``.

        Parameters
        ----------
        n : int
            Number of points to sample.
        rng : numpy.random.Generator
            Random generator.

        Returns
        -------
        ra, dec : numpy.ndarray
            Right ascension and declination in degrees, each of shape ``(n,)``.
        """
        if n == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        ra_c  = np.deg2rad(self.centre[0])
        dec_c = np.deg2rad(self.centre[1])

        # --- local-frame sampling (cap centre = north pole) ---
        cos_theta = rng.uniform(self._cos_radius, 1.0, size=n)
        phi       = rng.uniform(0.0, 2.0 * np.pi, size=n)

        sin_theta = np.sqrt(1.0 - cos_theta ** 2)
        x_local   = sin_theta * np.cos(phi)
        y_local   = sin_theta * np.sin(phi)
        z_local   = cos_theta

        # --- orthonormal frame at the cap centre ---
        n_hat   = np.array([
            np.cos(dec_c) * np.cos(ra_c),
            np.cos(dec_c) * np.sin(ra_c),
            np.sin(dec_c),
        ])
        e_east  = np.array([-np.sin(ra_c), np.cos(ra_c), 0.0])
        e_north = np.cross(n_hat, e_east)

        # --- rotate local samples into the equatorial frame ---
        x = e_east[0] * x_local + e_north[0] * y_local + n_hat[0] * z_local
        y = e_east[1] * x_local + e_north[1] * y_local + n_hat[1] * z_local
        z = e_east[2] * x_local + e_north[2] * y_local + n_hat[2] * z_local

        ra  = np.degrees(np.arctan2(y, x)) % 360.0
        dec = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))

        return ra, dec

    # -------------------------------------------------------------------------
    # Expected event counts
    # -------------------------------------------------------------------------

    def expected_n_in_window(
        self,
        n_events: int | float,
        exposure_model: "ExposureModel | None" = None,
    ) -> float:
        """
        Expected number of events in the window.

        When an :class:`ExposureModel` is supplied, the count is weighted by
        the analytical relative directional exposure ``omega(delta_centre)``
        evaluated at the window centre (see
        :meth:`ExposureModel.relative_exposure`)::

            expected_n = n_events * sky_fraction * omega(delta_centre)

        so that windows at well-exposed declinations get more events and
        poorly-exposed ones get fewer.  If no exposure model is provided,
        all declinations are weighted equally and the formula reduces to::

            expected_n = n_events * sky_fraction

        Parameters
        ----------
        n_events : int or float
            Total number of events in the full sky.
        exposure_model : ExposureModel or None, optional
            If provided, weights the result by
            ``exposure_model.relative_exposure(self.centre)``.  If ``None``
            (default), assumes uniform full-sky exposure.

        Returns
        -------
        float
            Expected number of events in the window.
        """
        if exposure_model is None:
            return float(n_events) * self.sky_fraction

        weight = exposure_model.relative_exposure(self.centre)
        return float(n_events) * self.sky_fraction * weight
