from __future__ import annotations

from .event_sample import EventSample

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


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
    This class is *geometry-only*. It provides selection masks and the
    spherical-cap sky fraction. Any expected-count computation is only valid
    under uniform full-sky exposure assumptions, and is provided separately
    as a convenience method.
    """

    centre: np.ndarray  # shape (2,) -> [RA_deg, Dec_deg]
    radius: float       # degrees

    # Cached private attributes (set in __post_init__)
    _center_vec: np.ndarray = field(init=False, repr=False, compare=False)
    _cos_radius: float      = field(init=False, repr=False, compare=False)
    _sky_fraction: float    = field(init=False, repr=False, compare=False)

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

    @property
    def sky_fraction(self) -> float:
        """Fraction of the full sky covered by this window (spherical cap)."""
        return self._sky_fraction

    def contains(self, sample: EventSample) -> np.ndarray:
        """Return boolean mask selecting events inside the window.

        Returns
        -------
        mask : np.ndarray of bool, shape (n_events,)
            True for events within angular radius of the centre.
        """
        if not sample.is_populated():
            raise ValueError("RA/Dec are not set in the sample. Call sample_equatorial_coordinates() first.")

        ra = np.asarray(sample.RA, dtype=float)
        dec = np.asarray(sample.Dec, dtype=float)

        if ra.shape != dec.shape:
            raise ValueError(f"sample.RA and sample.Dec must have the same shape, got {ra.shape} vs {dec.shape}.")

        # Convert to radians
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

        # Unit vectors for events
        event_vecs = np.column_stack(
            (
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            )
        )

        # Dot products with cached center vector
        dots = event_vecs @ self._center_vec
        dots = np.clip(dots, -1.0, 1.0)  # numeric safety

        # Angular cut
        print("RA min/max:", ra.min(), ra.max())
        print("Dec min/max:", dec.min(), dec.max())
        print("centre:", self.centre, "radius:", self.radius)
        print("_cos_radius:", self._cos_radius)
        print("center_vec norm:", np.linalg.norm(self._center_vec))
        print("dots min/max:", dots.min(), dots.max())
        print("any inside?:", np.any(dots >= self._cos_radius), "count:", np.count_nonzero(dots >= self._cos_radius))

        return dots >= self._cos_radius

    def uniform_expected_count(self, sample: EventSample) -> float:
        """Expected number of events in the window under uniform full-sky exposure."""
        return sample.n_events * self.sky_fraction

    def select(self, sample: EventSample) -> Tuple[EventSample, float]:
        """Convenience: return (subset_sample, uniform_expected_count)."""
        mask = self.contains(sample)
        if not np.any(mask):
            print("WARNING: No events found inside the sky window.")
        return sample.subset(mask), self.uniform_expected_count(sample)