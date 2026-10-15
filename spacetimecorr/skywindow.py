"""
Spherical-cap windows on the celestial sphere.

Defines :class:`SkyWindow`, a frozen geometry-only dataclass representing
a circular window (spherical cap) parameterised by a centre
``[RA_deg, Dec_deg]`` and an angular radius in degrees. The class
provides:

- containment masks (which events fall inside the cap),
- the spherical-cap sky fraction,
- uniform sampling within the cap (used by per-window event factories),
- the exposure-weighted expected event count
  (:meth:`expected_n_in_window`), which folds the directional exposure
  at the window centre into the bare sky-fraction expectation.

No exposure or event-generation state is held here — those live on
:class:`~spacetimecorr.exposure.ExposureModel` and
:class:`~spacetimecorr.event_sample.EventSample` respectively. The
window is treated as small enough that the exposure at its centre is a
good proxy for the exposure across the cap.

:class:`HealpixWindow` is the same kind of window built from one HEALPix
pixel instead of a cap. It offers the interface the exposure code reads
from a window (``centre``, ``solid_angle``, ``contains``,
``expected_n_in_window``), so either can be passed wherever a window is
expected. :class:`HealpixGrid` holds all the pixels at one resolution
whose centre lies inside the field of view.

This module also defines :class:`SkyGrid`, a container for ``N`` such
windows stored struct-of-arrays (``centres`` of shape ``(N, 2)`` and
``radii`` of shape ``(N,)``). Indexing a grid (``grid[i]``) materialises
an ordinary :class:`SkyWindow`, so a multi-window analysis is just the
single-window pipeline run in a loop, while leaving room to vectorise the
deterministic per-window scalars behind that boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from .event_sample import EventSample

import numpy as np
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .exposure import ExposureModel


def _healpy():
    """Import ``healpy`` lazily, with a helpful message if it is missing.

    ``healpy`` is an optional dependency (the ``skymap`` extra), only
    needed by the HEALPix windows.
    """
    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "HEALPix windows require the optional dependency 'healpy'. "
            "Install it with `pip install healpy`."
        ) from exc
    return hp


@dataclass(frozen=True)  # NOTE: no slots=True (requires Python >= 3.10)
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
    _e_east: np.ndarray     = field(init=False, repr=False, compare=False)
    _e_north: np.ndarray    = field(init=False, repr=False, compare=False)
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

        e_east  = np.array([-np.sin(ra_c_rad), np.cos(ra_c_rad), 0.0])
        e_north = np.cross(center_vec, e_east)

        object.__setattr__(self, "_center_vec", center_vec)
        object.__setattr__(self, "_e_east", e_east)
        object.__setattr__(self, "_e_north", e_north)
        object.__setattr__(self, "_cos_radius", float(np.cos(radius_rad)))
        object.__setattr__(self, "_sky_fraction", float((1.0 - np.cos(radius_rad)) / 2.0))

    # -------------------------------------------------------------------------
    # Basic properties
    # -------------------------------------------------------------------------

    @property
    def sky_fraction(self) -> float:
        """Fraction of the full sky covered by this window (spherical cap)."""
        return self._sky_fraction

    @property
    def solid_angle(self) -> float:
        """
        Solid angle subtended by the window, in steradians::

            Omega = 4 pi * sky_fraction = 2 pi (1 - cos radius)
        """
        return 4.0 * np.pi * self._sky_fraction

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

        cos_dec = np.cos(dec_rad)
        sin_dec = np.sin(dec_rad)

        sin_ra = np.sin(ra_rad)
        cos_ra = np.cos(ra_rad)

        cx, cy, cz = self._center_vec

        dots = (
            cos_dec * cos_ra * cx
            + cos_dec * sin_ra * cy
            + sin_dec * cz
        )

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

        # --- local-frame sampling (cap centre = north pole) ---
        cos_theta = rng.uniform(self._cos_radius, 1.0, size=n)
        phi       = rng.uniform(0.0, 2.0 * np.pi, size=n)

        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        x_local   = sin_theta * np.cos(phi)
        y_local   = sin_theta * np.sin(phi)
        z_local   = cos_theta

        # --- cached orthonormal frame ---
        e_east  = self._e_east
        e_north = self._e_north
        n_hat   = self._center_vec

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

        When an :class:`ExposureModel` is supplied, the count follows the
        share of the observatory's exposure that falls on this window::

            expected_n = n_events * E(window) / E_sky

        i.e. :meth:`ExposureModel.relative_window_exposure`, so windows at
        well-exposed declinations get more events and poorly-exposed ones
        fewer.  Both exposures run over the full interval ``[t0, tf]``, and
        because ``∫_sky eps_norm dOmega`` is analytic the per-window counts
        sum to ``n_events`` exactly over a full-sky tiling.

        If no exposure model is provided, all directions are weighted
        equally and the formula reduces to::

            expected_n = n_events * sky_fraction

        Parameters
        ----------
        n_events : int or float
            Total number of events in the full sky.
        exposure_model : ExposureModel or None, optional
            If provided, weights the result by
            ``exposure_model.relative_window_exposure(self)``.  If ``None``
            (default), assumes uniform full-sky exposure.

        Returns
        -------
        float
            Expected number of events in the window.
        """
        if exposure_model is None:
            return float(n_events) * self._sky_fraction

        return float(n_events) * exposure_model.relative_window_exposure(self)


@dataclass(frozen=True)
class HealpixWindow:
    """One HEALPix pixel used as a window on the celestial sphere.

    Parameters
    ----------
    nside : int
        HEALPix resolution. All ``12 nside^2`` pixels have the same solid
        angle, ``4 pi / (12 nside^2)`` sr.
    ipix : int
        Pixel index in RING ordering, in ``[0, 12 nside^2)``.

    Notes
    -----
    As for a :class:`SkyWindow`, the exposure towards the pixel is
    evaluated at its centre: the pixel enters only through ``centre`` and
    ``solid_angle``.
    """

    nside: int
    ipix: int

    # Set in __post_init__ from (nside, ipix)
    centre: np.ndarray = field(init=False, compare=False)            # [RA_deg, Dec_deg]
    solid_angle: float = field(init=False, repr=False, compare=False)  # sr

    def __post_init__(self) -> None:
        hp = _healpy()
        nside = int(self.nside)
        ipix = int(self.ipix)

        if not hp.isnsideok(nside):
            raise ValueError(f"nside={self.nside} is not a valid HEALPix resolution.")
        npix = hp.nside2npix(nside)
        if not (0 <= ipix < npix):
            raise ValueError(f"ipix must be in [0, {npix}) for nside={nside}; got {self.ipix}.")

        ra, dec = hp.pix2ang(nside, ipix, lonlat=True)

        object.__setattr__(self, "nside", nside)
        object.__setattr__(self, "ipix", ipix)
        object.__setattr__(self, "centre", np.array([float(ra), float(dec)]))
        object.__setattr__(self, "solid_angle", float(hp.nside2pixarea(nside)))

    def contains(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """Return a boolean mask, True for coordinates (deg) inside this pixel."""
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)

        if ra.shape != dec.shape:
            raise ValueError(
                f"ra and dec must have the same shape, got {ra.shape} vs {dec.shape}."
            )

        return _healpy().ang2pix(self.nside, ra, dec, lonlat=True) == self.ipix

    def expected_n_in_window(
        self,
        n_events: int | float,
        exposure_model: "ExposureModel | None" = None,
    ) -> float:
        """
        Expected number of events in the pixel.

        Same as :meth:`SkyWindow.expected_n_in_window`::

            expected_n = n_events * E(pixel) / E_sky

        with ``E(pixel)`` from the exposure at the pixel centre, or
        ``n_events * solid_angle / 4 pi`` if no exposure model is given.
        """
        if exposure_model is None:
            return float(n_events) * self.solid_angle / (4.0 * np.pi)

        return float(n_events) * exposure_model.relative_window_exposure(self)


class SkyGrid:
    """A collection of ``N`` spherical-cap windows.

    Parameters
    ----------
    centres : array-like of shape (N, 2)
        One ``[RA_deg, Dec_deg]`` per window. RA in ``[0, 360)``,
        Dec in ``[-90, 90]``.
    radii : float or array-like of shape (N,)
        Angular radius in degrees, in ``(0, 180]``. A scalar is
        broadcast to all ``N`` windows (the common case for a uniform
        tiling or a fixed search radius).

    Notes
    -----
    Storage is *struct-of-arrays*: the grid keeps only the raw ``centres``
    and ``radii`` arrays, never any :class:`SkyWindow` objects. This keeps
    construction cheap for large grids, keeps memory flat across a long
    loop (each window is freed once its iteration ends), and leaves the
    door open to vectorising the deterministic per-window scalars
    (sky fraction, expected counts, containment) behind this boundary
    without changing the public API.

    Usage mirrors indexing an array: ``grid[i]`` builds and returns a
    fresh :class:`SkyWindow`, so a multi-window analysis is just the
    existing single-window pipeline run in a loop::

        grid = SkyGrid(centres, radii)
        for window in grid:
            sample = EventSample.in_window(window=window, ...)
            sample.assign_cumul_exposure(window=window, ...)
            lam = lambda_estimator(sample)

    Validation mirrors :class:`SkyWindow` so that a window obtained via
    ``grid[i]`` is indistinguishable from one built directly.
    """

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(self, centres: np.ndarray, radii: float | np.ndarray):
        # --- coerce + validate centres ---
        c = np.asarray(centres, dtype=float)
        if c.ndim != 2 or c.shape[1] != 2:
            raise ValueError(
                f"centres must have shape (N, 2) -> [RA_deg, Dec_deg]; got {c.shape}."
            )
        n = c.shape[0]
        if n == 0:
            raise ValueError("centres must contain at least one window.")

        ra = c[:, 0]
        dec = c[:, 1]
        if np.any((ra < 0.0) | (ra >= 360.0)):
            raise ValueError("All RA values must be in [0, 360).")
        if np.any((dec < -90.0) | (dec > 90.0)):
            raise ValueError("All Dec values must be in [-90, 90].")

        # --- coerce + validate radii (scalar broadcasts) ---
        r = np.asarray(radii, dtype=float)
        if r.ndim == 0:
            r = np.full(n, float(r))
        else:
            r = r.reshape(-1)
            if r.shape[0] != n:
                raise ValueError(
                    f"radii must be a scalar or have shape (N,) with N={n}; "
                    f"got shape {r.shape}."
                )
        if np.any((r <= 0.0) | (r > 180.0)):
            raise ValueError("All radii must be in (0, 180].")

        self._centres = c
        self._radii = r

    @classmethod
    def from_arrays(
        cls,
        centres: np.ndarray,
        radii: float | np.ndarray,
    ) -> "SkyGrid":
        """
        Build a grid from explicit centre and radius arrays.

        Named factory parallel to the
        :class:`~spacetimecorr.event_sample.EventSample` constructors;
        equivalent to ``SkyGrid(centres, radii)``.

        Parameters
        ----------
        centres : array-like of shape (N, 2)
            One ``[RA_deg, Dec_deg]`` per window.
        radii : float or array-like of shape (N,)
            Angular radius/radii in degrees; a scalar broadcasts to all
            windows.

        Returns
        -------
        SkyGrid
        """
        return cls(centres=centres, radii=radii)

    # -------------------------------------------------------------------------
    # Container protocol
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return self._centres.shape[0]

    def __getitem__(self, index: int | slice | np.ndarray) -> "SkyWindow | SkyGrid":
        """
        Index the grid like an array.

        - An integer returns the corresponding :class:`SkyWindow`,
          freshly built from the stored ``centres``/``radii`` and ready
          to drop into the existing single-window pipeline.
        - A slice or an array of indices returns a new :class:`SkyGrid`
          sub-grid.

        Notes
        -----
        The grid stores no window objects: each integer index builds and
        returns a new :class:`SkyWindow`. Bind it once per iteration
        (``for window in grid`` or ``window = grid[i]``) and reuse that
        reference — passing the same ``window`` to several callers does
        not rebuild it. Only re-indexing the grid (``grid[i]`` again)
        constructs another instance. This keeps memory flat across a long
        loop, since each window is freed once the iteration that built it
        ends.
        """
        if isinstance(index, (int, np.integer)):
            i = int(index)
            if i < 0:
                i += len(self)
            if not (0 <= i < len(self)):
                raise IndexError(
                    f"window index {index} out of range for grid of size {len(self)}."
                )
            return SkyWindow(centre=self._centres[i], radius=float(self._radii[i]))

        # slice / fancy index -> sub-grid
        return SkyGrid(centres=self._centres[index], radii=self._radii[index])

    def __iter__(self) -> Iterator["SkyWindow"]:
        for i in range(len(self)):
            yield self[i]  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"SkyGrid(n_windows={len(self)})"

    # -------------------------------------------------------------------------
    # Array accessors
    # -------------------------------------------------------------------------

    @property
    def centres(self) -> np.ndarray:
        """Window centres, shape ``(N, 2)`` -> ``[RA_deg, Dec_deg]``."""
        return self._centres

    @property
    def radii(self) -> np.ndarray:
        """Window radii in degrees, shape ``(N,)``."""
        return self._radii

    # -------------------------------------------------------------------------
    # Vectorised deterministic per-window scalars
    # -------------------------------------------------------------------------

    @property
    def sky_fraction(self) -> np.ndarray:
        """Per-window full-sky fraction (spherical cap), shape ``(N,)``.

        Vectorised counterpart of :attr:`SkyWindow.sky_fraction`:
        ``(1 - cos(radius)) / 2`` evaluated over all radii at once.
        """
        return (1.0 - np.cos(np.deg2rad(self._radii))) / 2.0

    def expected_n_in_window(
        self,
        n_events: int | float,
        exposure_model: "ExposureModel | None" = None,
    ) -> np.ndarray:
        """
        Expected event count in each window, shape ``(N,)``.

        Mirrors :meth:`SkyWindow.expected_n_in_window` for every window.
        With no exposure model the result is ``n_events * sky_fraction``;
        with one, each window takes its share of the observatory's total
        exposure (see
        :meth:`ExposureModel.relative_window_exposure`), so the counts sum
        to ``n_events`` over a full-sky tiling.

        Notes
        -----
        Currently delegates per window (the exposure weight loops over
        centres). It is exposed here as a single ``(N,)`` array so the
        weighting can be vectorised later without changing callers.
        """
        if exposure_model is None:
            return float(n_events) * self.sky_fraction

        weights = np.array(
            [
                exposure_model.relative_window_exposure(
                    SkyWindow(centre=c, radius=r)
                )
                for c, r in zip(self._centres, self._radii)
            ],
            dtype=float,
        )
        return float(n_events) * weights


class HealpixGrid:
    """All HEALPix pixels at resolution ``nside`` whose centre lies inside
    the field of view.

    Parameters
    ----------
    exposure_model : ExposureModel
        Defines the field of view: a pixel is kept when the exposure per
        solid angle at its centre, :meth:`ExposureModel.spatial_exposure`,
        is positive, i.e. when its centre ever enters the zenith cut.
    nside : int, optional
        HEALPix resolution, RING ordering. Defaults to 32: pixels of
        ~3.4 deg², about the area of a cap of radius 1.05 deg.

    Notes
    -----
    Only the indices and centres of the kept pixels are stored;
    ``grid[i]`` builds the :class:`HealpixWindow` of the ``i``-th one.
    A pixel whose centre lies outside the FoV is dropped even if part of
    it lies inside (see ``TODO.md``).

    Assigning all events to their pixel at once, instead of filtering
    the sample once per window::

        grid = HealpixGrid(exposure_model)
        pix = grid.pixel_of(sample.ra, sample.dec)
        for window in grid:
            subsample = sample.select_subsample(window, mask=pix == window.ipix)
    """

    def __init__(self, exposure_model: "ExposureModel", nside: int = 32):
        hp = _healpy()
        nside = int(nside)
        if not hp.isnsideok(nside):
            raise ValueError(f"nside={nside} is not a valid HEALPix resolution.")

        ipix = np.arange(hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, ipix, lonlat=True)
        #in_fov = exposure_model.spatial_exposure(dec) > 0.0
        lat = exposure_model.observatory.latitude
        in_fov = np.abs(dec - lat) < exposure_model.theta_max_deg

        self._nside = nside
        self._pixels = ipix[in_fov]
        self._centres = np.column_stack((ra[in_fov], dec[in_fov]))

    # -------------------------------------------------------------------------
    # Container protocol
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return self._pixels.size

    def __getitem__(self, index: int) -> HealpixWindow:
        """The :class:`HealpixWindow` of the ``index``-th kept pixel."""
        return HealpixWindow(nside=self._nside, ipix=int(self._pixels[index]))

    def __iter__(self) -> Iterator[HealpixWindow]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"HealpixGrid(nside={self._nside}, n_windows={len(self)})"

    # -------------------------------------------------------------------------
    # Array accessors
    # -------------------------------------------------------------------------

    @property
    def nside(self) -> int:
        """HEALPix resolution."""
        return self._nside

    @property
    def pixels(self) -> np.ndarray:
        """RING indices of the kept pixels, shape ``(N,)``."""
        return self._pixels

    @property
    def centres(self) -> np.ndarray:
        """Pixel centres, shape ``(N, 2)`` -> ``[RA_deg, Dec_deg]``."""
        return self._centres

    @property
    def solid_angle(self) -> float:
        """Solid angle of every pixel, in sr."""
        return float(_healpy().nside2pixarea(self._nside))

    # -------------------------------------------------------------------------
    # Event assignment and expected counts
    # -------------------------------------------------------------------------

    def pixel_of(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """RING index of the pixel containing each direction (deg).

        Directions falling in a dropped pixel get an index that matches
        no window of the grid.
        """
        return _healpy().ang2pix(self._nside, ra, dec, lonlat=True)

    def expected_n_in_window(
        self,
        n_events: int | float,
        exposure_model: "ExposureModel | None" = None,
    ) -> np.ndarray:
        """
        Expected event count in each window, shape ``(N,)``.

        :meth:`HealpixWindow.expected_n_in_window` evaluated for every
        window of the grid.
        """
        return np.array(
            [window.expected_n_in_window(n_events, exposure_model) for window in self],
            dtype=float,
        )
