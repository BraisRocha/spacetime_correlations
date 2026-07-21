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

This module also defines :class:`SkyGrid`, a container for ``N`` such
windows stored struct-of-arrays (``centres`` of shape ``(N, 2)`` and
``radii`` of shape ``(N,)``). Indexing a grid (``grid[i]``) materialises
an ordinary :class:`SkyWindow`, so a multi-window analysis is just the
single-window pipeline run in a loop, while leaving room to vectorise the
deterministic per-window scalars behind that boundary.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterator

from .event_sample import EventSample

import numpy as np
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .exposure import ExposureModel

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

        When an :class:`ExposureModel` is supplied, the count is weighted by
        the analytical relative directional exposure ``omega(delta_centre)``
        evaluated at the window centre (see
        :meth:`ExposureModel.relative_exposure`), normalised by its sky
        average ``<omega>`` (see
        :attr:`ExposureModel.mean_relative_exposure`)::

            expected_n = n_events * sky_fraction * omega(delta_centre) / <omega>

        so that windows at well-exposed declinations get more events and
        poorly-exposed ones get fewer.  The ``/ <omega>`` normalisation makes
        the weight a proper probability density (unit sky average), so the
        per-window counts sum to ``n_events`` over a full-sky tiling.  If no
        exposure model is provided, all declinations are weighted equally and
        the formula reduces to::

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
            return float(n_events) * self._sky_fraction

        weight = exposure_model.relative_exposure(self.centre)
        weight /= exposure_model.mean_relative_exposure
        return float(n_events) * self._sky_fraction * weight


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
            sample.assign_directional_exposure(window=window, ...)
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
        presently equivalent to ``SkyGrid(centres, radii)`` and present so
        that future construction routes (e.g. a HEALPix tiling) can sit
        alongside it as sibling classmethods.

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
    # HEALPix construction
    # -------------------------------------------------------------------------

    @staticmethod
    def _healpy():
        """Import ``healpy`` lazily, with a helpful message if it is missing.

        ``healpy`` is only needed for the HEALPix factory, so the rest of the
        module (and ``from_arrays``) stays dependency-free.
        """
        try:
            import healpy as hp
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "SkyGrid.from_healpix / min_covering_nside require the optional "
                "dependency 'healpy'. Install it with `pip install healpy`."
            ) from exc
        return hp

    @staticmethod
    def min_covering_nside(radius: float, nside_max: int = 1 << 20) -> int:
        """Coarsest power-of-two ``nside`` whose pixels are covered by caps of ``radius``.

        A cap of angular radius ``radius`` (deg) centred on a pixel fully
        contains that pixel iff ``radius >= max_pixrad(nside)`` (the largest
        centre-to-corner distance). Since HEALPix pixels tile the sphere with
        no gaps, this guarantees the union of caps covers everything. The
        smallest such ``nside`` is returned: the fewest, least-overlapping
        windows that still leave no gaps at the given radius.

        Parameters
        ----------
        radius : float
            Search-cap angular radius in degrees, in ``(0, 180]``.
        nside_max : int, optional
            Upper bound on the search over powers of two (safety stop).

        Returns
        -------
        int
            The minimal covering ``nside``.
        """
        hp = SkyGrid._healpy()
        radius = float(radius)
        if not (0.0 < radius <= 180.0):
            raise ValueError("radius must be in (0, 180] degrees.")

        radius_rad = np.deg2rad(radius)
        nside = 1
        while nside <= nside_max:
            if hp.max_pixrad(nside) <= radius_rad:
                return int(nside)
            nside *= 2
        raise ValueError(
            f"No power-of-two nside <= {nside_max} has pixels small enough to be "
            f"covered by a cap of radius {radius} deg."
        )

    @classmethod
    def from_healpix(
        cls,
        radius: float,
        nside: int | None = None,
        *,
        observatory=None,
        theta_max_deg: float = 60.0,
        nest: bool = False,
    ) -> "SkyGrid":
        """Build a grid of fixed-radius windows on HEALPix pixel centres.

        The windows all share the search radius ``radius``; their centres are
        the centres of the HEALPix pixels at resolution ``nside``. The radius
        is the physics-driven input (e.g. the SNR-optimal search scale), and
        ``nside`` defaults to the coarsest grid that radius still covers
        without gaps (see :meth:`min_covering_nside`).

        Optionally restricts the grid to windows that lie *entirely* inside an
        observatory field of view: given ``observatory`` (anything exposing a
        ``latitude`` in degrees) and a zenith cut ``theta_max_deg``, the
        visible declination band is

            ``dec_lo = max(-90, latitude - theta_max_deg)``
            ``dec_hi = min(+90, latitude + theta_max_deg)``

        and a window survives only when its whole cap fits inside it,

            ``min(90, dec_c + radius) <= dec_hi``  and
            ``max(-90, dec_c - radius) >= dec_lo``,

        i.e. the centre sits at least one radius from each FoV edge. This
        drops the thin, near-zero-exposure strip at the FoV boundary where the
        windows would otherwise sample directions outside the FoV and the
        exposure-at-centre approximation is least reliable.

        Parameters
        ----------
        radius : float
            Window angular radius in degrees, in ``(0, 180]``.
        nside : int or None, optional
            HEALPix resolution. ``None`` (default) auto-selects the minimal
            covering ``nside`` for ``radius``. An explicit value is used as
            given, with a warning if its pixels are larger than the caps can
            cover (possible gaps).
        observatory : optional
            If given, restrict the grid to windows fully inside the FoV
            derived from ``observatory.latitude`` and ``theta_max_deg``.
            ``None`` (default) keeps the full sky.
        theta_max_deg : float, optional
            Zenith-angle cut defining the FoV, in ``(0, 90]``. Defaults to
            60 deg, matching :class:`~spacetimecorr.exposure.ExposureModel`.
            Only used when ``observatory`` is provided.
        nest : bool, optional
            HEALPix ordering. Defaults to RING (``False``).

        Returns
        -------
        SkyGrid
        """
        hp = cls._healpy()

        radius = float(radius)
        if not (0.0 < radius <= 180.0):
            raise ValueError("radius must be in (0, 180] degrees.")

        if nside is None:
            nside = cls.min_covering_nside(radius)
        else:
            nside = int(nside)
            if not hp.isnsideok(nside, nest=nest):
                raise ValueError(f"nside={nside} is not a valid HEALPix resolution.")
            if hp.max_pixrad(nside) > np.deg2rad(radius):
                warnings.warn(
                    f"radius={radius} deg is smaller than "
                    f"max_pixrad(nside={nside})="
                    f"{np.degrees(hp.max_pixrad(nside)):.3f} deg: caps do not fully "
                    f"cover their pixels, so the tiling may leave gaps.",
                    stacklevel=2,
                )

        ipix = np.arange(hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, ipix, nest=nest, lonlat=True)

        if observatory is not None:
            if not (0.0 < theta_max_deg <= 90.0):
                raise ValueError("theta_max_deg must be in (0, 90].")
            latitude = float(observatory.latitude)
            dec_hi = min(90.0, latitude + theta_max_deg)
            dec_lo = max(-90.0, latitude - theta_max_deg)

            contained = (
                (np.minimum(90.0, dec + radius) <= dec_hi)
                & (np.maximum(-90.0, dec - radius) >= dec_lo)
            )
            ra, dec = ra[contained], dec[contained]
            if ra.size == 0:
                raise ValueError(
                    f"No HEALPix windows of radius {radius} deg fit entirely "
                    f"inside the FoV declination band "
                    f"[{dec_lo:.2f}, {dec_hi:.2f}] deg "
                    f"(latitude={latitude:.2f}, theta_max={theta_max_deg:.2f})."
                )

        centres = np.column_stack((ra, dec))
        return cls(centres=centres, radii=radius)

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
        with one it is additionally weighted by the relative directional
        exposure at each centre, normalised by its sky average ``<omega>``
        (see :attr:`ExposureModel.mean_relative_exposure`) so the counts sum
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
            [exposure_model.relative_exposure(c) for c in self._centres],
            dtype=float,
        )
        weights /= exposure_model.mean_relative_exposure
        return float(n_events) * self.sky_fraction * weights
