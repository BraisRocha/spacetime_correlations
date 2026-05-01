from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time
from astropy.coordinates import EarthLocation

if TYPE_CHECKING:
    from .skywindow import SkyWindow
    from .exposure import ExposureModel
    from .flare import Flare


class EventSample:
    """Generate and store samples of events in equatorial coordinates.

    Parameters
    ----------
    n_events : int
        Number of events to generate (must be non-negative).
    t0 : astropy.time.Time
        Observation start time.
    tf : astropy.time.Time
        Observation end time (must be later than ``t0``).
    rng : numpy.random.Generator
        Random generator used for reproducible sampling
        (e.g. obtained from ``RNGManager.get(name)``).

    Notes
    -----
    - The constructor automatically samples isotropic ``(RA, Dec)``
      coordinates by calling :meth:`assign_equatorial_coordinates`.
    - All event coordinates are stored in degrees.
    - Optional state (exposure values, flare bookkeeping, sample-type
      labels) is set lazily by the corresponding ``assign_*`` /
      ``inject_flare`` methods and inspected via the ``has_*`` properties.
    """

    # -------------------------------------------------------------------------
    # Construction and basic initialization
    # -------------------------------------------------------------------------

    @staticmethod
    def _healpy():
        return importlib.import_module("healpy")

    @staticmethod
    def _pyplot():
        return importlib.import_module("matplotlib.pyplot")

    def __init__(
        self,
        n_events: int,
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
        *,
        _auto_sample: bool = True,
    ):
        # ---- Input validation ------------------------------------------------
        if not isinstance(n_events, int) or isinstance(n_events, bool):
            raise TypeError("n_events must be an integer.")
        if n_events < 0:
            raise ValueError("n_events must be non-negative.")

        if not isinstance(t0, Time) or not isinstance(tf, Time):
            raise TypeError("t0 and tf must be astropy.time.Time objects.")

        if tf <= t0:
            raise ValueError("tf must be strictly later than t0.")

        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be a numpy.random.Generator. "
                "Obtain one from RNGManager.get(name) and pass it here."
            )

        # ---- Core configuration ----------------------------------------------
        self.rng = rng
        self.n_events = int(n_events)
        self.expected_n = float(n_events)
        self.t0 = t0
        self.tf = tf

        # ---- Sample metadata / state labels ----------------------------------
        self.spatial_type: str | None = None
        self.exposure_type: str | None = None
        self.flare_type: str | None = None

        # ---- Event coordinates (stored in degrees) ---------------------------
        self.RA: np.ndarray | None = None
        self.Dec: np.ndarray | None = None

        # ---- Exposure-related attributes -------------------------------------
        self.expected_exposure_rate: float | None = None
        self.exposure: np.ndarray | None = None

        # ---- Flare bookkeeping -----------------------------------------------
        self.flare_mask: np.ndarray | None = None

        # ---- Optional automatic coordinate generation ------------------------
        if _auto_sample:
            self.assign_equatorial_coordinates()

    @classmethod
    def _from_arrays(
        cls,
        RA: np.ndarray,
        Dec: np.ndarray,
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
        *,
        spatial_type: str | None = None,
        expected_n: float | None = None,
        exposure: np.ndarray | None = None,
        exposure_type: str | None = None,
        expected_exposure_rate: float | None = None,
        flare_mask: np.ndarray | None = None,
        flare_type: str | None = None,
    ) -> "EventSample":
        """
        Create an EventSample from existing arrays without drawing new coordinates.

        Optional metadata and event-level attributes can also be attached, such as
        exposure values and flare bookkeeping.
        """

        RA = np.asarray(RA, dtype=float)
        Dec = np.asarray(Dec, dtype=float)

        if RA.shape != Dec.shape:
            raise ValueError(
                f"RA and Dec must have the same shape, got {RA.shape} vs {Dec.shape}."
            )
        if RA.ndim != 1:
            raise ValueError(f"RA and Dec must be 1D arrays, got ndim={RA.ndim}.")
        
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)
            if exposure.shape != RA.shape:
                raise ValueError(
                    f"exposure must have the same shape as RA/Dec, "
                    f"got {exposure.shape} vs {RA.shape}."
                )
            
        if flare_mask is not None:
            flare_mask = np.asarray(flare_mask, dtype=bool)
            if flare_mask.shape != RA.shape:
                raise ValueError(
                    f"flare_mask must have same shape as RA/Dec, "
                    f"got {flare_mask.shape} vs {RA.shape}."
                )

        obj = cls(
            n_events=int(RA.size),
            t0=t0,
            tf=tf,
            rng=rng,
            _auto_sample=False,
        )

        # Coordinates
        obj.RA = RA
        obj.Dec = Dec
        obj.spatial_type = spatial_type

        # Expected counts
        if expected_n is not None:
            obj.expected_n = float(expected_n)

        # Exposure-related attributes
        obj.exposure = exposure
        obj.exposure_type = exposure_type
        obj.expected_exposure_rate = expected_exposure_rate

        # Flare bookkeeping
        obj.flare_mask = flare_mask
        obj.flare_type = flare_type

        return obj

    # -------------------------------------------------------------------------
    # Basic derived properties and state checks
    # -------------------------------------------------------------------------

    @property
    def T_obs(self) -> Quantity:
        """Observation duration as an astropy Quantity."""
        return (self.tf - self.t0).to(u.s)

    @property
    def expected_temporal_rate(self) -> float:
        """Expected rate of events per unit of time."""
        return float(self.n_events / self.T_obs.to(u.s).value)

    @property
    def has_coordinates(self) -> bool:
        """Return True if coordinates have been assigned."""
        return self.RA is not None and self.Dec is not None

    @property
    def has_exposure(self) -> bool:
        """Return True if exposure values have been assigned."""
        return self.exposure is not None

    @property
    def has_flare(self) -> bool:
        """Return True if flare events have been identified in the sample."""
        return self.flare_mask is not None and np.any(self.flare_mask)

    # -------------------------------------------------------------------------
    # Core sampling and low-level data manipulation
    # -------------------------------------------------------------------------

    def _generate_equatorial_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate an isotropic distribution on the sphere in equatorial coordinates.

        RA is uniform in ``[0, 360)``.
        Dec is distributed so that ``sin(Dec)`` is uniform in ``[-1, 1]``
        (isotropic on the sphere). Coordinates are returned in degrees.
        """
        RA = self.rng.uniform(0.0, 360.0, size=self.n_events)
        u_rand = self.rng.uniform(-1.0, 1.0, size=self.n_events)
        Dec = np.degrees(np.arcsin(u_rand))

        return np.asarray(RA, dtype=float), np.asarray(Dec, dtype=float)

    def assign_equatorial_coordinates(self) -> None:
        """
        Sample isotropic equatorial coordinates and store them on ``self``.

        Sets ``self.RA``, ``self.Dec`` and tags ``self.spatial_type`` as
        ``"equatorial"``.
        """
        RA, Dec = self._generate_equatorial_coordinates()
        self.RA = RA
        self.Dec = Dec
        self.spatial_type = "equatorial"

    # -------------------------------------------------------------------------
    # Public selection and exposure methods
    # -------------------------------------------------------------------------

    def _subset(self, mask: np.ndarray) -> "EventSample":
        """
        Return a new ``EventSample`` containing only events where ``mask`` is True.

        All optional per-event arrays present on ``self`` (``exposure``,
        ``flare_mask``) are sliced consistently. Sample-level metadata
        (``spatial_type``, ``exposure_type``, ``flare_type``,
        ``expected_n``, ``expected_exposure_rate``) is propagated as-is.
        Note that ``expected_n`` is *not* rescaled by the subset fraction
        and should be set explicitly by the caller when its meaning changes
        (e.g. by :meth:`select_subsample`).
        """

        if self.RA is None or self.Dec is None:
            raise ValueError("RA and Dec are not available.")

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.RA.shape:
            raise ValueError(f"Mask must have shape {self.RA.shape}, got {mask.shape}.")

        exposure = None
        if self.exposure is not None:
            exposure = self.exposure[mask]

        flare_mask = None
        if self.flare_mask is not None:
            flare_mask = self.flare_mask[mask]

        return EventSample._from_arrays(
            RA=self.RA[mask],
            Dec=self.Dec[mask],
            t0=self.t0,
            tf=self.tf,
            rng=self.rng,
            spatial_type=self.spatial_type,
            expected_n=self.expected_n,
            exposure=exposure,
            exposure_type=self.exposure_type,
            expected_exposure_rate=self.expected_exposure_rate,
            flare_mask=flare_mask,
            flare_type=self.flare_type,
        )

    def select_subsample(
        self,
        window: SkyWindow,
    ) -> EventSample:
        """
        Return a new ``EventSample`` containing only events inside ``window``.

        The returned sample carries an updated ``expected_n`` set to
        ``window.expected_n_in_window(self.n_events)`` (i.e. the expected
        number of events inside the window under the *uniform full-sky*
        assumption built into :class:`SkyWindow`).

        Parameters
        ----------
        window : SkyWindow
            Spherical-cap window used to define the subset.

        Returns
        -------
        EventSample
            New sample with sliced ``RA``, ``Dec``, and any optional
            per-event arrays.

        Raises
        ------
        ValueError
            If coordinates have not been generated, or if no event lies
            inside the window.
        """
        if not self.has_coordinates:
            raise ValueError("RA and Dec are not available.")

        mask = window.contains(self.RA, self.Dec)

        if not np.any(mask):
            raise ValueError("No events found inside the sky window.")

        subsample = self._subset(mask)
        subsample.expected_n = window.expected_n_in_window(self.n_events)

        return subsample

    def generate_directional_exposure(
        self,
        window: "SkyWindow",
        exposure_model: "ExposureModel",
    ) -> tuple[np.ndarray, np.ndarray, float, str]:
        """
        Generate sampled cumulative directional exposure values for this EventSample,
        using the window definition as the reference direction.

        Intended workflow
        -----------------
        1) Start from a full dataset (parent sample) spanning [t0, tf].
        2) Apply a sky-window selection to build a subsample:
            `subsample = parent.select_subsample(window)`
        3) Call this method on the subsample to generate per-event epsilon values.

        Important notes
        ---------------
        - This method is designed for *window-selected subsamples*. It assumes that the
        sample was obtained via `select_subsample(...)` and does not validate that
        the events lie inside `window`.
        - No event times are generated or required. The method samples values directly
        in cumulative exposure space.
        - If flare events are present, exposure values are generated only for the
        non-flare events.
        - This method does not modify the sample.

        Parameters
        ----------
        window : SkyWindow
            The sky region that defined this subsample.
        exposure_model : ExposureModel
            Model providing the exposure-space sampling machinery.

        Returns
        -------
        eps : np.ndarray
            Sampled exposure values for the selected events.
        target_mask : np.ndarray
            Boolean mask indicating which events the sampled values correspond to.
        expected_exposure_rate : float
            Expected rate used in the exposure sampling.
        method : str
            Identifier of the sampling method used.
        """

        max_exposure = exposure_model.max_directional_exposure(window.centre)
        expected_exposure_rate = self.expected_n/ max_exposure

        if self.has_flare:
            isotropy_mask = ~self.flare_mask
        else:
            isotropy_mask = np.ones(self.n_events, dtype=bool)

        n_target = int(np.count_nonzero(isotropy_mask))

        eps, method = exposure_model.sample_directional_exposure(
            n_events=n_target,
            expected_exposure_rate=expected_exposure_rate,
            max_dir_exposure=max_exposure,
        )

        eps = np.asarray(eps, dtype=float)

        return eps, isotropy_mask, expected_exposure_rate, str(method)

    def assign_directional_exposure(
        self,
        window: "SkyWindow",
        exposure_model: "ExposureModel",
    ) -> None:
        """
        Generate and assign cumulative directional exposure values to this EventSample.

        This method is intended for *window-selected subsamples* and acts as an
        in-place wrapper around `generate_directional_exposure(...)`.

        Important notes
        ---------------
        - If flare events are present, exposure values are generated only for the
        non-flare events, and existing flare exposure values are preserved.
        - If the sample does not yet contain an exposure array, it is initialized with
        `np.nan` values and then populated at the relevant positions.
        - This method modifies the current sample in-place.

        Parameters
        ----------
        window : SkyWindow
            The sky region that defined this subsample.
        exposure_model : ExposureModel
            Model providing the exposure-space sampling machinery.
        """
        eps, target_mask, expected_exposure_rate, method = (
            self.generate_directional_exposure(window, exposure_model)
        )

        if self.exposure is None:
            self.exposure = np.full(self.n_events, np.nan, dtype=float)

        self.exposure[target_mask] = eps
        self.expected_exposure_rate = expected_exposure_rate
        self.exposure_type = method

    # -------------------------------------------------------------------------
    # Public flare manipulation
    # -------------------------------------------------------------------------

    def inject_flare(self, flare: "Flare") -> None:
        """
        Inject a fully-generated flare into the current sample in place.

        ``flare.n_events`` event slots are chosen uniformly at random
        (without replacement) from the existing events; their ``RA``,
        ``Dec`` and ``exposure`` are overwritten by the flare values, and
        a boolean ``flare_mask`` is recorded so that downstream code can
        identify the injected events.

        Parameters
        ----------
        flare : Flare
            Flare with ``RA``, ``Dec`` and ``exposure`` already populated
            (typically via :meth:`Flare.generate_in_window`).

        Raises
        ------
        TypeError
            If ``flare`` is not a :class:`Flare` instance.
        RuntimeError
            If the sample already contains an injected flare.
        ValueError
            If ``self`` has no coordinates, ``flare`` is not fully
            generated, or ``flare.n_events`` exceeds ``self.n_events``.

        Notes
        -----
        - If ``self.exposure`` is ``None`` it is allocated as ``NaN`` and
          only the flare slots are filled. The caller is responsible for
          subsequently calling :meth:`assign_directional_exposure` to fill
          the remaining background slots.
        - Sample size (``self.n_events``) is preserved by construction.
        """
        from .flare import Flare

        if not isinstance(flare, Flare):
            raise TypeError("flare must be an instance of Flare.")

        if self.has_flare:
            raise RuntimeError("This sample already contains an injected flare.")

        if not self.has_coordinates:
            raise ValueError("Sample coordinates are not available.")
        
        if flare.RA is None or flare.Dec is None or flare.exposure is None:
            raise ValueError(
                "Flare is not fully generated. "
                "Coordinates and exposure must be computed before injection."
            )

        if flare.n_events > self.n_events:
            raise ValueError(
                "Cannot inject flare: flare has more events than the sample."
            )

        idx = self.rng.choice(self.n_events, size=flare.n_events, replace=False)

        self.RA[idx] = flare.RA
        self.Dec[idx] = flare.Dec

        # Check if an exposure array already exists
        if self.exposure is None:
            self.exposure = np.full(self.n_events, np.nan, dtype=float)

        self.exposure[idx] = flare.exposure

        self.flare_mask = np.zeros(self.n_events, dtype=bool)
        self.flare_mask[idx] = True
        self.flare_type = flare.flare_type

    # -------------------------------------------------------------------------
    # Public skymap / visualization interface
    # -------------------------------------------------------------------------

    def get_healpix_skymap(
        self,
        nside: int = 32,
        *,
        mask_fov: bool = False,
        location: EarthLocation | None = None,
        zenith_max: u.Quantity | None = None,
    ) -> np.ndarray:
        """
        Build a HEALPix counts map from the sample event coordinates.

        Parameters
        ----------
        nside : int, optional
            HEALPix NSIDE parameter. Must be a valid HEALPix value.
        mask_fov : bool, optional
            If True, mask pixels outside the declination band visible from
            the observatory defined by ``location`` and ``zenith_max``.
        location : astropy.coordinates.EarthLocation, optional
            Observatory location. Required if ``mask_fov=True``.
        zenith_max : astropy.units.Quantity, optional
            Maximum zenith angle. Required if ``mask_fov=True``.

        Returns
        -------
        skymap : numpy.ndarray or numpy.ma.MaskedArray
            HEALPix map of event counts. If ``mask_fov=True``, a masked map
            is returned.

        Notes
        -----
        This method assumes ``self.RA`` and ``self.Dec`` are stored in degrees.
        """
        if self.RA is None or self.Dec is None:
            raise ValueError(
                "RA and Dec are not available. "
                "Sample coordinates before building the skymap."
            )

        hp = self._healpy()

        if not hp.isnsideok(nside):
            raise ValueError("nside must be a valid HEALPix NSIDE value.")

        ra_deg = np.asarray(self.RA, dtype=float)
        dec_deg = np.asarray(self.Dec, dtype=float)

        if ra_deg.shape != dec_deg.shape:
            raise ValueError("RA and Dec must have the same shape.")

        valid = np.isfinite(ra_deg) & np.isfinite(dec_deg)
        ra_deg = ra_deg[valid]
        dec_deg = dec_deg[valid]

        ra_rad = np.deg2rad(np.mod(ra_deg, 360.0))
        dec_rad = np.deg2rad(dec_deg)

        npix = hp.nside2npix(nside)
        skymap = np.zeros(npix, dtype=float)

        theta = 0.5 * np.pi - dec_rad
        phi = ra_rad

        pix = hp.ang2pix(nside, theta, phi)
        np.add.at(skymap, pix, 1)

        if mask_fov:
            if location is None or zenith_max is None:
                raise ValueError(
                    "location and zenith_max must be provided when mask_fov=True."
                )
            skymap = self._mask_skymap_outside_fov(
                skymap=skymap,
                nside=nside,
                location=location,
                zenith_max=zenith_max,
            )

        return skymap

    def plot_skymap(
        self,
        nside: int = 32,
        *,
        mask_fov: bool = False,
        location: EarthLocation | None = None,
        zenith_max: u.Quantity | None = None,
        title: str = "Sky map",
        cmap: str = "magma",
        output_file: str | None = None,
        astronomical: bool = True,
        show: bool = True,
        xticks_deg: np.ndarray | list | None = None,
        yticks_deg: np.ndarray | list | None = None,
    ):
        """
        Plot the event sample as a HEALPix-binned sky map in Hammer projection.

        Parameters
        ----------
        nside : int, optional
            HEALPix resolution parameter.
        mask_fov : bool, optional
            If True, mask pixels outside the observatory declination band
            defined by ``location`` and ``zenith_max``.
        location : astropy.coordinates.EarthLocation, optional
            Required when ``mask_fov=True``.
        zenith_max : astropy.units.Quantity, optional
            Required when ``mask_fov=True``.
        title : str, optional
            Figure title.
        cmap : str, optional
            Matplotlib colormap name.
        output_file : str or None, optional
            If given, the figure is saved at this path with ``dpi=300``.
        astronomical : bool, optional
            If True (default), display the sky in the astronomical
            convention (RA increases to the left).
        show : bool, optional
            If True, call ``plt.show()`` after drawing.
        xticks_deg, yticks_deg : array-like or None, optional
            Override the default tick locations (in degrees) on the
            longitude/latitude axes.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
            Created figure and axes objects.
        """
        skymap = self.get_healpix_skymap(
            nside=nside,
            mask_fov=mask_fov,
            location=location,
            zenith_max=zenith_max,
        )
        plt = self._pyplot()

        lon_edges, lat_edges, image = self._healpix_to_lonlat_image(
            skymap=skymap,
            nside=nside,
            astronomical=astronomical,
        )

        fig, ax = plt.subplots(
            figsize=(8, 4.8),
            subplot_kw={"projection": "hammer"},
        )

        vmin = np.nanmin(image)
        vmax = np.nanmax(image)

        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            image,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title)
        ax.grid(True, alpha=0.6)
        ax.set_facecolor("lightgrey")

        cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08)
        cbar.set_label("Number of events")

        if xticks_deg is None:
            xticks_deg = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
        else:
            xticks_deg = np.asarray(xticks_deg, dtype=float)
        ax.set_xticks(np.deg2rad(xticks_deg))

        if astronomical:
            ax.set_xticklabels([f"{(-x) % 360:.0f}°" for x in xticks_deg])
            ax.set_xlabel(r"Right ascension $\alpha$")
        else:
            ax.set_xticklabels([f"{x:.0f}°" for x in xticks_deg])
            ax.set_xlabel("Longitude")

        if yticks_deg is None:
            yticks_deg = np.array([-60, -30, 0, 30, 60])
        else:
            yticks_deg = np.asarray(yticks_deg, dtype=float)
        ax.set_yticks(np.deg2rad(yticks_deg))
        ax.set_yticklabels([f"{y:.0f}°" for y in yticks_deg])
        ax.set_ylabel(r"Declination $\delta$")

        fig.tight_layout()

        if output_file is not None:
            fig.savefig(output_file, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax

    # -------------------------------------------------------------------------
    # Internal skymap helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _visible_declination_band(
        location: EarthLocation,
        zenith_max: u.Quantity,
    ) -> tuple[float, float]:
        """
        Compute the accessible declination band for an observatory.

        Returns
        -------
        dec_min, dec_max : tuple of float
            Minimum and maximum visible declinations in radians.
        """
        if not isinstance(location, EarthLocation):
            raise TypeError("location must be an astropy.coordinates.EarthLocation.")

        if not isinstance(zenith_max, u.Quantity):
            raise TypeError("zenith_max must be an astropy.units.Quantity.")

        zenith_max_rad = zenith_max.to_value(u.rad)
        lat_rad = location.lat.to_value(u.rad)

        dec_min = max(-0.5 * np.pi, lat_rad - zenith_max_rad)
        dec_max = min(+0.5 * np.pi, lat_rad + zenith_max_rad)

        return dec_min, dec_max

    @classmethod
    def _mask_skymap_outside_fov(
        cls,
        skymap: np.ndarray,
        nside: int,
        *,
        location: EarthLocation,
        zenith_max: u.Quantity,
    ):
        """
        Mask HEALPix pixels outside the observatory declination band.
        """
        dec_min, dec_max = cls._visible_declination_band(location, zenith_max)
        hp = cls._healpy()

        masked = np.array(skymap, copy=True)
        npix = hp.nside2npix(nside)
        ipix = np.arange(npix)

        theta, _phi = hp.pix2ang(nside, ipix)
        dec = 0.5 * np.pi - theta

        outside = (dec < dec_min) | (dec > dec_max)
        masked[outside] = hp.UNSEEN

        return hp.ma(masked)

    @staticmethod
    def _healpix_to_lonlat_image(
        skymap,
        nside: int,
        *,
        nx: int = 361,
        ny: int = 181,
        astronomical: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample a HEALPix map onto a regular lon/lat grid for pcolormesh.

        Parameters
        ----------
        skymap : numpy.ndarray or numpy.ma.MaskedArray
            Input HEALPix map.
        nside : int
            HEALPix NSIDE parameter.
        nx, ny : int, optional
            Number of longitude/latitude grid edges.
        astronomical : bool, optional
            If True, use the standard astronomical convention in which
            right ascension increases to the left.

        Returns
        -------
        lon_edges, lat_edges, image : tuple of numpy.ndarray
            Grid edges in radians and the gridded image.
        """
        hp = EventSample._healpy()
        lon_edges = np.linspace(-np.pi, np.pi, nx)
        lat_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, ny)

        lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])

        lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)

        if astronomical:
            phi = np.mod(-lon2d, 2.0 * np.pi)
        else:
            phi = np.mod(lon2d, 2.0 * np.pi)

        theta = 0.5 * np.pi - lat2d
        pix = hp.ang2pix(nside, theta, phi)

        image = np.asarray(skymap[pix], dtype=float)

        if np.ma.isMaskedArray(skymap):
            image = np.where(np.isfinite(image), image, np.nan)

        return lon_edges, lat_edges, image
    
