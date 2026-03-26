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
    auto_sample : bool, optional
        If True (default), coordinates are sampled at construction time.
        Set to False when constructing from pre-existing arrays to avoid
        unnecessary random draws.

    Notes
    -----
    - Default coordinates are generated using
      :meth:`sample_equatorial_coordinates`.
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
        auto_sample: bool = True,
    ):
        # ---- Input validation ------------------------------------------------
        if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events < 0:
            raise TypeError("n_events must be a non-negative integer.")

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
        self.t0 = t0
        self.tf = tf

        # ---- Event properties ------------------------------------------------
        self.spatial_type: str | None = None

        # Event coordinates (stored in degrees)
        self.RA: np.ndarray | None = None
        self.Dec: np.ndarray | None = None
        if auto_sample:
            self.sample_equatorial_coordinates()

        # ---- Attributes populated after window selection ---------------------
        self.expected_counts: float | None = None

        # ---- Attributes populated after exposure model selection -------------
        self.exp_rate_exposure: float | None = None
        self.dir_exposure: np.ndarray | None = None
        self.dir_exposure_method: str | None = None

    @classmethod
    def _from_arrays(
        cls,
        RA: np.ndarray,
        Dec: np.ndarray,
        t0: Time,
        tf: Time,
        rng: np.random.Generator,
        spatial_type: str | None = "equatorial",
    ) -> "EventSample":
        """
        Create an EventSample from existing RA/Dec arrays (no new random draws).
        """

        RA = np.asarray(RA, dtype=float)
        Dec = np.asarray(Dec, dtype=float)

        if RA.shape != Dec.shape:
            raise ValueError(
                f"RA and Dec must have the same shape, got {RA.shape} vs {Dec.shape}."
            )
        if RA.ndim != 1:
            raise ValueError(f"RA and Dec must be 1D arrays, got ndim={RA.ndim}.")

        obj = cls(
            n_events=int(RA.size),
            t0=t0,
            tf=tf,
            rng=rng,
            auto_sample=False,
        )
        obj.RA = RA
        obj.Dec = Dec
        obj.spatial_type = spatial_type

        return obj

    # -------------------------------------------------------------------------
    # Basic derived properties and state checks
    # -------------------------------------------------------------------------

    @property
    def T_obs(self) -> Quantity:
        """
        Observation duration as an astropy Quantity.
        """
        return (self.tf - self.t0).to(u.s)

    @property
    def exp_rate_time(self) -> float:
        """
        Expected rate of events per unit of time.
        """
        return float(self.n_events / self.T_obs.to(u.s).value)

    @property
    def is_populated(self) -> bool:
        """Return True if RA/Dec have been generated/assigned."""
        return self.RA is not None and self.Dec is not None

    @property
    def has_exposure(self) -> bool:
        """
        Return True if directional exposure has been generated/assigned.
        """
        return getattr(self, "dir_exposure", None) is not None

    @property
    def has_flare(self) -> bool:
        """
        Return True if a flare has been already introduced
        into the sample.
        """
        return getattr(self, "flare_type", None) is not None

    # -------------------------------------------------------------------------
    # Core sampling and low-level data manipulation
    # -------------------------------------------------------------------------

    def sample_equatorial_coordinates(self) -> None:
        """
        Simulate an isotropic distribution on the sphere in equatorial coordinates.

        RA is uniform in [0, 360).
        Dec is distributed such that sin(Dec) is uniform in [-1, 1]
        (isotropic on the sphere).

        Notes
        -----
        Coordinates are stored in degrees.
        """
        RA = self.rng.uniform(0.0, 360.0, size=self.n_events)
        u_rand = self.rng.uniform(-1.0, 1.0, size=self.n_events)
        Dec = np.degrees(np.arcsin(u_rand))

        self.RA = np.asarray(RA, dtype=float)
        self.Dec = np.asarray(Dec, dtype=float)
        self.spatial_type = "equatorial"

    def _subset(self, mask: np.ndarray) -> "EventSample":
        """
        Return a new EventSample containing only events where mask is True.
        """

        if self.RA is None or self.Dec is None:
            raise ValueError("RA and Dec are not available.")

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.RA.shape:
            raise ValueError(f"Mask must have shape {self.RA.shape}, got {mask.shape}.")

        return EventSample._from_arrays(
            RA=self.RA[mask],
            Dec=self.Dec[mask],
            t0=self.t0,
            tf=self.tf,
            rng=self.rng,
            spatial_type=self.spatial_type,
        )

    # -------------------------------------------------------------------------
    # Public selection and exposure methods
    # -------------------------------------------------------------------------

    def select_subsample(
        self,
        window: SkyWindow,
    ) -> EventSample:
        """
        Return a new ``EventSample`` containing only the events within the window.

        The returned sample includes an additional attribute, ``expected_counts``,
        representing the expected number of events inside the window.
        """
        if self.RA is None or self.Dec is None:
            raise ValueError("RA and Dec are not available.")

        mask = window.contains(self.RA, self.Dec)

        if not np.any(mask):
            raise ValueError("No events found inside the sky window.")

        subsample = self._subset(mask)
        subsample.expected_counts = window.expected_counts_in_window(self)

        return subsample

    def add_directional_exposure(
        self,
        window: "SkyWindow",
        exposure_model: "ExposureModel",
    ) -> None:
        """
        Attach sampled cumulative directional exposure values to this EventSample,
        using the window definition as the reference direction.

        Intended workflow
        -----------------
        1) Start from a full dataset (parent sample) spanning [t0, tf].
        2) Apply a sky-window selection to build a subsample:
            `subsample = parent.select_subsample(window)`
        3) Call this method on the subsample `sub` to generate per-event epsilon values:
            `subsample.add_directional_exposure(...)`

        Important notes
        ---------------
        - This method is designed for *window-selected subsamples*.
        It does not validate that the events actually lie inside `window`; it assumes
        you already applied the window cut via `subset(...)`.
        - No event times are generated or required. The method samples values directly
        in cumulative exposure space.

        Parameters
        ----------
        window : SkyWindow
            The sky region that defined this subsample.
        exposure_model : ExposureModel
            Model providing the exposure-space sampling machinery.
        """
        
        if self.expected_counts is None:
            raise ValueError(
                "expected_counts is not set. "
                "Build this sample through a sky-window selection first."
            )

        max_dir_exposure = exposure_model.max_directional_exposure(window.centre)
        self.exp_rate_exposure = self.expected_counts / max_dir_exposure

        eps, method = exposure_model.sample_directional_exposure(
            n_events=self.n_events,
            exp_rate_exposure=self.exp_rate_exposure,
            max_dir_exposure=max_dir_exposure,
        )

        self.dir_exposure = np.asarray(eps, dtype=float)
        self.dir_exposure_method = str(method)

    # -------------------------------------------------------------------------
    # Public flare manipulation
    # -------------------------------------------------------------------------

    def inject_flare(self, flare: "Flare") -> None:
        """
        Inject a flare into the current sample by replacing random events.

        The injection replaces ``flare.n_events`` randomly selected events in
        the sample with the flare events.

        Parameters
        ----------
        flare : Flare
            Flare object containing RA, Dec, and directional exposure arrays.
        """
        from .flare import Flare

        if not isinstance(flare, Flare):
            raise TypeError("flare must be an instance of Flare.")

        if self.has_flare:
            raise RuntimeError("This sample already contains an injected flare.")

        if self.RA is None or self.Dec is None:
            raise ValueError("Sample coordinates are not available.")

        if self.dir_exposure is None:
            raise ValueError(
                "Sample directional exposure is not available. "
                "Generate or assign it before injecting a flare."
            )

        if flare.RA is None or flare.Dec is None or flare.dir_exposure is None:
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
        self.dir_exposure[idx] = flare.dir_exposure

        self.flare_type = flare.flare_type
        self.flare_indices = idx

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
        cmap: str = "viridis",
        output_file: str | None = None,
        astronomical: bool = True,
        show: bool = True,
    ):
        """
        Plot the event sample as a HEALPix-binned sky map in Hammer projection.
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

        xticks_deg = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
        ax.set_xticks(np.deg2rad(xticks_deg))

        if astronomical:
            ax.set_xticklabels([f"{(-x) % 360:.0f}°" for x in xticks_deg])
            ax.set_xlabel(r"Right ascension $\alpha$")
        else:
            ax.set_xticklabels([f"{x:.0f}°" for x in xticks_deg])
            ax.set_xlabel("Longitude")

        yticks_deg = np.array([-60, -30, 0, 30, 60])
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
    
