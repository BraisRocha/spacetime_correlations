"""
Ground-based observatory location.

Defines :class:`Observatory`, a frozen dataclass holding the geodetic
position (latitude, longitude, altitude) of the detector and caching
the corresponding :class:`astropy.coordinates.EarthLocation`. This is
the single source of truth for the observatory geometry consumed by
:class:`~spacetimecorr.exposure.ExposureModel` when computing zenith
angles and directional exposure.

Latitudes are stored in degrees in ``[-90, 90]``, longitudes in degrees
in ``[-180, 180]``, altitude in metres (non-negative). The area of the
array, in km**2, is either a constant or the active area A(t) as steps.
Instances are immutable.
"""

from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)  # NOTE: no slots=True (requires Python >= 3.10)
class Observatory:
    """
    Define the position of an observatory.

    Parameters
    ----------
    latitude : float
        Latitude in degrees, must be in [-90, 90].
    longitude : float
        Longitude in degrees, must be in [-180, 180].
    altitude : float
        Altitude in meters (non-negative).
    area : float or (Time, array)
        Area of the array in km**2. Either a constant (positive), or the
        active area A(t) as steps, e.g. from
        :func:`~spacetimecorr.io.load_active_area`: the start time of each
        step and A over it (non-negative). Each value holds until the next
        start time.

    Attributes
    ----------
    max_area : float
        Largest area in km**2: the constant, or the maximum of A(t).
    """

    latitude: float
    longitude: float
    altitude: float
    area: float | tuple[Time, np.ndarray]
    location: EarthLocation = field(init=False, repr=False)
    max_area: float = field(init=False)

    def __post_init__(self):

        # Type + range validation
        if not isinstance(self.latitude, (int, float)) or isinstance(self.latitude, bool):
            raise TypeError("Latitude must be a numeric value in degrees.")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("Latitude must be between -90 and 90 degrees.")

        if not isinstance(self.longitude, (int, float)) or isinstance(self.longitude, bool):
            raise TypeError("Longitude must be a numeric value in degrees.")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("Longitude must be between -180 and 180 degrees.")

        if not isinstance(self.altitude, (int, float)) or isinstance(self.altitude, bool):
            raise TypeError("Altitude must be a numeric value in meters.")
        if self.altitude < 0:
            raise ValueError("Altitude must be non-negative.")

        if isinstance(self.area, tuple):
            object.__setattr__(self, "area", self._check_active_area(*self.area))
            max_area = float(np.max(self.area[1]))
        else:
            if not isinstance(self.area, (int, float)) or isinstance(self.area, bool):
                raise TypeError("Area must be a numeric value in km**2 or (Time, array) steps.")
            if self.area <= 0:
                raise ValueError("Area must be positive.")
            object.__setattr__(self, "area", float(self.area))
            max_area = self.area
        object.__setattr__(self, "max_area", max_area)

        # Enforce internal float consistency
        object.__setattr__(self, "latitude", float(self.latitude))
        object.__setattr__(self, "longitude", float(self.longitude))
        object.__setattr__(self, "altitude", float(self.altitude))

        # Create EarthLocation once (cached geometry)
        location = EarthLocation(
            lat=self.latitude * u.deg,
            lon=self.longitude * u.deg,
            height=self.altitude * u.m
        )

        object.__setattr__(self, "location", location)

    @staticmethod
    def _check_active_area(start_times: Time, area: np.ndarray) -> tuple[Time, np.ndarray]:
        """Validate the steps of A(t) and return them as (Time, float array)."""
        if not isinstance(start_times, Time):
            raise TypeError("Active-area start times must be an astropy.time.Time array.")
        area = np.asarray(area, dtype=float)
        if area.ndim != 1 or start_times.shape != area.shape:
            raise ValueError("Active-area times and values must be 1D arrays of the same length.")
        if np.any(area < 0.0) or not np.any(area > 0.0):
            raise ValueError("Active-area values must be non-negative, and not all zero.")
        if np.any(np.diff((start_times - start_times[0]).to_value("sec")) <= 0.0):
            raise ValueError("Active-area start times must be strictly increasing.")
        return start_times, area
