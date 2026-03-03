from astropy.coordinates import EarthLocation
import astropy.units as u
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
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
    """

    latitude: float
    longitude: float
    altitude: float
    location: EarthLocation = field(init=False, repr=False)

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
