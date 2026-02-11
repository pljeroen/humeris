"""
Topocentric observation geometry.

Computes azimuth, elevation, and slant range from a ground station
to a satellite given in ECEF coordinates.

No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass

from constellation_generator.domain.coordinate_frames import geodetic_to_ecef


@dataclass(frozen=True)
class GroundStation:
    """A ground observation station."""
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0


@dataclass(frozen=True)
class Observation:
    """Topocentric observation: azimuth, elevation, slant range."""
    azimuth_deg: float
    elevation_deg: float
    slant_range_m: float


def _ecef_to_enu(
    range_ecef: tuple[float, float, float],
    lat_rad: float,
    lon_rad: float,
) -> tuple[float, float, float]:
    """
    Rotate ECEF range vector to East-North-Up (ENU) frame.

    Args:
        range_ecef: (dx, dy, dz) range vector in ECEF.
        lat_rad: Station geodetic latitude in radians.
        lon_rad: Station geodetic longitude in radians.

    Returns:
        (E, N, U) components in meters.
    """
    dx, dy, dz = range_ecef
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def compute_observation(
    station: GroundStation,
    satellite_ecef: tuple[float, float, float],
) -> Observation:
    """
    Compute topocentric azimuth, elevation, and slant range.

    Args:
        station: Ground station with geodetic coordinates.
        satellite_ecef: Satellite ECEF position (x, y, z) in meters.

    Returns:
        Observation with azimuth [0, 360), elevation [-90, 90],
        and slant range in meters.
    """
    station_ecef = geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)

    range_ecef = (
        satellite_ecef[0] - station_ecef[0],
        satellite_ecef[1] - station_ecef[1],
        satellite_ecef[2] - station_ecef[2],
    )

    lat_rad = math.radians(station.lat_deg)
    lon_rad = math.radians(station.lon_deg)
    e, n, u = _ecef_to_enu(range_ecef, lat_rad, lon_rad)

    slant_range = math.sqrt(e**2 + n**2 + u**2)
    horizontal = math.sqrt(e**2 + n**2)

    elevation_deg = math.degrees(math.atan2(u, horizontal))
    azimuth_deg = math.degrees(math.atan2(e, n)) % 360.0

    return Observation(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        slant_range_m=slant_range,
    )
