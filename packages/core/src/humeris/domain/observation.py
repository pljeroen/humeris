# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Topocentric observation geometry.

Computes azimuth, elevation, and slant range from a ground station
to a satellite given in ECEF coordinates.

"""
import math
from dataclasses import dataclass

import numpy as np

from humeris.domain.coordinate_frames import geodetic_to_ecef


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
    sin_lat = float(np.sin(lat_rad))
    cos_lat = float(np.cos(lat_rad))
    sin_lon = float(np.sin(lon_rad))
    cos_lon = float(np.cos(lon_rad))

    rot = np.array([
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ])
    enu = rot @ np.array([dx, dy, dz])

    return float(enu[0]), float(enu[1]), float(enu[2])


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

    range_vec = np.array(satellite_ecef) - np.array(station_ecef)
    range_ecef = (float(range_vec[0]), float(range_vec[1]), float(range_vec[2]))

    lat_rad = float(np.radians(station.lat_deg))
    lon_rad = float(np.radians(station.lon_deg))
    e, n, u = _ecef_to_enu(range_ecef, lat_rad, lon_rad)

    enu_arr = np.array([e, n, u])
    slant_range = float(np.linalg.norm(enu_arr))
    horizontal = float(np.sqrt(e**2 + n**2))

    elevation_deg = float(np.degrees(np.arctan2(u, horizontal)))
    azimuth_deg = float(np.degrees(np.arctan2(e, n))) % 360.0

    return Observation(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        slant_range_m=slant_range,
    )
