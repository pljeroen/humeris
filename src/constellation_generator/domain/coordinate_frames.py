# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Coordinate frame conversions.

Pure mathematical transformations between ECI, ECEF, and Geodetic frames.
No external dependencies — only stdlib math/datetime.

Reference frames:
    ECI  — Earth-Centered Inertial (non-rotating, J2000)
    ECEF — Earth-Centered Earth-Fixed (rotating with Earth)
    Geodetic — Latitude, Longitude, Altitude (WGS84 ellipsoid)

The ECI→ECEF rotation is a simple Z-axis rotation by the Greenwich
Mean Sidereal Time (GMST) angle. ECEF→Geodetic uses the iterative
Bowring method on the WGS84 ellipsoid.
"""
import math
from datetime import datetime, timezone

from constellation_generator.domain.orbital_mechanics import OrbitalConstants


def gmst_rad(epoch: datetime) -> float:
    """
    Compute Greenwich Mean Sidereal Time for a given UTC epoch.

    Uses the IAU formula based on Julian centuries from J2000.0:
        GMST(°) = 280.46061837 + 360.98564736629 * (JD - 2451545.0)
                  + 0.000387933 * T² - T³/38710000

    Args:
        epoch: UTC datetime.

    Returns:
        GMST in radians, normalized to [0, 2π).
    """
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)

    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    delta = epoch - j2000
    jd_since_j2000 = delta.total_seconds() / 86400.0
    t_centuries = jd_since_j2000 / 36525.0

    gmst_deg = (
        280.46061837
        + 360.98564736629 * jd_since_j2000
        + 0.000387933 * t_centuries**2
        - t_centuries**3 / 38710000.0
    )

    gmst_deg = gmst_deg % 360.0
    if gmst_deg < 0:
        gmst_deg += 360.0

    return math.radians(gmst_deg)


def eci_to_ecef(
    pos_eci: tuple[float, float, float],
    vel_eci: tuple[float, float, float],
    gmst_angle_rad: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Convert ECI state vectors to ECEF via Z-axis rotation by GMST.

    The rotation matrix R_z(-θ) rotates from inertial to Earth-fixed:
        [x_ecef]   [ cos(θ)  sin(θ)  0] [x_eci]
        [y_ecef] = [-sin(θ)  cos(θ)  0] [y_eci]
        [z_ecef]   [   0       0     1] [z_eci]

    Args:
        pos_eci: Position in ECI frame (x, y, z) in meters.
        vel_eci: Velocity in ECI frame (vx, vy, vz) in m/s.
        gmst_angle_rad: GMST angle in radians (from gmst_rad()).

    Returns:
        (pos_ecef, vel_ecef) as tuples of (x, y, z) in meters and m/s.
    """
    cos_t = math.cos(gmst_angle_rad)
    sin_t = math.sin(gmst_angle_rad)

    pos_ecef = (
        cos_t * pos_eci[0] + sin_t * pos_eci[1],
        -sin_t * pos_eci[0] + cos_t * pos_eci[1],
        pos_eci[2],
    )

    vel_ecef = (
        cos_t * vel_eci[0] + sin_t * vel_eci[1],
        -sin_t * vel_eci[0] + cos_t * vel_eci[1],
        vel_eci[2],
    )

    return pos_ecef, vel_ecef


def ecef_to_geodetic(
    pos_ecef: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Convert ECEF position to geodetic coordinates (WGS84 ellipsoid).

    Uses the iterative Bowring method for latitude convergence.

    Args:
        pos_ecef: Position in ECEF frame (x, y, z) in meters.

    Returns:
        (latitude_deg, longitude_deg, altitude_m)
        Latitude in [-90, 90], longitude in (-180, 180].
    """
    c = OrbitalConstants
    a = c.R_EARTH_EQUATORIAL
    b = c.R_EARTH_POLAR
    e2 = c.E_SQUARED

    x, y, z = pos_ecef
    p = math.sqrt(x**2 + y**2)

    # Longitude
    lon_rad = math.atan2(y, x)

    # Iterative Bowring method for latitude
    # Initial estimate using spherical approximation
    lat_rad = math.atan2(z, p * (1.0 - e2))

    for _ in range(10):
        sin_lat = math.sin(lat_rad)
        n = a / math.sqrt(1.0 - e2 * sin_lat**2)
        lat_rad = math.atan2(z + e2 * n * sin_lat, p)

    # Altitude
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    n = a / math.sqrt(1.0 - e2 * sin_lat**2)

    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - n
    else:
        alt = abs(z) - b

    lat_deg = math.degrees(lat_rad)
    lon_deg = math.degrees(lon_rad)

    return lat_deg, lon_deg, alt


def geodetic_to_ecef(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
) -> tuple[float, float, float]:
    """
    Convert geodetic coordinates to ECEF position (WGS84 ellipsoid).

    Inverse of ecef_to_geodetic.

    Args:
        lat_deg: Geodetic latitude in degrees [-90, 90].
        lon_deg: Geodetic longitude in degrees.
        alt_m: Altitude above WGS84 ellipsoid in meters.

    Returns:
        (x, y, z) in meters, ECEF frame.
    """
    c = OrbitalConstants
    a = c.R_EARTH_EQUATORIAL
    e2 = c.E_SQUARED

    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    n = a / math.sqrt(1.0 - e2 * sin_lat**2)

    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + alt_m) * sin_lat

    return x, y, z
