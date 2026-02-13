# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Sensor/payload FOV geometry, ground footprint, and sensor-constrained coverage.

Computes sensor field-of-view geometry, swath width with Earth curvature
correction, nadir footprint, and sensor-constrained coverage analysis.

No external dependencies — only stdlib math/dataclasses/enum/datetime
+ domain imports.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from humeris.domain.coverage import CoveragePoint
from humeris.domain.propagation import OrbitalState, propagate_to


# --- Types ---

class SensorType(Enum):
    CIRCULAR = "circular"
    RECTANGULAR = "rectangular"


@dataclass(frozen=True)
class SensorConfig:
    """Sensor configuration for FOV geometry."""
    sensor_type: SensorType
    half_angle_deg: float
    cross_track_half_angle_deg: float = 0.0
    along_track_half_angle_deg: float = 0.0


@dataclass(frozen=True)
class GroundFootprint:
    """Ground footprint of a nadir-pointing sensor."""
    center_lat_deg: float
    center_lon_deg: float
    swath_width_km: float
    along_track_extent_km: float


@dataclass(frozen=True)
class SensorAccessResult:
    """Result of sensor FOV visibility check."""
    is_visible: bool
    off_nadir_angle_deg: float
    ground_range_km: float


_R_EARTH_KM = OrbitalConstants.R_EARTH / 1000.0


# --- Functions ---

def compute_swath_width(altitude_km: float, half_angle_deg: float) -> float:
    """Swath width with Earth curvature correction.

    Geometry: satellite at altitude h, half-cone angle alpha.
    eta = asin((R_e + h) / R_e * sin(alpha))
    lambda = eta - alpha
    swath = 2 * R_e * lambda

    Args:
        altitude_km: Satellite altitude in km (must be > 0).
        half_angle_deg: Sensor half-angle in degrees (must be in (0, 90)).

    Returns:
        Swath width in km.

    Raises:
        ValueError: If altitude_km <= 0 or half_angle_deg not in (0, 90).
    """
    if altitude_km <= 0:
        raise ValueError(f"altitude_km must be positive, got {altitude_km}")
    if half_angle_deg <= 0 or half_angle_deg >= 90:
        raise ValueError(f"half_angle_deg must be in (0, 90), got {half_angle_deg}")

    r_e = _R_EARTH_KM
    h = altitude_km
    alpha = math.radians(half_angle_deg)

    sin_eta = (r_e + h) / r_e * math.sin(alpha)
    if sin_eta >= 1.0:
        # FOV extends beyond Earth horizon
        sin_eta = 1.0
    eta = math.asin(sin_eta)
    lam = eta - alpha
    return 2.0 * r_e * lam


def compute_nadir_footprint(
    altitude_km: float,
    sensor: SensorConfig,
    center_lat_deg: float = 0.0,
    center_lon_deg: float = 0.0,
) -> GroundFootprint:
    """Nadir-pointing footprint.

    CIRCULAR: symmetric (swath == along_track).
    RECTANGULAR: asymmetric using cross-track and along-track half-angles.

    Args:
        altitude_km: Satellite altitude in km.
        sensor: SensorConfig.
        center_lat_deg: Sub-satellite latitude.
        center_lon_deg: Sub-satellite longitude.

    Returns:
        GroundFootprint with swath and along-track dimensions.

    Raises:
        ValueError: If altitude_km <= 0.
    """
    if altitude_km <= 0:
        raise ValueError(f"altitude_km must be positive, got {altitude_km}")

    if sensor.sensor_type == SensorType.CIRCULAR:
        sw = compute_swath_width(altitude_km, sensor.half_angle_deg)
        return GroundFootprint(
            center_lat_deg=center_lat_deg,
            center_lon_deg=center_lon_deg,
            swath_width_km=sw,
            along_track_extent_km=sw,
        )
    else:
        sw = compute_swath_width(altitude_km, sensor.cross_track_half_angle_deg)
        at = compute_swath_width(altitude_km, sensor.along_track_half_angle_deg)
        return GroundFootprint(
            center_lat_deg=center_lat_deg,
            center_lon_deg=center_lon_deg,
            swath_width_km=sw,
            along_track_extent_km=at,
        )


def is_in_sensor_fov(
    sat_position_eci: tuple[float, float, float],
    ground_point_ecef: tuple[float, float, float],
    sensor: SensorConfig,
    gmst_angle_rad: float,
) -> SensorAccessResult:
    """Check if ground point is in nadir-pointing sensor FOV.

    Computes off-nadir angle and compares to half-angle.
    CIRCULAR: simple cone check.
    RECTANGULAR: uses max of cross/along-track half-angles as cone check.

    Args:
        sat_position_eci: Satellite ECI position (meters).
        ground_point_ecef: Ground point ECEF position (meters).
        sensor: SensorConfig.
        gmst_angle_rad: GMST angle (radians).

    Returns:
        SensorAccessResult with visibility, off-nadir angle, ground range.
    """
    # Convert satellite to ECEF
    sat_ecef, _ = eci_to_ecef(sat_position_eci, (0.0, 0.0, 0.0), gmst_angle_rad)

    sat_vec = np.array(sat_ecef)
    sat_mag = float(np.linalg.norm(sat_vec))

    # Earth obscuration check: ground point must be visible from satellite
    # (satellite must be above horizon at the ground point).
    # Condition: cos(angle between ground and sat vectors) > R_e / sat_mag
    ground_vec = np.array(ground_point_ecef)
    ground_mag = float(np.linalg.norm(ground_vec))
    if ground_mag < 1e-10 or sat_mag < 1e-10:
        return SensorAccessResult(is_visible=False, off_nadir_angle_deg=180.0, ground_range_km=0.0)

    dot_gs = float(np.dot(ground_vec, sat_vec))
    cos_angle_gs = dot_gs / (ground_mag * sat_mag)
    r_earth_m = OrbitalConstants.R_EARTH
    horizon_cos = r_earth_m / sat_mag

    if cos_angle_gs < horizon_cos:
        # Ground point not visible (behind Earth horizon)
        # Compute approximate off-nadir for reporting
        nadir = -sat_vec / sat_mag
        to_g = ground_vec - sat_vec
        to_g_mag = float(np.linalg.norm(to_g))
        if to_g_mag > 1e-10:
            cos_on = float(np.dot(to_g, nadir)) / to_g_mag
            cos_on = max(-1.0, min(1.0, cos_on))
            off_nadir_deg = math.degrees(math.acos(cos_on))
        else:
            off_nadir_deg = 0.0
        ground_range_km = math.acos(max(-1.0, min(1.0, cos_angle_gs))) * _R_EARTH_KM
        return SensorAccessResult(is_visible=False, off_nadir_angle_deg=off_nadir_deg, ground_range_km=ground_range_km)

    # Nadir direction (toward Earth center)
    nadir = -sat_vec / sat_mag

    # Vector from satellite to ground point
    to_ground = ground_vec - sat_vec

    to_ground_mag = float(np.linalg.norm(to_ground))
    if to_ground_mag < 1e-10:
        return SensorAccessResult(is_visible=True, off_nadir_angle_deg=0.0, ground_range_km=0.0)

    # Off-nadir angle
    cos_off_nadir = float(np.dot(to_ground, nadir)) / to_ground_mag
    cos_off_nadir = max(-1.0, min(1.0, cos_off_nadir))
    off_nadir_rad = math.acos(cos_off_nadir)
    off_nadir_deg = math.degrees(off_nadir_rad)

    # Ground range (Earth central angle between sub-satellite and ground point)
    ground_range_km = math.acos(max(-1.0, min(1.0, cos_angle_gs))) * _R_EARTH_KM

    # FOV check
    if sensor.sensor_type == SensorType.CIRCULAR:
        is_visible = off_nadir_deg <= sensor.half_angle_deg
    else:
        # Use max of cross/along-track as bounding cone
        max_half = max(sensor.cross_track_half_angle_deg, sensor.along_track_half_angle_deg)
        is_visible = off_nadir_deg <= max_half

    return SensorAccessResult(
        is_visible=is_visible,
        off_nadir_angle_deg=off_nadir_deg,
        ground_range_km=ground_range_km,
    )


def compute_sensor_coverage(
    orbital_states: list[OrbitalState],
    analysis_time: datetime,
    sensor: SensorConfig,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
) -> list[CoveragePoint]:
    """Sensor-constrained coverage snapshot.

    Like compute_coverage_snapshot but uses sensor FOV instead of
    elevation-based visibility.

    Args:
        orbital_states: List of OrbitalState objects.
        analysis_time: UTC time for snapshot.
        sensor: SensorConfig for FOV check.
        lat_step_deg: Latitude grid spacing.
        lon_step_deg: Longitude grid spacing.

    Returns:
        List of CoveragePoint for the grid.
    """
    gmst = gmst_rad(analysis_time)

    # Pre-compute satellite ECI positions
    sat_ecis: list[tuple[float, float, float]] = []
    for state in orbital_states:
        pos_eci, _ = propagate_to(state, analysis_time)
        sat_ecis.append((pos_eci[0], pos_eci[1], pos_eci[2]))

    grid: list[CoveragePoint] = []
    lat = -90.0
    while lat <= 90.0 + 1e-9:
        lon = -180.0
        while lon <= 180.0 - lon_step_deg + 1e-9:
            ground_ecef = geodetic_to_ecef(lat, lon, 0.0)
            count = 0
            for sat_eci in sat_ecis:
                result = is_in_sensor_fov(sat_eci, ground_ecef, sensor, gmst)
                if result.is_visible:
                    count += 1
            grid.append(CoveragePoint(lat_deg=lat, lon_deg=lon, visible_count=count))
            lon += lon_step_deg
        lat += lat_step_deg

    return grid
