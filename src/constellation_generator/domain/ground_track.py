"""Ground track computation for circular orbits.

Propagates a satellite's position over time using Keplerian two-body
mechanics (optionally with J2 secular perturbations) and converts to
geodetic coordinates (lat/lon/alt) via the ECI -> ECEF -> Geodetic
pipeline. Includes ascending node detection and ground track crossing
(self-intersection) analysis.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.propagation import (
    derive_orbital_state,
    propagate_to,
)
from constellation_generator.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)
from constellation_generator.domain.numerical_propagation import PropagationStep


@dataclass(frozen=True)
class GroundTrackPoint:
    """A single point on a satellite's ground track."""
    time: datetime
    lat_deg: float
    lon_deg: float
    alt_km: float


def compute_ground_track(
    satellite,
    start: datetime,
    duration: timedelta,
    step: timedelta,
    include_j2: bool = False,
) -> list[GroundTrackPoint]:
    """
    Compute the ground track of a satellite over a time interval.

    Uses Keplerian two-body propagation for circular orbits, optionally
    with J2 secular perturbations for RAAN and argument of perigee drift.

    Args:
        satellite: Satellite domain object with position_eci, velocity_eci,
            raan_deg, true_anomaly_deg, and optional epoch.
        start: UTC datetime for the first ground track point.
        duration: Total time span to compute.
        step: Time between consecutive points.
        include_j2: If True, apply J2 secular perturbations.

    Returns:
        List of GroundTrackPoint objects from start to start+duration.

    Raises:
        ValueError: If step is zero or negative.
    """
    step_seconds = step.total_seconds()
    if step_seconds <= 0:
        raise ValueError(f"Step must be positive, got {step}")

    duration_seconds = duration.total_seconds()
    state = derive_orbital_state(satellite, start, include_j2=include_j2)

    points: list[GroundTrackPoint] = []
    elapsed = 0.0

    while elapsed <= duration_seconds + 1e-9:
        current_time = start + timedelta(seconds=elapsed)
        pos_eci, vel_eci = propagate_to(state, current_time)

        gmst_angle = gmst_rad(current_time)
        pos_ecef, _ = eci_to_ecef(
            (pos_eci[0], pos_eci[1], pos_eci[2]),
            (vel_eci[0], vel_eci[1], vel_eci[2]),
            gmst_angle,
        )

        lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)

        points.append(GroundTrackPoint(
            time=current_time,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            alt_km=alt_m / 1000.0,
        ))

        elapsed += step_seconds

    return points


def compute_ground_track_numerical(
    steps: tuple[PropagationStep, ...],
) -> list[GroundTrackPoint]:
    """Convert numerical propagation steps to ground track points.

    Pipeline per step: gmst_rad -> eci_to_ecef -> ecef_to_geodetic -> GroundTrackPoint.

    Args:
        steps: Tuple of PropagationStep from numerical propagation.

    Returns:
        List of GroundTrackPoint, one per step.
    """
    points: list[GroundTrackPoint] = []

    for step in steps:
        gmst_angle = gmst_rad(step.time)
        pos_ecef, _ = eci_to_ecef(
            step.position_eci,
            step.velocity_eci,
            gmst_angle,
        )
        lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)
        points.append(GroundTrackPoint(
            time=step.time,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            alt_km=alt_m / 1000.0,
        ))

    return points


@dataclass(frozen=True)
class AscendingNodePass:
    """Equator crossing where satellite moves south-to-north."""
    time: datetime
    longitude_deg: float


@dataclass(frozen=True)
class GroundTrackCrossing:
    """Where ascending and descending ground track segments intersect."""
    lat_deg: float
    lon_deg: float
    time_ascending: datetime
    time_descending: datetime


def find_ascending_nodes(track: list[GroundTrackPoint]) -> list[AscendingNodePass]:
    """Find equator crossings where latitude goes negative to positive.

    Uses linear interpolation for exact crossing time and longitude.

    Args:
        track: List of GroundTrackPoint from compute_ground_track.

    Returns:
        List of AscendingNodePass at each equator crossing.
    """
    nodes: list[AscendingNodePass] = []
    for i in range(len(track) - 1):
        p0 = track[i]
        p1 = track[i + 1]
        if p0.lat_deg < 0.0 and p1.lat_deg >= 0.0:
            # Linear interpolation for zero crossing
            if abs(p1.lat_deg - p0.lat_deg) < 1e-12:
                frac = 0.5
            else:
                frac = -p0.lat_deg / (p1.lat_deg - p0.lat_deg)
            dt = (p1.time - p0.time).total_seconds()
            crossing_time = p0.time + timedelta(seconds=frac * dt)

            # Interpolate longitude with wrap-around handling
            lon0 = p0.lon_deg
            lon1 = p1.lon_deg
            dlon = lon1 - lon0
            if dlon > 180.0:
                dlon -= 360.0
            elif dlon < -180.0:
                dlon += 360.0
            crossing_lon = lon0 + frac * dlon
            if crossing_lon > 180.0:
                crossing_lon -= 360.0
            elif crossing_lon < -180.0:
                crossing_lon += 360.0

            nodes.append(AscendingNodePass(
                time=crossing_time, longitude_deg=crossing_lon,
            ))
    return nodes


def _segment_intersection_2d(
    lat_a0: float, lon_a0: float, lat_a1: float, lon_a1: float,
    lat_b0: float, lon_b0: float, lat_b1: float, lon_b1: float,
) -> tuple[float, float, float, float] | None:
    """2D line segment intersection in lat/lon space.

    Returns (lat, lon, t_a, t_b) if segments intersect, None otherwise.
    t_a and t_b are interpolation parameters along segments A and B.
    """
    dx_a = lat_a1 - lat_a0
    dy_a = lon_a1 - lon_a0
    dx_b = lat_b1 - lat_b0
    dy_b = lon_b1 - lon_b0

    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) < 1e-15:
        return None  # Parallel

    dx_ab = lat_b0 - lat_a0
    dy_ab = lon_b0 - lon_a0

    t_a = (dx_ab * dy_b - dy_ab * dx_b) / denom
    t_b = (dx_ab * dy_a - dy_ab * dx_a) / denom

    if 0.0 <= t_a <= 1.0 and 0.0 <= t_b <= 1.0:
        lat = lat_a0 + t_a * dx_a
        lon = lon_a0 + t_a * dy_a
        return (lat, lon, t_a, t_b)
    return None


def find_ground_track_crossings(
    track: list[GroundTrackPoint],
) -> list[GroundTrackCrossing]:
    """Find where ascending and descending ground track segments intersect.

    1. Classify each segment as ascending (lat increases) or descending.
    2. Collect ascending and descending segment indices.
    3. Test all ascending-descending pairs for intersection.
    4. Handle longitude wrap-around by skipping large jumps.

    Args:
        track: List of GroundTrackPoint from compute_ground_track.

    Returns:
        List of GroundTrackCrossing at intersection points.
    """
    if len(track) < 2:
        return []

    ascending_segs: list[int] = []
    descending_segs: list[int] = []

    for i in range(len(track) - 1):
        dlat = track[i + 1].lat_deg - track[i].lat_deg
        dlon = abs(track[i + 1].lon_deg - track[i].lon_deg)
        # Skip segments with longitude wrapping (>90° jump)
        if dlon > 90.0:
            continue
        if dlat > 0.01:
            ascending_segs.append(i)
        elif dlat < -0.01:
            descending_segs.append(i)

    crossings: list[GroundTrackCrossing] = []
    for ia in ascending_segs:
        p_a0 = track[ia]
        p_a1 = track[ia + 1]
        for id_ in descending_segs:
            p_d0 = track[id_]
            p_d1 = track[id_ + 1]

            # Skip if latitude ranges don't overlap
            lat_a_min = min(p_a0.lat_deg, p_a1.lat_deg)
            lat_a_max = max(p_a0.lat_deg, p_a1.lat_deg)
            lat_d_min = min(p_d0.lat_deg, p_d1.lat_deg)
            lat_d_max = max(p_d0.lat_deg, p_d1.lat_deg)
            if lat_a_max < lat_d_min or lat_d_max < lat_a_min:
                continue

            result = _segment_intersection_2d(
                p_a0.lat_deg, p_a0.lon_deg, p_a1.lat_deg, p_a1.lon_deg,
                p_d0.lat_deg, p_d0.lon_deg, p_d1.lat_deg, p_d1.lon_deg,
            )
            if result is not None:
                lat, lon, t_a, t_d = result
                dt_a = (p_a1.time - p_a0.time).total_seconds()
                dt_d = (p_d1.time - p_d0.time).total_seconds()
                time_asc = p_a0.time + timedelta(seconds=t_a * dt_a)
                time_desc = p_d0.time + timedelta(seconds=t_d * dt_d)
                crossings.append(GroundTrackCrossing(
                    lat_deg=lat, lon_deg=lon,
                    time_ascending=time_asc, time_descending=time_desc,
                ))

    return crossings
