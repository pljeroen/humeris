"""
Ground track computation for circular orbits.

Propagates a satellite's position over time using Keplerian two-body
mechanics (optionally with J2 secular perturbations) and converts to
geodetic coordinates (lat/lon/alt) via the ECI -> ECEF -> Geodetic
pipeline.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
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
