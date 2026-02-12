# Copyright (c) 2026 Jeroen. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""
Eclipse and shadow prediction.

Cylindrical shadow model for Earth eclipse detection, beta angle
computation, and eclipse window/fraction calculation.

No external dependencies — only stdlib math/dataclasses/datetime/enum.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.solar import sun_position_eci
from constellation_generator.domain.propagation import OrbitalState, propagate_to

_R_EARTH = OrbitalConstants.R_EARTH


class EclipseType(Enum):
    NONE = "none"
    PENUMBRA = "penumbra"
    UMBRA = "umbra"


@dataclass(frozen=True)
class EclipseEvent:
    """A single eclipse entry/exit event."""
    entry_time: datetime
    exit_time: datetime
    eclipse_type: EclipseType
    duration_seconds: float


def is_eclipsed(
    sat_position_eci: tuple[float, float, float],
    sun_position_eci_m: tuple[float, float, float],
    earth_radius_m: float | None = None,
) -> EclipseType:
    """Determine if satellite is in Earth's shadow.

    Cylindrical shadow model:
    1. Compute vector from satellite to Sun
    2. If satellite is on the sunlit side (dot product > 0), NONE
    3. Project satellite onto Earth-Sun line
    4. Compute perpendicular distance
    5. If perpendicular distance < R_Earth → UMBRA

    Args:
        sat_position_eci: Satellite ECI position (m).
        sun_position_eci_m: Sun ECI position (m).
        earth_radius_m: Override Earth radius (m).

    Returns:
        EclipseType (NONE, PENUMBRA, or UMBRA).
    """
    r_earth = earth_radius_m if earth_radius_m is not None else _R_EARTH

    sx, sy, sz = sat_position_eci
    ux, uy, uz = sun_position_eci_m

    # Vector from satellite to Sun
    to_sun_x = ux - sx
    to_sun_y = uy - sy
    to_sun_z = uz - sz

    # If dot(sat_pos, to_sun) > 0, satellite is on the sunlit side
    dot_sat_sun = sx * to_sun_x + sy * to_sun_y + sz * to_sun_z
    if dot_sat_sun > 0:
        return EclipseType.NONE

    # Satellite is behind Earth relative to Sun.
    # Project satellite position onto Earth-Sun line (unit direction from Earth to Sun).
    sun_dist = math.sqrt(ux**2 + uy**2 + uz**2)
    if sun_dist == 0:
        return EclipseType.NONE

    sun_hat_x = ux / sun_dist
    sun_hat_y = uy / sun_dist
    sun_hat_z = uz / sun_dist

    # Projection of satellite onto Sun direction
    proj = sx * sun_hat_x + sy * sun_hat_y + sz * sun_hat_z

    # Perpendicular distance from Earth-Sun line
    perp_x = sx - proj * sun_hat_x
    perp_y = sy - proj * sun_hat_y
    perp_z = sz - proj * sun_hat_z
    perp_dist = math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)

    if perp_dist < r_earth:
        return EclipseType.UMBRA

    return EclipseType.NONE


def compute_beta_angle(
    raan_rad: float,
    inclination_rad: float,
    epoch: datetime,
) -> float:
    """Orbital beta angle (angle between Sun and orbital plane).

    beta = arcsin(cos(dec_sun)*sin(RAAN - RA_sun)*sin(i) + sin(dec_sun)*cos(i))

    Ref: Vallado Section 5.4.

    Args:
        raan_rad: Right ascension of ascending node (radians).
        inclination_rad: Orbital inclination (radians).
        epoch: Epoch for Sun position.

    Returns:
        Beta angle in degrees.
    """
    sun = sun_position_eci(epoch)
    ra_sun = sun.right_ascension_rad
    dec_sun = sun.declination_rad

    sin_beta = (
        math.cos(dec_sun) * math.sin(raan_rad - ra_sun) * math.sin(inclination_rad)
        + math.sin(dec_sun) * math.cos(inclination_rad)
    )
    # Clamp for numerical safety
    sin_beta = max(-1.0, min(1.0, sin_beta))
    return math.degrees(math.asin(sin_beta))


def compute_eclipse_windows(
    state: OrbitalState,
    start: datetime,
    duration: timedelta,
    step: timedelta,
) -> list[EclipseEvent]:
    """Compute eclipse entry/exit times over a time window.

    Time-sweep pattern: propagate satellite, compute Sun position,
    check is_eclipsed, track state transitions.

    Args:
        state: Satellite orbital state.
        start: Start time (UTC).
        duration: Total analysis duration.
        step: Time step for sweep.

    Returns:
        List of EclipseEvent objects.
    """
    step_seconds = step.total_seconds()
    duration_seconds = duration.total_seconds()
    windows: list[EclipseEvent] = []

    in_eclipse = False
    entry_time = start
    elapsed = 0.0

    while elapsed <= duration_seconds + 1e-9:
        current_time = start + timedelta(seconds=elapsed)
        pos_eci, _ = propagate_to(state, current_time)
        sun = sun_position_eci(current_time)

        sat_pos = (pos_eci[0], pos_eci[1], pos_eci[2])
        eclipse_type = is_eclipsed(sat_pos, sun.position_eci_m)
        is_dark = eclipse_type != EclipseType.NONE

        if is_dark and not in_eclipse:
            entry_time = current_time
            in_eclipse = True
        elif not is_dark and in_eclipse:
            dur = (current_time - entry_time).total_seconds()
            windows.append(EclipseEvent(
                entry_time=entry_time,
                exit_time=current_time,
                eclipse_type=EclipseType.UMBRA,
                duration_seconds=dur,
            ))
            in_eclipse = False

        elapsed += step_seconds

    if in_eclipse:
        exit_time = start + timedelta(seconds=duration_seconds)
        dur = (exit_time - entry_time).total_seconds()
        windows.append(EclipseEvent(
            entry_time=entry_time,
            exit_time=exit_time,
            eclipse_type=EclipseType.UMBRA,
            duration_seconds=dur,
        ))

    return windows


def eclipse_fraction(
    state: OrbitalState,
    epoch: datetime,
    num_points: int = 360,
) -> float:
    """Fraction of orbit in eclipse at given epoch.

    Sweeps one orbital period at num_points intervals.

    Args:
        state: Satellite orbital state.
        epoch: Reference epoch.
        num_points: Number of sample points around orbit.

    Returns:
        Fraction of orbit in eclipse (0.0 to 1.0).
    """
    T = 2.0 * math.pi / state.mean_motion_rad_s
    dt = T / num_points
    eclipsed_count = 0

    for i in range(num_points):
        t = epoch + timedelta(seconds=i * dt)
        pos_eci, _ = propagate_to(state, t)
        sun = sun_position_eci(t)
        sat_pos = (pos_eci[0], pos_eci[1], pos_eci[2])
        if is_eclipsed(sat_pos, sun.position_eci_m) != EclipseType.NONE:
            eclipsed_count += 1

    return eclipsed_count / num_points
