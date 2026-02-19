# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Eclipse and shadow prediction.

Cylindrical shadow model for Earth eclipse detection, beta angle
computation, and eclipse window/fraction calculation.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.solar import sun_position_eci
from humeris.domain.propagation import OrbitalState, propagate_to

_R_EARTH = OrbitalConstants.R_EARTH
_R_SUN = 6.957e8  # metres — solar radius

# Eclipse beta-angle threshold: orbits with |beta| below this experience eclipses
_ECLIPSE_BETA_THRESHOLD_DEG = 70.0


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

    Conical shadow model with penumbra:
    1. Compute vector from satellite to Sun
    2. If satellite is on the sunlit side (dot product > 0), NONE
    3. Project satellite onto Earth-Sun line
    4. Compute perpendicular distance from shadow axis
    5. Compute umbra and penumbra cone radii at satellite's axial distance
    6. Classify as UMBRA, PENUMBRA, or NONE

    Args:
        sat_position_eci: Satellite ECI position (m).
        sun_position_eci_m: Sun ECI position (m).
        earth_radius_m: Override Earth radius (m).

    Returns:
        EclipseType (NONE, PENUMBRA, or UMBRA).
    """
    r_earth = earth_radius_m if earth_radius_m is not None else _R_EARTH

    sat = np.array(sat_position_eci)
    sun = np.array(sun_position_eci_m)

    # Vector from satellite to Sun
    to_sun = sun - sat

    # If dot(sat_pos, to_sun) > 0, satellite is on the sunlit side
    dot_sat_sun = float(np.dot(sat, to_sun))
    if dot_sat_sun > 0:
        return EclipseType.NONE

    # Satellite is behind Earth relative to Sun.
    sun_dist = float(np.linalg.norm(sun))
    if sun_dist < 1e-3:
        return EclipseType.NONE

    sun_hat = sun / sun_dist

    # Projection of satellite onto Sun direction (negative = behind Earth)
    proj = float(np.dot(sat, sun_hat))

    # Perpendicular distance from Earth-Sun line
    perp = sat - proj * sun_hat
    perp_dist = float(np.linalg.norm(perp))

    # Distance from Earth center along shadow axis (positive = behind Earth)
    d_along = -proj

    # Cone radii at satellite's axial distance behind Earth
    # From similar triangles (Montenbruck & Gill):
    #   umbra: R_e - d * (R_sun - R_e) / D
    #   penumbra: R_e + d * (R_sun + R_e) / D
    umbra_radius = r_earth - d_along * (_R_SUN - r_earth) / sun_dist
    penumbra_radius = r_earth + d_along * (_R_SUN + r_earth) / sun_dist

    if perp_dist < umbra_radius:
        return EclipseType.UMBRA
    if perp_dist < penumbra_radius:
        return EclipseType.PENUMBRA

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
        float(np.cos(dec_sun)) * float(np.sin(raan_rad - ra_sun)) * float(np.sin(inclination_rad))
        + float(np.sin(dec_sun)) * float(np.cos(inclination_rad))
    )
    # Clamp for numerical safety
    sin_beta = max(-1.0, min(1.0, sin_beta))
    return float(np.degrees(np.arcsin(sin_beta)))


def _bisect_event(
    f: "Callable[[float], float]",
    t_a: float,
    t_b: float,
    tol_s: float = 1e-3,
    max_iter: int = 50,
) -> float:
    """Find root of f(t) between t_a and t_b via bisection.

    Assumes f(t_a) and f(t_b) have opposite signs.

    Args:
        f: Function f(t) -> float (sign change indicates event).
        t_a: Left bound (seconds since reference).
        t_b: Right bound (seconds since reference).
        tol_s: Tolerance in seconds.
        max_iter: Maximum bisection iterations.

    Returns:
        Approximate root time (seconds since reference).
    """
    f_a = f(t_a)
    for _ in range(max_iter):
        t_mid = (t_a + t_b) / 2.0
        f_mid = f(t_mid)
        if f_mid * f_a < 0:
            t_b = t_mid
        else:
            t_a = t_mid
            f_a = f_mid
        if abs(t_b - t_a) < tol_s:
            break
    return (t_a + t_b) / 2.0


def compute_eclipse_windows(
    state: OrbitalState,
    start: datetime,
    duration: timedelta,
    step: timedelta,
) -> list[EclipseEvent]:
    """Compute eclipse entry/exit times over a time window.

    Time-sweep pattern with bisection refinement at transitions:
    1. Propagate satellite, compute Sun position, check is_eclipsed
    2. When a transition is detected, refine with bisection to ~1ms accuracy
    3. Track state transitions and record EclipseEvent

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

    def eclipse_indicator(elapsed_s: float) -> float:
        """Returns +1 if eclipsed, -1 if not."""
        t = start + timedelta(seconds=elapsed_s)
        pos_eci, _ = propagate_to(state, t)
        sun = sun_position_eci(t)
        sat_pos = (pos_eci[0], pos_eci[1], pos_eci[2])
        if is_eclipsed(sat_pos, sun.position_eci_m) != EclipseType.NONE:
            return 1.0
        return -1.0

    in_eclipse = False
    entry_elapsed = 0.0
    elapsed = 0.0
    prev_indicator = eclipse_indicator(0.0)

    while elapsed <= duration_seconds + 1e-9:
        current_indicator = eclipse_indicator(elapsed)

        if current_indicator > 0 and not in_eclipse:
            # Eclipse entry — refine
            prev_elapsed = max(0.0, elapsed - step_seconds)
            refined = _bisect_event(eclipse_indicator, prev_elapsed, elapsed)
            entry_elapsed = refined
            in_eclipse = True
        elif current_indicator < 0 and in_eclipse:
            # Eclipse exit — refine
            prev_elapsed = elapsed - step_seconds
            refined = _bisect_event(eclipse_indicator, prev_elapsed, elapsed)
            entry_time = start + timedelta(seconds=entry_elapsed)
            exit_time = start + timedelta(seconds=refined)
            dur = refined - entry_elapsed
            windows.append(EclipseEvent(
                entry_time=entry_time,
                exit_time=exit_time,
                eclipse_type=EclipseType.UMBRA,
                duration_seconds=dur,
            ))
            in_eclipse = False

        elapsed += step_seconds

    if in_eclipse:
        exit_time = start + timedelta(seconds=duration_seconds)
        dur = duration_seconds - entry_elapsed
        windows.append(EclipseEvent(
            entry_time=start + timedelta(seconds=entry_elapsed),
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


@dataclass(frozen=True)
class BetaAngleSnapshot:
    """Beta angle at a single point in time."""
    time: datetime
    beta_deg: float


@dataclass(frozen=True)
class BetaAngleHistory:
    """Time series of beta angle evolution."""
    snapshots: tuple[BetaAngleSnapshot, ...]
    duration_s: float


def compute_beta_angle_history(
    raan_rad: float,
    inclination_rad: float,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    raan_drift_rad_s: float = 0.0,
) -> BetaAngleHistory:
    """Time series of beta angle, optionally with RAAN drift.

    Args:
        raan_rad: Initial RAAN (radians).
        inclination_rad: Orbital inclination (radians).
        epoch: Start epoch.
        duration_s: Total duration (seconds).
        step_s: Time step (seconds).
        raan_drift_rad_s: RAAN precession rate (rad/s), e.g. from J2.

    Returns:
        BetaAngleHistory with snapshots at each time step.
    """
    snapshots: list[BetaAngleSnapshot] = []
    elapsed = 0.0
    while elapsed <= duration_s + 1e-9:
        current_time = epoch + timedelta(seconds=elapsed)
        current_raan = raan_rad + raan_drift_rad_s * elapsed
        beta = compute_beta_angle(current_raan, inclination_rad, current_time)
        snapshots.append(BetaAngleSnapshot(time=current_time, beta_deg=beta))
        elapsed += step_s

    return BetaAngleHistory(snapshots=tuple(snapshots), duration_s=duration_s)


def predict_eclipse_seasons(
    raan_rad: float,
    inclination_rad: float,
    epoch: datetime,
    duration_days: float,
    step_days: float = 1.0,
    raan_drift_rad_s: float = 0.0,
) -> list[tuple[datetime, datetime]]:
    """Predict intervals where |beta| < threshold (eclipse occurrence).

    Args:
        raan_rad: Initial RAAN (radians).
        inclination_rad: Orbital inclination (radians).
        epoch: Start epoch.
        duration_days: Analysis duration (days).
        step_days: Time step (days).
        raan_drift_rad_s: RAAN precession rate (rad/s).

    Returns:
        List of (start, end) datetime tuples for eclipse seasons.
    """
    if duration_days <= 0:
        return []

    duration_s = duration_days * 86400.0
    step_s = step_days * 86400.0

    seasons: list[tuple[datetime, datetime]] = []
    in_season = False
    season_start = epoch
    elapsed = 0.0

    while elapsed <= duration_s + 1e-9:
        current_time = epoch + timedelta(seconds=elapsed)
        current_raan = raan_rad + raan_drift_rad_s * elapsed
        beta = compute_beta_angle(current_raan, inclination_rad, current_time)
        is_eclipse = abs(beta) < _ECLIPSE_BETA_THRESHOLD_DEG

        if is_eclipse and not in_season:
            season_start = current_time
            in_season = True
        elif not is_eclipse and in_season:
            seasons.append((season_start, current_time))
            in_season = False

        elapsed += step_s

    if in_season:
        end_time = epoch + timedelta(seconds=duration_s)
        seasons.append((season_start, end_time))

    return seasons
