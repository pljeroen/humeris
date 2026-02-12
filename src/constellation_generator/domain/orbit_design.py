# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""
Orbit design utilities.

Sun-synchronous orbit with LTAN, frozen orbit design (J2/J3 balanced),
and repeat ground track orbit computation.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime

from constellation_generator.domain.orbital_mechanics import (
    OrbitalConstants,
    sso_inclination_deg,
)
from constellation_generator.domain.solar import sun_position_eci

_MU = OrbitalConstants.MU_EARTH
_R_E = OrbitalConstants.R_EARTH
_J2 = OrbitalConstants.J2_EARTH
_J3 = OrbitalConstants.J3_EARTH

# Earth sidereal day in seconds
_SIDEREAL_DAY_S = 86164.0905


@dataclass(frozen=True)
class SSODesign:
    """Sun-synchronous orbit design parameters."""
    altitude_km: float
    inclination_deg: float
    raan_deg: float
    ltan_hours: float


@dataclass(frozen=True)
class FrozenOrbitDesign:
    """Frozen orbit parameters (J2/J3 balanced)."""
    altitude_km: float
    inclination_deg: float
    eccentricity: float
    arg_perigee_deg: float  # 90° or 270°


@dataclass(frozen=True)
class RepeatGroundTrackDesign:
    """Repeat ground track orbit parameters."""
    semi_major_axis_m: float
    altitude_km: float
    inclination_deg: float
    repeat_days: int
    repeat_revolutions: int
    revolutions_per_day: float


def design_sso_orbit(
    altitude_km: float, ltan_hours: float, epoch: datetime,
) -> SSODesign:
    """Design SSO with specified LTAN.

    1. Compute inclination using existing sso_inclination_deg()
    2. Compute Sun RA at epoch using solar ephemeris
    3. RAAN = sun_RA + (LTAN - 12) * 15 degrees

    Ref: Vallado Ch. 11.

    Args:
        altitude_km: Target orbital altitude (km).
        ltan_hours: Local Time of Ascending Node (hours, 0-24).
        epoch: Reference epoch for RAAN computation.

    Returns:
        SSODesign with computed inclination and RAAN.
    """
    inc_deg = sso_inclination_deg(altitude_km)
    sun = sun_position_eci(epoch)
    sun_ra_deg = math.degrees(sun.right_ascension_rad)
    raan_deg = (sun_ra_deg + (ltan_hours - 12.0) * 15.0) % 360.0

    return SSODesign(
        altitude_km=altitude_km,
        inclination_deg=inc_deg,
        raan_deg=raan_deg,
        ltan_hours=ltan_hours,
    )


def design_frozen_orbit(
    altitude_km: float, inclination_deg: float,
) -> FrozenOrbitDesign:
    """Frozen orbit where J2/J3 perturbations cancel eccentricity drift.

    e_frozen = -(J3 * R_E) / (2 * J2 * a) * sin(i)
    omega = 90° (ascending node) or 270° (descending)

    Ref: Vallado Ch. 9.

    Args:
        altitude_km: Target orbital altitude (km).
        inclination_deg: Orbital inclination (degrees).

    Returns:
        FrozenOrbitDesign with balanced eccentricity.
    """
    a = _R_E + altitude_km * 1000.0
    i_rad = math.radians(inclination_deg)

    # Frozen eccentricity: e = -(J3 * R_E) / (2 * J2 * a) * sin(i)
    e_frozen = -(_J3 * _R_E) / (2.0 * _J2 * a) * math.sin(i_rad)

    # Choose arg_perigee based on sign convention
    if e_frozen >= 0:
        arg_perigee_deg = 90.0
    else:
        arg_perigee_deg = 270.0
        e_frozen = abs(e_frozen)

    return FrozenOrbitDesign(
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        eccentricity=e_frozen,
        arg_perigee_deg=arg_perigee_deg,
    )


def design_repeat_ground_track(
    inclination_deg: float,
    repeat_days: int,
    repeat_revolutions: int,
    tolerance_m: float = 1.0,
) -> RepeatGroundTrackDesign:
    """Repeat ground track orbit — find SMA matching the repeat condition.

    T_orbit = T_sidereal_day * repeat_days / repeat_revolutions
    Including J2 correction via bisection on SMA until nodal period matches.

    Ref: Vallado Ch. 11.

    Args:
        inclination_deg: Orbital inclination (degrees).
        repeat_days: Number of days for repeat cycle.
        repeat_revolutions: Number of revolutions in repeat cycle.
        tolerance_m: Convergence tolerance on SMA (m).

    Returns:
        RepeatGroundTrackDesign.

    Raises:
        ValueError: If repeat_days or repeat_revolutions <= 0.
    """
    if repeat_days <= 0:
        raise ValueError(f"repeat_days must be positive, got {repeat_days}")
    if repeat_revolutions <= 0:
        raise ValueError(f"repeat_revolutions must be positive, got {repeat_revolutions}")

    i_rad = math.radians(inclination_deg)
    target_period = _SIDEREAL_DAY_S * repeat_days / repeat_revolutions

    # Initial guess from Kepler's 3rd law (no J2)
    a_kepler = (_MU * (target_period / (2.0 * math.pi))**2)**(1.0 / 3.0)

    # Bisection to find J2-corrected SMA
    a_lo = a_kepler * 0.99
    a_hi = a_kepler * 1.01

    for _ in range(100):
        a_mid = (a_lo + a_hi) / 2.0
        T_nodal = _nodal_period(a_mid, i_rad)
        if T_nodal < target_period:
            a_lo = a_mid
        else:
            a_hi = a_mid
        if abs(a_hi - a_lo) < tolerance_m:
            break

    a_result = (a_lo + a_hi) / 2.0
    alt_km = (a_result - _R_E) / 1000.0
    revs_per_day = repeat_revolutions / repeat_days

    return RepeatGroundTrackDesign(
        semi_major_axis_m=a_result,
        altitude_km=alt_km,
        inclination_deg=inclination_deg,
        repeat_days=repeat_days,
        repeat_revolutions=repeat_revolutions,
        revolutions_per_day=revs_per_day,
    )


def _nodal_period(a: float, i_rad: float) -> float:
    """Nodal period including J2 effect.

    T_nodal = 2*pi/n * (1 - 3/2 * J2 * (R_E/a)^2 * (3 - 4*sin^2(i)) / (1 - e^2)^2)
    For circular orbits (e=0).
    """
    n = math.sqrt(_MU / a**3)
    T_kepler = 2.0 * math.pi / n
    p_ratio = (_R_E / a)**2
    correction = 1.0 - 1.5 * _J2 * p_ratio * (3.0 - 4.0 * math.sin(i_rad)**2)
    return T_kepler * correction
