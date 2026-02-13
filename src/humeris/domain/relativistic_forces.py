# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Relativistic force models for high-fidelity orbit propagation.

Post-Newtonian corrections per IERS 2010 Conventions (Chapter 10):
- Schwarzschild: dominant relativistic term (~2e-8 m/s² at LEO)
- Lense-Thirring: frame-dragging from Earth rotation (~2e-10 m/s² at LEO)
- de Sitter: geodesic precession from heliocentric motion (~5e-13 m/s²)

All three implement the ForceModel protocol (structural typing).
No external dependencies — only stdlib math/datetime.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

# --- Physical constants ---

_GM_EARTH: float = 3.986004418e14  # m³/s² (Earth gravitational parameter)
_C_LIGHT: float = 299792458.0  # m/s (speed of light)
_J_EARTH: float = 5.86e33  # kg·m²/s (Earth angular momentum = C·ω, Lense-Thirring)
_GM_SUN: float = 1.32712440041e20  # m³/s² (Sun gravitational parameter)
_G_NEWTON: float = 6.67430e-11  # m³/(kg·s²) (gravitational constant, for LT)

# --- Helpers ---

_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_AU = 1.495978707e11  # meters


def _datetime_to_jd(dt: datetime) -> float:
    """Convert datetime to Julian Date.

    Uses the standard algorithm: JD = 2451545.0 + seconds_since_J2000 / 86400.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (dt - _J2000).total_seconds()
    return 2451545.0 + delta / 86400.0


def _sun_position_approx(dt: datetime) -> tuple[float, float, float]:
    """Approximate Sun position in ECI (meters).

    Low-precision solar coordinates based on Meeus (1991) truncated series.
    Accuracy ~1° in ecliptic longitude, sufficient for de Sitter correction.
    """
    jd = _datetime_to_jd(dt)
    T = (jd - 2451545.0) / 36525.0

    # Mean anomaly of Sun (degrees)
    M_deg = (357.5291092 + 35999.0502909 * T) % 360.0
    M_rad = float(np.radians(M_deg))

    # Equation of center (degrees)
    C_deg = (
        (1.9146 - 0.004817 * T) * float(np.sin(M_rad))
        + 0.019993 * float(np.sin(2.0 * M_rad))
    )

    # Sun's mean longitude (degrees)
    L0_deg = (280.46646 + 36000.76983 * T) % 360.0

    # Sun's true longitude (radians)
    sun_lon = float(np.radians((L0_deg + C_deg) % 360.0))

    # Obliquity of ecliptic (radians)
    eps = float(np.radians(23.439291 - 0.0130042 * T))

    # Orbital eccentricity
    e = 0.016708634 - 0.000042037 * T

    # True anomaly
    nu = M_rad + float(np.radians(C_deg))

    # Sun–Earth distance (meters)
    r_au = 1.000001018 * (1.0 - e * e) / (1.0 + e * float(np.cos(nu)))
    r = r_au * _AU

    # ECI coordinates (equatorial frame via obliquity rotation)
    cos_lon = float(np.cos(sun_lon))
    sin_lon = float(np.sin(sun_lon))
    cos_eps = float(np.cos(eps))
    sin_eps = float(np.sin(eps))

    x = r * cos_lon
    y = r * sin_lon * cos_eps
    z = r * sin_lon * sin_eps

    return (x, y, z)


# --- Force models ---


@dataclass(frozen=True)
class SchwarzschildForce:
    """Post-Newtonian Schwarzschild correction (IERS 2010, eq. 10.4).

    Dominant relativistic perturbation for Earth-orbiting satellites.
    Isotropic PPN gauge with beta = gamma = 1 (General Relativity).

    Acceleration (isotropic gauge, β=γ=1):
        a = (GM/(c²r³)) * [(4GM/r - v²)r + 4(r·v)v]

    For circular orbits (r·v = 0), the acceleration is purely radial.
    For eccentric orbits, the 4(r·v)v term produces an along-track
    component. Secular precession: 6πGM/(c²a(1-e²)) rad/orbit.

    Magnitude at LEO (~400 km): ~2e-8 m/s².
    """

    gm: float = _GM_EARTH
    c: float = _C_LIGHT

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        pos = np.array(position)
        vel = np.array(velocity)

        r_sq = float(np.dot(pos, pos))
        r = float(np.sqrt(r_sq))
        v_sq = float(np.dot(vel, vel))
        r_dot_v = float(np.dot(pos, vel))

        c2 = self.c * self.c
        coeff = self.gm / (c2 * r * r_sq)

        gm_over_r = self.gm / r
        pos_factor = 4.0 * gm_over_r - v_sq
        vel_factor = 4.0 * r_dot_v

        a = coeff * (pos_factor * pos + vel_factor * vel)

        return (float(a[0]), float(a[1]), float(a[2]))


@dataclass(frozen=True)
class LenseThirringForce:
    """Lense-Thirring frame-dragging acceleration (IERS 2010, eq. 10.5).

    Arises from Earth's rotation (angular momentum J along z-axis).
    Post-Newtonian formulation (IERS 2010, Soffel et al. 2003):

    Acceleration:
        a = (2G/(c²r³)) * { (3/r²)(r·J)(r × v) + (v × J) }

    where J = (0, 0, J_EARTH) is Earth's spin angular momentum vector
    (J = C·ω ≈ 5.86e33 kg·m²/s), G is Newton's gravitational constant.

    Magnitude at LEO: ~2e-10 m/s² (equatorial), varies with inclination.
    """

    j_earth: float = _J_EARTH
    c: float = _C_LIGHT
    g_newton: float = _G_NEWTON

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        pos = np.array(position)
        vel = np.array(velocity)

        r_sq = float(np.dot(pos, pos))
        r = float(np.sqrt(r_sq))

        # J = (0, 0, J_EARTH) — Earth spin angular momentum along z-axis.
        J = np.array([0.0, 0.0, self.j_earth])

        # r · J = z * J_EARTH
        r_dot_J = float(np.dot(pos, J))

        # r × v
        rxv = np.cross(pos, vel)

        # v × J
        vxJ = np.cross(vel, J)

        c2 = self.c * self.c
        coeff = 2.0 * self.g_newton / (c2 * r * r_sq)
        inner_coeff = 3.0 * r_dot_J / r_sq

        a = coeff * (inner_coeff * rxv + vxJ)

        return (float(a[0]), float(a[1]), float(a[2]))


@dataclass(frozen=True)
class DeSitterForce:
    """Geodesic (de Sitter) precession from Earth's heliocentric motion.

    IERS 2010, eq. 10.6. Accounts for the curvature of spacetime
    due to the Sun's gravitational field as Earth orbits through it.

    Acceleration:
        a_dS = -(3 GM_S / (2 c² R³)) * (V_E × (R_sun × v_sat))

    where R_sun is the Earth-to-Sun vector, V_E is Earth's heliocentric
    velocity (computed via finite differences of the Sun ephemeris),
    and v_sat is the satellite geocentric velocity.

    Magnitude at LEO: ~5e-13 m/s².
    """

    gm_sun: float = _GM_SUN
    c: float = _C_LIGHT

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        vel_arr = np.array(velocity)

        # Sun position (Earth-to-Sun vector) at epoch
        sun = np.array(_sun_position_approx(epoch))
        R_sq = float(np.dot(sun, sun))
        R = float(np.sqrt(R_sq))

        # Earth heliocentric velocity via finite difference.
        # Earth velocity = -d(Sun_pos_ECI)/dt
        dt_s = 3600.0  # 1 hour step
        epoch2 = epoch + timedelta(seconds=dt_s)
        sun2 = np.array(_sun_position_approx(epoch2))

        Ve = -(sun2 - sun) / dt_s

        # R_sun × v_sat
        Rxv = np.cross(sun, vel_arr)

        # V_E × (R_sun × v_sat)
        cross = np.cross(Ve, Rxv)

        c2 = self.c * self.c
        R_cubed = R * R_sq
        coeff = -3.0 * self.gm_sun / (2.0 * c2 * R_cubed)

        a = coeff * cross
        return (float(a[0]), float(a[1]), float(a[2]))
