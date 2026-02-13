# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see LICENSE-COMMERCIAL.md.
"""Earth albedo and infrared radiation pressure force model.

Models two effects on satellite orbits:
1. Earth albedo: reflected solar radiation from Earth's surface
2. Earth IR: thermal infrared radiation emitted by Earth

Both produce radially outward acceleration that decreases with altitude.
"""

import math
from datetime import datetime


# --- Constants ---

_SOLAR_FLUX = 1361.0  # W/m², solar constant at 1 AU
_EARTH_ALBEDO = 0.3  # mean Earth albedo
_EARTH_IR = 237.0  # W/m², mean Earth IR emission
_R_EARTH = 6378137.0  # m
_C_LIGHT = 299792458.0  # m/s
_AU = 1.495978707e11  # m


# --- Vector helpers ---

def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _mag(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    m = _mag(v)
    if m < 1e-30:
        return (0.0, 0.0, 0.0)
    return (v[0] / m, v[1] / m, v[2] / m)


# --- Sun position ---

def _datetime_to_jd(dt: datetime) -> float:
    """Convert datetime to Julian Date."""
    y = dt.year
    m = dt.month
    d = dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def _sun_position_approx(dt: datetime) -> tuple[float, float, float]:
    """Approximate Sun position in ECI (meters)."""
    jd = _datetime_to_jd(dt)
    t = (jd - 2451545.0) / 36525.0
    mean_anom = math.radians((357.5291092 + 35999.0502909 * t) % 360)
    center = (
        (1.9146 - 0.004817 * t) * math.sin(mean_anom)
        + 0.019993 * math.sin(2 * mean_anom)
    )
    sun_lon = math.radians((280.46646 + 36000.76983 * t + center) % 360)
    eps = math.radians(23.439291 - 0.0130042 * t)
    e = 0.016708634 - 0.000042037 * t
    nu = mean_anom + math.radians(center)
    r = 1.000001018 * (1 - e * e) / (1 + e * math.cos(nu)) * _AU

    x = r * math.cos(sun_lon)
    y = r * math.sin(sun_lon) * math.cos(eps)
    z = r * math.sin(sun_lon) * math.sin(eps)
    return (x, y, z)


# --- Force model ---

class AlbedoRadiationPressure:
    """Earth albedo and infrared radiation pressure acceleration.

    Models reflected solar radiation (albedo) and thermal infrared
    emission from Earth acting on a satellite. Both components produce
    acceleration directed radially away from Earth.

    Implements the ForceModel protocol.
    """

    def __init__(
        self,
        cr: float,
        area_m2: float,
        mass_kg: float,
        n_sectors: int = 6,
    ) -> None:
        self._cr = cr
        self._area_m2 = area_m2
        self._mass_kg = mass_kg
        self._n_sectors = n_sectors

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Compute combined albedo + IR radiation pressure acceleration.

        Args:
            epoch: UTC datetime of the evaluation.
            position: ECI position in meters (x, y, z).
            velocity: ECI velocity in m/s (vx, vy, vz).

        Returns:
            Acceleration vector in m/s^2 (ax, ay, az).
        """
        r = _mag(position)
        if r < _R_EARTH:
            return (0.0, 0.0, 0.0)

        r_hat = _unit(position)
        am_ratio = self._area_m2 / self._mass_kg

        # --- Earth albedo component ---
        sun_pos = _sun_position_approx(epoch)
        sun_hat = _unit(sun_pos)

        # Phase angle: how much of the sunlit hemisphere the satellite sees
        cos_phase = max(0.0, _dot(r_hat, sun_hat))

        # Reflected solar flux at satellite altitude
        re_over_r = _R_EARTH / r
        p_albedo = (
            _EARTH_ALBEDO * _SOLAR_FLUX * re_over_r * re_over_r
            * cos_phase * (2.0 / 3.0)
        )

        a_albedo_mag = self._cr * am_ratio * p_albedo / _C_LIGHT

        # --- Earth IR component ---
        p_ir = _EARTH_IR * re_over_r * re_over_r
        a_ir_mag = self._cr * am_ratio * p_ir / _C_LIGHT

        # --- Total (radially outward) ---
        total_mag = a_albedo_mag + a_ir_mag

        return (
            total_mag * r_hat[0],
            total_mag * r_hat[1],
            total_mag * r_hat[2],
        )
