# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Atmospheric density model and drag acceleration.

Exponential atmospheric density with altitude-dependent scale height
(Vallado Table 8-4 / CIRA reference values). Covers 100-2000 km.

"""
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants


class AtmosphereModel(Enum):
    """Exponential atmosphere density table selection."""
    VALLADO_4TH = "vallado_4th"      # Vallado 4th ed. Table 8-4 (moderate solar activity)
    HIGH_ACTIVITY = "high_activity"   # Higher-density table (~solar maximum)


@dataclass(frozen=True)
class DragConfig:
    """Drag configuration for a satellite.

    cd: drag coefficient (dimensionless, typically 2.0-2.5)
    area_m2: cross-sectional area (m²)
    mass_kg: satellite mass (kg)
    """
    cd: float
    area_m2: float
    mass_kg: float

    @property
    def ballistic_coefficient(self) -> float:
        """Ballistic coefficient B_c = C_d * A / m (m²/kg)."""
        return self.cd * self.area_m2 / self.mass_kg


# High-activity exponential atmosphere table (original, ~solar maximum)
# Source: CIRA reference atmosphere / higher solar activity conditions
_ATMOSPHERE_TABLE_HIGH: tuple[tuple[float, float, float], ...] = (
    (100, 5.297e-07, 5.877),
    (150, 2.070e-09, 22.523),
    (200, 2.541e-10, 53.298),
    (250, 6.967e-11, 68.019),
    (300, 2.508e-11, 76.680),
    (350, 1.172e-11, 84.852),
    (400, 6.097e-12, 89.412),
    (450, 3.510e-12, 97.498),
    (500, 2.150e-12, 112.458),
    (600, 8.620e-13, 133.060),
    (700, 3.614e-13, 150.580),
    (800, 1.454e-13, 164.441),
    (900, 5.811e-14, 175.579),
    (1000, 2.302e-14, 188.667),
    (1100, 9.661e-15, 200.000),
    (1200, 4.297e-15, 210.000),
    (1300, 2.036e-15, 218.000),
    (1400, 1.024e-15, 225.000),
    (1500, 5.448e-16, 231.000),
    (1600, 3.059e-16, 236.000),
    (1700, 1.806e-16, 240.000),
    (1800, 1.115e-16, 243.000),
    (1900, 7.170e-17, 245.000),
    (2000, 4.789e-17, 247.000),
)

# Vallado 4th ed. Table 8-4 (moderate solar activity)
# Includes entries at 110-140km to avoid interpolation gap
_ATMOSPHERE_TABLE_VALLADO: tuple[tuple[float, float, float], ...] = (
    (100, 5.297e-07, 5.877),
    (110, 9.661e-08, 7.263),
    (120, 2.438e-08, 9.473),
    (130, 8.484e-09, 12.636),
    (140, 3.845e-09, 16.149),
    (150, 2.070e-09, 22.523),
    (180, 5.464e-10, 29.740),
    (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822),
    (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64),
    (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
)

_MODEL_TABLES = {
    AtmosphereModel.VALLADO_4TH: _ATMOSPHERE_TABLE_VALLADO,
    AtmosphereModel.HIGH_ACTIVITY: _ATMOSPHERE_TABLE_HIGH,
}


def atmospheric_density(
    altitude_km: float,
    model: AtmosphereModel = AtmosphereModel.HIGH_ACTIVITY,
) -> float:
    """Atmospheric density at given altitude using piecewise exponential model.

    Binary-searches the lookup table for the altitude bracket, then
    interpolates: rho = rho_base * exp(-(h - h_base) / H)

    Args:
        altitude_km: Altitude above Earth surface in km.
        model: Atmosphere model table to use.

    Returns:
        Atmospheric density in kg/m³.

    Raises:
        ValueError: If altitude is outside table range.
    """
    table = _MODEL_TABLES[model]

    if altitude_km < table[0][0] or altitude_km > table[-1][0]:
        raise ValueError(
            f"Altitude {altitude_km} km outside valid range "
            f"[{table[0][0]}, {table[-1][0]}] km"
        )

    # Binary search for bracket
    lo, hi = 0, len(table) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if table[mid][0] <= altitude_km:
            lo = mid
        else:
            hi = mid

    h_base, rho_base, scale_height = table[lo]
    return float(rho_base * np.exp(-(altitude_km - h_base) / scale_height))


def drag_acceleration(density: float, velocity: float, drag_config: DragConfig) -> float:
    """Drag acceleration magnitude.

    a_drag = 0.5 * rho * v² * B_c

    Args:
        density: Atmospheric density (kg/m³).
        velocity: Orbital velocity magnitude (m/s).
        drag_config: Satellite drag configuration.

    Returns:
        Drag acceleration in m/s².
    """
    return 0.5 * density * velocity ** 2 * drag_config.ballistic_coefficient


def semi_major_axis_decay_rate(
    a: float,
    e: float,
    drag_config: DragConfig,
    model: AtmosphereModel = AtmosphereModel.HIGH_ACTIVITY,
) -> float:
    """Rate of semi-major axis decay due to atmospheric drag.

    da/dt = -rho(h) * v * B_c * a

    where v = sqrt(mu/a), h = (a - R_E) / 1000.

    Args:
        a: Semi-major axis in meters.
        e: Eccentricity (used for perigee altitude in future extensions).
        drag_config: Satellite drag configuration.
        model: Atmosphere model table to use.

    Returns:
        da/dt in m/s (negative — orbit decays).
    """
    h_km = (a - OrbitalConstants.R_EARTH) / 1000.0
    v = float(np.sqrt(OrbitalConstants.MU_EARTH / a))
    rho = atmospheric_density(h_km, model=model)
    return -rho * v * drag_config.ballistic_coefficient * a


# ── P48: Boundary Layer Flow Regime Transition ─────────────────────
#
# Knudsen number-based flow regime classification and C_D interpolation
# between free-molecular and continuum regimes.


# Boltzmann constant (J/K)
_K_BOLTZMANN = 1.380649e-23
# Effective molecular diameter of N2 (dominant species, meters)
_SIGMA_MOLECULE = 3.7e-10
# Mean molecular mass of air (kg) — ~29 amu
_M_MOLECULE = 4.81e-26


@dataclass(frozen=True)
class FlowRegimeTransition:
    """Boundary layer flow regime transition analysis.

    Attributes:
        boundary_layer_altitude_km: Altitude where Kn = 1 (transition boundary).
        boundary_layer_thickness_km: Width of transition zone (Kn 0.01 to 10).
        knudsen_at_altitude: Knudsen number at the queried altitude.
        cd_corrected: Drag coefficient corrected for flow regime.
        cd_free_molecular: Free-molecular drag coefficient (Kn >> 1).
        cd_continuum: Continuum drag coefficient (Kn << 1).
        lifetime_correction_factor: Ratio of corrected to constant-C_D lifetime.
    """
    boundary_layer_altitude_km: float
    boundary_layer_thickness_km: float
    knudsen_at_altitude: float
    cd_corrected: float
    cd_free_molecular: float
    cd_continuum: float
    lifetime_correction_factor: float


def _number_density(altitude_km: float, model: AtmosphereModel = AtmosphereModel.HIGH_ACTIVITY) -> float:
    """Compute atmospheric number density from mass density.

    n = rho / m_molecule

    Args:
        altitude_km: Altitude above Earth surface (km).
        model: Atmosphere model table to use.

    Returns:
        Number density in particles/m^3.
    """
    rho = atmospheric_density(altitude_km, model=model)
    return rho / _M_MOLECULE


def _mean_free_path(altitude_km: float, model: AtmosphereModel = AtmosphereModel.HIGH_ACTIVITY) -> float:
    """Compute mean free path from number density.

    lambda_mfp = 1 / (sqrt(2) * pi * d^2 * n)

    Args:
        altitude_km: Altitude above Earth surface (km).
        model: Atmosphere model table to use.

    Returns:
        Mean free path in meters.
    """
    n = _number_density(altitude_km, model=model)
    if n < 1e-10:
        return float('inf')
    return 1.0 / (math.sqrt(2.0) * math.pi * _SIGMA_MOLECULE ** 2 * n)


def _knudsen_number(altitude_km: float, char_length_m: float, model: AtmosphereModel = AtmosphereModel.HIGH_ACTIVITY) -> float:
    """Compute Knudsen number: Kn = lambda_mfp / L_char.

    Args:
        altitude_km: Altitude above Earth surface (km).
        char_length_m: Characteristic length of the spacecraft (m).
        model: Atmosphere model table to use.

    Returns:
        Knudsen number (dimensionless).
    """
    if char_length_m <= 0:
        raise ValueError(f"char_length_m must be positive, got {char_length_m}")
    mfp = _mean_free_path(altitude_km, model=model)
    return mfp / char_length_m


def _cd_interpolation(kn: float, cd_fm: float = 2.2, cd_cont: float = 1.0, kn_ref: float = 1.0) -> float:
    """Interpolate drag coefficient between free-molecular and continuum.

    C_D(Kn) = C_D_fm * f(Kn) + C_D_cont * (1 - f(Kn))
    where f(Kn) = 1 / (1 + Kn_ref / Kn) = Kn / (Kn + Kn_ref)

    For Kn >> Kn_ref: f -> 1, C_D -> C_D_fm (free molecular)
    For Kn << Kn_ref: f -> 0, C_D -> C_D_cont (continuum)

    Args:
        kn: Knudsen number.
        cd_fm: Free-molecular drag coefficient (default 2.2).
        cd_cont: Continuum drag coefficient (default 1.0).
        kn_ref: Reference Knudsen number for transition (default 1.0).

    Returns:
        Interpolated drag coefficient.
    """
    if kn <= 0:
        return cd_cont
    f = kn / (kn + kn_ref)
    return cd_fm * f + cd_cont * (1.0 - f)


def compute_flow_regime_transition(
    altitude_km: float,
    char_length_m: float = 1.0,
    cd_free_molecular: float = 2.2,
    cd_continuum: float = 1.0,
    model: AtmosphereModel = AtmosphereModel.HIGH_ACTIVITY,
) -> FlowRegimeTransition:
    """Compute flow regime transition analysis for a spacecraft.

    Determines the Knudsen number at the given altitude, interpolates
    the drag coefficient between free-molecular and continuum regimes,
    and finds the boundary layer altitude where Kn = 1.

    Args:
        altitude_km: Altitude above Earth surface (km).
        char_length_m: Characteristic length of the spacecraft (m).
        cd_free_molecular: Free-molecular C_D (default 2.2, flat plate).
        cd_continuum: Continuum C_D (default 1.0, sphere).
        model: Atmosphere model table to use.

    Returns:
        FlowRegimeTransition with Knudsen number, C_D, and boundary altitude.

    Raises:
        ValueError: If char_length_m <= 0 or altitude_km out of table range.
    """
    if char_length_m <= 0:
        raise ValueError(f"char_length_m must be positive, got {char_length_m}")

    table = _MODEL_TABLES[model]
    alt_min = table[0][0]
    alt_max = table[-1][0]

    # Knudsen at query altitude
    kn = _knudsen_number(altitude_km, char_length_m, model=model)

    # Interpolated C_D at query altitude
    cd_corr = _cd_interpolation(kn, cd_free_molecular, cd_continuum)

    # Find boundary layer altitude where Kn = 1 via bisection
    # Kn increases with altitude, so search upward
    bl_alt = altitude_km  # fallback
    lo_alt = alt_min
    hi_alt = alt_max

    for _ in range(64):
        mid_alt = (lo_alt + hi_alt) / 2.0
        kn_mid = _knudsen_number(mid_alt, char_length_m, model=model)
        if kn_mid < 1.0:
            lo_alt = mid_alt
        else:
            hi_alt = mid_alt
    bl_alt = (lo_alt + hi_alt) / 2.0

    # Boundary layer thickness: zone where Kn goes from 0.01 to 10
    # Lower bound: Kn = 0.01
    lo_kn_alt = alt_min
    hi_kn_alt = alt_max
    for _ in range(64):
        mid_alt = (lo_kn_alt + hi_kn_alt) / 2.0
        kn_mid = _knudsen_number(mid_alt, char_length_m, model=model)
        if kn_mid < 0.01:
            lo_kn_alt = mid_alt
        else:
            hi_kn_alt = mid_alt
    lower_bound = (lo_kn_alt + hi_kn_alt) / 2.0

    # Upper bound: Kn = 10
    lo_kn_alt = alt_min
    hi_kn_alt = alt_max
    for _ in range(64):
        mid_alt = (lo_kn_alt + hi_kn_alt) / 2.0
        kn_mid = _knudsen_number(mid_alt, char_length_m, model=model)
        if kn_mid < 10.0:
            lo_kn_alt = mid_alt
        else:
            hi_kn_alt = mid_alt
    upper_bound = (lo_kn_alt + hi_kn_alt) / 2.0

    bl_thickness = upper_bound - lower_bound

    # Lifetime correction factor: ratio of corrected to constant C_D
    # At the query altitude, lifetime ~ 1/C_D, so correction = C_D_const / C_D_corr
    cd_const = cd_free_molecular  # typical assumption for LEO
    if cd_corr > 1e-15:
        lifetime_correction = cd_const / cd_corr
    else:
        lifetime_correction = 1.0

    return FlowRegimeTransition(
        boundary_layer_altitude_km=bl_alt,
        boundary_layer_thickness_km=bl_thickness,
        knudsen_at_altitude=kn,
        cd_corrected=cd_corr,
        cd_free_molecular=cd_free_molecular,
        cd_continuum=cd_continuum,
        lifetime_correction_factor=lifetime_correction,
    )
