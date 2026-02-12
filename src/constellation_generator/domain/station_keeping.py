# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""
Station-keeping delta-V budgets and propellant computation.

Drag compensation, plane maintenance, Tsiolkovsky rocket equation,
and combined operational lifetime budgets.

No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.atmosphere import (
    AtmosphereModel,
    DragConfig,
    atmospheric_density,
    semi_major_axis_decay_rate,
)


_G0 = 9.80665  # standard gravity m/s²
_SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass(frozen=True)
class StationKeepingConfig:
    """Configuration for station-keeping budget computation."""
    target_altitude_km: float
    inclination_deg: float
    drag_config: DragConfig
    isp_s: float
    dry_mass_kg: float
    propellant_mass_kg: float


@dataclass(frozen=True)
class StationKeepingBudget:
    """Station-keeping delta-V and propellant budget."""
    drag_dv_per_year_ms: float
    plane_dv_per_year_ms: float
    total_dv_per_year_ms: float
    propellant_per_year_kg: float
    operational_lifetime_years: float
    total_dv_capacity_ms: float


def drag_compensation_dv_per_year(
    altitude_km: float,
    drag_config: DragConfig,
    atmosphere_model: AtmosphereModel | None = None,
) -> float:
    """Annual delta-V for drag compensation at given altitude.

    Uses vis-viva linearization: dV/dt = |da/dt| * n / 2, annualized.
    Ref: Wertz SMAD Ch. 6.

    Args:
        altitude_km: Target orbit altitude (km).
        drag_config: Satellite drag configuration.

    Returns:
        Delta-V in m/s per year.
    """
    a = OrbitalConstants.R_EARTH + altitude_km * 1000.0
    n = math.sqrt(OrbitalConstants.MU_EARTH / a ** 3)
    decay_kwargs: dict = {}
    if atmosphere_model is not None:
        decay_kwargs["model"] = atmosphere_model
    da_dt = semi_major_axis_decay_rate(a, 0.0, drag_config, **decay_kwargs)
    dv_per_second = abs(da_dt) * n / 2.0
    return dv_per_second * _SECONDS_PER_YEAR


def plane_maintenance_dv_per_year(
    inclination_deg: float,
    altitude_km: float,
    delta_inclination_deg: float = 0.05,
) -> float:
    """Annual delta-V for orbital plane maintenance.

    dV = 2 * v * sin(di/2) for inclination correction budget.

    Args:
        inclination_deg: Orbit inclination (degrees) — unused in formula
            but kept for API consistency and future J2 drift models.
        altitude_km: Orbit altitude (km).
        delta_inclination_deg: Annual inclination correction budget (degrees).

    Returns:
        Delta-V in m/s per year.
    """
    a = OrbitalConstants.R_EARTH + altitude_km * 1000.0
    v = math.sqrt(OrbitalConstants.MU_EARTH / a)
    di_rad = math.radians(delta_inclination_deg)
    return 2.0 * v * math.sin(di_rad / 2.0)


def tsiolkovsky_dv(
    isp_s: float,
    dry_mass_kg: float,
    propellant_mass_kg: float,
) -> float:
    """Tsiolkovsky rocket equation: delta-V from propellant mass.

    dV = Isp * g0 * ln(m0 / mf)

    Args:
        isp_s: Specific impulse (seconds).
        dry_mass_kg: Dry mass (kg).
        propellant_mass_kg: Propellant mass (kg).

    Returns:
        Delta-V capacity in m/s.

    Raises:
        ValueError: If dry_mass <= 0 or propellant < 0.
    """
    if dry_mass_kg <= 0:
        raise ValueError(f"dry_mass_kg must be positive, got {dry_mass_kg}")
    if propellant_mass_kg < 0:
        raise ValueError(f"propellant_mass_kg must be non-negative, got {propellant_mass_kg}")
    if propellant_mass_kg == 0.0:
        return 0.0
    m0 = dry_mass_kg + propellant_mass_kg
    return isp_s * _G0 * math.log(m0 / dry_mass_kg)


def propellant_mass_for_dv(
    isp_s: float,
    dry_mass_kg: float,
    dv_ms: float,
) -> float:
    """Propellant mass required for a given delta-V.

    m_prop = m_dry * (exp(dV / (Isp * g0)) - 1)

    Args:
        isp_s: Specific impulse (seconds).
        dry_mass_kg: Dry mass (kg).
        dv_ms: Required delta-V (m/s).

    Returns:
        Propellant mass in kg.
    """
    return dry_mass_kg * (math.exp(dv_ms / (isp_s * _G0)) - 1.0)


def compute_station_keeping_budget(
    config: StationKeepingConfig,
) -> StationKeepingBudget:
    """Compute combined station-keeping budget.

    Combines drag compensation and plane maintenance delta-V,
    computes total capacity via Tsiolkovsky, derives operational lifetime.

    Args:
        config: Station-keeping configuration.

    Returns:
        StationKeepingBudget with annual budgets and lifetime.
    """
    drag_dv = drag_compensation_dv_per_year(config.target_altitude_km, config.drag_config)
    plane_dv = plane_maintenance_dv_per_year(config.inclination_deg, config.target_altitude_km)
    total_dv_year = drag_dv + plane_dv

    total_capacity = tsiolkovsky_dv(
        config.isp_s, config.dry_mass_kg, config.propellant_mass_kg,
    )

    if total_dv_year > 0:
        lifetime_years = total_capacity / total_dv_year
    else:
        lifetime_years = float('inf')

    propellant_year = propellant_mass_for_dv(
        config.isp_s, config.dry_mass_kg, total_dv_year,
    )

    return StationKeepingBudget(
        drag_dv_per_year_ms=drag_dv,
        plane_dv_per_year_ms=plane_dv,
        total_dv_per_year_ms=total_dv_year,
        propellant_per_year_kg=propellant_year,
        operational_lifetime_years=lifetime_years,
        total_dv_capacity_ms=total_capacity,
    )
