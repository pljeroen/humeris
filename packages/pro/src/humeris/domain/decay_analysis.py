# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Exponential decay scale map extraction.

Extracts and compares exponential scale parameters from all exp(-kx)
processes the codebase computes: atmospheric density, collision survival,
propellant depletion, and orbit decay.

"""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from humeris.domain.propagation import OrbitalState
from humeris.domain.atmosphere import DragConfig, atmospheric_density
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.station_keeping import (
    drag_compensation_dv_per_year,
    propellant_mass_for_dv,
)

_R_EARTH = OrbitalConstants.R_EARTH
_MU = OrbitalConstants.MU_EARTH
_G0 = 9.80665
_SECONDS_PER_YEAR = 365.25 * 86400.0
_LN2 = float(np.log(2.0))


@dataclass(frozen=True)
class ExponentialProcess:
    """A single exponential decay process."""
    name: str
    scale_parameter: float
    scale_unit: str
    current_value: float
    half_life: float
    e_folding: float


@dataclass(frozen=True)
class ExponentialScaleMap:
    """Unified view of all exponential decay processes."""
    processes: tuple
    fastest_process: str
    slowest_process: str
    scale_ratio: float


def _atmospheric_scale_height(altitude_km: float, delta_km: float = 1.0) -> float:
    """Compute local scale height via finite-difference of atmospheric_density.

    H = -delta_h / ln(rho(h+delta) / rho(h))
    """
    rho_base = atmospheric_density(altitude_km)
    rho_upper = atmospheric_density(altitude_km + delta_km)
    if rho_upper <= 0 or rho_base <= 0:
        return delta_km
    ratio = rho_upper / rho_base
    if ratio >= 1.0 or ratio <= 0.0:
        return delta_km
    return -delta_km / float(np.log(ratio))


def compute_exponential_scale_map(
    state: OrbitalState,
    drag_config: DragConfig,
    epoch: datetime,
    isp_s: float,
    dry_mass_kg: float,
    propellant_budget_kg: float,
    conjunction_rate_per_year: float = 0.1,
    density_func: Optional[Callable[[float, datetime], float]] = None,
) -> ExponentialScaleMap:
    """Extract and compare exponential scale parameters.

    Processes:
    1. Atmospheric density: scale height H (km)
    2. Collision survival: 1/lambda_conj (years)
    3. Propellant depletion: budget/rate (years)
    4. Orbit decay: altitude/decay_rate (years)

    All half-lives are normalized to the same unit (years) for comparison.
    """
    alt_km = (state.semi_major_axis_m - _R_EARTH) / 1000.0
    processes: list[ExponentialProcess] = []

    # 1. Atmospheric scale height
    h_km = _atmospheric_scale_height(alt_km)
    # Convert to time scale: how long for density to change by factor e
    # at current altitude. Use decay rate for conversion.
    # Half-life in km: H * ln(2)
    # We express as km for the scale parameter, but compute a time half-life too
    # For unified comparison, convert H to a time: t_H = H_km / |da/dt| in km/year
    a = state.semi_major_axis_m
    n = state.mean_motion_rad_s
    if density_func is not None:
        rho = density_func(alt_km, epoch)
    else:
        rho = atmospheric_density(alt_km)
    v = float(np.sqrt(_MU / a))
    bc = drag_config.ballistic_coefficient
    da_dt_m_s = rho * v * bc * a  # m/s (positive = rate of decay)

    # Underflow guard: at high altitudes (>1000 km) atmospheric density approaches
    # machine epsilon, producing denormalized products. Clamp to infinity.
    if da_dt_m_s < 1e-20:
        da_dt_km_year = 0.0
        atmo_half_life_years = float('inf')
    else:
        da_dt_km_year = (da_dt_m_s / 1000.0) * _SECONDS_PER_YEAR
        atmo_half_life_years = (h_km * _LN2) / da_dt_km_year if da_dt_km_year > 0 else float('inf')
        atmo_half_life_years = max(atmo_half_life_years, 1e-15)

    processes.append(ExponentialProcess(
        name="atmospheric_density",
        scale_parameter=h_km,
        scale_unit="km",
        current_value=rho,
        half_life=atmo_half_life_years,
        e_folding=h_km,
    ))

    # 2. Collision survival: P(t) = exp(-lambda * t)
    # tau = 1/lambda, half_life = ln(2)/lambda
    if conjunction_rate_per_year > 0:
        conj_tau_years = 1.0 / conjunction_rate_per_year
        conj_half_life = _LN2 * conj_tau_years
    else:
        conj_tau_years = float('inf')
        conj_half_life = float('inf')

    processes.append(ExponentialProcess(
        name="collision_survival",
        scale_parameter=conj_tau_years,
        scale_unit="years",
        current_value=1.0,
        half_life=conj_half_life,
        e_folding=conj_tau_years,
    ))

    # 3. Propellant depletion: linear consumption approximated as exponential
    # Rocket equation: m_prop = m_dry * (exp(dv/(Isp*g0)) - 1)
    # Annual dV from drag compensation
    dv_per_year = drag_compensation_dv_per_year(
        alt_km, drag_config, density_func=density_func, epoch=epoch,
    )
    max_dv = isp_s * _G0 * 20.0
    capped_dv = min(dv_per_year, max_dv)
    prop_per_year = propellant_mass_for_dv(isp_s, dry_mass_kg, capped_dv)

    if prop_per_year > 0:
        # Depletion time scale: budget / rate
        prop_tau_years = propellant_budget_kg / prop_per_year
        prop_half_life = prop_tau_years * _LN2
    else:
        prop_tau_years = float('inf')
        prop_half_life = float('inf')

    processes.append(ExponentialProcess(
        name="propellant_depletion",
        scale_parameter=prop_tau_years,
        scale_unit="years",
        current_value=propellant_budget_kg,
        half_life=prop_half_life,
        e_folding=prop_tau_years,
    ))

    # 4. Orbit decay: altitude decrease
    # tau = alt / decay_rate
    if da_dt_km_year > 0:
        decay_tau_years = alt_km / da_dt_km_year
        decay_half_life = decay_tau_years * _LN2
    else:
        decay_tau_years = float('inf')
        decay_half_life = float('inf')

    processes.append(ExponentialProcess(
        name="orbit_decay",
        scale_parameter=decay_tau_years,
        scale_unit="years",
        current_value=alt_km,
        half_life=decay_half_life,
        e_folding=decay_tau_years,
    ))

    # Identify fastest and slowest by half-life
    sorted_procs = sorted(processes, key=lambda p: p.half_life)
    fastest = sorted_procs[0].name
    slowest = sorted_procs[-1].name

    # Scale ratio: slowest / fastest
    if sorted_procs[0].half_life > 0:
        scale_ratio = sorted_procs[-1].half_life / sorted_procs[0].half_life
    else:
        scale_ratio = float('inf')

    return ExponentialScaleMap(
        processes=tuple(processes),
        fastest_process=fastest,
        slowest_process=slowest,
        scale_ratio=scale_ratio,
    )
