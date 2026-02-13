# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Maintenance planning compositions.

Composes perturbation models (J2, J3, drag), station-keeping, and
maneuver modules to produce perturbation budgets and maintenance
schedules.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.propagation import OrbitalState
from humeris.domain.atmosphere import DragConfig, semi_major_axis_decay_rate
from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    j2_raan_rate,
    j2_arg_perigee_rate,
)
from humeris.domain.station_keeping import drag_compensation_dv_per_year

_R_EARTH = OrbitalConstants.R_EARTH
_MU = OrbitalConstants.MU_EARTH
_J2 = OrbitalConstants.J2_EARTH
_J3 = OrbitalConstants.J3_EARTH
_SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass(frozen=True)
class ElementPerturbation:
    """Perturbation rate for a single orbital element."""
    element_name: str
    j2_rate: float
    drag_rate: float
    total_rate: float
    dominant_source: str


@dataclass(frozen=True)
class PerturbationBudget:
    """Perturbation budget for all orbital elements."""
    elements: tuple[ElementPerturbation, ...]
    altitude_km: float


@dataclass(frozen=True)
class MaintenanceBurn:
    """A single maintenance burn."""
    time: datetime
    element: str
    delta_v_ms: float
    description: str


@dataclass(frozen=True)
class MaintenanceSchedule:
    """Complete maintenance schedule."""
    burns: tuple[MaintenanceBurn, ...]
    total_dv_per_year_ms: float
    burn_frequency_per_year: int
    dominant_correction: str


def compute_perturbation_budget(
    state: OrbitalState,
    drag_config: DragConfig | None = None,
) -> PerturbationBudget:
    """Compute perturbation budget for all orbital elements.

    Evaluates J2 rates for RAAN and argument of perigee,
    drag rate for semi-major axis.

    Args:
        state: Satellite orbital state.
        drag_config: Drag configuration (optional, for drag rates).

    Returns:
        PerturbationBudget with element-wise perturbation breakdown.
    """
    a = state.semi_major_axis_m
    e = state.eccentricity
    i = state.inclination_rad
    n = state.mean_motion_rad_s
    alt_km = (a - _R_EARTH) / 1000.0

    elements: list[ElementPerturbation] = []

    # RAAN: dominated by J2
    raan_j2 = abs(j2_raan_rate(n, a, e, i))
    raan_drag = 0.0  # drag doesn't directly affect RAAN
    raan_total = raan_j2 + raan_drag
    raan_dominant = "j2" if raan_j2 > raan_drag else "drag"
    elements.append(ElementPerturbation(
        element_name="raan",
        j2_rate=raan_j2,
        drag_rate=raan_drag,
        total_rate=raan_total,
        dominant_source=raan_dominant,
    ))

    # Argument of perigee: dominated by J2
    argp_j2 = abs(j2_arg_perigee_rate(n, a, e, i))
    argp_drag = 0.0
    argp_total = argp_j2 + argp_drag
    argp_dominant = "j2" if argp_j2 > argp_drag else "drag"
    elements.append(ElementPerturbation(
        element_name="arg_perigee",
        j2_rate=argp_j2,
        drag_rate=argp_drag,
        total_rate=argp_total,
        dominant_source=argp_dominant,
    ))

    # Semi-major axis: dominated by drag (J2 is conservative for circular orbits)
    sma_j2 = 0.0  # J2 is conservative (no secular SMA change for circular)
    if drag_config is not None:
        sma_drag = abs(semi_major_axis_decay_rate(a, e, drag_config))
    else:
        sma_drag = 0.0
    sma_total = sma_j2 + sma_drag
    sma_dominant = "drag" if sma_drag > sma_j2 else "j2"
    elements.append(ElementPerturbation(
        element_name="sma",
        j2_rate=sma_j2,
        drag_rate=sma_drag,
        total_rate=sma_total,
        dominant_source=sma_dominant,
    ))

    return PerturbationBudget(
        elements=tuple(elements),
        altitude_km=alt_km,
    )


def compute_maintenance_schedule(
    state: OrbitalState,
    drag_config: DragConfig,
    epoch: datetime,
    altitude_tolerance_km: float = 5.0,
    mission_duration_days: float = 365.0,
) -> MaintenanceSchedule:
    """Compute maintenance burn schedule.

    Determines burn frequency based on altitude tolerance and drag rate,
    computes per-burn dV, and generates a schedule.

    Args:
        state: Satellite orbital state.
        drag_config: Drag configuration.
        epoch: Mission start epoch.
        altitude_tolerance_km: Maximum allowed altitude drop (km).
        mission_duration_days: Mission duration (days).

    Returns:
        MaintenanceSchedule with burns and total dV.
    """
    a = state.semi_major_axis_m
    e = state.eccentricity
    alt_km = (a - _R_EARTH) / 1000.0
    n = state.mean_motion_rad_s

    # Compute SMA decay rate
    da_dt = abs(semi_major_axis_decay_rate(a, e, drag_config))

    # Time to decay by tolerance
    tolerance_m = altitude_tolerance_km * 1000.0
    if da_dt > 0:
        time_to_tolerance_s = tolerance_m / da_dt
    else:
        time_to_tolerance_s = mission_duration_days * 86400.0

    # Burn frequency
    mission_s = mission_duration_days * 86400.0
    if time_to_tolerance_s > 0:
        num_burns = max(1, int(np.ceil(mission_s / time_to_tolerance_s)))
    else:
        num_burns = 1

    burn_interval_s = mission_s / num_burns

    # Annual dV from station-keeping
    annual_dv = drag_compensation_dv_per_year(alt_km, drag_config)
    per_burn_dv = annual_dv * (burn_interval_s / _SECONDS_PER_YEAR)

    burns: list[MaintenanceBurn] = []
    for i in range(num_burns):
        burn_time = epoch + timedelta(seconds=(i + 1) * burn_interval_s)
        burns.append(MaintenanceBurn(
            time=burn_time,
            element="sma",
            delta_v_ms=per_burn_dv,
            description=f"Altitude maintenance burn {i+1}/{num_burns}",
        ))

    # Annualize
    burns_per_year = int(num_burns * (_SECONDS_PER_YEAR / mission_s))
    if burns_per_year < 1:
        burns_per_year = num_burns

    # Dominant correction
    pert = compute_perturbation_budget(state, drag_config)
    dominant = max(pert.elements, key=lambda el: el.total_rate)

    return MaintenanceSchedule(
        burns=tuple(burns),
        total_dv_per_year_ms=annual_dv,
        burn_frequency_per_year=burns_per_year,
        dominant_correction=dominant.element_name,
    )
