# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Maintenance planning compositions.

Composes perturbation models (J2, J3, drag), station-keeping, and
maneuver modules to produce perturbation budgets and maintenance
schedules.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

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
    density_func: Optional[Callable[[float, datetime], float]] = None,
    epoch: datetime | None = None,
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
        if density_func is not None:
            if epoch is not None:
                ref_epoch = epoch
            else:
                from datetime import timezone
                ref_epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
            rho = density_func(alt_km, ref_epoch)
            v = math.sqrt(OrbitalConstants.MU_EARTH / a)
            sma_drag = abs(-rho * v * drag_config.ballistic_coefficient * a)
        else:
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
    density_func: Optional[Callable[[float, datetime], float]] = None,
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
    if density_func is not None:
        rho = density_func(alt_km, epoch)
        v = math.sqrt(OrbitalConstants.MU_EARTH / a)
        da_dt = abs(-rho * v * drag_config.ballistic_coefficient * a)
    else:
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
    annual_dv = drag_compensation_dv_per_year(
        alt_km, drag_config, density_func=density_func, epoch=epoch,
    )
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
    pert = compute_perturbation_budget(
        state, drag_config, density_func=density_func, epoch=epoch,
    )
    dominant = max(pert.elements, key=lambda el: el.total_rate)

    return MaintenanceSchedule(
        burns=tuple(burns),
        total_dv_per_year_ms=annual_dv,
        burn_frequency_per_year=burns_per_year,
        dominant_correction=dominant.element_name,
    )


# ── P45: Vorticity in Orbital Element Space ───────────────────────


@dataclass(frozen=True)
class OrbitalVorticityField:
    """Vorticity analysis of J2-driven velocity field in (a, i) space.

    The J2 perturbation induces secular rates dOmega/dt(a, i) and
    domega/dt(a, i) that define a velocity field in orbital element space.
    The curl (vorticity) of this field measures differential precession
    shear that drives constellation strain.

    Attributes:
        vorticity_grid: 2D tuple of vorticity values omega_z(a, i)
            in rad/s per (m * rad). Shape: (n_a, n_i).
        circulation: Line integral of v . dl around the constellation
            boundary in the (a, i) plane (rad/s * m).
        max_vorticity: Maximum absolute vorticity in the grid (rad/s per m*rad).
        shear_rate_rad_per_year: Maximum shear rate converted to rad/year.
        constellation_strain_time_years: Time for one full differential
            rotation cycle at max vorticity: 2*pi / max_vorticity_annualized.
    """
    vorticity_grid: tuple
    circulation: float
    max_vorticity: float
    shear_rate_rad_per_year: float
    constellation_strain_time_years: float


def compute_orbital_vorticity_field(
    a_range_m: tuple[float, float],
    i_range_rad: tuple[float, float],
    n_a: int = 20,
    n_i: int = 20,
    eccentricity: float = 0.0,
) -> OrbitalVorticityField:
    """Compute vorticity of the J2 velocity field in (a, i) space.

    The J2 perturbation defines a 2D velocity field:
        v_Omega(a, i) = dOmega/dt  (RAAN precession rate)
        v_omega(a, i) = domega/dt  (arg perigee rate)

    Treating (a, i) as spatial coordinates, the 2D curl (vorticity) is:
        omega_z = d(v_omega)/da - d(v_Omega)/di

    computed via central finite differences on a grid.

    Circulation is the line integral of v . dl around the rectangular
    boundary in (a, i) space, which by Green's theorem equals the area
    integral of vorticity.

    Args:
        a_range_m: (a_min, a_max) semi-major axis range in metres.
        i_range_rad: (i_min, i_max) inclination range in radians.
        n_a: Number of grid points in a-direction.
        n_i: Number of grid points in i-direction.
        eccentricity: Eccentricity (default 0 for circular orbits).

    Returns:
        OrbitalVorticityField with vorticity grid and derived quantities.

    Raises:
        ValueError: If ranges are invalid or grid too small.
    """
    if a_range_m[0] >= a_range_m[1]:
        raise ValueError(
            f"a_range_m must be (a_min, a_max) with a_min < a_max, "
            f"got {a_range_m}"
        )
    if i_range_rad[0] >= i_range_rad[1]:
        raise ValueError(
            f"i_range_rad must be (i_min, i_max) with i_min < i_max, "
            f"got {i_range_rad}"
        )
    if n_a < 3 or n_i < 3:
        raise ValueError(f"Grid must be at least 3x3, got {n_a}x{n_i}")

    mu = _MU
    e = eccentricity

    a_vals = np.linspace(a_range_m[0], a_range_m[1], n_a)
    i_vals = np.linspace(i_range_rad[0], i_range_rad[1], n_i)
    da = float(a_vals[1] - a_vals[0])
    di = float(i_vals[1] - i_vals[0])

    # Compute velocity field: v_Omega[j, k] and v_omega[j, k]
    v_omega_grid = np.zeros((n_a, n_i))
    v_Omega_grid = np.zeros((n_a, n_i))

    for j in range(n_a):
        a = float(a_vals[j])
        n_mean = math.sqrt(mu / a ** 3)
        for k in range(n_i):
            inc = float(i_vals[k])
            v_Omega_grid[j, k] = j2_raan_rate(n_mean, a, e, inc)
            v_omega_grid[j, k] = j2_arg_perigee_rate(n_mean, a, e, inc)

    # Compute vorticity via central finite differences:
    # omega_z = d(v_omega)/da - d(v_Omega)/di
    vorticity = np.zeros((n_a, n_i))
    for j in range(1, n_a - 1):
        for k in range(1, n_i - 1):
            dv_omega_da = (v_omega_grid[j + 1, k] - v_omega_grid[j - 1, k]) / (2.0 * da)
            dv_Omega_di = (v_Omega_grid[j, k + 1] - v_Omega_grid[j, k - 1]) / (2.0 * di)
            vorticity[j, k] = dv_omega_da - dv_Omega_di

    # Forward/backward differences at boundaries
    for k in range(n_i):
        dv_omega_da_0 = (v_omega_grid[1, k] - v_omega_grid[0, k]) / da
        dv_omega_da_n = (v_omega_grid[-1, k] - v_omega_grid[-2, k]) / da
        if 0 < k < n_i - 1:
            dv_Omega_di_k = (v_Omega_grid[0, k + 1] - v_Omega_grid[0, k - 1]) / (2.0 * di)
            dv_Omega_di_k_n = (v_Omega_grid[-1, k + 1] - v_Omega_grid[-1, k - 1]) / (2.0 * di)
        elif k == 0:
            dv_Omega_di_k = (v_Omega_grid[0, 1] - v_Omega_grid[0, 0]) / di
            dv_Omega_di_k_n = (v_Omega_grid[-1, 1] - v_Omega_grid[-1, 0]) / di
        else:
            dv_Omega_di_k = (v_Omega_grid[0, -1] - v_Omega_grid[0, -2]) / di
            dv_Omega_di_k_n = (v_Omega_grid[-1, -1] - v_Omega_grid[-1, -2]) / di
        vorticity[0, k] = dv_omega_da_0 - dv_Omega_di_k
        vorticity[-1, k] = dv_omega_da_n - dv_Omega_di_k_n

    for j in range(1, n_a - 1):
        dv_Omega_di_0 = (v_Omega_grid[j, 1] - v_Omega_grid[j, 0]) / di
        dv_Omega_di_n = (v_Omega_grid[j, -1] - v_Omega_grid[j, -2]) / di
        dv_omega_da_j = (v_omega_grid[j + 1, 0] - v_omega_grid[j - 1, 0]) / (2.0 * da)
        dv_omega_da_j_n = (v_omega_grid[j + 1, -1] - v_omega_grid[j - 1, -1]) / (2.0 * da)
        vorticity[j, 0] = dv_omega_da_j - dv_Omega_di_0
        vorticity[j, -1] = dv_omega_da_j_n - dv_Omega_di_n

    max_vorticity = float(np.max(np.abs(vorticity)))

    # Circulation: line integral around boundary of (a, i) rectangle
    # Gamma = integral v . dl = sum of 4 edges
    # Bottom edge (i = i_min, a from a_min to a_max): v_omega along +a direction
    circ_bottom = float(np.trapezoid(v_omega_grid[:, 0], a_vals))
    # Right edge (a = a_max, i from i_min to i_max): v_Omega is not along i,
    # but the velocity component along +i direction is v_Omega (RAAN rate)
    # Actually for the line integral in (a,i) space:
    # bottom: v_a * da (v_a = v_omega component along a)
    # right:  v_i * di (v_i = v_Omega component along i)
    # top:    -v_a * da (reversed direction)
    # left:   -v_i * di (reversed direction)
    circ_right = float(np.trapezoid(v_Omega_grid[-1, :], i_vals))
    circ_top = -float(np.trapezoid(v_omega_grid[:, -1], a_vals))
    circ_left = -float(np.trapezoid(v_Omega_grid[0, :], i_vals))
    circulation = circ_bottom + circ_right + circ_top + circ_left

    # Convert max vorticity to annual shear rate
    # vorticity has units rad/s / (m * rad), multiply by characteristic scales
    # Shear rate = max_vorticity * da_char * di_char gives rad/s
    # But for a direct measure, use vorticity * area element = circulation density
    shear_rate_per_s = max_vorticity * da * di  # rad/s from one grid cell
    shear_rate_per_year = shear_rate_per_s * _SECONDS_PER_YEAR

    # Strain time: time for one full differential rotation cycle
    if shear_rate_per_year > 0:
        strain_time = 2.0 * math.pi / shear_rate_per_year
    else:
        strain_time = float('inf')

    # Convert vorticity grid to nested tuple
    vorticity_tuple = tuple(
        tuple(float(vorticity[j, k]) for k in range(n_i))
        for j in range(n_a)
    )

    return OrbitalVorticityField(
        vorticity_grid=vorticity_tuple,
        circulation=circulation,
        max_vorticity=max_vorticity,
        shear_rate_rad_per_year=shear_rate_per_year,
        constellation_strain_time_years=strain_time,
    )
