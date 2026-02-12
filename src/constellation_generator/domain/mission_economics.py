# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Survival-weighted design optimization and reliability-weighted cost.

Composes mass efficiency frontier with lifetime survival curves and
mission availability to produce operationally realistic cost metrics.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime

from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.design_optimization import (
    compute_mass_efficiency_frontier,
    MassEfficiencyFrontier,
)
from constellation_generator.domain.lifetime import compute_orbit_lifetime
from constellation_generator.domain.statistical_analysis import (
    compute_lifetime_survival_curve,
    compute_mission_availability,
)
from constellation_generator.domain.radiation import compute_orbit_radiation_summary


@dataclass(frozen=True)
class SurvivalWeightedEfficiencyPoint:
    """Single point on the survival-weighted mass efficiency frontier."""
    altitude_km: float
    mass_efficiency: float
    expected_lifetime_fraction: float
    weighted_efficiency: float


@dataclass(frozen=True)
class SurvivalWeightedMassFrontier:
    """Mass frontier weighted by orbital survival probability."""
    points: tuple
    unweighted_optimal_km: float
    weighted_optimal_km: float
    altitude_shift_km: float


@dataclass(frozen=True)
class ReliabilityWeightedCostPoint:
    """Single point in the reliability-weighted cost profile."""
    altitude_km: float
    constellation_mass_kg: float
    integrated_availability_days: float
    dose_factor: float
    rwccd_kg_per_day: float


@dataclass(frozen=True)
class ReliabilityWeightedCostProfile:
    """Cost per useful coverage-day across altitudes."""
    points: tuple
    optimal_altitude_km: float
    min_rwccd: float
    mass_optimal_altitude_km: float


def compute_survival_weighted_frontier(
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    injection_altitude_km: float,
    mission_years: float,
    num_sats: int,
    epoch: datetime,
    alt_min_km: float = 300.0,
    alt_max_km: float = 800.0,
    alt_step_km: float = 25.0,
) -> SurvivalWeightedMassFrontier:
    """Compute mass efficiency weighted by expected orbital lifetime.

    eta_weighted(h) = eta(h) * E[T|h] / T_mission
    """
    frontier = compute_mass_efficiency_frontier(
        drag_config, isp_s, dry_mass_kg, injection_altitude_km,
        mission_years, num_sats,
        alt_min_km=alt_min_km, alt_max_km=alt_max_km, alt_step_km=alt_step_km,
    )

    mission_days = mission_years * 365.25
    points = []

    for fp in frontier.points:
        alt_km = fp.altitude_km
        a_m = OrbitalConstants.R_EARTH + alt_km * 1000.0

        # Compute orbit lifetime
        try:
            lifetime = compute_orbit_lifetime(
                a_m, 0.0, drag_config, epoch,
                step_days=max(1.0, mission_days / 100.0),
                max_years=mission_years * 2.0,
            )
            survival = compute_lifetime_survival_curve(lifetime)
            lifetime_frac = min(1.0, lifetime.lifetime_days / mission_days)
        except (ValueError, ZeroDivisionError):
            lifetime_frac = 0.0

        weighted = fp.mass_efficiency * lifetime_frac

        points.append(SurvivalWeightedEfficiencyPoint(
            altitude_km=alt_km,
            mass_efficiency=fp.mass_efficiency,
            expected_lifetime_fraction=lifetime_frac,
            weighted_efficiency=weighted,
        ))

    if not points:
        return SurvivalWeightedMassFrontier(
            points=(), unweighted_optimal_km=0.0,
            weighted_optimal_km=0.0, altitude_shift_km=0.0,
        )

    unweighted_best = max(points, key=lambda p: p.mass_efficiency)
    weighted_best = max(points, key=lambda p: p.weighted_efficiency)

    return SurvivalWeightedMassFrontier(
        points=tuple(points),
        unweighted_optimal_km=unweighted_best.altitude_km,
        weighted_optimal_km=weighted_best.altitude_km,
        altitude_shift_km=weighted_best.altitude_km - unweighted_best.altitude_km,
    )


def compute_reliability_weighted_cost(
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    injection_altitude_km: float,
    mission_years: float,
    num_sats: int,
    epoch: datetime,
    inclination_rad: float,
    alt_min_km: float = 300.0,
    alt_max_km: float = 800.0,
    alt_step_km: float = 25.0,
    propellant_budget_kg: float = 20.0,
    conjunction_rate_per_year: float = 0.1,
    dose_limit_rad: float = 5000.0,
) -> ReliabilityWeightedCostProfile:
    """Compute cost per useful coverage-day at each altitude.

    RWCCD(h) = M_constellation(h) / (integral A(t,h) dt * (1 - dose_factor(h)))
    """
    frontier = compute_mass_efficiency_frontier(
        drag_config, isp_s, dry_mass_kg, injection_altitude_km,
        mission_years, num_sats,
        alt_min_km=alt_min_km, alt_max_km=alt_max_km, alt_step_km=alt_step_km,
    )

    points = []

    for fp in frontier.points:
        alt_km = fp.altitude_km
        a_m = OrbitalConstants.R_EARTH + alt_km * 1000.0
        n_rad_s = math.sqrt(OrbitalConstants.MU_EARTH / a_m ** 3)

        # Build a reference OrbitalState at this altitude
        state = OrbitalState(
            semi_major_axis_m=a_m, eccentricity=0.0,
            inclination_rad=inclination_rad,
            raan_rad=0.0, arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=n_rad_s,
            reference_epoch=epoch,
        )

        # Mission availability
        avail = compute_mission_availability(
            state, drag_config, epoch,
            isp_s=isp_s, dry_mass_kg=dry_mass_kg,
            propellant_budget_kg=propellant_budget_kg,
            mission_years=mission_years,
            conjunction_rate_per_year=conjunction_rate_per_year,
        )
        integrated_days = avail.mission_reliability * mission_years * 365.25

        # Radiation dose factor
        rad = compute_orbit_radiation_summary(state, epoch)
        dose_factor = min(0.99, rad.annual_dose_rad / dose_limit_rad)

        # RWCCD
        usable_factor = max(0.01, integrated_days * (1.0 - dose_factor))
        rwccd = fp.constellation_mass_kg / usable_factor

        points.append(ReliabilityWeightedCostPoint(
            altitude_km=alt_km,
            constellation_mass_kg=fp.constellation_mass_kg,
            integrated_availability_days=integrated_days,
            dose_factor=dose_factor,
            rwccd_kg_per_day=rwccd,
        ))

    if not points:
        return ReliabilityWeightedCostProfile(
            points=(), optimal_altitude_km=0.0,
            min_rwccd=float('inf'), mass_optimal_altitude_km=0.0,
        )

    best_rwccd = min(points, key=lambda p: p.rwccd_kg_per_day)
    mass_optimal = frontier.optimal_altitude_km

    return ReliabilityWeightedCostProfile(
        points=tuple(points),
        optimal_altitude_km=best_rwccd.altitude_km,
        min_rwccd=best_rwccd.rwccd_kg_per_day,
        mass_optimal_altitude_km=mass_optimal,
    )
