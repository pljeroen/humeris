# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Multi-objective constellation design.

Pareto-optimal navigation-control-mass surface and entropy-collision
efficiency for information-theoretically optimal constellation sizing.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime

from constellation_generator.domain.propagation import OrbitalState, propagate_ecef_to
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.design_optimization import (
    compute_mass_efficiency_frontier,
    compute_positioning_information,
)
from constellation_generator.domain.control_analysis import compute_cw_controllability
from constellation_generator.domain.information_theory import compute_marginal_satellite_value
from constellation_generator.domain.statistical_analysis import compute_analytical_collision_probability


@dataclass(frozen=True)
class ParetoPoint:
    """Single point in the 3D objective space."""
    altitude_km: float
    information_efficiency: float
    controllability: float
    mass_efficiency: float
    is_pareto_optimal: bool


@dataclass(frozen=True)
class ParetoSurface:
    """Non-dominated configurations in navigation-control-mass space."""
    points: tuple
    pareto_front: tuple
    num_pareto_optimal: int


@dataclass(frozen=True)
class EntropyCollisionEfficiency:
    """Information gain per unit collision risk."""
    entropy_gain: float
    aggregate_collision_prob: float
    information_risk_ratio: float
    marginal_value_index: float


def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """True if a dominates b (a >= b in all objectives, a > b in at least one)."""
    geq = (
        a.information_efficiency >= b.information_efficiency
        and a.controllability >= b.controllability
        and a.mass_efficiency >= b.mass_efficiency
    )
    gt = (
        a.information_efficiency > b.information_efficiency
        or a.controllability > b.controllability
        or a.mass_efficiency > b.mass_efficiency
    )
    return geq and gt


def compute_pareto_surface(
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    injection_altitude_km: float,
    mission_years: float,
    num_sats: int,
    epoch: datetime,
    inclination_rad: float,
    alt_min_km: float = 400.0,
    alt_max_km: float = 800.0,
    alt_step_km: float = 50.0,
    control_duration_s: float = 5400.0,
    observer_lat_deg: float = 45.0,
    observer_lon_deg: float = 0.0,
) -> ParetoSurface:
    """Sweep altitudes for navigation-control-mass Pareto frontier.

    P(h) = (information_efficiency(h), 1/kappa(W_c(h)), mass_efficiency(h))
    """
    frontier = compute_mass_efficiency_frontier(
        drag_config, isp_s, dry_mass_kg, injection_altitude_km,
        mission_years, num_sats,
        alt_min_km=alt_min_km, alt_max_km=alt_max_km, alt_step_km=alt_step_km,
    )

    raw_points = []

    for fp in frontier.points:
        alt_km = fp.altitude_km
        a_m = OrbitalConstants.R_EARTH + alt_km * 1000.0
        n_rad_s = math.sqrt(OrbitalConstants.MU_EARTH / a_m ** 3)

        # Controllability
        ctrl = compute_cw_controllability(n_rad_s, control_duration_s)
        if ctrl.condition_number > 0:
            controllability = 1.0 / ctrl.condition_number
        else:
            controllability = 0.0

        # Information efficiency: build constellation at this altitude
        # and compute positioning information
        states_at_alt = []
        raan_step = 360.0 / max(1, int(math.sqrt(num_sats)))
        ta_step = 360.0 / max(1, num_sats // max(1, int(math.sqrt(num_sats))))
        n_planes = max(1, int(math.sqrt(num_sats)))
        n_per_plane = max(1, num_sats // n_planes)

        for p in range(n_planes):
            for s in range(n_per_plane):
                states_at_alt.append(OrbitalState(
                    semi_major_axis_m=a_m, eccentricity=0.0,
                    inclination_rad=inclination_rad,
                    raan_rad=math.radians(p * 360.0 / n_planes),
                    arg_perigee_rad=0.0,
                    true_anomaly_rad=math.radians(s * 360.0 / n_per_plane),
                    mean_motion_rad_s=n_rad_s,
                    reference_epoch=epoch,
                ))

        sat_ecef = [propagate_ecef_to(s, epoch) for s in states_at_alt]
        pos_info = compute_positioning_information(
            observer_lat_deg, observer_lon_deg, sat_ecef,
        )
        info_eff = pos_info.information_efficiency

        raw_points.append(ParetoPoint(
            altitude_km=alt_km,
            information_efficiency=info_eff,
            controllability=controllability,
            mass_efficiency=fp.mass_efficiency,
            is_pareto_optimal=False,
        ))

    # Identify Pareto front
    pareto_indices = []
    for i, p in enumerate(raw_points):
        dominated = False
        for j, q in enumerate(raw_points):
            if i != j and _dominates(q, p):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    points = []
    for i, p in enumerate(raw_points):
        points.append(ParetoPoint(
            altitude_km=p.altitude_km,
            information_efficiency=p.information_efficiency,
            controllability=p.controllability,
            mass_efficiency=p.mass_efficiency,
            is_pareto_optimal=(i in pareto_indices),
        ))

    return ParetoSurface(
        points=tuple(points),
        pareto_front=tuple(pareto_indices),
        num_pareto_optimal=len(pareto_indices),
    )


def compute_entropy_collision_efficiency(
    states: list,
    candidate: OrbitalState,
    epoch: datetime,
    miss_distance_m: float = 500.0,
    sigma_radial_m: float = 100.0,
    sigma_cross_m: float = 200.0,
    combined_radius_m: float = 5.0,
    duration_s: float = 5400.0,
    step_s: float = 60.0,
    lat_step_deg: float = 30.0,
    lon_step_deg: float = 30.0,
) -> EntropyCollisionEfficiency:
    """Compute information gain per unit collision risk.

    eta = Delta_H / P_aggregate
    """
    # Information gain
    marginal = compute_marginal_satellite_value(
        states, candidate, epoch,
        duration_s=duration_s, step_s=step_s,
        lat_step_deg=lat_step_deg, lon_step_deg=lon_step_deg,
    )
    entropy_gain = marginal.coverage_information_gain

    # Aggregate collision probability
    n = len(states)
    if n == 0:
        return EntropyCollisionEfficiency(
            entropy_gain=entropy_gain,
            aggregate_collision_prob=0.0,
            information_risk_ratio=float('inf') if entropy_gain > 0 else 0.0,
            marginal_value_index=marginal.total_information_value,
        )

    # P_aggregate = 1 - product(1 - P_c(i, candidate))
    survival_product = 1.0
    for i in range(n):
        pc = compute_analytical_collision_probability(
            miss_distance_m=miss_distance_m,
            b_radial_m=miss_distance_m * 0.7,
            b_cross_m=miss_distance_m * 0.7,
            sigma_radial_m=sigma_radial_m,
            sigma_cross_m=sigma_cross_m,
            combined_radius_m=combined_radius_m,
        )
        survival_product *= (1.0 - pc.analytical_pc)

    p_aggregate = 1.0 - survival_product

    # Ratio
    if p_aggregate > 1e-20:
        ratio = entropy_gain / p_aggregate
    else:
        ratio = float('inf') if entropy_gain > 0 else 0.0

    return EntropyCollisionEfficiency(
        entropy_gain=entropy_gain,
        aggregate_collision_prob=p_aggregate,
        information_risk_ratio=ratio,
        marginal_value_index=marginal.total_information_value,
    )
