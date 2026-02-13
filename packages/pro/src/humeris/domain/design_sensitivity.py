# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Composite design sensitivity metrics.

Combines eigenvalue problems from topology, controllability, and positioning
domains. Sweeps altitude for coverage-connectivity trade. Computes coupled
altitude sensitivity Jacobian.

"""
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from humeris.domain.propagation import OrbitalState
from humeris.domain.graph_analysis import compute_topology_resilience
from humeris.domain.control_analysis import compute_cw_controllability
from humeris.domain.dilution_of_precision import compute_dop
from humeris.domain.link_budget import LinkConfig
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state, propagate_ecef_to
from humeris.domain.revisit import compute_single_coverage_fraction
from humeris.domain.orbital_mechanics import OrbitalConstants, j2_raan_rate
from humeris.domain.atmosphere import (
    DragConfig,
    atmospheric_density,
    semi_major_axis_decay_rate,
)
from humeris.domain.radiation import compute_orbit_radiation_summary
from humeris.domain.eclipse import eclipse_fraction

_R_EARTH = OrbitalConstants.R_EARTH
_MU = OrbitalConstants.MU_EARTH


@dataclass(frozen=True)
class SpectralFragility:
    """Composite spectral fragility from three eigenvalue domains."""
    topology_isotropy: float
    controllability_isotropy: float
    positioning_efficiency: float
    composite_fragility: float
    limiting_dimension: str


@dataclass(frozen=True)
class CoverageConnectivityPoint:
    """Coverage and connectivity at a single altitude."""
    altitude_km: float
    coverage_fraction: float
    fiedler_value: float
    coverage_connectivity_product: float


@dataclass(frozen=True)
class CoverageConnectivityCrossover:
    """Coverage-connectivity product sweep over altitude."""
    points: tuple
    peak_product_altitude_km: float
    peak_product: float


@dataclass(frozen=True)
class AltitudeSensitivity:
    """Finite-difference altitude sensitivity Jacobian."""
    altitude_km: float
    delta_altitude_m: float
    raan_drift_sensitivity: float
    drag_decay_sensitivity: float
    dose_rate_sensitivity: float
    thermal_cycle_sensitivity: float
    fiedler_sensitivity: float
    dominant_sensitivity: str


def compute_spectral_fragility(
    states: list,
    time: datetime,
    link_config: LinkConfig,
    n_rad_s: float,
    control_duration_s: float,
    lat_deg: float,
    lon_deg: float,
) -> SpectralFragility:
    """Compute composite spectral fragility from three domains.

    composite = (lambda2/lambda_max) * (1/condition_number) * (1/gdop^2)
    limiting_dimension = which of the three is smallest.
    """
    # 1. Topology: Fiedler value / max eigenvalue = isotropy
    resilience = compute_topology_resilience(states, time, link_config)
    if resilience.fiedler_value > 0 and resilience.spectral_gap > 0:
        # Spectral gap + fiedler gives isotropy measure
        topology_isotropy = resilience.fiedler_value / (resilience.fiedler_value + resilience.spectral_gap)
    else:
        topology_isotropy = 0.0

    # 2. Controllability: 1/condition_number
    ctrl = compute_cw_controllability(n_rad_s, control_duration_s, step_s=60.0)
    if ctrl.condition_number > 0 and ctrl.condition_number < float('inf'):
        controllability_isotropy = 1.0 / ctrl.condition_number
    else:
        controllability_isotropy = 0.0

    # 3. Positioning: 1/GDOP^2
    sat_ecefs = [propagate_ecef_to(s, time) for s in states]
    dop = compute_dop(lat_deg, lon_deg, sat_ecefs)
    if dop.gdop > 0 and dop.gdop < float('inf'):
        positioning_efficiency = 1.0 / (dop.gdop ** 2)
    else:
        positioning_efficiency = 0.0

    # Composite
    composite = topology_isotropy * controllability_isotropy * positioning_efficiency

    # Limiting dimension
    metrics = {
        "topology": topology_isotropy,
        "controllability": controllability_isotropy,
        "positioning": positioning_efficiency,
    }
    limiting = min(metrics, key=metrics.get)

    return SpectralFragility(
        topology_isotropy=topology_isotropy,
        controllability_isotropy=controllability_isotropy,
        positioning_efficiency=positioning_efficiency,
        composite_fragility=composite,
        limiting_dimension=limiting,
    )


def compute_coverage_connectivity_crossover(
    inclination_deg: float,
    num_planes: int,
    sats_per_plane: int,
    link_config: LinkConfig,
    epoch: datetime,
    alt_min_km: float = 300.0,
    alt_max_km: float = 1200.0,
    alt_step_km: float = 50.0,
) -> CoverageConnectivityCrossover:
    """Sweep altitude for coverage * connectivity product.

    At each altitude, generates a Walker shell, computes coverage fraction
    and Fiedler value, and returns the product.
    """
    points: list[CoverageConnectivityPoint] = []

    alt = alt_min_km
    while alt <= alt_max_km + 1e-9:
        shell = ShellConfig(
            altitude_km=alt,
            inclination_deg=inclination_deg,
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            phase_factor=1,
            raan_offset_deg=0.0,
            shell_name="sweep",
        )
        sats = generate_walker_shell(shell)
        states = [derive_orbital_state(s, epoch) for s in sats]

        # Coverage
        cov = compute_single_coverage_fraction(
            states, epoch, lat_step_deg=30.0, lon_step_deg=30.0,
        )

        # Connectivity (Fiedler value)
        resilience = compute_topology_resilience(states, epoch, link_config)
        fiedler = resilience.fiedler_value

        product = cov * fiedler

        points.append(CoverageConnectivityPoint(
            altitude_km=alt,
            coverage_fraction=cov,
            fiedler_value=fiedler,
            coverage_connectivity_product=product,
        ))

        alt += alt_step_km

    if points:
        best = max(points, key=lambda p: p.coverage_connectivity_product)
        peak_alt = best.altitude_km
        peak_product = best.coverage_connectivity_product
    else:
        peak_alt = alt_min_km
        peak_product = 0.0

    return CoverageConnectivityCrossover(
        points=tuple(points),
        peak_product_altitude_km=peak_alt,
        peak_product=peak_product,
    )


def _compute_metrics_at_altitude(
    alt_km: float,
    inclination_rad: float,
    eccentricity: float,
    drag_config: DragConfig,
    epoch: datetime,
    link_config: LinkConfig,
    states: list,
) -> dict:
    """Compute a set of metrics at a given altitude for sensitivity analysis."""
    a = _R_EARTH + alt_km * 1000.0
    n = float(np.sqrt(_MU / a ** 3))

    # RAAN drift rate
    raan_rate = abs(j2_raan_rate(n, a, eccentricity, inclination_rad))

    # Drag decay rate
    da_dt = abs(semi_major_axis_decay_rate(a, eccentricity, drag_config))

    # Radiation dose rate
    temp_state = OrbitalState(
        semi_major_axis_m=a, eccentricity=eccentricity,
        inclination_rad=inclination_rad, raan_rad=0.0,
        arg_perigee_rad=0.0, true_anomaly_rad=0.0,
        mean_motion_rad_s=n, reference_epoch=epoch,
    )
    rad = compute_orbit_radiation_summary(temp_state, epoch)
    dose_rate = rad.mean_dose_rate_rad_s

    # Thermal cycling (eclipse-driven)
    ecl_frac = eclipse_fraction(temp_state, epoch, num_points=36)
    t_orbit = 2.0 * np.pi / n
    cycles_per_day = 86400.0 / t_orbit * ecl_frac

    # Fiedler value
    if states:
        resilience = compute_topology_resilience(states, epoch, link_config)
        fiedler = resilience.fiedler_value
    else:
        fiedler = 0.0

    return {
        "raan_drift": raan_rate,
        "drag_decay": da_dt,
        "dose_rate": dose_rate,
        "thermal_cycle": cycles_per_day,
        "fiedler": fiedler,
    }


def compute_altitude_sensitivity(
    state: OrbitalState,
    drag_config: DragConfig,
    epoch: datetime,
    link_config: LinkConfig,
    states: list,
    delta_altitude_m: float = 100.0,
) -> AltitudeSensitivity:
    """Compute finite-difference altitude sensitivity Jacobian.

    For each metric, computes (metric(alt+delta) - metric(alt-delta)) / (2*delta).
    """
    alt_km = (state.semi_major_axis_m - _R_EARTH) / 1000.0
    delta_km = delta_altitude_m / 1000.0

    alt_plus = alt_km + delta_km
    alt_minus = alt_km - delta_km

    # Clamp to valid atmosphere range
    alt_minus = max(alt_minus, 100.1)
    alt_plus = min(alt_plus, 1999.0)

    metrics_plus = _compute_metrics_at_altitude(
        alt_plus, state.inclination_rad, state.eccentricity,
        drag_config, epoch, link_config, states,
    )
    metrics_minus = _compute_metrics_at_altitude(
        alt_minus, state.inclination_rad, state.eccentricity,
        drag_config, epoch, link_config, states,
    )

    actual_delta_km = alt_plus - alt_minus
    if actual_delta_km <= 0:
        actual_delta_km = 2 * delta_km

    sensitivities = {}
    for key in metrics_plus:
        sensitivities[key] = (metrics_plus[key] - metrics_minus[key]) / actual_delta_km

    # Find dominant (largest absolute sensitivity, normalized)
    # Normalize each sensitivity by the base value to get relative sensitivity
    base_metrics = _compute_metrics_at_altitude(
        alt_km, state.inclination_rad, state.eccentricity,
        drag_config, epoch, link_config, states,
    )

    relative_sens = {}
    for key in sensitivities:
        base_val = base_metrics[key]
        if abs(base_val) > 1e-20:
            relative_sens[key] = abs(sensitivities[key] / base_val)
        else:
            relative_sens[key] = abs(sensitivities[key])

    dominant = max(relative_sens, key=relative_sens.get)

    return AltitudeSensitivity(
        altitude_km=alt_km,
        delta_altitude_m=delta_altitude_m,
        raan_drift_sensitivity=sensitivities["raan_drift"],
        drag_decay_sensitivity=sensitivities["drag_decay"],
        dose_rate_sensitivity=sensitivities["dose_rate"],
        thermal_cycle_sensitivity=sensitivities["thermal_cycle"],
        fiedler_sensitivity=sensitivities["fiedler"],
        dominant_sensitivity=dominant,
    )
