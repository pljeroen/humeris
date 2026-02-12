# Copyright (c) 2026 Jeroen. All rights reserved.
"""Constellation operability assessment.

Constellation Operability Index (COI) combining graph connectivity,
communication capacity, and controllability. Common-cause failure
detection via triple correlation.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.link_budget import LinkConfig
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.graph_analysis import (
    compute_topology_resilience,
    compute_fragmentation_timeline,
)
from constellation_generator.domain.information_theory import compute_eclipse_channel_capacity
from constellation_generator.domain.control_analysis import compute_cw_controllability
from constellation_generator.domain.statistical_analysis import (
    compute_mission_availability,
    _pearson_correlation,
)
from constellation_generator.domain.orbital_mechanics import OrbitalConstants


@dataclass(frozen=True)
class ConstellationOperabilityIndex:
    """Single scalar measuring simultaneous operability."""
    connectivity_factor: float
    communication_factor: float
    controllability_factor: float
    coi: float
    is_operable: bool


@dataclass(frozen=True)
class CommonCauseFailureResult:
    """Triple correlation detecting shared degradation root cause."""
    fiedler_bec_correlation: float
    fiedler_availability_correlation: float
    degradation_correlation: float
    is_common_cause: bool
    dominant_cause: str


def compute_operability_index(
    states: list,
    epoch: datetime,
    link_config: LinkConfig,
    max_range_km: float = 5000.0,
    eclipse_power_fraction: float = 0.5,
    control_duration_s: float = 5400.0,
    isl_distance_m: float = 2000e3,
) -> ConstellationOperabilityIndex:
    """Compute Constellation Operability Index.

    COI = (lambda_2 / lambda_2_max) * C_ratio * (1 / log(kappa(W_c)))
    """
    n = len(states)
    if n <= 1:
        return ConstellationOperabilityIndex(
            connectivity_factor=0.0, communication_factor=0.0,
            controllability_factor=0.0, coi=0.0, is_operable=False,
        )

    # Graph connectivity factor
    resilience = compute_topology_resilience(
        states, epoch, link_config,
        max_range_km=max_range_km,
        eclipse_power_fraction=eclipse_power_fraction,
    )
    # Normalize: lambda_2_max for complete graph with uniform weight
    # Use a reference SNR for normalization
    budget = compute_eclipse_channel_capacity(
        states[0], epoch, link_config, isl_distance_m,
    )
    reference_snr = 10.0 ** (20.0 / 10.0)  # Reference: 20 dB SNR
    lambda_2_max = n * reference_snr * eclipse_power_fraction
    if lambda_2_max > 0:
        connectivity = min(1.0, resilience.fiedler_value / lambda_2_max)
    else:
        connectivity = 0.0

    # Communication factor (BEC capacity ratio)
    if budget.awgn_capacity_bps > 0:
        communication = budget.bec_capacity_bps / budget.awgn_capacity_bps
    else:
        communication = 0.0

    # Controllability factor
    if states:
        alt_m = states[0].semi_major_axis_m
        n_rad_s = math.sqrt(OrbitalConstants.MU_EARTH / alt_m ** 3)
    else:
        n_rad_s = 0.001
    ctrl = compute_cw_controllability(n_rad_s, control_duration_s)
    kappa = ctrl.condition_number
    if kappa > math.e:
        controllability = 1.0 / math.log(kappa)
    else:
        controllability = 1.0

    coi = connectivity * communication * min(1.0, controllability)

    return ConstellationOperabilityIndex(
        connectivity_factor=connectivity,
        communication_factor=communication,
        controllability_factor=min(1.0, controllability),
        coi=min(1.0, max(0.0, coi)),
        is_operable=coi > 0.01,
    )


def compute_common_cause_failure(
    states: list,
    link_config: LinkConfig,
    epoch: datetime,
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    propellant_budget_kg: float,
    duration_s: float = 5400.0,
    step_s: float = 300.0,
    max_range_km: float = 5000.0,
    eclipse_power_fraction: float = 0.5,
    isl_distance_m: float = 2000e3,
    mission_years: float = 5.0,
    conjunction_rate_per_year: float = 0.1,
) -> CommonCauseFailureResult:
    """Detect common-cause failure via triple correlation.

    D = pearson(fiedler(t), bec_capacity(t)) * pearson(fiedler(t), availability(t))

    High |D| → eclipse causes simultaneous graph, communication, and mission degradation.
    """
    # Compute fragmentation timeline
    timeline = compute_fragmentation_timeline(
        states, link_config, epoch, duration_s, step_s,
        max_range_km=max_range_km,
        eclipse_power_fraction=eclipse_power_fraction,
    )

    fiedler_series = [e.fiedler_value for e in timeline.events]

    # Compute BEC capacity at each timestep
    bec_series = []
    for event in timeline.events:
        if states:
            cap = compute_eclipse_channel_capacity(
                states[0], event.time, link_config, isl_distance_m,
            )
            bec_series.append(cap.bec_capacity_bps)
        else:
            bec_series.append(0.0)

    # Compute availability at each timestep
    avail_series = []
    if states:
        avail = compute_mission_availability(
            states[0], drag_config, epoch,
            isp_s=isp_s, dry_mass_kg=dry_mass_kg,
            propellant_budget_kg=propellant_budget_kg,
            mission_years=mission_years,
            conjunction_rate_per_year=conjunction_rate_per_year,
        )
        # Resample availability to match fragmentation timesteps
        # Availability has (num_steps+1) monthly points, fragmentation has more
        num_avail = len(avail.total_availability)
        for i in range(len(timeline.events)):
            frac = i / max(1, len(timeline.events) - 1)
            idx = min(int(frac * (num_avail - 1)), num_avail - 1)
            avail_series.append(avail.total_availability[idx])
    else:
        avail_series = [0.0] * len(fiedler_series)

    # Triple correlation
    r_fiedler_bec = _pearson_correlation(fiedler_series, bec_series)
    r_fiedler_avail = _pearson_correlation(fiedler_series, avail_series)
    degradation = r_fiedler_bec * r_fiedler_avail

    is_common = abs(degradation) > 0.25

    # Dominant cause
    if abs(r_fiedler_bec) > abs(r_fiedler_avail):
        cause = "eclipse_communication"
    else:
        cause = "eclipse_availability"

    return CommonCauseFailureResult(
        fiedler_bec_correlation=r_fiedler_bec,
        fiedler_availability_correlation=r_fiedler_avail,
        degradation_correlation=degradation,
        is_common_cause=is_common,
        dominant_cause=cause,
    )
