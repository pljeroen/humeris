# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Communication analysis compositions.

Composes ISL topology, link budget, eclipse, access windows, and Doppler
modules to produce eclipse-degraded topologies, pass throughput, ISL
distance predictions, and network capacity timelines.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.inter_satellite_links import (
    ISLLink,
    ISLTopology,
    compute_isl_topology,
)
from humeris.domain.link_budget import (
    LinkConfig,
    LinkBudgetResult,
    compute_link_budget,
    free_space_path_loss_db,
)
from humeris.domain.eclipse import is_eclipsed, EclipseType
from humeris.domain.solar import sun_position_eci
from humeris.domain.observation import GroundStation, compute_observation
from humeris.domain.access_windows import AccessWindow, compute_access_windows
from humeris.domain.pass_analysis import compute_doppler_shift
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
)
from humeris.domain.orbital_mechanics import OrbitalConstants

_R_EARTH = OrbitalConstants.R_EARTH


@dataclass(frozen=True)
class DegradedLink:
    """An ISL link with eclipse degradation info."""
    link: ISLLink
    budget: LinkBudgetResult
    is_eclipsed_a: bool
    is_eclipsed_b: bool
    has_positive_margin: bool


@dataclass(frozen=True)
class EclipseDegradedTopology:
    """ISL topology with eclipse-induced degradation."""
    links: tuple[DegradedLink, ...]
    active_link_count: int
    degraded_link_count: int
    total_link_count: int
    capacity_fraction: float


@dataclass(frozen=True)
class PassDataPoint:
    """Data rate at a single point during a pass."""
    time: datetime
    snr_db: float
    doppler_hz: float
    data_rate_bps: float
    slant_range_km: float


@dataclass(frozen=True)
class PassThroughput:
    """Total data throughput for a pass."""
    window: AccessWindow
    data_points: tuple[PassDataPoint, ...]
    total_bytes: float
    effective_rate_fraction: float
    peak_data_rate_bps: float


@dataclass(frozen=True)
class ISLDistancePrediction:
    """Predicted ISL distances from ascending node spacing."""
    plane_pairs: tuple[tuple[int, int], ...]
    predicted_distances_m: tuple[float, ...]
    node_spacing_deg: float


@dataclass(frozen=True)
class NetworkCapacitySnapshot:
    """Network capacity at a single instant."""
    time: datetime
    active_isl_count: int
    degraded_isl_count: int
    ground_contact_count: int
    eclipsed_sat_count: int


@dataclass(frozen=True)
class NetworkCapacityTimeline:
    """Time series of network capacity."""
    snapshots: tuple[NetworkCapacitySnapshot, ...]
    min_active_isl_count: int
    min_capacity_time: datetime
    mean_active_isl_count: float


def compute_eclipse_degraded_topology(
    states: list[OrbitalState],
    time: datetime,
    link_config: LinkConfig,
    eclipse_power_fraction: float = 0.5,
    max_range_km: float = 5000.0,
) -> EclipseDegradedTopology:
    """ISL topology with eclipse-induced power degradation.

    For each ISL, checks if either endpoint is eclipsed. If so,
    computes link budget with reduced transmit power.

    Args:
        states: List of satellite orbital states.
        time: Evaluation time.
        link_config: RF link configuration.
        eclipse_power_fraction: Power fraction during eclipse.
        max_range_km: Maximum ISL range (km).

    Returns:
        EclipseDegradedTopology with degraded link assessment.
    """
    topology = compute_isl_topology(states, time, max_range_km=max_range_km)
    sun = sun_position_eci(time)

    # Determine eclipse state for each satellite
    positions = []
    for state in states:
        pos, _ = propagate_to(state, time)
        positions.append((pos[0], pos[1], pos[2]))

    eclipsed = []
    for pos in positions:
        ecl = is_eclipsed(pos, sun.position_eci_m)
        eclipsed.append(ecl != EclipseType.NONE)

    degraded_links: list[DegradedLink] = []
    active_count = 0
    degraded_count = 0

    for link in topology.links:
        if link.is_blocked or link.distance_m > max_range_km * 1000.0:
            continue

        ecl_a = eclipsed[link.sat_idx_a] if link.sat_idx_a < len(eclipsed) else False
        ecl_b = eclipsed[link.sat_idx_b] if link.sat_idx_b < len(eclipsed) else False

        # Compute link budget (with possible power reduction)
        if ecl_a or ecl_b:
            # Reduced power config
            reduced_config = LinkConfig(
                frequency_hz=link_config.frequency_hz,
                transmit_power_w=link_config.transmit_power_w * eclipse_power_fraction,
                tx_antenna_gain_dbi=link_config.tx_antenna_gain_dbi,
                rx_antenna_gain_dbi=link_config.rx_antenna_gain_dbi,
                system_noise_temp_k=link_config.system_noise_temp_k,
                bandwidth_hz=link_config.bandwidth_hz,
                additional_losses_db=link_config.additional_losses_db,
                required_snr_db=link_config.required_snr_db,
            )
            budget = compute_link_budget(reduced_config, link.distance_m)
            degraded_count += 1
        else:
            budget = compute_link_budget(link_config, link.distance_m)

        has_margin = budget.link_margin_db > 0
        if has_margin:
            active_count += 1

        degraded_links.append(DegradedLink(
            link=link,
            budget=budget,
            is_eclipsed_a=ecl_a,
            is_eclipsed_b=ecl_b,
            has_positive_margin=has_margin,
        ))

    total_count = len(degraded_links)
    capacity_frac = active_count / total_count if total_count > 0 else 0.0

    return EclipseDegradedTopology(
        links=tuple(degraded_links),
        active_link_count=active_count,
        degraded_link_count=degraded_count,
        total_link_count=total_count,
        capacity_fraction=capacity_frac,
    )


def compute_pass_data_throughput(
    station: GroundStation,
    state: OrbitalState,
    window: AccessWindow,
    link_config: LinkConfig,
    freq_hz: float,
    step_s: float = 10.0,
) -> PassThroughput:
    """Compute data throughput for a single pass.

    Steps through the access window, computes link budget and Doppler
    at each point, integrates total bytes.

    Args:
        station: Ground station.
        state: Satellite orbital state.
        window: Access window.
        link_config: RF link configuration.
        freq_hz: Communication frequency (Hz).
        step_s: Time step (seconds).

    Returns:
        PassThroughput with data points and total bytes.
    """
    data_points: list[PassDataPoint] = []
    total_bytes = 0.0
    peak_rate = 0.0

    duration = window.duration_seconds
    t = 0.0
    while t <= duration + 1e-9:
        current_time = window.rise_time + timedelta(seconds=t)
        pos_eci, vel_eci = propagate_to(state, current_time)

        # Compute slant range
        from humeris.domain.propagation import propagate_ecef_to
        sat_ecef = propagate_ecef_to(state, current_time)
        obs = compute_observation(station, sat_ecef)
        slant_range_km = obs.slant_range_m / 1000.0

        # Link budget at this range
        budget = compute_link_budget(link_config, obs.slant_range_m)

        # Doppler
        doppler = compute_doppler_shift(
            station, pos_eci, vel_eci, current_time, freq_hz,
        )

        rate = max(0.0, budget.max_data_rate_bps)
        if rate > peak_rate:
            peak_rate = rate

        data_points.append(PassDataPoint(
            time=current_time,
            snr_db=budget.snr_db,
            doppler_hz=doppler.shift_hz,
            data_rate_bps=rate,
            slant_range_km=slant_range_km,
        ))

        # Integrate bytes
        dt = min(step_s, duration - t + 1e-9)
        total_bytes += rate * dt / 8.0

        t += step_s

    effective_frac = 0.0
    if peak_rate > 0 and duration > 0:
        avg_rate = total_bytes * 8.0 / duration
        effective_frac = avg_rate / peak_rate

    return PassThroughput(
        window=window,
        data_points=tuple(data_points),
        total_bytes=total_bytes,
        effective_rate_fraction=min(effective_frac, 1.0),
        peak_data_rate_bps=peak_rate,
    )


def predict_isl_distances(
    node_longitudes_deg: list[float],
    altitude_km: float,
) -> ISLDistancePrediction:
    """Predict ISL distances from ascending node longitudes.

    For adjacent pairs of ascending nodes, computes the chord distance
    at orbital altitude.

    Args:
        node_longitudes_deg: List of ascending node longitudes (degrees).
        altitude_km: Orbital altitude (km).

    Returns:
        ISLDistancePrediction with plane pairs and distances.
    """
    r = _R_EARTH + altitude_km * 1000.0
    sorted_lons = sorted(node_longitudes_deg)

    pairs: list[tuple[int, int]] = []
    distances: list[float] = []

    for i in range(len(sorted_lons) - 1):
        dlon_deg = sorted_lons[i + 1] - sorted_lons[i]
        dlon_rad = math.radians(dlon_deg)
        # Chord distance at orbital altitude
        dist = 2.0 * r * math.sin(dlon_rad / 2.0)
        pairs.append((i, i + 1))
        distances.append(dist)

    # Compute average spacing
    if len(sorted_lons) >= 2:
        total_span = sorted_lons[-1] - sorted_lons[0]
        spacing = total_span / (len(sorted_lons) - 1)
    else:
        spacing = 0.0

    return ISLDistancePrediction(
        plane_pairs=tuple(pairs),
        predicted_distances_m=tuple(distances),
        node_spacing_deg=spacing,
    )


def compute_network_capacity_timeline(
    states: list[OrbitalState],
    stations: list[GroundStation],
    link_config: LinkConfig,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    max_range_km: float = 5000.0,
) -> NetworkCapacityTimeline:
    """Time series of network capacity (ISL + ground contacts).

    At each time step, evaluates ISL topology and ground contacts.

    Args:
        states: List of satellite orbital states.
        stations: List of ground stations.
        link_config: RF link configuration.
        epoch: Start time.
        duration_s: Duration (seconds).
        step_s: Time step (seconds).
        max_range_km: Maximum ISL range (km).

    Returns:
        NetworkCapacityTimeline with snapshots.
    """
    snapshots: list[NetworkCapacitySnapshot] = []
    t = 0.0

    while t <= duration_s + 1e-9:
        current_time = epoch + timedelta(seconds=t)

        # ISL topology
        topology = compute_isl_topology(states, current_time, max_range_km=max_range_km)

        # Eclipse state
        sun = sun_position_eci(current_time)
        eclipsed_count = 0
        positions = []
        for state in states:
            pos, _ = propagate_to(state, current_time)
            pos_tuple = (pos[0], pos[1], pos[2])
            positions.append(pos_tuple)
            if is_eclipsed(pos_tuple, sun.position_eci_m) != EclipseType.NONE:
                eclipsed_count += 1

        # Count degraded ISLs (at least one endpoint eclipsed)
        eclipsed_set = set()
        for i, pos in enumerate(positions):
            if is_eclipsed(pos, sun.position_eci_m) != EclipseType.NONE:
                eclipsed_set.add(i)

        degraded = 0
        for link in topology.links:
            if not link.is_blocked and link.distance_m <= max_range_km * 1000.0:
                if link.sat_idx_a in eclipsed_set or link.sat_idx_b in eclipsed_set:
                    degraded += 1

        # Ground contacts: check which stations can see at least one satellite
        from humeris.domain.propagation import propagate_ecef_to
        ground_contacts = 0
        for station in stations:
            for state in states:
                sat_ecef = propagate_ecef_to(state, current_time)
                obs = compute_observation(station, sat_ecef)
                if obs.elevation_deg >= 10.0:
                    ground_contacts += 1
                    break

        snapshots.append(NetworkCapacitySnapshot(
            time=current_time,
            active_isl_count=topology.num_active_links,
            degraded_isl_count=degraded,
            ground_contact_count=ground_contacts,
            eclipsed_sat_count=eclipsed_count,
        ))

        t += step_s

    # Summary
    if snapshots:
        min_active = min(s.active_isl_count for s in snapshots)
        min_time = next(s.time for s in snapshots if s.active_isl_count == min_active)
        mean_active = sum(s.active_isl_count for s in snapshots) / len(snapshots)
    else:
        min_active = 0
        min_time = epoch
        mean_active = 0.0

    return NetworkCapacityTimeline(
        snapshots=tuple(snapshots),
        min_active_isl_count=min_active,
        min_capacity_time=min_time,
        mean_active_isl_count=mean_active,
    )
