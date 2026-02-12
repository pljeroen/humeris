# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""ISL topology graph analysis — algebraic connectivity and fragmentation timeline.

Computes the graph Laplacian of the ISL network with SNR-weighted edges,
then uses Jacobi eigendecomposition to find the Fiedler value (lambda_2).

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.propagation import OrbitalState, propagate_to
from constellation_generator.domain.link_budget import LinkConfig, compute_link_budget
from constellation_generator.domain.inter_satellite_links import compute_isl_topology
from constellation_generator.domain.eclipse import is_eclipsed, EclipseType
from constellation_generator.domain.solar import sun_position_eci
from constellation_generator.domain.linalg import mat_eigenvalues_symmetric, mat_zeros


@dataclass(frozen=True)
class TopologyResilience:
    """Graph-theoretic resilience metrics for ISL topology."""
    fiedler_value: float
    fiedler_vector: tuple
    num_components: int
    is_connected: bool
    spectral_gap: float
    total_capacity: float


@dataclass(frozen=True)
class FragmentationEvent:
    """Snapshot of topology state at a point in time."""
    time: datetime
    fiedler_value: float
    eclipsed_count: int
    active_links: int


@dataclass(frozen=True)
class FragmentationTimeline:
    """Time series of topology fragmentation."""
    events: tuple
    min_fiedler_value: float
    min_fiedler_time: datetime
    fragmentation_count: int
    mean_fiedler_value: float
    resilience_margin: float


def _compute_laplacian(
    states: list,
    time: datetime,
    link_config: LinkConfig,
    max_range_km: float,
    eclipse_power_fraction: float,
) -> tuple:
    """Build weighted graph Laplacian from ISL topology.

    Returns (laplacian_matrix, total_capacity, active_links, eclipsed_count).
    """
    n = len(states)
    if n <= 1:
        lap = mat_zeros(max(n, 1), max(n, 1))
        return lap, 0.0, 0, 0

    topology = compute_isl_topology(states, time, max_range_km=max_range_km)
    sun_pos = sun_position_eci(time)

    # Check eclipse status for each satellite
    eclipsed = []
    for s in states:
        pos, _ = propagate_to(s, time)
        pos_tuple = (pos[0], pos[1], pos[2])
        sun_tuple = (sun_pos.position_eci_m[0], sun_pos.position_eci_m[1], sun_pos.position_eci_m[2])
        eclipse_type = is_eclipsed(pos_tuple, sun_tuple)
        eclipsed.append(eclipse_type != EclipseType.NONE)

    eclipsed_count = sum(1 for e in eclipsed if e)

    lap = mat_zeros(n, n)
    total_cap = 0.0
    active_links = 0

    for link in topology.links:
        if link.is_blocked:
            continue
        i = link.sat_idx_a
        j = link.sat_idx_b

        budget = compute_link_budget(link_config, link.distance_m)
        snr_linear = 10.0 ** (budget.snr_db / 10.0)
        weight = snr_linear

        # Apply eclipse degradation
        if eclipsed[i] or eclipsed[j]:
            weight *= eclipse_power_fraction

        lap[i][j] = -weight
        lap[j][i] = -weight
        lap[i][i] += weight
        lap[j][j] += weight
        total_cap += budget.max_data_rate_bps
        active_links += 1

    return lap, total_cap, active_links, eclipsed_count


def compute_topology_resilience(
    states: list,
    time: datetime,
    link_config: LinkConfig,
    max_range_km: float = 5000.0,
    eclipse_power_fraction: float = 0.5,
) -> TopologyResilience:
    """Compute algebraic connectivity (Fiedler value) of ISL topology.

    The Fiedler value lambda_2 is the second-smallest eigenvalue of the
    graph Laplacian. It measures the bottleneck of information flow:
    - lambda_2 > 0 means the graph is connected
    - lambda_2 = 0 means the graph is disconnected
    """
    n = len(states)
    if n <= 1:
        return TopologyResilience(
            fiedler_value=0.0,
            fiedler_vector=(0.0,) if n == 1 else (),
            num_components=n,
            is_connected=(n <= 1),
            spectral_gap=0.0,
            total_capacity=0.0,
        )

    lap, total_cap, _, _ = _compute_laplacian(
        states, time, link_config, max_range_km, eclipse_power_fraction,
    )

    eig = mat_eigenvalues_symmetric(lap)
    eigenvalues = list(eig.eigenvalues)

    # lambda_1 should be ~0, lambda_2 is the Fiedler value
    fiedler_val = max(0.0, eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    fiedler_vec = eig.eigenvectors[1] if len(eig.eigenvectors) > 1 else (0.0,) * n

    # Count components: number of eigenvalues near zero
    num_components = sum(1 for v in eigenvalues if abs(v) < 1e-8)
    is_connected = num_components <= 1

    spectral_gap = 0.0
    if len(eigenvalues) > 2:
        spectral_gap = max(0.0, eigenvalues[2] - eigenvalues[1])

    return TopologyResilience(
        fiedler_value=fiedler_val,
        fiedler_vector=fiedler_vec,
        num_components=num_components,
        is_connected=is_connected,
        spectral_gap=spectral_gap,
        total_capacity=total_cap,
    )


def compute_fragmentation_timeline(
    states: list,
    link_config: LinkConfig,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    max_range_km: float = 5000.0,
    eclipse_power_fraction: float = 0.5,
) -> FragmentationTimeline:
    """Compute time series of Fiedler value with eclipse-degraded weights."""
    n = len(states)
    num_steps = int(duration_s / step_s) + 1
    events = []
    min_fiedler = float('inf')
    min_fiedler_time = epoch
    frag_count = 0

    for step in range(num_steps):
        t = epoch + timedelta(seconds=step * step_s)

        if n <= 1:
            events.append(FragmentationEvent(
                time=t, fiedler_value=0.0, eclipsed_count=0, active_links=0,
            ))
            if 0.0 < min_fiedler:
                min_fiedler = 0.0
                min_fiedler_time = t
            frag_count += 1
            continue

        lap, _, active_links, eclipsed_count = _compute_laplacian(
            states, t, link_config, max_range_km, eclipse_power_fraction,
        )
        eig = mat_eigenvalues_symmetric(lap)
        fiedler_val = max(0.0, eig.eigenvalues[1]) if len(eig.eigenvalues) > 1 else 0.0

        events.append(FragmentationEvent(
            time=t,
            fiedler_value=fiedler_val,
            eclipsed_count=eclipsed_count,
            active_links=active_links,
        ))

        if fiedler_val < min_fiedler:
            min_fiedler = fiedler_val
            min_fiedler_time = t

        if fiedler_val <= 1e-10:
            frag_count += 1

    if min_fiedler == float('inf'):
        min_fiedler = 0.0

    fiedler_values = [e.fiedler_value for e in events]
    mean_fiedler = sum(fiedler_values) / len(fiedler_values) if fiedler_values else 0.0
    resilience_margin = min_fiedler / mean_fiedler if mean_fiedler > 1e-15 else 0.0

    return FragmentationTimeline(
        events=tuple(events),
        min_fiedler_value=min_fiedler,
        min_fiedler_time=min_fiedler_time,
        fragmentation_count=frag_count,
        mean_fiedler_value=mean_fiedler,
        resilience_margin=resilience_margin,
    )
