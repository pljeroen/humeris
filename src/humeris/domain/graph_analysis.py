# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""ISL topology graph analysis — algebraic connectivity, fragmentation timeline,
and Hodge Laplacian higher-order topology.

Computes the graph Laplacian of the ISL network with SNR-weighted edges,
then uses Jacobi eigendecomposition to find the Fiedler value (lambda_2).

Also provides Hodge Laplacian analysis via simplicial complexes: boundary
operators B1 (edge-node) and B2 (triangle-edge) yield the edge Laplacian
L1 = B1 B1^T + B2^T B2, whose null space dimension is the first Betti
number beta_1 (independent routing cycles not filled by triangles).

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.link_budget import LinkConfig, compute_link_budget
from humeris.domain.inter_satellite_links import compute_isl_topology
from humeris.domain.eclipse import is_eclipsed, EclipseType
from humeris.domain.solar import sun_position_eci
from humeris.domain.linalg import mat_eigenvalues_symmetric, mat_zeros


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


# ---------------------------------------------------------------------------
# Hodge Laplacian topology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HodgeTopology:
    """Higher-order topological analysis of ISL network."""
    betti_1: int                    # First Betti number (independent routing cycles)
    l1_spectral_gap: float          # Spectral gap of edge Laplacian
    triangle_count: int             # Number of triangles (2-simplices)
    routing_redundancy: float       # beta_1 / edge_count (cycle density)
    l1_smallest_nonzero: float      # Smallest nonzero eigenvalue of L_1


def compute_hodge_topology(
    adjacency: list[list[float]],
    n_nodes: int,
) -> HodgeTopology:
    """Compute Hodge Laplacian topology from adjacency matrix.

    Builds the simplicial complex (edges + triangles), computes the boundary
    operators B1 (edges -> nodes) and B2 (triangles -> edges), then assembles
    the edge Laplacian L1 = B1 B1^T + B2^T B2.

    The null space of L1 corresponds to harmonic 1-forms, whose dimension is
    the first Betti number beta_1 — the number of independent cycles in the
    graph that are NOT boundaries of 2-simplices (triangles).

    Args:
        adjacency: n_nodes x n_nodes adjacency matrix (symmetric, weight > 0
            indicates an edge).
        n_nodes: Number of nodes in the graph.

    Returns:
        HodgeTopology with Betti number, spectral gap, and cycle metrics.
    """
    # Step 1: Extract oriented edges from upper triangle (i < j, weight > 0)
    edges = []
    edge_index = {}  # (i, j) -> edge index
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i][j] > 0:
                idx = len(edges)
                edges.append((i, j))
                edge_index[(i, j)] = idx

    n_edges = len(edges)

    # Handle degenerate case: no edges
    if n_edges == 0:
        return HodgeTopology(
            betti_1=0,
            l1_spectral_gap=0.0,
            triangle_count=0,
            routing_redundancy=0.0,
            l1_smallest_nonzero=0.0,
        )

    # Step 2: Enumerate triangles (i < j < k where all three edges exist)
    triangles = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if (i, j) not in edge_index:
                continue
            for k in range(j + 1, n_nodes):
                if (i, k) in edge_index and (j, k) in edge_index:
                    triangles.append((i, j, k))

    n_triangles = len(triangles)

    # Step 3: Build boundary operator B1 (n_edges x n_nodes)
    # For edge (i -> j): B1[edge_idx, i] = -1, B1[edge_idx, j] = +1
    b1 = np.zeros((n_edges, n_nodes), dtype=np.float64)
    for e_idx, (i, j) in enumerate(edges):
        b1[e_idx, i] = -1.0
        b1[e_idx, j] = 1.0

    # Step 4: Build boundary operator B2 (n_triangles x n_edges)
    # For triangle (i, j, k) with i < j < k:
    #   boundary = edge(i,j) - edge(i,k) + edge(j,k)
    #   B2[tri_idx, edge(i,j)] = +1
    #   B2[tri_idx, edge(j,k)] = +1
    #   B2[tri_idx, edge(i,k)] = -1
    b2 = np.zeros((n_triangles, n_edges), dtype=np.float64)
    for t_idx, (i, j, k) in enumerate(triangles):
        b2[t_idx, edge_index[(i, j)]] = 1.0
        b2[t_idx, edge_index[(j, k)]] = 1.0
        b2[t_idx, edge_index[(i, k)]] = -1.0

    # Step 5: Assemble Hodge Laplacian L_1 (n_edges x n_edges)
    #
    # Convention: standard B_1 is (n_nodes x n_edges), B_2 is (n_edges x n_triangles).
    # Our b1 = B_1^T (n_edges x n_nodes), b2 = B_2^T (n_triangles x n_edges).
    # L_1 = B_1^T B_1 + B_2 B_2^T = b1 b1^T + b2^T b2
    l1 = b1 @ b1.T + b2.T @ b2

    # Step 6: Eigendecompose L1 (symmetric positive semi-definite)
    eigenvalues = np.linalg.eigh(l1)[0]
    eigenvalues = np.sort(eigenvalues)

    # Count zero eigenvalues (tolerance 1e-10) = beta_1
    tol = 1e-10
    betti_1 = int(np.sum(np.abs(eigenvalues) < tol))

    # Smallest nonzero eigenvalue
    nonzero_eigs = eigenvalues[np.abs(eigenvalues) >= tol]
    if len(nonzero_eigs) > 0:
        l1_smallest_nonzero = float(nonzero_eigs[0])
        l1_spectral_gap = l1_smallest_nonzero
    else:
        l1_smallest_nonzero = 0.0
        l1_spectral_gap = 0.0

    # Routing redundancy
    routing_redundancy = betti_1 / n_edges if n_edges > 0 else 0.0

    return HodgeTopology(
        betti_1=betti_1,
        l1_spectral_gap=l1_spectral_gap,
        triangle_count=n_triangles,
        routing_redundancy=routing_redundancy,
        l1_smallest_nonzero=l1_smallest_nonzero,
    )
