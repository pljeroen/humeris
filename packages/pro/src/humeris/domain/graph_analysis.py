# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""ISL topology graph analysis — algebraic connectivity, fragmentation timeline,
Hodge Laplacian higher-order topology, belief-propagation max-flow routing,
and controllability-aware ISL topology.

Computes the graph Laplacian of the ISL network with SNR-weighted edges,
then uses Jacobi eigendecomposition to find the Fiedler value (lambda_2).

Also provides Hodge Laplacian analysis via simplicial complexes: boundary
operators B1 (edge-node) and B2 (triangle-edge) yield the edge Laplacian
L1 = B1 B1^T + B2^T B2, whose null space dimension is the first Betti
number beta_1 (independent routing cycles not filled by triangles).

Belief-propagation max-flow: treats the ISL capacity graph as a factor graph
and runs min-sum message passing to find the max-flow from source to sink.
Falls back to BFS augmenting-path (Edmonds-Karp) if BP does not converge.

Controllability-aware topology: weights ISL adjacency by both link capacity
and CW Gramian controllability, revealing satellites that are connected but
uncontrollable (poor Gramian condition number).

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


# ---------------------------------------------------------------------------
# P1: Belief Propagation ISL Max-Flow Routing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISLRoutingSolution:
    """Result of max-flow computation on the ISL capacity graph."""
    max_flow_bps: float
    bottleneck_link: tuple
    convergence_iterations: int
    capacity_utilization: float
    converged_via_heuristic: bool


def _bfs_augmenting_path(
    capacity: np.ndarray,
    flow: np.ndarray,
    source: int,
    sink: int,
    n: int,
) -> list:
    """BFS to find an augmenting path in the residual graph.

    Returns the path as a list of node indices, or empty list if no path.
    """
    from collections import deque
    visited = np.zeros(n, dtype=bool)
    visited[source] = True
    parent = np.full(n, -1, dtype=int)
    queue = deque([source])

    while queue:
        u = queue.popleft()
        if u == sink:
            # Reconstruct path
            path = []
            v = sink
            while v != source:
                path.append(v)
                v = int(parent[v])
            path.append(source)
            path.reverse()
            return path
        for v in range(n):
            residual = capacity[u, v] - flow[u, v]
            if not visited[v] and residual > 1e-12:
                visited[v] = True
                parent[v] = u
                queue.append(v)

    return []


def _edmonds_karp_max_flow(
    capacity: np.ndarray,
    source: int,
    sink: int,
    n: int,
) -> tuple:
    """Edmonds-Karp (BFS augmenting path) max-flow.

    Returns (max_flow_value, flow_matrix, iterations).
    """
    flow = np.zeros_like(capacity)
    total_flow = 0.0
    iterations = 0

    while True:
        path = _bfs_augmenting_path(capacity, flow, source, sink, n)
        if not path:
            break
        iterations += 1

        # Find bottleneck capacity along path
        bottleneck = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            residual = capacity[u, v] - flow[u, v]
            if residual < bottleneck:
                bottleneck = residual

        # Update flow along path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            flow[u, v] += bottleneck
            flow[v, u] -= bottleneck

        total_flow += bottleneck

    return total_flow, flow, iterations


def _iterative_flow_heuristic(
    capacity: np.ndarray,
    source: int,
    sink: int,
    n: int,
    max_iterations: int,
    convergence_tol: float,
) -> tuple:
    """Iterative flow heuristic for max-flow approximation.

    This is NOT belief propagation despite superficial similarity. It is a
    custom iterative heuristic that adjusts flow beliefs to satisfy
    conservation constraints at each node. It may not converge to the
    true max-flow on general graphs.

    Uses iterative message passing with damping for stability.
    Returns (max_flow_value, converged, iterations).
    Falls back to None if not converged.
    """
    # Build edge list from capacity matrix
    edges = []
    edge_cap = []
    for i in range(n):
        for j in range(n):
            if capacity[i, j] > 1e-12:
                edges.append((i, j))
                edge_cap.append(capacity[i, j])

    n_edges = len(edges)
    if n_edges == 0:
        return 0.0, True, 0

    edge_cap_arr = np.array(edge_cap)

    # For each node, find incident edge indices and directions
    # node_edges[v] = list of (edge_idx, direction) where direction=+1 if v is source of edge, -1 if sink
    node_edges = [[] for _ in range(n)]
    for idx, (i, j) in enumerate(edges):
        node_edges[i].append((idx, 1))   # outgoing
        node_edges[j].append((idx, -1))  # incoming

    # Initialize flow beliefs (midpoint of capacity)
    flow_belief = edge_cap_arr * 0.5

    # Messages from factor (node conservation) to variable (edge flow)
    # msg[node][edge_idx] = suggested flow adjustment
    damping = 0.5
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        old_belief = flow_belief.copy()

        # For each internal node (not source, not sink), enforce conservation
        for v in range(n):
            if v == source or v == sink:
                continue
            incident = node_edges[v]
            if not incident:
                continue

            # Net flow at v: sum of incoming - sum of outgoing should = 0
            net = 0.0
            for (eidx, direction) in incident:
                if direction == -1:  # incoming
                    net += flow_belief[eidx]
                else:  # outgoing
                    net -= flow_belief[eidx]

            # Distribute excess equally among incident edges
            if len(incident) > 0:
                correction = net / len(incident)
                for (eidx, direction) in incident:
                    if direction == -1:
                        flow_belief[eidx] -= damping * correction
                    else:
                        flow_belief[eidx] += damping * correction

        # Enforce capacity constraints
        flow_belief = np.clip(flow_belief, 0.0, edge_cap_arr)

        # Check convergence
        diff = np.max(np.abs(flow_belief - old_belief))
        if diff < convergence_tol:
            converged = True
            break

    if not converged:
        return None, False, iteration

    # Compute max flow: total flow out of source
    total_flow = 0.0
    for (eidx, direction) in node_edges[source]:
        if direction == 1:  # outgoing from source
            total_flow += flow_belief[eidx]
        else:
            total_flow -= flow_belief[eidx]

    return max(0.0, total_flow), True, iteration


def compute_isl_max_flow(
    adjacency: list,
    source: int,
    sink: int,
    n_nodes: int,
    max_iterations: int = 50,
    convergence_tol: float = 1e-6,
) -> ISLRoutingSolution:
    """Compute max-flow on an ISL capacity graph.

    Attempts an iterative flow heuristic first for convergence diagnostics.
    Always validates with Edmonds-Karp (BFS augmenting path) for the
    exact max-flow value. The heuristic on loopy graphs is an approximation;
    the converged_via_heuristic flag indicates whether the heuristic alone
    matched the exact result.

    Args:
        adjacency: n_nodes x n_nodes capacity matrix (bps). Entry [i][j]
            is the capacity of the directed link from i to j.
        source: Source node index.
        sink: Sink node index.
        n_nodes: Number of nodes in the graph.
        max_iterations: Maximum heuristic iterations.
        convergence_tol: Heuristic convergence tolerance.

    Returns:
        ISLRoutingSolution with max-flow value and diagnostics.
    """
    cap = np.array(adjacency, dtype=np.float64)
    if cap.shape != (n_nodes, n_nodes):
        cap = np.zeros((n_nodes, n_nodes))

    if source == sink or n_nodes < 2:
        return ISLRoutingSolution(
            max_flow_bps=0.0,
            bottleneck_link=(source, sink),
            convergence_iterations=0,
            capacity_utilization=0.0,
            converged_via_heuristic=True,
        )

    # Run iterative heuristic for convergence diagnostics
    heuristic_result, heuristic_converged, h_iters = _iterative_flow_heuristic(
        cap, source, sink, n_nodes, max_iterations, convergence_tol,
    )

    # Always run Edmonds-Karp for exact result
    max_flow_val, flow_matrix, ek_iters = _edmonds_karp_max_flow(
        cap, source, sink, n_nodes,
    )

    # Check if heuristic matched the exact result
    heuristic_matched = bool(
        heuristic_converged
        and heuristic_result is not None
        and abs(heuristic_result - max_flow_val) < convergence_tol * max(1.0, max_flow_val)
    )
    iterations = h_iters if heuristic_matched else ek_iters

    # Find bottleneck: edge with smallest residual capacity that carries flow
    bottleneck_link = (source, sink)
    min_residual = float('inf')
    for i in range(n_nodes):
        for j in range(n_nodes):
            if flow_matrix[i, j] > 1e-12 and cap[i, j] > 1e-12:
                residual = cap[i, j] - flow_matrix[i, j]
                if residual < min_residual:
                    min_residual = residual
                    bottleneck_link = (i, j)

    total_cap = float(np.sum(cap))
    cap_util = max_flow_val / total_cap if total_cap > 0 else 0.0

    return ISLRoutingSolution(
        max_flow_bps=max_flow_val,
        bottleneck_link=bottleneck_link,
        convergence_iterations=iterations,
        capacity_utilization=cap_util,
        converged_via_heuristic=heuristic_matched,
    )


# ---------------------------------------------------------------------------
# P17: Controllability-Aware ISL Topology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllabilityAwareTopology:
    """Joint controllability + connectivity metric for ISL topology."""
    modified_fiedler_value: float
    standard_fiedler_value: float
    controllability_degradation: float
    weakest_link_pair: tuple
    weakest_link_controllability: float


def compute_controllability_routing(
    adjacency: list,
    controllability_scores: list,
    n_nodes: int,
) -> ControllabilityAwareTopology:
    """Compute controllability-aware ISL topology metrics.

    Weights the ISL adjacency matrix by the minimum controllability
    score (1/condition_number) of each link's endpoints. A link to
    a poorly controllable satellite gets lower weight.

    The modified Fiedler value measures bottleneck connectivity
    accounting for both communication and orbital dynamics.

    Args:
        adjacency: n_nodes x n_nodes adjacency/capacity matrix.
            Symmetric, positive entries indicate edges.
        controllability_scores: Per-satellite controllability score.
            Typically 1.0 / condition_number from the CW Gramian.
            Higher is better (more controllable).
        n_nodes: Number of nodes in the graph.

    Returns:
        ControllabilityAwareTopology with standard and modified
        Fiedler values and degradation metric.
    """
    adj = np.array(adjacency, dtype=np.float64)
    scores = np.array(controllability_scores, dtype=np.float64)

    # Clamp scores to [0, 1] for numerical sanity
    scores = np.clip(scores, 0.0, 1.0)

    if n_nodes < 2:
        return ControllabilityAwareTopology(
            modified_fiedler_value=0.0,
            standard_fiedler_value=0.0,
            controllability_degradation=0.0,
            weakest_link_pair=(0, 0),
            weakest_link_controllability=0.0,
        )

    # Build standard Laplacian
    std_lap = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and adj[i, j] > 0:
                std_lap[i, j] = -adj[i, j]
                std_lap[i, i] += adj[i, j]

    # Build modified adjacency: weight by min controllability of endpoints
    mod_lap = np.zeros((n_nodes, n_nodes))
    weakest_pair = (0, 0)
    weakest_score = float('inf')

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and adj[i, j] > 0:
                ctrl_weight = min(scores[i], scores[j])
                mod_weight = adj[i, j] * ctrl_weight
                mod_lap[i, j] = -mod_weight
                mod_lap[i, i] += mod_weight

                if ctrl_weight < weakest_score:
                    weakest_score = ctrl_weight
                    weakest_pair = (i, j)

    if weakest_score == float('inf'):
        weakest_score = 0.0

    # Eigendecompose both Laplacians
    std_eigs = np.sort(np.linalg.eigh(std_lap)[0])
    mod_eigs = np.sort(np.linalg.eigh(mod_lap)[0])

    std_fiedler = max(0.0, float(std_eigs[1])) if len(std_eigs) > 1 else 0.0
    mod_fiedler = max(0.0, float(mod_eigs[1])) if len(mod_eigs) > 1 else 0.0

    degradation = 1.0 - (mod_fiedler / std_fiedler) if std_fiedler > 1e-15 else 0.0
    degradation = max(0.0, min(1.0, degradation))

    return ControllabilityAwareTopology(
        modified_fiedler_value=mod_fiedler,
        standard_fiedler_value=std_fiedler,
        controllability_degradation=degradation,
        weakest_link_pair=weakest_pair,
        weakest_link_controllability=weakest_score,
    )


# ---------------------------------------------------------------------------
# P25: Ising Model Phase Transition for ISL Networks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISLPhaseTransition:
    """Ising model phase transition analysis for ISL network.

    Maps the ISL network to an Ising model where spin sigma_i = +1 (sunlit)
    or -1 (eclipsed). Coupling J_ij comes from link SNR. Eclipse fraction
    maps to temperature via T = 1/(1 - eclipse_fraction). The mean-field
    self-consistency equation m = tanh((J_eff * m + h) / T) is solved
    iteratively to find the magnetization (network coherence).

    A phase transition from ordered (coherent network) to disordered
    (fragmented) occurs at the critical eclipse fraction.
    """
    critical_eclipse_fraction: float
    current_magnetization: float
    susceptibility: float
    mean_coupling_strength: float
    is_ordered_phase: bool
    phase_transition_sharpness: float


def compute_isl_phase_transition(
    adjacency: list,
    eclipse_fractions: list,
    n_nodes: int,
    external_field: float = 0.0,
    eclipse_fraction_steps: int = 50,
    max_iterations: int = 200,
    convergence_tol: float = 1e-8,
) -> ISLPhaseTransition:
    """Compute Ising model phase transition for an ISL network.

    Maps the ISL adjacency to an Ising model:
    - Spin sigma_i = +1 (sunlit) or -1 (eclipsed)
    - Coupling J_ij = adjacency weight / 10.0 (SNR-based)
    - Eclipse fraction maps to temperature: T = 1 / (1 - eclipse_fraction)
    - Mean-field self-consistency: m = tanh((J_eff * m + h) / T)
    - Critical temperature T_c = J_eff (mean coupling)
    - Susceptibility chi = (1 - m^2) / (T - J_eff * (1 - m^2))

    Scans over eclipse fractions to find the critical point where
    the network transitions from ordered (coherent) to disordered.

    Args:
        adjacency: n_nodes x n_nodes adjacency/weight matrix (symmetric).
            Positive entries indicate edges; weight typically SNR-based.
        eclipse_fractions: Per-node eclipse fraction in [0, 1].
            Used to compute the current operating temperature.
        n_nodes: Number of nodes in the ISL network.
        external_field: External magnetic field h (bias toward sunlit).
        eclipse_fraction_steps: Number of steps for eclipse fraction sweep.
        max_iterations: Maximum iterations for mean-field self-consistency.
        convergence_tol: Convergence tolerance for magnetization.

    Returns:
        ISLPhaseTransition with critical eclipse fraction and diagnostics.
    """
    adj = np.array(adjacency, dtype=np.float64)
    ecl = np.array(eclipse_fractions, dtype=np.float64)

    if n_nodes < 2:
        return ISLPhaseTransition(
            critical_eclipse_fraction=1.0,
            current_magnetization=1.0,
            susceptibility=0.0,
            mean_coupling_strength=0.0,
            is_ordered_phase=True,
            phase_transition_sharpness=0.0,
        )

    # Compute coupling matrix: J_ij = adjacency weight / 10.0
    j_matrix = adj / 10.0

    # Mean coupling strength: J_eff = mean of all nonzero couplings per node
    # J_eff = (1/N) * sum_j J_ij for an average node
    total_coupling = 0.0
    coupling_count = 0
    for i in range(n_nodes):
        node_coupling = 0.0
        for j in range(n_nodes):
            if i != j and j_matrix[i, j] > 0:
                node_coupling += j_matrix[i, j]
        total_coupling += node_coupling
        if node_coupling > 0:
            coupling_count += 1

    j_eff = total_coupling / n_nodes if n_nodes > 0 else 0.0

    if j_eff < 1e-15:
        return ISLPhaseTransition(
            critical_eclipse_fraction=0.0,
            current_magnetization=0.0,
            susceptibility=0.0,
            mean_coupling_strength=0.0,
            is_ordered_phase=False,
            phase_transition_sharpness=0.0,
        )

    # Critical temperature: T_c = J_eff
    t_c = j_eff

    # Current operating temperature from mean eclipse fraction
    mean_eclipse = float(np.mean(ecl)) if len(ecl) > 0 else 0.0
    mean_eclipse = max(0.0, min(mean_eclipse, 0.999))
    current_temp = 1.0 / (1.0 - mean_eclipse)

    # Solve mean-field self-consistency at current temperature
    current_m = _solve_mean_field(j_eff, external_field, current_temp,
                                  max_iterations, convergence_tol)

    # Susceptibility: chi = (1 - m^2) / (T - J_eff * (1 - m^2))
    m_sq = current_m * current_m
    denom = current_temp - j_eff * (1.0 - m_sq)
    if abs(denom) > 1e-15:
        susceptibility = (1.0 - m_sq) / denom
    else:
        susceptibility = float('inf')

    # Sweep eclipse fractions to find critical point
    # Critical eclipse fraction: where magnetization drops sharply
    sweep_ecl = np.linspace(0.0, 0.99, eclipse_fraction_steps)
    sweep_m = np.zeros(eclipse_fraction_steps)

    for idx in range(eclipse_fraction_steps):
        t_sweep = 1.0 / (1.0 - sweep_ecl[idx])
        sweep_m[idx] = _solve_mean_field(j_eff, external_field, t_sweep,
                                         max_iterations, convergence_tol)

    # Critical eclipse fraction: where T = T_c, i.e. 1/(1-f) = J_eff
    # f_c = 1 - 1/J_eff  (if J_eff > 1, otherwise no transition in [0,1])
    if j_eff > 1.0:
        critical_ecl = 1.0 - 1.0 / j_eff
    else:
        # Find where magnetization drops below 0.5 (approximate)
        critical_ecl = 0.99
        for idx in range(eclipse_fraction_steps):
            if sweep_m[idx] < 0.5:
                critical_ecl = float(sweep_ecl[idx])
                break

    # Phase transition sharpness: max |dm/df| over the sweep
    if eclipse_fraction_steps > 1:
        dm = np.diff(sweep_m)
        df = np.diff(sweep_ecl)
        with np.errstate(divide='ignore', invalid='ignore'):
            dm_df = np.where(np.abs(df) > 1e-15, np.abs(dm / df), 0.0)
        sharpness = float(np.max(dm_df))
    else:
        sharpness = 0.0

    is_ordered = bool(current_temp < t_c)

    return ISLPhaseTransition(
        critical_eclipse_fraction=max(0.0, min(1.0, critical_ecl)),
        current_magnetization=current_m,
        susceptibility=susceptibility,
        mean_coupling_strength=j_eff,
        is_ordered_phase=is_ordered,
        phase_transition_sharpness=sharpness,
    )


def _solve_mean_field(
    j_eff: float,
    h: float,
    temperature: float,
    max_iterations: int,
    tol: float,
) -> float:
    """Solve mean-field self-consistency equation m = tanh((J_eff*m + h)/T).

    Uses iterative fixed-point method with damping for stability.

    Returns the converged magnetization m in [0, 1].
    """
    if temperature < 1e-15:
        return 1.0

    m = 0.5  # Initial guess
    damping = 0.3

    for _ in range(max_iterations):
        arg = (j_eff * m + h) / temperature
        # Clamp argument to avoid overflow in tanh (effectively +-1 beyond +-20)
        arg = max(-20.0, min(20.0, arg))
        m_new = math.tanh(arg)
        m_updated = (1.0 - damping) * m + damping * m_new

        if abs(m_updated - m) < tol:
            return abs(m_updated)

        m = m_updated

    return abs(m)


# ---------------------------------------------------------------------------
# P41: Quantum Walk Coverage Exploration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantumWalkCoverage:
    """Coined discrete-time quantum walk analysis on ISL graph."""
    mixing_time_steps: int
    classical_mixing_time_steps: int
    speedup_ratio: float
    hitting_time_to_antipodal: int
    propagation_velocity: float
    stationary_distribution: tuple


def compute_quantum_walk_coverage(
    adjacency: list,
    n_nodes: int,
    epsilon: float = 0.01,
    max_steps: int = 2000,
    start_node: int = 0,
) -> QuantumWalkCoverage:
    """Coined discrete-time quantum walk on the ISL graph.

    Uses Grover diffusion coin C = 2|s><s| - I on each node's internal
    (coin) space, where |s> is the uniform superposition over the node's
    edges.  Tracks probability distribution p(v,t) and computes the
    mixing time (total-variation distance to stationary < epsilon).

    Args:
        adjacency: n_nodes x n_nodes adjacency matrix (symmetric, positive = edge).
        n_nodes: Number of nodes.
        epsilon: TV distance threshold for mixing.
        max_steps: Maximum walk steps.
        start_node: Starting node for hitting time measurement.

    Returns:
        QuantumWalkCoverage with mixing/hitting diagnostics.
    """
    adj = np.array(adjacency, dtype=np.float64)

    if n_nodes <= 1:
        return QuantumWalkCoverage(
            mixing_time_steps=0,
            classical_mixing_time_steps=0,
            speedup_ratio=1.0,
            hitting_time_to_antipodal=0,
            propagation_velocity=0.0,
            stationary_distribution=(1.0,) if n_nodes == 1 else (),
        )

    # Build degree vector and neighbour lists
    degrees = np.zeros(n_nodes, dtype=int)
    neighbours = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and adj[i, j] > 0:
                neighbours[i].append(j)
        degrees[i] = len(neighbours[i])

    # Total number of directed arcs (coin dimension)
    # State space: |v, c> where v is node, c indexes neighbour of v
    # We map each directed arc (v -> neighbours[v][c]) to a flat index
    arc_to_idx = {}
    idx_to_arc = []
    idx = 0
    for v in range(n_nodes):
        for c, u in enumerate(neighbours[v]):
            arc_to_idx[(v, u)] = idx
            idx_to_arc.append((v, u))
            idx += 1
    dim = idx

    if dim == 0:
        # Isolated nodes
        return QuantumWalkCoverage(
            mixing_time_steps=max_steps,
            classical_mixing_time_steps=max_steps,
            speedup_ratio=1.0,
            hitting_time_to_antipodal=max_steps,
            propagation_velocity=0.0,
            stationary_distribution=tuple(1.0 / n_nodes for _ in range(n_nodes)),
        )

    # Build the coined quantum walk unitary U = S * C
    # C = Grover diffusion on each node's coin space
    # S = flip/shift: |v, u> -> |u, v>

    # Coin operator (block diagonal, one Grover per node)
    coin = np.zeros((dim, dim), dtype=np.complex128)
    for v in range(n_nodes):
        d = degrees[v]
        if d == 0:
            continue
        # Indices of arcs leaving v
        arc_indices = [arc_to_idx[(v, u)] for u in neighbours[v]]
        # Grover coin: 2/d * |1><1| - I  (all-ones projector scaled)
        for a in arc_indices:
            for b in arc_indices:
                coin[a, b] += 2.0 / d
            coin[a, a] -= 1.0

    # Shift operator: |v, u> -> |u, v>
    shift = np.zeros((dim, dim), dtype=np.float64)
    for (v, u), src_idx in arc_to_idx.items():
        if (u, v) in arc_to_idx:
            dst_idx = arc_to_idx[(u, v)]
            shift[dst_idx, src_idx] = 1.0

    # Full step operator
    U = shift @ coin

    # Initial state: uniform superposition over arcs leaving start_node
    psi = np.zeros(dim, dtype=np.complex128)
    d_start = degrees[start_node]
    if d_start > 0:
        for u in neighbours[start_node]:
            psi[arc_to_idx[(start_node, u)]] = 1.0 / math.sqrt(d_start)
    else:
        # Fallback: uniform over all arcs
        psi[:] = 1.0 / math.sqrt(dim)

    # Stationary distribution for a regular/ergodic graph: proportional to degree
    total_degree = float(np.sum(degrees))
    if total_degree > 0:
        stationary = np.array([float(degrees[v]) / total_degree for v in range(n_nodes)])
    else:
        stationary = np.ones(n_nodes) / n_nodes

    # Antipodal node: the node farthest from start_node by BFS
    antipodal = _bfs_farthest(neighbours, start_node, n_nodes)

    # Walk simulation
    mixing_time = max_steps
    hitting_time = max_steps
    hit_antipodal = False

    for step in range(1, max_steps + 1):
        psi = U @ psi

        # Node probability: p(v) = sum over coin states |<v,c|psi>|^2
        prob = np.zeros(n_nodes)
        for arc_idx in range(dim):
            v, _ = idx_to_arc[arc_idx]
            prob[v] += abs(psi[arc_idx]) ** 2

        # Normalise (numerical safety)
        pn = prob.sum()
        if pn > 1e-15:
            prob /= pn

        # TV distance to stationary
        tv = 0.5 * np.sum(np.abs(prob - stationary))
        if tv < epsilon and mixing_time == max_steps:
            mixing_time = step

        # Hitting time to antipodal
        if not hit_antipodal and prob[antipodal] > 1.0 / (2.0 * n_nodes):
            hitting_time = step
            hit_antipodal = True

        if mixing_time < max_steps and hit_antipodal:
            break

    # Classical mixing time: 1 / spectral_gap of random walk transition matrix
    classical_mixing = _classical_mixing_time(adj, n_nodes)

    speedup = classical_mixing / mixing_time if mixing_time > 0 else 0.0

    # Propagation velocity: antipodal distance / hitting time
    antipodal_dist = _bfs_distance(neighbours, start_node, antipodal, n_nodes)
    prop_vel = antipodal_dist / hitting_time if hitting_time > 0 else 0.0

    return QuantumWalkCoverage(
        mixing_time_steps=mixing_time,
        classical_mixing_time_steps=classical_mixing,
        speedup_ratio=speedup,
        hitting_time_to_antipodal=hitting_time,
        propagation_velocity=prop_vel,
        stationary_distribution=tuple(float(s) for s in stationary),
    )


def _bfs_farthest(neighbours: list, start: int, n: int) -> int:
    """Return the node farthest from start by BFS."""
    from collections import deque
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    farthest = start
    max_d = 0
    while q:
        v = q.popleft()
        for u in neighbours[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                if dist[u] > max_d:
                    max_d = dist[u]
                    farthest = u
                q.append(u)
    return farthest


def _bfs_distance(neighbours: list, start: int, target: int, n: int) -> int:
    """BFS shortest path distance from start to target."""
    from collections import deque
    if start == target:
        return 0
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        v = q.popleft()
        for u in neighbours[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                if u == target:
                    return dist[u]
                q.append(u)
    return n  # unreachable fallback


def _classical_mixing_time(adj: np.ndarray, n: int) -> int:
    """Estimate classical random walk mixing time as ceil(1 / spectral_gap)."""
    degrees = np.sum(adj > 0, axis=1).astype(float)
    if np.any(degrees < 1e-15):
        return n * n  # disconnected fallback

    # Transition matrix P = D^{-1} A
    P = np.zeros((n, n))
    for i in range(n):
        if degrees[i] > 0:
            for j in range(n):
                if adj[i, j] > 0:
                    P[i, j] = 1.0 / degrees[i]

    eigs = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    # Second largest eigenvalue magnitude
    if len(eigs) > 1:
        lambda2 = float(eigs[1])
        gap = 1.0 - lambda2
        if gap > 1e-12:
            return max(1, int(math.ceil(1.0 / gap)))
    return n * n


# ---------------------------------------------------------------------------
# P43: Density Matrix Entanglement Entropy for ISL
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkEntanglement:
    """Entanglement entropy of an ISL network partition."""
    von_neumann_entropy: float
    max_entropy: float
    entanglement_ratio: float
    partition_independence: bool
    eigenvalues: tuple


def compute_network_entanglement(
    adjacency: list,
    n_nodes: int,
    partition_a: list,
) -> NetworkEntanglement:
    """Compute density matrix entanglement entropy for an ISL partition.

    Builds a normalised density matrix rho from the adjacency matrix,
    computes the reduced density matrix rho_A by partial trace over
    partition B, then evaluates the von Neumann entropy.

    Args:
        adjacency: n_nodes x n_nodes adjacency/weight matrix (symmetric).
        n_nodes: Number of nodes.
        partition_a: List of node indices in partition A.

    Returns:
        NetworkEntanglement with entropy and eigenvalue diagnostics.
    """
    adj = np.array(adjacency, dtype=np.float64)
    n_a = len(partition_a)
    partition_b = [i for i in range(n_nodes) if i not in partition_a]
    n_b = len(partition_b)

    max_entropy = math.log2(n_a) if n_a > 1 else 0.0

    if n_a == 0 or n_b == 0 or n_nodes < 2:
        return NetworkEntanglement(
            von_neumann_entropy=0.0,
            max_entropy=max_entropy,
            entanglement_ratio=0.0,
            partition_independence=True,
            eigenvalues=(1.0,) if n_a > 0 else (),
        )

    # Build correlation matrix from adjacency: rho = (A + D) / Tr(A + D)
    # Adding diagonal degree makes it positive semi-definite
    rho_raw = adj.copy()
    for i in range(n_nodes):
        rho_raw[i, i] = np.sum(np.abs(adj[i, :]))

    trace = np.trace(rho_raw)
    if trace < 1e-15:
        return NetworkEntanglement(
            von_neumann_entropy=0.0,
            max_entropy=max_entropy,
            entanglement_ratio=0.0,
            partition_independence=True,
            eigenvalues=tuple(0.0 for _ in range(n_a)),
        )

    rho = rho_raw / trace

    # Partial trace over B to get rho_A
    # rho_A[i,j] = sum over k in B: rho[a_i*n_b + k, a_j*n_b + k]
    # But our matrix is not in tensor product form, so we use the
    # submatrix extraction approach: rho_A = rho[A,A] (principal submatrix)
    # This is the standard network-science approach for graph entanglement.
    idx_a = np.array(partition_a)
    rho_a = rho[np.ix_(idx_a, idx_a)]

    # Normalise rho_A to have trace 1
    tr_a = np.trace(rho_a)
    if tr_a > 1e-15:
        rho_a = rho_a / tr_a

    # Eigendecompose
    eigenvalues = np.linalg.eigh(rho_a)[0]
    eigenvalues = np.clip(eigenvalues, 0.0, None)  # numerical safety

    # Von Neumann entropy: S = -sum lambda_k * log2(lambda_k)
    entropy = 0.0
    for lam in eigenvalues:
        if lam > 1e-15:
            entropy -= lam * math.log2(lam)

    entanglement_ratio = entropy / max_entropy if max_entropy > 1e-15 else 0.0
    entanglement_ratio = min(1.0, max(0.0, entanglement_ratio))

    # Partition independence: entropy near zero means nearly independent
    partition_independence = bool(entropy < 1e-6)

    return NetworkEntanglement(
        von_neumann_entropy=max(0.0, entropy),
        max_entropy=max_entropy,
        entanglement_ratio=entanglement_ratio,
        partition_independence=partition_independence,
        eigenvalues=tuple(float(e) for e in eigenvalues),
    )


# ---------------------------------------------------------------------------
# P44: Spin Glass Frustration Index
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulingFrustration:
    """Spin glass frustration analysis of ISL operational couplings."""
    frustration_index: float
    ground_state_energy: float
    greedy_energy: float
    energy_gap: float
    frustrated_loops: int
    total_loops: int


def compute_scheduling_frustration(
    couplings: list,
    n_nodes: int,
    max_local_search_iters: int = 200,
) -> SchedulingFrustration:
    """Spin glass frustration index for ISL operational scheduling.

    Maps ISL couplings to a spin glass with signed J_ij.  Enumerates
    triangular plaquettes and computes the product of coupling signs
    around each.  Negative products indicate frustrated loops.

    The ground state is approximated by greedy local search that
    flips spins to minimise the Ising energy E = -sum J_ij s_i s_j.

    Args:
        couplings: n_nodes x n_nodes signed coupling matrix (symmetric).
            Positive = ferromagnetic, negative = antiferromagnetic.
        n_nodes: Number of nodes.
        max_local_search_iters: Maximum iterations for greedy search.

    Returns:
        SchedulingFrustration with frustration index and energy analysis.
    """
    J = np.array(couplings, dtype=np.float64)

    # Extract edges (upper triangle with nonzero coupling)
    edges = []
    edge_set = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if abs(J[i, j]) > 1e-15:
                edges.append((i, j))
                edge_set.add((i, j))

    if len(edges) == 0:
        return SchedulingFrustration(
            frustration_index=0.0,
            ground_state_energy=0.0,
            greedy_energy=0.0,
            energy_gap=0.0,
            frustrated_loops=0,
            total_loops=0,
        )

    # Enumerate triangular plaquettes
    triangles = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, n_nodes):
                if (i, k) in edge_set and (j, k) in edge_set:
                    triangles.append((i, j, k))

    total_loops = len(triangles)

    # Compute frustration: product of signs around each triangle
    frustrated_count = 0
    for (i, j, k) in triangles:
        sign_ij = 1.0 if J[i, j] > 0 else -1.0
        sign_jk = 1.0 if J[j, k] > 0 else -1.0
        sign_ik = 1.0 if J[i, k] > 0 else -1.0
        product = sign_ij * sign_jk * sign_ik
        if product < 0:
            frustrated_count += 1

    frustration_index = frustrated_count / total_loops if total_loops > 0 else 0.0

    # Greedy local search for approximate ground state
    # E = -sum_{i<j} J_ij s_i s_j
    spins = np.ones(n_nodes)  # start all +1

    def _ising_energy(s):
        e = 0.0
        for (i, j) in edges:
            e -= J[i, j] * s[i] * s[j]
        return e

    greedy_energy = _ising_energy(spins)

    for _ in range(max_local_search_iters):
        improved = False
        for v in range(n_nodes):
            spins[v] *= -1
            new_e = _ising_energy(spins)
            if new_e < greedy_energy - 1e-15:
                greedy_energy = new_e
                improved = True
            else:
                spins[v] *= -1  # flip back
        if not improved:
            break

    # Try a few random restarts for better ground state estimate
    best_energy = greedy_energy
    rng = np.random.RandomState(42)
    for _ in range(10):
        s = rng.choice([-1.0, 1.0], size=n_nodes)
        e = _ising_energy(s)
        for _ in range(max_local_search_iters):
            improved = False
            for v in range(n_nodes):
                s[v] *= -1
                ne = _ising_energy(s)
                if ne < e - 1e-15:
                    e = ne
                    improved = True
                else:
                    s[v] *= -1
            if not improved:
                break
        if e < best_energy:
            best_energy = e

    ground_state_energy = best_energy
    energy_gap = greedy_energy - ground_state_energy

    return SchedulingFrustration(
        frustration_index=frustration_index,
        ground_state_energy=ground_state_energy,
        greedy_energy=greedy_energy,
        energy_gap=max(0.0, energy_gap),
        frustrated_loops=frustrated_count,
        total_loops=total_loops,
    )


# ---------------------------------------------------------------------------
# P53: Hebbian Learning ISL Adaptation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HebbianISLTopology:
    """Result of Hebbian learning adaptation on ISL topology."""
    adapted_weights: tuple
    emergent_link_count: int
    geometric_link_count: int
    adaptation_efficiency: float
    strongest_links: tuple
    weakest_pruned: tuple


def compute_hebbian_isl_topology(
    adjacency: list,
    activity: list,
    n_nodes: int,
    learning_rate: float = 0.1,
    threshold: float = 0.3,
    epochs: int = 10,
) -> HebbianISLTopology:
    """Hebbian/Oja learning adaptation of ISL link weights.

    Applies Oja's rule: dw_ij/dt = eta * (x_i * x_j - w_ij) over
    the activity timeseries, then thresholds the adapted weight matrix
    to obtain the emergent topology.

    Args:
        adjacency: n_nodes x n_nodes adjacency/weight matrix (initial topology).
        activity: n_nodes x n_nodes activity correlation matrix.  Entry [i][j]
            represents the co-activation strength between nodes i and j.
        n_nodes: Number of nodes.
        learning_rate: Oja learning rate eta.
        threshold: Weight threshold for emergent link inclusion.
        epochs: Number of learning iterations.

    Returns:
        HebbianISLTopology with adapted weights and topology metrics.
    """
    adj = np.array(adjacency, dtype=np.float64)
    act = np.array(activity, dtype=np.float64)

    # Count geometric (initial) links (undirected, upper triangle)
    geometric_links = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj[i, j] > 0:
                geometric_links += 1

    if n_nodes < 2 or geometric_links == 0:
        return HebbianISLTopology(
            adapted_weights=tuple(tuple(0.0 for _ in range(n_nodes)) for _ in range(n_nodes)),
            emergent_link_count=0,
            geometric_link_count=geometric_links,
            adaptation_efficiency=0.0,
            strongest_links=(),
            weakest_pruned=(),
        )

    # Initialise weights from adjacency (normalised)
    max_w = np.max(np.abs(adj))
    if max_w > 1e-15:
        W = adj / max_w
    else:
        W = adj.copy()

    # Oja's rule iterations
    for _ in range(epochs):
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj[i, j] > 0 or adj[j, i] > 0:
                    # Hebbian term: correlation of activity
                    hebbian = act[i, j] if i < len(act) and j < len(act[i]) else 0.0
                    # Oja's rule: dw = eta * (x_i * x_j - w_ij)
                    dw = learning_rate * (hebbian - W[i, j])
                    W[i, j] += dw
                    W[j, i] = W[i, j]

        # Normalise weights (Oja normalisation for stability)
        norm = np.max(np.abs(W))
        if norm > 1e-15:
            W /= norm

    # Threshold to get emergent topology
    emergent_links = 0
    strongest = []
    pruned = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            w = W[i, j]
            if w >= threshold:
                emergent_links += 1
                strongest.append((i, j, float(w)))
            elif adj[i, j] > 0:
                # Was a geometric link, now pruned
                pruned.append((i, j, float(w)))

    # Sort strongest by weight descending
    strongest.sort(key=lambda x: x[2], reverse=True)
    pruned.sort(key=lambda x: x[2])

    # Adaptation efficiency: fraction of geometric links that survived
    efficiency = emergent_links / geometric_links if geometric_links > 0 else 0.0

    return HebbianISLTopology(
        adapted_weights=tuple(tuple(float(W[i, j]) for j in range(n_nodes)) for i in range(n_nodes)),
        emergent_link_count=emergent_links,
        geometric_link_count=geometric_links,
        adaptation_efficiency=efficiency,
        strongest_links=tuple((i, j, w) for i, j, w in strongest[:10]),
        weakest_pruned=tuple((i, j, w) for i, j, w in pruned[:10]),
    )


# ---------------------------------------------------------------------------
# P55: Ant Colony Optimization ISL Routing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ACORoutingSolution:
    """Result of ant colony optimisation routing on ISL graph."""
    best_route: tuple
    best_cost: float
    convergence_iteration: int
    pheromone_distribution: tuple
    route_diversity: float


def compute_aco_routing(
    adjacency: list,
    n_nodes: int,
    source: int,
    sink: int,
    n_ants: int = 10,
    max_iterations: int = 50,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.3,
    initial_pheromone: float = 1.0,
    seed: int = 42,
) -> ACORoutingSolution:
    """Ant colony optimisation routing on an ISL graph.

    Probabilistic route construction with pheromone update and evaporation.
    p_ij = (tau_ij^alpha * eta_ij^beta) / sum_k (tau_ik^alpha * eta_ik^beta)
    where eta_ij = 1/distance (here 1/weight or weight as attractiveness).

    Args:
        adjacency: n_nodes x n_nodes adjacency/weight matrix.
        n_nodes: Number of nodes.
        source: Source node index.
        sink: Sink node index.
        n_ants: Number of ants per iteration.
        max_iterations: Maximum iterations.
        alpha: Pheromone influence exponent.
        beta: Heuristic influence exponent.
        evaporation_rate: Pheromone evaporation rate rho.
        initial_pheromone: Initial pheromone level.
        seed: Random seed.

    Returns:
        ACORoutingSolution with best route and pheromone diagnostics.
    """
    adj = np.array(adjacency, dtype=np.float64)
    rng = np.random.RandomState(seed)

    # Check connectivity first
    if not _has_path(adj, source, sink, n_nodes):
        return ACORoutingSolution(
            best_route=(),
            best_cost=float('inf'),
            convergence_iteration=0,
            pheromone_distribution=tuple(
                tuple(initial_pheromone if adj[i, j] > 0 else 0.0 for j in range(n_nodes))
                for i in range(n_nodes)
            ),
            route_diversity=0.0,
        )

    # Initialise pheromone matrix
    tau = np.full((n_nodes, n_nodes), initial_pheromone)
    # Only on existing edges
    tau = tau * (adj > 0).astype(float)

    # Heuristic: weight as attractiveness (higher weight = more desirable)
    eta = adj.copy()
    eta[eta < 1e-15] = 0.0

    best_route = ()
    best_cost = float('inf')
    convergence_iter = max_iterations
    prev_best = float('inf')
    stable_count = 0

    all_routes_set = set()

    for iteration in range(1, max_iterations + 1):
        routes = []
        costs = []

        for _ in range(n_ants):
            route = [source]
            visited = {source}
            current = source
            stuck = False

            while current != sink:
                # Find unvisited neighbours
                candidates = []
                probs = []
                for j in range(n_nodes):
                    if j not in visited and adj[current, j] > 0:
                        p = (tau[current, j] ** alpha) * (eta[current, j] ** beta)
                        candidates.append(j)
                        probs.append(p)

                if not candidates:
                    stuck = True
                    break

                total_p = sum(probs)
                if total_p < 1e-15:
                    # Uniform random
                    next_node = candidates[rng.randint(len(candidates))]
                else:
                    probs = [p / total_p for p in probs]
                    next_node = candidates[rng.choice(len(candidates), p=probs)]

                route.append(next_node)
                visited.add(next_node)
                current = next_node

            if stuck:
                continue

            # Cost = number of hops (lower is better)
            cost = float(len(route) - 1)
            routes.append(route)
            costs.append(cost)
            all_routes_set.add(tuple(route))

            if cost < best_cost:
                best_cost = cost
                best_route = tuple(route)

        # Pheromone evaporation
        tau *= (1.0 - evaporation_rate)

        # Pheromone deposit
        for route, cost in zip(routes, costs):
            deposit = 1.0 / cost if cost > 0 else 1.0
            for k in range(len(route) - 1):
                tau[route[k], route[k + 1]] += deposit

        # Convergence check
        if abs(best_cost - prev_best) < 1e-10:
            stable_count += 1
            if stable_count >= 5:
                convergence_iter = iteration
                break
        else:
            stable_count = 0
            prev_best = best_cost

    if convergence_iter == max_iterations and stable_count < 5:
        convergence_iter = max_iterations

    # Route diversity: number of distinct routes found / total ants
    route_diversity = float(len(all_routes_set))

    return ACORoutingSolution(
        best_route=best_route,
        best_cost=best_cost,
        convergence_iteration=convergence_iter,
        pheromone_distribution=tuple(
            tuple(float(tau[i, j]) for j in range(n_nodes)) for i in range(n_nodes)
        ),
        route_diversity=route_diversity,
    )


def _has_path(adj: np.ndarray, source: int, sink: int, n: int) -> bool:
    """BFS check for path existence."""
    from collections import deque
    if source == sink:
        return True
    visited = set()
    visited.add(source)
    q = deque([source])
    while q:
        v = q.popleft()
        for u in range(n):
            if u not in visited and adj[v, u] > 0:
                if u == sink:
                    return True
                visited.add(u)
                q.append(u)
    return False


# ---------------------------------------------------------------------------
# P60: Spectral Graph Wavelet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralGraphWavelet:
    """Spectral graph wavelet transform result."""
    scales: tuple
    wavelet_coefficients: tuple
    cross_scale_correlations: tuple
    dominant_scale: int
    localization_index: float


def compute_spectral_graph_wavelet(
    adjacency: list,
    n_nodes: int,
    signal: list,
    num_scales: int = 5,
    scale_min: float = 0.5,
    scale_max: float = 10.0,
) -> SpectralGraphWavelet:
    """Spectral graph wavelet transform.

    Eigendecomposes the graph Laplacian L = U Lambda U^T, applies the
    wavelet generating kernel g(x) = x * exp(-x) at multiple scales,
    and computes wavelet coefficients W_f(s,n) = sum_k g(s*lambda_k) *
    f_hat(k) * u_k(n).

    Args:
        adjacency: n_nodes x n_nodes adjacency/weight matrix (symmetric).
        n_nodes: Number of nodes.
        signal: Length-n_nodes signal on the graph.
        num_scales: Number of wavelet scales.
        scale_min: Minimum scale.
        scale_max: Maximum scale.

    Returns:
        SpectralGraphWavelet with coefficients and cross-scale correlations.
    """
    if n_nodes <= 1:
        return SpectralGraphWavelet(
            scales=(1.0,),
            wavelet_coefficients=((0.0,),) if n_nodes == 1 else ((),),
            cross_scale_correlations=((1.0,),),
            dominant_scale=0,
            localization_index=0.0,
        )

    adj = np.array(adjacency, dtype=np.float64)
    f = np.array(signal, dtype=np.float64)

    # Build graph Laplacian
    L = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and adj[i, j] > 0:
                L[i, j] = -adj[i, j]
                L[i, i] += adj[i, j]

    # Eigendecompose
    eigenvalues, U = np.linalg.eigh(L)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Graph Fourier transform of signal
    f_hat = U.T @ f

    # Generate scales (logarithmic spacing)
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num_scales)

    # Wavelet kernel: g(x) = x * exp(-x)
    def wavelet_kernel(x):
        return x * np.exp(-x)

    # Compute wavelet coefficients at each scale
    coefficients = np.zeros((num_scales, n_nodes))
    for s_idx, s in enumerate(scales):
        # g(s * lambda_k) for each eigenvalue
        g_vals = wavelet_kernel(s * eigenvalues)
        # W_f(s, n) = sum_k g(s*lambda_k) * f_hat(k) * u_k(n)
        # = U @ diag(g_vals) @ f_hat
        coefficients[s_idx, :] = U @ (g_vals * f_hat)

    # Cross-scale correlations
    cross_corr = np.zeros((num_scales, num_scales))
    for i in range(num_scales):
        for j in range(num_scales):
            norm_i = np.linalg.norm(coefficients[i])
            norm_j = np.linalg.norm(coefficients[j])
            if norm_i > 1e-15 and norm_j > 1e-15:
                cross_corr[i, j] = np.dot(coefficients[i], coefficients[j]) / (norm_i * norm_j)
            elif i == j:
                cross_corr[i, j] = 1.0

    # Dominant scale: scale with maximum energy
    energies = np.array([np.sum(coefficients[s] ** 2) for s in range(num_scales)])
    dominant_scale = int(np.argmax(energies))

    # Localization index: ratio of max coefficient to total energy
    total_energy = np.sum(energies)
    if total_energy > 1e-15:
        max_coeff = np.max(np.abs(coefficients))
        localization_index = float(max_coeff ** 2 / total_energy)
    else:
        localization_index = 0.0

    return SpectralGraphWavelet(
        scales=tuple(float(s) for s in scales),
        wavelet_coefficients=tuple(
            tuple(float(c) for c in coefficients[s]) for s in range(num_scales)
        ),
        cross_scale_correlations=tuple(
            tuple(float(cross_corr[i, j]) for j in range(num_scales))
            for i in range(num_scales)
        ),
        dominant_scale=dominant_scale,
        localization_index=localization_index,
    )


# ---------------------------------------------------------------------------
# P63: Cheeger Constant ISL Bottleneck
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheegerBottleneck:
    """Cheeger constant and spectral bisection bottleneck analysis."""
    cheeger_constant: float
    fiedler_lower_bound: float
    spectral_upper_bound: float
    bottleneck_cut_edges: tuple
    partition_a: tuple
    partition_b: tuple
    cut_capacity: float


def compute_cheeger_bottleneck(
    adjacency: list,
    n_nodes: int,
) -> CheegerBottleneck:
    """Compute the Cheeger constant of an ISL graph via spectral bisection.

    Uses the Fiedler vector (eigenvector of second-smallest Laplacian
    eigenvalue) to partition the graph, then computes the Cheeger constant
    h = cut_weight / min(vol_A, vol_B) where vol is the sum of degrees
    in the partition.

    Also verifies the Cheeger inequality:
        lambda_2 / 2 <= h <= sqrt(2 * lambda_2 * d_max)

    Args:
        adjacency: n_nodes x n_nodes adjacency/weight matrix (symmetric).
        n_nodes: Number of nodes.

    Returns:
        CheegerBottleneck with Cheeger constant, bounds, and partition info.
    """
    adj = np.array(adjacency, dtype=np.float64)

    if n_nodes < 2:
        return CheegerBottleneck(
            cheeger_constant=0.0,
            fiedler_lower_bound=0.0,
            spectral_upper_bound=0.0,
            bottleneck_cut_edges=(),
            partition_a=(0,) if n_nodes == 1 else (),
            partition_b=(),
            cut_capacity=0.0,
        )

    # Build graph Laplacian
    L = np.zeros((n_nodes, n_nodes))
    degrees = np.zeros(n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and adj[i, j] > 0:
                L[i, j] = -adj[i, j]
                L[i, i] += adj[i, j]
                degrees[i] += adj[i, j]

    # Eigendecompose
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Fiedler value and vector
    lambda_2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    fiedler_vec = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(n_nodes)

    # Spectral bisection: sweep all threshold cuts on sorted Fiedler vector
    # to find the partition minimising h(S) = cut / min(vol_A, vol_B).
    sorted_indices = np.argsort(fiedler_vec)
    total_vol = float(np.sum(degrees))

    best_cheeger = float('inf')
    best_cut_pos = 1  # number of nodes in partition A

    for k in range(1, n_nodes):
        # partition A = first k nodes in sorted order, B = rest
        a_nodes = sorted_indices[:k]
        b_nodes = sorted_indices[k:]
        a_set = set(int(x) for x in a_nodes)

        cut_w = 0.0
        for i in a_nodes:
            for j in range(n_nodes):
                if j not in a_set and adj[int(i), j] > 0:
                    cut_w += adj[int(i), j]

        vol_a_k = sum(float(degrees[int(i)]) for i in a_nodes)
        vol_b_k = total_vol - vol_a_k
        min_vol_k = min(vol_a_k, vol_b_k)

        if min_vol_k > 1e-15:
            h_k = cut_w / min_vol_k
            if h_k < best_cheeger:
                best_cheeger = h_k
                best_cut_pos = k

    # Build the best partition
    partition_a = [int(x) for x in sorted_indices[:best_cut_pos]]
    partition_b = [int(x) for x in sorted_indices[best_cut_pos:]]

    # Compute cut weight and cut edges for the best partition
    set_a = set(partition_a)
    cut_weight = 0.0
    cut_edges = []
    for i in partition_a:
        for j in partition_b:
            if adj[i, j] > 0:
                cut_weight += adj[i, j]
                cut_edges.append((i, j))

    # Compute volumes
    vol_a = sum(float(degrees[i]) for i in partition_a)
    vol_b = sum(float(degrees[i]) for i in partition_b)
    min_vol = min(vol_a, vol_b)

    # Cheeger constant
    cheeger = cut_weight / min_vol if min_vol > 1e-15 else 0.0

    # Cheeger inequality bounds (volume-based formulation)
    # For h = cut / min(vol_A, vol_B) with unnormalised Laplacian eigenvalue lambda_2:
    #   lambda_2 / (2 * d_max) <= h <= sqrt(2 * lambda_2 * d_max)
    d_max = float(np.max(degrees)) if n_nodes > 0 else 0.0
    fiedler_lower = lambda_2 / (2.0 * d_max) if d_max > 1e-15 else 0.0
    spectral_upper = math.sqrt(2.0 * lambda_2 * d_max) if lambda_2 > 0 and d_max > 0 else 0.0

    return CheegerBottleneck(
        cheeger_constant=cheeger,
        fiedler_lower_bound=fiedler_lower,
        spectral_upper_bound=spectral_upper,
        bottleneck_cut_edges=tuple(cut_edges),
        partition_a=tuple(sorted(partition_a)),
        partition_b=tuple(sorted(partition_b)),
        cut_capacity=cut_weight,
    )
