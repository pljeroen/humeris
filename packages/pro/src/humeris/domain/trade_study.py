# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Parametric Walker constellation trade studies.

Sweeps constellation parameters (altitude, inclination, planes, sats/plane,
phase factor) and evaluates coverage FoMs for each configuration. Supports
Pareto frontier extraction for cost-vs-performance analysis.

"""
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.constellation import (
    ShellConfig,
    generate_walker_shell,
)
from humeris.domain.propagation import (
    OrbitalState,
    derive_orbital_state,
)
from humeris.domain.revisit import (
    CoverageResult,
    compute_revisit,
)


@dataclass(frozen=True)
class WalkerConfig:
    """Walker constellation configuration for parametric trade study."""
    altitude_km: float
    inclination_deg: float
    num_planes: int
    sats_per_plane: int
    phase_factor: int


@dataclass(frozen=True)
class TradePoint:
    """Single evaluated configuration in a trade study."""
    config: WalkerConfig
    total_satellites: int
    coverage: CoverageResult


@dataclass(frozen=True)
class TradeStudyResult:
    """Complete trade study results."""
    points: tuple[TradePoint, ...]
    analysis_duration_s: float
    min_elevation_deg: float


def generate_walker_configs(
    altitude_range: tuple[float, ...],
    inclination_range: tuple[float, ...],
    planes_range: tuple[int, ...],
    sats_per_plane_range: tuple[int, ...],
    phase_factor_range: tuple[int, ...] | None = None,
) -> list[WalkerConfig]:
    """Generate Cartesian product of parameter ranges as WalkerConfigs.

    If phase_factor_range is None, uses 0 for all configs.
    Validates: planes >= 1, sats_per_plane >= 1, altitude > 0.
    """
    for alt in altitude_range:
        if alt <= 0:
            raise ValueError(f"altitude must be > 0, got {alt}")
    for p in planes_range:
        if p < 1:
            raise ValueError(f"num_planes must be >= 1, got {p}")
    for s in sats_per_plane_range:
        if s < 1:
            raise ValueError(f"sats_per_plane must be >= 1, got {s}")

    if phase_factor_range is None:
        phase_factor_range = (0,)

    configs: list[WalkerConfig] = []
    for alt in altitude_range:
        for inc in inclination_range:
            for planes in planes_range:
                for spp in sats_per_plane_range:
                    for pf in phase_factor_range:
                        configs.append(WalkerConfig(
                            altitude_km=alt,
                            inclination_deg=inc,
                            num_planes=planes,
                            sats_per_plane=spp,
                            phase_factor=pf,
                        ))
    return configs


def run_walker_trade_study(
    configs: list[WalkerConfig],
    analysis_epoch: datetime,
    analysis_duration: timedelta,
    analysis_step: timedelta,
    min_elevation_deg: float = 10.0,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
) -> TradeStudyResult:
    """Run coverage analysis for each Walker configuration.

    For each config:
    1. Generate Walker shell via generate_walker_shell
    2. Derive orbital states for all satellites
    3. Run compute_revisit for the full analysis window
    4. Collect results as TradePoint
    """
    points: list[TradePoint] = []

    for config in configs:
        shell = ShellConfig(
            altitude_km=config.altitude_km,
            inclination_deg=config.inclination_deg,
            num_planes=config.num_planes,
            sats_per_plane=config.sats_per_plane,
            phase_factor=config.phase_factor,
            raan_offset_deg=0.0,
            shell_name='TradeStudy',
        )
        sats = generate_walker_shell(shell)
        states = [derive_orbital_state(s, analysis_epoch) for s in sats]

        coverage = compute_revisit(
            states, analysis_epoch, analysis_duration, analysis_step,
            min_elevation_deg=min_elevation_deg,
            lat_step_deg=lat_step_deg, lon_step_deg=lon_step_deg,
            lat_range=lat_range, lon_range=lon_range,
        )

        points.append(TradePoint(
            config=config,
            total_satellites=config.num_planes * config.sats_per_plane,
            coverage=coverage,
        ))

    return TradeStudyResult(
        points=tuple(points),
        analysis_duration_s=analysis_duration.total_seconds(),
        min_elevation_deg=min_elevation_deg,
    )


def pareto_front_indices(
    costs: list[float],
    metrics: list[float],
) -> list[int]:
    """Return indices of Pareto-optimal points (minimizing both cost and metric).

    A point (c_i, m_i) is non-dominated if no other point (c_j, m_j) has
    c_j <= c_i AND m_j <= m_i (with at least one strict inequality).

    Returns sorted list of indices into the input arrays.
    """
    n = len(costs)
    if n == 0:
        return []

    front: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if costs[j] <= costs[i] and metrics[j] <= metrics[i]:
                if costs[j] < costs[i] or metrics[j] < metrics[i]:
                    dominated = True
                    break
        if not dominated:
            front.append(i)

    return sorted(front)


# ── P49: Burnside Symmetry Counting ──────────────────────────────


@dataclass(frozen=True)
class SymmetryEquivalence:
    """Result of Burnside symmetry counting for Walker constellations.

    A Walker(T, P, F) constellation has T total satellites in P planes
    with S = T/P satellites per plane. The symmetry group is
    G = Z_P x Z_S (cyclic shifts of planes and satellites within planes).

    For binary configurations (m active out of T), Burnside's lemma gives
    the number of truly distinct configurations modulo symmetry.

    Attributes:
        total_configurations: C(T, m) total binary configurations.
        symmetry_group_order: |G| = P * S.
        distinct_configurations: Number of orbits under G (Burnside count).
        redundancy_factor: total_configurations / distinct_configurations.
        fixed_point_counts: Per group element |Fix(g)| values as tuple.
    """
    total_configurations: int
    symmetry_group_order: int
    distinct_configurations: int
    redundancy_factor: float
    fixed_point_counts: tuple


def compute_burnside_symmetry(
    total_sats: int,
    num_planes: int,
    num_active: int,
) -> SymmetryEquivalence:
    """Count distinct constellation configurations via Burnside's lemma.

    For a Walker constellation with T satellites in P planes (S = T/P per
    plane), the symmetry group G = Z_P x Z_S acts by cyclic shifts.

    A group element (k, j) shifts planes by k and satellites within each
    plane by j. A binary configuration (m active out of T) is fixed by
    (k, j) if the configuration is invariant under this combined shift.

    Burnside: distinct = (1/|G|) * sum_{g in G} |Fix(g)|

    For each (k, j), |Fix(g)| counts configurations of T bits with exactly
    m ones that are invariant under the permutation induced by (k, j).

    The permutation induced by (k, j) on satellite index (p, s) is:
        (p, s) -> ((p + k) mod P, (s + j) mod S)

    A configuration is fixed iff it is constant on each cycle of this
    permutation. The number of cycles is gcd(k, P) * gcd(j, S).
    A fixed configuration with m active satellites must distribute m among
    c cycles, each cycle being entirely active or inactive.
    So |Fix(k,j)| = C(c, m/cycle_length) if cycle_length divides m,
    where c = num_cycles and cycle_length = T/c.

    More precisely: cycle_length = lcm(P/gcd(k,P), S/gcd(j,S)) and
    num_cycles = T / cycle_length. Fixed configs = C(num_cycles, m/cycle_length)
    if cycle_length | m, else 0.

    Args:
        total_sats: Total satellites T.
        num_planes: Number of planes P (must divide T).
        num_active: Number of active satellites m.

    Returns:
        SymmetryEquivalence with Burnside counting results.

    Raises:
        ValueError: If parameters are invalid.
    """
    import math as _math

    T = total_sats
    P = num_planes

    if T <= 0:
        raise ValueError(f"total_sats must be positive, got {T}")
    if P <= 0:
        raise ValueError(f"num_planes must be positive, got {P}")
    if T % P != 0:
        raise ValueError(f"total_sats ({T}) must be divisible by num_planes ({P})")
    if num_active < 0 or num_active > T:
        raise ValueError(
            f"num_active must be in [0, {T}], got {num_active}"
        )

    S = T // P
    group_order = P * S

    # Total configurations: C(T, m)
    total_configs = _math.comb(T, num_active)

    # Burnside counting
    fixed_counts = []
    total_fixed = 0

    for k in range(P):
        for j in range(S):
            # Number of cycles of the permutation (k, j) on T satellites
            # The permutation acts on (p, s) -> ((p+k) mod P, (s+j) mod S)
            # This is a direct product of two cyclic permutations.
            # Number of cycles = gcd(k, P) * gcd(j, S) when acting on Z_P x Z_S
            # Cycle length for the combined permutation:
            # Each cycle of the Z_P action has length P/gcd(k,P)
            # Each cycle of the Z_S action has length S/gcd(j,S)
            # Combined cycle length = lcm(P/gcd(k,P), S/gcd(j,S))
            # Number of combined cycles = T / combined_cycle_length

            c_p = _math.gcd(k, P) if k > 0 else P
            c_s = _math.gcd(j, S) if j > 0 else S
            len_p = P // c_p
            len_s = S // c_s
            cycle_len = _lcm(len_p, len_s)
            num_cycles = T // cycle_len

            # Fixed iff m is divisible by cycle_len
            if cycle_len > 0 and num_active % cycle_len == 0:
                active_cycles = num_active // cycle_len
                fix_count = _math.comb(num_cycles, active_cycles)
            else:
                fix_count = 0

            fixed_counts.append(fix_count)
            total_fixed += fix_count

    distinct = total_fixed // group_order
    # Handle rounding (Burnside guarantees integer result)
    if total_fixed % group_order != 0:
        # Floating point: use exact division
        distinct = round(total_fixed / group_order)

    if distinct > 0:
        redundancy = total_configs / distinct
    else:
        redundancy = float('inf') if total_configs > 0 else 1.0

    return SymmetryEquivalence(
        total_configurations=total_configs,
        symmetry_group_order=group_order,
        distinct_configurations=distinct,
        redundancy_factor=redundancy,
        fixed_point_counts=tuple(fixed_counts),
    )


def _lcm(a: int, b: int) -> int:
    """Least common multiple."""
    import math as _math
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // _math.gcd(a, b)


# ── P52: Lattice Partial Orders ──────────────────────────────────


@dataclass(frozen=True)
class ConstellationLattice:
    """Partial order analysis of constellation configurations.

    Given a set of constellation configurations with coverage values
    at multiple grid points, build a partial order: C1 <= C2 iff
    f_g(C1) <= f_g(C2) for all grid points g.

    Attributes:
        num_configurations: Number of configurations analysed.
        partial_order_width: Maximum antichain size (greedy approximation).
        partial_order_height: Longest chain length.
        meet_irreducibles: Number of meet-irreducible elements.
        join_irreducibles: Number of join-irreducible elements.
        num_comparable_pairs: Number of pairs (i, j) with i <= j.
        comparability_fraction: Fraction of pairs that are comparable.
    """
    num_configurations: int
    partial_order_width: int
    partial_order_height: int
    meet_irreducibles: int
    join_irreducibles: int
    num_comparable_pairs: int
    comparability_fraction: float


def compute_constellation_lattice(
    coverage_matrix: list[list[float]],
) -> ConstellationLattice:
    """Analyse the partial order structure of constellation configurations.

    Args:
        coverage_matrix: List of configurations, each a list of coverage
            values at grid points. Shape: (num_configs, num_grid_points).
            C1 <= C2 iff C1[g] <= C2[g] for all g.

    Returns:
        ConstellationLattice with partial order metrics.

    Raises:
        ValueError: If coverage_matrix is empty or inconsistent.
    """
    if not coverage_matrix:
        raise ValueError("coverage_matrix must be non-empty")

    n = len(coverage_matrix)
    if n == 0:
        raise ValueError("coverage_matrix must be non-empty")

    num_grid = len(coverage_matrix[0])
    for row in coverage_matrix:
        if len(row) != num_grid:
            raise ValueError("All configurations must have the same number of grid points")

    mat = np.array(coverage_matrix, dtype=np.float64)

    # Build comparison matrix: leq[i][j] = True if config i <= config j
    leq = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            leq[i, j] = bool(np.all(mat[i] <= mat[j]))

    # Count comparable pairs (i < j where i <= j or j <= i, i != j)
    num_comparable = 0
    total_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            if leq[i, j] or leq[j, i]:
                num_comparable += 1

    comparability = num_comparable / total_pairs if total_pairs > 0 else 1.0

    # Height: longest chain (greedy DFS from each element)
    # Build strict order: strictly_less[i][j] = leq[i][j] and not leq[j][i]
    strictly_less = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and leq[i, j] and not leq[j, i]:
                strictly_less[i, j] = True

    height = _longest_chain(strictly_less, n)

    # Width: maximum antichain (greedy approximation via Dilworth)
    width = _max_antichain_greedy(leq, n)

    # Meet-irreducible: x is meet-irreducible if it has exactly one
    # element covering it from above (exactly one element y such that
    # x < y and there is no z with x < z < y)
    meet_irr = 0
    join_irr = 0

    for x in range(n):
        # Elements covering x (immediate successors)
        covers_x = []
        for y in range(n):
            if not strictly_less[x, y]:
                continue
            # Check no z with x < z < y
            is_cover = True
            for z in range(n):
                if z == x or z == y:
                    continue
                if strictly_less[x, z] and strictly_less[z, y]:
                    is_cover = False
                    break
            if is_cover:
                covers_x.append(y)
        if len(covers_x) == 1:
            meet_irr += 1

        # Elements covered by x (immediate predecessors)
        covered_by_x = []
        for y in range(n):
            if not strictly_less[y, x]:
                continue
            is_cover = True
            for z in range(n):
                if z == x or z == y:
                    continue
                if strictly_less[y, z] and strictly_less[z, x]:
                    is_cover = False
                    break
            if is_cover:
                covered_by_x.append(y)
        if len(covered_by_x) == 1:
            join_irr += 1

    return ConstellationLattice(
        num_configurations=n,
        partial_order_width=width,
        partial_order_height=height,
        meet_irreducibles=meet_irr,
        join_irreducibles=join_irr,
        num_comparable_pairs=num_comparable,
        comparability_fraction=comparability,
    )


def _longest_chain(strictly_less: np.ndarray, n: int) -> int:
    """Find the longest chain in the partial order (longest path in DAG)."""
    # Memoized DFS
    memo: dict[int, int] = {}

    def dfs(node: int) -> int:
        if node in memo:
            return memo[node]
        best = 1
        for succ in range(n):
            if strictly_less[node, succ]:
                best = max(best, 1 + dfs(succ))
        memo[node] = best
        return best

    return max((dfs(i) for i in range(n)), default=1)


def _max_antichain_greedy(leq: np.ndarray, n: int) -> int:
    """Greedy approximation of maximum antichain size.

    Iteratively pick the element with the fewest comparable elements,
    add it to the antichain, remove all comparable elements.
    """
    available = set(range(n))
    antichain_size = 0

    while available:
        # Pick element with fewest comparable elements in available set
        best = min(available, key=lambda x: sum(
            1 for y in available if y != x and (leq[x, y] or leq[y, x])
        ))
        antichain_size += 1
        # Remove best and all comparable elements
        to_remove = {best}
        for y in available:
            if y != best and (leq[best, y] or leq[y, best]):
                # Only remove if strictly comparable (not just equal)
                if not (leq[best, y] and leq[y, best]):
                    to_remove.add(y)
        available -= to_remove

    return antichain_size


# ── P61: Mapper Algorithm for Design Space TDA ──────────────────


@dataclass(frozen=True)
class DesignSpaceTopology:
    """Topological analysis of design space via the Mapper algorithm.

    The Mapper algorithm discretizes a continuous design space into
    a simplicial complex that preserves topological features.

    Attributes:
        num_components: Number of connected components (beta_0).
        num_loops: Number of loops / 1-cycles (beta_1).
        cluster_sizes: Number of design points in each cluster.
        edges: Edges in the nerve complex as (cluster_i, cluster_j) pairs.
        design_families: Mapping of cluster index to list of design point indices.
        bottleneck_transitions: Cluster indices that connect different components
            (bridge nodes in the nerve graph).
    """
    num_components: int
    num_loops: int
    cluster_sizes: tuple
    edges: tuple
    design_families: tuple
    bottleneck_transitions: tuple


def compute_design_space_topology(
    design_points: list[list[float]],
    filter_values: list[float],
    num_intervals: int = 10,
    overlap_fraction: float = 0.3,
    cluster_threshold: float = 1.0,
) -> DesignSpaceTopology:
    """Compute topological features of design space via Mapper algorithm.

    1. Filter function f partitions design points into overlapping intervals.
    2. Within each interval, single-linkage clustering groups nearby points.
    3. Build nerve complex: clusters are vertices, edges connect clusters
       that share design points.
    4. Count components (beta_0) and loops (beta_1 = edges - vertices + components).

    Args:
        design_points: List of design vectors (each a list of floats).
        filter_values: Scalar filter value for each design point.
        num_intervals: Number of intervals covering the filter range.
        overlap_fraction: Fraction of overlap between adjacent intervals.
        cluster_threshold: Distance threshold for single-linkage clustering.

    Returns:
        DesignSpaceTopology with topological invariants.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    if len(design_points) != len(filter_values):
        raise ValueError(
            f"design_points ({len(design_points)}) and filter_values "
            f"({len(filter_values)}) must have the same length"
        )
    if len(design_points) == 0:
        raise ValueError("design_points must be non-empty")
    if num_intervals < 1:
        raise ValueError(f"num_intervals must be >= 1, got {num_intervals}")
    if overlap_fraction < 0 or overlap_fraction >= 1:
        raise ValueError(
            f"overlap_fraction must be in [0, 1), got {overlap_fraction}"
        )

    n = len(design_points)
    pts = np.array(design_points, dtype=np.float64)
    f_vals = np.array(filter_values, dtype=np.float64)

    f_min = float(np.min(f_vals))
    f_max = float(np.max(f_vals))

    if f_min == f_max:
        # All same filter value: one interval, one cluster per connected component
        f_max = f_min + 1.0

    # Build overlapping intervals
    interval_width = (f_max - f_min) / num_intervals
    step = interval_width * (1.0 - overlap_fraction)
    if step <= 0:
        step = interval_width

    intervals = []
    start = f_min
    while start < f_max:
        end = start + interval_width
        intervals.append((start, end))
        start += step

    # Cluster within each interval using single-linkage
    all_clusters: list[list[int]] = []  # each cluster is a list of point indices
    point_to_clusters: dict[int, list[int]] = {i: [] for i in range(n)}

    for interval_start, interval_end in intervals:
        # Find points in this interval
        in_interval = [
            i for i in range(n)
            if interval_start <= f_vals[i] <= interval_end
        ]
        if not in_interval:
            continue

        # Single-linkage clustering
        clusters = _single_linkage_cluster(pts, in_interval, cluster_threshold)

        for cluster in clusters:
            cluster_idx = len(all_clusters)
            all_clusters.append(cluster)
            for pt_idx in cluster:
                point_to_clusters[pt_idx].append(cluster_idx)

    num_clusters = len(all_clusters)
    if num_clusters == 0:
        return DesignSpaceTopology(
            num_components=0, num_loops=0,
            cluster_sizes=(), edges=(),
            design_families=(), bottleneck_transitions=(),
        )

    # Build nerve complex: edge between clusters sharing points
    edge_set: set[tuple[int, int]] = set()
    for pt_idx in range(n):
        cl_list = point_to_clusters[pt_idx]
        for a in range(len(cl_list)):
            for b in range(a + 1, len(cl_list)):
                e = (min(cl_list[a], cl_list[b]), max(cl_list[a], cl_list[b]))
                edge_set.add(e)

    edges = sorted(edge_set)
    num_edges = len(edges)

    # Count connected components via union-find
    parent = list(range(num_clusters))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for u, v in edges:
        union(u, v)

    num_components = len(set(find(i) for i in range(num_clusters)))

    # Betti numbers: beta_0 = components, beta_1 = edges - vertices + components
    # (Euler characteristic for 1-complex)
    beta_1 = max(0, num_edges - num_clusters + num_components)

    # Cluster sizes
    cluster_sizes = tuple(len(c) for c in all_clusters)

    # Design families: cluster index -> point indices
    design_families = tuple(tuple(c) for c in all_clusters)

    # Bottleneck transitions: bridge edges (removal disconnects the graph)
    bottleneck_nodes: set[int] = set()
    for u, v in edges:
        # Test if removing this edge increases components
        temp_parent = list(range(num_clusters))

        def temp_find(x: int) -> int:
            while temp_parent[x] != x:
                temp_parent[x] = temp_parent[temp_parent[x]]
                x = temp_parent[x]
            return x

        for eu, ev in edges:
            if (eu, ev) == (u, v):
                continue
            ru, rv = temp_find(eu), temp_find(ev)
            if ru != rv:
                temp_parent[ru] = rv

        temp_components = len(set(temp_find(i) for i in range(num_clusters)))
        if temp_components > num_components:
            bottleneck_nodes.add(u)
            bottleneck_nodes.add(v)

    return DesignSpaceTopology(
        num_components=num_components,
        num_loops=beta_1,
        cluster_sizes=cluster_sizes,
        edges=tuple(edges),
        design_families=design_families,
        bottleneck_transitions=tuple(sorted(bottleneck_nodes)),
    )


def _single_linkage_cluster(
    pts: np.ndarray,
    indices: list[int],
    threshold: float,
) -> list[list[int]]:
    """Single-linkage clustering of points at given indices.

    Two points are in the same cluster if they can be connected through
    a chain of points each within `threshold` distance of the next.
    """
    n = len(indices)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a in range(n):
        for b in range(a + 1, n):
            dist = float(np.linalg.norm(pts[indices[a]] - pts[indices[b]]))
            if dist <= threshold:
                union(a, b)

    clusters: dict[int, list[int]] = {}
    for a in range(n):
        root = find(a)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(indices[a])

    return list(clusters.values())
