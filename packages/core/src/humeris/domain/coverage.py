# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Grid-based coverage analysis.

Computes a snapshot of how many satellites are visible from each point
on a latitude/longitude grid at a given time.

"""
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from humeris.domain.propagation import (
    OrbitalState,
    propagate_ecef_to,
)
from humeris.domain.observation import (
    GroundStation,
    compute_observation,
)


@dataclass(frozen=True)
class CoveragePoint:
    """A grid point with its satellite visibility count."""
    lat_deg: float
    lon_deg: float
    visible_count: int


def compute_coverage_snapshot(
    orbital_states: list[OrbitalState],
    analysis_time: datetime,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    min_elevation_deg: float = 10.0,
    lat_range: tuple[float, float] = (-90, 90),
    lon_range: tuple[float, float] = (-180, 180),
) -> list[CoveragePoint]:
    """
    Compute a coverage snapshot: how many satellites are visible per grid point.

    Precomputes all satellite ECEF positions, then for each grid point
    counts how many are above min_elevation.

    Args:
        orbital_states: List of OrbitalState objects to evaluate.
        analysis_time: UTC time for the snapshot.
        lat_step_deg: Latitude grid spacing in degrees.
        lon_step_deg: Longitude grid spacing in degrees.
        min_elevation_deg: Minimum elevation for visibility.
        lat_range: (min_lat, max_lat) in degrees.
        lon_range: (min_lon, max_lon) in degrees.

    Returns:
        List of CoveragePoint objects for the entire grid.
    """
    sat_ecefs = [propagate_ecef_to(state, analysis_time) for state in orbital_states]

    grid: list[CoveragePoint] = []

    lat = lat_range[0]
    while lat <= lat_range[1] + 1e-9:
        lon = lon_range[0]
        while lon <= lon_range[1] - lon_step_deg + 1e-9:
            station = GroundStation(name='grid', lat_deg=lat, lon_deg=lon, alt_m=0.0)
            count = 0
            for sat_ecef in sat_ecefs:
                obs = compute_observation(station, sat_ecef)
                if obs.elevation_deg >= min_elevation_deg:
                    count += 1
            grid.append(CoveragePoint(lat_deg=lat, lon_deg=lon, visible_count=count))
            lon += lon_step_deg
        lat += lat_step_deg

    return grid


# ── Morse Theory for Coverage Phase Transitions (P7) ──────────────

@dataclass(frozen=True)
class CoverageCriticalPoint:
    """A critical point of the coverage function on S^2.

    Index 0 = minimum (coverage gap center).
    Index 1 = saddle (gap boundary / transition).
    Index 2 = maximum (coverage hotspot).
    """
    lat_deg: float
    lon_deg: float
    coverage_count: int
    index: int          # Morse index: 0=min, 1=saddle, 2=max
    depth: float        # For minima: deficit below mean neighbors; for maxima: excess


@dataclass(frozen=True)
class CoverageMorseAnalysis:
    """Morse-theoretic analysis of coverage topology on a rectangular grid.

    Critical points are where the discrete gradient of the coverage
    function vanishes. The Morse index at each critical point is the
    number of negative eigenvalues of the discrete Hessian.

    Note: the grid is rectangular (lat-lon with boundaries), NOT the
    sphere S^2. For a rectangular domain with boundary, the Euler
    characteristic is chi = 1, not chi(S^2) = 2. In practice, on a
    discrete grid this is approximate due to discretization artifacts
    and boundary effects.

    Phase transition altitudes are identified by sweeping altitude
    and detecting where the Euler characteristic of the coverage
    sub-level sets changes.

    References:
        Milnor (1963). Morse Theory. Princeton University Press.
        Forman (1998). Morse theory for cell complexes.
    """
    critical_points: tuple[CoverageCriticalPoint, ...]
    num_maxima: int
    num_saddles: int
    num_minima: int
    euler_characteristic: int   # #maxima - #saddles + #minima (rectangular domain: ~1)
    phase_transition_altitudes: tuple[float, ...]  # Altitudes where topology changes


def compute_coverage_morse(
    coverage_grid: list[CoveragePoint],
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
) -> CoverageMorseAnalysis:
    """Compute Morse-theoretic coverage analysis from a coverage grid.

    Identifies critical points (minima, saddles, maxima) of the discrete
    coverage function on the latitude-longitude grid, classifies them
    by Morse index, and verifies the Euler characteristic constraint.

    The algorithm:
    1. Organize coverage data into a 2D array indexed by (lat, lon).
    2. For each interior grid point, compute the discrete Hessian from
       finite differences of coverage counts.
    3. Classify critical points by eigenvalue signs of the Hessian.
    4. Compute Euler characteristic as #maxima - #saddles + #minima.

    Args:
        coverage_grid: List of CoveragePoint from compute_coverage_snapshot.
        lat_step_deg: Latitude grid spacing (must match the snapshot).
        lon_step_deg: Longitude grid spacing (must match the snapshot).

    Returns:
        CoverageMorseAnalysis with critical points and Euler characteristic.
    """
    if not coverage_grid:
        return CoverageMorseAnalysis(
            critical_points=(),
            num_maxima=0,
            num_saddles=0,
            num_minima=0,
            euler_characteristic=0,
            phase_transition_altitudes=(),
        )

    # Build 2D grid from coverage points
    lats = sorted(set(cp.lat_deg for cp in coverage_grid))
    lons = sorted(set(cp.lon_deg for cp in coverage_grid))

    if len(lats) < 3 or len(lons) < 3:
        return CoverageMorseAnalysis(
            critical_points=(),
            num_maxima=0,
            num_saddles=0,
            num_minima=0,
            euler_characteristic=0,
            phase_transition_altitudes=(),
        )

    n_lat = len(lats)
    n_lon = len(lons)

    # Map (lat, lon) -> coverage count
    cov_map: dict[tuple[float, float], int] = {}
    for cp in coverage_grid:
        cov_map[(cp.lat_deg, cp.lon_deg)] = cp.visible_count

    # Build numpy array
    grid = np.zeros((n_lat, n_lon))
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            grid[i, j] = cov_map.get((lat, lon), 0)

    # Find critical points using discrete gradient and Hessian
    critical_points: list[CoverageCriticalPoint] = []
    mean_coverage = float(np.mean(grid))

    for i in range(1, n_lat - 1):
        for j in range(1, n_lon - 1):
            val = grid[i, j]

            # Discrete gradient (forward differences)
            g_lat = (grid[i + 1, j] - grid[i - 1, j]) / 2.0
            g_lon = (grid[i, j + 1] - grid[i, j - 1]) / 2.0

            # Critical point: gradient ~ 0
            # For integer coverage counts, check if this is a local extremum
            # or saddle by examining neighbors directly
            neighbors = [
                grid[i - 1, j], grid[i + 1, j],
                grid[i, j - 1], grid[i, j + 1],
            ]

            n_below = sum(1 for n_val in neighbors if n_val < val)
            n_above = sum(1 for n_val in neighbors if n_val > val)
            n_equal = sum(1 for n_val in neighbors if n_val == val)

            # Discrete Hessian (second derivatives)
            h_ll = grid[i + 1, j] - 2.0 * val + grid[i - 1, j]  # d^2f/dlat^2
            h_nn = grid[i, j + 1] - 2.0 * val + grid[i, j - 1]  # d^2f/dlon^2
            h_ln = (grid[i + 1, j + 1] - grid[i + 1, j - 1]
                    - grid[i - 1, j + 1] + grid[i - 1, j - 1]) / 4.0

            # Eigenvalues of Hessian
            trace = h_ll + h_nn
            det = h_ll * h_nn - h_ln * h_ln
            discriminant = trace * trace - 4.0 * det

            if discriminant < 0:
                discriminant = 0.0
            sqrt_disc = float(np.sqrt(discriminant))
            eig1 = (trace + sqrt_disc) / 2.0
            eig2 = (trace - sqrt_disc) / 2.0

            # Classify critical points
            is_critical = False
            morse_index = -1

            if n_below == 4:
                # Local maximum (all neighbors lower)
                is_critical = True
                morse_index = 2
            elif n_above == 4:
                # Local minimum (all neighbors higher)
                is_critical = True
                morse_index = 0
            elif n_below >= 1 and n_above >= 1 and abs(g_lat) < 1.0 and abs(g_lon) < 1.0:
                # Potential saddle: check Hessian eigenvalue signs
                if eig1 > 0 and eig2 < 0:
                    is_critical = True
                    morse_index = 1
                elif eig1 < 0 and eig2 > 0:
                    is_critical = True
                    morse_index = 1
                # Also classify based on neighbor pattern:
                # If some neighbors above and some below with small gradient
                elif n_below >= 2 and n_above >= 2:
                    is_critical = True
                    morse_index = 1

            if is_critical and morse_index >= 0:
                # Depth: for minima, how far below mean neighbors;
                # for maxima, how far above
                mean_neighbors = sum(neighbors) / len(neighbors)
                if morse_index == 0:
                    depth = mean_neighbors - val
                elif morse_index == 2:
                    depth = val - mean_neighbors
                else:
                    depth = abs(val - mean_neighbors)

                critical_points.append(CoverageCriticalPoint(
                    lat_deg=lats[i],
                    lon_deg=lons[j],
                    coverage_count=int(val),
                    index=morse_index,
                    depth=depth,
                ))

    # Count by type
    num_maxima = sum(1 for cp in critical_points if cp.index == 2)
    num_saddles = sum(1 for cp in critical_points if cp.index == 1)
    num_minima = sum(1 for cp in critical_points if cp.index == 0)

    # Euler characteristic: should be ~1 for rectangular domain with boundary
    euler = num_maxima - num_saddles + num_minima

    # Phase transition altitudes: find coverage levels where topology changes
    # (connected components of sub-level sets merge or split)
    coverage_levels = sorted(set(int(grid[i, j]) for i in range(n_lat) for j in range(n_lon)))
    transition_altitudes: list[float] = []

    prev_components = 0
    for level in coverage_levels:
        # Count connected components of {(lat,lon) : coverage >= level}
        mask = (grid >= level).astype(int)
        components = _count_connected_components(mask)
        if prev_components != 0 and components != prev_components:
            transition_altitudes.append(float(level))
        prev_components = components

    return CoverageMorseAnalysis(
        critical_points=tuple(critical_points),
        num_maxima=num_maxima,
        num_saddles=num_saddles,
        num_minima=num_minima,
        euler_characteristic=euler,
        phase_transition_altitudes=tuple(transition_altitudes),
    )


def _count_connected_components(mask: np.ndarray) -> int:
    """Count connected components in a 2D binary mask using BFS.

    Uses 4-connectivity (up, down, left, right).

    Args:
        mask: 2D binary array (1 = occupied, 0 = empty).

    Returns:
        Number of connected components.
    """
    n_rows, n_cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    count = 0

    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j] == 1 and not visited[i, j]:
                # BFS from this cell
                count += 1
                queue = [(i, j)]
                visited[i, j] = True
                while queue:
                    ci, cj = queue.pop(0)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < n_rows and 0 <= nj < n_cols:
                            if mask[ni, nj] == 1 and not visited[ni, nj]:
                                visited[ni, nj] = True
                                queue.append((ni, nj))
    return count


# ── Persistent Homology for Coverage Holes (P30) ──────────────────

@dataclass(frozen=True)
class PersistenceInterval:
    """A single persistence interval (birth, death, dimension).

    birth: filtration value at which the feature appears.
    death: filtration value at which the feature is destroyed.
           math.inf for features that never die.
    dimension: 0 = connected component, 1 = loop/hole.
    """
    birth: float
    death: float
    dimension: int


@dataclass(frozen=True)
class PersistentCoverageHomology:
    """Persistent homology analysis of coverage holes.

    Sublevel set filtration F_k = {grid points with coverage_count <= k}
    reveals how coverage gaps connect (H_0) and enclose holes (H_1)
    as the coverage threshold increases.

    References:
        Edelsbrunner & Harer (2010). Computational Topology.
        Carlsson (2009). Topology and Data.
    """
    intervals: tuple[PersistenceInterval, ...]
    num_significant_holes: int      # H_1 features with persistence > threshold
    max_hole_persistence: float     # Largest H_1 persistence
    betti_0_curve: tuple[int, ...]  # beta_0(k) for each coverage level k
    betti_1_curve: tuple[int, ...]  # beta_1(k) for each coverage level k
    total_persistence: float        # Sum of all persistence values


class _UnionFind:
    """Union-Find data structure for tracking connected components.

    Uses path compression and union by rank for near-O(1) amortized operations.
    Tracks birth times for the elder rule.
    """

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}
        self._birth: dict[int, float] = {}

    def make_set(self, x: int, birth: float) -> None:
        """Create a new component with the given birth time."""
        self._parent[x] = x
        self._rank[x] = 0
        self._birth[x] = birth

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: int, y: int) -> tuple[int, int] | None:
        """Union two components. Returns (survivor_root, dying_root) or None if same.

        Uses elder rule: the component born earlier survives.
        """
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return None

        # Elder rule: component with smaller (earlier) birth survives
        if self._birth[rx] < self._birth[ry]:
            survivor, dying = rx, ry
        elif self._birth[ry] < self._birth[rx]:
            survivor, dying = ry, rx
        else:
            # Same birth time: use rank as tiebreaker
            if self._rank[rx] >= self._rank[ry]:
                survivor, dying = rx, ry
            else:
                survivor, dying = ry, rx

        self._parent[dying] = survivor
        if self._rank[survivor] == self._rank[dying]:
            self._rank[survivor] += 1

        return survivor, dying

    def get_birth(self, x: int) -> float:
        """Get birth time of the component containing x."""
        return self._birth[self.find(x)]


def compute_persistent_coverage_homology(
    coverage_grid: list[CoveragePoint],
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    significance_threshold: float = 1.0,
) -> PersistentCoverageHomology:
    """Compute persistent homology of the coverage function.

    Builds a sublevel set filtration F_k = {points with coverage <= k}
    and tracks topological features as k increases.

    H_0 (connected components): Tracked via Union-Find with elder rule.
    This computation is robust to triangulation choice.

    H_1 (holes/loops): Tracked via boundary matrix reduction on the
    grid's simplicial complex (edges and triangles from 4-connected grid).
    The H_1 computation uses lower-left triangulation of grid squares
    (each square split into two triangles along the lower-left to
    upper-right diagonal). H_1 results are sensitive to this
    triangulation choice; different diagonal splits can yield different
    H_1 persistence intervals.

    Args:
        coverage_grid: List of CoveragePoint from compute_coverage_snapshot.
        lat_step_deg: Latitude grid spacing.
        lon_step_deg: Longitude grid spacing.
        significance_threshold: Minimum persistence to count as significant.

    Returns:
        PersistentCoverageHomology with intervals, Betti curves, etc.
    """
    if not coverage_grid:
        return PersistentCoverageHomology(
            intervals=(),
            num_significant_holes=0,
            max_hole_persistence=0.0,
            betti_0_curve=(),
            betti_1_curve=(),
            total_persistence=0.0,
        )

    # Build 2D grid
    lats = sorted(set(cp.lat_deg for cp in coverage_grid))
    lons = sorted(set(cp.lon_deg for cp in coverage_grid))
    n_lat = len(lats)
    n_lon = len(lons)

    if n_lat < 2 or n_lon < 2:
        return PersistentCoverageHomology(
            intervals=(),
            num_significant_holes=0,
            max_hole_persistence=0.0,
            betti_0_curve=(),
            betti_1_curve=(),
            total_persistence=0.0,
        )

    lat_idx = {lat: i for i, lat in enumerate(lats)}
    lon_idx = {lon: j for j, lon in enumerate(lons)}

    # Build coverage array
    grid = np.zeros((n_lat, n_lon))
    for cp in coverage_grid:
        i = lat_idx[cp.lat_deg]
        j = lon_idx[cp.lon_deg]
        grid[i, j] = cp.visible_count

    # Flatten grid to 1D index: idx = i * n_lon + j
    def to_idx(i: int, j: int) -> int:
        return i * n_lon + j

    # Coverage levels (unique sorted)
    all_values = sorted(set(float(grid[i, j]) for i in range(n_lat) for j in range(n_lon)))
    max_level = max(all_values) if all_values else 0.0

    # ── H_0: Union-Find with elder rule ──
    # Process vertices in order of coverage value (sublevel filtration)
    # Sort all grid points by coverage value
    vertices = []
    for i in range(n_lat):
        for j in range(n_lon):
            vertices.append((float(grid[i, j]), i, j))
    vertices.sort(key=lambda x: x[0])

    uf = _UnionFind()
    active = np.zeros((n_lat, n_lon), dtype=bool)
    h0_intervals: list[PersistenceInterval] = []

    for val, i, j in vertices:
        idx = to_idx(i, j)
        uf.make_set(idx, val)
        active[i, j] = True

        # Check 4-connected neighbors already in filtration
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n_lat and 0 <= nj < n_lon and active[ni, nj]:
                n_idx = to_idx(ni, nj)
                result = uf.union(idx, n_idx)
                if result is not None:
                    _, dying_root = result
                    birth = uf._birth[dying_root]
                    death = val
                    if death > birth:
                        h0_intervals.append(PersistenceInterval(
                            birth=birth, death=death, dimension=0,
                        ))

    # Surviving H_0 components (never die)
    roots = set()
    for i in range(n_lat):
        for j in range(n_lon):
            roots.add(uf.find(to_idx(i, j)))
    for root in roots:
        h0_intervals.append(PersistenceInterval(
            birth=uf._birth[root], death=float('inf'), dimension=0,
        ))

    # ── H_1: Boundary matrix reduction ──
    # Build simplicial complex from grid:
    # 0-simplices: vertices (grid points)
    # 1-simplices: edges (4-connectivity)
    # 2-simplices: triangles (each square cell -> 2 triangles)
    #
    # Filtration value of a simplex = max of its vertex values

    # Edges: horizontal and vertical
    edges: list[tuple[float, int, int]] = []  # (filtration_val, v0, v1)
    for i in range(n_lat):
        for j in range(n_lon):
            idx0 = to_idx(i, j)
            val0 = float(grid[i, j])
            # Right neighbor
            if j + 1 < n_lon:
                idx1 = to_idx(i, j + 1)
                val1 = float(grid[i, j + 1])
                edges.append((max(val0, val1), idx0, idx1))
            # Down neighbor
            if i + 1 < n_lat:
                idx1 = to_idx(i + 1, j)
                val1 = float(grid[i + 1, j])
                edges.append((max(val0, val1), idx0, idx1))

    # Triangles: each grid cell (i,j)-(i,j+1)-(i+1,j)-(i+1,j+1) -> 2 triangles
    triangles: list[tuple[float, int, int, int]] = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            v00 = to_idx(i, j)
            v01 = to_idx(i, j + 1)
            v10 = to_idx(i + 1, j)
            v11 = to_idx(i + 1, j + 1)
            vals = [float(grid[i, j]), float(grid[i, j + 1]),
                    float(grid[i + 1, j]), float(grid[i + 1, j + 1])]
            # Lower-left triangle: v00, v10, v01
            triangles.append((max(vals[0], vals[2], vals[1]), v00, v10, v01))
            # Upper-right triangle: v01, v10, v11
            triangles.append((max(vals[1], vals[2], vals[3]), v01, v10, v11))

    # Sort edges and triangles by filtration value
    edges.sort(key=lambda x: x[0])
    triangles.sort(key=lambda x: x[0])

    # Build boundary matrix for edges -> vertices and triangles -> edges
    # For H_1, we need to reduce the boundary matrix of 2-simplices (triangles)
    # with respect to 1-simplices (edges).
    #
    # An edge is an H_1 birth if it creates a cycle (already connected vertices).
    # A triangle kills a cycle if its boundary matches an unpaired edge cycle.
    #
    # Simplified approach: use Union-Find to detect cycle-creating edges,
    # then pair them with triangles using boundary matrix column reduction.

    edge_uf = _UnionFind()
    edge_map: dict[tuple[int, int], int] = {}  # (v0, v1) -> edge_index
    cycle_edges: list[tuple[float, int]] = []  # (filtration_val, edge_index)

    for e_idx, (filt_val, v0, v1) in enumerate(edges):
        key = (min(v0, v1), max(v0, v1))
        edge_map[key] = e_idx

        if v0 not in edge_uf._parent:
            edge_uf.make_set(v0, 0.0)
        if v1 not in edge_uf._parent:
            edge_uf.make_set(v1, 0.0)

        r0 = edge_uf.find(v0)
        r1 = edge_uf.find(v1)
        if r0 == r1:
            # This edge creates a cycle -> H_1 birth
            cycle_edges.append((filt_val, e_idx))
        else:
            edge_uf.union(v0, v1)

    # For each triangle, compute its boundary (set of 3 edges)
    # Column reduction to pair triangles with cycle-creating edges
    h1_intervals: list[PersistenceInterval] = []
    paired_cycles: set[int] = set()

    # Build boundary columns for triangles
    # Each triangle has boundary = set of its 3 edges
    tri_boundaries: list[tuple[float, set[int]]] = []
    for filt_val, v0, v1, v2 in triangles:
        e0_key = (min(v0, v1), max(v0, v1))
        e1_key = (min(v1, v2), max(v1, v2))
        e2_key = (min(v0, v2), max(v0, v2))
        boundary = set()
        for key in [e0_key, e1_key, e2_key]:
            if key in edge_map:
                boundary.add(edge_map[key])
        if boundary:
            tri_boundaries.append((filt_val, boundary))

    # Column reduction (left-to-right) over Z_2
    # Pivot = lowest (highest-index) element in each column
    reduced_columns: list[tuple[float, set[int]]] = []
    pivot_to_col: dict[int, int] = {}

    for col_idx, (filt_val, boundary) in enumerate(tri_boundaries):
        col = set(boundary)  # copy
        while col:
            pivot = max(col)
            if pivot in pivot_to_col:
                # XOR (symmetric difference) with the column that has this pivot
                other_col = reduced_columns[pivot_to_col[pivot]][1]
                col = col.symmetric_difference(other_col)
            else:
                break

        if col:
            pivot = max(col)
            pivot_to_col[pivot] = col_idx
            reduced_columns.append((filt_val, col))

            # This triangle pairs with the cycle-creating edge at `pivot`
            # Find the birth time of that cycle
            for cycle_val, cycle_e_idx in cycle_edges:
                if cycle_e_idx == pivot and cycle_e_idx not in paired_cycles:
                    paired_cycles.add(cycle_e_idx)
                    if filt_val > cycle_val:
                        h1_intervals.append(PersistenceInterval(
                            birth=cycle_val, death=filt_val, dimension=1,
                        ))
                    break
        else:
            reduced_columns.append((filt_val, set()))

    # Unpaired cycle edges -> infinite H_1 features
    for cycle_val, cycle_e_idx in cycle_edges:
        if cycle_e_idx not in paired_cycles:
            h1_intervals.append(PersistenceInterval(
                birth=cycle_val, death=float('inf'), dimension=1,
            ))

    # ── Combine intervals ──
    all_intervals = h0_intervals + h1_intervals

    # ── Betti curves ──
    # Compute beta_0(k) and beta_1(k) for each coverage level k
    int_levels = sorted(set(int(v) for v in all_values))
    if not int_levels:
        int_levels = [0]

    betti_0: list[int] = []
    betti_1: list[int] = []

    for k in int_levels:
        level = float(k)
        b0 = sum(1 for iv in h0_intervals
                 if iv.birth <= level and (iv.death > level or iv.death == float('inf')))
        b1 = sum(1 for iv in h1_intervals
                 if iv.birth <= level and (iv.death > level or iv.death == float('inf')))
        betti_0.append(b0)
        betti_1.append(b1)

    # ── Summary statistics ──
    finite_h1 = [iv for iv in h1_intervals if iv.death != float('inf')]
    max_hole_persistence = 0.0
    if finite_h1:
        max_hole_persistence = max(iv.death - iv.birth for iv in finite_h1)

    num_significant = sum(
        1 for iv in h1_intervals
        if (iv.death - iv.birth if iv.death != float('inf') else float('inf'))
        > significance_threshold
    )

    total_persistence = sum(
        iv.death - iv.birth
        for iv in all_intervals
        if iv.death != float('inf')
    )

    return PersistentCoverageHomology(
        intervals=tuple(all_intervals),
        num_significant_holes=num_significant,
        max_hole_persistence=max_hole_persistence,
        betti_0_curve=tuple(betti_0),
        betti_1_curve=tuple(betti_1),
        total_persistence=total_persistence,
    )


# ── Wavelet Multi-Resolution Coverage Analysis (P37) ─────────────

@dataclass(frozen=True)
class WaveletCoverageAnalysis:
    """Haar wavelet decomposition of a 1D coverage time series.

    The signal is decomposed into approximation (lowpass) and detail
    (highpass) coefficients at multiple resolution levels using the
    discrete wavelet transform (DWT) with Haar wavelets.

    Energy at each scale identifies dominant temporal patterns:
    - Large-scale approximation energy = trend-dominated coverage.
    - Large-scale detail energy = transient/periodic variations.

    References:
        Mallat (2009). A Wavelet Tour of Signal Processing.
        Daubechies (1992). Ten Lectures on Wavelets.
    """
    approximation_energy: float         # Energy in final approximation
    detail_energies: tuple[float, ...]  # Energy at each detail level (finest first)
    dominant_scale: int                 # Scale index with highest detail energy
    dominant_period_s: float            # Physical period of dominant scale
    trend_coefficient: float            # Final approximation coefficient (DC level)
    transient_amplitude: float          # Max absolute detail coefficient
    is_trend_dominated: bool            # True if approximation energy > sum of detail energies


def compute_wavelet_coverage(
    coverage_values: list[float],
    sample_interval_s: float = 60.0,
    max_levels: int | None = None,
) -> WaveletCoverageAnalysis:
    """Compute Haar wavelet multi-resolution analysis of coverage time series.

    Decomposes the coverage signal into approximation (lowpass) and detail
    (highpass) components at progressively coarser scales.

    Haar wavelet filters:
        h = [1/sqrt(2), 1/sqrt(2)]   (lowpass / scaling)
        g = [1/sqrt(2), -1/sqrt(2)]  (highpass / wavelet)

    At each level, the signal is convolved with h and g, then downsampled
    by factor 2 to produce approximation and detail coefficients.

    Args:
        coverage_values: 1D time series of coverage counts or metrics.
        sample_interval_s: Time between samples in seconds.
        max_levels: Maximum decomposition levels (None = auto from signal length).

    Returns:
        WaveletCoverageAnalysis with energy distribution across scales.
    """
    if not coverage_values:
        return WaveletCoverageAnalysis(
            approximation_energy=0.0,
            detail_energies=(),
            dominant_scale=0,
            dominant_period_s=0.0,
            trend_coefficient=0.0,
            transient_amplitude=0.0,
            is_trend_dominated=True,
        )

    # Pad to next power of 2
    n = len(coverage_values)
    n_padded = 1
    while n_padded < n:
        n_padded *= 2

    signal = np.zeros(n_padded)
    signal[:n] = coverage_values

    # Haar wavelet coefficients
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    # Determine number of decomposition levels
    if max_levels is None:
        num_levels = int(np.log2(n_padded))
    else:
        num_levels = min(max_levels, int(np.log2(n_padded)))

    if num_levels < 1:
        approx_energy = float(np.sum(signal ** 2)) / n_padded
        trend_coeff = float(signal[0]) if n_padded > 0 else 0.0
        return WaveletCoverageAnalysis(
            approximation_energy=approx_energy,
            detail_energies=(),
            dominant_scale=0,
            dominant_period_s=0.0,
            trend_coefficient=trend_coeff,
            transient_amplitude=0.0,
            is_trend_dominated=True,
        )

    # Iterative DWT decomposition
    approx = signal.copy()
    detail_coeffs_list: list[np.ndarray] = []

    for level in range(num_levels):
        length = len(approx)
        if length < 2:
            break

        half = length // 2
        new_approx = np.zeros(half)
        new_detail = np.zeros(half)

        for k in range(half):
            new_approx[k] = inv_sqrt2 * (approx[2 * k] + approx[2 * k + 1])
            new_detail[k] = inv_sqrt2 * (approx[2 * k] - approx[2 * k + 1])

        detail_coeffs_list.append(new_detail)
        approx = new_approx

    # Energy at each scale: E_j = sum |d_j,k|^2 / N
    detail_energies: list[float] = []
    for d in detail_coeffs_list:
        energy = float(np.sum(d ** 2)) / n_padded
        detail_energies.append(energy)

    approx_energy = float(np.sum(approx ** 2)) / n_padded

    # Dominant scale: argmax of detail energies
    if detail_energies:
        dominant_scale = int(np.argmax(detail_energies))
    else:
        dominant_scale = 0

    # Dominant period: at scale j, period = 2^(j+1) * sample_interval
    dominant_period_s = (2.0 ** (dominant_scale + 1)) * sample_interval_s

    # Trend coefficient: final approximation (DC level)
    trend_coefficient = float(approx[0]) if len(approx) > 0 else 0.0

    # Transient amplitude: max absolute detail coefficient
    all_detail = np.concatenate(detail_coeffs_list) if detail_coeffs_list else np.array([0.0])
    transient_amplitude = float(np.max(np.abs(all_detail)))

    # Is trend dominated?
    total_detail_energy = sum(detail_energies)
    is_trend_dominated = approx_energy > total_detail_energy

    return WaveletCoverageAnalysis(
        approximation_energy=approx_energy,
        detail_energies=tuple(detail_energies),
        dominant_scale=dominant_scale,
        dominant_period_s=dominant_period_s,
        trend_coefficient=trend_coefficient,
        transient_amplitude=transient_amplitude,
        is_trend_dominated=is_trend_dominated,
    )
