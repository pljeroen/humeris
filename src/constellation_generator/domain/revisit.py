# Copyright (c) 2026 Jeroen. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""
Time-domain revisit coverage analysis.

Computes standard coverage Figures of Merit (FoMs) over a time window:
mean/max revisit time, coverage fraction, mean response time, pass count.
Uses a brute-force time sweep with an optimized Earth Central Angle (ECA)
visibility check for performance.

Refs: Wertz SMAD Ch. 7, Vallado Ch. 11.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.propagation import OrbitalState, propagate_to
from constellation_generator.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    geodetic_to_ecef,
)


@dataclass(frozen=True)
class GridPoint:
    """Precomputed grid point with ECEF unit vector for fast visibility check."""
    lat_deg: float
    lon_deg: float
    ecef_x: float
    ecef_y: float
    ecef_z: float
    unit_x: float
    unit_y: float
    unit_z: float


@dataclass(frozen=True)
class PointRevisitResult:
    """Revisit metrics for a single grid point over the analysis window."""
    lat_deg: float
    lon_deg: float
    num_passes: int
    total_visible_s: float
    total_gap_s: float
    mean_gap_s: float
    max_gap_s: float
    mean_response_time_s: float
    coverage_fraction: float


@dataclass(frozen=True)
class CoverageResult:
    """Aggregate coverage metrics for a constellation over an analysis window."""
    analysis_duration_s: float
    num_grid_points: int
    num_satellites: int
    mean_coverage_fraction: float
    min_coverage_fraction: float
    mean_revisit_s: float
    max_revisit_s: float
    mean_response_time_s: float
    percent_coverage_single: float
    point_results: tuple[PointRevisitResult, ...]


def _earth_central_angle_limit(
    altitude_m: float,
    min_elevation_deg: float,
) -> float:
    """Compute cos(rho_max) for the fast ECA visibility check.

    rho_max = 90° - el_min - asin(R_E * cos(el_min) / (R_E + h))
    Returns cos(rho_max) — the threshold for dot(s_unit, g_unit).
    """
    r_e = OrbitalConstants.R_EARTH
    el_rad = math.radians(min_elevation_deg)
    rho_max = (
        math.radians(90.0)
        - el_rad
        - math.asin(r_e * math.cos(el_rad) / (r_e + altitude_m))
    )
    return math.cos(rho_max)


def _generate_grid(
    lat_step_deg: float,
    lon_step_deg: float,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> list[GridPoint]:
    """Generate grid points with precomputed ECEF unit vectors.

    Uses geodetic_to_ecef for exact WGS84 position, then normalizes.
    Grid covers lat_range inclusive, lon_range inclusive of min but
    exclusive of max (wraps at 360°).
    """
    grid: list[GridPoint] = []
    lat = lat_range[0]
    while lat <= lat_range[1] + 1e-9:
        lon = lon_range[0]
        while lon <= lon_range[1] - lon_step_deg + 1e-9:
            ecef_x, ecef_y, ecef_z = geodetic_to_ecef(lat, lon, 0.0)
            mag = math.sqrt(ecef_x**2 + ecef_y**2 + ecef_z**2)
            grid.append(GridPoint(
                lat_deg=lat, lon_deg=lon,
                ecef_x=ecef_x, ecef_y=ecef_y, ecef_z=ecef_z,
                unit_x=ecef_x / mag, unit_y=ecef_y / mag, unit_z=ecef_z / mag,
            ))
            lon += lon_step_deg
        lat += lat_step_deg
    return grid


def compute_revisit(
    orbital_states: list[OrbitalState],
    start: datetime,
    duration: timedelta,
    step: timedelta,
    min_elevation_deg: float = 10.0,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
) -> CoverageResult:
    """Compute time-domain coverage metrics via brute-force time sweep.

    Algorithm:
    1. Precompute grid ECEF unit vectors and ECA cos threshold
    2. For each timestep:
       a. Compute GMST once (shared across all satellites)
       b. Propagate each satellite to ECEF (via propagate_to + eci_to_ecef)
       c. For each grid point: check visibility via dot product against all sats
    3. Track per-point state machine: gap start/end, pass count, visible time
    4. After sweep: compute per-point FoMs and aggregate metrics

    Complexity: O(N_sats × N_grid × N_timesteps) dot products.

    Refs: Wertz SMAD Ch. 7, Vallado Ch. 11.
    """
    grid = _generate_grid(lat_step_deg, lon_step_deg, lat_range, lon_range)
    n_grid = len(grid)
    n_sats = len(orbital_states)
    duration_s = duration.total_seconds()
    step_s = step.total_seconds() if step.total_seconds() > 0 else 1.0

    # Determine altitude from first satellite (assumes homogeneous constellation)
    if n_sats > 0:
        altitude_m = orbital_states[0].semi_major_axis_m - OrbitalConstants.R_EARTH
        cos_rho_threshold = _earth_central_angle_limit(altitude_m, min_elevation_deg)
        # For dot product check: sat_ecef · grid_unit >= cos_rho * |sat_ecef|
        # For circular orbits, |sat_ecef| = a = constant
        a_m = orbital_states[0].semi_major_axis_m
        dot_threshold = cos_rho_threshold * a_m
    else:
        cos_rho_threshold = 0.0
        a_m = 0.0
        dot_threshold = 0.0

    # Per-point tracking state
    was_visible = [False] * n_grid
    gap_start_s = [0.0] * n_grid
    total_visible_s = [0.0] * n_grid
    gaps: list[list[float]] = [[] for _ in range(n_grid)]
    num_passes = [0] * n_grid

    # Time sweep
    num_steps = max(1, int(duration_s / step_s) + 1) if duration_s > 0 else 1
    for step_idx in range(num_steps):
        t_offset_s = step_idx * step_s if duration_s > 0 else 0.0
        current_time = start + timedelta(seconds=t_offset_s)

        # Compute GMST once for this timestep
        gmst_angle = gmst_rad(current_time)

        # Propagate all satellites to ECEF
        sat_ecefs: list[tuple[float, float, float]] = []
        for state in orbital_states:
            pos_eci, vel_eci = propagate_to(state, current_time)
            pos_ecef, _ = eci_to_ecef(
                (pos_eci[0], pos_eci[1], pos_eci[2]),
                (vel_eci[0], vel_eci[1], vel_eci[2]),
                gmst_angle,
            )
            sat_ecefs.append(pos_ecef)

        # Check visibility for each grid point
        for g_idx in range(n_grid):
            gp = grid[g_idx]
            visible = False
            for sx, sy, sz in sat_ecefs:
                dot = sx * gp.unit_x + sy * gp.unit_y + sz * gp.unit_z
                if dot >= dot_threshold:
                    visible = True
                    break

            # State machine transitions
            if visible:
                total_visible_s[g_idx] += step_s if duration_s > 0 else 0.0
                if not was_visible[g_idx]:
                    # Gap → visible: record gap
                    if step_idx > 0 or was_visible[g_idx] is False:
                        gap_duration = t_offset_s - gap_start_s[g_idx]
                        if gap_duration > 0 or step_idx > 0:
                            gaps[g_idx].append(gap_duration)
                    num_passes[g_idx] += 1
            else:
                if was_visible[g_idx]:
                    # Visible → gap: start new gap
                    gap_start_s[g_idx] = t_offset_s

            was_visible[g_idx] = visible

    # Close trailing gaps
    for g_idx in range(n_grid):
        if not was_visible[g_idx] and duration_s > 0:
            gap_duration = duration_s - gap_start_s[g_idx]
            if gap_duration > 0:
                gaps[g_idx].append(gap_duration)

    # Compute per-point results
    point_results: list[PointRevisitResult] = []
    for g_idx in range(n_grid):
        gp = grid[g_idx]
        visible_time = total_visible_s[g_idx]
        gap_time = max(0.0, duration_s - visible_time)
        point_gaps = gaps[g_idx]

        if len(point_gaps) > 0:
            mean_gap = sum(point_gaps) / len(point_gaps)
            max_gap = max(point_gaps)
        else:
            mean_gap = 0.0
            max_gap = 0.0

        cov_frac = visible_time / duration_s if duration_s > 0 else (1.0 if num_passes[g_idx] > 0 else 0.0)

        point_results.append(PointRevisitResult(
            lat_deg=gp.lat_deg,
            lon_deg=gp.lon_deg,
            num_passes=num_passes[g_idx],
            total_visible_s=visible_time,
            total_gap_s=gap_time,
            mean_gap_s=mean_gap,
            max_gap_s=max_gap,
            mean_response_time_s=mean_gap / 2.0,
            coverage_fraction=cov_frac,
        ))

    # Aggregate metrics
    if n_grid > 0:
        fractions = [pr.coverage_fraction for pr in point_results]
        mean_cov = sum(fractions) / n_grid
        min_cov = min(fractions)
        mean_revisit = sum(pr.mean_gap_s for pr in point_results) / n_grid
        max_revisit = max(pr.max_gap_s for pr in point_results)
        mean_resp = sum(pr.mean_response_time_s for pr in point_results) / n_grid
        pct_single = 100.0 * sum(1 for pr in point_results if pr.coverage_fraction >= 1.0 - 1e-9) / n_grid
    else:
        mean_cov = 0.0
        min_cov = 0.0
        mean_revisit = 0.0
        max_revisit = 0.0
        mean_resp = 0.0
        pct_single = 0.0

    return CoverageResult(
        analysis_duration_s=duration_s,
        num_grid_points=n_grid,
        num_satellites=n_sats,
        mean_coverage_fraction=mean_cov,
        min_coverage_fraction=min_cov,
        mean_revisit_s=mean_revisit,
        max_revisit_s=max_revisit,
        mean_response_time_s=mean_resp,
        percent_coverage_single=pct_single,
        point_results=tuple(point_results),
    )


def compute_single_coverage_fraction(
    orbital_states: list[OrbitalState],
    analysis_time: datetime,
    min_elevation_deg: float = 10.0,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
) -> float:
    """Quick spatial coverage fraction at a single epoch.

    Uses the fast ECA check. Returns fraction of grid points
    with ≥1 satellite visible.
    """
    grid = _generate_grid(lat_step_deg, lon_step_deg, lat_range, lon_range)
    n_grid = len(grid)
    if n_grid == 0:
        return 0.0

    n_sats = len(orbital_states)
    if n_sats == 0:
        return 0.0

    altitude_m = orbital_states[0].semi_major_axis_m - OrbitalConstants.R_EARTH
    cos_rho_threshold = _earth_central_angle_limit(altitude_m, min_elevation_deg)
    a_m = orbital_states[0].semi_major_axis_m
    dot_threshold = cos_rho_threshold * a_m

    gmst_angle = gmst_rad(analysis_time)
    sat_ecefs: list[tuple[float, float, float]] = []
    for state in orbital_states:
        pos_eci, vel_eci = propagate_to(state, analysis_time)
        pos_ecef, _ = eci_to_ecef(
            (pos_eci[0], pos_eci[1], pos_eci[2]),
            (vel_eci[0], vel_eci[1], vel_eci[2]),
            gmst_angle,
        )
        sat_ecefs.append(pos_ecef)

    visible_count = 0
    for gp in grid:
        for sx, sy, sz in sat_ecefs:
            dot = sx * gp.unit_x + sy * gp.unit_y + sz * gp.unit_z
            if dot >= dot_threshold:
                visible_count += 1
                break

    return visible_count / n_grid
