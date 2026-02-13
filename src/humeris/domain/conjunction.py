# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Conjunction screening, TCA refinement, B-plane decomposition,
and collision probability assessment.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.numerical_propagation import NumericalPropagationResult


@dataclass(frozen=True)
class PositionCovariance:
    """3x3 symmetric position covariance matrix (upper triangle, meters^2)."""
    sigma_xx: float
    sigma_yy: float
    sigma_zz: float
    sigma_xy: float
    sigma_xz: float
    sigma_yz: float


@dataclass(frozen=True)
class ConjunctionEvent:
    """Result of a conjunction assessment."""
    sat1_name: str
    sat2_name: str
    tca: datetime
    miss_distance_m: float
    relative_velocity_ms: float
    collision_probability: float
    max_collision_probability: float
    b_plane_radial_m: float
    b_plane_cross_track_m: float


def _distance_at_time(
    state1: OrbitalState,
    state2: OrbitalState,
    t: datetime,
) -> tuple[float, list[float], list[float], list[float], list[float]]:
    """Compute distance between two states at time t.

    Returns (distance, pos1, vel1, pos2, vel2).
    """
    p1, v1 = propagate_to(state1, t)
    p2, v2 = propagate_to(state2, t)
    dp = np.array(p1) - np.array(p2)
    dist = float(np.linalg.norm(dp))
    return dist, p1, v1, p2, v2


def screen_conjunctions(
    states: list[OrbitalState],
    names: list[str],
    start: datetime,
    duration: timedelta,
    step: timedelta,
    distance_threshold_m: float = 50_000.0,
) -> list[tuple[int, int, datetime, float]]:
    """Screen for conjunction candidates via pairwise distance checks.

    Propagates all states at each timestep, checks pairwise distances.

    Args:
        states: List of OrbitalState objects.
        names: Corresponding satellite names (same length as states).
        start: Start time for screening window.
        duration: Duration of screening window.
        step: Time step between checks.
        distance_threshold_m: Distance threshold for flagging (meters).

    Returns:
        List of (i, j, time, distance) tuples, sorted by distance.

    Raises:
        ValueError: If len(states) != len(names) or step <= 0.
    """
    if len(states) != len(names):
        raise ValueError(
            f"states length ({len(states)}) != names length ({len(names)})"
        )
    step_seconds = step.total_seconds()
    if step_seconds <= 0:
        raise ValueError(f"step must be positive, got {step_seconds}s")

    duration_seconds = duration.total_seconds()
    n_sats = len(states)
    results: list[tuple[int, int, datetime, float]] = []

    t_elapsed = 0.0
    while t_elapsed <= duration_seconds:
        t = start + timedelta(seconds=t_elapsed)

        # Propagate all states to current time
        positions = []
        for s in states:
            pos, _ = propagate_to(s, t)
            positions.append(pos)

        # Pairwise distance check
        for i in range(n_sats):
            for j in range(i + 1, n_sats):
                dp = np.array(positions[i]) - np.array(positions[j])
                dist = float(np.linalg.norm(dp))
                if dist <= distance_threshold_m:
                    results.append((i, j, t, dist))

        t_elapsed += step_seconds

    results.sort(key=lambda x: x[3])
    return results


def refine_tca(
    state1: OrbitalState,
    state2: OrbitalState,
    t_guess: datetime,
    search_window_s: float = 300.0,
    tolerance_seconds: float = 0.1,
) -> tuple[datetime, float, float]:
    """Refine time of closest approach using golden-section search.

    Minimizes ||pos1(t) - pos2(t)|| within search window around guess.

    Args:
        state1: First satellite state.
        state2: Second satellite state.
        t_guess: Initial guess for TCA.
        search_window_s: Half-width of search window (seconds).
        tolerance_seconds: Convergence tolerance (seconds).

    Returns:
        (tca, miss_distance_m, relative_velocity_ms)
    """
    golden_ratio = (math.sqrt(5) - 1) / 2

    a_s = -search_window_s
    b_s = search_window_s

    c_s = b_s - golden_ratio * (b_s - a_s)
    d_s = a_s + golden_ratio * (b_s - a_s)

    while abs(b_s - a_s) > tolerance_seconds:
        t_c = t_guess + timedelta(seconds=c_s)
        t_d = t_guess + timedelta(seconds=d_s)

        fc, _, _, _, _ = _distance_at_time(state1, state2, t_c)
        fd, _, _, _, _ = _distance_at_time(state1, state2, t_d)

        if fc < fd:
            b_s = d_s
        else:
            a_s = c_s

        c_s = b_s - golden_ratio * (b_s - a_s)
        d_s = a_s + golden_ratio * (b_s - a_s)

    best_s = (a_s + b_s) / 2.0
    tca = t_guess + timedelta(seconds=best_s)
    dist, p1, v1, p2, v2 = _distance_at_time(state1, state2, tca)

    dv = np.array(v1) - np.array(v2)
    rel_vel = float(np.linalg.norm(dv))

    return tca, dist, rel_vel


def compute_b_plane(
    pos1: list[float],
    vel1: list[float],
    pos2: list[float],
    vel2: list[float],
) -> tuple[float, float]:
    """Compute B-plane decomposition of miss vector.

    Projects miss vector onto plane perpendicular to relative velocity.

    Args:
        pos1, vel1: Position/velocity of satellite 1 (m, m/s).
        pos2, vel2: Position/velocity of satellite 2 (m, m/s).

    Returns:
        (b_radial_m, b_cross_track_m) — B-plane components.
    """
    # Relative velocity direction (encounter frame)
    dv = np.array(vel1) - np.array(vel2)
    dv_mag = float(np.linalg.norm(dv))

    if dv_mag < 1e-10:
        return 0.0, 0.0

    # Unit vector along relative velocity
    s_hat = dv / dv_mag

    # Miss vector (from sat2 to sat1)
    m_vec = np.array(pos1) - np.array(pos2)

    # Project miss vector onto B-plane (perpendicular to s_hat)
    m_dot_s = float(np.dot(m_vec, s_hat))
    b_vec = m_vec - m_dot_s * s_hat

    # Define radial direction: component of position in B-plane
    # Use orbit normal approximation: z-component dominant for radial
    p1 = np.array(pos1)
    r1_mag = float(np.linalg.norm(p1))
    if r1_mag < 1e-10:
        return float(np.linalg.norm(b_vec)), 0.0

    r_hat = p1 / r1_mag

    # Radial component in B-plane
    r_proj_s = float(np.dot(r_hat, s_hat))
    r_b = r_hat - r_proj_s * s_hat
    r_b_mag = float(np.linalg.norm(r_b))

    if r_b_mag < 1e-10:
        return float(np.linalg.norm(b_vec)), 0.0

    r_b_hat = r_b / r_b_mag

    # Cross-track direction: s_hat x r_b_hat
    ct_hat = np.cross(s_hat, r_b_hat)

    b_radial = float(np.dot(b_vec, r_b_hat))
    b_cross = float(np.dot(b_vec, ct_hat))

    return b_radial, b_cross


def foster_max_collision_probability(
    miss_distance_m: float,
    combined_radius_m: float,
    combined_covariance_trace_m2: float,
) -> float:
    """Foster conservative upper bound on collision probability.

    Pc_max = r^2 / (2 * e * sigma^2)

    where r = combined_radius, sigma^2 = combined_covariance_trace.
    Ref: NASA conjunction assessment.

    Args:
        miss_distance_m: Miss distance (meters) — unused in max bound formula
            but kept for API consistency.
        combined_radius_m: Combined hard-body radius (meters).
        combined_covariance_trace_m2: Trace of combined position covariance (m^2).

    Returns:
        Maximum collision probability (dimensionless). 0 if covariance <= 0.
    """
    if combined_covariance_trace_m2 <= 0:
        return 0.0
    return combined_radius_m ** 2 / (2.0 * math.e * combined_covariance_trace_m2)


def collision_probability_2d(
    miss_distance_m: float,
    b_radial_m: float,
    b_cross_m: float,
    sigma_radial_m: float,
    sigma_cross_m: float,
    combined_radius_m: float,
    num_steps: int = 100,
) -> float:
    """2D collision probability via numerical integration in B-plane.

    Integrates bivariate normal distribution over hard-body disk.

    Args:
        miss_distance_m: Total miss distance (meters).
        b_radial_m: B-plane radial component (meters).
        b_cross_m: B-plane cross-track component (meters).
        sigma_radial_m: Radial position uncertainty (meters).
        sigma_cross_m: Cross-track position uncertainty (meters).
        combined_radius_m: Combined hard-body radius (meters).
        num_steps: Grid resolution per axis for integration.

    Returns:
        Collision probability (dimensionless, 0-1).
    """
    if sigma_radial_m <= 0 or sigma_cross_m <= 0:
        return 0.0

    r = combined_radius_m
    dx = 2.0 * r / num_steps
    dy = 2.0 * r / num_steps

    probability = 0.0
    for i in range(num_steps):
        x = -r + (i + 0.5) * dx
        for j in range(num_steps):
            y = -r + (j + 0.5) * dy

            # Check if point is within hard-body disk
            if x * x + y * y > r * r:
                continue

            # Bivariate normal centered at miss vector in B-plane
            zx = (x - b_radial_m) / sigma_radial_m
            zy = (y - b_cross_m) / sigma_cross_m
            pdf = (1.0 / (2.0 * math.pi * sigma_radial_m * sigma_cross_m)) * math.exp(
                -0.5 * (zx * zx + zy * zy)
            )
            probability += pdf * dx * dy

    return min(probability, 1.0)


def assess_conjunction(
    state1: OrbitalState,
    name1: str,
    state2: OrbitalState,
    name2: str,
    t_guess: datetime,
    combined_radius_m: float = 10.0,
    cov1: PositionCovariance | None = None,
    cov2: PositionCovariance | None = None,
) -> ConjunctionEvent:
    """Full conjunction assessment: TCA refinement, B-plane, collision probability.

    Args:
        state1: First satellite state.
        name1: First satellite name.
        state2: Second satellite state.
        name2: Second satellite name.
        t_guess: Approximate conjunction time.
        combined_radius_m: Combined hard-body radius (meters).
        cov1: Position covariance for satellite 1 (optional).
        cov2: Position covariance for satellite 2 (optional).

    Returns:
        ConjunctionEvent with full assessment results.
    """
    tca, miss_dist, rel_vel = refine_tca(state1, state2, t_guess)

    # Propagate to TCA for B-plane
    p1, v1 = propagate_to(state1, tca)
    p2, v2 = propagate_to(state2, tca)

    b_radial, b_cross = compute_b_plane(p1, v1, p2, v2)

    pc = 0.0
    pc_max = 0.0

    if cov1 is not None and cov2 is not None:
        # Combined covariance trace (sum of diagonal elements)
        combined_trace = (
            cov1.sigma_xx + cov1.sigma_yy + cov1.sigma_zz
            + cov2.sigma_xx + cov2.sigma_yy + cov2.sigma_zz
        )

        pc_max = foster_max_collision_probability(
            miss_dist, combined_radius_m, combined_trace,
        )

        # Approximate radial/cross-track sigmas from combined covariance
        sigma_combined = math.sqrt(combined_trace / 3.0)
        pc = collision_probability_2d(
            miss_dist, b_radial, b_cross,
            sigma_combined, sigma_combined,
            combined_radius_m,
        )

    return ConjunctionEvent(
        sat1_name=name1,
        sat2_name=name2,
        tca=tca,
        miss_distance_m=miss_dist,
        relative_velocity_ms=rel_vel,
        collision_probability=pc,
        max_collision_probability=pc_max,
        b_plane_radial_m=b_radial,
        b_plane_cross_track_m=b_cross,
    )


def screen_conjunctions_numerical(
    results: list[NumericalPropagationResult],
    names: list[str],
    distance_threshold_m: float = 50_000.0,
) -> list[tuple[int, int, datetime, float]]:
    """Screen conjunctions from pre-computed numerical propagation results.

    Assumes all results have the same time steps. Checks pairwise distances
    at each time step.

    Args:
        results: List of NumericalPropagationResult (one per satellite).
        names: Corresponding satellite names (same length as results).
        distance_threshold_m: Distance threshold for flagging (meters).

    Returns:
        List of (i, j, time, distance) tuples, sorted by distance ascending.

    Raises:
        ValueError: If len(results) != len(names), or results is empty.
    """
    if len(results) != len(names):
        raise ValueError(
            f"results length ({len(results)}) != names length ({len(names)})"
        )
    if len(results) == 0:
        raise ValueError("results must not be empty")

    n_sats = len(results)
    if n_sats < 2:
        return []

    candidates: list[tuple[int, int, datetime, float]] = []
    n_steps = len(results[0].steps)

    for step_idx in range(n_steps):
        positions = [r.steps[step_idx].position_eci for r in results]
        t = results[0].steps[step_idx].time

        for i in range(n_sats):
            for j in range(i + 1, n_sats):
                dp = np.array(positions[i]) - np.array(positions[j])
                dist = float(np.linalg.norm(dp))
                if dist <= distance_threshold_m:
                    candidates.append((i, j, t, dist))

    candidates.sort(key=lambda x: x[3])
    return candidates
