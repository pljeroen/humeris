# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Conjunction screening, TCA refinement, B-plane decomposition,
and collision probability assessment.

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

    # Altitude-band pre-filter: skip pairs whose semi-major axes differ
    # by more than the distance threshold (they can never be close enough).
    sma = np.array([s.semi_major_axis_m for s in states])
    candidate_pairs: list[tuple[int, int]] = []
    for i in range(n_sats):
        for j in range(i + 1, n_sats):
            if abs(sma[i] - sma[j]) <= distance_threshold_m:
                candidate_pairs.append((i, j))

    if not candidate_pairs:
        return results

    t_elapsed = 0.0
    while t_elapsed <= duration_seconds:
        t = start + timedelta(seconds=t_elapsed)

        # Propagate all states to current time
        positions: list[list[float]] = [[] for _ in range(n_sats)]
        propagated: set[int] = set()

        for i, j in candidate_pairs:
            for idx in (i, j):
                if idx not in propagated:
                    pos, _ = propagate_to(states[idx], t)
                    positions[idx] = pos
                    propagated.add(idx)

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

    # Altitude-band pre-filter using initial positions
    initial_radii = np.array([
        float(np.linalg.norm(r.steps[0].position_eci)) for r in results
    ])
    candidate_pairs: list[tuple[int, int]] = []
    for i in range(n_sats):
        for j in range(i + 1, n_sats):
            if abs(initial_radii[i] - initial_radii[j]) <= distance_threshold_m:
                candidate_pairs.append((i, j))

    if not candidate_pairs:
        return []

    candidates: list[tuple[int, int, datetime, float]] = []
    n_steps = len(results[0].steps)

    for step_idx in range(n_steps):
        positions = [r.steps[step_idx].position_eci for r in results]
        t = results[0].steps[step_idx].time

        for i, j in candidate_pairs:
            dp = np.array(positions[i]) - np.array(positions[j])
            dist = float(np.linalg.norm(dp))
            if dist <= distance_threshold_m:
                candidates.append((i, j, t, dist))

    candidates.sort(key=lambda x: x[3])
    return candidates


# ── FTLE for Conjunction Risk Classification ───────────────────────

@dataclass(frozen=True)
class ConjunctionPredictability:
    """Finite-Time Lyapunov Exponent for conjunction risk assessment."""
    ftle: float                        # Finite-time Lyapunov exponent (1/s)
    max_singular_value: float          # Largest singular value of relative STM
    is_chaotic: bool                   # FTLE > threshold
    margin_multiplier: float           # Suggested safety margin multiplier (1.0 = nominal)
    predictability_horizon_s: float    # Time horizon for reliable prediction


def compute_conjunction_ftle(
    state1: OrbitalState,
    state2: OrbitalState,
    tca: datetime,
    window_s: float = 1800.0,
    perturbation_m: float = 1.0,
    chaos_threshold: float = 1e-4,
) -> ConjunctionPredictability:
    """Compute finite-time Lyapunov exponent for conjunction risk assessment.

    Estimates dynamical sensitivity of the relative state at TCA by
    numerical differentiation of the position flow map.

    Steps:
    1. Propagate both objects to TCA (nominal relative state).
    2. Perturb position of object 1 in each of 3 directions.
    3. Propagate nominal and perturbed states for window_s.
    4. Build 3x3 position sensitivity matrix from finite differences.
    5. SVD to get maximum singular value.
    6. FTLE = ln(sigma_max) / window_s.

    Args:
        state1: First satellite orbital state.
        state2: Second satellite orbital state.
        tca: Time of closest approach.
        window_s: Propagation window for FTLE computation (seconds).
        perturbation_m: Position perturbation magnitude (meters).
        chaos_threshold: FTLE threshold for chaotic classification (1/s).

    Returns:
        ConjunctionPredictability with FTLE, singular value, and margins.
    """
    from humeris.domain.orbital_mechanics import kepler_to_cartesian, OrbitalConstants

    t_end = tca + timedelta(seconds=window_s)

    # Nominal propagation to end of window
    pos1_end, _ = propagate_to(state1, t_end)
    pos2_end, _ = propagate_to(state2, t_end)
    dr_nominal = np.array(pos1_end) - np.array(pos2_end)

    # Build 3x3 position sensitivity matrix via finite differences.
    # Under two-body Keplerian dynamics, the FTLE depends only on the
    # individual orbit shape (semi-major axis, eccentricity). Encounter
    # geometry (crossing vs co-planar) is captured separately via relative
    # velocity in the margin_multiplier.
    phi = np.zeros((3, 3))

    pos1_tca, vel1_tca = propagate_to(state1, tca)
    pos2_tca, vel2_tca = propagate_to(state2, tca)

    for axis in range(3):
        pos1_pert = list(pos1_tca)
        pos1_pert[axis] += perturbation_m
        perturbed_state1 = _cartesian_to_orbital_state(
            pos1_pert, vel1_tca, tca,
        )
        pos1p_end, _ = propagate_to(perturbed_state1, t_end)
        dr_perturbed = np.array(pos1p_end) - np.array(pos2_end)
        phi[:, axis] = (dr_perturbed - dr_nominal) / perturbation_m

    # SVD of the sensitivity matrix
    _, singular_values, _ = np.linalg.svd(phi)
    sigma_max = float(singular_values[0])

    # Ensure sigma_max >= 1 (identity map has sigma = 1)
    sigma_max = max(sigma_max, 1.0)

    # FTLE
    ftle = math.log(sigma_max) / window_s

    # Ensure non-negative (log(1) = 0 is the minimum)
    ftle = max(ftle, 0.0)

    is_chaotic = ftle > chaos_threshold

    # Encounter geometry factor: relative velocity at TCA determines
    # conjunction duration and miss-distance sensitivity. Crossing orbits
    # have higher relative velocity → shorter conjunction → less predictable.
    rel_vel = np.linalg.norm(np.array(vel1_tca) - np.array(vel2_tca))
    v_circ = math.sqrt(OrbitalConstants.MU_EARTH / state1.semi_major_axis_m)
    geometry_factor = 1.0 + rel_vel / v_circ

    margin_multiplier = max(1.0, sigma_max / 10.0) * geometry_factor
    predictability_horizon_s = 1.0 / max(ftle, 1e-12)

    return ConjunctionPredictability(
        ftle=ftle,
        max_singular_value=sigma_max,
        is_chaotic=is_chaotic,
        margin_multiplier=margin_multiplier,
        predictability_horizon_s=predictability_horizon_s,
    )


def _cartesian_to_orbital_state(
    position: list[float],
    velocity: list[float],
    epoch: datetime,
) -> OrbitalState:
    """Convert Cartesian ECI state to OrbitalState for two-body propagation.

    Uses standard Keplerian element recovery from position/velocity vectors.
    Assumes two-body dynamics (no J2 corrections).

    Args:
        position: ECI position [x, y, z] in meters.
        velocity: ECI velocity [vx, vy, vz] in m/s.
        epoch: Reference epoch for the state.

    Returns:
        OrbitalState suitable for propagate_to.
    """
    from humeris.domain.orbital_mechanics import OrbitalConstants

    mu = OrbitalConstants.MU_EARTH

    r_vec = np.array(position)
    v_vec = np.array(velocity)
    r_mag = float(np.linalg.norm(r_vec))
    v_mag = float(np.linalg.norm(v_vec))

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h_mag = float(np.linalg.norm(h_vec))

    # Node vector
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n_mag = float(np.linalg.norm(n_vec))

    # Eccentricity vector
    e_vec = ((v_mag ** 2 - mu / r_mag) * r_vec
             - float(np.dot(r_vec, v_vec)) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))

    # Semi-major axis (vis-viva)
    energy = v_mag ** 2 / 2.0 - mu / r_mag
    if abs(energy) < 1e-20:
        # Parabolic — use r_mag as approximation
        a = r_mag
    else:
        a = -mu / (2.0 * energy)

    # Inclination
    i_rad = math.acos(max(-1.0, min(1.0, h_vec[2] / h_mag))) if h_mag > 0 else 0.0

    # RAAN
    if n_mag > 1e-10:
        raan_rad = math.acos(max(-1.0, min(1.0, n_vec[0] / n_mag)))
        if n_vec[1] < 0:
            raan_rad = 2.0 * math.pi - raan_rad
    else:
        raan_rad = 0.0

    # Argument of perigee
    if n_mag > 1e-10 and e > 1e-10:
        arg_perigee_rad = math.acos(
            max(-1.0, min(1.0, float(np.dot(n_vec, e_vec)) / (n_mag * e)))
        )
        if e_vec[2] < 0:
            arg_perigee_rad = 2.0 * math.pi - arg_perigee_rad
    else:
        arg_perigee_rad = 0.0

    # True anomaly
    if e > 1e-10:
        cos_nu = max(-1.0, min(1.0, float(np.dot(e_vec, r_vec)) / (e * r_mag)))
        nu_rad = math.acos(cos_nu)
        if float(np.dot(r_vec, v_vec)) < 0:
            nu_rad = 2.0 * math.pi - nu_rad
    else:
        # Circular orbit: use argument of latitude
        if n_mag > 1e-10:
            cos_u = max(-1.0, min(1.0, float(np.dot(n_vec, r_vec)) / (n_mag * r_mag)))
            nu_rad = math.acos(cos_u)
            if r_vec[2] < 0:
                nu_rad = 2.0 * math.pi - nu_rad
            nu_rad = nu_rad - arg_perigee_rad
        else:
            # Equatorial circular: use longitude
            nu_rad = math.atan2(r_vec[1], r_vec[0])

    # Mean motion
    n = math.sqrt(mu / abs(a) ** 3) if a > 0 else math.sqrt(mu / r_mag ** 3)

    return OrbitalState(
        semi_major_axis_m=a,
        eccentricity=e,
        inclination_rad=i_rad,
        raan_rad=raan_rad,
        arg_perigee_rad=arg_perigee_rad,
        true_anomaly_rad=nu_rad,
        mean_motion_rad_s=n,
        reference_epoch=epoch,
    )


# ── Contact Geometry for Conjunction Assessment (P6) ───────────────

@dataclass(frozen=True)
class ContactConjunctionMetric:
    """Heuristic encounter-geometry metric for conjunction assessment.

    Constructs a contact form from the relative state at encounter
    and computes a heuristic encounter-geometry metric combining
    along-track separation, B-plane miss distance, and relative
    velocity. This is NOT a true topological invariant (contact
    volume), but a practical scalar that captures encounter severity
    in a frame-independent way.

    In the B-plane frame:
        alpha = dz - p_x * dx - p_y * dy
    where z = along-track, (x, y) = B-plane coordinates,
    (p_x, p_y) = conjugate momenta (velocity components scaled by 1/v_rel).

    The Reeb vector field R (satisfying alpha(R)=1, d(alpha)(R,.)=0)
    gives the direction of closest approach.

    References:
        Arnold (1989). Mathematical Methods of Classical Mechanics.
        Geiges (2008). An Introduction to Contact Topology.
    """
    contact_metric: float          # Heuristic encounter-geometry metric
    reeb_direction: tuple[float, float, float]  # Direction of closest approach (unit)
    legendrian_angle_rad: float    # Angle of B-plane in contact structure
    miss_distance_m: float         # Miss distance at TCA
    relative_velocity_ms: float    # Relative velocity magnitude
    is_transverse: bool            # True if encounter is transverse to contact planes


def compute_contact_conjunction(
    pos1: list[float],
    vel1: list[float],
    pos2: list[float],
    vel2: list[float],
    tca: datetime | None = None,
) -> ContactConjunctionMetric:
    """Compute heuristic encounter-geometry metric for conjunction assessment.

    Constructs a contact form from the relative state at encounter and
    computes a heuristic encounter-geometry metric. This is frame-independent
    (same value regardless of coordinate system) but is NOT a true
    topological invariant -- it is a practical scalar combining along-track
    separation, B-plane miss distance, and relative velocity.

    The contact form on the relative state space is:
        alpha = p . dq - H dt
    In the encounter frame, this becomes:
        alpha = (v_rel . dr_perp) / |v_rel|

    The metric is: |alpha_value| * b_mag * dv_mag, combining the
    along-track miss, B-plane distance, and relative velocity.

    Args:
        pos1: Position of satellite 1 (m), ECI or any inertial frame.
        vel1: Velocity of satellite 1 (m/s).
        pos2: Position of satellite 2 (m).
        vel2: Velocity of satellite 2 (m/s).
        tca: Time of closest approach (optional, for metadata).

    Returns:
        ContactConjunctionMetric with heuristic encounter-geometry metric.
    """
    r1 = np.array(pos1, dtype=np.float64)
    r2 = np.array(pos2, dtype=np.float64)
    v1 = np.array(vel1, dtype=np.float64)
    v2 = np.array(vel2, dtype=np.float64)

    # Relative state
    dr = r1 - r2   # relative position (miss vector)
    dv = v1 - v2   # relative velocity

    dv_mag = float(np.linalg.norm(dv))
    dr_mag = float(np.linalg.norm(dr))

    if dv_mag < 1e-10:
        # No relative motion: degenerate encounter
        return ContactConjunctionMetric(
            contact_metric=0.0,
            reeb_direction=(0.0, 0.0, 1.0),
            legendrian_angle_rad=0.0,
            miss_distance_m=dr_mag,
            relative_velocity_ms=0.0,
            is_transverse=False,
        )

    # Unit vector along relative velocity (encounter direction)
    s_hat = dv / dv_mag

    # B-plane: project miss vector onto plane perpendicular to s_hat
    dr_along = float(np.dot(dr, s_hat))
    b_vec = dr - dr_along * s_hat  # B-plane miss vector
    b_mag = float(np.linalg.norm(b_vec))

    # Build encounter frame: s_hat, t_hat, w_hat
    # t_hat: component of r1 direction in B-plane (radial-ish)
    r1_mag = float(np.linalg.norm(r1))
    if r1_mag > 1e-10:
        r_hat = r1 / r1_mag
        r_proj = r_hat - float(np.dot(r_hat, s_hat)) * s_hat
        r_proj_mag = float(np.linalg.norm(r_proj))
        if r_proj_mag > 1e-10:
            t_hat = r_proj / r_proj_mag
        else:
            # s_hat aligned with position: pick arbitrary perpendicular
            perp = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(s_hat, perp))) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            t_hat = np.cross(s_hat, perp)
            t_hat = t_hat / float(np.linalg.norm(t_hat))
    else:
        t_hat = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(s_hat, t_hat))) > 0.9:
            t_hat = np.array([0.0, 1.0, 0.0])
        t_hat = t_hat - float(np.dot(t_hat, s_hat)) * s_hat
        t_hat = t_hat / float(np.linalg.norm(t_hat))

    w_hat = np.cross(s_hat, t_hat)

    # B-plane components
    b_t = float(np.dot(b_vec, t_hat))  # radial component
    b_w = float(np.dot(b_vec, w_hat))  # cross-track component

    # Conjugate momenta in B-plane (velocity components perpendicular to s_hat)
    dv_perp = dv - float(np.dot(dv, s_hat)) * s_hat  # should be ~0 at TCA
    p_t = float(np.dot(dv_perp, t_hat)) / dv_mag
    p_w = float(np.dot(dv_perp, w_hat)) / dv_mag

    # Contact form: alpha = dz - p_t * dx - p_w * dy
    # At the encounter point, alpha evaluated on the relative state:
    #   alpha(dr, dv) = dr_along - p_t * b_t - p_w * b_w
    alpha_value = dr_along - p_t * b_t - p_w * b_w

    # Contact volume: |alpha ^ (d alpha)^2|
    # For the standard contact form alpha = dz - p1 dx1 - p2 dx2,
    # d(alpha) = dp1 ^ dx1 + dp2 ^ dx2 (symplectic on the B-plane).
    # alpha ^ d(alpha)^2 = alpha ^ dp1^dx1^dp2^dx2
    # The magnitude is |alpha_value| * |d(alpha)^2|
    # |d(alpha)^2| is the area element of the B-plane times velocity space
    #
    # In practice, the invariant contact volume combines:
    # Heuristic encounter-geometry metric:
    # |alpha| * |b_mag| * |v_rel| = |along-track miss| * B-plane distance * relative velocity.
    # Not a true topological invariant, but a practical frame-independent scalar
    # that captures encounter severity.
    contact_metric = abs(alpha_value) * b_mag * dv_mag

    # Reeb vector field: direction satisfying alpha(R)=1 and d(alpha)(R,.)=0
    # For alpha = dz - p_t dx - p_w dy, the Reeb vector is R = d/dz = s_hat
    reeb = tuple(float(x) for x in s_hat)

    # Legendrian angle: angle of the B-plane miss vector
    legendrian_angle = math.atan2(b_w, b_t) if b_mag > 1e-10 else 0.0

    # Transversality: encounter is transverse if the miss vector has
    # significant B-plane component (not purely along-track)
    is_transverse = b_mag > 0.1 * abs(dr_along) if abs(dr_along) > 1e-10 else b_mag > 1e-10

    return ContactConjunctionMetric(
        contact_metric=contact_metric,
        reeb_direction=reeb,
        legendrian_angle_rad=legendrian_angle,
        miss_distance_m=dr_mag,
        relative_velocity_ms=dv_mag,
        is_transverse=is_transverse,
    )


# ── P15: Hamilton-Jacobi Reachability for Collision Avoidance ──────


@dataclass(frozen=True)
class AvoidanceReachability:
    """Result of Hamilton-Jacobi reachability analysis for collision avoidance.

    Determines whether a collision is avoidable given the available delta-V
    budget and time to TCA, using the Clohessy-Wiltshire relative dynamics.

    The backward reachable tube is computed: starting from the collision set
    C = {||x_pos|| <= r_combined}, we propagate backward in time under CW
    dynamics with bounded control ||u|| <= u_max. If the current relative
    state lies outside the backward reachable tube, the collision is avoidable.

    Attributes:
        is_avoidable: True if the miss distance can be increased above
            the combined radius using available dV.
        optimal_avoidance_direction: Unit vector (vx, vy, vz) for the
            optimal impulse in the LVLH frame.
        min_avoidance_dv_ms: Minimum delta-V to achieve safe miss distance.
        achievable_miss_distance_m: Maximum miss distance achievable with
            the full available dV budget.
        safety_margin: Ratio of achievable_miss_distance to combined_radius.
            > 1 means avoidable, < 1 means unavoidable with available dV.
    """
    is_avoidable: bool
    optimal_avoidance_direction: tuple[float, float, float]
    min_avoidance_dv_ms: float
    achievable_miss_distance_m: float
    safety_margin: float


def compute_avoidance_reachability(
    rel_x_m: float,
    rel_y_m: float,
    rel_z_m: float,
    rel_vx_ms: float,
    rel_vy_ms: float,
    rel_vz_ms: float,
    n_rad_s: float,
    combined_radius_m: float,
    max_dv_ms: float,
    time_to_tca_s: float,
) -> AvoidanceReachability:
    """Determine collision avoidability via HJ reachability on CW dynamics.

    Uses the closed-form Clohessy-Wiltshire state transition matrix to
    propagate the relative state forward to TCA, then evaluates the
    reachable set boundary analytically.

    The CW equations give: x(TCA) = Phi_rr * x0 + Phi_rv * v0
    An impulsive maneuver dv applied now adds: Phi_rv * dv to the position
    at TCA. The question is: can we find ||dv|| <= max_dv such that
    ||x(TCA) + Phi_rv * dv|| > combined_radius?

    This is equivalent to: find dv that maximises ||x_nom + Phi_rv * dv||,
    subject to ||dv|| <= max_dv. The optimal dv is along the direction
    Phi_rv^T * x_nom / ||Phi_rv^T * x_nom|| (gradient ascent on distance).

    The minimum dV to escape the collision sphere is found by solving:
    ||x_nom + Phi_rv * dv|| = combined_radius for the minimum ||dv||.

    Args:
        rel_x_m: Relative radial position (m) in LVLH.
        rel_y_m: Relative along-track position (m) in LVLH.
        rel_z_m: Relative cross-track position (m) in LVLH.
        rel_vx_ms: Relative radial velocity (m/s).
        rel_vy_ms: Relative along-track velocity (m/s).
        rel_vz_ms: Relative cross-track velocity (m/s).
        n_rad_s: Chief orbit mean motion (rad/s).
        combined_radius_m: Combined hard-body radius (m).
        max_dv_ms: Available delta-V budget (m/s).
        time_to_tca_s: Time until closest approach (seconds, positive).

    Returns:
        AvoidanceReachability with avoidability assessment and optimal direction.

    Raises:
        ValueError: If n_rad_s <= 0, combined_radius_m <= 0,
            max_dv_ms < 0, or time_to_tca_s <= 0.
    """
    if n_rad_s <= 0:
        raise ValueError(f"n_rad_s must be positive, got {n_rad_s}")
    if combined_radius_m <= 0:
        raise ValueError(
            f"combined_radius_m must be positive, got {combined_radius_m}"
        )
    if max_dv_ms < 0:
        raise ValueError(f"max_dv_ms must be non-negative, got {max_dv_ms}")
    if time_to_tca_s <= 0:
        raise ValueError(f"time_to_tca_s must be positive, got {time_to_tca_s}")

    n = n_rad_s
    t = time_to_tca_s
    nt = n * t
    cos_nt = math.cos(nt)
    sin_nt = math.sin(nt)

    # CW state transition matrix (in-plane: x, y; cross-track: z)
    # Position part: x(t) = Phi_rr * r0 + Phi_rv * v0

    # Phi_rv (3x3): maps velocity impulse at t=0 to position at t=TCA
    phi_rv = np.array([
        [sin_nt / n, 2.0 * (1.0 - cos_nt) / n, 0.0],
        [-2.0 * (1.0 - cos_nt) / n, (4.0 * sin_nt / n - 3.0 * t), 0.0],
        [0.0, 0.0, sin_nt / n],
    ])

    # Phi_rr: maps initial position to position at TCA
    phi_rr = np.array([
        [4.0 - 3.0 * cos_nt, 0.0, 0.0],
        [6.0 * (sin_nt - nt), 1.0, 0.0],
        [0.0, 0.0, cos_nt],
    ])

    r0 = np.array([rel_x_m, rel_y_m, rel_z_m])
    v0 = np.array([rel_vx_ms, rel_vy_ms, rel_vz_ms])

    # Nominal position at TCA (no maneuver)
    x_nom = phi_rr @ r0 + phi_rv @ v0
    nom_dist = float(np.linalg.norm(x_nom))

    # The effect of an impulse dv at t=0 on position at TCA is: Phi_rv @ dv
    # To maximise miss distance: dv should be along Phi_rv^T @ x_nom
    # (gradient of ||x_nom + Phi_rv @ dv||^2 w.r.t. dv)
    gradient = phi_rv.T @ x_nom
    grad_norm = float(np.linalg.norm(gradient))

    if grad_norm < 1e-15:
        # Degenerate case: no preferred direction
        # Try all principal directions and pick the best
        best_dist = nom_dist
        best_dir = np.array([1.0, 0.0, 0.0])
        for axis in range(3):
            for sign in [1.0, -1.0]:
                dv_test = np.zeros(3)
                dv_test[axis] = sign * max_dv_ms
                test_pos = x_nom + phi_rv @ dv_test
                test_dist = float(np.linalg.norm(test_pos))
                if test_dist > best_dist:
                    best_dist = test_dist
                    best_dir = np.zeros(3)
                    best_dir[axis] = sign
        opt_dir = best_dir
        achievable_dist = best_dist
    else:
        opt_dir = gradient / grad_norm

        # Achievable miss distance with full dV budget
        dv_optimal = max_dv_ms * opt_dir
        x_maneuver = x_nom + phi_rv @ dv_optimal
        achievable_dist = float(np.linalg.norm(x_maneuver))

    # Minimum dV to reach combined_radius (escape collision sphere)
    # Solve: ||x_nom + alpha * Phi_rv @ opt_dir||^2 = combined_radius^2
    phi_rv_d = phi_rv @ opt_dir
    a_coeff = float(np.dot(phi_rv_d, phi_rv_d))
    b_coeff = 2.0 * float(np.dot(x_nom, phi_rv_d))
    c_coeff = float(np.dot(x_nom, x_nom)) - combined_radius_m ** 2

    if nom_dist >= combined_radius_m:
        # Already safe — no maneuver needed
        min_dv = 0.0
    elif a_coeff < 1e-30:
        # Phi_rv @ opt_dir ~ 0: maneuver has no effect
        min_dv = float('inf')
    else:
        discriminant = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
        if discriminant < 0:
            # No real solution: cannot escape with this direction
            min_dv = float('inf')
        else:
            sqrt_disc = math.sqrt(discriminant)
            alpha1 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff)
            alpha2 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff)
            # Pick the smallest positive alpha
            candidates = [x for x in [alpha1, alpha2] if x > 0]
            if candidates:
                min_dv = min(candidates)
            else:
                min_dv = 0.0

    is_avoidable = achievable_dist > combined_radius_m
    safety = achievable_dist / combined_radius_m if combined_radius_m > 0 else float('inf')

    return AvoidanceReachability(
        is_avoidable=is_avoidable,
        optimal_avoidance_direction=(
            float(opt_dir[0]), float(opt_dir[1]), float(opt_dir[2]),
        ),
        min_avoidance_dv_ms=min_dv,
        achievable_miss_distance_m=achievable_dist,
        safety_margin=safety,
    )


# ── P21: Bayesian Conjunction Probability via Beta-Binomial ────────


@dataclass(frozen=True)
class BayesianCollisionProbability:
    """Bayesian collision probability estimate via Beta-Binomial model.

    Models the collision probability Pc as a Beta(alpha, beta) posterior,
    where the prior strength (kappa) is derived from the observation
    quality (number of observations and covariance condition number).

    Attributes:
        posterior_mean: E[Pc] = alpha / (alpha + beta).
        posterior_std: Std[Pc] from Beta distribution.
        credible_interval_95: (lower, upper) 95% credible interval.
        exceedance_probability: P(Pc > threshold) via regularized
            incomplete beta function.
        point_estimate: Original point estimate Pc used as prior center.
    """
    posterior_mean: float
    posterior_std: float
    credible_interval_95: tuple[float, float]
    exceedance_probability: float
    point_estimate: float


def _log_gamma(x: float) -> float:
    """Log-gamma function via Stirling's approximation with Lanczos correction.

    Accurate to ~15 digits for x > 0.5. For x < 0.5 uses reflection formula.
    """
    if x < 0.5:
        # Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        return math.log(math.pi / math.sin(math.pi * x)) - _log_gamma(1.0 - x)

    x -= 1.0
    # Lanczos coefficients (g=7, n=9)
    coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    g = 7.0
    t = x + g + 0.5
    s = coeffs[0]
    for i in range(1, len(coeffs)):
        s += coeffs[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(s)


def _log_beta(a: float, b: float) -> float:
    """Log of the Beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)."""
    return _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction.

    Uses Lentz's method for the continued fraction representation.
    I_x(a, b) = x^a * (1-x)^b / (a * B(a,b)) * CF(a, b, x)

    The continued fraction coefficients:
        d_{2m+1} = -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))
        d_{2m}   = m(b-m)x / ((a+2m-1)(a+2m))

    Args:
        x: Evaluation point, 0 <= x <= 1.
        a: First shape parameter, a > 0.
        b: Second shape parameter, b > 0.

    Returns:
        I_x(a, b) in [0, 1].
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use symmetry relation when x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)

    # Prefactor: x^a * (1-x)^b / (a * B(a,b))
    log_prefix = a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b) - math.log(a)

    # Continued fraction via modified Lentz's method
    # CF = 1 / (1 + d1/(1 + d2/(1 + ...)))
    # We evaluate as a standard CF: f = a0 + a1/(b1 + a2/(b2 + ...))
    tiny = 1e-30
    max_iter = 200
    eps = 1e-14

    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # d_{2m}: even term
        numerator = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        f *= c * d

        # d_{2m+1}: odd term
        numerator = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < eps:
            break

    result = math.exp(log_prefix) * f
    return max(0.0, min(1.0, result))


def _beta_quantile(p: float, a: float, b: float) -> float:
    """Approximate quantile of Beta(a, b) distribution via bisection.

    Finds x such that I_x(a, b) = p.

    Args:
        p: Target probability (0 < p < 1).
        a: First shape parameter.
        b: Second shape parameter.

    Returns:
        x in [0, 1] such that I_x(a, b) ≈ p.
    """
    lo = 0.0
    hi = 1.0
    for _ in range(64):  # 64 bisection steps → ~1e-19 precision
        mid = (lo + hi) / 2.0
        if _regularized_incomplete_beta(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def bayesian_collision_probability(
    pc_point: float,
    n_obs: int,
    cov_condition_number: float,
    threshold: float = 1e-4,
) -> BayesianCollisionProbability:
    """Bayesian collision probability via Beta-Binomial model.

    Models the true collision probability Pc as a Beta(alpha, beta)
    posterior. The prior concentration kappa = n_obs^2 / cond(P) reflects
    observation quality: more observations and better-conditioned
    covariance yield a tighter posterior.

    The prior center is the point estimate: alpha = Pc * kappa,
    beta = (1 - Pc) * kappa. The posterior mean = alpha / (alpha + beta).

    Args:
        pc_point: Point estimate of collision probability (0 to 1).
        n_obs: Number of observations used in the conjunction assessment.
        cov_condition_number: Condition number of the combined covariance
            matrix. Higher values indicate more uncertain geometry.
        threshold: Collision probability threshold for exceedance
            computation P(Pc > threshold).

    Returns:
        BayesianCollisionProbability with posterior statistics.

    Raises:
        ValueError: If pc_point not in [0, 1], n_obs < 1, or
            cov_condition_number < 1.
    """
    if not (0.0 <= pc_point <= 1.0):
        raise ValueError(f"pc_point must be in [0, 1], got {pc_point}")
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")
    if cov_condition_number < 1.0:
        raise ValueError(
            f"cov_condition_number must be >= 1, got {cov_condition_number}"
        )

    # Prior concentration from observation quality
    kappa = (n_obs * n_obs) / cov_condition_number
    kappa = max(kappa, 2.0)  # minimum concentration for valid Beta

    # Clamp pc_point away from exact 0 and 1 for Beta distribution validity
    pc_clamped = max(1e-10, min(1.0 - 1e-10, pc_point))

    alpha = pc_clamped * kappa
    beta_param = (1.0 - pc_clamped) * kappa

    # Posterior statistics
    posterior_mean = alpha / (alpha + beta_param)
    var_num = alpha * beta_param
    var_den = (alpha + beta_param) ** 2 * (alpha + beta_param + 1.0)
    posterior_std = math.sqrt(var_num / var_den)

    # 95% credible interval via quantile function
    lower = _beta_quantile(0.025, alpha, beta_param)
    upper = _beta_quantile(0.975, alpha, beta_param)

    # Exceedance probability: P(Pc > threshold)
    threshold_clamped = max(0.0, min(1.0, threshold))
    exceedance = 1.0 - _regularized_incomplete_beta(
        threshold_clamped, alpha, beta_param,
    )

    return BayesianCollisionProbability(
        posterior_mean=posterior_mean,
        posterior_std=posterior_std,
        credible_interval_95=(lower, upper),
        exceedance_probability=exceedance,
        point_estimate=pc_point,
    )


# ── P51: Generating Function for Conjunction Counting ──────────────
#
# Given pairwise collision probabilities p_ij (upper triangular),
# compute the probability generating function G(z) = product((1-p) + p*z)
# and extract the full distribution P(K=k) of total conjunction count.


@dataclass(frozen=True)
class ConjunctionCountDistribution:
    """Distribution of total conjunction count from pairwise probabilities.

    Attributes:
        mean_conjunctions: E[K] = sum(p_ij).
        variance: Var[K] = sum(p_ij * (1 - p_ij)).
        skewness: Skewness of the distribution.
        prob_zero_conjunctions: P(K=0) = product(1 - p_ij).
        prob_at_most_k: Tuple of P(K <= k) for k = 0, 1, ..., max_k.
        max_k: Maximum k for which P(K <= k) is computed.
    """
    mean_conjunctions: float
    variance: float
    skewness: float
    prob_zero_conjunctions: float
    prob_at_most_k: tuple[float, ...]
    max_k: int


def compute_conjunction_count_distribution(
    pairwise_probabilities: list[float],
    max_k: int | None = None,
) -> ConjunctionCountDistribution:
    """Compute conjunction count distribution via probability generating function.

    Given independent pairwise conjunction probabilities p_1, p_2, ..., p_n
    (e.g., upper triangular entries of a conjunction probability matrix),
    the total count K = sum of independent Bernoulli(p_i).

    The PGF is: G(z) = product_i ((1-p_i) + p_i * z)

    P(K=k) = coefficient of z^k in G(z), obtained via polynomial
    multiplication.

    Args:
        pairwise_probabilities: List of pairwise conjunction probabilities
            (upper triangular entries). Each must be in [0, 1].
        max_k: Maximum k for CDF computation (default: len(probabilities)).

    Returns:
        ConjunctionCountDistribution with full distribution.

    Raises:
        ValueError: If any probability is outside [0, 1] or list is empty.
    """
    if not pairwise_probabilities:
        raise ValueError("pairwise_probabilities must not be empty")

    for i, p in enumerate(pairwise_probabilities):
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"Probability at index {i} must be in [0, 1], got {p}"
            )

    n = len(pairwise_probabilities)
    if max_k is None:
        max_k = n

    # Compute PGF coefficients via polynomial multiplication
    # Start with [1.0] (constant polynomial = 1)
    # Multiply by ((1-p_i) + p_i * z) for each i
    coeffs = np.array([1.0])  # polynomial coefficients: coeffs[k] = coeff of z^k

    for p in pairwise_probabilities:
        # Multiply current polynomial by ((1-p) + p*z)
        # New polynomial has one more term
        new_coeffs = np.zeros(len(coeffs) + 1)
        new_coeffs[:len(coeffs)] += (1.0 - p) * coeffs
        new_coeffs[1:len(coeffs) + 1] += p * coeffs
        coeffs = new_coeffs

    # P(K=k) = coeffs[k] for k = 0, 1, ..., n
    # Truncate to max_k
    actual_max_k = min(max_k, len(coeffs) - 1)

    # Mean: E[K] = sum(p_i)
    mean = sum(pairwise_probabilities)

    # Variance: Var[K] = sum(p_i * (1 - p_i))
    variance = sum(p * (1.0 - p) for p in pairwise_probabilities)

    # Skewness: sum(p_i * (1-p_i) * (1-2*p_i)) / variance^(3/2)
    if variance > 1e-30:
        third_moment = sum(
            p * (1.0 - p) * (1.0 - 2.0 * p) for p in pairwise_probabilities
        )
        skewness = third_moment / (variance ** 1.5)
    else:
        skewness = 0.0

    # P(K=0) = product(1 - p_i) = coeffs[0]
    prob_zero = float(coeffs[0])

    # CDF: P(K <= k) = sum_{j=0}^{k} coeffs[j]
    cdf = np.cumsum(coeffs[:actual_max_k + 1])
    # Clamp to [0, 1] for numerical safety
    cdf = np.clip(cdf, 0.0, 1.0)

    return ConjunctionCountDistribution(
        mean_conjunctions=mean,
        variance=variance,
        skewness=skewness,
        prob_zero_conjunctions=prob_zero,
        prob_at_most_k=tuple(float(x) for x in cdf),
        max_k=actual_max_k,
    )
