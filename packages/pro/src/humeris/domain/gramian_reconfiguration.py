# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Gramian-guided constellation reconfiguration (G-RECON).

Uses CW controllability Gramian eigenstructure to find minimum-fuel
reconfiguration maneuvers. The Gramian reveals which directions in
relative state space are cheapest to reach, enabling optimal maneuver
planning that exploits orbital dynamics instead of fighting them.

"""
import math
from dataclasses import dataclass

import numpy as np

from humeris.domain.control_analysis import (
    ControllabilityAnalysis,
    compute_cw_controllability,
)
from humeris.domain.station_keeping import propellant_mass_for_dv


@dataclass(frozen=True)
class ReconfigurationTarget:
    """Desired relative state change for one satellite."""
    satellite_index: int
    delta_state: tuple  # (dx, dy, dz, dvx, dvy, dvz) desired change in LVLH


@dataclass(frozen=True)
class ReconfigurationManeuver:
    """Optimal maneuver for one satellite."""
    satellite_index: int
    delta_v: tuple  # (dvx, dvy, dvz) in LVLH frame (m/s)
    delta_v_magnitude: float  # ||delta_v|| in m/s
    fuel_cost_index: float  # Relative cost (1.0 = average, <1 = cheap, >1 = expensive)
    gramian_alignment: float  # Cosine similarity with max-eigenvalue direction
    propellant_mass_kg: float  # If Isp provided


@dataclass(frozen=True)
class ReconfigurationPlan:
    """Complete reconfiguration plan for constellation."""
    maneuvers: tuple  # Tuple of ReconfigurationManeuver
    total_delta_v: float  # Sum of all ||dv||
    total_propellant_kg: float  # Sum of propellant
    max_single_dv: float  # Maximum single-satellite dV
    mean_gramian_alignment: float  # How well maneuvers align with cheap directions
    is_feasible: bool  # All maneuvers within capability
    efficiency_score: float  # 0-1, higher = better Gramian exploitation


def compute_gramian_optimal_dv(
    target_delta_state: tuple,
    n_rad_s: float,
    duration_s: float,
    step_s: float = 10.0,
) -> tuple:
    """Compute minimum-energy delta-V for a desired state change.

    Projects the target state change onto the Gramian eigenvectors and
    reconstructs the delta-V using inverse-eigenvalue weighting. Directions
    with large eigenvalues (high controllability) require less control effort.

    The Gramian W_c satisfies: minimum-energy input u that drives the state
    from 0 to x_f has cost J = x_f^T W_c^{-1} x_f. The eigenvectors of W_c
    diagonalise this quadratic form.

    For each eigenvector e_i with eigenvalue lambda_i, the component of
    control effort along that direction is proportional to the projection
    of the target onto e_i divided by lambda_i.

    The returned delta-V is the velocity part (components 3,4,5) of the
    optimal control vector in the 6D state space.

    Args:
        target_delta_state: Desired state change (dx, dy, dz, dvx, dvy, dvz).
        n_rad_s: Chief orbit mean motion (rad/s).
        duration_s: Maneuver window duration (s).
        step_s: Gramian integration step (s).

    Returns:
        Tuple (dvx, dvy, dvz) of optimal delta-V in LVLH frame (m/s).
    """
    target = np.array(target_delta_state, dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    if target_norm < 1e-15:
        return (0.0, 0.0, 0.0)

    analysis = compute_cw_controllability(n_rad_s, duration_s, step_s)
    eigenvalues = analysis.gramian_eigenvalues
    eigenvectors = analysis.gramian_eigenvectors

    # Reconstruct optimal control via Gramian inverse eigenbasis
    # u_opt = W_c^{-1} x_f, decomposed in eigenbasis
    optimal_control = np.zeros(6, dtype=np.float64)
    for i, (lam, evec) in enumerate(zip(eigenvalues, eigenvectors)):
        ev = np.array(evec, dtype=np.float64)
        projection = float(np.dot(target, ev))
        if abs(lam) > 1e-15:
            optimal_control += (projection / lam) * ev

    # Extract the velocity components (indices 3, 4, 5) as the delta-V
    dvx = float(optimal_control[3])
    dvy = float(optimal_control[4])
    dvz = float(optimal_control[5])
    return (dvx, dvy, dvz)


def compute_fuel_cost_index(
    delta_state: tuple,
    gramian_analysis: ControllabilityAnalysis,
) -> float:
    """Compute relative fuel cost for a maneuver direction.

    Projects the desired state change onto the Gramian eigenvectors and
    computes the quadratic cost x^T W_c^{-1} x relative to what the cost
    would be if all eigenvalues were equal to the mean eigenvalue.

    A cost index < 1 means the maneuver is cheaper than average (aligned
    with high-controllability directions). A cost index > 1 means it is
    more expensive (aligned with low-controllability directions).

    Args:
        delta_state: Desired state change (6-vector).
        gramian_analysis: Precomputed Gramian analysis.

    Returns:
        Fuel cost ratio vs average cost (1.0 = average).
    """
    state = np.array(delta_state, dtype=np.float64)
    state_norm_sq = float(np.dot(state, state))
    if state_norm_sq < 1e-30:
        return 1.0

    eigenvalues = gramian_analysis.gramian_eigenvalues
    eigenvectors = gramian_analysis.gramian_eigenvectors

    if not eigenvalues:
        return 1.0

    mean_eigenvalue = sum(eigenvalues) / len(eigenvalues)
    if mean_eigenvalue < 1e-30:
        return 1.0

    # Weighted cost: sum(component_i^2 / eigenvalue_i)
    weighted_cost = 0.0
    for lam, evec in zip(eigenvalues, eigenvectors):
        ev = np.array(evec, dtype=np.float64)
        component = float(np.dot(state, ev))
        if abs(lam) > 1e-15:
            weighted_cost += component ** 2 / lam

    # Reference cost: same state, but all eigenvalues = mean
    reference_cost = state_norm_sq / mean_eigenvalue

    if reference_cost < 1e-30:
        return 1.0

    return weighted_cost / reference_cost


def _gramian_alignment(
    delta_state: tuple,
    gramian_analysis: ControllabilityAnalysis,
) -> float:
    """Compute cosine similarity between state change and max-eigenvalue direction.

    Returns value in [-1, 1]. Values near +/-1 indicate the maneuver is
    aligned with the cheapest controllability direction.

    Args:
        delta_state: Desired state change (6-vector).
        gramian_analysis: Precomputed Gramian analysis.

    Returns:
        Cosine similarity with max-eigenvalue eigenvector.
    """
    state = np.array(delta_state, dtype=np.float64)
    state_norm = float(np.linalg.norm(state))
    if state_norm < 1e-15:
        return 0.0

    max_dir = np.array(gramian_analysis.max_energy_direction, dtype=np.float64)
    max_dir_norm = float(np.linalg.norm(max_dir))
    if max_dir_norm < 1e-15:
        return 0.0

    cosine = float(np.dot(state, max_dir)) / (state_norm * max_dir_norm)
    return max(-1.0, min(1.0, cosine))


def plan_reconfiguration(
    targets: list,
    n_rad_s: float,
    duration_s: float,
    isp_s: float = 0.0,
    dry_mass_kg: float = 0.0,
    max_dv_per_sat: float = float('inf'),
) -> ReconfigurationPlan:
    """Plan Gramian-optimal reconfiguration for multiple satellites.

    For each target, computes the minimum-energy delta-V using the CW
    Gramian eigenstructure, then assembles a constellation-level plan
    with feasibility checks and efficiency scoring.

    Args:
        targets: List of ReconfigurationTarget objects.
        n_rad_s: Chief orbit mean motion (rad/s).
        duration_s: Maneuver window duration (s).
        isp_s: Specific impulse (s). If 0, propellant not computed.
        dry_mass_kg: Satellite dry mass (kg). Required if isp_s > 0.
        max_dv_per_sat: Maximum delta-V per satellite (m/s).

    Returns:
        ReconfigurationPlan with maneuvers and aggregate metrics.
    """
    if not targets:
        return ReconfigurationPlan(
            maneuvers=(),
            total_delta_v=0.0,
            total_propellant_kg=0.0,
            max_single_dv=0.0,
            mean_gramian_alignment=0.0,
            is_feasible=True,
            efficiency_score=1.0,
        )

    # Compute Gramian once (same orbit for all sats)
    analysis = compute_cw_controllability(n_rad_s, duration_s, step_s=10.0)

    maneuvers = []
    is_feasible = True
    cost_indices = []

    for target in targets:
        dv = compute_gramian_optimal_dv(
            target.delta_state, n_rad_s, duration_s,
        )
        dv_mag = float(np.linalg.norm(dv))

        cost_index = compute_fuel_cost_index(target.delta_state, analysis)
        alignment = _gramian_alignment(target.delta_state, analysis)

        prop_mass = 0.0
        if isp_s > 0 and dry_mass_kg > 0 and dv_mag > 0:
            prop_mass = propellant_mass_for_dv(isp_s, dry_mass_kg, dv_mag)

        if dv_mag > max_dv_per_sat:
            is_feasible = False

        maneuver = ReconfigurationManeuver(
            satellite_index=target.satellite_index,
            delta_v=dv,
            delta_v_magnitude=dv_mag,
            fuel_cost_index=cost_index,
            gramian_alignment=alignment,
            propellant_mass_kg=prop_mass,
        )
        maneuvers.append(maneuver)
        cost_indices.append(cost_index)

    total_dv = sum(m.delta_v_magnitude for m in maneuvers)
    total_prop = sum(m.propellant_mass_kg for m in maneuvers)
    max_dv = max(m.delta_v_magnitude for m in maneuvers)
    mean_alignment = sum(abs(m.gramian_alignment) for m in maneuvers) / len(maneuvers)

    # Efficiency score: how much cheaper than average the maneuvers are.
    # A plan where all cost indices = 1.0 gets score 0.5.
    # All cost indices → 0 gives score → 1.0.
    # All cost indices → inf gives score → 0.0.
    # Formula: score = 1 / (1 + mean_cost_index)
    if cost_indices:
        mean_cost = sum(cost_indices) / len(cost_indices)
        efficiency = 1.0 / (1.0 + mean_cost)
    else:
        efficiency = 1.0

    return ReconfigurationPlan(
        maneuvers=tuple(maneuvers),
        total_delta_v=total_dv,
        total_propellant_kg=total_prop,
        max_single_dv=max_dv,
        mean_gramian_alignment=mean_alignment,
        is_feasible=is_feasible,
        efficiency_score=efficiency,
    )


def find_cheapest_reconfig_path(
    current_states: list,
    desired_states: list,
    n_rad_s: float,
    duration_s: float,
) -> tuple:
    """Find optimal assignment of satellites to target slots.

    Uses a greedy Hungarian-style matching based on Gramian-weighted cost.
    For each possible (satellite, target) pair, computes the fuel cost index
    of the required state change, then greedily assigns lowest-cost pairs.

    Args:
        current_states: List of current relative states as 6-tuples.
        desired_states: List of desired relative states as 6-tuples.
        n_rad_s: Chief orbit mean motion (rad/s).
        duration_s: Maneuver window duration (s).

    Returns:
        Tuple of (satellite_idx, target_idx, cost) assignments.
    """
    n_sats = len(current_states)
    n_targets = len(desired_states)
    n_assign = min(n_sats, n_targets)

    if n_assign == 0:
        return ()

    analysis = compute_cw_controllability(n_rad_s, duration_s, step_s=10.0)

    # Build cost matrix
    cost_matrix = []
    for i in range(n_sats):
        row = []
        for j in range(n_targets):
            current = np.array(current_states[i], dtype=np.float64)
            desired = np.array(desired_states[j], dtype=np.float64)
            delta = tuple((desired - current).tolist())
            delta_norm = float(np.linalg.norm(desired - current))
            if delta_norm < 1e-15:
                row.append(0.0)
            else:
                cost_index = compute_fuel_cost_index(delta, analysis)
                # Total cost = dV magnitude * cost_index
                dv = compute_gramian_optimal_dv(delta, n_rad_s, duration_s)
                dv_mag = float(np.linalg.norm(dv))
                row.append(dv_mag * cost_index)
        cost_matrix.append(row)

    # Greedy assignment: pick lowest cost pair, remove both from pool
    assigned_sats = set()
    assigned_targets = set()
    assignments = []

    for _ in range(n_assign):
        best_cost = float('inf')
        best_i = -1
        best_j = -1
        for i in range(n_sats):
            if i in assigned_sats:
                continue
            for j in range(n_targets):
                if j in assigned_targets:
                    continue
                if cost_matrix[i][j] < best_cost:
                    best_cost = cost_matrix[i][j]
                    best_i = i
                    best_j = j
        if best_i >= 0:
            assignments.append((best_i, best_j, best_cost))
            assigned_sats.add(best_i)
            assigned_targets.add(best_j)

    return tuple(assignments)


def compute_reconfig_window(
    n_rad_s: float,
    durations: list,
) -> tuple:
    """Analyze Gramian condition number variation with maneuver duration.

    Helps identify optimal timing windows for reconfiguration. Lower
    condition numbers mean more isotropic controllability (all directions
    are similarly easy to reach), which is desirable for arbitrary
    reconfigurations.

    Args:
        n_rad_s: Chief orbit mean motion (rad/s).
        durations: List of candidate maneuver durations (s).

    Returns:
        Tuple of (duration_s, condition_number, gramian_trace) for each duration.
    """
    results = []
    for dur in durations:
        analysis = compute_cw_controllability(n_rad_s, dur, step_s=max(10.0, dur / 100.0))
        results.append((dur, analysis.condition_number, analysis.gramian_trace))
    return tuple(results)
