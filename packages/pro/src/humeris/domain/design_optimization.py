# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Design optimization — DOP/Fisher information, coverage drift, mass efficiency frontier.

Reinterprets DOP geometry as Fisher Information Matrix, computes RAAN drift
sensitivity for coverage maintenance, and maps the Tsiolkovsky mass wall.

"""
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_ecef_to
from humeris.domain.atmosphere import DragConfig
from humeris.domain.dilution_of_precision import compute_dop
from humeris.domain.orbital_mechanics import OrbitalConstants, j2_raan_rate
from humeris.domain.station_keeping import drag_compensation_dv_per_year, propellant_mass_for_dv
from humeris.domain.maneuvers import hohmann_transfer
from humeris.domain.revisit import compute_single_coverage_fraction


@dataclass(frozen=True)
class PositioningInformationMetric:
    """DOP reinterpreted as Fisher Information."""
    gdop: float
    pdop: float
    fisher_determinant: float
    d_optimal_criterion: float
    crlb_position_m: float
    information_efficiency: float


@dataclass(frozen=True)
class CoverageDriftAnalysis:
    """Sensitivity of coverage to RAAN drift from altitude errors."""
    raan_sensitivity_rad_s_per_m: float
    coverage_drift_rate_per_s: float
    coverage_half_life_s: float
    maintenance_interval_s: float


@dataclass(frozen=True)
class MassEfficiencyPoint:
    """Single point on the mass efficiency frontier."""
    altitude_km: float
    total_dv_ms: float
    wet_mass_kg: float
    constellation_mass_kg: float
    mass_efficiency: float


@dataclass(frozen=True)
class MassEfficiencyFrontier:
    """Tsiolkovsky mass efficiency across altitude range."""
    points: tuple
    optimal_altitude_km: float
    peak_efficiency: float
    mass_wall_altitude_km: float


def compute_positioning_information(
    lat_deg: float,
    lon_deg: float,
    sat_positions_ecef: list,
    sigma_measurement_m: float = 1.0,
    min_elevation_deg: float = 10.0,
) -> PositioningInformationMetric:
    """Reinterpret DOP geometry as Fisher Information Matrix.

    FIM = H^T H, where H is the geometry matrix from DOP computation.
    D-optimality = det(FIM)^(1/N) where N=4 unknowns.
    CRLB_position = sigma_meas * PDOP.
    Information efficiency = 1 / GDOP^2.
    """
    dop = compute_dop(lat_deg, lon_deg, sat_positions_ecef, min_elevation_deg)

    # FIM = H^T H. Q = (H^T H)^-1. det(FIM) = 1/det(Q).
    # Compute det(Q) from individual DOP components rather than GDOP^4.
    # Q diagonal elements: HDOP^2/2 (x), HDOP^2/2 (y), VDOP^2 (z), TDOP^2 (t).
    # For the off-diagonal-free approximation:
    #   det(Q) ~= (HDOP^2/2)^2 * VDOP^2 * TDOP^2

    if dop.gdop > 0 and dop.gdop < float('inf'):
        # Approximate det(Q) from DOP components
        hdop_sq_half = dop.hdop ** 2 / 2.0  # split HDOP^2 across x,y
        vdop_sq = dop.vdop ** 2
        tdop_sq = dop.tdop ** 2
        det_q = hdop_sq_half * hdop_sq_half * vdop_sq * tdop_sq
        fisher_det = 1.0 / det_q if det_q > 1e-30 else 0.0
        d_optimal = fisher_det ** 0.25  # det^(1/4)
        crlb = sigma_measurement_m * dop.pdop
        gdop_sq = dop.gdop ** 2
        efficiency = 1.0 / gdop_sq
    else:
        fisher_det = 0.0
        d_optimal = 0.0
        crlb = float('inf')
        efficiency = 0.0

    return PositioningInformationMetric(
        gdop=dop.gdop,
        pdop=dop.pdop,
        fisher_determinant=fisher_det,
        d_optimal_criterion=d_optimal,
        crlb_position_m=crlb,
        information_efficiency=min(1.0, efficiency),
    )


def compute_coverage_drift(
    states: list,
    epoch: datetime,
    altitude_error_m: float = 100.0,
    coverage_threshold: float = 0.05,
    duration_s: float = 5400.0,
    step_s: float = 60.0,
) -> CoverageDriftAnalysis:
    """Compute coverage sensitivity to J2 RAAN drift from altitude errors.

    dOmega/dt sensitivity to semi-major axis:
    d(dOmega/dt)/da = -7/2 * (dOmega/dt)_nominal / a

    Note: the -7/2 sensitivity coefficient assumes circular orbits (e=0).
    For eccentric orbits, the full expression includes additional terms
    in e and the coefficient differs.

    Coverage half-life: time until coverage degrades by threshold.
    """
    if not states:
        return CoverageDriftAnalysis(
            raan_sensitivity_rad_s_per_m=0.0,
            coverage_drift_rate_per_s=0.0,
            coverage_half_life_s=float('inf'),
            maintenance_interval_s=float('inf'),
        )

    # Use first satellite as reference
    ref = states[0]
    a = ref.semi_major_axis_m
    n = ref.mean_motion_rad_s
    inc = ref.inclination_rad

    # J2 RAAN rate and its sensitivity to semi-major axis
    raan_rate = j2_raan_rate(n, a, 0.0, inc)
    # ∂(dΩ/dt)/∂a = -7/2 · (dΩ/dt) / a
    raan_sensitivity = -3.5 * raan_rate / a

    # Differential RAAN drift from altitude error
    d_raan_rate = raan_sensitivity * altitude_error_m

    # Coverage drift: approximate as linear degradation
    # Compute baseline coverage
    cov_baseline = compute_single_coverage_fraction(
        states, epoch, lat_step_deg=30.0, lon_step_deg=30.0,
    )

    if cov_baseline < 1e-10:
        return CoverageDriftAnalysis(
            raan_sensitivity_rad_s_per_m=raan_sensitivity,
            coverage_drift_rate_per_s=0.0,
            coverage_half_life_s=float('inf'),
            maintenance_interval_s=float('inf'),
        )

    # Coverage drift rate: proportional to differential RAAN drift
    # One radian of RAAN shift ≈ 1/N_planes of coverage cell width
    n_planes = max(1, len(set(s.raan_rad for s in states)))
    coverage_sensitivity = 1.0 / (2.0 * np.pi / n_planes)
    coverage_drift = abs(d_raan_rate) * coverage_sensitivity

    # Half-life: time until coverage degrades by threshold fraction
    if coverage_drift > 1e-20:
        half_life = coverage_threshold / coverage_drift
    else:
        half_life = float('inf')

    # Maintenance interval: half of half-life (for proactive maintenance)
    maintenance = half_life / 2.0 if half_life < float('inf') else float('inf')

    return CoverageDriftAnalysis(
        raan_sensitivity_rad_s_per_m=raan_sensitivity,
        coverage_drift_rate_per_s=coverage_drift,
        coverage_half_life_s=half_life,
        maintenance_interval_s=maintenance,
    )


def compute_mass_efficiency_frontier(
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    injection_altitude_km: float,
    mission_years: float,
    num_sats: int,
    alt_min_km: float = 300.0,
    alt_max_km: float = 800.0,
    alt_step_km: float = 25.0,
) -> MassEfficiencyFrontier:
    """Map the Tsiolkovsky mass efficiency frontier across altitudes.

    ΔV_total(alt) = ΔV_raise(alt) + ΔV_SK(alt) * T_mission
    M_wet(alt) = m_dry * exp(ΔV_total / (Isp*g0))
    Efficiency(alt) = 1 / (M_wet * num_sats)  [higher = better]
    Mass wall: altitude where M_wet grows 10x from minimum.
    """
    g0 = 9.80665
    r_e = OrbitalConstants.R_EARTH
    points = []

    num_alt_steps = max(0, int(round((alt_max_km - alt_min_km) / alt_step_km)))
    for alt_idx in range(num_alt_steps + 1):
        alt = alt_min_km + alt_idx * alt_step_km
        r_injection = r_e + injection_altitude_km * 1000.0
        r_operational = r_e + alt * 1000.0

        # Raise maneuver dV
        if abs(alt - injection_altitude_km) > 0.1:
            transfer = hohmann_transfer(r_injection, r_operational)
            dv_raise = transfer.total_delta_v_ms
        else:
            dv_raise = 0.0

        # Station-keeping dV over mission
        dv_sk_year = drag_compensation_dv_per_year(alt, drag_config)
        max_dv_sk = isp_s * g0 * 15.0  # Cap to prevent overflow
        dv_sk_total = min(dv_sk_year, max_dv_sk) * mission_years

        dv_total = dv_raise + dv_sk_total

        # Tsiolkovsky: m_wet = m_dry * exp(dv/(Isp*g0))
        exponent = dv_total / (isp_s * g0)
        if exponent > 20.0:
            exponent = 20.0  # Cap to prevent overflow
        wet_mass = dry_mass_kg * float(np.exp(exponent))
        constellation_mass = wet_mass * num_sats

        # Efficiency = inverse of total mass (higher = better)
        efficiency = 1.0 / constellation_mass if constellation_mass > 0 else 0.0

        points.append(MassEfficiencyPoint(
            altitude_km=alt,
            total_dv_ms=dv_total,
            wet_mass_kg=wet_mass,
            constellation_mass_kg=constellation_mass,
            mass_efficiency=efficiency,
        ))

    if not points:
        return MassEfficiencyFrontier(
            points=(), optimal_altitude_km=0.0,
            peak_efficiency=0.0, mass_wall_altitude_km=0.0,
        )

    # Find optimal altitude (peak efficiency)
    best_idx = max(range(len(points)), key=lambda i: points[i].mass_efficiency)
    optimal_alt = points[best_idx].altitude_km
    peak_eff = points[best_idx].mass_efficiency

    # Mass wall: lowest altitude where wet mass > 10x minimum wet mass
    min_wet = min(p.wet_mass_kg for p in points)
    mass_wall_alt = points[0].altitude_km
    for p in points:
        if p.wet_mass_kg > 10.0 * min_wet:
            mass_wall_alt = p.altitude_km
            break

    # If no mass wall found (all within 10x), set to min altitude
    if mass_wall_alt == points[0].altitude_km:
        mass_wall_found = False
        for p in points:
            if p.wet_mass_kg > 10.0 * min_wet:
                mass_wall_alt = p.altitude_km
                mass_wall_found = True
                break
        if not mass_wall_found:
            mass_wall_alt = points[0].altitude_km

    return MassEfficiencyFrontier(
        points=tuple(points),
        optimal_altitude_km=optimal_alt,
        peak_efficiency=peak_eff,
        mass_wall_altitude_km=mass_wall_alt,
    )


# ── Wasserstein Distance for Constellation Reconfiguration (P19) ──

@dataclass(frozen=True)
class ReconfigurationPlan:
    """Optimal transport plan for constellation reconfiguration.

    The Wasserstein-1 distance (Earth Mover's Distance) between the
    current and target constellation configurations gives the minimum
    total delta-V for reconfiguration, with optimal satellite-to-slot
    assignment computed via the Hungarian algorithm.

    Attributes:
        total_dv_ms: Total delta-V for all satellite transfers (m/s).
        assignments: Tuple of (current_idx, target_idx, dv_ms) assignments.
        wasserstein_distance: W_1 distance (same as total_dv_ms for
            balanced assignment, normalized otherwise).
        savings_vs_naive_pct: Savings vs naive sequential assignment.
    """
    total_dv_ms: float
    assignments: tuple[tuple[int, int, float], ...]
    wasserstein_distance: float
    savings_vs_naive_pct: float


def compute_optimal_reconfiguration(
    current_states: list[OrbitalState],
    target_states: list[OrbitalState],
) -> ReconfigurationPlan:
    """Compute optimal constellation reconfiguration via optimal transport.

    Uses the Hungarian algorithm (Kuhn-Munkres) to solve the assignment
    problem: which current satellite should be moved to which target slot
    to minimize total delta-V.

    The cost of moving satellite i to slot j is the sum of:
    1. Hohmann transfer dV for altitude change (if needed).
    2. Plane change dV for inclination/RAAN change.

    For unbalanced cases (different number of current and target sats),
    dummy rows/columns are added with zero cost.

    The 1D Wasserstein distance on RAAN distributions is also computed
    as a secondary metric for plane reconfiguration cost.

    Args:
        current_states: List of current satellite OrbitalState objects.
        target_states: List of target slot OrbitalState objects.

    Returns:
        ReconfigurationPlan with optimal assignment and total dV.
    """
    if len(current_states) > 500:
        raise ValueError(
            "Hungarian algorithm limited to 500 satellites; "
            "use greedy for larger sets"
        )
    if len(target_states) > 500:
        raise ValueError(
            "Hungarian algorithm limited to 500 satellites; "
            "use greedy for larger sets"
        )

    n_current = len(current_states)
    n_target = len(target_states)

    if n_current == 0 or n_target == 0:
        return ReconfigurationPlan(
            total_dv_ms=0.0,
            assignments=(),
            wasserstein_distance=0.0,
            savings_vs_naive_pct=0.0,
        )

    # Build cost matrix: cost[i][j] = dV to move current[i] to target[j]
    n = max(n_current, n_target)
    cost = np.zeros((n, n))

    mu = OrbitalConstants.MU_EARTH

    for i in range(n_current):
        for j in range(n_target):
            a_i = current_states[i].semi_major_axis_m
            a_j = target_states[j].semi_major_axis_m

            # Hohmann transfer dV for altitude change
            if abs(a_i - a_j) > 1.0:  # >1 meter difference
                transfer = hohmann_transfer(a_i, a_j)
                dv_alt = transfer.total_delta_v_ms
            else:
                dv_alt = 0.0

            # Plane change dV: combined inclination and RAAN change
            # dV = 2 * v * sin(d_theta / 2)
            # where d_theta is the angle between the orbital planes
            inc_i = current_states[i].inclination_rad
            inc_j = target_states[j].inclination_rad
            raan_i = current_states[i].raan_rad
            raan_j = target_states[j].raan_rad

            # Angle between orbital planes via spherical geometry
            # cos(d_theta) = cos(i1)*cos(i2) + sin(i1)*sin(i2)*cos(dOmega)
            d_raan = raan_j - raan_i
            cos_d_theta = (math.cos(inc_i) * math.cos(inc_j)
                           + math.sin(inc_i) * math.sin(inc_j) * math.cos(d_raan))
            cos_d_theta = max(-1.0, min(1.0, cos_d_theta))
            d_theta = math.acos(cos_d_theta)

            if d_theta > 1e-8:
                # Use velocity at the higher orbit (more efficient)
                a_plane = max(a_i, a_j)
                v_circ = float(np.sqrt(mu / a_plane))
                dv_plane = 2.0 * v_circ * math.sin(d_theta / 2.0)
            else:
                dv_plane = 0.0

            cost[i, j] = dv_alt + dv_plane

    # Solve assignment problem using Hungarian algorithm
    row_ind, col_ind = _hungarian_algorithm(cost)

    # Build assignment list
    assignments: list[tuple[int, int, float]] = []
    total_dv = 0.0
    for r, c in zip(row_ind, col_ind):
        if r < n_current and c < n_target:
            dv = cost[r, c]
            assignments.append((r, c, dv))
            total_dv += dv

    # Naive cost: sequential assignment (current[0]->target[0], etc.)
    naive_dv = 0.0
    for k in range(min(n_current, n_target)):
        naive_dv += cost[k, k]

    savings = ((naive_dv - total_dv) / naive_dv * 100.0) if naive_dv > 0 else 0.0
    savings = max(0.0, savings)

    # Wasserstein distance: total optimal transport cost
    # Normalized by number of assignments for comparability
    n_assignments = len(assignments)
    wasserstein = total_dv / n_assignments if n_assignments > 0 else 0.0

    return ReconfigurationPlan(
        total_dv_ms=total_dv,
        assignments=tuple(assignments),
        wasserstein_distance=wasserstein,
        savings_vs_naive_pct=savings,
    )


def _hungarian_algorithm(cost_matrix: np.ndarray) -> tuple[list[int], list[int]]:
    """Solve the assignment problem using the Hungarian (Kuhn-Munkres) algorithm.

    Finds the assignment that minimizes the total cost.

    This is a pure-numpy implementation suitable for small to medium matrices
    (up to ~1000 x 1000). Uses the successive shortest paths approach.

    Args:
        cost_matrix: n x n cost matrix (non-negative).

    Returns:
        (row_indices, col_indices) of the optimal assignment.
    """
    n = cost_matrix.shape[0]
    if n == 0:
        return [], []

    # For small matrices, use brute-force-friendly approach
    # For larger, use the O(n^3) Hungarian algorithm

    # O(n^3) Hungarian algorithm (Jonker-Volgenant variant)
    # Adapted from scipy.optimize.linear_sum_assignment logic

    cost = cost_matrix.copy()

    # Step 1: Subtract row minimums
    row_min = cost.min(axis=1, keepdims=True)
    cost = cost - row_min

    # Step 2: Subtract column minimums
    col_min = cost.min(axis=0, keepdims=True)
    cost = cost - col_min

    # Greedy initial assignment
    row_assign = [-1] * n
    col_assign = [-1] * n

    for i in range(n):
        for j in range(n):
            if cost[i, j] < 1e-12 and col_assign[j] == -1:
                row_assign[i] = j
                col_assign[j] = i
                break

    # Augment until all rows assigned
    for _ in range(n):
        unassigned = [i for i in range(n) if row_assign[i] == -1]
        if not unassigned:
            break

        for start_row in unassigned:
            # BFS/Dijkstra to find augmenting path
            dist = np.full(n, np.inf)
            prev_col = [-1] * n
            prev_row = [-1] * n
            visited_col = [False] * n
            visited_row = [False] * n

            # Initialize distances from unassigned row
            for j in range(n):
                dist[j] = cost[start_row, j]
                prev_row[j] = start_row

            found_col = -1
            while True:
                # Find minimum-distance unvisited column
                min_dist = np.inf
                min_col = -1
                for j in range(n):
                    if not visited_col[j] and dist[j] < min_dist:
                        min_dist = dist[j]
                        min_col = j

                if min_col == -1:
                    break

                visited_col[min_col] = True

                if col_assign[min_col] == -1:
                    # Found augmenting path
                    found_col = min_col
                    break

                # Extend through assigned row
                assigned_row = col_assign[min_col]
                visited_row[assigned_row] = True

                for j in range(n):
                    if not visited_col[j]:
                        new_dist = min_dist + cost[assigned_row, j] - cost[assigned_row, min_col]
                        if new_dist < dist[j]:
                            dist[j] = new_dist
                            prev_col[j] = min_col
                            prev_row[j] = assigned_row

            if found_col == -1:
                continue

            # Update dual variables (reduce costs along the path)
            for j in range(n):
                if visited_col[j]:
                    # Reduce cost along the augmenting path
                    cost[:, j] -= (dist[j] - dist[found_col])
                    cost[:, j] = np.maximum(cost[:, j], 0.0)

            # Augment: trace back through prev_col/prev_row chain,
            # alternating assigned/unassigned edges (standard Hungarian
            # augmenting path reconstruction).
            col = found_col
            while True:
                row = prev_row[col]
                # Save the column that this row was previously assigned to
                old_col = row_assign[row]
                # Assign this row to the new column
                row_assign[row] = col
                col_assign[col] = row
                # Move to the previous column in the alternating path
                if row == start_row:
                    break
                col = old_col

    # All rows should be assigned after augmentation. If not, the
    # augmenting path reconstruction has a bug. This should not happen
    # for well-formed cost matrices.

    return list(range(n)), row_assign


# ── P26: Free Energy Landscape for Constellation Design ──

@dataclass(frozen=True)
class DesignFreeEnergyLandscape:
    """Statistical mechanics free energy landscape for constellation design.

    Maps constellation design optimization to a thermodynamic analogy:
    - Energy E(x) = -score (lower energy = better design)
    - Partition function Z(T) = sum exp(-E(x)/T)
    - Free energy F(T) = -T * ln(Z(T))
    - Boltzmann probabilities P(x) = exp(-E(x)/T) / Z(T)
    - Entropy S = -sum P(x) * ln(P(x))
    - Heat capacity C = Var(E) / T^2

    Attributes:
        optimal_config_index: Index of the best configuration (lowest energy).
        free_energy: Free energy F at the operating temperature.
        design_entropy: Shannon entropy of the Boltzmann distribution.
        heat_capacity: Heat capacity C_V = Var(E) / T^2.
        metastable_configs: Indices of local minima (high Boltzmann probability).
        temperature_sweep: Tuple of temperatures used in the sweep.
        free_energy_curve: Tuple of free energies at each temperature.
    """
    optimal_config_index: int
    free_energy: float
    design_entropy: float
    heat_capacity: float
    metastable_configs: tuple
    temperature_sweep: tuple
    free_energy_curve: tuple


def compute_design_free_energy(
    scores: list,
    temperature: float = 1.0,
    n_temp_steps: int = 20,
    temp_min: float = 0.1,
    temp_max: float = 10.0,
    metastable_threshold: float = 0.01,
) -> DesignFreeEnergyLandscape:
    """Compute the free energy landscape for a set of constellation designs.

    Each configuration has a score (higher is better). The score is mapped
    to energy E = -score. The Boltzmann distribution at temperature T gives
    the probability of each configuration. At low T, the distribution
    concentrates on the optimum; at high T, all configs are equiprobable.

    Metastable configurations are those with Boltzmann probability above
    the threshold, excluding the global optimum.

    Args:
        scores: List of design scores (higher is better). One per config.
        temperature: Operating temperature for the main analysis.
        n_temp_steps: Number of temperature steps for sweep.
        temp_min: Minimum temperature for sweep.
        temp_max: Maximum temperature for sweep.
        metastable_threshold: Minimum Boltzmann probability to be metastable.

    Returns:
        DesignFreeEnergyLandscape with free energy, entropy, heat capacity,
        metastable configs, and temperature sweep data.
    """
    if not scores:
        return DesignFreeEnergyLandscape(
            optimal_config_index=0,
            free_energy=0.0,
            design_entropy=0.0,
            heat_capacity=0.0,
            metastable_configs=(),
            temperature_sweep=(),
            free_energy_curve=(),
        )

    scores_arr = np.array(scores, dtype=np.float64)
    n_configs = len(scores_arr)

    # Energy: E(x) = -score (lower energy = better)
    energies = -scores_arr

    # Optimal configuration (lowest energy = highest score)
    optimal_idx = int(np.argmin(energies))

    # Boltzmann distribution at operating temperature
    probs, log_z, free_en = _boltzmann_distribution(energies, temperature)

    # Design entropy: S = -sum P(x) * ln(P(x))
    entropy = 0.0
    for p in probs:
        if p > 1e-300:
            entropy -= p * math.log(p)

    # Heat capacity: C = Var(E) / T^2
    mean_e = float(np.dot(probs, energies))
    mean_e_sq = float(np.dot(probs, energies ** 2))
    var_e = mean_e_sq - mean_e ** 2
    heat_capacity = var_e / (temperature ** 2) if temperature > 1e-15 else 0.0

    # Metastable configurations: high probability, not the global optimum
    metastable = []
    for i in range(n_configs):
        if i != optimal_idx and probs[i] >= metastable_threshold:
            metastable.append(i)

    # Temperature sweep
    temps = np.linspace(temp_min, temp_max, n_temp_steps)
    free_energies = np.zeros(n_temp_steps)
    for t_idx in range(n_temp_steps):
        _, _, f_t = _boltzmann_distribution(energies, float(temps[t_idx]))
        free_energies[t_idx] = f_t

    return DesignFreeEnergyLandscape(
        optimal_config_index=optimal_idx,
        free_energy=free_en,
        design_entropy=entropy,
        heat_capacity=heat_capacity,
        metastable_configs=tuple(metastable),
        temperature_sweep=tuple(float(t) for t in temps),
        free_energy_curve=tuple(float(f) for f in free_energies),
    )


def _boltzmann_distribution(
    energies: np.ndarray,
    temperature: float,
) -> tuple:
    """Compute Boltzmann distribution for given energies and temperature.

    Uses the log-sum-exp trick for numerical stability.

    Returns (probabilities, log_partition_function, free_energy).
    """
    if temperature < 1e-15:
        # Zero temperature: all weight on minimum energy
        probs = np.zeros(len(energies))
        min_idx = int(np.argmin(energies))
        probs[min_idx] = 1.0
        free_energy = float(energies[min_idx])
        return probs, 0.0, free_energy

    beta = 1.0 / temperature
    neg_beta_e = -beta * energies

    # Log-sum-exp trick
    max_val = float(np.max(neg_beta_e))
    log_z = max_val + math.log(float(np.sum(np.exp(neg_beta_e - max_val))))

    log_probs = neg_beta_e - log_z
    probs = np.exp(log_probs)

    # Normalize (belt and suspenders)
    total = float(np.sum(probs))
    if total > 0:
        probs = probs / total

    free_energy = -temperature * log_z

    return probs, log_z, free_energy


# ── P28: Vickrey Auction for Orbital Slot Allocation ──

@dataclass(frozen=True)
class AuctionResult:
    """Result of Vickrey-Clarke-Groves auction for orbital slot allocation.

    Implements the VCG mechanism for truthful multi-item allocation:
    - Each operator bids a valuation V_i for each slot.
    - Allocation maximizes total social welfare (sum of winning valuations).
    - Payment for each winner = welfare of others without winner - welfare
      of others with winner (externality-based pricing).
    - For single-item case, reduces to second-price (Vickrey) auction.

    Attributes:
        allocations: Tuple of (bidder_index, slot_index) assignments.
        payments: Tuple of payment amounts per allocated bidder.
        social_welfare: Total valuation of the winning allocation.
        total_revenue: Sum of all payments (revenue to auctioneer).
        price_per_slot: Average revenue per allocated slot.
    """
    allocations: tuple
    payments: tuple
    social_welfare: float
    total_revenue: float
    price_per_slot: float


def compute_orbital_slot_auction(
    valuations: list,
    n_slots: int,
) -> AuctionResult:
    """Run a VCG (Vickrey-Clarke-Groves) auction for orbital slots.

    Supports multi-slot allocation with truthful revelation:
    - Each bidder submits valuations for each slot.
    - The mechanism allocates slots to maximize social welfare.
    - Payments are based on the externality each winner imposes.

    For the single-slot case, this reduces to a standard Vickrey
    (second-price sealed-bid) auction.

    Args:
        valuations: List of lists. valuations[i][j] = bidder i's
            valuation for slot j. Shape: (n_bidders, n_slots).
            Each bidder can win at most one slot.
        n_slots: Number of slots to allocate.

    Returns:
        AuctionResult with allocations, payments, and welfare metrics.
    """
    if not valuations or n_slots <= 0:
        return AuctionResult(
            allocations=(),
            payments=(),
            social_welfare=0.0,
            total_revenue=0.0,
            price_per_slot=0.0,
        )

    val = np.array(valuations, dtype=np.float64)
    n_bidders = val.shape[0]

    if val.ndim == 1:
        # Single slot: each bidder has one valuation
        val = val.reshape(-1, 1)

    actual_slots = min(n_slots, val.shape[1])

    # Step 1: Find welfare-maximizing allocation using Hungarian algorithm
    # Negate valuations for cost minimization
    n = max(n_bidders, actual_slots)
    cost = np.zeros((n, n))
    for i in range(n_bidders):
        for j in range(actual_slots):
            cost[i, j] = -val[i, j]

    row_ind, col_ind = _hungarian_algorithm(cost)

    # Extract valid allocations (bidder < n_bidders and slot < actual_slots)
    allocations = []
    total_welfare = 0.0
    winners = set()

    for r, c in zip(row_ind, col_ind):
        if r < n_bidders and c < actual_slots and val[r, c] > 0:
            allocations.append((r, c))
            total_welfare += val[r, c]
            winners.add(r)

    # Step 2: Compute VCG payments
    # Payment for bidder i = (welfare of others without i) - (welfare of others with i)
    payments = []
    for bidder_idx, slot_idx in allocations:
        # Welfare of others with bidder present
        welfare_others_with = total_welfare - val[bidder_idx, slot_idx]

        # Welfare without bidder: re-run allocation excluding bidder
        n_excl = max(n_bidders - 1, actual_slots)
        cost_excl = np.zeros((n_excl, n_excl))
        # Map original bidder indices (excluding bidder_idx) to new indices
        excl_map = []
        for i in range(n_bidders):
            if i != bidder_idx:
                excl_map.append(i)

        for new_i, orig_i in enumerate(excl_map):
            for j in range(actual_slots):
                cost_excl[new_i, j] = -val[orig_i, j]

        row_excl, col_excl = _hungarian_algorithm(cost_excl)
        welfare_without = 0.0
        for r, c in zip(row_excl, col_excl):
            if r < len(excl_map) and c < actual_slots:
                welfare_without += val[excl_map[r], c]

        # VCG payment: externality
        payment = welfare_without - welfare_others_with
        payments.append(max(0.0, payment))

    total_revenue = sum(payments)
    n_allocated = len(allocations)
    price_per = total_revenue / n_allocated if n_allocated > 0 else 0.0

    return AuctionResult(
        allocations=tuple(allocations),
        payments=tuple(payments),
        social_welfare=total_welfare,
        total_revenue=total_revenue,
        price_per_slot=price_per,
    )


# ── P35: r/K Selection Strategy for Constellation Design ──

@dataclass(frozen=True)
class ConstellationStrategy:
    """r/K ecological selection strategy for constellation design.

    Maps biological r/K selection theory to constellation design:
    - r-selected: many cheap, short-lived satellites (high throughput)
    - K-selected: fewer expensive, long-lived satellites (high efficiency)

    The environment r/K ratio indicates whether conditions favor
    high-turnover (hostile: high drag, high conjunction rate) or
    long-duration (benign) strategies.

    Mismatch penalty quantifies how far the chosen design strategy
    deviates from what the environment demands.

    Attributes:
        strategy_type: "r-selected" or "K-selected".
        r_metric: r-selection metric (coverage throughput per dollar-year).
        k_metric: K-selection metric (coverage efficiency * durability).
        environment_rk_ratio: Environmental r/K pressure ratio.
        design_rk_ratio: Design's r/K characteristic ratio.
        mismatch_penalty: |log(design_rk / environment_rk)| penalty.
    """
    strategy_type: str
    r_metric: float
    k_metric: float
    environment_rk_ratio: float
    design_rk_ratio: float
    mismatch_penalty: float


def compute_constellation_strategy(
    coverage_fraction: float,
    cost_per_satellite: float,
    lifetime_years: float,
    n_satellites: int,
    survival_probability: float,
    drag_rate_per_year: float,
    conjunction_rate_per_year: float,
    max_useful_sats: int,
    launch_rate_per_year: float,
) -> ConstellationStrategy:
    """Compute r/K selection strategy for constellation design.

    Evaluates whether the constellation design matches its environment
    using ecological selection theory:

    r-metric = (coverage_per_dollar_per_year) * survival_probability
        High r = good at rapid, cheap deployment (r-selected)

    K-metric = (coverage_per_satellite) * lifetime / cost
        High K = good at efficient, long-duration operation (K-selected)

    Environment r/K ratio = (drag_rate + conjunction_rate) / max_useful_sats
        High ratio = hostile environment favoring r-selection

    Design r/K ratio = (launch_rate * coverage_per_sat) / (cost * lifetime)
        High ratio = r-selected design

    Args:
        coverage_fraction: Fraction of Earth covered (0-1).
        cost_per_satellite: Cost per satellite (arbitrary units).
        lifetime_years: Expected satellite lifetime in years.
        n_satellites: Number of satellites in constellation.
        survival_probability: Probability a satellite survives its lifetime.
        drag_rate_per_year: Atmospheric drag decay rate (1/year).
        conjunction_rate_per_year: Conjunction/collision rate (events/year).
        max_useful_sats: Maximum useful satellites in the orbital regime.
        launch_rate_per_year: Constellation replenishment rate (sats/year).

    Returns:
        ConstellationStrategy with strategy type and metrics.
    """
    # Guard against zero/negative inputs
    cost_safe = max(cost_per_satellite, 1e-10)
    lifetime_safe = max(lifetime_years, 1e-10)
    n_sats_safe = max(n_satellites, 1)
    max_useful_safe = max(max_useful_sats, 1)
    launch_rate_safe = max(launch_rate_per_year, 1e-10)

    # Coverage per satellite
    coverage_per_sat = coverage_fraction / n_sats_safe

    # r-metric: coverage throughput per dollar-year, weighted by survival
    # coverage / (cost * lifetime) per satellite, times survival
    coverage_per_dollar_per_year = coverage_fraction / (cost_safe * lifetime_safe)
    r_metric = coverage_per_dollar_per_year * survival_probability

    # K-metric: coverage efficiency * durability per cost
    k_metric = coverage_per_sat * lifetime_safe / cost_safe

    # Environment r/K ratio: pressure from hostile environment
    environment_rk = (drag_rate_per_year + conjunction_rate_per_year) / max_useful_safe

    # Design r/K ratio: design's inherent characteristic
    design_rk = (launch_rate_safe * coverage_per_sat) / (cost_safe * lifetime_safe)

    # Mismatch penalty: how far design diverges from environment demand
    if environment_rk > 1e-15 and design_rk > 1e-15:
        mismatch = abs(math.log(design_rk / environment_rk))
    elif environment_rk < 1e-15 and design_rk < 1e-15:
        mismatch = 0.0
    else:
        mismatch = float('inf')

    # Strategy classification
    if design_rk > environment_rk:
        strategy_type = "r-selected"
    else:
        strategy_type = "K-selected"

    return ConstellationStrategy(
        strategy_type=strategy_type,
        r_metric=r_metric,
        k_metric=k_metric,
        environment_rk_ratio=environment_rk,
        design_rk_ratio=design_rk,
        mismatch_penalty=mismatch,
    )
