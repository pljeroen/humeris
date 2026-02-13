# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Hodge-CUSUM topology change detector.

Sequential detection of ISL network topology changes using Hodge Laplacian
spectral features monitored by CUSUM change-point detection.

Tracks betti_1 (cycle count), L1 spectral gap, and routing redundancy
as time series. Applies two-sided CUSUM to detect statistically significant
shifts indicating topology reconfiguration, link failure, or degradation.

"""
from dataclasses import dataclass

import numpy as np

from humeris.domain.graph_analysis import compute_hodge_topology


@dataclass(frozen=True)
class TopologySnapshot:
    """Single topology measurement at a point in time."""
    time_index: int
    betti_1: int  # Number of independent cycles
    l1_spectral_gap: float  # Gap in L1 Hodge Laplacian spectrum
    routing_redundancy: float  # Redundancy metric
    triangle_count: int  # Number of 3-cliques


@dataclass(frozen=True)
class TopologyChangeEvent:
    """Detected topology change event."""
    time_index: int
    feature_name: str  # Which feature triggered: "betti_1", "spectral_gap", "redundancy"
    cusum_value: float  # CUSUM statistic at detection
    direction: str  # "increase" or "decrease"
    magnitude: float  # Estimated shift magnitude


@dataclass(frozen=True)
class HodgeCusumResult:
    """Result of Hodge-CUSUM topology monitoring."""
    snapshots: tuple  # Tuple of TopologySnapshot
    events: tuple  # Tuple of TopologyChangeEvent
    cusum_betti: tuple  # CUSUM+ history for betti_1
    cusum_spectral: tuple  # CUSUM+ history for spectral gap
    cusum_redundancy: tuple  # CUSUM+ history for redundancy
    mean_betti_1: float
    mean_spectral_gap: float
    mean_redundancy: float
    num_topology_changes: int


def compute_topology_snapshot(
    adjacency: list,
    n_nodes: int,
    time_index: int,
) -> TopologySnapshot:
    """Compute Hodge topology features from current adjacency matrix.

    Uses compute_hodge_topology() internally to extract spectral and
    combinatorial features from the ISL network at the given time index.

    Args:
        adjacency: n_nodes x n_nodes adjacency matrix (symmetric, weight > 0
            indicates an edge).
        n_nodes: Number of nodes in the graph.
        time_index: Integer time index for this snapshot.

    Returns:
        TopologySnapshot with Hodge topology features.
    """
    hodge = compute_hodge_topology(adjacency, n_nodes)
    return TopologySnapshot(
        time_index=time_index,
        betti_1=hodge.betti_1,
        l1_spectral_gap=hodge.l1_spectral_gap,
        routing_redundancy=hodge.routing_redundancy,
        triangle_count=hodge.triangle_count,
    )


def monitor_topology_cusum(
    snapshots: list,
    threshold: float = 5.0,
    drift: float = 0.5,
    baseline_window: int = 0,
) -> HodgeCusumResult:
    """Monitor topology time series with two-sided CUSUM change detection.

    Normalizes each feature (betti_1, spectral_gap, redundancy) by its
    baseline mean and standard deviation, then runs two-sided CUSUM on each
    normalized feature. Detects events when CUSUM exceeds threshold. Uses
    Hawkins-Olwell reset (S = S - threshold) to allow multiple detections.

    Args:
        snapshots: List of TopologySnapshot objects (time-ordered).
        threshold: CUSUM detection threshold. Default 5.0.
        drift: CUSUM allowance parameter. Default 0.5.
        baseline_window: Number of initial snapshots for baseline estimation.
            0 = auto (first 20% of snapshots, minimum 1).

    Returns:
        HodgeCusumResult with detection events and CUSUM histories.
    """
    n = len(snapshots)
    if n == 0:
        return HodgeCusumResult(
            snapshots=(),
            events=(),
            cusum_betti=(),
            cusum_spectral=(),
            cusum_redundancy=(),
            mean_betti_1=0.0,
            mean_spectral_gap=0.0,
            mean_redundancy=0.0,
            num_topology_changes=0,
        )

    # Extract feature arrays
    betti_arr = np.array([s.betti_1 for s in snapshots], dtype=np.float64)
    spectral_arr = np.array([s.l1_spectral_gap for s in snapshots], dtype=np.float64)
    redundancy_arr = np.array([s.routing_redundancy for s in snapshots], dtype=np.float64)

    # Determine baseline window
    if baseline_window <= 0:
        baseline_window = max(1, n // 5)
    baseline_window = min(baseline_window, n)

    # Compute baseline statistics
    def _baseline_stats(arr: np.ndarray, bw: int) -> tuple:
        baseline = arr[:bw]
        mu = float(np.mean(baseline))
        if bw > 1:
            sigma = float(np.std(baseline, ddof=1))
        else:
            sigma = 0.0
        return mu, sigma

    mu_b, sigma_b = _baseline_stats(betti_arr, baseline_window)
    mu_s, sigma_s = _baseline_stats(spectral_arr, baseline_window)
    mu_r, sigma_r = _baseline_stats(redundancy_arr, baseline_window)

    # Normalize features: z_i = (f_i - mu) / sigma
    # When sigma=0 (constant baseline), any deviation from the mean is
    # infinitely significant in theory. Use the absolute deviation directly
    # so that genuine changes are still detectable.
    def _normalize(arr: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        if sigma < 1e-15:
            # Constant baseline: use raw deviation as z-score.
            # A difference of 1 unit from baseline counts as 1 sigma.
            return arr - mu
        return (arr - mu) / sigma

    z_betti = _normalize(betti_arr, mu_b, sigma_b)
    z_spectral = _normalize(spectral_arr, mu_s, sigma_s)
    z_redundancy = _normalize(redundancy_arr, mu_r, sigma_r)

    # Run two-sided CUSUM on each feature
    events = []

    # Map feature names to their baseline sigmas for magnitude denormalization
    _sigma_map = {
        "betti_1": sigma_b,
        "spectral_gap": sigma_s,
        "redundancy": sigma_r,
    }

    def _run_cusum(
        z_arr: np.ndarray,
        feature_name: str,
    ) -> tuple:
        """Run two-sided CUSUM and return (cusum_plus_history, detected_events)."""
        cusum_plus = 0.0
        cusum_minus = 0.0
        cusum_plus_hist = []
        detected = []

        sigma_feat = _sigma_map[feature_name]

        for i in range(n):
            z = float(z_arr[i])

            # Positive side: detects increases
            cusum_plus = max(0.0, cusum_plus + z - drift)
            # Negative side: detects decreases
            cusum_minus = max(0.0, cusum_minus - z - drift)

            cusum_plus_hist.append(cusum_plus)

            if cusum_plus > threshold:
                # Denormalize magnitude: if sigma was used, multiply back;
                # if sigma was 0, z is already in original units.
                if sigma_feat >= 1e-15:
                    magnitude = abs(float(z_arr[i]) * sigma_feat)
                else:
                    magnitude = abs(float(z_arr[i]))
                detected.append(TopologyChangeEvent(
                    time_index=snapshots[i].time_index,
                    feature_name=feature_name,
                    cusum_value=cusum_plus,
                    direction="increase",
                    magnitude=magnitude,
                ))
                # Hawkins-Olwell reset: preserve accumulated evidence
                cusum_plus = max(0.0, cusum_plus - threshold)

            if cusum_minus > threshold:
                if sigma_feat >= 1e-15:
                    magnitude = abs(float(z_arr[i]) * sigma_feat)
                else:
                    magnitude = abs(float(z_arr[i]))
                detected.append(TopologyChangeEvent(
                    time_index=snapshots[i].time_index,
                    feature_name=feature_name,
                    cusum_value=cusum_minus,
                    direction="decrease",
                    magnitude=magnitude,
                ))
                # Hawkins-Olwell reset
                cusum_minus = max(0.0, cusum_minus - threshold)

        return tuple(cusum_plus_hist), detected

    cusum_betti_hist, events_betti = _run_cusum(z_betti, "betti_1")
    cusum_spectral_hist, events_spectral = _run_cusum(z_spectral, "spectral_gap")
    cusum_redundancy_hist, events_redundancy = _run_cusum(z_redundancy, "redundancy")

    all_events = events_betti + events_spectral + events_redundancy
    # Sort events by time_index
    all_events.sort(key=lambda e: e.time_index)

    # Overall means (across all snapshots)
    mean_betti = float(np.mean(betti_arr))
    mean_spectral = float(np.mean(spectral_arr))
    mean_redundancy = float(np.mean(redundancy_arr))

    return HodgeCusumResult(
        snapshots=tuple(snapshots),
        events=tuple(all_events),
        cusum_betti=cusum_betti_hist,
        cusum_spectral=cusum_spectral_hist,
        cusum_redundancy=cusum_redundancy_hist,
        mean_betti_1=mean_betti,
        mean_spectral_gap=mean_spectral,
        mean_redundancy=mean_redundancy,
        num_topology_changes=len(all_events),
    )


def detect_link_failure(
    adjacency_before: list,
    adjacency_after: list,
    n_nodes: int,
) -> tuple:
    """Compare two adjacency matrices to identify which links changed.

    Identifies links that were lost (present before, absent after) and
    links that were gained (absent before, present after), plus the
    topology impact in terms of Hodge features.

    Args:
        adjacency_before: n_nodes x n_nodes adjacency matrix before change.
        adjacency_after: n_nodes x n_nodes adjacency matrix after change.
        n_nodes: Number of nodes in the graph.

    Returns:
        Tuple of (lost_links, gained_links, topology_impact) where:
        - lost_links: tuple of (i, j) pairs for removed links
        - gained_links: tuple of (i, j) pairs for added links
        - topology_impact: dict with delta_betti_1, delta_spectral_gap
    """
    before = np.array(adjacency_before, dtype=np.float64)
    after = np.array(adjacency_after, dtype=np.float64)

    lost_links = []
    gained_links = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            had_link = before[i, j] > 0
            has_link = after[i, j] > 0
            if had_link and not has_link:
                lost_links.append((i, j))
            elif not had_link and has_link:
                gained_links.append((i, j))

    # Compute topology impact
    hodge_before = compute_hodge_topology(adjacency_before, n_nodes)
    hodge_after = compute_hodge_topology(adjacency_after, n_nodes)

    topology_impact = {
        "delta_betti_1": hodge_after.betti_1 - hodge_before.betti_1,
        "delta_spectral_gap": hodge_after.l1_spectral_gap - hodge_before.l1_spectral_gap,
    }

    return tuple(lost_links), tuple(gained_links), topology_impact


def compute_topology_resilience_score(snapshots: list) -> float:
    """Compute a 0-1 resilience score from topology time series.

    Based on: min(spectral_gap) / max(spectral_gap) * mean(redundancy).
    Higher values indicate more resilient topology over the observation
    window. Returns 0.0 for empty or degenerate inputs.

    Args:
        snapshots: List of TopologySnapshot objects.

    Returns:
        Float in [0, 1] representing topology resilience.
    """
    if not snapshots:
        return 0.0

    spectral_gaps = np.array(
        [s.l1_spectral_gap for s in snapshots], dtype=np.float64,
    )
    redundancies = np.array(
        [s.routing_redundancy for s in snapshots], dtype=np.float64,
    )

    max_sg = float(np.max(spectral_gaps))
    min_sg = float(np.min(spectral_gaps))
    mean_red = float(np.mean(redundancies))

    if max_sg < 1e-15:
        return 0.0

    score = (min_sg / max_sg) * mean_red
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))
