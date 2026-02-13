# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/hodge_cusum.py — Hodge-CUSUM topology change detector."""
import math

from humeris.domain.hodge_cusum import (
    TopologySnapshot,
    TopologyChangeEvent,
    HodgeCusumResult,
    compute_topology_snapshot,
    monitor_topology_cusum,
    detect_link_failure,
    compute_topology_resilience_score,
)


# ---------------------------------------------------------------------------
# Test adjacency matrices
# ---------------------------------------------------------------------------

def _complete_4():
    """Complete graph on 4 nodes (K4)."""
    return [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]


def _ring_4():
    """Ring graph on 4 nodes (C4)."""
    return [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]


def _ring_5():
    """Ring graph on 5 nodes (C5)."""
    return [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ]


def _broken_ring_4():
    """Ring on 4 nodes with one link removed (path graph P4: 0-1-2-3)."""
    return [
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ]


def _complete_5():
    """Complete graph on 5 nodes (K5)."""
    n = 5
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i][j] = 1
    return adj


# ---------------------------------------------------------------------------
# TopologySnapshot tests
# ---------------------------------------------------------------------------

class TestTopologySnapshotCompleteGraph:
    def test_topology_snapshot_complete_graph(self):
        """Complete graph K4 has known Hodge features: 4 triangles, betti_1=0."""
        snap = compute_topology_snapshot(_complete_4(), 4, time_index=0)
        assert isinstance(snap, TopologySnapshot)
        assert snap.time_index == 0
        # K4 has 4 triangles (C(4,3) = 4)
        assert snap.triangle_count == 4
        # K4: all cycles are boundaries of triangles, so betti_1 = 0
        assert snap.betti_1 == 0
        # Spectral gap should be positive (connected, no harmonic 1-forms)
        assert snap.l1_spectral_gap > 0.0


class TestTopologySnapshotRingGraph:
    def test_topology_snapshot_ring_graph(self):
        """Ring graph C4 has betti_1 = 1 (one independent cycle)."""
        snap = compute_topology_snapshot(_ring_4(), 4, time_index=1)
        assert snap.betti_1 == 1
        # Ring has no triangles
        assert snap.triangle_count == 0
        # Routing redundancy = betti_1 / n_edges = 1/4 = 0.25
        assert abs(snap.routing_redundancy - 0.25) < 1e-10


# ---------------------------------------------------------------------------
# CUSUM: no change
# ---------------------------------------------------------------------------

class TestCusumNoChangeNoEvents:
    def test_cusum_no_change_no_events(self):
        """Constant topology produces no change events."""
        # Build 30 identical snapshots from the same ring graph
        snapshots = [
            compute_topology_snapshot(_ring_4(), 4, time_index=i)
            for i in range(30)
        ]
        result = monitor_topology_cusum(snapshots, threshold=5.0, drift=0.5)
        assert isinstance(result, HodgeCusumResult)
        assert result.num_topology_changes == 0
        assert len(result.events) == 0


# ---------------------------------------------------------------------------
# CUSUM: detects link loss
# ---------------------------------------------------------------------------

class TestCusumDetectsLinkLoss:
    def test_cusum_detects_link_loss(self):
        """Removing a link from the ring should trigger a detection."""
        # 20 snapshots of ring_4, then 20 snapshots of broken_ring_4
        snapshots = []
        for i in range(20):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))
        for i in range(20, 40):
            snapshots.append(compute_topology_snapshot(_broken_ring_4(), 4, time_index=i))

        result = monitor_topology_cusum(
            snapshots, threshold=3.0, drift=0.5, baseline_window=20,
        )
        # Should detect at least one event after the link removal
        assert result.num_topology_changes > 0
        # At least one event should occur at or after time_index 20
        late_events = [e for e in result.events if e.time_index >= 20]
        assert len(late_events) > 0


# ---------------------------------------------------------------------------
# CUSUM: detects link addition
# ---------------------------------------------------------------------------

class TestCusumDetectsLinkAddition:
    def test_cusum_detects_link_addition(self):
        """Adding links (ring -> complete) should trigger detection."""
        snapshots = []
        for i in range(20):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))
        for i in range(20, 40):
            snapshots.append(compute_topology_snapshot(_complete_4(), 4, time_index=i))

        result = monitor_topology_cusum(
            snapshots, threshold=3.0, drift=0.5, baseline_window=20,
        )
        assert result.num_topology_changes > 0
        late_events = [e for e in result.events if e.time_index >= 20]
        assert len(late_events) > 0


# ---------------------------------------------------------------------------
# CUSUM: multiple changes
# ---------------------------------------------------------------------------

class TestCusumMultipleChanges:
    def test_cusum_multiple_changes(self):
        """Two separate topology changes should produce events in both regions."""
        snapshots = []
        # Phase 1: ring (0-19)
        for i in range(20):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))
        # Phase 2: complete (20-39)
        for i in range(20, 40):
            snapshots.append(compute_topology_snapshot(_complete_4(), 4, time_index=i))
        # Phase 3: back to ring (40-59)
        for i in range(40, 60):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))

        result = monitor_topology_cusum(
            snapshots, threshold=3.0, drift=0.5, baseline_window=20,
        )
        # Should detect changes in both transition regions
        assert result.num_topology_changes >= 2


# ---------------------------------------------------------------------------
# CUSUM: threshold sensitivity
# ---------------------------------------------------------------------------

class TestCusumThresholdSensitivity:
    def test_cusum_threshold_sensitivity(self):
        """Higher threshold produces fewer or equal events."""
        snapshots = []
        for i in range(20):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))
        for i in range(20, 40):
            snapshots.append(compute_topology_snapshot(_complete_4(), 4, time_index=i))

        result_low = monitor_topology_cusum(
            snapshots, threshold=2.0, drift=0.5, baseline_window=20,
        )
        result_high = monitor_topology_cusum(
            snapshots, threshold=10.0, drift=0.5, baseline_window=20,
        )
        assert result_high.num_topology_changes <= result_low.num_topology_changes


# ---------------------------------------------------------------------------
# CUSUM: drift sensitivity
# ---------------------------------------------------------------------------

class TestCusumDriftSensitivity:
    def test_cusum_drift_sensitivity(self):
        """Higher drift produces fewer or equal events."""
        snapshots = []
        for i in range(20):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))
        for i in range(20, 40):
            snapshots.append(compute_topology_snapshot(_complete_4(), 4, time_index=i))

        result_low = monitor_topology_cusum(
            snapshots, threshold=3.0, drift=0.1, baseline_window=20,
        )
        result_high = monitor_topology_cusum(
            snapshots, threshold=3.0, drift=5.0, baseline_window=20,
        )
        assert result_high.num_topology_changes <= result_low.num_topology_changes


# ---------------------------------------------------------------------------
# CUSUM: baseline window
# ---------------------------------------------------------------------------

class TestCusumBaselineWindow:
    def test_cusum_baseline_window(self):
        """Custom baseline window works and influences results."""
        snapshots = []
        for i in range(20):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))
        for i in range(20, 40):
            snapshots.append(compute_topology_snapshot(_complete_4(), 4, time_index=i))

        # baseline_window=10 uses first 10 snapshots
        result = monitor_topology_cusum(
            snapshots, threshold=3.0, drift=0.5, baseline_window=10,
        )
        assert isinstance(result, HodgeCusumResult)
        assert result.mean_betti_1 >= 0.0
        # Should still detect the change
        assert result.num_topology_changes > 0


# ---------------------------------------------------------------------------
# Link failure detection
# ---------------------------------------------------------------------------

class TestLinkFailureDetection:
    def test_link_failure_detection(self):
        """detect_link_failure identifies the correct removed link."""
        before = _ring_4()
        after = _broken_ring_4()  # link (0,3) removed
        lost, gained, impact = detect_link_failure(before, after, 4)
        assert (0, 3) in lost
        assert len(gained) == 0
        # Removing a link from a ring changes betti_1
        assert impact["delta_betti_1"] == -1  # ring betti=1, path betti=0


class TestLinkFailureNoChange:
    def test_link_failure_no_change(self):
        """Same adjacency produces no lost or gained links."""
        adj = _ring_4()
        lost, gained, impact = detect_link_failure(adj, adj, 4)
        assert len(lost) == 0
        assert len(gained) == 0
        assert impact["delta_betti_1"] == 0
        assert abs(impact["delta_spectral_gap"]) < 1e-10


# ---------------------------------------------------------------------------
# Resilience score
# ---------------------------------------------------------------------------

class TestResilienceScoreStable:
    def test_resilience_score_stable(self):
        """Stable topology with positive spectral gap yields high score."""
        snapshots = [
            compute_topology_snapshot(_complete_4(), 4, time_index=i)
            for i in range(20)
        ]
        score = compute_topology_resilience_score(snapshots)
        # All identical -> min/max spectral gap = 1.0, so score = 1.0 * mean_redundancy
        # K4: betti_1=0, so routing_redundancy=0.0, thus score=0.0
        # Use ring instead: betti_1=1, spectral_gap>0, redundancy>0
        snapshots_ring = [
            compute_topology_snapshot(_ring_4(), 4, time_index=i)
            for i in range(20)
        ]
        score_ring = compute_topology_resilience_score(snapshots_ring)
        # Stable: min_sg / max_sg = 1.0, mean_redundancy = 0.25
        assert abs(score_ring - 0.25) < 1e-10


class TestResilienceScoreUnstable:
    def test_resilience_score_unstable(self):
        """Varying topology yields lower resilience score."""
        snapshots = []
        for i in range(10):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))
        for i in range(10, 20):
            snapshots.append(compute_topology_snapshot(_complete_4(), 4, time_index=i))

        score = compute_topology_resilience_score(snapshots)
        # Ring has spectral_gap > 0 and K4 has different spectral_gap
        # The min/max ratio will be < 1.0
        # Also, K4 has redundancy=0.0 which pulls mean_redundancy down
        # So score should be less than the stable ring score of 0.25
        score_stable = compute_topology_resilience_score([
            compute_topology_snapshot(_ring_4(), 4, time_index=i)
            for i in range(20)
        ])
        assert score < score_stable


# ---------------------------------------------------------------------------
# Hawkins-Olwell reset
# ---------------------------------------------------------------------------

class TestHawkinsOlwellReset:
    def test_hawkins_olwell_reset(self):
        """After detection, CUSUM resets but can detect again.

        Uses three epochs: ring baseline -> complete (first change) ->
        back to ring (return to baseline) -> complete again (second change).
        The Hawkins-Olwell reset (S = S - threshold) preserves accumulated
        evidence, allowing the second departure to be detected.
        """
        snapshots = []
        # Epoch 1: ring baseline (0-19)
        for i in range(20):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))
        # Epoch 2: complete — first departure from baseline (20-34)
        for i in range(20, 35):
            snapshots.append(compute_topology_snapshot(_complete_4(), 4, time_index=i))
        # Epoch 3: ring — return to baseline (35-49)
        for i in range(35, 50):
            snapshots.append(compute_topology_snapshot(_ring_4(), 4, time_index=i))
        # Epoch 4: complete — second departure from baseline (50-69)
        for i in range(50, 70):
            snapshots.append(compute_topology_snapshot(_complete_4(), 4, time_index=i))

        result = monitor_topology_cusum(
            snapshots, threshold=3.0, drift=0.5, baseline_window=20,
        )
        event_times = [e.time_index for e in result.events]
        # Should have events in both departure windows
        first_departure = [t for t in event_times if 20 <= t < 35]
        second_departure = [t for t in event_times if t >= 50]
        assert len(first_departure) > 0, "Should detect first topology change"
        assert len(second_departure) > 0, "Should detect second topology change after reset"


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

class TestSnapshotFrozenDataclass:
    def test_snapshot_frozen_dataclass(self):
        """Verify TopologySnapshot is immutable."""
        snap = compute_topology_snapshot(_ring_4(), 4, time_index=0)
        raised = False
        try:
            snap.betti_1 = 99
        except AttributeError:
            raised = True
        assert raised, "TopologySnapshot should be frozen"

    def test_event_frozen_dataclass(self):
        """Verify TopologyChangeEvent is immutable."""
        event = TopologyChangeEvent(
            time_index=0,
            feature_name="betti_1",
            cusum_value=6.0,
            direction="increase",
            magnitude=1.0,
        )
        raised = False
        try:
            event.cusum_value = 0.0
        except AttributeError:
            raised = True
        assert raised, "TopologyChangeEvent should be frozen"

    def test_result_frozen_dataclass(self):
        """Verify HodgeCusumResult is immutable."""
        result = HodgeCusumResult(
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
        raised = False
        try:
            result.num_topology_changes = 99
        except AttributeError:
            raised = True
        assert raised, "HodgeCusumResult should be frozen"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_snapshots(self):
        """Empty snapshot list returns empty result."""
        result = monitor_topology_cusum([])
        assert result.num_topology_changes == 0
        assert len(result.snapshots) == 0

    def test_single_snapshot(self):
        """Single snapshot produces no events."""
        snap = compute_topology_snapshot(_ring_4(), 4, time_index=0)
        result = monitor_topology_cusum([snap])
        assert result.num_topology_changes == 0

    def test_resilience_score_empty(self):
        """Empty snapshot list returns 0.0 resilience."""
        assert compute_topology_resilience_score([]) == 0.0
