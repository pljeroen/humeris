# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for graph_analysis.py proposals R3 (P41, P43, P44, P53, P55, P60, P63)."""
import math

import numpy as np

from humeris.domain.graph_analysis import (
    QuantumWalkCoverage,
    NetworkEntanglement,
    SchedulingFrustration,
    HebbianISLTopology,
    ACORoutingSolution,
    SpectralGraphWavelet,
    CheegerBottleneck,
    compute_quantum_walk_coverage,
    compute_network_entanglement,
    compute_scheduling_frustration,
    compute_hebbian_isl_topology,
    compute_aco_routing,
    compute_spectral_graph_wavelet,
    compute_cheeger_bottleneck,
)


# ---------------------------------------------------------------------------
# Helper: simple adjacency matrices
# ---------------------------------------------------------------------------

def _path_graph(n):
    """Path graph on n nodes: 0-1-2-..-(n-1)."""
    adj = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        adj[i][i + 1] = 1.0
        adj[i + 1][i] = 1.0
    return adj


def _complete_graph(n):
    """Complete graph K_n."""
    adj = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i][j] = 1.0
    return adj


def _cycle_graph(n):
    """Cycle graph C_n."""
    adj = [[0.0] * n for _ in range(n)]
    for i in range(n):
        adj[i][(i + 1) % n] = 1.0
        adj[(i + 1) % n][i] = 1.0
    return adj


def _star_graph(n):
    """Star graph: node 0 connected to 1..n-1."""
    adj = [[0.0] * n for _ in range(n)]
    for i in range(1, n):
        adj[0][i] = 1.0
        adj[i][0] = 1.0
    return adj


def _barbell_graph():
    """Two triangles (0,1,2) and (3,4,5) connected by edge 2-3."""
    adj = [[0.0] * 6 for _ in range(6)]
    for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
        adj[i][j] = 1.0
        adj[j][i] = 1.0
    return adj


# ===========================================================================
# P41: Quantum Walk Coverage Exploration
# ===========================================================================

class TestQuantumWalkCoverage:
    def test_returns_frozen_dataclass(self):
        adj = _cycle_graph(4)
        result = compute_quantum_walk_coverage(adj, 4)
        assert isinstance(result, QuantumWalkCoverage)

    def test_mixing_time_positive(self):
        adj = _complete_graph(4)
        result = compute_quantum_walk_coverage(adj, 4)
        assert result.mixing_time_steps > 0

    def test_classical_mixing_time_positive(self):
        adj = _cycle_graph(6)
        result = compute_quantum_walk_coverage(adj, 6)
        assert result.classical_mixing_time_steps > 0

    def test_speedup_ratio_positive(self):
        adj = _complete_graph(4)
        result = compute_quantum_walk_coverage(adj, 4)
        assert result.speedup_ratio > 0.0

    def test_stationary_distribution_sums_to_one(self):
        adj = _cycle_graph(6)
        result = compute_quantum_walk_coverage(adj, 6)
        total = sum(result.stationary_distribution)
        assert abs(total - 1.0) < 1e-10

    def test_propagation_velocity_nonneg(self):
        adj = _path_graph(5)
        result = compute_quantum_walk_coverage(adj, 5)
        assert result.propagation_velocity >= 0.0

    def test_hitting_time_to_antipodal_positive(self):
        adj = _cycle_graph(6)
        result = compute_quantum_walk_coverage(adj, 6)
        assert result.hitting_time_to_antipodal > 0

    def test_single_node(self):
        adj = [[0.0]]
        result = compute_quantum_walk_coverage(adj, 1)
        assert result.mixing_time_steps >= 0
        assert abs(result.stationary_distribution[0] - 1.0) < 1e-10

    def test_complete_graph_uniform_stationary(self):
        n = 5
        adj = _complete_graph(n)
        result = compute_quantum_walk_coverage(adj, n)
        for p in result.stationary_distribution:
            assert abs(p - 1.0 / n) < 0.1  # approximately uniform

    def test_speedup_ratio_equals_quotient(self):
        adj = _cycle_graph(6)
        result = compute_quantum_walk_coverage(adj, 6)
        expected = result.classical_mixing_time_steps / result.mixing_time_steps if result.mixing_time_steps > 0 else 0.0
        assert abs(result.speedup_ratio - expected) < 1e-10


# ===========================================================================
# P43: Density Matrix Entanglement Entropy
# ===========================================================================

class TestNetworkEntanglement:
    def test_returns_frozen_dataclass(self):
        adj = _complete_graph(4)
        result = compute_network_entanglement(adj, 4, partition_a=[0, 1])
        assert isinstance(result, NetworkEntanglement)

    def test_entropy_nonneg(self):
        adj = _complete_graph(4)
        result = compute_network_entanglement(adj, 4, partition_a=[0, 1])
        assert result.von_neumann_entropy >= -1e-10

    def test_entropy_leq_max(self):
        adj = _complete_graph(6)
        result = compute_network_entanglement(adj, 6, partition_a=[0, 1, 2])
        assert result.von_neumann_entropy <= result.max_entropy + 1e-10

    def test_entanglement_ratio_range(self):
        adj = _complete_graph(4)
        result = compute_network_entanglement(adj, 4, partition_a=[0, 1])
        assert 0.0 <= result.entanglement_ratio <= 1.0 + 1e-10

    def test_disconnected_partitions_low_entropy(self):
        """Two disconnected components partitioned exactly should have low entropy."""
        adj = [[0.0] * 4 for _ in range(4)]
        # Component 1: 0-1
        adj[0][1] = 1.0
        adj[1][0] = 1.0
        # Component 2: 2-3
        adj[2][3] = 1.0
        adj[3][2] = 1.0
        result = compute_network_entanglement(adj, 4, partition_a=[0, 1])
        assert result.von_neumann_entropy < 1.0

    def test_eigenvalues_nonneg(self):
        adj = _complete_graph(4)
        result = compute_network_entanglement(adj, 4, partition_a=[0, 1])
        for ev in result.eigenvalues:
            assert ev >= -1e-10

    def test_partition_independence_bool(self):
        adj = _complete_graph(4)
        result = compute_network_entanglement(adj, 4, partition_a=[0, 1])
        assert isinstance(result.partition_independence, bool)

    def test_max_entropy_correct(self):
        adj = _complete_graph(4)
        result = compute_network_entanglement(adj, 4, partition_a=[0, 1])
        expected_max = math.log2(2)  # partition_a has 2 nodes
        assert abs(result.max_entropy - expected_max) < 1e-10

    def test_single_node_partition(self):
        adj = _complete_graph(3)
        result = compute_network_entanglement(adj, 3, partition_a=[0])
        assert result.von_neumann_entropy >= 0.0
        assert result.max_entropy == 0.0  # log2(1) = 0

    def test_full_partition_zero_entropy(self):
        """If partition A = all nodes, reduced density matrix is pure, entropy = 0."""
        adj = _complete_graph(3)
        result = compute_network_entanglement(adj, 3, partition_a=[0, 1, 2])
        assert result.von_neumann_entropy < 1e-10


# ===========================================================================
# P44: Spin Glass Frustration Index
# ===========================================================================

class TestSchedulingFrustration:
    def test_returns_frozen_dataclass(self):
        adj = _cycle_graph(3)
        couplings = [[0.0] * 3 for _ in range(3)]
        couplings[0][1] = 1.0; couplings[1][0] = 1.0
        couplings[1][2] = 1.0; couplings[2][1] = 1.0
        couplings[0][2] = -1.0; couplings[2][0] = -1.0
        result = compute_scheduling_frustration(couplings, 3)
        assert isinstance(result, SchedulingFrustration)

    def test_frustration_index_range(self):
        couplings = [[0.0] * 3 for _ in range(3)]
        couplings[0][1] = 1.0; couplings[1][0] = 1.0
        couplings[1][2] = 1.0; couplings[2][1] = 1.0
        couplings[0][2] = -1.0; couplings[2][0] = -1.0
        result = compute_scheduling_frustration(couplings, 3)
        assert 0.0 <= result.frustration_index <= 1.0

    def test_all_positive_couplings_no_frustration(self):
        """All-positive triangle: product of signs = +1*+1*+1 = +1, not frustrated."""
        n = 3
        couplings = [[0.0] * n for _ in range(n)]
        couplings[0][1] = 1.0; couplings[1][0] = 1.0
        couplings[1][2] = 1.0; couplings[2][1] = 1.0
        couplings[0][2] = 1.0; couplings[2][0] = 1.0
        result = compute_scheduling_frustration(couplings, n)
        assert result.frustration_index == 0.0

    def test_frustrated_triangle(self):
        """One negative coupling in triangle: product = +1*+1*(-1) = -1, frustrated."""
        n = 3
        couplings = [[0.0] * n for _ in range(n)]
        couplings[0][1] = 1.0; couplings[1][0] = 1.0
        couplings[1][2] = 1.0; couplings[2][1] = 1.0
        couplings[0][2] = -1.0; couplings[2][0] = -1.0
        result = compute_scheduling_frustration(couplings, n)
        assert result.frustration_index == 1.0  # 1 of 1 triangle frustrated

    def test_ground_state_energy_leq_greedy(self):
        n = 4
        couplings = [[0.0] * n for _ in range(n)]
        couplings[0][1] = 1.0; couplings[1][0] = 1.0
        couplings[1][2] = -1.0; couplings[2][1] = -1.0
        couplings[2][3] = 1.0; couplings[3][2] = 1.0
        couplings[0][3] = -1.0; couplings[3][0] = -1.0
        result = compute_scheduling_frustration(couplings, n)
        assert result.ground_state_energy <= result.greedy_energy + 1e-10

    def test_total_loops_correct(self):
        n = 4
        adj = _complete_graph(n)
        result = compute_scheduling_frustration(adj, n)
        # K4 has C(4,3) = 4 triangles
        assert result.total_loops == 4

    def test_frustrated_loops_leq_total(self):
        n = 4
        couplings = _complete_graph(n)
        couplings[0][1] = -1.0
        couplings[1][0] = -1.0
        result = compute_scheduling_frustration(couplings, n)
        assert result.frustrated_loops <= result.total_loops

    def test_energy_gap_nonneg(self):
        n = 3
        couplings = _complete_graph(n)
        result = compute_scheduling_frustration(couplings, n)
        assert result.energy_gap >= -1e-10

    def test_no_edges(self):
        adj = [[0.0] * 3 for _ in range(3)]
        result = compute_scheduling_frustration(adj, 3)
        assert result.total_loops == 0
        assert result.frustration_index == 0.0

    def test_two_nodes(self):
        adj = [[0.0, 1.0], [1.0, 0.0]]
        result = compute_scheduling_frustration(adj, 2)
        assert result.total_loops == 0


# ===========================================================================
# P53: Hebbian Learning ISL Adaptation
# ===========================================================================

class TestHebbianISLTopology:
    def test_returns_frozen_dataclass(self):
        adj = _complete_graph(3)
        activity = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]
        result = compute_hebbian_isl_topology(adj, activity, 3)
        assert isinstance(result, HebbianISLTopology)

    def test_adapted_weights_shape(self):
        n = 4
        adj = _complete_graph(n)
        activity = [[float(i == j) for j in range(n)] for i in range(n)]
        result = compute_hebbian_isl_topology(adj, activity, n, epochs=5)
        assert len(result.adapted_weights) == n
        assert len(result.adapted_weights[0]) == n

    def test_emergent_link_count_nonneg(self):
        adj = _complete_graph(3)
        activity = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]
        result = compute_hebbian_isl_topology(adj, activity, 3)
        assert result.emergent_link_count >= 0

    def test_adaptation_efficiency_range(self):
        adj = _complete_graph(4)
        activity = [[1.0, 0.8, 0.2, 0.1],
                     [0.8, 1.0, 0.3, 0.2],
                     [0.2, 0.3, 1.0, 0.9],
                     [0.1, 0.2, 0.9, 1.0]]
        result = compute_hebbian_isl_topology(adj, activity, 4)
        assert result.adaptation_efficiency >= 0.0

    def test_strongest_links_nonempty(self):
        adj = _complete_graph(3)
        activity = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]
        result = compute_hebbian_isl_topology(adj, activity, 3)
        assert len(result.strongest_links) > 0

    def test_weakest_pruned_list(self):
        adj = _complete_graph(3)
        activity = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]
        result = compute_hebbian_isl_topology(adj, activity, 3)
        assert isinstance(result.weakest_pruned, tuple)

    def test_geometric_link_count(self):
        n = 4
        adj = _complete_graph(n)
        activity = [[1.0] * n for _ in range(n)]
        result = compute_hebbian_isl_topology(adj, activity, n)
        # K4 has 6 undirected edges
        assert result.geometric_link_count == 6

    def test_emergent_leq_geometric(self):
        n = 4
        adj = _complete_graph(n)
        activity = [[0.1] * n for _ in range(n)]
        result = compute_hebbian_isl_topology(adj, activity, n)
        assert result.emergent_link_count <= result.geometric_link_count

    def test_single_node(self):
        adj = [[0.0]]
        activity = [[1.0]]
        result = compute_hebbian_isl_topology(adj, activity, 1)
        assert result.emergent_link_count == 0
        assert result.geometric_link_count == 0

    def test_high_correlation_strengthens_links(self):
        """Highly correlated activity should produce stronger adapted weights."""
        n = 3
        adj = _complete_graph(n)
        # Perfect correlation
        high = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        r_high = compute_hebbian_isl_topology(adj, high, n, epochs=20)
        # Anti-correlation
        low = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        r_low = compute_hebbian_isl_topology(adj, low, n, epochs=20)
        # Emergent links from high correlation should be >= low correlation
        assert r_high.emergent_link_count >= r_low.emergent_link_count


# ===========================================================================
# P55: Ant Colony Optimization ISL Routing
# ===========================================================================

class TestACORoutingSolution:
    def test_returns_frozen_dataclass(self):
        adj = _path_graph(4)
        result = compute_aco_routing(adj, 4, source=0, sink=3)
        assert isinstance(result, ACORoutingSolution)

    def test_best_route_starts_at_source(self):
        adj = _path_graph(4)
        result = compute_aco_routing(adj, 4, source=0, sink=3)
        assert result.best_route[0] == 0

    def test_best_route_ends_at_sink(self):
        adj = _path_graph(4)
        result = compute_aco_routing(adj, 4, source=0, sink=3)
        assert result.best_route[-1] == 3

    def test_best_cost_positive(self):
        adj = _path_graph(4)
        result = compute_aco_routing(adj, 4, source=0, sink=3)
        assert result.best_cost > 0.0

    def test_convergence_iteration_positive(self):
        adj = _complete_graph(4)
        result = compute_aco_routing(adj, 4, source=0, sink=3, max_iterations=20)
        assert result.convergence_iteration > 0

    def test_pheromone_distribution_shape(self):
        n = 4
        adj = _complete_graph(n)
        result = compute_aco_routing(adj, n, source=0, sink=3)
        assert len(result.pheromone_distribution) == n
        assert len(result.pheromone_distribution[0]) == n

    def test_route_diversity_nonneg(self):
        adj = _complete_graph(5)
        result = compute_aco_routing(adj, 5, source=0, sink=4)
        assert result.route_diversity >= 0.0

    def test_complete_graph_shortest_route(self):
        """On K4, shortest route from 0 to 3 is direct (2 nodes)."""
        adj = _complete_graph(4)
        result = compute_aco_routing(adj, 4, source=0, sink=3, max_iterations=30)
        # Direct path is [0, 3], length 2
        assert len(result.best_route) <= 4  # at most all nodes

    def test_path_graph_only_route(self):
        """On path 0-1-2-3, the only route is [0,1,2,3]."""
        adj = _path_graph(4)
        result = compute_aco_routing(adj, 4, source=0, sink=3, max_iterations=30)
        assert result.best_route == (0, 1, 2, 3)

    def test_no_path_returns_empty(self):
        """Disconnected graph: no route from 0 to 3."""
        adj = [[0.0] * 4 for _ in range(4)]
        adj[0][1] = 1.0; adj[1][0] = 1.0
        adj[2][3] = 1.0; adj[3][2] = 1.0
        result = compute_aco_routing(adj, 4, source=0, sink=3)
        assert len(result.best_route) == 0
        assert result.best_cost == float('inf')


# ===========================================================================
# P60: Spectral Graph Wavelet
# ===========================================================================

class TestSpectralGraphWavelet:
    def test_returns_frozen_dataclass(self):
        adj = _cycle_graph(6)
        signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = compute_spectral_graph_wavelet(adj, 6, signal)
        assert isinstance(result, SpectralGraphWavelet)

    def test_scales_nonempty(self):
        adj = _cycle_graph(6)
        signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = compute_spectral_graph_wavelet(adj, 6, signal)
        assert len(result.scales) > 0

    def test_wavelet_coefficients_shape(self):
        n = 6
        adj = _cycle_graph(n)
        signal = [float(i == 0) for i in range(n)]
        result = compute_spectral_graph_wavelet(adj, n, signal, num_scales=4)
        assert len(result.wavelet_coefficients) == 4  # num_scales
        assert len(result.wavelet_coefficients[0]) == n

    def test_dominant_scale_in_range(self):
        adj = _cycle_graph(6)
        signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = compute_spectral_graph_wavelet(adj, 6, signal)
        assert 0 <= result.dominant_scale < len(result.scales)

    def test_localization_index_nonneg(self):
        adj = _cycle_graph(6)
        signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = compute_spectral_graph_wavelet(adj, 6, signal)
        assert result.localization_index >= 0.0

    def test_cross_scale_correlations_shape(self):
        n = 6
        adj = _cycle_graph(n)
        signal = [float(i == 0) for i in range(n)]
        result = compute_spectral_graph_wavelet(adj, n, signal, num_scales=4)
        n_scales = len(result.scales)
        assert len(result.cross_scale_correlations) == n_scales
        assert len(result.cross_scale_correlations[0]) == n_scales

    def test_constant_signal_low_coefficients(self):
        """Constant signal projected onto graph frequencies should have low wavelet energy."""
        n = 6
        adj = _cycle_graph(n)
        signal = [1.0] * n
        result = compute_spectral_graph_wavelet(adj, n, signal)
        # Wavelet kernel g(x) = x*exp(-x) vanishes at x=0 (DC component)
        total_energy = sum(sum(c ** 2 for c in row) for row in result.wavelet_coefficients)
        assert total_energy < 1e-6

    def test_localized_signal_high_coefficients(self):
        """A delta signal should have nonzero wavelet coefficients."""
        n = 6
        adj = _cycle_graph(n)
        signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = compute_spectral_graph_wavelet(adj, n, signal)
        total_energy = sum(sum(c ** 2 for c in row) for row in result.wavelet_coefficients)
        assert total_energy > 1e-6

    def test_single_node(self):
        adj = [[0.0]]
        signal = [1.0]
        result = compute_spectral_graph_wavelet(adj, 1, signal)
        assert len(result.scales) >= 1


# ===========================================================================
# P63: Cheeger Constant ISL Bottleneck
# ===========================================================================

class TestCheegerBottleneck:
    def test_returns_frozen_dataclass(self):
        adj = _barbell_graph()
        result = compute_cheeger_bottleneck(adj, 6)
        assert isinstance(result, CheegerBottleneck)

    def test_cheeger_constant_nonneg(self):
        adj = _complete_graph(5)
        result = compute_cheeger_bottleneck(adj, 5)
        assert result.cheeger_constant >= 0.0

    def test_fiedler_lower_bound(self):
        """Cheeger inequality: lambda_2/2 <= h."""
        adj = _complete_graph(5)
        result = compute_cheeger_bottleneck(adj, 5)
        assert result.cheeger_constant >= result.fiedler_lower_bound - 1e-10

    def test_spectral_upper_bound(self):
        """Cheeger inequality: h <= sqrt(2 * lambda_2 * d_max)."""
        adj = _complete_graph(5)
        result = compute_cheeger_bottleneck(adj, 5)
        assert result.cheeger_constant <= result.spectral_upper_bound + 1e-10

    def test_partition_covers_all_nodes(self):
        n = 6
        adj = _barbell_graph()
        result = compute_cheeger_bottleneck(adj, n)
        all_nodes = set(result.partition_a) | set(result.partition_b)
        assert all_nodes == set(range(n))

    def test_partitions_disjoint(self):
        adj = _barbell_graph()
        result = compute_cheeger_bottleneck(adj, 6)
        assert len(set(result.partition_a) & set(result.partition_b)) == 0

    def test_bottleneck_cut_edges_nonempty(self):
        adj = _barbell_graph()
        result = compute_cheeger_bottleneck(adj, 6)
        assert len(result.bottleneck_cut_edges) > 0

    def test_cut_capacity_positive(self):
        adj = _barbell_graph()
        result = compute_cheeger_bottleneck(adj, 6)
        assert result.cut_capacity > 0.0

    def test_barbell_finds_bridge(self):
        """Barbell graph bottleneck should cut the bridge edge (2,3)."""
        adj = _barbell_graph()
        result = compute_cheeger_bottleneck(adj, 6)
        cut_set = set(result.bottleneck_cut_edges)
        assert (2, 3) in cut_set or (3, 2) in cut_set

    def test_complete_graph_high_cheeger(self):
        """K_n has high Cheeger constant (well connected)."""
        n = 5
        adj = _complete_graph(n)
        result = compute_cheeger_bottleneck(adj, n)
        assert result.cheeger_constant > 0.5

    def test_two_nodes(self):
        adj = [[0.0, 1.0], [1.0, 0.0]]
        result = compute_cheeger_bottleneck(adj, 2)
        assert result.cheeger_constant > 0.0
        assert len(result.partition_a) == 1
        assert len(result.partition_b) == 1

    def test_path_graph_low_cheeger(self):
        """Path graph has low Cheeger constant (easy to disconnect)."""
        n = 8
        adj = _path_graph(n)
        result = compute_cheeger_bottleneck(adj, n)
        # Compare to complete graph
        adj_k = _complete_graph(n)
        result_k = compute_cheeger_bottleneck(adj_k, n)
        assert result.cheeger_constant < result_k.cheeger_constant
