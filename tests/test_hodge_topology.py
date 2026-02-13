# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/graph_analysis.py — Hodge Laplacian topology analysis."""
import ast
import math

import numpy as np

from humeris.domain.graph_analysis import (
    HodgeTopology,
    compute_hodge_topology,
)


def _complete_graph(n):
    """Build adjacency matrix for complete graph K_n."""
    adj = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i][j] = 1.0
    return adj


def _tree_graph_4():
    """Build adjacency for a tree: 0-1, 1-2, 1-3 (star with center 1).
    3 edges, 0 triangles."""
    adj = [[0.0] * 4 for _ in range(4)]
    edges = [(0, 1), (1, 2), (1, 3)]
    for i, j in edges:
        adj[i][j] = 1.0
        adj[j][i] = 1.0
    return adj


def _single_triangle():
    """Build adjacency for a single triangle (K3): 3 nodes, 3 edges, 1 triangle."""
    return _complete_graph(3)


def _two_disconnected_triangles():
    """Two separate triangles: 6 nodes, 6 edges, 2 triangles, no shared edges."""
    adj = [[0.0] * 6 for _ in range(6)]
    # Triangle 1: nodes 0, 1, 2
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        adj[i][j] = 1.0
        adj[j][i] = 1.0
    # Triangle 2: nodes 3, 4, 5
    for i, j in [(3, 4), (4, 5), (3, 5)]:
        adj[i][j] = 1.0
        adj[j][i] = 1.0
    return adj


def _two_triangles_shared_edge():
    """Two triangles sharing edge (1,2): nodes 0,1,2,3.
    Edges: 0-1, 0-2, 1-2, 1-3, 2-3 => 5 edges, 2 triangles.
    beta_1 = edges - nodes + components - triangles_that_fill_cycles
    In simplicial homology: beta_1 = dim(ker(B1^T)) - dim(im(B2)).
    For this graph: 2 triangles fill both potential cycles, so beta_1 = 0."""
    adj = [[0.0] * 4 for _ in range(4)]
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        adj[i][j] = 1.0
        adj[j][i] = 1.0
    return adj


class TestHodgeTopologyK4:
    """K4: 4 nodes, 6 edges, 4 triangles (2-skeleton of the tetrahedron).

    Simplicial homology: dim C_0=4, dim C_1=6, dim C_2=4.
    rank(B1) = 3 (connected on 4 nodes), so dim(ker B1^T) = 6 - 3 = 3.
    rank(B2) = 3 (4 triangles with one dependency: boundary of tetrahedron = 0).
    beta_1 = dim(ker B1^T) - dim(im B2) = 3 - 3 = 0.

    All cycles in K4 are filled by triangles, so there are no harmonic 1-forms.
    """

    def test_returns_hodge_topology_type(self):
        result = compute_hodge_topology(_complete_graph(4), n_nodes=4)
        assert isinstance(result, HodgeTopology)

    def test_k4_betti_1(self):
        """K4 should have beta_1 as specified."""
        result = compute_hodge_topology(_complete_graph(4), n_nodes=4)
        # K4 simplicial complex (with all triangles): beta_1 = 0
        # The Hodge Laplacian L1 = B1 B1^T + B2^T B2 captures simplicial homology.
        # With 4 triangles filling all cycles except the "hollow tetrahedron" cycle,
        # beta_1 = 0 for the 2-skeleton of K4.
        assert result.betti_1 == 0

    def test_k4_triangle_count(self):
        result = compute_hodge_topology(_complete_graph(4), n_nodes=4)
        assert result.triangle_count == 4

    def test_k4_spectral_gap_positive(self):
        result = compute_hodge_topology(_complete_graph(4), n_nodes=4)
        assert result.l1_spectral_gap > 0.0

    def test_k4_routing_redundancy(self):
        result = compute_hodge_topology(_complete_graph(4), n_nodes=4)
        # beta_1 = 0, so routing_redundancy = 0/6 = 0
        assert result.routing_redundancy == 0.0


class TestHodgeTopologyTree:
    """Tree graph: no cycles, no triangles.
    L1 = B1 B1^T (since B2 is empty).
    For a tree, beta_1 = 0 (no cycles at all).
    All eigenvalues of L1 should be positive (no harmonic 1-forms on a tree)."""

    def test_tree_betti_1_zero(self):
        result = compute_hodge_topology(_tree_graph_4(), n_nodes=4)
        assert result.betti_1 == 0

    def test_tree_no_triangles(self):
        result = compute_hodge_topology(_tree_graph_4(), n_nodes=4)
        assert result.triangle_count == 0

    def test_tree_spectral_gap_positive(self):
        """Tree has no cycles so L1 = B1 B1^T has all positive eigenvalues."""
        result = compute_hodge_topology(_tree_graph_4(), n_nodes=4)
        assert result.l1_spectral_gap > 0.0

    def test_tree_routing_redundancy_zero(self):
        result = compute_hodge_topology(_tree_graph_4(), n_nodes=4)
        assert result.routing_redundancy == 0.0


class TestHodgeTopologySingleTriangle:
    """Single triangle (K3): 3 nodes, 3 edges, 1 triangle.
    The single cycle is filled by the triangle, making beta_1 = 0
    in simplicial homology."""

    def test_single_triangle_betti_1_zero(self):
        result = compute_hodge_topology(_single_triangle(), n_nodes=3)
        assert result.betti_1 == 0

    def test_single_triangle_count(self):
        result = compute_hodge_topology(_single_triangle(), n_nodes=3)
        assert result.triangle_count == 1

    def test_single_triangle_spectral_gap_positive(self):
        result = compute_hodge_topology(_single_triangle(), n_nodes=3)
        assert result.l1_spectral_gap > 0.0


class TestHodgeTopologyDisconnectedTriangles:
    """Two disconnected triangles: 6 nodes, 6 edges, 2 triangles.
    Each triangle's cycle is filled, so beta_1 = 0."""

    def test_disconnected_triangles_betti_1(self):
        result = compute_hodge_topology(_two_disconnected_triangles(), n_nodes=6)
        assert result.betti_1 == 0

    def test_disconnected_triangles_count(self):
        result = compute_hodge_topology(_two_disconnected_triangles(), n_nodes=6)
        assert result.triangle_count == 2


class TestHodgeTopologySharedEdge:
    """Two triangles sharing an edge: 4 nodes, 5 edges, 2 triangles.
    Both cycles are filled by triangles, so beta_1 = 0."""

    def test_shared_edge_betti_1(self):
        result = compute_hodge_topology(_two_triangles_shared_edge(), n_nodes=4)
        assert result.betti_1 == 0

    def test_shared_edge_triangle_count(self):
        result = compute_hodge_topology(_two_triangles_shared_edge(), n_nodes=4)
        assert result.triangle_count == 2


class TestHodgeTopologyEmptyGraph:
    """Empty graph with no edges should return zeros gracefully."""

    def test_empty_returns_zeros(self):
        adj = [[0.0] * 3 for _ in range(3)]
        result = compute_hodge_topology(adj, n_nodes=3)
        assert result.betti_1 == 0
        assert result.l1_spectral_gap == 0.0
        assert result.triangle_count == 0
        assert result.routing_redundancy == 0.0
        assert result.l1_smallest_nonzero == 0.0

    def test_single_node(self):
        adj = [[0.0]]
        result = compute_hodge_topology(adj, n_nodes=1)
        assert result.betti_1 == 0
        assert result.triangle_count == 0


class TestHodgeTopologyL1Symmetry:
    """Verify L1 is symmetric by checking eigenvalues are real (which they
    will be if np.linalg.eigh is used on a symmetric matrix)."""

    def test_k4_l1_smallest_nonzero_positive(self):
        result = compute_hodge_topology(_complete_graph(4), n_nodes=4)
        assert result.l1_smallest_nonzero > 0.0

    def test_tree_l1_smallest_nonzero_positive(self):
        result = compute_hodge_topology(_tree_graph_4(), n_nodes=4)
        assert result.l1_smallest_nonzero > 0.0


class TestHodgeTopologyCycleGraph:
    """A square (cycle graph C4): 4 nodes, 4 edges forming a cycle, no diagonals.
    No triangles exist, so B2 is empty. L1 = B1 B1^T only.
    beta_1 = E - V + components = 4 - 4 + 1 = 1 (one independent cycle, unfilled).
    """

    def _cycle_graph_4(self):
        adj = [[0.0] * 4 for _ in range(4)]
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        for i, j in edges:
            adj[i][j] = 1.0
            adj[j][i] = 1.0
        return adj

    def test_cycle_betti_1(self):
        result = compute_hodge_topology(self._cycle_graph_4(), n_nodes=4)
        assert result.betti_1 == 1

    def test_cycle_no_triangles(self):
        result = compute_hodge_topology(self._cycle_graph_4(), n_nodes=4)
        assert result.triangle_count == 0

    def test_cycle_routing_redundancy(self):
        result = compute_hodge_topology(self._cycle_graph_4(), n_nodes=4)
        # beta_1 = 1, edges = 4
        assert abs(result.routing_redundancy - 1.0 / 4.0) < 1e-10


class TestGraphAnalysisPurityExtended:
    """Ensure graph_analysis module still passes purity after Hodge additions."""

    def test_module_pure(self):
        import humeris.domain.graph_analysis as mod
        source = ast.parse(open(mod.__file__).read())
        for node in ast.walk(source):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                else:
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                assert top in {"math", "dataclasses", "datetime", "typing", "enum", "numpy", "humeris", "__future__", "collections"}, (
                    f"Forbidden import: {top}"
                )
