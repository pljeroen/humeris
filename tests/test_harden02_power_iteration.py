# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""H02-R03: Power iteration convergence diagnostic tests.

Verifies that _power_iteration returns convergence information.
"""
import numpy as np
import pytest


class TestPowerIterationConvergence:
    """Power iteration must return (vector, converged, iterations) tuple."""

    def test_returns_tuple_of_three(self):
        """_power_iteration must return (vector, converged, iterations)."""
        from humeris.domain.kessler_heatmap import _power_iteration
        matrix = np.eye(3)
        result = _power_iteration(matrix)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        vec, converged, iters = result
        assert isinstance(vec, np.ndarray)
        assert isinstance(converged, bool)
        assert isinstance(iters, int)

    def test_identity_matrix_converges_fast(self):
        """Identity matrix should converge in very few iterations."""
        from humeris.domain.kessler_heatmap import _power_iteration
        matrix = np.eye(5)
        vec, converged, iters = _power_iteration(matrix)
        assert converged is True
        assert iters <= 2, f"Identity matrix should converge in ≤2 iters, took {iters}"
        assert len(vec) == 5
        assert abs(np.sum(vec) - 1.0) < 1e-10, "Vector should be normalized to sum=1"

    def test_diagonal_matrix_distinct_eigenvalues(self):
        """Diagonal matrix with distinct eigenvalues should converge."""
        from humeris.domain.kessler_heatmap import _power_iteration
        matrix = np.diag([5.0, 3.0, 1.0, 0.5])
        vec, converged, iters = _power_iteration(matrix)
        assert converged is True
        # Dominant eigenvector should point toward first element
        assert vec[0] > vec[1] > vec[2] > vec[3], (
            f"Expected dominant eigenvector to weight first element highest, got {vec}"
        )

    def test_nearly_degenerate_eigenvalues_slow_convergence(self):
        """Matrix with eigenvalue ratio > 0.99 may not converge in 100 iters."""
        from humeris.domain.kessler_heatmap import _power_iteration
        # Two eigenvalues very close: 1.0 and 0.999
        matrix = np.diag([1.0, 0.999, 0.5])
        vec, converged, iters = _power_iteration(matrix, max_iter=100, tol=1e-10)
        # With spectral gap of 0.001, convergence is slow
        # The key test: we get the converged flag honestly
        assert isinstance(converged, bool)
        assert isinstance(iters, int)
        assert iters <= 100

    def test_zero_matrix_returns_uniform(self):
        """Zero matrix should return uniform vector, converged=True."""
        from humeris.domain.kessler_heatmap import _power_iteration
        matrix = np.zeros((4, 4))
        vec, converged, iters = _power_iteration(matrix)
        assert converged is True
        expected = np.ones(4) / 4.0
        np.testing.assert_allclose(vec, expected, atol=1e-10)

    def test_empty_matrix(self):
        """Empty matrix should return empty vector."""
        from humeris.domain.kessler_heatmap import _power_iteration
        matrix = np.zeros((0, 0))
        vec, converged, iters = _power_iteration(matrix)
        assert len(vec) == 0
        assert converged is True

    def test_callers_still_work(self):
        """compute_kessler_heatmap() should still produce valid output."""
        from humeris.domain.kessler_heatmap import (
            compute_kessler_heatmap,
            KesslerHeatMap,
        )
        from humeris.domain.propagation import OrbitalState
        from humeris.domain.orbital_mechanics import OrbitalConstants
        from datetime import datetime, timezone
        import math

        # Create minimal OrbitalState objects
        a = OrbitalConstants.R_EARTH + 550_000.0
        n = math.sqrt(OrbitalConstants.MU_EARTH / a**3)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        states = []
        for i in range(5):
            states.append(OrbitalState(
                semi_major_axis_m=a,
                eccentricity=0.0,
                inclination_rad=math.radians(53.0),
                raan_rad=0.0,
                arg_perigee_rad=0.0,
                true_anomaly_rad=math.radians(i * 72.0),
                mean_motion_rad_s=n,
                reference_epoch=epoch,
            ))

        heatmap = compute_kessler_heatmap(states)
        assert isinstance(heatmap, KesslerHeatMap)
        assert len(heatmap.cells) > 0
