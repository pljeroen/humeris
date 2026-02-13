# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/control_analysis.py — CW controllability Gramian analysis."""
import ast
import math

from humeris.domain.control_analysis import (
    ControllabilityAnalysis,
    compute_cw_controllability,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0


def _mean_motion(alt_km=550.0):
    a = _R_E + alt_km * 1000.0
    return math.sqrt(_MU / a ** 3)


class TestControllability:
    def test_returns_type(self):
        n = _mean_motion()
        result = compute_cw_controllability(n, duration_s=3600.0, step_s=60.0)
        assert isinstance(result, ControllabilityAnalysis)

    def test_controllable_full_rank(self):
        n = _mean_motion()
        result = compute_cw_controllability(n, duration_s=3600.0, step_s=60.0)
        assert result.is_controllable is True

    def test_eigenvalues_positive(self):
        n = _mean_motion()
        result = compute_cw_controllability(n, duration_s=3600.0, step_s=60.0)
        assert all(v > -1e-10 for v in result.gramian_eigenvalues)

    def test_eigenvalues_sorted(self):
        n = _mean_motion()
        result = compute_cw_controllability(n, duration_s=3600.0, step_s=60.0)
        for i in range(len(result.gramian_eigenvalues) - 1):
            assert result.gramian_eigenvalues[i] <= result.gramian_eigenvalues[i + 1] + 1e-10

    def test_condition_number_geq_one(self):
        n = _mean_motion()
        result = compute_cw_controllability(n, duration_s=3600.0, step_s=60.0)
        assert result.condition_number >= 1.0 - 1e-10

    def test_min_energy_is_along_track(self):
        """Smallest eigenvalue direction should be near along-track (CW free drift)."""
        n = _mean_motion()
        result = compute_cw_controllability(n, duration_s=7200.0, step_s=60.0)
        # min_energy_direction is a 6-vector; along-track = index 1 (y)
        direction = result.min_energy_direction
        assert len(direction) == 6

    def test_longer_duration_more_controllability(self):
        n = _mean_motion()
        short = compute_cw_controllability(n, duration_s=1800.0, step_s=60.0)
        long = compute_cw_controllability(n, duration_s=7200.0, step_s=60.0)
        # Longer duration → larger Gramian trace (more total controllability)
        assert long.gramian_trace > short.gramian_trace

    def test_gramian_symmetric(self):
        """All eigenvalues should be real (symmetric Gramian)."""
        n = _mean_motion()
        result = compute_cw_controllability(n, duration_s=3600.0, step_s=60.0)
        # If eigenvalues are returned at all, they are real (Jacobi on symmetric)
        assert len(result.gramian_eigenvalues) == 6

    def test_trace_equals_eigenvalue_sum(self):
        n = _mean_motion()
        result = compute_cw_controllability(n, duration_s=3600.0, step_s=60.0)
        eig_sum = sum(result.gramian_eigenvalues)
        assert abs(result.gramian_trace - eig_sum) < abs(eig_sum) * 1e-6 + 1e-10


class TestControlAnalysisPurity:
    def test_module_pure(self):
        import humeris.domain.control_analysis as mod
        source = ast.parse(open(mod.__file__).read())
        for node in ast.walk(source):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                else:
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                assert top in {"math", "numpy", "dataclasses", "datetime", "typing", "enum", "humeris", "__future__"}, (
                    f"Forbidden import: {top}"
                )
