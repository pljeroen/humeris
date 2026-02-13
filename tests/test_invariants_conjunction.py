# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Invariant tests for conjunction/collision probability assessment.

These verify mathematical properties of collision probability
calculations that must always hold.

Invariants H1-H5 from the formal invariant specification.
"""

import math

import pytest

from humeris.domain.conjunction import (
    collision_probability_2d,
    foster_max_collision_probability,
)


class TestH1ProbabilityBounds:
    """H1: 0 <= Pc <= 1 for all valid inputs."""

    _CASES = [
        (0.0, 0.0, 0.0, 100.0, 100.0, 10.0, "zero miss"),
        (100.0, 80.0, 60.0, 100.0, 100.0, 10.0, "near miss"),
        (1000.0, 800.0, 600.0, 100.0, 100.0, 10.0, "medium miss"),
        (50000.0, 40000.0, 30000.0, 100.0, 100.0, 10.0, "far miss"),
        (10.0, 5.0, 5.0, 1.0, 1.0, 10.0, "tight sigma"),
        (10.0, 5.0, 5.0, 10000.0, 10000.0, 10.0, "loose sigma"),
    ]

    @pytest.mark.parametrize("miss,br,bc,sr,sc,radius,label", _CASES)
    def test_bounds(self, miss, br, bc, sr, sc, radius, label):
        pc = collision_probability_2d(miss, br, bc, sr, sc, radius)
        assert 0.0 <= pc <= 1.0, f"{label}: Pc={pc} out of [0,1]"


class TestH2Symmetry:
    """H2: Pc(A,B) == Pc(B,A) — swapping objects doesn't change Pc.

    For the 2D probability integral, symmetry means negating the
    B-plane components should give the same result (the distribution
    is symmetric around the origin with respect to sign flip).
    """

    def test_symmetry_sign_flip(self):
        """Negating b_radial and b_cross gives same Pc."""
        pc1 = collision_probability_2d(
            miss_distance_m=500.0,
            b_radial_m=300.0, b_cross_m=200.0,
            sigma_radial_m=100.0, sigma_cross_m=100.0,
            combined_radius_m=10.0,
        )
        pc2 = collision_probability_2d(
            miss_distance_m=500.0,
            b_radial_m=-300.0, b_cross_m=-200.0,
            sigma_radial_m=100.0, sigma_cross_m=100.0,
            combined_radius_m=10.0,
        )
        assert abs(pc1 - pc2) < 1e-10, f"Pc={pc1} vs Pc_swapped={pc2}"


class TestH3MonotonicityWithMissDistance:
    """H3: Increasing miss distance must not increase Pc (fixed covariance/radius)."""

    def test_miss_distance_monotonicity(self):
        sigma = 100.0
        radius = 10.0
        distances = [0, 10, 50, 100, 200, 500, 1000, 5000]
        prev_pc = 1.0
        for d in distances:
            pc = collision_probability_2d(
                miss_distance_m=float(d),
                b_radial_m=float(d), b_cross_m=0.0,
                sigma_radial_m=sigma, sigma_cross_m=sigma,
                combined_radius_m=radius,
            )
            assert pc <= prev_pc + 1e-10, \
                f"dist={d}: Pc={pc} > prev Pc={prev_pc}"
            prev_pc = pc


class TestH4RadiusMonotonicity:
    """H4: Increasing combined hard-body radius must not decrease Pc."""

    def test_radius_monotonicity(self):
        radii = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        prev_pc = 0.0
        for r in radii:
            pc = collision_probability_2d(
                miss_distance_m=100.0,
                b_radial_m=50.0, b_cross_m=50.0,
                sigma_radial_m=100.0, sigma_cross_m=100.0,
                combined_radius_m=r,
            )
            assert pc >= prev_pc - 1e-10, \
                f"radius={r}: Pc={pc} < prev Pc={prev_pc}"
            prev_pc = pc


class TestH5DegenerateCovarianceHandling:
    """H5: Zero/negative sigma produces Pc=0 (not NaN or crash)."""

    def test_zero_sigma_radial(self):
        pc = collision_probability_2d(
            miss_distance_m=100.0,
            b_radial_m=50.0, b_cross_m=50.0,
            sigma_radial_m=0.0, sigma_cross_m=100.0,
            combined_radius_m=10.0,
        )
        assert pc == 0.0

    def test_zero_sigma_cross(self):
        pc = collision_probability_2d(
            miss_distance_m=100.0,
            b_radial_m=50.0, b_cross_m=50.0,
            sigma_radial_m=100.0, sigma_cross_m=0.0,
            combined_radius_m=10.0,
        )
        assert pc == 0.0

    def test_negative_sigma(self):
        pc = collision_probability_2d(
            miss_distance_m=100.0,
            b_radial_m=50.0, b_cross_m=50.0,
            sigma_radial_m=-10.0, sigma_cross_m=100.0,
            combined_radius_m=10.0,
        )
        assert pc == 0.0


class TestFosterMaxPcInvariants:
    """Additional invariants for foster_max_collision_probability."""

    def test_bounds(self):
        """Result is non-negative."""
        pc = foster_max_collision_probability(500.0, 10.0, 10000.0)
        assert pc >= 0.0

    def test_radius_monotonicity(self):
        """Larger radius -> larger max Pc (Pc_max proportional to r^2)."""
        pc_small = foster_max_collision_probability(500.0, 5.0, 10000.0)
        pc_large = foster_max_collision_probability(500.0, 20.0, 10000.0)
        assert pc_large > pc_small

    def test_covariance_monotonicity(self):
        """Larger covariance -> smaller max Pc."""
        pc_small_cov = foster_max_collision_probability(500.0, 10.0, 1000.0)
        pc_large_cov = foster_max_collision_probability(500.0, 10.0, 100000.0)
        assert pc_large_cov < pc_small_cov
