# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for conjunction screening, TCA refinement, and collision probability."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.propagation import OrbitalState


# ── Helpers ──────────────────────────────────────────────────────────

EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _circular_state(altitude_km, inclination_deg, raan_deg=0.0, nu_deg=0.0):
    """Create a circular OrbitalState at given altitude."""
    a = OrbitalConstants.R_EARTH + altitude_km * 1000.0
    n = math.sqrt(OrbitalConstants.MU_EARTH / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a,
        eccentricity=0.0,
        inclination_rad=math.radians(inclination_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=0.0,
        true_anomaly_rad=math.radians(nu_deg),
        mean_motion_rad_s=n,
        reference_epoch=EPOCH,
    )


# ── PositionCovariance / ConjunctionEvent ────────────────────────────

class TestPositionCovariance:

    def test_frozen(self):
        """PositionCovariance is immutable."""
        from constellation_generator.domain.conjunction import PositionCovariance

        cov = PositionCovariance(
            sigma_xx=100.0, sigma_yy=100.0, sigma_zz=100.0,
            sigma_xy=0.0, sigma_xz=0.0, sigma_yz=0.0,
        )
        with pytest.raises(AttributeError):
            cov.sigma_xx = 200.0

    def test_fields(self):
        """PositionCovariance has all 6 upper-triangle fields."""
        from constellation_generator.domain.conjunction import PositionCovariance

        cov = PositionCovariance(
            sigma_xx=1.0, sigma_yy=2.0, sigma_zz=3.0,
            sigma_xy=0.1, sigma_xz=0.2, sigma_yz=0.3,
        )
        assert cov.sigma_xx == 1.0
        assert cov.sigma_yz == 0.3


class TestConjunctionEvent:

    def test_frozen(self):
        """ConjunctionEvent is immutable."""
        from constellation_generator.domain.conjunction import ConjunctionEvent

        evt = ConjunctionEvent(
            sat1_name="A", sat2_name="B", tca=EPOCH,
            miss_distance_m=1000.0, relative_velocity_ms=14000.0,
            collision_probability=0.0, max_collision_probability=1e-5,
            b_plane_radial_m=500.0, b_plane_cross_track_m=800.0,
        )
        with pytest.raises(AttributeError):
            evt.miss_distance_m = 500.0


# ── screen_conjunctions ─────────────────────────────────────────────

class TestScreenConjunctions:

    def test_close_offset_detected(self):
        """Two satellites in same plane with small offset are detected."""
        from constellation_generator.domain.conjunction import screen_conjunctions

        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=0.5)
        results = screen_conjunctions(
            [s1, s2], ["Sat-1", "Sat-2"],
            EPOCH, timedelta(minutes=10), timedelta(seconds=10),
            distance_threshold_m=200_000,
        )
        assert len(results) > 0
        # All returned distances should be within threshold
        for _, _, _, dist in results:
            assert dist <= 200_000

    def test_well_separated_empty(self):
        """Satellites at very different altitudes → no conjunctions within tight threshold."""
        from constellation_generator.domain.conjunction import screen_conjunctions

        s1 = _circular_state(400, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(800, 53, raan_deg=90, nu_deg=90)
        results = screen_conjunctions(
            [s1, s2], ["Sat-1", "Sat-2"],
            EPOCH, timedelta(minutes=10), timedelta(seconds=10),
            distance_threshold_m=1_000,  # very tight
        )
        assert len(results) == 0

    def test_mismatched_lengths_raises(self):
        """Mismatched state/name lengths raises ValueError."""
        from constellation_generator.domain.conjunction import screen_conjunctions

        s1 = _circular_state(550, 53)
        with pytest.raises(ValueError):
            screen_conjunctions(
                [s1], ["A", "B"],
                EPOCH, timedelta(hours=1), timedelta(seconds=30),
            )

    def test_negative_step_raises(self):
        """Negative step raises ValueError."""
        from constellation_generator.domain.conjunction import screen_conjunctions

        s1 = _circular_state(550, 53)
        s2 = _circular_state(550, 53, nu_deg=1)
        with pytest.raises(ValueError):
            screen_conjunctions(
                [s1, s2], ["A", "B"],
                EPOCH, timedelta(hours=1), timedelta(seconds=-10),
            )


# ── refine_tca ───────────────────────────────────────────────────────

class TestRefineTca:

    def test_refined_distance_less_than_coarse(self):
        """Refined miss distance ≤ coarse distance at guess time."""
        from constellation_generator.domain.conjunction import refine_tca
        from constellation_generator.domain.propagation import propagate_to

        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=0.5)

        t_guess = EPOCH + timedelta(minutes=5)
        p1, _ = propagate_to(s1, t_guess)
        p2, _ = propagate_to(s2, t_guess)
        coarse_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

        tca, miss_dist, _ = refine_tca(s1, s2, t_guess)
        assert miss_dist <= coarse_dist + 1.0  # allow 1m numerical tolerance

    def test_tca_within_search_window(self):
        """Refined TCA is within the search window of guess."""
        from constellation_generator.domain.conjunction import refine_tca

        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=0.5)
        t_guess = EPOCH + timedelta(minutes=5)

        tca, _, _ = refine_tca(s1, s2, t_guess, search_window_s=300.0)
        assert abs((tca - t_guess).total_seconds()) <= 300.0


# ── compute_b_plane ──────────────────────────────────────────────────

class TestBPlane:

    def test_known_geometry(self):
        """B-plane components are finite for non-trivial relative geometry."""
        from constellation_generator.domain.conjunction import compute_b_plane

        # Two sats at same altitude, different RAAN → crossing geometry
        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=45)
        s2 = _circular_state(550, 53, raan_deg=1, nu_deg=45)

        from constellation_generator.domain.propagation import propagate_to
        t = EPOCH
        p1, v1 = propagate_to(s1, t)
        p2, v2 = propagate_to(s2, t)

        b_r, b_ct = compute_b_plane(p1, v1, p2, v2)
        assert math.isfinite(b_r)
        assert math.isfinite(b_ct)


# ── foster_max_collision_probability ─────────────────────────────────

class TestFosterMaxPc:

    def test_positive_for_reasonable_inputs(self):
        """Foster max Pc is positive for reasonable miss/covariance."""
        from constellation_generator.domain.conjunction import foster_max_collision_probability

        pc = foster_max_collision_probability(
            miss_distance_m=500.0, combined_radius_m=10.0,
            combined_covariance_trace_m2=10000.0,
        )
        assert pc > 0

    def test_larger_covariance_smaller_pc(self):
        """Larger covariance → smaller max Pc (more uncertainty spreads risk)."""
        from constellation_generator.domain.conjunction import foster_max_collision_probability

        pc_small_cov = foster_max_collision_probability(100.0, 10.0, 1000.0)
        pc_large_cov = foster_max_collision_probability(100.0, 10.0, 100000.0)
        assert pc_large_cov < pc_small_cov

    def test_zero_covariance_returns_zero(self):
        """Zero combined covariance → Pc = 0."""
        from constellation_generator.domain.conjunction import foster_max_collision_probability

        assert foster_max_collision_probability(100.0, 10.0, 0.0) == 0.0


# ── collision_probability_2d ─────────────────────────────────────────

class TestCollisionProbability2D:

    def test_zero_miss_high_pc(self):
        """Zero miss distance + small sigmas → Pc near 1."""
        from constellation_generator.domain.conjunction import collision_probability_2d

        pc = collision_probability_2d(
            miss_distance_m=0.0, b_radial_m=0.0, b_cross_m=0.0,
            sigma_radial_m=5.0, sigma_cross_m=5.0,
            combined_radius_m=10.0, num_steps=200,
        )
        assert pc > 0.5

    def test_large_miss_low_pc(self):
        """Large miss distance → Pc near 0."""
        from constellation_generator.domain.conjunction import collision_probability_2d

        pc = collision_probability_2d(
            miss_distance_m=50000.0, b_radial_m=50000.0, b_cross_m=0.0,
            sigma_radial_m=100.0, sigma_cross_m=100.0,
            combined_radius_m=10.0,
        )
        assert pc < 1e-6


# ── assess_conjunction ───────────────────────────────────────────────

class TestAssessConjunction:

    def test_with_covariance_pc_positive(self):
        """Assessment with covariance produces Pc > 0."""
        from constellation_generator.domain.conjunction import (
            assess_conjunction, PositionCovariance,
        )

        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=0.5)

        cov = PositionCovariance(
            sigma_xx=10000.0, sigma_yy=10000.0, sigma_zz=10000.0,
            sigma_xy=0.0, sigma_xz=0.0, sigma_yz=0.0,
        )
        event = assess_conjunction(
            s1, "Sat-1", s2, "Sat-2",
            EPOCH, combined_radius_m=10.0, cov1=cov, cov2=cov,
        )
        assert event.max_collision_probability > 0

    def test_without_covariance_pc_zero(self):
        """Assessment without covariance → Pc = 0."""
        from constellation_generator.domain.conjunction import assess_conjunction

        s1 = _circular_state(550, 53, raan_deg=0, nu_deg=0)
        s2 = _circular_state(550, 53, raan_deg=0, nu_deg=0.5)

        event = assess_conjunction(s1, "Sat-1", s2, "Sat-2", EPOCH)
        assert event.collision_probability == 0.0
        assert event.max_collision_probability == 0.0


# ── Domain purity ───────────────────────────────────────────────────

class TestScreenConjunctionsNumerical:
    """Tests for screen_conjunctions_numerical."""

    def _make_numerical_results(self, altitude_offsets_km, nu_offsets_deg):
        """Create numerical propagation results for testing.

        Each result is for a satellite at 550 km + offset, 53 deg inclination.
        """
        from constellation_generator import derive_orbital_state, propagate_numerical
        from constellation_generator.domain.numerical_propagation import TwoBodyGravity
        from constellation_generator.domain.constellation import (
            ShellConfig, generate_walker_shell,
        )

        results = []
        for alt_off, nu_off in zip(altitude_offsets_km, nu_offsets_deg):
            s = _circular_state(550 + alt_off, 53, raan_deg=0, nu_deg=nu_off)
            result = propagate_numerical(
                s, timedelta(minutes=10), timedelta(seconds=60),
                [TwoBodyGravity()], epoch=EPOCH,
            )
            results.append(result)
        return results

    def test_close_offset_detected(self):
        from constellation_generator.domain.conjunction import screen_conjunctions_numerical
        results = self._make_numerical_results([0, 0], [0, 0.5])
        events = screen_conjunctions_numerical(results, ["A", "B"], distance_threshold_m=200_000)
        assert len(events) > 0

    def test_well_separated_empty(self):
        from constellation_generator.domain.conjunction import screen_conjunctions_numerical
        results = self._make_numerical_results([0, 400], [0, 90])
        events = screen_conjunctions_numerical(results, ["A", "B"], distance_threshold_m=1_000)
        assert len(events) == 0

    def test_mismatched_lengths_raises(self):
        from constellation_generator.domain.conjunction import screen_conjunctions_numerical
        results = self._make_numerical_results([0], [0])
        with pytest.raises(ValueError):
            screen_conjunctions_numerical(results, ["A", "B"])

    def test_empty_results_raises(self):
        from constellation_generator.domain.conjunction import screen_conjunctions_numerical
        with pytest.raises(ValueError):
            screen_conjunctions_numerical([], [])

    def test_sorted_by_distance(self):
        from constellation_generator.domain.conjunction import screen_conjunctions_numerical
        results = self._make_numerical_results([0, 0, 0], [0, 0.3, 0.6])
        events = screen_conjunctions_numerical(
            results, ["A", "B", "C"], distance_threshold_m=500_000,
        )
        if len(events) > 1:
            dists = [e[3] for e in events]
            assert dists == sorted(dists)

    def test_all_distances_within_threshold(self):
        from constellation_generator.domain.conjunction import screen_conjunctions_numerical
        threshold = 200_000.0
        results = self._make_numerical_results([0, 0], [0, 0.5])
        events = screen_conjunctions_numerical(results, ["A", "B"], distance_threshold_m=threshold)
        for _, _, _, dist in events:
            assert dist <= threshold

    def test_single_satellite_no_conjunctions(self):
        from constellation_generator.domain.conjunction import screen_conjunctions_numerical
        results = self._make_numerical_results([0], [0])
        events = screen_conjunctions_numerical(results, ["A"])
        assert len(events) == 0


class TestConjunctionPurity:

    def test_conjunction_module_pure(self):
        """conjunction.py must only import stdlib modules."""
        import constellation_generator.domain.conjunction as mod

        allowed = {'math', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('constellation_generator'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'constellation_generator':
                        assert False, f"Disallowed import from '{node.module}'"
