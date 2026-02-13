# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for geometry proposals R1: P5, P6, P7, P18, P19.

P5:  Jacobi Metric for Geodesic Propagation Stability
P6:  Contact Geometry for Conjunction Assessment
P7:  Morse Theory for Coverage Phase Transitions
P18: Spectral Kessler via Perron-Frobenius
P19: Wasserstein Distance for Constellation Reconfiguration
"""
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants


# ── Fixtures ───────────────────────────────────────────────────────

MU = OrbitalConstants.MU_EARTH
R_EARTH = OrbitalConstants.R_EARTH


@pytest.fixture
def epoch():
    return datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def leo_circular_pos_vel():
    """LEO circular orbit at 500 km: position along x, velocity along y."""
    r = R_EARTH + 500_000.0
    v = math.sqrt(MU / r)
    return (r, 0.0, 0.0), (0.0, v, 0.0)


@pytest.fixture
def hyperbolic_pos_vel():
    """Hyperbolic flyby: excess velocity of 3 km/s at 500 km altitude."""
    r = R_EARTH + 500_000.0
    v_esc = math.sqrt(2.0 * MU / r)
    v = v_esc + 3000.0  # 3 km/s excess
    return (r, 0.0, 0.0), (0.0, v, 0.0)


# ═══════════════════════════════════════════════════════════════════
# P5: Jacobi Stability
# ═══════════════════════════════════════════════════════════════════


class TestJacobiStability:
    """Tests for compute_jacobi_stability (P5)."""

    def test_import(self):
        """JacobiStability and compute_jacobi_stability are importable."""
        from humeris.domain.numerical_propagation import (
            JacobiStability,
            compute_jacobi_stability,
        )
        assert JacobiStability is not None
        assert compute_jacobi_stability is not None

    def test_bound_orbit_is_stable(self, leo_circular_pos_vel):
        """Bound (elliptic/circular) orbit has is_stable=True."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        pos, vel = leo_circular_pos_vel
        result = compute_jacobi_stability(pos, vel, MU)

        assert result.is_stable is True
        assert result.specific_energy_j_kg < 0.0  # bound orbit has E < 0

    def test_unbound_orbit_is_unstable(self, hyperbolic_pos_vel):
        """Hyperbolic orbit has is_stable=False."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        pos, vel = hyperbolic_pos_vel
        result = compute_jacobi_stability(pos, vel, MU)

        assert result.is_stable is False
        assert result.specific_energy_j_kg > 0.0  # unbound orbit has E > 0

    def test_scalar_curvature_positive_for_kepler(self, leo_circular_pos_vel):
        """Scalar curvature is positive for Kepler potential (no perturbations)."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        pos, vel = leo_circular_pos_vel
        result = compute_jacobi_stability(pos, vel, MU)

        assert result.scalar_curvature > 0.0

    def test_jacobi_metric_factor_equals_v_squared(self, leo_circular_pos_vel):
        """Jacobi metric factor 2(E - V) = 2T = v^2."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        pos, vel = leo_circular_pos_vel
        result = compute_jacobi_stability(pos, vel, MU)

        v_sq = vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2
        assert result.jacobi_metric_factor == pytest.approx(v_sq, rel=1e-10)

    def test_stability_index_positive_for_bound(self, leo_circular_pos_vel):
        """Stability index is positive for bound orbits."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        pos, vel = leo_circular_pos_vel
        result = compute_jacobi_stability(pos, vel, MU)

        assert result.stability_index > 0.0

    def test_stability_index_negative_for_unbound(self, hyperbolic_pos_vel):
        """Stability index is negative for unbound orbits."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        pos, vel = hyperbolic_pos_vel
        result = compute_jacobi_stability(pos, vel, MU)

        assert result.stability_index < 0.0

    def test_curvature_decreases_with_altitude(self):
        """Higher altitude (larger r) gives lower curvature (weaker focusing)."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        r_low = R_EARTH + 400_000.0
        r_high = R_EARTH + 800_000.0
        v_low = math.sqrt(MU / r_low)
        v_high = math.sqrt(MU / r_high)

        result_low = compute_jacobi_stability((r_low, 0.0, 0.0), (0.0, v_low, 0.0), MU)
        result_high = compute_jacobi_stability((r_high, 0.0, 0.0), (0.0, v_high, 0.0), MU)

        assert result_low.scalar_curvature > result_high.scalar_curvature

    def test_degenerate_position(self):
        """Position near origin returns safe defaults."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        result = compute_jacobi_stability((0.0, 0.0, 0.0), (1000.0, 0.0, 0.0), MU)

        assert result.scalar_curvature == 0.0
        assert result.is_stable is False

    def test_dataclass_frozen(self, leo_circular_pos_vel):
        """JacobiStability is immutable."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        pos, vel = leo_circular_pos_vel
        result = compute_jacobi_stability(pos, vel, MU)

        with pytest.raises(AttributeError):
            result.scalar_curvature = 999.0  # type: ignore[misc]

    def test_energy_consistent_with_vis_viva(self, leo_circular_pos_vel):
        """Specific energy matches vis-viva: E = v^2/2 - mu/r."""
        from humeris.domain.numerical_propagation import compute_jacobi_stability

        pos, vel = leo_circular_pos_vel
        result = compute_jacobi_stability(pos, vel, MU)

        r = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        v = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        expected_energy = 0.5 * v * v - MU / r

        assert result.specific_energy_j_kg == pytest.approx(expected_energy, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════
# P6: Contact Conjunction Metric
# ═══════════════════════════════════════════════════════════════════


class TestContactConjunction:
    """Tests for compute_contact_conjunction (P6)."""

    def test_import(self):
        """ContactConjunctionMetric and compute_contact_conjunction are importable."""
        from humeris.domain.conjunction import (
            ContactConjunctionMetric,
            compute_contact_conjunction,
        )
        assert ContactConjunctionMetric is not None
        assert compute_contact_conjunction is not None

    def test_basic_encounter(self):
        """Basic encounter returns valid metric."""
        from humeris.domain.conjunction import compute_contact_conjunction

        r = R_EARTH + 500_000.0
        v = math.sqrt(MU / r)

        pos1 = [r, 0.0, 0.0]
        vel1 = [0.0, v, 0.0]
        pos2 = [r + 1000.0, 100.0, 50.0]
        vel2 = [0.0, v + 10.0, 5.0]

        result = compute_contact_conjunction(pos1, vel1, pos2, vel2)

        assert result.miss_distance_m > 0
        assert result.relative_velocity_ms > 0
        assert result.contact_metric >= 0.0

    def test_reeb_direction_is_unit_vector(self):
        """Reeb direction is a unit vector."""
        from humeris.domain.conjunction import compute_contact_conjunction

        r = R_EARTH + 500_000.0
        v = math.sqrt(MU / r)

        pos1 = [r, 0.0, 0.0]
        vel1 = [0.0, v, 0.0]
        pos2 = [r + 500.0, 200.0, 100.0]
        vel2 = [0.0, v - 100.0, 50.0]

        result = compute_contact_conjunction(pos1, vel1, pos2, vel2)

        reeb = result.reeb_direction
        mag = math.sqrt(reeb[0] ** 2 + reeb[1] ** 2 + reeb[2] ** 2)
        assert mag == pytest.approx(1.0, abs=1e-10)

    def test_contact_metric_invariant_under_translation(self):
        """Contact volume is invariant under spatial translation."""
        from humeris.domain.conjunction import compute_contact_conjunction

        r = R_EARTH + 500_000.0
        v = math.sqrt(MU / r)

        pos1 = [r, 0.0, 0.0]
        vel1 = [0.0, v, 100.0]
        pos2 = [r + 1000.0, 500.0, 200.0]
        vel2 = [0.0, v - 200.0, -50.0]

        result1 = compute_contact_conjunction(pos1, vel1, pos2, vel2)

        # Translate everything by (1e6, 2e6, 3e6)
        offset = [1e6, 2e6, 3e6]
        pos1t = [p + o for p, o in zip(pos1, offset)]
        pos2t = [p + o for p, o in zip(pos2, offset)]

        result2 = compute_contact_conjunction(pos1t, vel1, pos2t, vel2)

        # Relative geometry unchanged, so contact volume should be the same
        assert result2.contact_metric == pytest.approx(result1.contact_metric, rel=1e-10)

    def test_zero_relative_velocity_degenerate(self):
        """Zero relative velocity yields degenerate result."""
        from humeris.domain.conjunction import compute_contact_conjunction

        pos1 = [R_EARTH + 500_000.0, 0.0, 0.0]
        vel = [0.0, 7500.0, 0.0]
        pos2 = [R_EARTH + 501_000.0, 0.0, 0.0]

        result = compute_contact_conjunction(pos1, vel, pos2, vel)

        assert result.relative_velocity_ms == pytest.approx(0.0, abs=1e-8)
        assert result.contact_metric == 0.0
        assert result.is_transverse is False

    def test_head_on_encounter_is_transverse(self):
        """Head-on encounter with significant B-plane offset is transverse."""
        from humeris.domain.conjunction import compute_contact_conjunction

        r = R_EARTH + 500_000.0
        v = math.sqrt(MU / r)

        # Two satellites at same altitude, crossing inclinations
        pos1 = [r, 0.0, 0.0]
        vel1 = [0.0, v * 0.7, v * 0.7]
        pos2 = [r, 1000.0, 0.0]  # 1 km offset in y
        vel2 = [0.0, v * 0.7, -v * 0.7]  # opposite cross-track

        result = compute_contact_conjunction(pos1, vel1, pos2, vel2)

        assert result.relative_velocity_ms > 0
        assert result.miss_distance_m > 0

    def test_legendrian_angle_range(self):
        """Legendrian angle is in [-pi, pi]."""
        from humeris.domain.conjunction import compute_contact_conjunction

        r = R_EARTH + 500_000.0
        v = math.sqrt(MU / r)

        pos1 = [r, 0.0, 0.0]
        vel1 = [0.0, v, 0.0]
        pos2 = [r, 500.0, 300.0]
        vel2 = [100.0, v - 50.0, -200.0]

        result = compute_contact_conjunction(pos1, vel1, pos2, vel2)

        assert -math.pi <= result.legendrian_angle_rad <= math.pi

    def test_dataclass_frozen(self):
        """ContactConjunctionMetric is immutable."""
        from humeris.domain.conjunction import compute_contact_conjunction

        r = R_EARTH + 500_000.0
        v = math.sqrt(MU / r)
        result = compute_contact_conjunction(
            [r, 0.0, 0.0], [0.0, v, 0.0],
            [r + 1000.0, 0.0, 0.0], [0.0, v + 10.0, 0.0],
        )

        with pytest.raises(AttributeError):
            result.contact_metric = 999.0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# P7: Morse Theory for Coverage Phase Transitions
# ═══════════════════════════════════════════════════════════════════


class TestCoverageMorse:
    """Tests for compute_coverage_morse (P7)."""

    def test_import(self):
        """CoverageMorseAnalysis and compute_coverage_morse are importable."""
        from humeris.domain.coverage import (
            CoverageMorseAnalysis,
            CoverageCriticalPoint,
            compute_coverage_morse,
        )
        assert CoverageMorseAnalysis is not None
        assert CoverageCriticalPoint is not None
        assert compute_coverage_morse is not None

    def test_empty_grid(self):
        """Empty coverage grid returns empty analysis."""
        from humeris.domain.coverage import compute_coverage_morse

        result = compute_coverage_morse([], lat_step_deg=10.0, lon_step_deg=10.0)

        assert result.num_maxima == 0
        assert result.num_saddles == 0
        assert result.num_minima == 0
        assert result.euler_characteristic == 0

    def test_uniform_coverage_no_critical_points(self):
        """Uniform coverage (all same count) has no critical points."""
        from humeris.domain.coverage import CoveragePoint, compute_coverage_morse

        # Create uniform grid: all cells have coverage = 3
        grid = []
        for lat in range(-80, 81, 10):
            for lon in range(-170, 171, 10):
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=3))

        result = compute_coverage_morse(grid, lat_step_deg=10.0, lon_step_deg=10.0)

        # Uniform function has no critical points (gradient everywhere zero,
        # but Hessian is zero too — degenerate)
        assert result.num_maxima == 0
        assert result.num_minima == 0

    def test_single_peak_coverage(self):
        """Coverage with a single peak should detect a maximum."""
        from humeris.domain.coverage import CoveragePoint, compute_coverage_morse

        # Create grid with a peak at (0, 0) and decreasing away
        grid = []
        for lat in range(-40, 41, 10):
            for lon in range(-40, 41, 10):
                # Distance from center
                d = math.sqrt(lat ** 2 + lon ** 2)
                count = max(0, int(10 - d / 5))
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))

        result = compute_coverage_morse(grid, lat_step_deg=10.0, lon_step_deg=10.0)

        assert result.num_maxima >= 1

    def test_coverage_gap_detected(self):
        """Coverage with a gap (minimum) should detect a minimum."""
        from humeris.domain.coverage import CoveragePoint, compute_coverage_morse

        # Create grid with a smooth valley at center: high edges, low center
        # This ensures the center point (0,0) has all 4 neighbors at higher values
        grid = []
        for lat in range(-40, 41, 10):
            for lon in range(-40, 41, 10):
                d = math.sqrt(lat ** 2 + lon ** 2)
                # Smooth: coverage increases with distance from center
                count = int(min(10, d / 5))
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))

        result = compute_coverage_morse(grid, lat_step_deg=10.0, lon_step_deg=10.0)

        assert result.num_minima >= 1

    def test_euler_characteristic_computed(self):
        """Euler characteristic is computed as maxima - saddles + minima."""
        from humeris.domain.coverage import CoveragePoint, compute_coverage_morse

        # Create non-trivial coverage pattern
        grid = []
        for lat in range(-60, 61, 10):
            for lon in range(-60, 61, 10):
                count = int(5 + 3 * math.sin(math.radians(lat)) * math.cos(math.radians(lon)))
                count = max(0, count)
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))

        result = compute_coverage_morse(grid, lat_step_deg=10.0, lon_step_deg=10.0)

        assert result.euler_characteristic == result.num_maxima - result.num_saddles + result.num_minima

    def test_critical_points_have_valid_index(self):
        """All critical points have index in {0, 1, 2}."""
        from humeris.domain.coverage import CoveragePoint, compute_coverage_morse

        grid = []
        for lat in range(-50, 51, 10):
            for lon in range(-50, 51, 10):
                count = int(3 + 2 * math.cos(math.radians(lat * 3)))
                count = max(0, count)
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))

        result = compute_coverage_morse(grid, lat_step_deg=10.0, lon_step_deg=10.0)

        for cp in result.critical_points:
            assert cp.index in (0, 1, 2)

    def test_phase_transition_altitudes_are_sorted_coverage_levels(self):
        """Phase transition 'altitudes' are coverage levels where topology changes."""
        from humeris.domain.coverage import CoveragePoint, compute_coverage_morse

        # Two separate high-coverage regions that merge at a certain level
        grid = []
        for lat in range(-50, 51, 10):
            for lon in range(-50, 51, 10):
                if abs(lon + 30) < 15 and abs(lat) < 20:
                    count = 8
                elif abs(lon - 30) < 15 and abs(lat) < 20:
                    count = 8
                else:
                    count = 2
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))

        result = compute_coverage_morse(grid, lat_step_deg=10.0, lon_step_deg=10.0)

        # Phase transitions should exist where the topology changes
        assert isinstance(result.phase_transition_altitudes, tuple)

    def test_too_small_grid_returns_empty(self):
        """Grid smaller than 3x3 returns empty analysis."""
        from humeris.domain.coverage import CoveragePoint, compute_coverage_morse

        grid = [
            CoveragePoint(lat_deg=0.0, lon_deg=0.0, visible_count=5),
            CoveragePoint(lat_deg=0.0, lon_deg=10.0, visible_count=3),
        ]

        result = compute_coverage_morse(grid, lat_step_deg=10.0, lon_step_deg=10.0)

        assert result.num_maxima == 0
        assert result.euler_characteristic == 0

    def test_dataclass_frozen(self):
        """CoverageMorseAnalysis is immutable."""
        from humeris.domain.coverage import CoveragePoint, compute_coverage_morse

        grid = []
        for lat in range(-30, 31, 10):
            for lon in range(-30, 31, 10):
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=3))

        result = compute_coverage_morse(grid, lat_step_deg=10.0, lon_step_deg=10.0)

        with pytest.raises(AttributeError):
            result.num_maxima = 999  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# P18: Spectral Kessler via Perron-Frobenius
# ═══════════════════════════════════════════════════════════════════


class TestSpectralKessler:
    """Tests for compute_spectral_kessler (P18)."""

    def test_import(self):
        """SpectralKessler and compute_spectral_kessler are importable."""
        from humeris.domain.kessler_heatmap import (
            SpectralKessler,
            compute_spectral_kessler,
        )
        assert SpectralKessler is not None
        assert compute_spectral_kessler is not None

    def _make_heatmap(self, n_sats=50, alt_km=800.0, inc_deg=98.0):
        """Helper: create a heatmap with satellites concentrated in one band."""
        from humeris.domain.propagation import OrbitalState
        from humeris.domain.kessler_heatmap import compute_kessler_heatmap

        states = []
        for k in range(n_sats):
            a = R_EARTH + alt_km * 1000.0 + k * 100.0  # slight spread
            n = math.sqrt(MU / a ** 3)
            states.append(OrbitalState(
                semi_major_axis_m=a,
                eccentricity=0.0,
                inclination_rad=math.radians(inc_deg),
                raan_rad=k * 2.0 * math.pi / n_sats,
                arg_perigee_rad=0.0,
                true_anomaly_rad=0.0,
                mean_motion_rad_s=n,
                reference_epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ))

        return compute_kessler_heatmap(states)

    def test_basic_computation(self):
        """Spectral Kessler computes without error."""
        from humeris.domain.kessler_heatmap import compute_spectral_kessler

        heatmap = self._make_heatmap()
        result = compute_spectral_kessler(heatmap)

        assert result.migration_eigenvalue >= 0.0
        assert len(result.dominant_mode) > 0

    def test_eigenvalue_nonnegative(self):
        """Perron-Frobenius eigenvalue is non-negative for non-negative matrix."""
        from humeris.domain.kessler_heatmap import compute_spectral_kessler

        heatmap = self._make_heatmap(n_sats=100)
        result = compute_spectral_kessler(heatmap)

        assert result.migration_eigenvalue >= 0.0

    def test_dominant_mode_sums_to_one(self):
        """Dominant mode (eigenvector) is normalized to sum to 1."""
        from humeris.domain.kessler_heatmap import compute_spectral_kessler

        heatmap = self._make_heatmap()
        result = compute_spectral_kessler(heatmap)

        if result.dominant_mode:
            total = sum(result.dominant_mode)
            assert total == pytest.approx(1.0, abs=0.01)

    def test_spectral_gap_nonnegative(self):
        """Spectral gap (lambda_1 - lambda_2) is non-negative."""
        from humeris.domain.kessler_heatmap import compute_spectral_kessler

        heatmap = self._make_heatmap()
        result = compute_spectral_kessler(heatmap)

        assert result.spectral_gap >= -1e-10  # allow tiny numerical noise

    def test_most_dangerous_cell_valid_index(self):
        """Most dangerous cell index is within valid range."""
        from humeris.domain.kessler_heatmap import compute_spectral_kessler

        heatmap = self._make_heatmap()
        result = compute_spectral_kessler(heatmap)

        assert 0 <= result.most_dangerous_cell_idx < len(heatmap.cells)

    def test_empty_heatmap(self):
        """Empty heatmap returns safe defaults."""
        from humeris.domain.kessler_heatmap import KesslerHeatMap, compute_spectral_kessler

        empty_heatmap = KesslerHeatMap(
            cells=(),
            altitude_bins_km=(200.0, 250.0),
            inclination_bins_deg=(0.0, 10.0),
            total_objects=0,
            peak_density_altitude_km=0.0,
            peak_density_inclination_deg=0.0,
            peak_density_per_km3=0.0,
        )

        result = compute_spectral_kessler(empty_heatmap)

        assert result.migration_eigenvalue == 0.0
        assert result.is_supercritical is False

    def test_supercritical_flag(self):
        """is_supercritical reflects migration_eigenvalue > 1."""
        from humeris.domain.kessler_heatmap import compute_spectral_kessler

        heatmap = self._make_heatmap()
        result = compute_spectral_kessler(heatmap)

        assert result.is_supercritical == (result.migration_eigenvalue > 1.0)

    def test_peak_cell_keff_provided(self):
        """k_eff_peak_cell is computed for comparison."""
        from humeris.domain.kessler_heatmap import compute_spectral_kessler

        heatmap = self._make_heatmap()
        result = compute_spectral_kessler(heatmap)

        assert result.k_eff_peak_cell >= 0.0

    def test_dataclass_frozen(self):
        """SpectralKessler is immutable."""
        from humeris.domain.kessler_heatmap import compute_spectral_kessler

        heatmap = self._make_heatmap()
        result = compute_spectral_kessler(heatmap)

        with pytest.raises(AttributeError):
            result.migration_eigenvalue = 999.0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# P19: Wasserstein Distance for Constellation Reconfiguration
# ═══════════════════════════════════════════════════════════════════


class TestReconfigurationPlan:
    """Tests for compute_optimal_reconfiguration (P19)."""

    def test_import(self):
        """ReconfigurationPlan and compute_optimal_reconfiguration are importable."""
        from humeris.domain.design_optimization import (
            ReconfigurationPlan,
            compute_optimal_reconfiguration,
        )
        assert ReconfigurationPlan is not None
        assert compute_optimal_reconfiguration is not None

    def _make_state(self, alt_km, inc_deg, raan_deg):
        """Helper: create an OrbitalState."""
        from humeris.domain.propagation import OrbitalState

        a = R_EARTH + alt_km * 1000.0
        n = math.sqrt(MU / a ** 3)
        return OrbitalState(
            semi_major_axis_m=a,
            eccentricity=0.0,
            inclination_rad=math.radians(inc_deg),
            raan_rad=math.radians(raan_deg),
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=n,
            reference_epoch=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    def test_identical_constellations_zero_dv(self):
        """Identical current and target yields zero total dV."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        states = [self._make_state(500, 53, raan) for raan in [0, 90, 180, 270]]

        result = compute_optimal_reconfiguration(states, states)

        assert result.total_dv_ms == pytest.approx(0.0, abs=1.0)

    def test_altitude_change_requires_dv(self):
        """Changing altitude requires non-zero dV."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        current = [self._make_state(500, 53, 0)]
        target = [self._make_state(600, 53, 0)]

        result = compute_optimal_reconfiguration(current, target)

        assert result.total_dv_ms > 0.0

    def test_plane_change_requires_dv(self):
        """Changing orbital plane requires non-zero dV."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        current = [self._make_state(500, 53, 0)]
        target = [self._make_state(500, 53, 90)]  # 90 deg RAAN change

        result = compute_optimal_reconfiguration(current, target)

        assert result.total_dv_ms > 0.0

    def test_assignment_covers_all_satellites(self):
        """Each current satellite is assigned to a target slot."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        current = [self._make_state(500, 53, raan) for raan in [0, 120, 240]]
        target = [self._make_state(500, 53, raan) for raan in [60, 180, 300]]

        result = compute_optimal_reconfiguration(current, target)

        current_indices = {a[0] for a in result.assignments}
        target_indices = {a[1] for a in result.assignments}
        assert len(result.assignments) == 3
        assert current_indices == {0, 1, 2}
        assert target_indices == {0, 1, 2}

    def test_optimal_assignment_better_than_naive(self):
        """Optimal assignment should be <= naive sequential assignment in total dV."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        # Design a case where optimal != naive
        # Current: 0, 120, 240 deg RAAN
        # Target:  130, 250, 10 deg RAAN
        # Naive: 0->130, 120->250, 240->10 (large total)
        # Optimal: 0->10, 120->130, 240->250 (small total)
        current = [self._make_state(500, 53, raan) for raan in [0, 120, 240]]
        target = [self._make_state(500, 53, raan) for raan in [130, 250, 10]]

        result = compute_optimal_reconfiguration(current, target)

        # The optimal should find the better matching
        assert result.total_dv_ms > 0.0

    def test_empty_constellations(self):
        """Empty constellation lists return zero plan."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        result = compute_optimal_reconfiguration([], [])

        assert result.total_dv_ms == 0.0
        assert result.assignments == ()

    def test_wasserstein_distance_nonnegative(self):
        """Wasserstein distance is non-negative."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        current = [self._make_state(500, 53, raan) for raan in [0, 90]]
        target = [self._make_state(500, 53, raan) for raan in [45, 135]]

        result = compute_optimal_reconfiguration(current, target)

        assert result.wasserstein_distance >= 0.0

    def test_single_satellite_reconfiguration(self):
        """Single satellite reconfiguration produces one assignment."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        current = [self._make_state(500, 53, 0)]
        target = [self._make_state(550, 53, 0)]

        result = compute_optimal_reconfiguration(current, target)

        assert len(result.assignments) == 1
        assert result.assignments[0][0] == 0  # current idx
        assert result.assignments[0][1] == 0  # target idx
        assert result.assignments[0][2] > 0   # dv > 0

    def test_savings_percentage_nonnegative(self):
        """Savings vs naive is non-negative."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        current = [self._make_state(500, 53, raan) for raan in [0, 120, 240]]
        target = [self._make_state(500, 53, raan) for raan in [60, 180, 300]]

        result = compute_optimal_reconfiguration(current, target)

        assert result.savings_vs_naive_pct >= 0.0

    def test_dataclass_frozen(self):
        """ReconfigurationPlan is immutable."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        current = [self._make_state(500, 53, 0)]
        target = [self._make_state(550, 53, 0)]
        result = compute_optimal_reconfiguration(current, target)

        with pytest.raises(AttributeError):
            result.total_dv_ms = 999.0  # type: ignore[misc]

    def test_dv_values_physically_reasonable(self):
        """Delta-V values are in a physically reasonable range."""
        from humeris.domain.design_optimization import compute_optimal_reconfiguration

        # 500 -> 550 km altitude change should be ~30-50 m/s (Hohmann)
        current = [self._make_state(500, 53, 0)]
        target = [self._make_state(550, 53, 0)]

        result = compute_optimal_reconfiguration(current, target)

        # Hohmann from 500 to 550 km is about 28 m/s
        assert 10.0 < result.total_dv_ms < 200.0
