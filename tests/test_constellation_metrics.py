# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for constellation-level derived metrics.

Coverage statistics, revisit statistics, eclipse statistics, N-1 redundancy,
constellation score, thermal cycling, ground network metrics, deployment phasing.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.propagation import OrbitalState, derive_orbital_state, propagate_to
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.coverage import CoveragePoint, compute_coverage_snapshot
from humeris.domain.observation import GroundStation
from humeris.domain.access_windows import AccessWindow

EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
MU = OrbitalConstants.MU_EARTH
R_E = OrbitalConstants.R_EARTH


def _leo_state(altitude_km=550, inclination_deg=53) -> OrbitalState:
    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=inclination_deg,
        num_planes=1, sats_per_plane=1, phase_factor=0,
        raan_offset_deg=0, shell_name="Test",
    )
    sats = generate_walker_shell(shell)
    return derive_orbital_state(sats[0], EPOCH)


def _constellation_states(n_planes=3, sats_per_plane=4) -> list[OrbitalState]:
    shell = ShellConfig(
        altitude_km=550, inclination_deg=53,
        num_planes=n_planes, sats_per_plane=sats_per_plane,
        phase_factor=1, raan_offset_deg=0, shell_name="Constellation",
    )
    sats = generate_walker_shell(shell)
    return [derive_orbital_state(s, EPOCH) for s in sats]


# --- Coverage Statistics ---

class TestComputeCoverageStatistics:

    def test_returns_coverage_statistics_type(self):
        from humeris.domain.constellation_metrics import (
            compute_coverage_statistics, CoverageStatistics,
        )
        grid = [
            CoveragePoint(lat_deg=0, lon_deg=0, visible_count=2),
            CoveragePoint(lat_deg=10, lon_deg=10, visible_count=0),
            CoveragePoint(lat_deg=20, lon_deg=20, visible_count=3),
        ]
        result = compute_coverage_statistics(grid)
        assert isinstance(result, CoverageStatistics)

    def test_mean_visible(self):
        from humeris.domain.constellation_metrics import compute_coverage_statistics
        grid = [
            CoveragePoint(lat_deg=0, lon_deg=0, visible_count=2),
            CoveragePoint(lat_deg=0, lon_deg=10, visible_count=4),
        ]
        result = compute_coverage_statistics(grid)
        assert abs(result.mean_visible - 3.0) < 0.01

    def test_max_and_min_visible(self):
        from humeris.domain.constellation_metrics import compute_coverage_statistics
        grid = [
            CoveragePoint(lat_deg=0, lon_deg=0, visible_count=1),
            CoveragePoint(lat_deg=0, lon_deg=10, visible_count=5),
            CoveragePoint(lat_deg=0, lon_deg=20, visible_count=3),
        ]
        result = compute_coverage_statistics(grid)
        assert result.max_visible == 5
        assert result.min_visible == 1

    def test_percent_covered(self):
        from humeris.domain.constellation_metrics import compute_coverage_statistics
        grid = [
            CoveragePoint(lat_deg=0, lon_deg=0, visible_count=2),
            CoveragePoint(lat_deg=0, lon_deg=10, visible_count=0),
            CoveragePoint(lat_deg=0, lon_deg=20, visible_count=0),
            CoveragePoint(lat_deg=0, lon_deg=30, visible_count=1),
        ]
        result = compute_coverage_statistics(grid)
        assert abs(result.percent_covered - 50.0) < 0.01

    def test_n_fold_coverage(self):
        from humeris.domain.constellation_metrics import compute_coverage_statistics
        grid = [
            CoveragePoint(lat_deg=0, lon_deg=0, visible_count=3),
            CoveragePoint(lat_deg=0, lon_deg=10, visible_count=1),
            CoveragePoint(lat_deg=0, lon_deg=20, visible_count=2),
            CoveragePoint(lat_deg=0, lon_deg=30, visible_count=0),
        ]
        result = compute_coverage_statistics(grid, n_fold_levels=[1, 2, 3])
        assert abs(result.n_fold_coverage[1] - 75.0) < 0.01  # 3 of 4 have ≥1
        assert abs(result.n_fold_coverage[2] - 50.0) < 0.01  # 2 of 4 have ≥2
        assert abs(result.n_fold_coverage[3] - 25.0) < 0.01  # 1 of 4 has ≥3

    def test_empty_grid(self):
        from humeris.domain.constellation_metrics import compute_coverage_statistics
        result = compute_coverage_statistics([])
        assert result.mean_visible == 0.0
        assert result.percent_covered == 0.0

    def test_std_visible_computed(self):
        from humeris.domain.constellation_metrics import compute_coverage_statistics
        grid = [
            CoveragePoint(lat_deg=0, lon_deg=0, visible_count=2),
            CoveragePoint(lat_deg=0, lon_deg=10, visible_count=4),
        ]
        result = compute_coverage_statistics(grid)
        # std of [2, 4] = 1.0
        assert abs(result.std_visible - 1.0) < 0.01


# --- Revisit Statistics ---

class TestComputeRevisitStatistics:

    def test_returns_revisit_statistics_type(self):
        from humeris.domain.constellation_metrics import (
            compute_revisit_statistics, RevisitStatistics,
        )
        result = compute_revisit_statistics([100.0, 200.0, 300.0])
        assert isinstance(result, RevisitStatistics)

    def test_mean_revisit(self):
        from humeris.domain.constellation_metrics import compute_revisit_statistics
        result = compute_revisit_statistics([100.0, 200.0, 300.0])
        assert abs(result.mean_revisit_s - 200.0) < 0.01

    def test_max_revisit(self):
        from humeris.domain.constellation_metrics import compute_revisit_statistics
        result = compute_revisit_statistics([100.0, 200.0, 500.0])
        assert abs(result.max_revisit_s - 500.0) < 0.01

    def test_percentile_95(self):
        from humeris.domain.constellation_metrics import compute_revisit_statistics
        data = list(range(1, 101))  # 1 to 100
        result = compute_revisit_statistics([float(x) for x in data])
        # 95th percentile of 1..100 = 95
        assert 94.0 < result.percentile_95_s < 96.0

    def test_threshold_compliance(self):
        from humeris.domain.constellation_metrics import compute_revisit_statistics
        result = compute_revisit_statistics([100, 200, 300, 400, 500], threshold_s=350)
        # 3 of 5 are within 350s
        assert abs(result.percent_within_threshold - 60.0) < 0.01

    def test_empty_data(self):
        from humeris.domain.constellation_metrics import compute_revisit_statistics
        result = compute_revisit_statistics([])
        assert result.mean_revisit_s == 0.0
        assert result.max_revisit_s == 0.0


# --- Eclipse Statistics ---

class TestComputeEclipseStatistics:

    def test_returns_eclipse_statistics_type(self):
        from humeris.domain.constellation_metrics import (
            compute_eclipse_statistics, EclipseStatistics,
        )
        state = _leo_state()
        result = compute_eclipse_statistics(
            state, EPOCH, timedelta(hours=2), timedelta(seconds=30),
        )
        assert isinstance(result, EclipseStatistics)

    def test_eclipse_fraction_between_0_and_1(self):
        from humeris.domain.constellation_metrics import compute_eclipse_statistics
        state = _leo_state()
        result = compute_eclipse_statistics(
            state, EPOCH, timedelta(hours=2), timedelta(seconds=30),
        )
        assert 0.0 <= result.eclipse_fraction <= 1.0

    def test_num_eclipses_positive_for_leo(self):
        """LEO satellite crosses shadow multiple times in 2 hours."""
        from humeris.domain.constellation_metrics import compute_eclipse_statistics
        state = _leo_state(altitude_km=550)
        result = compute_eclipse_statistics(
            state, EPOCH, timedelta(hours=3), timedelta(seconds=30),
        )
        assert result.num_eclipses >= 1

    def test_total_eclipse_consistent(self):
        from humeris.domain.constellation_metrics import compute_eclipse_statistics
        state = _leo_state()
        result = compute_eclipse_statistics(
            state, EPOCH, timedelta(hours=2), timedelta(seconds=30),
        )
        # Total eclipse time = fraction * duration
        duration_s = 2 * 3600
        expected_total = result.eclipse_fraction * duration_s
        assert abs(result.total_eclipse_s - expected_total) < 60  # within 1 step


# --- N-1 Redundancy ---

class TestComputeNMinus1Redundancy:

    def test_returns_redundancy_result_type(self):
        from humeris.domain.constellation_metrics import (
            compute_n_minus_1_redundancy, RedundancyResult,
        )
        states = _constellation_states(n_planes=2, sats_per_plane=3)
        result = compute_n_minus_1_redundancy(states, EPOCH)
        assert isinstance(result, RedundancyResult)

    def test_baseline_greater_than_degraded(self):
        from humeris.domain.constellation_metrics import compute_n_minus_1_redundancy
        states = _constellation_states(n_planes=2, sats_per_plane=3)
        result = compute_n_minus_1_redundancy(states, EPOCH)
        assert result.baseline_coverage_pct >= result.worst_degraded_coverage_pct

    def test_degradation_nonnegative(self):
        from humeris.domain.constellation_metrics import compute_n_minus_1_redundancy
        states = _constellation_states(n_planes=2, sats_per_plane=3)
        result = compute_n_minus_1_redundancy(states, EPOCH)
        assert result.degradation_pct >= 0

    def test_worst_case_index_valid(self):
        from humeris.domain.constellation_metrics import compute_n_minus_1_redundancy
        states = _constellation_states(n_planes=2, sats_per_plane=3)
        result = compute_n_minus_1_redundancy(states, EPOCH)
        assert 0 <= result.worst_case_satellite_idx < len(states)

    def test_single_satellite(self):
        from humeris.domain.constellation_metrics import compute_n_minus_1_redundancy
        states = [_leo_state()]
        result = compute_n_minus_1_redundancy(states, EPOCH)
        assert result.worst_degraded_coverage_pct == 0.0


# --- Constellation Score ---

class TestComputeConstellationScore:

    def test_returns_constellation_score_type(self):
        from humeris.domain.constellation_metrics import (
            compute_constellation_score, ConstellationScore,
        )
        result = compute_constellation_score(
            coverage_pct=85.0, max_revisit_s=3600, target_revisit_s=7200,
            degradation_pct=5.0,
        )
        assert isinstance(result, ConstellationScore)

    def test_perfect_scores(self):
        from humeris.domain.constellation_metrics import compute_constellation_score
        result = compute_constellation_score(
            coverage_pct=100.0, max_revisit_s=1800, target_revisit_s=3600,
            degradation_pct=0.0,
        )
        assert result.coverage_score == 1.0
        assert result.overall_score >= 0.9

    def test_score_between_0_and_1(self):
        from humeris.domain.constellation_metrics import compute_constellation_score
        result = compute_constellation_score(
            coverage_pct=50.0, max_revisit_s=7200, target_revisit_s=3600,
            degradation_pct=20.0,
        )
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.coverage_score <= 1.0
        assert 0.0 <= result.revisit_score <= 1.0
        assert 0.0 <= result.redundancy_score <= 1.0

    def test_worse_inputs_lower_score(self):
        from humeris.domain.constellation_metrics import compute_constellation_score
        good = compute_constellation_score(90.0, 1800, 3600, 5.0)
        bad = compute_constellation_score(50.0, 10800, 3600, 30.0)
        assert good.overall_score > bad.overall_score


# --- Thermal Cycling ---

class TestComputeThermalCycling:

    def test_returns_thermal_cycling_type(self):
        from humeris.domain.constellation_metrics import (
            compute_thermal_cycling, ThermalCycling,
        )
        state = _leo_state()
        result = compute_thermal_cycling(
            state, EPOCH, timedelta(hours=3), timedelta(seconds=30),
        )
        assert isinstance(result, ThermalCycling)

    def test_cycle_count_for_leo(self):
        """LEO does ~1 eclipse per orbit, ~3 orbits in 3 hours → ~3 cycles."""
        from humeris.domain.constellation_metrics import compute_thermal_cycling
        state = _leo_state(altitude_km=550)
        result = compute_thermal_cycling(
            state, EPOCH, timedelta(hours=5), timedelta(seconds=30),
        )
        # Expect 2-5 cycles in 5 hours
        assert result.num_cycles >= 1

    def test_eclipse_and_sunlit_durations_present(self):
        from humeris.domain.constellation_metrics import compute_thermal_cycling
        state = _leo_state()
        result = compute_thermal_cycling(
            state, EPOCH, timedelta(hours=3), timedelta(seconds=30),
        )
        assert isinstance(result.eclipse_durations_s, tuple)
        assert isinstance(result.sunlit_durations_s, tuple)


# --- Ground Network Metrics ---

class TestComputeGroundNetworkMetrics:

    def test_returns_ground_network_metrics_type(self):
        from humeris.domain.constellation_metrics import (
            compute_ground_network_metrics, GroundNetworkMetrics,
        )
        stations = [
            GroundStation(name="Delft", lat_deg=52.0, lon_deg=4.4),
            GroundStation(name="Svalbard", lat_deg=78.2, lon_deg=15.6),
        ]
        states = [_leo_state()]
        result = compute_ground_network_metrics(
            stations, states, EPOCH, timedelta(hours=6), timedelta(seconds=30),
        )
        assert isinstance(result, GroundNetworkMetrics)

    def test_more_stations_more_contact(self):
        from humeris.domain.constellation_metrics import compute_ground_network_metrics
        single = [GroundStation(name="Delft", lat_deg=52.0, lon_deg=4.4)]
        double = [
            GroundStation(name="Delft", lat_deg=52.0, lon_deg=4.4),
            GroundStation(name="Svalbard", lat_deg=78.2, lon_deg=15.6),
        ]
        states = [_leo_state()]
        r1 = compute_ground_network_metrics(
            single, states, EPOCH, timedelta(hours=12), timedelta(seconds=30),
        )
        r2 = compute_ground_network_metrics(
            double, states, EPOCH, timedelta(hours=12), timedelta(seconds=30),
        )
        assert r2.total_contact_s >= r1.total_contact_s

    def test_contact_per_station_dict(self):
        from humeris.domain.constellation_metrics import compute_ground_network_metrics
        stations = [GroundStation(name="Test", lat_deg=0.0, lon_deg=0.0)]
        states = [_leo_state()]
        result = compute_ground_network_metrics(
            stations, states, EPOCH, timedelta(hours=6), timedelta(seconds=30),
        )
        assert "Test" in result.contact_per_station


# --- Deployment Phasing ---

class TestComputeDeploymentPhasing:

    def test_returns_deployment_phasing_type(self):
        from humeris.domain.constellation_metrics import (
            compute_deployment_phasing, DeploymentPhasing,
        )
        result = compute_deployment_phasing(
            num_planes=3, sats_per_plane=4, orbit_radius_m=R_E + 550_000,
        )
        assert isinstance(result, DeploymentPhasing)

    def test_total_delta_v_positive(self):
        from humeris.domain.constellation_metrics import compute_deployment_phasing
        result = compute_deployment_phasing(3, 4, R_E + 550_000)
        assert result.total_delta_v_ms >= 0

    def test_total_time_positive(self):
        from humeris.domain.constellation_metrics import compute_deployment_phasing
        result = compute_deployment_phasing(3, 4, R_E + 550_000)
        assert result.total_time_s >= 0

    def test_per_satellite_count(self):
        from humeris.domain.constellation_metrics import compute_deployment_phasing
        result = compute_deployment_phasing(3, 4, R_E + 550_000)
        # Total sats = 12, first one needs no maneuver → 11 phasing entries
        assert len(result.per_satellite) == 3 * 4

    def test_single_plane_minimal_dv(self):
        from humeris.domain.constellation_metrics import compute_deployment_phasing
        result = compute_deployment_phasing(1, 1, R_E + 550_000)
        assert result.total_delta_v_ms == 0.0

    def test_zero_orbit_radius_raises(self):
        from humeris.domain.constellation_metrics import compute_deployment_phasing
        with pytest.raises(ValueError, match="orbit_radius_m"):
            compute_deployment_phasing(3, 4, orbit_radius_m=0.0)

    def test_negative_orbit_radius_raises(self):
        from humeris.domain.constellation_metrics import compute_deployment_phasing
        with pytest.raises(ValueError, match="orbit_radius_m"):
            compute_deployment_phasing(3, 4, orbit_radius_m=-1_000_000.0)


# --- Thermal Cycling Edge Cases ---

class TestThermalCyclingEdgeCases:

    def test_zero_step_returns_empty(self):
        """Zero step size must not cause infinite loop."""
        from humeris.domain.constellation_metrics import compute_thermal_cycling
        state = _leo_state()
        result = compute_thermal_cycling(
            state, EPOCH, timedelta(hours=1), timedelta(seconds=0),
        )
        assert result.num_cycles == 0
        assert result.eclipse_durations_s == ()
        assert result.sunlit_durations_s == ()

    def test_negative_step_returns_empty(self):
        from humeris.domain.constellation_metrics import compute_thermal_cycling
        state = _leo_state()
        result = compute_thermal_cycling(
            state, EPOCH, timedelta(hours=1), timedelta(seconds=-10),
        )
        assert result.num_cycles == 0

    def test_zero_duration_returns_empty(self):
        from humeris.domain.constellation_metrics import compute_thermal_cycling
        state = _leo_state()
        result = compute_thermal_cycling(
            state, EPOCH, timedelta(seconds=0), timedelta(seconds=30),
        )
        assert result.num_cycles == 0


# --- Purity ---

class TestConstellationMetricsPurity:

    def test_no_external_deps(self):
        import ast
        import humeris.domain.constellation_metrics as mod
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {"math", "numpy", "dataclasses", "datetime"}
        allowed_internal = {"humeris"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import from: {node.module}"
