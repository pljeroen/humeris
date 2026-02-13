# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for Round 3 creative proposals: P42, P47, P50, P54, P64.

P42: Partition Function Constellation Thermodynamics (multi_objective_design.py)
P47: Potential Flow Coverage via Conformal Mapping (coverage_optimization.py)
P50: Matroid Independence for Minimum Coverage Sets (coverage_optimization.py)
P54: Neural Field Equation for Coverage Demand (coverage_optimization.py)
P64: Ergodic Coverage Partition (coverage_optimization.py)
"""
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
MU = OrbitalConstants.MU_EARTH
R_EARTH = OrbitalConstants.R_EARTH


# ═══════════════════════════════════════════════════════════════════
# P42: Partition Function Constellation Thermodynamics
# ═══════════════════════════════════════════════════════════════════


class TestConstellationThermodynamicsDataclass:
    """Verify ConstellationThermodynamics is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.multi_objective_design import ConstellationThermodynamics

        result = ConstellationThermodynamics(
            partition_function_log=np.array([1.0]),
            helmholtz_free_energy=np.array([0.5]),
            mean_energy=np.array([1.0]),
            energy_variance=np.array([0.1]),
            heat_capacity=np.array([0.5]),
            entropy=np.array([0.3]),
            critical_temperature=2.0,
        )
        with pytest.raises(AttributeError):
            result.critical_temperature = 99.0

    def test_all_fields_present(self):
        from humeris.domain.multi_objective_design import ConstellationThermodynamics

        result = ConstellationThermodynamics(
            partition_function_log=np.array([1.0]),
            helmholtz_free_energy=np.array([0.5]),
            mean_energy=np.array([1.0]),
            energy_variance=np.array([0.1]),
            heat_capacity=np.array([0.5]),
            entropy=np.array([0.3]),
            critical_temperature=2.0,
        )
        assert hasattr(result, "partition_function_log")
        assert hasattr(result, "helmholtz_free_energy")
        assert hasattr(result, "mean_energy")
        assert hasattr(result, "energy_variance")
        assert hasattr(result, "heat_capacity")
        assert hasattr(result, "entropy")
        assert hasattr(result, "critical_temperature")


class TestComputeConstellationThermodynamics:
    """Test compute_constellation_thermodynamics function."""

    def test_single_beta_single_energy(self):
        """Simplest case: one energy, one temperature."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([0.0])
        betas = np.array([1.0])
        result = compute_constellation_thermodynamics(energies, betas)
        # Z = exp(0) = 1, log(Z) = 0
        assert abs(result.partition_function_log[0]) < 1e-10
        # F = -ln(Z)/beta = 0
        assert abs(result.helmholtz_free_energy[0]) < 1e-10

    def test_partition_function_two_energies(self):
        """Z = exp(-beta*E1) + exp(-beta*E2), verify log-sum-exp."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([0.0, 1.0])
        betas = np.array([1.0])
        result = compute_constellation_thermodynamics(energies, betas)
        expected_log_z = np.log(np.exp(0.0) + np.exp(-1.0))
        assert abs(result.partition_function_log[0] - expected_log_z) < 1e-10

    def test_helmholtz_free_energy(self):
        """F = -ln(Z)/beta."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([1.0, 2.0, 3.0])
        betas = np.array([0.5])
        result = compute_constellation_thermodynamics(energies, betas)
        log_z = result.partition_function_log[0]
        expected_f = -log_z / 0.5
        assert abs(result.helmholtz_free_energy[0] - expected_f) < 1e-10

    def test_mean_energy_uniform_at_zero_beta(self):
        """At high temperature (small beta), <E> ~ arithmetic mean."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([1.0, 2.0, 3.0])
        betas = np.array([0.001])
        result = compute_constellation_thermodynamics(energies, betas)
        # At beta ~ 0, all states equiprobable, <E> ~ mean
        assert abs(result.mean_energy[0] - 2.0) < 0.05

    def test_mean_energy_low_temperature(self):
        """At low temperature (large beta), <E> ~ ground state energy."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([1.0, 5.0, 10.0])
        betas = np.array([100.0])
        result = compute_constellation_thermodynamics(energies, betas)
        # At very low T, system occupies ground state
        assert abs(result.mean_energy[0] - 1.0) < 0.01

    def test_heat_capacity_nonnegative(self):
        """C_v = beta^2 * Var(E) >= 0."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([0.0, 1.0, 2.0, 5.0])
        betas = np.linspace(0.1, 10.0, 20)
        result = compute_constellation_thermodynamics(energies, betas)
        assert np.all(result.heat_capacity >= -1e-15)

    def test_entropy_nonnegative(self):
        """Entropy S >= 0."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([0.0, 1.0, 3.0])
        betas = np.linspace(0.1, 10.0, 15)
        result = compute_constellation_thermodynamics(energies, betas)
        # Allow tiny numerical noise
        assert np.all(result.entropy >= -1e-10)

    def test_critical_temperature_exists(self):
        """Critical temperature is where C_v peaks."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([0.0, 1.0, 4.0, 9.0])
        betas = np.linspace(0.01, 5.0, 200)
        result = compute_constellation_thermodynamics(energies, betas)
        assert result.critical_temperature > 0.0

    def test_log_sum_exp_stability(self):
        """Large energy values should not cause overflow."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([0.0, 1000.0, 2000.0])
        betas = np.array([1.0])
        result = compute_constellation_thermodynamics(energies, betas)
        assert np.isfinite(result.partition_function_log[0])
        assert np.isfinite(result.helmholtz_free_energy[0])

    def test_multiple_betas_output_shapes(self):
        """Output arrays match beta array length."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([0.0, 1.0, 2.0])
        betas = np.linspace(0.1, 5.0, 50)
        result = compute_constellation_thermodynamics(energies, betas)
        assert len(result.partition_function_log) == 50
        assert len(result.helmholtz_free_energy) == 50
        assert len(result.mean_energy) == 50
        assert len(result.energy_variance) == 50
        assert len(result.heat_capacity) == 50
        assert len(result.entropy) == 50

    def test_energy_variance_correct(self):
        """Var(E) = <E^2> - <E>^2."""
        from humeris.domain.multi_objective_design import compute_constellation_thermodynamics

        energies = np.array([1.0, 2.0, 3.0])
        betas = np.array([1.0])
        result = compute_constellation_thermodynamics(energies, betas)
        # Compute manually
        boltz = np.exp(-betas[0] * energies)
        Z = np.sum(boltz)
        probs = boltz / Z
        mean_e = np.sum(energies * probs)
        mean_e2 = np.sum(energies**2 * probs)
        var_e = mean_e2 - mean_e**2
        assert abs(result.energy_variance[0] - var_e) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# P47: Potential Flow Coverage via Conformal Mapping
# ═══════════════════════════════════════════════════════════════════


class TestPotentialFlowCoverageDataclass:
    """Verify PotentialFlowCoverage is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.coverage_optimization import PotentialFlowCoverage

        result = PotentialFlowCoverage(
            stagnation_points_latlon=((0.0, 0.0),),
            coverage_potential_grid=np.array([[1.0]]),
            stream_function_grid=np.array([[0.0]]),
            uniformity_metric=0.9,
            num_stagnation_points=1,
        )
        with pytest.raises(AttributeError):
            result.uniformity_metric = 0.5

    def test_all_fields_present(self):
        from humeris.domain.coverage_optimization import PotentialFlowCoverage

        result = PotentialFlowCoverage(
            stagnation_points_latlon=((0.0, 0.0),),
            coverage_potential_grid=np.array([[1.0]]),
            stream_function_grid=np.array([[0.0]]),
            uniformity_metric=0.9,
            num_stagnation_points=1,
        )
        assert hasattr(result, "stagnation_points_latlon")
        assert hasattr(result, "coverage_potential_grid")
        assert hasattr(result, "stream_function_grid")
        assert hasattr(result, "uniformity_metric")
        assert hasattr(result, "num_stagnation_points")


class TestComputePotentialFlowCoverage:
    """Test compute_potential_flow_coverage function."""

    def test_single_source(self):
        """Single source at north pole, stagnation at south pole."""
        from humeris.domain.coverage_optimization import compute_potential_flow_coverage

        lats = np.array([90.0])
        lons = np.array([0.0])
        strengths = np.array([1.0])
        result = compute_potential_flow_coverage(lats, lons, strengths, grid_resolution=10)
        assert result.num_stagnation_points >= 0
        assert result.coverage_potential_grid.shape[0] > 0
        assert result.coverage_potential_grid.shape[1] > 0

    def test_two_equal_sources(self):
        """Two equal sources produce stagnation point between them."""
        from humeris.domain.coverage_optimization import compute_potential_flow_coverage

        lats = np.array([45.0, -45.0])
        lons = np.array([0.0, 0.0])
        strengths = np.array([1.0, 1.0])
        result = compute_potential_flow_coverage(lats, lons, strengths, grid_resolution=10)
        # Two equal sources should produce at least one stagnation point
        assert result.num_stagnation_points >= 1

    def test_uniformity_metric_range(self):
        """Uniformity metric should be between 0 and 1."""
        from humeris.domain.coverage_optimization import compute_potential_flow_coverage

        lats = np.array([0.0, 60.0, -60.0])
        lons = np.array([0.0, 120.0, 240.0])
        strengths = np.array([1.0, 1.0, 1.0])
        result = compute_potential_flow_coverage(lats, lons, strengths, grid_resolution=8)
        assert 0.0 <= result.uniformity_metric <= 1.0

    def test_coverage_and_stream_same_shape(self):
        """Coverage potential and stream function grids have same shape."""
        from humeris.domain.coverage_optimization import compute_potential_flow_coverage

        lats = np.array([30.0, -30.0])
        lons = np.array([0.0, 180.0])
        strengths = np.array([1.0, 1.0])
        result = compute_potential_flow_coverage(lats, lons, strengths, grid_resolution=8)
        assert result.coverage_potential_grid.shape == result.stream_function_grid.shape

    def test_stagnation_latlon_within_bounds(self):
        """Stagnation points should have valid lat/lon."""
        from humeris.domain.coverage_optimization import compute_potential_flow_coverage

        lats = np.array([45.0, -45.0])
        lons = np.array([0.0, 180.0])
        strengths = np.array([1.0, 1.0])
        result = compute_potential_flow_coverage(lats, lons, strengths, grid_resolution=10)
        for lat, lon in result.stagnation_points_latlon:
            assert -90.0 <= lat <= 90.0
            assert -180.0 <= lon <= 360.0

    def test_stronger_source_larger_potential(self):
        """A stronger source produces higher potential at its location."""
        from humeris.domain.coverage_optimization import compute_potential_flow_coverage

        # Single strong source
        result_strong = compute_potential_flow_coverage(
            np.array([0.0]), np.array([0.0]), np.array([10.0]), grid_resolution=8,
        )
        result_weak = compute_potential_flow_coverage(
            np.array([0.0]), np.array([0.0]), np.array([1.0]), grid_resolution=8,
        )
        # Max potential should be higher for stronger source
        assert np.max(result_strong.coverage_potential_grid) > np.max(result_weak.coverage_potential_grid)

    def test_empty_sources(self):
        """Empty source arrays should produce zero grids."""
        from humeris.domain.coverage_optimization import compute_potential_flow_coverage

        result = compute_potential_flow_coverage(
            np.array([]), np.array([]), np.array([]), grid_resolution=5,
        )
        assert result.num_stagnation_points == 0
        assert np.all(result.coverage_potential_grid == 0.0)

    def test_grid_resolution_affects_shape(self):
        """Grid resolution controls output dimensions."""
        from humeris.domain.coverage_optimization import compute_potential_flow_coverage

        lats = np.array([0.0])
        lons = np.array([0.0])
        strengths = np.array([1.0])
        r5 = compute_potential_flow_coverage(lats, lons, strengths, grid_resolution=5)
        r10 = compute_potential_flow_coverage(lats, lons, strengths, grid_resolution=10)
        assert r10.coverage_potential_grid.size > r5.coverage_potential_grid.size


# ═══════════════════════════════════════════════════════════════════
# P50: Matroid Independence for Minimum Coverage Sets
# ═══════════════════════════════════════════════════════════════════


class TestCoverageMatroidDataclass:
    """Verify CoverageMatroid is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.coverage_optimization import CoverageMatroid

        result = CoverageMatroid(
            rank=3,
            minimum_basis=(0, 1, 2),
            redundant_satellites=(3,),
            redundancy_ratio=0.25,
            greedy_weight_basis=(0.5, 0.3, 0.2),
        )
        with pytest.raises(AttributeError):
            result.rank = 99

    def test_all_fields_present(self):
        from humeris.domain.coverage_optimization import CoverageMatroid

        result = CoverageMatroid(
            rank=3,
            minimum_basis=(0, 1, 2),
            redundant_satellites=(3,),
            redundancy_ratio=0.25,
            greedy_weight_basis=(0.5, 0.3, 0.2),
        )
        assert hasattr(result, "rank")
        assert hasattr(result, "minimum_basis")
        assert hasattr(result, "redundant_satellites")
        assert hasattr(result, "redundancy_ratio")
        assert hasattr(result, "greedy_weight_basis")


class TestComputeCoverageMatroid:
    """Test compute_coverage_matroid function."""

    def test_single_satellite_full_coverage(self):
        """One satellite covering everything: rank=1, no redundancy."""
        from humeris.domain.coverage_optimization import compute_coverage_matroid

        # coverage_matrix[i, j]: satellite i covers grid point j
        coverage = np.array([[1, 1, 1, 1]])
        result = compute_coverage_matroid(coverage)
        assert result.rank == 1
        assert len(result.redundant_satellites) == 0
        assert result.redundancy_ratio == 0.0

    def test_two_nonoverlapping_satellites(self):
        """Two disjoint satellites: both required, rank=2."""
        from humeris.domain.coverage_optimization import compute_coverage_matroid

        coverage = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])
        result = compute_coverage_matroid(coverage)
        assert result.rank == 2
        assert len(result.redundant_satellites) == 0

    def test_fully_redundant_satellite(self):
        """Third satellite is subset of first: it's redundant."""
        from humeris.domain.coverage_optimization import compute_coverage_matroid

        coverage = np.array([
            [1, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],  # subset of satellite 0
        ])
        result = compute_coverage_matroid(coverage)
        assert result.rank == 2
        assert 2 in result.redundant_satellites
        assert result.redundancy_ratio > 0.0

    def test_greedy_selects_largest_first(self):
        """Greedy picks satellite covering the most points first."""
        from humeris.domain.coverage_optimization import compute_coverage_matroid

        coverage = np.array([
            [1, 0, 0, 0],  # covers 1
            [1, 1, 1, 1],  # covers 4
            [0, 0, 1, 0],  # covers 1
        ])
        result = compute_coverage_matroid(coverage)
        # Greedy should pick satellite 1 first
        assert result.minimum_basis[0] == 1

    def test_empty_coverage(self):
        """No satellites: rank=0."""
        from humeris.domain.coverage_optimization import compute_coverage_matroid

        coverage = np.zeros((0, 5))
        result = compute_coverage_matroid(coverage)
        assert result.rank == 0
        assert len(result.minimum_basis) == 0

    def test_all_identical_satellites(self):
        """N identical satellites: rank=1, N-1 redundant."""
        from humeris.domain.coverage_optimization import compute_coverage_matroid

        coverage = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])
        result = compute_coverage_matroid(coverage)
        assert result.rank == 1
        assert len(result.redundant_satellites) == 3
        assert abs(result.redundancy_ratio - 0.75) < 1e-10

    def test_rank_leq_num_satellites(self):
        """Rank cannot exceed number of satellites."""
        from humeris.domain.coverage_optimization import compute_coverage_matroid

        rng = np.random.RandomState(42)
        coverage = rng.randint(0, 2, size=(5, 20))
        result = compute_coverage_matroid(coverage)
        assert result.rank <= 5

    def test_greedy_weight_basis_descending(self):
        """Greedy weights should be in descending order."""
        from humeris.domain.coverage_optimization import compute_coverage_matroid

        coverage = np.array([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ])
        result = compute_coverage_matroid(coverage)
        weights = result.greedy_weight_basis
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1] - 1e-10


# ═══════════════════════════════════════════════════════════════════
# P54: Neural Field Equation for Coverage Demand
# ═══════════════════════════════════════════════════════════════════


class TestDemandFieldEquilibriumDataclass:
    """Verify DemandFieldEquilibrium is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.coverage_optimization import DemandFieldEquilibrium

        result = DemandFieldEquilibrium(
            demand_field=np.array([[1.0]]),
            supply_field=np.array([[0.5]]),
            deficit_field=np.array([[0.5]]),
            peak_deficit_latlon=(0.0, 0.0),
            total_deficit=0.5,
            demand_concentration_index=0.8,
        )
        with pytest.raises(AttributeError):
            result.total_deficit = 0.0

    def test_all_fields_present(self):
        from humeris.domain.coverage_optimization import DemandFieldEquilibrium

        result = DemandFieldEquilibrium(
            demand_field=np.array([[1.0]]),
            supply_field=np.array([[0.5]]),
            deficit_field=np.array([[0.5]]),
            peak_deficit_latlon=(0.0, 0.0),
            total_deficit=0.5,
            demand_concentration_index=0.8,
        )
        assert hasattr(result, "demand_field")
        assert hasattr(result, "supply_field")
        assert hasattr(result, "deficit_field")
        assert hasattr(result, "peak_deficit_latlon")
        assert hasattr(result, "total_deficit")
        assert hasattr(result, "demand_concentration_index")


class TestComputeDemandFieldEquilibrium:
    """Test compute_demand_field_equilibrium function."""

    def test_zero_demand_zero_deficit(self):
        """Zero external input produces zero deficit."""
        from humeris.domain.coverage_optimization import compute_demand_field_equilibrium

        n = 10
        supply = np.ones((n, 2 * n)) * 0.5
        demand_ext = np.zeros((n, 2 * n))
        result = compute_demand_field_equilibrium(supply, demand_ext, grid_resolution=n)
        assert result.total_deficit < 1e-10

    def test_deficit_nonnegative(self):
        """Deficit field should be nonnegative everywhere."""
        from humeris.domain.coverage_optimization import compute_demand_field_equilibrium

        n = 10
        supply = np.random.RandomState(42).rand(n, 2 * n) * 0.3
        demand_ext = np.random.RandomState(43).rand(n, 2 * n)
        result = compute_demand_field_equilibrium(supply, demand_ext, grid_resolution=n)
        assert np.all(result.deficit_field >= -1e-10)

    def test_supply_exceeds_demand_no_deficit(self):
        """When supply >> demand, deficit should be near zero."""
        from humeris.domain.coverage_optimization import compute_demand_field_equilibrium

        n = 8
        supply = np.ones((n, 2 * n)) * 10.0
        demand_ext = np.ones((n, 2 * n)) * 0.01
        result = compute_demand_field_equilibrium(supply, demand_ext, grid_resolution=n)
        assert result.total_deficit < 0.1

    def test_peak_deficit_location_valid(self):
        """Peak deficit lat/lon should be within valid ranges."""
        from humeris.domain.coverage_optimization import compute_demand_field_equilibrium

        n = 10
        supply = np.zeros((n, 2 * n))
        demand_ext = np.random.RandomState(44).rand(n, 2 * n)
        result = compute_demand_field_equilibrium(supply, demand_ext, grid_resolution=n)
        lat, lon = result.peak_deficit_latlon
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 360.0

    def test_demand_concentration_index_range(self):
        """Concentration index in [0, 1]."""
        from humeris.domain.coverage_optimization import compute_demand_field_equilibrium

        n = 10
        supply = np.random.RandomState(45).rand(n, 2 * n) * 0.2
        demand_ext = np.random.RandomState(46).rand(n, 2 * n)
        result = compute_demand_field_equilibrium(supply, demand_ext, grid_resolution=n)
        assert 0.0 <= result.demand_concentration_index <= 1.0 + 1e-10

    def test_demand_field_shape_matches_supply(self):
        """Demand field should have same shape as supply."""
        from humeris.domain.coverage_optimization import compute_demand_field_equilibrium

        n = 12
        supply = np.ones((n, 2 * n))
        demand_ext = np.ones((n, 2 * n))
        result = compute_demand_field_equilibrium(supply, demand_ext, grid_resolution=n)
        assert result.demand_field.shape == supply.shape
        assert result.supply_field.shape == supply.shape
        assert result.deficit_field.shape == supply.shape

    def test_concentrated_demand_high_concentration(self):
        """Demand concentrated in one spot should yield high concentration index."""
        from humeris.domain.coverage_optimization import compute_demand_field_equilibrium

        n = 10
        supply = np.zeros((n, 2 * n))
        demand_ext = np.zeros((n, 2 * n))
        demand_ext[n // 2, n] = 100.0  # Single hotspot
        result = compute_demand_field_equilibrium(supply, demand_ext, grid_resolution=n)
        # With highly concentrated demand, index should be relatively high
        assert result.demand_concentration_index > 0.3

    def test_total_deficit_consistent_with_field(self):
        """Total deficit should equal sum of deficit field."""
        from humeris.domain.coverage_optimization import compute_demand_field_equilibrium

        n = 8
        supply = np.random.RandomState(47).rand(n, 2 * n) * 0.3
        demand_ext = np.random.RandomState(48).rand(n, 2 * n)
        result = compute_demand_field_equilibrium(supply, demand_ext, grid_resolution=n)
        field_sum = float(np.sum(result.deficit_field))
        assert abs(result.total_deficit - field_sum) < 1e-8


# ═══════════════════════════════════════════════════════════════════
# P64: Ergodic Coverage Partition
# ═══════════════════════════════════════════════════════════════════


class TestErgodicPartitionDataclass:
    """Verify ErgodicPartition is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.coverage_optimization import ErgodicPartition

        result = ErgodicPartition(
            coverage_time_averages=np.array([[0.5]]),
            ergodicity_defect=0.1,
            num_ergodic_components=1,
            component_boundaries=np.array([[0]]),
            mixing_time_orbits=5.0,
            is_ergodic=True,
        )
        with pytest.raises(AttributeError):
            result.is_ergodic = False

    def test_all_fields_present(self):
        from humeris.domain.coverage_optimization import ErgodicPartition

        result = ErgodicPartition(
            coverage_time_averages=np.array([[0.5]]),
            ergodicity_defect=0.1,
            num_ergodic_components=1,
            component_boundaries=np.array([[0]]),
            mixing_time_orbits=5.0,
            is_ergodic=True,
        )
        assert hasattr(result, "coverage_time_averages")
        assert hasattr(result, "ergodicity_defect")
        assert hasattr(result, "num_ergodic_components")
        assert hasattr(result, "component_boundaries")
        assert hasattr(result, "mixing_time_orbits")
        assert hasattr(result, "is_ergodic")


class TestComputeErgodicPartition:
    """Test compute_ergodic_partition function."""

    def test_uniform_coverage_is_ergodic(self):
        """Uniform coverage across time should be ergodic."""
        from humeris.domain.coverage_optimization import compute_ergodic_partition

        n_grid = 8
        n_time = 20
        # Constant coverage everywhere
        coverage_timeseries = np.ones((n_time, n_grid, 2 * n_grid)) * 0.5
        result = compute_ergodic_partition(coverage_timeseries, orbit_period_steps=5)
        assert result.is_ergodic
        assert result.ergodicity_defect < 0.1

    def test_defect_is_max_minus_min(self):
        """Ergodicity defect = max(f_g) - min(f_g)."""
        from humeris.domain.coverage_optimization import compute_ergodic_partition

        n_grid = 6
        n_time = 10
        rng = np.random.RandomState(42)
        coverage = rng.rand(n_time, n_grid, 2 * n_grid)
        result = compute_ergodic_partition(coverage, orbit_period_steps=3)
        avgs = result.coverage_time_averages
        expected_defect = float(np.max(avgs) - np.min(avgs))
        assert abs(result.ergodicity_defect - expected_defect) < 1e-10

    def test_two_distinct_regions(self):
        """Coverage that separates into two regions yields >= 2 components."""
        from humeris.domain.coverage_optimization import compute_ergodic_partition

        n_grid = 10
        n_time = 15
        coverage = np.zeros((n_time, n_grid, 2 * n_grid))
        # Region 1: top half always covered
        coverage[:, :n_grid // 2, :] = 1.0
        # Region 2: bottom half never covered
        coverage[:, n_grid // 2:, :] = 0.0
        result = compute_ergodic_partition(coverage, orbit_period_steps=5)
        assert result.num_ergodic_components >= 2
        assert result.ergodicity_defect > 0.5

    def test_mixing_time_positive(self):
        """Mixing time should be positive."""
        from humeris.domain.coverage_optimization import compute_ergodic_partition

        n_grid = 6
        n_time = 20
        rng = np.random.RandomState(43)
        coverage = rng.rand(n_time, n_grid, 2 * n_grid)
        result = compute_ergodic_partition(coverage, orbit_period_steps=4)
        assert result.mixing_time_orbits > 0.0

    def test_component_boundaries_shape(self):
        """Component boundaries should have same spatial shape as averages."""
        from humeris.domain.coverage_optimization import compute_ergodic_partition

        n_grid = 6
        n_time = 10
        rng = np.random.RandomState(44)
        coverage = rng.rand(n_time, n_grid, 2 * n_grid)
        result = compute_ergodic_partition(coverage, orbit_period_steps=3)
        assert result.component_boundaries.shape == result.coverage_time_averages.shape

    def test_single_timestep(self):
        """Single timestep should still work."""
        from humeris.domain.coverage_optimization import compute_ergodic_partition

        n_grid = 5
        coverage = np.ones((1, n_grid, 2 * n_grid)) * 0.3
        result = compute_ergodic_partition(coverage, orbit_period_steps=1)
        assert result.mixing_time_orbits >= 1.0
        assert result.num_ergodic_components >= 1

    def test_coverage_time_averages_in_range(self):
        """Time averages should be within [0, 1] if input is [0, 1]."""
        from humeris.domain.coverage_optimization import compute_ergodic_partition

        n_grid = 6
        n_time = 15
        rng = np.random.RandomState(45)
        coverage = rng.rand(n_time, n_grid, 2 * n_grid)
        result = compute_ergodic_partition(coverage, orbit_period_steps=5)
        assert np.all(result.coverage_time_averages >= -1e-10)
        assert np.all(result.coverage_time_averages <= 1.0 + 1e-10)

    def test_ergodicity_defect_nonnegative(self):
        """Defect is max - min, so always >= 0."""
        from humeris.domain.coverage_optimization import compute_ergodic_partition

        n_grid = 5
        n_time = 8
        rng = np.random.RandomState(46)
        coverage = rng.rand(n_time, n_grid, 2 * n_grid)
        result = compute_ergodic_partition(coverage, orbit_period_steps=2)
        assert result.ergodicity_defect >= 0.0
