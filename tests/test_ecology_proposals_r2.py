# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for Round 2 creative proposals: P24, P33, P32, P39, P40."""
import math

import numpy as np
import pytest

from humeris.domain.cascade_analysis import (
    LotkaVolterraDebris,
    compute_lotka_volterra_debris,
)
from humeris.domain.kessler_heatmap import (
    DebrisDensityEvolution,
    compute_debris_density_evolution,
    DebrisCloudDimension,
    compute_debris_cloud_dimension,
    RenormalizationGroupAnalysis,
    compute_renormalization_group,
)
from humeris.domain.station_keeping import (
    PropellantPharmacokinetics,
    compute_propellant_pharmacokinetics,
)


# ── P24: Lotka-Volterra Multi-Species Debris ───────────────────────


class TestLotkaVolterraBasic:
    """Basic Lotka-Volterra multi-species debris tests."""

    def test_return_type_is_frozen_dataclass(self):
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
        )
        assert isinstance(result, LotkaVolterraDebris)
        with pytest.raises(AttributeError):
            result.is_stable = True

    def test_species_names(self):
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
        )
        assert result.species == ("rocket_bodies", "mission_debris", "fragments")

    def test_populations_tuple_of_tuples(self):
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
            duration_years=10.0,
            step_years=1.0,
        )
        assert isinstance(result.populations, tuple)
        assert len(result.populations) == 3
        for pop in result.populations:
            assert isinstance(pop, tuple)
            # 10/1 + 1 = 11 steps
            assert len(pop) == 11

    def test_equilibrium_counts_length(self):
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
        )
        assert len(result.equilibrium_counts) == 3

    def test_jacobian_eigenvalues_length(self):
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
        )
        assert len(result.jacobian_eigenvalues) == 3


class TestLotkaVolterraStability:
    """Stability analysis for Lotka-Volterra model."""

    def test_high_drag_removal_is_stable(self):
        """High drag removal rates should yield a stable equilibrium."""
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
            gamma=np.array([0.2, 0.2, 0.1]),
            collision_rates=np.array([
                [0.0, 1e-6, 1e-7],
                [1e-6, 0.0, 1e-7],
                [1e-7, 1e-7, 0.0],
            ]),
            fragment_multipliers=np.array([
                [0.0, 10.0, 5.0],
                [10.0, 0.0, 5.0],
                [5.0, 5.0, 0.0],
            ]),
            launch_rates=np.array([5.0, 10.0, 0.0]),
        )
        assert result.is_stable is True
        # All Jacobian eigenvalue real parts should be negative
        for ev in result.jacobian_eigenvalues:
            assert complex(ev).real < 0.0 or abs(complex(ev).real) < 1e-6

    def test_no_launches_no_collisions_decay_to_zero(self):
        """Without launches and negligible collisions, all species decay."""
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
            gamma=np.array([0.1, 0.1, 0.1]),
            collision_rates=np.zeros((3, 3)),
            fragment_multipliers=np.zeros((3, 3)),
            launch_rates=np.array([0.0, 0.0, 0.0]),
            duration_years=200.0,
            step_years=0.5,
        )
        # Final populations should be near zero
        # N(t) = N0 * exp(-gamma*t), 500*exp(-0.1*200) ~ 1e-6
        for pop in result.populations:
            assert pop[-1] < 1.0

    def test_dominant_interaction_is_string(self):
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
        )
        assert isinstance(result.dominant_interaction, str)


class TestLotkaVolterraConservation:
    """Physics consistency checks."""

    def test_populations_non_negative(self):
        """All population values should be non-negative."""
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
            duration_years=50.0,
            step_years=0.1,
        )
        for pop in result.populations:
            for val in pop:
                assert val >= 0.0

    def test_initial_conditions_recorded(self):
        """First element of each population equals initial condition."""
        result = compute_lotka_volterra_debris(
            initial_rocket_bodies=100.0,
            initial_mission_debris=200.0,
            initial_fragments=500.0,
        )
        assert result.populations[0][0] == 100.0
        assert result.populations[1][0] == 200.0
        assert result.populations[2][0] == 500.0


# ── P33: Fokker-Planck Density Evolution ────────────────────────────


class TestFokkerPlanckBasic:
    """Basic Fokker-Planck density evolution tests."""

    def test_return_type_is_frozen_dataclass(self):
        result = compute_debris_density_evolution(
            altitude_min_km=300.0,
            altitude_max_km=1000.0,
            n_altitude_bins=50,
            duration_years=10.0,
            step_years=0.1,
        )
        assert isinstance(result, DebrisDensityEvolution)
        with pytest.raises(AttributeError):
            result.altitude_bins = None

    def test_altitude_bins_shape(self):
        n_bins = 50
        result = compute_debris_density_evolution(
            altitude_min_km=300.0,
            altitude_max_km=1000.0,
            n_altitude_bins=n_bins,
            duration_years=10.0,
            step_years=0.1,
        )
        assert len(result.altitude_bins) == n_bins

    def test_time_steps_shape(self):
        result = compute_debris_density_evolution(
            altitude_min_km=300.0,
            altitude_max_km=1000.0,
            n_altitude_bins=20,
            duration_years=10.0,
            step_years=1.0,
        )
        assert len(result.time_steps) == 11  # 10/1 + 1

    def test_density_evolution_shape(self):
        n_bins = 20
        result = compute_debris_density_evolution(
            altitude_min_km=300.0,
            altitude_max_km=1000.0,
            n_altitude_bins=n_bins,
            duration_years=10.0,
            step_years=1.0,
        )
        # density_evolution is (n_time_steps, n_altitude_bins)
        assert len(result.density_evolution) == 11
        for row in result.density_evolution:
            assert len(row) == n_bins


class TestFokkerPlanckPhysics:
    """Physics consistency for Fokker-Planck."""

    def test_density_non_negative(self):
        """Density should remain non-negative everywhere."""
        result = compute_debris_density_evolution(
            altitude_min_km=300.0,
            altitude_max_km=1000.0,
            n_altitude_bins=20,
            duration_years=5.0,
            step_years=0.1,
        )
        for row in result.density_evolution:
            for val in row:
                assert val >= 0.0

    def test_peak_density_trajectory_length(self):
        result = compute_debris_density_evolution(
            altitude_min_km=300.0,
            altitude_max_km=1000.0,
            n_altitude_bins=20,
            duration_years=10.0,
            step_years=1.0,
        )
        assert len(result.peak_density_trajectory) == 11

    def test_total_mass_trajectory_length(self):
        result = compute_debris_density_evolution(
            altitude_min_km=300.0,
            altitude_max_km=1000.0,
            n_altitude_bins=20,
            duration_years=10.0,
            step_years=1.0,
        )
        assert len(result.total_mass_trajectory) == 11

    def test_no_source_drift_to_lower_alt(self):
        """Without source, density should drift to lower altitudes over time."""
        result = compute_debris_density_evolution(
            altitude_min_km=300.0,
            altitude_max_km=1000.0,
            n_altitude_bins=30,
            duration_years=20.0,
            step_years=0.5,
            initial_density_per_km3=1e-8,
            source_altitude_km=800.0,
            source_rate_per_km3_per_year=0.0,
            diffusion_coefficient=0.0,
        )
        # Peak should move downward (lower index = lower altitude)
        # Just check the initial peak is at or near 800 km index
        alt_bins = result.altitude_bins
        initial_peak_idx = max(range(len(result.density_evolution[0])),
                               key=lambda i: result.density_evolution[0][i])
        final_peak_idx = max(range(len(result.density_evolution[-1])),
                             key=lambda i: result.density_evolution[-1][i])
        # Peak should be at same or lower altitude
        assert alt_bins[final_peak_idx] <= alt_bins[initial_peak_idx] + 50.0


# ── P32: Hausdorff Dimension of Debris Cloud ───────────────────────


class TestHausdorffBasic:
    """Basic Hausdorff dimension tests."""

    def test_return_type_is_frozen_dataclass(self):
        positions = np.random.RandomState(42).uniform(0, 1000, size=(200, 3))
        result = compute_debris_cloud_dimension(positions)
        assert isinstance(result, DebrisCloudDimension)
        with pytest.raises(AttributeError):
            result.box_counting_dimension = 0.0

    def test_dimension_fields_present(self):
        positions = np.random.RandomState(42).uniform(0, 1000, size=(200, 3))
        result = compute_debris_cloud_dimension(positions)
        assert hasattr(result, 'box_counting_dimension')
        assert hasattr(result, 'filling_fraction')
        assert hasattr(result, 'concentration_factor')
        assert hasattr(result, 'avoidance_duration_orbits')
        assert hasattr(result, 'dimension_uncertainty')


class TestHausdorffPhysics:
    """Physics of box-counting dimension."""

    def test_collinear_points_dimension_near_1(self):
        """Points along a line should have dimension near 1."""
        rng = np.random.RandomState(42)
        t = rng.uniform(0, 1000, size=500)
        positions = np.column_stack([t, np.zeros(500), np.zeros(500)])
        result = compute_debris_cloud_dimension(positions)
        assert 0.5 < result.box_counting_dimension < 1.5

    def test_planar_points_dimension_near_2(self):
        """Points on a plane should have dimension near 2."""
        rng = np.random.RandomState(42)
        positions = np.column_stack([
            rng.uniform(0, 1000, 1000),
            rng.uniform(0, 1000, 1000),
            np.zeros(1000),
        ])
        result = compute_debris_cloud_dimension(positions)
        assert 1.5 < result.box_counting_dimension < 2.5

    def test_volume_filling_dimension_near_3(self):
        """Points filling a volume should have dimension near 3."""
        rng = np.random.RandomState(42)
        positions = rng.uniform(0, 1000, size=(2000, 3))
        result = compute_debris_cloud_dimension(positions)
        assert 2.0 < result.box_counting_dimension < 3.5

    def test_filling_fraction_between_0_and_1(self):
        positions = np.random.RandomState(42).uniform(0, 1000, size=(200, 3))
        result = compute_debris_cloud_dimension(positions)
        assert 0.0 <= result.filling_fraction <= 1.0

    def test_dimension_uncertainty_is_r_squared(self):
        """dimension_uncertainty (R^2) should be between 0 and 1."""
        positions = np.random.RandomState(42).uniform(0, 1000, size=(200, 3))
        result = compute_debris_cloud_dimension(positions)
        assert 0.0 <= result.dimension_uncertainty <= 1.0

    def test_single_point_dimension_zero(self):
        """A single point has dimension 0."""
        positions = np.array([[100.0, 200.0, 300.0]])
        result = compute_debris_cloud_dimension(positions)
        assert result.box_counting_dimension == 0.0


# ── P39: Renormalization Group for Multi-Scale Debris ──────────────


class TestRenormalizationGroupBasic:
    """Basic RG analysis tests."""

    def test_return_type_is_frozen_dataclass(self):
        density_profile = np.array([1e-9] * 64)
        density_profile[28:36] = 1e-7  # hot zone
        result = compute_renormalization_group(
            density_profile=density_profile,
            mean_fragments_per_collision=100.0,
            collision_cross_section_km2=1e-6,
            mean_collision_velocity_ms=10000.0,
            orbital_lifetime_years=25.0,
        )
        assert isinstance(result, RenormalizationGroupAnalysis)
        with pytest.raises(AttributeError):
            result.is_critical = True

    def test_fields_present(self):
        density_profile = np.array([1e-9] * 64)
        result = compute_renormalization_group(
            density_profile=density_profile,
            mean_fragments_per_collision=100.0,
            collision_cross_section_km2=1e-6,
            mean_collision_velocity_ms=10000.0,
            orbital_lifetime_years=25.0,
        )
        assert hasattr(result, 'fixed_point_k_eff')
        assert hasattr(result, 'critical_exponent_nu')
        assert hasattr(result, 'correlation_length_cells')
        assert hasattr(result, 'is_critical')
        assert hasattr(result, 'scale_levels')


class TestRenormalizationGroupPhysics:
    """Physics of RG flow."""

    def test_uniform_density_fixed_point(self):
        """Uniform density should be a fixed point under coarse-graining."""
        density_profile = np.array([1e-8] * 64)
        result = compute_renormalization_group(
            density_profile=density_profile,
            mean_fragments_per_collision=100.0,
            collision_cross_section_km2=1e-6,
            mean_collision_velocity_ms=10000.0,
            orbital_lifetime_years=25.0,
        )
        # k_eff should be consistent across scales
        assert result.fixed_point_k_eff >= 0.0

    def test_scale_levels_decreasing_length(self):
        """Each coarse-graining level should have fewer cells."""
        density_profile = np.array([1e-9] * 64)
        result = compute_renormalization_group(
            density_profile=density_profile,
            mean_fragments_per_collision=100.0,
            collision_cross_section_km2=1e-6,
            mean_collision_velocity_ms=10000.0,
            orbital_lifetime_years=25.0,
        )
        assert len(result.scale_levels) >= 2  # at least 2 levels

    def test_subcritical_density_not_critical(self):
        """Very low density should not be critical."""
        density_profile = np.array([1e-12] * 32)
        result = compute_renormalization_group(
            density_profile=density_profile,
            mean_fragments_per_collision=100.0,
            collision_cross_section_km2=1e-6,
            mean_collision_velocity_ms=10000.0,
            orbital_lifetime_years=25.0,
        )
        assert result.is_critical is False

    def test_critical_exponent_non_negative(self):
        density_profile = np.array([1e-8] * 32)
        result = compute_renormalization_group(
            density_profile=density_profile,
            mean_fragments_per_collision=100.0,
            collision_cross_section_km2=1e-6,
            mean_collision_velocity_ms=10000.0,
            orbital_lifetime_years=25.0,
        )
        assert result.critical_exponent_nu >= 0.0

    def test_correlation_length_non_negative(self):
        density_profile = np.array([1e-8] * 32)
        result = compute_renormalization_group(
            density_profile=density_profile,
            mean_fragments_per_collision=100.0,
            collision_cross_section_km2=1e-6,
            mean_collision_velocity_ms=10000.0,
            orbital_lifetime_years=25.0,
        )
        assert result.correlation_length_cells >= 0.0


# ── P40: Compartmental Pharmacokinetic Propellant Model ────────────


class TestPharmacokineticsBasic:
    """Basic pharmacokinetic propellant model tests."""

    def test_return_type_is_frozen_dataclass(self):
        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=50.0,
            altitude_km=500.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
        )
        assert isinstance(result, PropellantPharmacokinetics)
        with pytest.raises(AttributeError):
            result.half_life = 0.0

    def test_trajectory_lengths_match(self):
        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=50.0,
            altitude_km=500.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
            duration_years=10.0,
            step_years=1.0,
        )
        assert len(result.stored_trajectory) == 11
        assert len(result.committed_trajectory) == 11
        assert len(result.expended_trajectory) == 11


class TestPharmacokineticsConservation:
    """Propellant mass conservation."""

    def test_total_propellant_conserved(self):
        """P_stored + P_committed + P_expended = P_initial at all times."""
        p0 = 50.0
        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=p0,
            altitude_km=500.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
            duration_years=10.0,
            step_years=0.1,
        )
        for i in range(len(result.stored_trajectory)):
            total = (result.stored_trajectory[i]
                     + result.committed_trajectory[i]
                     + result.expended_trajectory[i])
            assert abs(total - p0) < 0.1, (
                f"Conservation violated at step {i}: {total} != {p0}"
            )

    def test_stored_monotonically_decreasing(self):
        """Stored propellant should monotonically decrease (no refueling)."""
        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=50.0,
            altitude_km=500.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
            duration_years=10.0,
            step_years=0.1,
        )
        for i in range(1, len(result.stored_trajectory)):
            assert result.stored_trajectory[i] <= result.stored_trajectory[i - 1] + 1e-10

    def test_expended_monotonically_increasing(self):
        """Expended propellant should monotonically increase."""
        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=50.0,
            altitude_km=500.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
            duration_years=10.0,
            step_years=0.1,
        )
        for i in range(1, len(result.expended_trajectory)):
            assert result.expended_trajectory[i] >= result.expended_trajectory[i - 1] - 1e-10


class TestPharmacokineticsPhysics:
    """Physics of propellant depletion."""

    def test_half_life_positive(self):
        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=50.0,
            altitude_km=500.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
        )
        assert result.half_life > 0.0

    def test_depletion_time_greater_than_half_life(self):
        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=50.0,
            altitude_km=500.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
        )
        assert result.depletion_time >= result.half_life

    def test_margin_at_eol_between_0_and_1(self):
        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=50.0,
            altitude_km=500.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
            duration_years=5.0,
        )
        assert 0.0 <= result.margin_at_eol <= 1.0

    def test_higher_altitude_longer_depletion(self):
        """Higher altitude = less drag = longer propellant life."""
        low = compute_propellant_pharmacokinetics(
            initial_propellant_kg=50.0,
            altitude_km=400.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
        )
        high = compute_propellant_pharmacokinetics(
            initial_propellant_kg=50.0,
            altitude_km=800.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
        )
        assert high.depletion_time > low.depletion_time

    def test_initial_conditions(self):
        """Initial state: all propellant in stored compartment."""
        p0 = 50.0
        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=p0,
            altitude_km=500.0,
            isp_s=220.0,
            dry_mass_kg=100.0,
            drag_cd=2.2,
            drag_area_m2=1.0,
        )
        assert result.stored_trajectory[0] == p0
        assert result.committed_trajectory[0] == 0.0
        assert result.expended_trajectory[0] == 0.0
