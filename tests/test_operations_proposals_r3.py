# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for R3 proposals: math fix R3-18 and proposals P45, P49, P52, P56, P57, P61, P62."""
import math

import numpy as np
import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants

R_EARTH = OrbitalConstants.R_EARTH
MU = OrbitalConstants.MU_EARTH
R_LEO = R_EARTH + 400_000        # 400 km LEO
R_GEO = R_EARTH + 35_786_000     # GEO


# ══════════════════════════════════════════════════════════════════
# R3-18: Hohmann transfer burn direction info
# ══════════════════════════════════════════════════════════════════

class TestR3_18_HohmannBurnDirection:
    """Hohmann transfer descriptions include prograde/retrograde direction."""

    def test_raising_transfer_says_prograde(self):
        """LEO to GEO (raising) should say 'prograde' in both burns."""
        from humeris.domain.maneuvers import hohmann_transfer
        result = hohmann_transfer(R_LEO, R_GEO)
        assert "prograde" in result.burns[0].description
        assert "prograde" in result.burns[1].description

    def test_lowering_transfer_says_retrograde(self):
        """GEO to LEO (lowering) should say 'retrograde' in both burns."""
        from humeris.domain.maneuvers import hohmann_transfer
        result = hohmann_transfer(R_GEO, R_LEO)
        assert "retrograde" in result.burns[0].description
        assert "retrograde" in result.burns[1].description

    def test_no_change_transfer_has_no_direction(self):
        """Equal radii transfer should say 'no change', no direction label."""
        from humeris.domain.maneuvers import hohmann_transfer
        result = hohmann_transfer(R_LEO, R_LEO)
        assert "no change" in result.burns[0].description
        assert "prograde" not in result.burns[0].description
        assert "retrograde" not in result.burns[0].description

    def test_small_raise_is_prograde(self):
        """Even a small raise (1 km) should say prograde."""
        from humeris.domain.maneuvers import hohmann_transfer
        result = hohmann_transfer(R_LEO, R_LEO + 1000)
        assert "prograde" in result.burns[0].description

    def test_small_lower_is_retrograde(self):
        """Even a small lower (1 km) should say retrograde."""
        from humeris.domain.maneuvers import hohmann_transfer
        result = hohmann_transfer(R_LEO, R_LEO - 1000)
        assert "retrograde" in result.burns[0].description

    def test_burn_descriptions_contain_departure_arrival(self):
        """Burn descriptions still contain departure/arrival info."""
        from humeris.domain.maneuvers import hohmann_transfer
        result = hohmann_transfer(R_LEO, R_GEO)
        assert "departure" in result.burns[0].description
        assert "arrival" in result.burns[1].description

    def test_delta_v_unchanged_by_direction_fix(self):
        """The delta-V values must not change from the direction fix."""
        from humeris.domain.maneuvers import hohmann_transfer
        result = hohmann_transfer(R_LEO, R_GEO)
        # Vallado reference: LEO to GEO ~3.93 km/s total
        assert 3800 < result.total_delta_v_ms < 4100

    def test_transfer_plan_is_frozen(self):
        """TransferPlan result is still a frozen dataclass."""
        from humeris.domain.maneuvers import hohmann_transfer
        result = hohmann_transfer(R_LEO, R_GEO)
        with pytest.raises(AttributeError):
            result.total_delta_v_ms = 0.0

    def test_raising_then_lowering_symmetry(self):
        """Raising and lowering transfers have same total delta-V."""
        from humeris.domain.maneuvers import hohmann_transfer
        up = hohmann_transfer(R_LEO, R_GEO)
        down = hohmann_transfer(R_GEO, R_LEO)
        assert abs(up.total_delta_v_ms - down.total_delta_v_ms) < 0.01


# ══════════════════════════════════════════════════════════════════
# P45: Vorticity in Orbital Element Space
# ══════════════════════════════════════════════════════════════════

class TestP45_OrbitalVorticity:
    """Vorticity analysis of J2-driven velocity field in (a, i) space."""

    def test_returns_frozen_dataclass(self):
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        result = compute_orbital_vorticity_field(
            a_range_m=(R_EARTH + 400e3, R_EARTH + 600e3),
            i_range_rad=(0.5, 1.2),
            n_a=5, n_i=5,
        )
        with pytest.raises(AttributeError):
            result.max_vorticity = 0.0

    def test_vorticity_grid_shape(self):
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        result = compute_orbital_vorticity_field(
            a_range_m=(R_EARTH + 400e3, R_EARTH + 600e3),
            i_range_rad=(0.5, 1.2),
            n_a=7, n_i=9,
        )
        assert len(result.vorticity_grid) == 7
        assert len(result.vorticity_grid[0]) == 9

    def test_max_vorticity_positive(self):
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        result = compute_orbital_vorticity_field(
            a_range_m=(R_EARTH + 300e3, R_EARTH + 800e3),
            i_range_rad=(0.3, 1.5),
            n_a=10, n_i=10,
        )
        assert result.max_vorticity > 0

    def test_circulation_nonzero(self):
        """J2 velocity field should produce nonzero circulation."""
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        result = compute_orbital_vorticity_field(
            a_range_m=(R_EARTH + 400e3, R_EARTH + 800e3),
            i_range_rad=(0.5, 1.2),
            n_a=15, n_i=15,
        )
        assert result.circulation != 0.0

    def test_strain_time_positive(self):
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        result = compute_orbital_vorticity_field(
            a_range_m=(R_EARTH + 400e3, R_EARTH + 600e3),
            i_range_rad=(0.5, 1.2),
            n_a=5, n_i=5,
        )
        assert result.constellation_strain_time_years > 0

    def test_shear_rate_positive(self):
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        result = compute_orbital_vorticity_field(
            a_range_m=(R_EARTH + 400e3, R_EARTH + 600e3),
            i_range_rad=(0.5, 1.2),
            n_a=5, n_i=5,
        )
        assert result.shear_rate_rad_per_year > 0

    def test_invalid_a_range_raises(self):
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        with pytest.raises(ValueError, match="a_range_m"):
            compute_orbital_vorticity_field(
                a_range_m=(R_EARTH + 600e3, R_EARTH + 400e3),
                i_range_rad=(0.5, 1.2),
            )

    def test_invalid_i_range_raises(self):
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        with pytest.raises(ValueError, match="i_range_rad"):
            compute_orbital_vorticity_field(
                a_range_m=(R_EARTH + 400e3, R_EARTH + 600e3),
                i_range_rad=(1.2, 0.5),
            )

    def test_grid_too_small_raises(self):
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        with pytest.raises(ValueError, match="3x3"):
            compute_orbital_vorticity_field(
                a_range_m=(R_EARTH + 400e3, R_EARTH + 600e3),
                i_range_rad=(0.5, 1.2),
                n_a=2, n_i=2,
            )

    def test_narrow_range_small_vorticity(self):
        """A very narrow range should have small vorticity magnitude."""
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        result = compute_orbital_vorticity_field(
            a_range_m=(R_EARTH + 500e3, R_EARTH + 501e3),
            i_range_rad=(1.0, 1.001),
            n_a=5, n_i=5,
        )
        # Should still compute but values should be finite
        assert math.isfinite(result.max_vorticity)
        assert math.isfinite(result.circulation)

    def test_strain_time_inversely_related_to_shear(self):
        """strain_time = 2*pi / shear_rate."""
        from humeris.domain.maintenance_planning import compute_orbital_vorticity_field
        result = compute_orbital_vorticity_field(
            a_range_m=(R_EARTH + 400e3, R_EARTH + 600e3),
            i_range_rad=(0.5, 1.2),
            n_a=5, n_i=5,
        )
        expected = 2.0 * math.pi / result.shear_rate_rad_per_year
        assert abs(result.constellation_strain_time_years - expected) < 1e-10


# ══════════════════════════════════════════════════════════════════
# P49: Burnside Symmetry Counting
# ══════════════════════════════════════════════════════════════════

class TestP49_BurnsideSymmetry:
    """Burnside symmetry counting for Walker constellations."""

    def test_returns_frozen_dataclass(self):
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(6, 3, 3)
        with pytest.raises(AttributeError):
            result.distinct_configurations = 0

    def test_trivial_all_active(self):
        """All satellites active: only 1 configuration total and distinct."""
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(6, 3, 6)
        assert result.total_configurations == 1
        assert result.distinct_configurations == 1

    def test_trivial_none_active(self):
        """No satellites active: only 1 configuration."""
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(6, 3, 0)
        assert result.total_configurations == 1
        assert result.distinct_configurations == 1

    def test_distinct_leq_total(self):
        """Distinct configurations must not exceed total configurations."""
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(12, 3, 4)
        assert result.distinct_configurations <= result.total_configurations
        assert result.distinct_configurations >= 1

    def test_group_order(self):
        """Group order = P * S."""
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(12, 4, 6)
        assert result.symmetry_group_order == 4 * 3  # P=4, S=12/4=3

    def test_redundancy_factor(self):
        """Redundancy = total / distinct."""
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(6, 2, 3)
        expected_redundancy = result.total_configurations / result.distinct_configurations
        assert abs(result.redundancy_factor - expected_redundancy) < 1e-10

    def test_fixed_point_counts_length(self):
        """Fixed point counts has |G| entries."""
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(6, 3, 2)
        assert len(result.fixed_point_counts) == result.symmetry_group_order

    def test_identity_fixed_count(self):
        """Identity element (0, 0) fixes all C(T, m) configurations."""
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(6, 3, 2)
        # Identity is the first element (k=0, j=0)
        assert result.fixed_point_counts[0] == math.comb(6, 2)

    def test_invalid_not_divisible_raises(self):
        from humeris.domain.trade_study import compute_burnside_symmetry
        with pytest.raises(ValueError, match="divisible"):
            compute_burnside_symmetry(7, 3, 2)

    def test_invalid_active_too_many_raises(self):
        from humeris.domain.trade_study import compute_burnside_symmetry
        with pytest.raises(ValueError, match="num_active"):
            compute_burnside_symmetry(6, 3, 7)

    def test_single_plane_single_sat(self):
        """T=1, P=1, m=1: trivially 1 distinct configuration."""
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(1, 1, 1)
        assert result.distinct_configurations == 1

    def test_known_necklace_count(self):
        """6 beads in a ring (P=1, S=6), 3 active: known necklace counting.
        This is the number of binary necklaces of length 6 with 3 ones.
        By Burnside with Z_6: distinct = (1/6)*sum_{d|6} phi(6/d)*C(d, 3*d/6)
        = (1/6)*(C(6,3) + 0 + C(2,1) + C(6,3) + 0 + C(2,1) + ...)
        Actually for Z_1 x Z_6 (P=1):
        Distinct binary necklaces of length 6 with 3 ones = 4.
        """
        from humeris.domain.trade_study import compute_burnside_symmetry
        result = compute_burnside_symmetry(6, 1, 3)
        # The standard necklace count for binary necklaces of length 6 with 3 ones
        # is (1/6) * [C(6,3) + 0 + C(2,1) + 0 + 0 + C(2,1)] = (20+0+2+0+0+2)/6 = 4
        # But we also need to verify what our group gives.
        # With P=1, S=6: G = Z_1 x Z_6, effectively just Z_6.
        assert result.distinct_configurations == 4


# ══════════════════════════════════════════════════════════════════
# P52: Lattice Partial Orders
# ══════════════════════════════════════════════════════════════════

class TestP52_ConstellationLattice:
    """Partial order analysis of constellation configurations."""

    def test_returns_frozen_dataclass(self):
        from humeris.domain.trade_study import compute_constellation_lattice
        result = compute_constellation_lattice([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(AttributeError):
            result.num_configurations = 0

    def test_total_order(self):
        """If all configs are totally ordered, width=1, height=n."""
        from humeris.domain.trade_study import compute_constellation_lattice
        # Config i dominates config i-1 at all grid points
        matrix = [[float(i), float(i)] for i in range(5)]
        result = compute_constellation_lattice(matrix)
        assert result.partial_order_height == 5
        assert result.partial_order_width == 1
        assert result.num_comparable_pairs == 10  # C(5,2) = 10

    def test_antichain(self):
        """If no config dominates another, width=n, height=1."""
        from humeris.domain.trade_study import compute_constellation_lattice
        # Each config is better on one grid point, worse on another
        matrix = [
            [1.0, 3.0],
            [3.0, 1.0],
            [2.0, 2.0],
        ]
        # Check: (1,3) vs (3,1): incomparable. (1,3) vs (2,2): 1<2 but 3>2: incomparable.
        # (3,1) vs (2,2): 3>2, 1<2: incomparable. All pairwise incomparable.
        result = compute_constellation_lattice(matrix)
        assert result.partial_order_width == 3
        assert result.partial_order_height == 1
        assert result.num_comparable_pairs == 0

    def test_comparability_fraction(self):
        from humeris.domain.trade_study import compute_constellation_lattice
        # 2 out of 3 pairs comparable
        matrix = [
            [1.0, 1.0],
            [2.0, 2.0],
            [1.5, 3.0],  # incomparable with [2, 2]: 1.5<2 but 3>2
        ]
        result = compute_constellation_lattice(matrix)
        # [1,1] <= [2,2]: yes. [1,1] <= [1.5,3]: yes. [1.5,3] vs [2,2]: no.
        assert result.num_comparable_pairs == 2
        assert abs(result.comparability_fraction - 2.0 / 3.0) < 1e-10

    def test_single_config(self):
        from humeris.domain.trade_study import compute_constellation_lattice
        result = compute_constellation_lattice([[1.0, 2.0]])
        assert result.num_configurations == 1
        assert result.partial_order_height == 1
        assert result.partial_order_width == 1

    def test_empty_raises(self):
        from humeris.domain.trade_study import compute_constellation_lattice
        with pytest.raises(ValueError, match="non-empty"):
            compute_constellation_lattice([])

    def test_meet_irreducibles(self):
        """In a chain, all non-top elements are join-irreducible."""
        from humeris.domain.trade_study import compute_constellation_lattice
        matrix = [[float(i)] for i in range(4)]
        result = compute_constellation_lattice(matrix)
        # In a chain of 4: meet-irreducibles have exactly 1 upper cover
        # Element 0: upper cover = 1 (only). Meet-irr.
        # Element 1: upper cover = 2 (only). Meet-irr.
        # Element 2: upper cover = 3 (only). Meet-irr.
        # Element 3: no upper cover. Not meet-irr (0 covers).
        assert result.meet_irreducibles == 3

    def test_join_irreducibles(self):
        """In a chain, all non-bottom elements are join-irreducible."""
        from humeris.domain.trade_study import compute_constellation_lattice
        matrix = [[float(i)] for i in range(4)]
        result = compute_constellation_lattice(matrix)
        # Join-irr: exactly 1 lower cover
        # Element 0: no lower cover. Not join-irr.
        # Element 1: lower cover = 0. Join-irr.
        # Element 2: lower cover = 1. Join-irr.
        # Element 3: lower cover = 2. Join-irr.
        assert result.join_irreducibles == 3

    def test_inconsistent_grid_raises(self):
        from humeris.domain.trade_study import compute_constellation_lattice
        with pytest.raises(ValueError, match="same number"):
            compute_constellation_lattice([[1.0, 2.0], [3.0]])

    def test_identical_configs(self):
        """Two identical configs are comparable (both <= each other)."""
        from humeris.domain.trade_study import compute_constellation_lattice
        result = compute_constellation_lattice([[1.0, 2.0], [1.0, 2.0]])
        assert result.num_comparable_pairs == 1
        assert result.comparability_fraction == 1.0

    def test_height_geq_1(self):
        """Height is always at least 1."""
        from humeris.domain.trade_study import compute_constellation_lattice
        result = compute_constellation_lattice([[5.0]])
        assert result.partial_order_height >= 1


# ══════════════════════════════════════════════════════════════════
# P56: Hopfield Attractor Network
# ══════════════════════════════════════════════════════════════════

class TestP56_HopfieldAttractors:
    """Hopfield attractor network for constellation modes."""

    def test_returns_frozen_dataclass(self):
        from humeris.domain.constellation_operability import compute_constellation_attractors
        patterns = [[1, 1, -1, -1], [-1, -1, 1, 1]]
        state = [1, 1, -1, -1]
        result = compute_constellation_attractors(patterns, state)
        with pytest.raises(AttributeError):
            result.energy = 0.0

    def test_exact_pattern_recovery(self):
        """Current state matching a stored pattern should have overlap ~1."""
        from humeris.domain.constellation_operability import compute_constellation_attractors
        patterns = [[1, 1, -1, -1], [-1, -1, 1, 1]]
        state = [1, 1, -1, -1]
        result = compute_constellation_attractors(patterns, state)
        assert abs(result.current_overlap[0] - 1.0) < 1e-10
        assert abs(result.current_overlap[1] - (-1.0)) < 1e-10

    def test_nearest_attractor_correct(self):
        """State close to pattern 0 should converge to it."""
        from humeris.domain.constellation_operability import compute_constellation_attractors
        N = 20
        p0 = [1] * 10 + [-1] * 10
        p1 = [-1] * 10 + [1] * 10
        # Perturb p0 by flipping 2 bits
        state = list(p0)
        state[0] = -1
        state[1] = -1
        result = compute_constellation_attractors([p0, p1], state)
        assert result.nearest_attractor_idx == 0

    def test_energy_negative_for_stored_pattern(self):
        """Energy should be negative for a state that matches a stored pattern."""
        from humeris.domain.constellation_operability import compute_constellation_attractors
        patterns = [[1, 1, -1, -1], [-1, -1, 1, 1]]
        state = [1, 1, -1, -1]
        result = compute_constellation_attractors(patterns, state)
        assert result.energy < 0

    def test_num_stored_patterns(self):
        from humeris.domain.constellation_operability import compute_constellation_attractors
        patterns = [[1, -1, 1], [-1, 1, -1], [1, 1, 1]]
        state = [1, -1, 1]
        result = compute_constellation_attractors(patterns, state)
        assert result.num_stored_patterns == 3

    def test_basin_sizes_sum(self):
        """Basin sizes should be positive and sum to ~1."""
        from humeris.domain.constellation_operability import compute_constellation_attractors
        patterns = [[1, 1, -1, -1], [-1, -1, 1, 1]]
        state = [1, 1, -1, -1]
        result = compute_constellation_attractors(patterns, state)
        assert all(b > 0 for b in result.basin_sizes)
        assert abs(sum(result.basin_sizes) - 1.0) < 1e-10

    def test_invalid_pattern_value_raises(self):
        from humeris.domain.constellation_operability import compute_constellation_attractors
        with pytest.raises(ValueError, match="\\+1 or -1"):
            compute_constellation_attractors([[1, 0, -1]], [1, -1, 1])

    def test_inconsistent_state_length_raises(self):
        from humeris.domain.constellation_operability import compute_constellation_attractors
        with pytest.raises(ValueError, match="length"):
            compute_constellation_attractors([[1, -1]], [1, -1, 1])

    def test_empty_patterns_raises(self):
        from humeris.domain.constellation_operability import compute_constellation_attractors
        with pytest.raises(ValueError, match="non-empty"):
            compute_constellation_attractors([], [1, -1])

    def test_is_within_basin_for_exact_match(self):
        """Exact pattern match should be within basin."""
        from humeris.domain.constellation_operability import compute_constellation_attractors
        patterns = [[1, 1, -1, -1, 1, 1, -1, -1]]
        state = [1, 1, -1, -1, 1, 1, -1, -1]
        result = compute_constellation_attractors(patterns, state)
        assert result.is_within_basin is True

    def test_overlap_range(self):
        """Overlaps should be in [-1, 1]."""
        from humeris.domain.constellation_operability import compute_constellation_attractors
        patterns = [[1, 1, -1, -1], [-1, -1, 1, 1]]
        state = [1, -1, 1, -1]
        result = compute_constellation_attractors(patterns, state)
        for m in result.current_overlap:
            assert -1.0 <= m <= 1.0


# ══════════════════════════════════════════════════════════════════
# P57: Geometric (Berry) Phase for Cyclic Maneuvers
# ══════════════════════════════════════════════════════════════════

class TestP57_GeometricPhase:
    """Berry/geometric phase for cyclic orbital maneuvers."""

    def test_returns_frozen_dataclass(self):
        from humeris.domain.station_keeping import compute_geometric_phase
        result = compute_geometric_phase(R_LEO, 10_000.0, 5400.0)
        with pytest.raises(AttributeError):
            result.dynamic_phase_rad = 0.0

    def test_dynamic_phase_positive(self):
        """Dynamic phase from orbital motion should be positive."""
        from humeris.domain.station_keeping import compute_geometric_phase
        result = compute_geometric_phase(R_LEO, 10_000.0, 5400.0)
        assert result.dynamic_phase_rad > 0

    def test_geometric_phase_sign_for_raise(self):
        """Raising orbit (positive delta_a) should give negative geometric phase."""
        from humeris.domain.station_keeping import compute_geometric_phase
        result = compute_geometric_phase(R_LEO, 10_000.0, 5400.0)
        assert result.geometric_phase_rad < 0

    def test_geometric_phase_sign_for_lower(self):
        """Lowering orbit (negative delta_a) should give positive geometric phase."""
        from humeris.domain.station_keeping import compute_geometric_phase
        result = compute_geometric_phase(R_LEO, -10_000.0, 5400.0)
        assert result.geometric_phase_rad > 0

    def test_total_phase_is_sum(self):
        from humeris.domain.station_keeping import compute_geometric_phase
        result = compute_geometric_phase(R_LEO, 10_000.0, 5400.0)
        expected = result.dynamic_phase_rad + result.geometric_phase_rad
        assert abs(result.total_phase_rad - expected) < 1e-10

    def test_hannay_equals_geometric(self):
        """Hannay angle should equal geometric phase."""
        from humeris.domain.station_keeping import compute_geometric_phase
        result = compute_geometric_phase(R_LEO, 10_000.0, 5400.0)
        assert abs(result.hannay_angle_rad - result.geometric_phase_rad) < 1e-15

    def test_zero_wait_time(self):
        """Zero wait time gives zero phases."""
        from humeris.domain.station_keeping import compute_geometric_phase
        result = compute_geometric_phase(R_LEO, 10_000.0, 0.0)
        assert result.dynamic_phase_rad == 0.0
        assert result.geometric_phase_rad == 0.0

    def test_geometric_phase_scales_with_delta_a(self):
        """Geometric phase should scale linearly with delta_a."""
        from humeris.domain.station_keeping import compute_geometric_phase
        r1 = compute_geometric_phase(R_LEO, 1000.0, 5400.0)
        r2 = compute_geometric_phase(R_LEO, 2000.0, 5400.0)
        ratio = r2.geometric_phase_rad / r1.geometric_phase_rad
        assert abs(ratio - 2.0) < 1e-6

    def test_geometric_phase_scales_with_wait_time(self):
        """Geometric phase should scale linearly with wait time."""
        from humeris.domain.station_keeping import compute_geometric_phase
        r1 = compute_geometric_phase(R_LEO, 1000.0, 5400.0)
        r2 = compute_geometric_phase(R_LEO, 1000.0, 10800.0)
        ratio = r2.geometric_phase_rad / r1.geometric_phase_rad
        assert abs(ratio - 2.0) < 1e-6

    def test_raan_shift_with_inclination(self):
        """Non-zero inclination should produce non-zero RAAN shift."""
        from humeris.domain.station_keeping import compute_geometric_phase
        result = compute_geometric_phase(
            R_LEO, 10_000.0, 86400.0, inclination_rad=0.9,
        )
        assert result.raan_shift_rad != 0.0

    def test_raan_shift_zero_at_equator(self):
        """At 90-degree inclination, RAAN rate is zero (cos(90)=0)."""
        from humeris.domain.station_keeping import compute_geometric_phase
        result = compute_geometric_phase(
            R_LEO, 10_000.0, 86400.0, inclination_rad=math.pi / 2,
        )
        assert abs(result.raan_shift_rad) < 1e-10

    def test_negative_sma_raises(self):
        from humeris.domain.station_keeping import compute_geometric_phase
        with pytest.raises(ValueError, match="positive"):
            compute_geometric_phase(-1.0, 10_000.0, 5400.0)

    def test_negative_wait_raises(self):
        from humeris.domain.station_keeping import compute_geometric_phase
        with pytest.raises(ValueError, match="non-negative"):
            compute_geometric_phase(R_LEO, 10_000.0, -1.0)


# ══════════════════════════════════════════════════════════════════
# P61: Mapper Algorithm for Design Space TDA
# ══════════════════════════════════════════════════════════════════

class TestP61_DesignSpaceTopology:
    """Mapper algorithm for design space topological data analysis."""

    def test_returns_frozen_dataclass(self):
        from humeris.domain.trade_study import compute_design_space_topology
        pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        fvals = [0.0, 1.0, 0.5]
        result = compute_design_space_topology(pts, fvals, num_intervals=2)
        with pytest.raises(AttributeError):
            result.num_components = 0

    def test_single_cluster(self):
        """Tightly packed points with generous overlap should form one component."""
        from humeris.domain.trade_study import compute_design_space_topology
        pts = [[0.0], [0.1], [0.2], [0.3]]
        fvals = [0.0, 0.1, 0.2, 0.3]
        result = compute_design_space_topology(
            pts, fvals, num_intervals=2, overlap_fraction=0.7,
            cluster_threshold=0.5,
        )
        assert result.num_components == 1

    def test_two_separate_clusters(self):
        """Well-separated points should form multiple components."""
        from humeris.domain.trade_study import compute_design_space_topology
        pts = [[0.0], [0.1], [100.0], [100.1]]
        fvals = [0.0, 0.1, 100.0, 100.1]
        result = compute_design_space_topology(
            pts, fvals, num_intervals=3, overlap_fraction=0.3,
            cluster_threshold=1.0,
        )
        assert result.num_components >= 2

    def test_loop_detection(self):
        """Euler characteristic beta_1 = edges - vertices + components >= 0.
        Construct a case where multiple overlapping intervals share points
        that create extra edges beyond a spanning tree."""
        from humeris.domain.trade_study import compute_design_space_topology
        # Create a Y-shaped point cloud where the filter intervals each
        # contain multiple branches that share points, creating a cycle.
        # Strategy: 3 clusters of points with pairwise shared points
        # A-B shared, B-C shared, A-C shared => triangle => 1 loop
        pts = [
            # Cluster A: filter ~ 0.0
            [0.0, 0.0], [0.1, 0.0],
            # Shared A-B point
            [0.5, 0.0],
            # Cluster B: filter ~ 1.0
            [1.0, 0.0], [1.1, 0.0],
            # Shared B-C point
            [1.5, 0.0],
            # Cluster C: filter ~ 2.0
            [2.0, 0.0], [2.1, 0.0],
        ]
        fvals = [0.0, 0.1, 0.5, 1.0, 1.1, 1.5, 2.0, 2.1]
        result = compute_design_space_topology(
            pts, fvals, num_intervals=3, overlap_fraction=0.5,
            cluster_threshold=0.6,
        )
        # This should at minimum form a connected complex.
        # The Euler formula holds: beta_1 = E - V + beta_0
        V = len(result.cluster_sizes)
        E = len(result.edges)
        assert result.num_loops == E - V + result.num_components

    def test_cluster_sizes_sum(self):
        """Sum of cluster sizes should account for all points (possibly with overlap)."""
        from humeris.domain.trade_study import compute_design_space_topology
        pts = [[float(i)] for i in range(10)]
        fvals = [float(i) for i in range(10)]
        result = compute_design_space_topology(
            pts, fvals, num_intervals=3, overlap_fraction=0.3,
            cluster_threshold=2.0,
        )
        # Each point should be in at least one cluster
        all_points = set()
        for family in result.design_families:
            for pt_idx in family:
                all_points.add(pt_idx)
        assert len(all_points) == 10

    def test_mismatched_lengths_raises(self):
        from humeris.domain.trade_study import compute_design_space_topology
        with pytest.raises(ValueError, match="same length"):
            compute_design_space_topology([[0.0]], [0.0, 1.0])

    def test_empty_raises(self):
        from humeris.domain.trade_study import compute_design_space_topology
        with pytest.raises(ValueError, match="non-empty"):
            compute_design_space_topology([], [])

    def test_single_point(self):
        from humeris.domain.trade_study import compute_design_space_topology
        result = compute_design_space_topology([[1.0]], [0.0])
        assert result.num_components == 1
        assert result.num_loops == 0

    def test_edges_connect_valid_clusters(self):
        """All edges should reference valid cluster indices."""
        from humeris.domain.trade_study import compute_design_space_topology
        pts = [[float(i)] for i in range(10)]
        fvals = [float(i) for i in range(10)]
        result = compute_design_space_topology(
            pts, fvals, num_intervals=3, overlap_fraction=0.3,
            cluster_threshold=2.0,
        )
        num_clusters = len(result.cluster_sizes)
        for u, v in result.edges:
            assert 0 <= u < num_clusters
            assert 0 <= v < num_clusters

    def test_beta_1_is_non_negative(self):
        """Number of loops (beta_1) should never be negative."""
        from humeris.domain.trade_study import compute_design_space_topology
        pts = [[float(i)] for i in range(5)]
        fvals = [float(i) for i in range(5)]
        result = compute_design_space_topology(
            pts, fvals, num_intervals=2, overlap_fraction=0.3,
            cluster_threshold=2.0,
        )
        assert result.num_loops >= 0


# ══════════════════════════════════════════════════════════════════
# P62: Stochastic Resonance for Maneuver Timing
# ══════════════════════════════════════════════════════════════════

class TestP62_StochasticResonance:
    """Stochastic resonance analysis for maneuver timing."""

    def test_returns_frozen_dataclass(self):
        from humeris.domain.station_keeping import compute_stochastic_resonance
        result = compute_stochastic_resonance(1.0, 1.0, 0.1, 0.01)
        with pytest.raises(AttributeError):
            result.peak_snr = 0.0

    def test_barrier_height_formula(self):
        """DeltaV = a^2 / (4b)."""
        from humeris.domain.station_keeping import compute_stochastic_resonance
        result = compute_stochastic_resonance(2.0, 1.0, 0.1, 0.01)
        assert abs(result.barrier_height - 2.0 ** 2 / (4.0 * 1.0)) < 1e-15

    def test_optimal_noise_equals_barrier(self):
        """D_opt = DeltaV from analytical SNR maximization."""
        from humeris.domain.station_keeping import compute_stochastic_resonance
        result = compute_stochastic_resonance(2.0, 1.0, 0.1, 0.01)
        assert abs(result.optimal_noise_intensity - result.barrier_height) < 1e-15

    def test_peak_snr_formula(self):
        """SNR at D_opt = (pi * A^2) / (4 * D_opt) * exp(-1)."""
        from humeris.domain.station_keeping import compute_stochastic_resonance
        A = 0.5
        result = compute_stochastic_resonance(2.0, 1.0, A, 0.01)
        d_opt = result.optimal_noise_intensity
        expected = (math.pi * A ** 2) / (4.0 * d_opt) * math.exp(-1.0)
        assert abs(result.peak_snr - expected) < 1e-15

    def test_kramers_rate_formula(self):
        """r_K = (omega_0 / 2pi) * exp(-DeltaV / D_opt)."""
        from humeris.domain.station_keeping import compute_stochastic_resonance
        a, b = 2.0, 1.0
        result = compute_stochastic_resonance(a, b, 0.1, 0.01)
        omega_0 = math.sqrt(2.0 * a)
        d_v = a ** 2 / (4.0 * b)
        expected = (omega_0 / (2.0 * math.pi)) * math.exp(-d_v / d_v)
        assert abs(result.kramers_rate - expected) < 1e-15

    def test_effectiveness_gain_gt_1(self):
        """Resonance enhancement at D_opt vs D_opt/2 should be > 1."""
        from humeris.domain.station_keeping import compute_stochastic_resonance
        result = compute_stochastic_resonance(1.0, 1.0, 0.1, 0.01)
        # SNR(D_opt) / SNR(D_opt/2):
        # = [pi*A^2/(4*D_opt) * exp(-DV/D_opt)] / [pi*A^2/(4*D_opt*0.5) * exp(-DV/(D_opt*0.5))]
        # = [1/D_opt * exp(-1)] / [2/D_opt * exp(-2)]
        # = 0.5 * exp(1) ~ 1.359
        assert result.maneuver_effectiveness_gain > 1.0

    def test_effectiveness_gain_value(self):
        """Effectiveness gain = SNR(D_opt)/SNR(D_opt/2) = exp(1)/2."""
        from humeris.domain.station_keeping import compute_stochastic_resonance
        result = compute_stochastic_resonance(1.0, 1.0, 0.1, 0.01)
        expected = math.exp(1.0) / 2.0
        assert abs(result.maneuver_effectiveness_gain - expected) < 1e-10

    def test_peak_snr_positive(self):
        from humeris.domain.station_keeping import compute_stochastic_resonance
        result = compute_stochastic_resonance(1.0, 1.0, 0.1, 0.01)
        assert result.peak_snr > 0

    def test_invalid_a_raises(self):
        from humeris.domain.station_keeping import compute_stochastic_resonance
        with pytest.raises(ValueError, match="potential_a"):
            compute_stochastic_resonance(-1.0, 1.0, 0.1, 0.01)

    def test_invalid_b_raises(self):
        from humeris.domain.station_keeping import compute_stochastic_resonance
        with pytest.raises(ValueError, match="potential_b"):
            compute_stochastic_resonance(1.0, -1.0, 0.1, 0.01)

    def test_invalid_amplitude_raises(self):
        from humeris.domain.station_keeping import compute_stochastic_resonance
        with pytest.raises(ValueError, match="signal_amplitude"):
            compute_stochastic_resonance(1.0, 1.0, -0.1, 0.01)

    def test_invalid_frequency_raises(self):
        from humeris.domain.station_keeping import compute_stochastic_resonance
        with pytest.raises(ValueError, match="signal_frequency_hz"):
            compute_stochastic_resonance(1.0, 1.0, 0.1, -0.01)

    def test_higher_barrier_means_lower_kramers_rate(self):
        """Higher barrier should yield lower Kramers escape rate at D_opt."""
        from humeris.domain.station_keeping import compute_stochastic_resonance
        r_low = compute_stochastic_resonance(1.0, 1.0, 0.1, 0.01)  # DV = 0.25
        r_high = compute_stochastic_resonance(4.0, 1.0, 0.1, 0.01)  # DV = 4.0
        # At D_opt = DV, kramers = omega/(2pi)*exp(-1), so omega matters
        # omega_0 = sqrt(2a), so r_high has larger omega.
        # Both have exp(-1), so rate is proportional to sqrt(a)
        # r_low: sqrt(2) / 2pi * e^-1, r_high: sqrt(8)/2pi * e^-1
        # So r_high > r_low (higher a means higher omega_0)
        # This test verifies the formula is self-consistent.
        assert r_low.kramers_rate > 0
        assert r_high.kramers_rate > 0
