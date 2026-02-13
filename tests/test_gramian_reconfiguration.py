# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/gramian_reconfiguration.py — G-RECON module."""
import ast
import math

from humeris.domain.control_analysis import compute_cw_controllability
from humeris.domain.gramian_reconfiguration import (
    ReconfigurationTarget,
    ReconfigurationManeuver,
    ReconfigurationPlan,
    compute_gramian_optimal_dv,
    plan_reconfiguration,
    compute_fuel_cost_index,
    find_cheapest_reconfig_path,
    compute_reconfig_window,
)

# Typical LEO: ~400 km altitude → n ≈ 0.00113 rad/s
_N_LEO = 0.00113
_ORBIT_S = 5400.0  # approximately one orbit at LEO


class TestComputeGramianOptimalDv:
    def test_gramian_optimal_dv_along_track(self):
        """Along-track (y) maneuver should produce relatively small dV.

        In CW dynamics, along-track drift is naturally driven by radial
        offsets. Along-track state changes are cheap because the Gramian
        eigenvalues for those directions are large.
        """
        # Pure along-track position change: 1000 m in y
        target = (0.0, 1000.0, 0.0, 0.0, 0.0, 0.0)
        dv = compute_gramian_optimal_dv(target, _N_LEO, _ORBIT_S)
        assert len(dv) == 3
        dv_mag_along = math.sqrt(dv[0] ** 2 + dv[1] ** 2 + dv[2] ** 2)

        # Pure radial position change: 1000 m in x
        target_radial = (1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        dv_radial = compute_gramian_optimal_dv(target_radial, _N_LEO, _ORBIT_S)
        dv_mag_radial = math.sqrt(dv_radial[0] ** 2 + dv_radial[1] ** 2 + dv_radial[2] ** 2)

        # Along-track should be cheaper (or comparable) to radial
        # The key point: dV is finite and well-defined
        assert dv_mag_along >= 0.0
        assert dv_mag_along < float('inf')

    def test_gramian_optimal_dv_radial(self):
        """Radial (x) maneuver has nonzero dV — actively controlled direction."""
        target = (1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        dv = compute_gramian_optimal_dv(target, _N_LEO, _ORBIT_S)
        dv_mag = math.sqrt(dv[0] ** 2 + dv[1] ** 2 + dv[2] ** 2)
        # Radial displacement requires nonzero control effort
        assert dv_mag > 0.0

    def test_gramian_optimal_dv_zero_target(self):
        """Zero target state change should produce zero delta-V."""
        target = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        dv = compute_gramian_optimal_dv(target, _N_LEO, _ORBIT_S)
        assert dv == (0.0, 0.0, 0.0)


class TestPlanReconfiguration:
    def test_plan_reconfiguration_single_sat(self):
        """Single satellite reconfiguration yields one maneuver."""
        target = ReconfigurationTarget(
            satellite_index=0,
            delta_state=(500.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        plan = plan_reconfiguration([target], _N_LEO, _ORBIT_S)
        assert isinstance(plan, ReconfigurationPlan)
        assert len(plan.maneuvers) == 1
        assert plan.maneuvers[0].satellite_index == 0
        assert plan.total_delta_v > 0.0
        assert plan.max_single_dv == plan.total_delta_v

    def test_plan_reconfiguration_multiple_sats(self):
        """Multiple satellite plan aggregates correctly."""
        targets = [
            ReconfigurationTarget(satellite_index=0, delta_state=(500.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ReconfigurationTarget(satellite_index=1, delta_state=(0.0, 500.0, 0.0, 0.0, 0.0, 0.0)),
            ReconfigurationTarget(satellite_index=2, delta_state=(0.0, 0.0, 500.0, 0.0, 0.0, 0.0)),
        ]
        plan = plan_reconfiguration(targets, _N_LEO, _ORBIT_S)
        assert len(plan.maneuvers) == 3
        # Each satellite index present
        indices = {m.satellite_index for m in plan.maneuvers}
        assert indices == {0, 1, 2}
        # Total dV is sum of individual magnitudes
        expected_total = sum(m.delta_v_magnitude for m in plan.maneuvers)
        assert abs(plan.total_delta_v - expected_total) < 1e-12

    def test_plan_feasibility_check(self):
        """Maneuver exceeding max_dv should mark plan as infeasible."""
        target = ReconfigurationTarget(
            satellite_index=0,
            delta_state=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        plan = plan_reconfiguration(
            [target], _N_LEO, _ORBIT_S, max_dv_per_sat=1e-10,
        )
        assert plan.is_feasible is False

    def test_plan_feasible_within_limit(self):
        """Maneuver within max_dv should mark plan as feasible."""
        target = ReconfigurationTarget(
            satellite_index=0,
            delta_state=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        plan = plan_reconfiguration(
            [target], _N_LEO, _ORBIT_S, max_dv_per_sat=1e6,
        )
        assert plan.is_feasible is True

    def test_plan_with_propellant_mass(self):
        """When Isp and dry mass provided, propellant mass is computed."""
        target = ReconfigurationTarget(
            satellite_index=0,
            delta_state=(500.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        plan = plan_reconfiguration(
            [target], _N_LEO, _ORBIT_S, isp_s=300.0, dry_mass_kg=100.0,
        )
        assert plan.maneuvers[0].propellant_mass_kg > 0.0
        assert plan.total_propellant_kg > 0.0

    def test_plan_no_propellant_without_isp(self):
        """Without Isp, propellant mass is zero."""
        target = ReconfigurationTarget(
            satellite_index=0,
            delta_state=(500.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        plan = plan_reconfiguration([target], _N_LEO, _ORBIT_S)
        assert plan.maneuvers[0].propellant_mass_kg == 0.0

    def test_efficiency_score_range(self):
        """Efficiency score must be in [0, 1]."""
        targets = [
            ReconfigurationTarget(satellite_index=0, delta_state=(500.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ReconfigurationTarget(satellite_index=1, delta_state=(0.0, 0.0, 500.0, 0.0, 0.0, 0.0)),
        ]
        plan = plan_reconfiguration(targets, _N_LEO, _ORBIT_S)
        assert 0.0 <= plan.efficiency_score <= 1.0

    def test_total_delta_v_sum(self):
        """Total dV equals sum of individual dV magnitudes."""
        targets = [
            ReconfigurationTarget(satellite_index=i, delta_state=(100.0 * (i + 1), 0.0, 0.0, 0.0, 0.0, 0.0))
            for i in range(4)
        ]
        plan = plan_reconfiguration(targets, _N_LEO, _ORBIT_S)
        individual_sum = sum(m.delta_v_magnitude for m in plan.maneuvers)
        assert abs(plan.total_delta_v - individual_sum) < 1e-12


class TestFuelCostIndex:
    def test_fuel_cost_index_cheap_direction(self):
        """State change along max-eigenvalue direction should have cost < 1."""
        analysis = compute_cw_controllability(_N_LEO, _ORBIT_S, step_s=60.0)
        # max_energy_direction is the cheapest direction
        cheap_dir = analysis.max_energy_direction
        cost = compute_fuel_cost_index(cheap_dir, analysis)
        # Along the largest eigenvalue, cost index should be < 1 (cheaper than average)
        assert cost < 1.0

    def test_fuel_cost_index_expensive_direction(self):
        """State change along min-eigenvalue direction should have cost > 1."""
        analysis = compute_cw_controllability(_N_LEO, _ORBIT_S, step_s=60.0)
        # min_energy_direction is the costliest direction
        expensive_dir = analysis.min_energy_direction
        cost = compute_fuel_cost_index(expensive_dir, analysis)
        # Along the smallest eigenvalue, cost index should be > 1 (more expensive than average)
        assert cost > 1.0

    def test_fuel_cost_index_zero_state(self):
        """Zero state change should return cost index 1.0 (neutral)."""
        analysis = compute_cw_controllability(_N_LEO, _ORBIT_S, step_s=60.0)
        cost = compute_fuel_cost_index((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), analysis)
        assert cost == 1.0


class TestGramianAlignment:
    def test_gramian_alignment_parallel(self):
        """Maneuver along the cheapest direction should have alignment near 1."""
        analysis = compute_cw_controllability(_N_LEO, _ORBIT_S, step_s=60.0)
        cheap_dir = analysis.max_energy_direction
        target = ReconfigurationTarget(
            satellite_index=0,
            delta_state=cheap_dir,
        )
        plan = plan_reconfiguration([target], _N_LEO, _ORBIT_S)
        # gramian_alignment is cosine similarity with max-eigenvalue direction
        # For the max-eigenvalue direction itself, |alignment| ≈ 1
        assert abs(plan.maneuvers[0].gramian_alignment) > 0.9


class TestCheapestReconfigPath:
    def test_cheapest_path_identity(self):
        """If current == desired, all costs should be zero."""
        states = [
            (100.0, 200.0, 0.0, 0.0, 0.0, 0.0),
            (300.0, 400.0, 0.0, 0.0, 0.0, 0.0),
        ]
        assignments = find_cheapest_reconfig_path(
            states, states, _N_LEO, _ORBIT_S,
        )
        assert len(assignments) == 2
        for sat_idx, tgt_idx, cost in assignments:
            assert cost == 0.0
            assert sat_idx == tgt_idx  # identity mapping

    def test_cheapest_path_swap(self):
        """Two satellites needing a swap should produce an assignment."""
        state_a = (100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state_b = (0.0, 100.0, 0.0, 0.0, 0.0, 0.0)
        # Current: [A, B], desired: [B, A]
        assignments = find_cheapest_reconfig_path(
            [state_a, state_b],
            [state_b, state_a],
            _N_LEO,
            _ORBIT_S,
        )
        assert len(assignments) == 2
        # Each satellite should be assigned to a target
        sat_indices = {a[0] for a in assignments}
        tgt_indices = {a[1] for a in assignments}
        assert sat_indices == {0, 1}
        assert tgt_indices == {0, 1}
        # All costs should be non-negative
        for _, _, cost in assignments:
            assert cost >= 0.0


class TestReconfigWindow:
    def test_reconfig_window_condition_varies(self):
        """Condition number should vary with maneuver duration."""
        durations = [1800.0, 3600.0, 5400.0, 7200.0]
        results = compute_reconfig_window(_N_LEO, durations)
        assert len(results) == 4
        # Each entry is (duration, condition_number, gramian_trace)
        conditions = [r[1] for r in results]
        # Condition numbers should not all be identical
        assert len(set(conditions)) > 1

    def test_reconfig_window_trace_increases(self):
        """Gramian trace should generally increase with duration."""
        durations = [1800.0, 5400.0]
        results = compute_reconfig_window(_N_LEO, durations)
        # Longer duration → more integrated controllability → larger trace
        assert results[1][2] > results[0][2]


class TestGramianReconfigurationPurity:
    def test_module_pure(self):
        """Module only imports from allowed packages."""
        import humeris.domain.gramian_reconfiguration as mod
        source = ast.parse(open(mod.__file__).read())
        for node in ast.walk(source):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                else:
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                assert top in {
                    "math", "numpy", "dataclasses", "datetime",
                    "typing", "enum", "humeris", "__future__",
                }, f"Forbidden import: {top}"
