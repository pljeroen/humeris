# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/competing_risks.py — competing-risks satellite population dynamics."""
import math

import numpy as np

from humeris.domain.competing_risks import (
    RiskProfile,
    CompetingRisksResult,
    PopulationProjection,
    create_drag_risk,
    create_collision_risk,
    create_component_risk,
    create_deorbit_risk,
    compute_competing_risks,
    project_population,
    compute_risk_sensitivity,
)

# Realistic LEO parameters
_ALT_KM = 500.0
_DRAG_DECAY_KM_PER_YEAR = 2.0
_SPATIAL_DENSITY = 1e-8  # objects/km^3
_REL_VELOCITY = 10_000.0  # m/s
_CROSS_SECTION = 10.0  # m^2
_MTBF_YEARS = 15.0
_PLANNED_LIFE_YEARS = 5.0
_DURATION = 25.0


def _standard_risks():
    """Create a standard set of four competing risks."""
    return [
        create_drag_risk(_ALT_KM, _DRAG_DECAY_KM_PER_YEAR),
        create_collision_risk(_SPATIAL_DENSITY, _REL_VELOCITY, _CROSS_SECTION),
        create_component_risk(_MTBF_YEARS),
        create_deorbit_risk(_PLANNED_LIFE_YEARS),
    ]


class TestSingleConstantRiskExponential:
    """One constant risk should produce exponential survival."""

    def test_single_constant_risk_exponential(self):
        risk = create_collision_risk(_SPATIAL_DENSITY, _REL_VELOCITY, _CROSS_SECTION)
        result = compute_competing_risks([risk], duration_years=10.0, dt_years=0.01)

        h_per_day = risk.hazard_rates[0]
        times = np.array(result.times_years)
        expected_survival = np.exp(-h_per_day * times * 365.25)
        actual_survival = np.array(result.overall_survival)

        # Trapezoidal integration should match exponential closely
        np.testing.assert_allclose(actual_survival, expected_survival, rtol=1e-3)


class TestTwoRisksFasterDecay:
    """Adding a second risk must shorten the median lifetime."""

    def test_two_risks_faster_decay(self):
        single = create_collision_risk(_SPATIAL_DENSITY, _REL_VELOCITY, _CROSS_SECTION)
        component = create_component_risk(_MTBF_YEARS)

        result_one = compute_competing_risks([single], duration_years=_DURATION)
        result_two = compute_competing_risks([single, component], duration_years=_DURATION)

        assert result_two.median_lifetime_years < result_one.median_lifetime_years


class TestDragRiskIncreasingHazard:
    """Drag hazard should increase as altitude drops."""

    def test_drag_risk_increasing_hazard(self):
        risk = create_drag_risk(_ALT_KM, _DRAG_DECAY_KM_PER_YEAR)
        rates = np.array(risk.hazard_rates)

        # First rate should be less than last rate (increasing hazard)
        assert rates[0] < rates[-1]
        # Monotonically non-decreasing
        assert np.all(np.diff(rates) >= 0)


class TestCollisionRiskConstant:
    """Collision risk should be time-invariant."""

    def test_collision_risk_constant(self):
        risk = create_collision_risk(_SPATIAL_DENSITY, _REL_VELOCITY, _CROSS_SECTION)
        assert risk.is_constant is True
        assert len(risk.hazard_rates) == 1
        assert risk.hazard_rates[0] > 0.0


class TestComponentRiskWeibull:
    """With wear_factor > 0, component hazard should increase over time."""

    def test_component_risk_weibull(self):
        risk = create_component_risk(_MTBF_YEARS, wear_factor=0.1)
        assert risk.is_constant is False
        rates = np.array(risk.hazard_rates)

        # Hazard should increase
        assert rates[-1] > rates[0]
        # Should be monotonically non-decreasing
        assert np.all(np.diff(rates) >= 0)


class TestDeorbitRiskStepFunction:
    """Deorbit risk should be zero before planned lifetime."""

    def test_deorbit_risk_step_function(self):
        risk = create_deorbit_risk(_PLANNED_LIFE_YEARS, compliance_probability=0.9)
        rates = np.array(risk.hazard_rates)
        n = len(rates)

        # Determine which indices correspond to t < planned_lifetime
        duration = max(_PLANNED_LIFE_YEARS * 2.0, 25.0)
        times = np.linspace(0.0, duration, n)
        pre_mission = times < _PLANNED_LIFE_YEARS

        # All rates before planned lifetime should be zero
        assert np.all(rates[pre_mission] == 0.0)

        # At least some rates after planned lifetime should be positive
        post_mission = times >= _PLANNED_LIFE_YEARS
        assert np.any(rates[post_mission] > 0.0)


class TestCompetingRisksCIFSum:
    """CIFs must sum to 1 - S(t) at every time step."""

    def test_competing_risks_cif_sum(self):
        risks = _standard_risks()
        result = compute_competing_risks(risks, duration_years=_DURATION, dt_years=0.05)

        survival = np.array(result.overall_survival)
        cif_sum = np.zeros(len(survival))
        for _name, cif_values in result.cause_specific_cif:
            cif_sum += np.array(cif_values)

        expected = 1.0 - survival
        np.testing.assert_allclose(cif_sum, expected, atol=1e-6)


class TestSurvivalMonotoneDecreasing:
    """S(t) must never increase."""

    def test_survival_monotone_decreasing(self):
        risks = _standard_risks()
        result = compute_competing_risks(risks, duration_years=_DURATION)

        survival = np.array(result.overall_survival)
        assert np.all(np.diff(survival) <= 1e-15)


class TestRiskAttributionSumsToOne:
    """Risk attribution fractions must sum to approximately 1.0."""

    def test_risk_attribution_sums_to_one(self):
        risks = _standard_risks()
        result = compute_competing_risks(risks, duration_years=_DURATION)

        total = sum(frac for _name, frac in result.risk_attribution)
        assert abs(total - 1.0) < 1e-6


class TestDominantRiskChanges:
    """At different times, different risks should dominate."""

    def test_dominant_risk_changes(self):
        # Drag starts low, deorbit kicks in after planned life,
        # so dominant risk should change
        risks = _standard_risks()
        result = compute_competing_risks(risks, duration_years=_DURATION, dt_years=0.1)

        dominant = result.dominant_risk_at_time
        unique_dominant = set(dominant)
        # With drag + collision + component + deorbit, we expect at least
        # 2 different dominant risks over 25 years
        assert len(unique_dominant) >= 2


class TestPopulationNoLaunchesDecays:
    """Without replenishment, population must decay to near zero."""

    def test_population_no_launches_decays(self):
        risks = _standard_risks()
        proj = project_population(
            risks, initial_population=100,
            launch_rate_per_year=0.0,
            duration_years=_DURATION, dt_years=0.1,
        )

        pop = np.array(proj.active_population)
        # Population should decrease
        assert pop[-1] < pop[0]
        # Should be significantly depleted over 25 years
        assert pop[-1] < 50.0


class TestPopulationWithReplenishment:
    """With launches, population should be higher than without."""

    def test_population_with_replenishment(self):
        risks = _standard_risks()
        proj_no = project_population(
            risks, initial_population=100,
            launch_rate_per_year=0.0,
            duration_years=_DURATION, dt_years=0.1,
        )
        proj_yes = project_population(
            risks, initial_population=100,
            launch_rate_per_year=20.0,
            duration_years=_DURATION, dt_years=0.1,
        )

        pop_no = np.array(proj_no.active_population)
        pop_yes = np.array(proj_yes.active_population)

        # With launches, final population should be higher
        assert pop_yes[-1] > pop_no[-1]


class TestPopulationTargetMaintenance:
    """Target population should be approximately maintained via launches."""

    def test_population_target_maintenance(self):
        risks = _standard_risks()
        target = 100
        proj = project_population(
            risks, initial_population=target,
            target_population=target,
            duration_years=_DURATION, dt_years=0.1,
        )

        pop = np.array(proj.active_population)
        # Population should stay near target (within 5%)
        # Skip first few steps for transient
        mid_to_end = pop[10:]
        assert np.all(mid_to_end >= target * 0.95)
        assert np.all(mid_to_end <= target * 1.05)

        # Cumulative launches should be positive (replacements needed)
        assert proj.cumulative_launches[-1] > 0.0


class TestSensitivityHigherRiskShorterLife:
    """Multiplying a risk by > 1 should shorten median lifetime."""

    def test_sensitivity_higher_risk_shorter_life(self):
        risks = _standard_risks()
        multipliers = (0.5, 1.0, 2.0, 5.0)
        results = compute_risk_sensitivity(
            risks, risk_index=1,  # collision risk
            multipliers=multipliers,
            duration_years=_DURATION,
        )

        lifetimes = [r[1] for r in results]
        # Lifetime should decrease as multiplier increases
        for i in range(len(lifetimes) - 1):
            assert lifetimes[i] >= lifetimes[i + 1]


class TestMeanLifetimePositive:
    """Mean lifetime must be positive for any risk profile."""

    def test_mean_lifetime_positive(self):
        risks = _standard_risks()
        result = compute_competing_risks(risks, duration_years=_DURATION)
        assert result.mean_lifetime_years > 0.0

    def test_mean_lifetime_positive_single_risk(self):
        risk = create_component_risk(5.0)
        result = compute_competing_risks([risk], duration_years=_DURATION)
        assert result.mean_lifetime_years > 0.0


class TestMedianLifetimeLessThanDuration:
    """Median should be found within the analysis window for strong enough risks."""

    def test_median_lifetime_less_than_duration(self):
        # Use a high hazard to ensure median is found
        risk = create_component_risk(3.0)  # 3-year MTBF
        result = compute_competing_risks([risk], duration_years=_DURATION, dt_years=0.01)

        # Median of exponential with rate 1/3 per year ≈ 3 * ln(2) ≈ 2.08 years
        assert result.median_lifetime_years < _DURATION
        expected_median = 3.0 * math.log(2)
        assert abs(result.median_lifetime_years - expected_median) < 0.1


class TestResultTypes:
    """Verify output types are correct frozen dataclasses."""

    def test_competing_risks_result_type(self):
        risks = _standard_risks()
        result = compute_competing_risks(risks, duration_years=10.0)
        assert isinstance(result, CompetingRisksResult)
        assert isinstance(result.times_years, tuple)
        assert isinstance(result.overall_survival, tuple)
        assert isinstance(result.cause_specific_cif, tuple)
        assert isinstance(result.risk_attribution, tuple)

    def test_population_projection_type(self):
        risks = _standard_risks()
        proj = project_population(risks, initial_population=50, duration_years=10.0)
        assert isinstance(proj, PopulationProjection)
        assert isinstance(proj.times_years, tuple)
        assert isinstance(proj.active_population, tuple)
        assert isinstance(proj.cumulative_failures, tuple)


class TestEdgeCases:
    """Edge case handling."""

    def test_single_risk_cif_equals_one_minus_survival(self):
        risk = create_collision_risk(_SPATIAL_DENSITY, _REL_VELOCITY)
        result = compute_competing_risks([risk], duration_years=10.0, dt_years=0.01)

        survival = np.array(result.overall_survival)
        _name, cif_values = result.cause_specific_cif[0]
        cif = np.array(cif_values)

        np.testing.assert_allclose(cif, 1.0 - survival, atol=1e-5)

    def test_zero_wear_factor_constant(self):
        risk = create_component_risk(_MTBF_YEARS, wear_factor=0.0)
        assert risk.is_constant is True

    def test_population_cost_tracking(self):
        risks = [create_component_risk(10.0)]
        proj = project_population(
            risks, initial_population=100,
            launch_rate_per_year=10.0,
            duration_years=10.0, dt_years=0.1,
            cost_per_launch=50_000_000.0,
        )
        costs = np.array(proj.cost_per_year)
        # Some costs should be positive (launches happening)
        assert np.any(costs > 0.0)
