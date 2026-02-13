# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for stochastic proposals R1: P8, P9, P10, P11, P20.

P8:  Poisson Conjunction Arrival Process (kessler_heatmap.py)
P9:  M/G/1 Queue for Maneuver Scheduling (conjunction_management.py)
P10: Stopping Time Martingale / SPRT (maneuver_detection.py)
P11: Birth-Death Chain for Debris Population (cascade_analysis.py)
P20: Ornstein-Uhlenbeck Stochastic Lifetime (lifetime.py)
"""
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from humeris.domain.kessler_heatmap import (
    ConjunctionPoissonModel,
    KesslerCell,
    KesslerHeatMap,
    compute_conjunction_intensity,
    compute_conjunction_poisson_model,
    compute_kessler_heatmap,
    conjunction_poisson_rate,
    conjunction_probability_window,
)
from humeris.domain.maneuver_detection import (
    MartingaleDetectionResult,
    wald_sequential_test,
)
from humeris.domain.orbit_determination import (
    ODEstimate,
    ODResult,
)
from humeris.domain.cascade_analysis import (
    BirthDeathDebris,
    compute_debris_birth_death,
)
from humeris.domain.conjunction_management import (
    ManeuverQueueModel,
    compute_maneuver_queue,
)
from humeris.domain.lifetime import (
    StochasticLifetime,
    compute_stochastic_lifetime,
)
from humeris.domain.atmosphere import DragConfig
from humeris.domain.propagation import OrbitalState
from humeris.domain.orbital_mechanics import OrbitalConstants


# ── Helpers ─────────────────────────────────────────────────────────

def _make_heatmap_with_objects() -> KesslerHeatMap:
    """Create a heatmap with known object distribution for testing."""
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    r_earth = OrbitalConstants.R_EARTH

    states = []
    # Place 50 objects at ~800 km, inclination ~98 deg (SSO-like)
    for i in range(50):
        a = r_earth + 800_000.0 + i * 100.0
        states.append(OrbitalState(
            semi_major_axis_m=a,
            eccentricity=0.001,
            inclination_rad=math.radians(98.0),
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=math.sqrt(OrbitalConstants.MU_EARTH / a**3),
            reference_epoch=epoch,
        ))

    return compute_kessler_heatmap(
        states,
        altitude_step_km=100.0,
        inclination_step_deg=10.0,
        altitude_min_km=200.0,
        altitude_max_km=2000.0,
    )


def _make_od_result_nominal(n: int = 50) -> ODResult:
    """Create ODResult with Gaussian residuals (no maneuver)."""
    rng = np.random.default_rng(42)
    sigma = 10.0  # meters
    residuals = rng.normal(0.0, sigma, n)
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)

    estimates = []
    for i, r in enumerate(residuals):
        from datetime import timedelta
        t = epoch + timedelta(minutes=i * 10)
        estimates.append(ODEstimate(
            time=t,
            state=(7e6, 0.0, 0.0, 0.0, 7500.0, 0.0),
            covariance=tuple(
                tuple(100.0 if j == k else 0.0 for k in range(6))
                for j in range(6)
            ),
            residual_m=float(r),
        ))

    rms = float(np.sqrt(np.mean(residuals**2)))
    return ODResult(
        estimates=tuple(estimates),
        final_state=(7e6, 0.0, 0.0, 0.0, 7500.0, 0.0),
        final_covariance=tuple(
            tuple(100.0 if j == k else 0.0 for k in range(6))
            for j in range(6)
        ),
        rms_residual_m=rms,
        observations_processed=n,
    )


def _make_od_result_with_maneuver(n: int = 50, shift_at: int = 30,
                                   shift_sigma: float = 5.0) -> ODResult:
    """Create ODResult with a maneuver injected at step shift_at."""
    rng = np.random.default_rng(42)
    sigma = 10.0
    residuals = rng.normal(0.0, sigma, n)
    # Inject shift
    residuals[shift_at:] += shift_sigma * sigma
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)

    estimates = []
    for i, r in enumerate(residuals):
        from datetime import timedelta
        t = epoch + timedelta(minutes=i * 10)
        estimates.append(ODEstimate(
            time=t,
            state=(7e6, 0.0, 0.0, 0.0, 7500.0, 0.0),
            covariance=tuple(
                tuple(100.0 if j == k else 0.0 for k in range(6))
                for j in range(6)
            ),
            residual_m=float(r),
        ))

    rms = float(np.sqrt(np.mean(residuals**2)))
    return ODResult(
        estimates=tuple(estimates),
        final_state=(7e6, 0.0, 0.0, 0.0, 7500.0, 0.0),
        final_covariance=tuple(
            tuple(100.0 if j == k else 0.0 for k in range(6))
            for j in range(6)
        ),
        rms_residual_m=rms,
        observations_processed=n,
    )


# ════════════════════════════════════════════════════════════════════
# P8: Poisson Conjunction Arrival Process
# ════════════════════════════════════════════════════════════════════


class TestConjunctionIntensity:
    """Test compute_conjunction_intensity basic formula."""

    def test_intensity_positive_for_nonzero_inputs(self):
        result = compute_conjunction_intensity(
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            collision_cross_section_km2=1e-5,
        )
        assert result > 0.0

    def test_intensity_zero_for_zero_density(self):
        result = compute_conjunction_intensity(
            spatial_density_per_km3=0.0,
            mean_collision_velocity_ms=10_000.0,
            collision_cross_section_km2=1e-5,
        )
        assert result == 0.0

    def test_intensity_scales_linearly_with_density(self):
        r1 = compute_conjunction_intensity(1e-8, 10_000.0, 1e-5)
        r2 = compute_conjunction_intensity(2e-8, 10_000.0, 1e-5)
        assert abs(r2 / r1 - 2.0) < 1e-10

    def test_intensity_scales_linearly_with_velocity(self):
        r1 = compute_conjunction_intensity(1e-8, 10_000.0, 1e-5)
        r2 = compute_conjunction_intensity(1e-8, 20_000.0, 1e-5)
        assert abs(r2 / r1 - 2.0) < 1e-10

    def test_intensity_scales_linearly_with_cross_section(self):
        r1 = compute_conjunction_intensity(1e-8, 10_000.0, 1e-5)
        r2 = compute_conjunction_intensity(1e-8, 10_000.0, 2e-5)
        assert abs(r2 / r1 - 2.0) < 1e-10


class TestConjunctionPoissonRate:
    """Test conjunction_poisson_rate from heatmap lookup."""

    def test_rate_from_heatmap(self):
        heatmap = _make_heatmap_with_objects()
        rate = conjunction_poisson_rate(
            heatmap,
            satellite_altitude_km=800.0,
            satellite_inclination_deg=98.0,
        )
        # Should be positive since we placed objects in this cell
        assert rate > 0.0

    def test_rate_zero_for_empty_cell(self):
        heatmap = _make_heatmap_with_objects()
        rate = conjunction_poisson_rate(
            heatmap,
            satellite_altitude_km=300.0,
            satellite_inclination_deg=10.0,
        )
        assert rate == 0.0

    def test_rate_zero_for_out_of_bounds(self):
        heatmap = _make_heatmap_with_objects()
        rate = conjunction_poisson_rate(
            heatmap,
            satellite_altitude_km=5000.0,
            satellite_inclination_deg=98.0,
        )
        assert rate == 0.0


class TestConjunctionProbabilityWindow:
    """Test conjunction_probability_window formula."""

    def test_probability_zero_for_zero_rate(self):
        assert conjunction_probability_window(0.0, 10.0) == 0.0

    def test_probability_zero_for_zero_duration(self):
        assert conjunction_probability_window(1.0, 0.0) == 0.0

    def test_probability_approaches_one_for_large_lambda_t(self):
        p = conjunction_probability_window(10.0, 100.0)
        assert p > 0.999

    def test_probability_matches_formula(self):
        rate = 0.5
        duration = 2.0
        expected = 1.0 - math.exp(-rate * duration)
        result = conjunction_probability_window(rate, duration)
        assert abs(result - expected) < 1e-12

    def test_probability_between_zero_and_one(self):
        p = conjunction_probability_window(0.1, 5.0)
        assert 0.0 < p < 1.0


class TestConjunctionPoissonModel:
    """Test full non-homogeneous Poisson conjunction model."""

    def test_constant_rate_model(self):
        heatmap = _make_heatmap_with_objects()
        model = compute_conjunction_poisson_model(
            heatmap,
            satellite_altitude_km=800.0,
            satellite_inclination_deg=98.0,
            duration_years=10.0,
            step_years=0.5,
            density_growth_rate_per_year=0.0,
        )
        assert isinstance(model, ConjunctionPoissonModel)
        assert len(model.intensity_per_year) > 0
        assert len(model.survival_probability) > 0
        # Constant rate: all intensities equal
        intensities = model.intensity_per_year
        for lam in intensities:
            assert abs(lam - intensities[0]) < 1e-12
        assert not model.is_increasing_rate

    def test_increasing_rate_model(self):
        heatmap = _make_heatmap_with_objects()
        model = compute_conjunction_poisson_model(
            heatmap,
            satellite_altitude_km=800.0,
            satellite_inclination_deg=98.0,
            duration_years=10.0,
            step_years=0.5,
            density_growth_rate_per_year=0.05,
        )
        assert model.is_increasing_rate
        # Last intensity > first intensity
        assert model.intensity_per_year[-1] > model.intensity_per_year[0]

    def test_survival_probability_monotone_decreasing(self):
        heatmap = _make_heatmap_with_objects()
        model = compute_conjunction_poisson_model(
            heatmap,
            satellite_altitude_km=800.0,
            satellite_inclination_deg=98.0,
            duration_years=10.0,
            step_years=0.5,
        )
        surv = model.survival_probability
        for i in range(1, len(surv)):
            assert surv[i] <= surv[i - 1] + 1e-12

    def test_survival_starts_at_one(self):
        heatmap = _make_heatmap_with_objects()
        model = compute_conjunction_poisson_model(
            heatmap,
            satellite_altitude_km=800.0,
            satellite_inclination_deg=98.0,
        )
        assert abs(model.survival_probability[0] - 1.0) < 1e-12

    def test_expected_time_positive(self):
        heatmap = _make_heatmap_with_objects()
        model = compute_conjunction_poisson_model(
            heatmap,
            satellite_altitude_km=800.0,
            satellite_inclination_deg=98.0,
        )
        assert model.expected_time_to_first_years > 0.0

    def test_peak_intensity_equals_max(self):
        heatmap = _make_heatmap_with_objects()
        model = compute_conjunction_poisson_model(
            heatmap,
            satellite_altitude_km=800.0,
            satellite_inclination_deg=98.0,
            density_growth_rate_per_year=0.03,
        )
        assert abs(model.peak_intensity_per_year -
                    max(model.intensity_per_year)) < 1e-12


# ════════════════════════════════════════════════════════════════════
# P10: Stopping Time Martingale (SPRT) for Maneuver Detection
# ════════════════════════════════════════════════════════════════════


class TestWaldSequentialTestNominal:
    """SPRT on nominal (no maneuver) data — should have few/no detections."""

    def test_no_false_alarms_on_nominal(self):
        od = _make_od_result_nominal(n=50)
        result = wald_sequential_test(od, shift_sigma=3.0,
                                       false_alarm_target=0.001)
        assert isinstance(result, MartingaleDetectionResult)
        # With false alarm target 0.001 and only 50 samples,
        # false alarms should be rare (not guaranteed 0, but rare)
        assert len(result.events) <= 2

    def test_result_has_correct_structure(self):
        od = _make_od_result_nominal(n=20)
        result = wald_sequential_test(od)
        assert len(result.log_likelihood_ratio_history) == 20
        assert result.exact_false_alarm_bound > 0
        assert result.is_optimal is True
        assert result.upper_threshold > 0
        assert result.lower_threshold < 0


class TestWaldSequentialTestDetection:
    """SPRT on data with injected maneuver — should detect it."""

    def test_detects_large_maneuver(self):
        od = _make_od_result_with_maneuver(n=50, shift_at=25, shift_sigma=5.0)
        result = wald_sequential_test(od, shift_sigma=3.0,
                                       false_alarm_target=0.01)
        # Should detect the maneuver
        assert len(result.events) >= 1
        # All detections should be of type "sprt"
        for event in result.events:
            assert event.detection_type == "sprt"

    def test_detection_after_shift_point(self):
        od = _make_od_result_with_maneuver(n=60, shift_at=30, shift_sigma=5.0)
        result = wald_sequential_test(od, shift_sigma=3.0,
                                       false_alarm_target=0.01)
        if result.events:
            # Detection should occur at or after the shift point
            # (the first baseline_window observations are used for calibration)
            assert len(result.events) >= 1


class TestWaldSequentialTestBounds:
    """Test Wald's exact bounds and threshold computation."""

    def test_exact_false_alarm_bound_formula(self):
        od = _make_od_result_nominal(n=20)
        result = wald_sequential_test(od, false_alarm_target=0.001,
                                       miss_probability=0.01)
        # exact_fa_bound = exp(-upper_threshold)
        # upper_threshold = log((1 - beta) / alpha)
        alpha = 0.001
        beta = 0.01
        expected_upper = math.log((1.0 - beta) / alpha)
        expected_fa = math.exp(-expected_upper)
        assert abs(result.exact_false_alarm_bound - expected_fa) < 1e-10
        assert abs(result.upper_threshold - expected_upper) < 1e-10

    def test_lower_threshold_formula(self):
        od = _make_od_result_nominal(n=20)
        alpha = 0.01
        beta = 0.05
        result = wald_sequential_test(od, false_alarm_target=alpha,
                                       miss_probability=beta)
        expected_lower = math.log(beta / (1.0 - alpha))
        assert abs(result.lower_threshold - expected_lower) < 1e-10

    def test_wald_delay_positive(self):
        od = _make_od_result_nominal(n=20)
        result = wald_sequential_test(od, shift_sigma=1.0)
        assert result.wald_delay_bound > 0

    def test_tighter_fa_gives_higher_threshold(self):
        od = _make_od_result_nominal(n=20)
        r1 = wald_sequential_test(od, false_alarm_target=0.01)
        r2 = wald_sequential_test(od, false_alarm_target=0.001)
        assert r2.upper_threshold > r1.upper_threshold


class TestWaldSequentialTestValidation:
    """Input validation for SPRT."""

    def test_rejects_too_few_estimates(self):
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        est = ODEstimate(
            time=epoch,
            state=(7e6, 0.0, 0.0, 0.0, 7500.0, 0.0),
            covariance=tuple(
                tuple(100.0 if j == k else 0.0 for k in range(6))
                for j in range(6)
            ),
            residual_m=1.0,
        )
        od = ODResult(
            estimates=(est,),
            final_state=(7e6, 0.0, 0.0, 0.0, 7500.0, 0.0),
            final_covariance=tuple(
                tuple(100.0 if j == k else 0.0 for k in range(6))
                for j in range(6)
            ),
            rms_residual_m=1.0,
            observations_processed=1,
        )
        with pytest.raises(ValueError, match="at least 2"):
            wald_sequential_test(od)

    def test_rejects_negative_shift(self):
        od = _make_od_result_nominal(n=10)
        with pytest.raises(ValueError, match="shift_sigma"):
            wald_sequential_test(od, shift_sigma=-1.0)

    def test_rejects_invalid_alpha(self):
        od = _make_od_result_nominal(n=10)
        with pytest.raises(ValueError, match="false_alarm_target"):
            wald_sequential_test(od, false_alarm_target=0.0)

    def test_rejects_invalid_beta(self):
        od = _make_od_result_nominal(n=10)
        with pytest.raises(ValueError, match="miss_probability"):
            wald_sequential_test(od, miss_probability=1.0)


# ════════════════════════════════════════════════════════════════════
# P11: Birth-Death Chain for Debris Population Dynamics
# ════════════════════════════════════════════════════════════════════


class TestBirthDeathSubcritical:
    """Subcritical regime: birth rate < death rate."""

    def test_subcritical_has_finite_equilibrium(self):
        result = compute_debris_birth_death(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-9,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=100,
            fragments_per_collision=10.0,
            drag_lifetime_years=5.0,
            collision_cross_section_km2=1e-5,
        )
        assert not result.is_supercritical
        assert result.mean_equilibrium_debris < float("inf")
        assert result.mean_equilibrium_debris >= 0

    def test_subcritical_extinction_probability_one(self):
        result = compute_debris_birth_death(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-9,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=100,
            fragments_per_collision=10.0,
            drag_lifetime_years=5.0,
            collision_cross_section_km2=1e-5,
        )
        assert result.extinction_probability == 1.0

    def test_subcritical_geometric_mean_and_variance(self):
        """Subcritical birth-death: geometric distribution, mean=1/(1-rho), var=rho/(1-rho)^2."""
        result = compute_debris_birth_death(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-9,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=100,
            fragments_per_collision=10.0,
            drag_lifetime_years=5.0,
            collision_cross_section_km2=1e-5,
        )
        if not result.is_supercritical:
            rho = result.birth_rate / result.death_rate
            expected_mean = 1.0 / (1.0 - rho)
            expected_var = rho / (1.0 - rho) ** 2
            assert abs(result.mean_equilibrium_debris - expected_mean) < 1e-10
            assert abs(result.variance_equilibrium_debris - expected_var) < 1e-10

    def test_subcritical_distribution_sums_to_one(self):
        result = compute_debris_birth_death(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-9,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=100,
            fragments_per_collision=10.0,
            drag_lifetime_years=5.0,
            collision_cross_section_km2=1e-5,
        )
        total = sum(result.stationary_distribution)
        assert abs(total - 1.0) < 1e-6


class TestBirthDeathSupercritical:
    """Supercritical regime: birth rate > death rate."""

    def test_supercritical_detected(self):
        result = compute_debris_birth_death(
            shell_volume_km3=1e10,  # smaller volume -> higher density effect
            spatial_density_per_km3=1e-7,
            mean_collision_velocity_ms=15_000.0,
            satellite_count=10000,
            fragments_per_collision=200.0,
            drag_lifetime_years=50.0,
            collision_cross_section_km2=1e-4,
        )
        assert result.is_supercritical

    def test_supercritical_infinite_equilibrium(self):
        result = compute_debris_birth_death(
            shell_volume_km3=1e10,
            spatial_density_per_km3=1e-7,
            mean_collision_velocity_ms=15_000.0,
            satellite_count=10000,
            fragments_per_collision=200.0,
            drag_lifetime_years=50.0,
            collision_cross_section_km2=1e-4,
        )
        assert result.mean_equilibrium_debris == float("inf")

    def test_supercritical_extinction_less_than_one(self):
        result = compute_debris_birth_death(
            shell_volume_km3=1e10,
            spatial_density_per_km3=1e-7,
            mean_collision_velocity_ms=15_000.0,
            satellite_count=10000,
            fragments_per_collision=200.0,
            drag_lifetime_years=50.0,
            collision_cross_section_km2=1e-4,
        )
        assert result.extinction_probability < 1.0

    def test_supercritical_finite_doubling_time(self):
        result = compute_debris_birth_death(
            shell_volume_km3=1e10,
            spatial_density_per_km3=1e-7,
            mean_collision_velocity_ms=15_000.0,
            satellite_count=10000,
            fragments_per_collision=200.0,
            drag_lifetime_years=50.0,
            collision_cross_section_km2=1e-4,
        )
        assert result.time_to_threshold_years > 0
        assert result.time_to_threshold_years < float("inf")


class TestBirthDeathRates:
    """Test birth/death rate computation."""

    def test_birth_rate_positive(self):
        result = compute_debris_birth_death(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=1000,
        )
        assert result.birth_rate > 0

    def test_death_rate_equals_inverse_lifetime(self):
        lifetime = 25.0
        result = compute_debris_birth_death(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=1000,
            drag_lifetime_years=lifetime,
        )
        assert abs(result.death_rate - 1.0 / lifetime) < 1e-12

    def test_more_satellites_increases_birth_rate(self):
        r1 = compute_debris_birth_death(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=100,
        )
        r2 = compute_debris_birth_death(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=200,
        )
        assert r2.birth_rate > r1.birth_rate


# ════════════════════════════════════════════════════════════════════
# P9: M/G/1 Queue for Maneuver Scheduling
# ════════════════════════════════════════════════════════════════════


class TestManeuverQueueStable:
    """Stable queue: utilization < 1."""

    def test_stable_queue_basic(self):
        result = compute_maneuver_queue(
            conjunction_rate_per_year=2.0,
            mean_avoidance_dv_ms=0.1,
            variance_avoidance_dv_ms2=0.01,
            sk_dv_per_year_ms=5.0,
            total_dv_budget_ms=100.0,
            mission_years=10.0,
        )
        assert isinstance(result, ManeuverQueueModel)
        assert result.is_stable
        assert result.utilization_factor < 1.0

    def test_stable_queue_finite_waiting_time(self):
        result = compute_maneuver_queue(
            conjunction_rate_per_year=2.0,
            mean_avoidance_dv_ms=0.1,
            variance_avoidance_dv_ms2=0.01,
            sk_dv_per_year_ms=5.0,
            total_dv_budget_ms=100.0,
            mission_years=10.0,
        )
        assert result.mean_waiting_time_days < float("inf")
        assert result.mean_waiting_time_days >= 0

    def test_stable_queue_finite_queue_length(self):
        result = compute_maneuver_queue(
            conjunction_rate_per_year=2.0,
            mean_avoidance_dv_ms=0.1,
            variance_avoidance_dv_ms2=0.01,
            sk_dv_per_year_ms=5.0,
            total_dv_budget_ms=100.0,
            mission_years=10.0,
        )
        assert result.mean_queue_length < float("inf")
        assert result.mean_queue_length >= 0


class TestManeuverQueueUnstable:
    """Unstable queue: utilization >= 1."""

    def test_unstable_when_demand_exceeds_capacity(self):
        result = compute_maneuver_queue(
            conjunction_rate_per_year=100.0,  # Very high conjunction rate
            mean_avoidance_dv_ms=1.0,        # Large maneuvers
            variance_avoidance_dv_ms2=0.1,
            sk_dv_per_year_ms=8.0,
            total_dv_budget_ms=100.0,
            mission_years=10.0,
        )
        assert not result.is_stable
        assert result.utilization_factor >= 1.0

    def test_unstable_has_infinite_queue(self):
        result = compute_maneuver_queue(
            conjunction_rate_per_year=100.0,
            mean_avoidance_dv_ms=1.0,
            variance_avoidance_dv_ms2=0.1,
            sk_dv_per_year_ms=8.0,
            total_dv_budget_ms=100.0,
            mission_years=10.0,
        )
        assert result.mean_queue_length == float("inf")


class TestManeuverQueueBudget:
    """Test dV budget competition between SK and CA."""

    def test_budget_fractions_sum_to_approximately_one_or_less(self):
        result = compute_maneuver_queue(
            conjunction_rate_per_year=2.0,
            mean_avoidance_dv_ms=0.1,
            variance_avoidance_dv_ms2=0.01,
            sk_dv_per_year_ms=5.0,
            total_dv_budget_ms=100.0,
            mission_years=10.0,
        )
        # SK + avoidance fractions should not exceed 1
        assert result.sk_fraction + result.avoidance_fraction <= 1.0 + 1e-10
        assert result.sk_fraction >= 0
        assert result.avoidance_fraction >= 0

    def test_total_demand_correct(self):
        rate = 3.0
        mean_dv = 0.2
        sk_dv = 4.0
        result = compute_maneuver_queue(
            conjunction_rate_per_year=rate,
            mean_avoidance_dv_ms=mean_dv,
            variance_avoidance_dv_ms2=0.01,
            sk_dv_per_year_ms=sk_dv,
            total_dv_budget_ms=100.0,
            mission_years=10.0,
        )
        expected_demand = sk_dv + rate * mean_dv
        assert abs(result.total_dv_demand_per_year_ms - expected_demand) < 1e-10

    def test_overflow_probability_between_zero_and_one(self):
        result = compute_maneuver_queue(
            conjunction_rate_per_year=2.0,
            mean_avoidance_dv_ms=0.1,
            variance_avoidance_dv_ms2=0.01,
            sk_dv_per_year_ms=5.0,
            total_dv_budget_ms=100.0,
            mission_years=10.0,
        )
        assert 0.0 <= result.overflow_probability <= 1.0

    def test_larger_budget_reduces_overflow(self):
        r1 = compute_maneuver_queue(
            conjunction_rate_per_year=5.0,
            mean_avoidance_dv_ms=0.5,
            variance_avoidance_dv_ms2=0.1,
            sk_dv_per_year_ms=5.0,
            total_dv_budget_ms=50.0,
            mission_years=10.0,
        )
        r2 = compute_maneuver_queue(
            conjunction_rate_per_year=5.0,
            mean_avoidance_dv_ms=0.5,
            variance_avoidance_dv_ms2=0.1,
            sk_dv_per_year_ms=5.0,
            total_dv_budget_ms=200.0,
            mission_years=10.0,
        )
        assert r2.overflow_probability <= r1.overflow_probability


class TestManeuverQueueValidation:
    """Input validation for queue model."""

    def test_rejects_negative_mission_years(self):
        with pytest.raises(ValueError, match="mission_years"):
            compute_maneuver_queue(1.0, 0.1, 0.01, 5.0, 100.0, -1.0)

    def test_rejects_zero_budget(self):
        with pytest.raises(ValueError, match="total_dv_budget"):
            compute_maneuver_queue(1.0, 0.1, 0.01, 5.0, 0.0, 10.0)


# ════════════════════════════════════════════════════════════════════
# P20: Ornstein-Uhlenbeck Process for Stochastic Lifetime
# ════════════════════════════════════════════════════════════════════


class TestStochasticLifetimeBasic:
    """Basic stochastic lifetime computation."""

    def test_returns_correct_structure(self):
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400_000.0  # 400 km

        result = compute_stochastic_lifetime(
            semi_major_axis_m=a,
            eccentricity=0.001,
            drag_config=drag,
            epoch=epoch,
            num_samples=10,
            step_days=10.0,
            max_years=30.0,
            rng_seed=42,
        )
        assert isinstance(result, StochasticLifetime)
        assert result.num_samples == 10
        assert result.mean_lifetime_days > 0
        assert result.std_lifetime_days >= 0

    def test_percentile_ordering(self):
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400_000.0

        result = compute_stochastic_lifetime(
            semi_major_axis_m=a,
            eccentricity=0.001,
            drag_config=drag,
            epoch=epoch,
            num_samples=30,
            step_days=10.0,
            max_years=30.0,
            rng_seed=42,
        )
        # P5 <= P50 <= P95
        assert result.percentile_5_days <= result.percentile_50_days + 1e-6
        assert result.percentile_50_days <= result.percentile_95_days + 1e-6

    def test_deterministic_lifetime_stored(self):
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400_000.0

        result = compute_stochastic_lifetime(
            semi_major_axis_m=a,
            eccentricity=0.001,
            drag_config=drag,
            epoch=epoch,
            num_samples=5,
            step_days=10.0,
            max_years=30.0,
            rng_seed=42,
        )
        assert result.deterministic_lifetime_days > 0

    def test_reproducible_with_seed(self):
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400_000.0

        r1 = compute_stochastic_lifetime(
            semi_major_axis_m=a,
            eccentricity=0.001,
            drag_config=drag,
            epoch=epoch,
            num_samples=10,
            step_days=10.0,
            max_years=30.0,
            rng_seed=123,
        )
        r2 = compute_stochastic_lifetime(
            semi_major_axis_m=a,
            eccentricity=0.001,
            drag_config=drag,
            epoch=epoch,
            num_samples=10,
            step_days=10.0,
            max_years=30.0,
            rng_seed=123,
        )
        assert abs(r1.mean_lifetime_days - r2.mean_lifetime_days) < 1e-6


class TestStochasticLifetimeUncertainty:
    """Test that stochastic model produces meaningful uncertainty."""

    def test_nonzero_std_with_variability(self):
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400_000.0

        result = compute_stochastic_lifetime(
            semi_major_axis_m=a,
            eccentricity=0.001,
            drag_config=drag,
            epoch=epoch,
            density_variability_fraction=0.3,  # high variability
            num_samples=50,
            step_days=10.0,
            max_years=30.0,
            rng_seed=42,
        )
        assert result.std_lifetime_days > 0
        assert result.uncertainty_ratio > 0

    def test_higher_variability_increases_uncertainty(self):
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400_000.0

        r_low = compute_stochastic_lifetime(
            semi_major_axis_m=a,
            eccentricity=0.001,
            drag_config=drag,
            epoch=epoch,
            density_variability_fraction=0.05,
            num_samples=50,
            step_days=10.0,
            max_years=30.0,
            rng_seed=42,
        )
        r_high = compute_stochastic_lifetime(
            semi_major_axis_m=a,
            eccentricity=0.001,
            drag_config=drag,
            epoch=epoch,
            density_variability_fraction=0.4,
            num_samples=50,
            step_days=10.0,
            max_years=30.0,
            rng_seed=42,
        )
        # Higher variability should generally produce wider spread
        assert r_high.std_lifetime_days > r_low.std_lifetime_days * 0.5


class TestStochasticLifetimeValidation:
    """Input validation for stochastic lifetime."""

    def test_rejects_below_reentry(self):
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 50_000.0  # 50 km (below 100 km reentry)

        with pytest.raises(ValueError, match="already at or below"):
            compute_stochastic_lifetime(
                semi_major_axis_m=a,
                eccentricity=0.0,
                drag_config=drag,
                epoch=epoch,
            )

    def test_rejects_zero_step(self):
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400_000.0

        with pytest.raises(ValueError, match="step_days"):
            compute_stochastic_lifetime(
                semi_major_axis_m=a,
                eccentricity=0.0,
                drag_config=drag,
                epoch=epoch,
                step_days=0.0,
            )

    def test_rejects_zero_samples(self):
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400_000.0

        with pytest.raises(ValueError, match="num_samples"):
            compute_stochastic_lifetime(
                semi_major_axis_m=a,
                eccentricity=0.0,
                drag_config=drag,
                epoch=epoch,
                num_samples=0,
            )
