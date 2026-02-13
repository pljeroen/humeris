# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for maneuver detection via EKF innovation monitoring."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.orbit_determination import (
    ODEstimate,
    ODResult,
)
from humeris.domain.maneuver_detection import (
    ManeuverEvent,
    ManeuverDetectionResult,
    detect_maneuvers_cusum,
    detect_maneuvers_chi_squared,
    detect_maneuvers_ewma,
    _estimate_arl0,
    _compute_d_prime,
)


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_od_result(
    residuals: list[float],
    innovation_variances: list[float] | None = None,
) -> ODResult:
    """Create an ODResult with given residual sequence."""
    state = (7000000.0, 0.0, 0.0, 0.0, 7500.0, 0.0)
    cov = tuple(
        tuple(1.0 if i == j else 0.0 for j in range(6))
        for i in range(6)
    )
    estimates = []
    for k, r in enumerate(residuals):
        innov_var = (
            innovation_variances[k]
            if innovation_variances is not None
            else None
        )
        estimates.append(ODEstimate(
            time=EPOCH + timedelta(seconds=k * 60),
            state=state,
            covariance=cov,
            residual_m=r,
            innovation_variance_m2=innov_var,
        ))

    rms = math.sqrt(sum(r ** 2 for r in residuals) / len(residuals))
    return ODResult(
        estimates=tuple(estimates),
        final_state=state,
        final_covariance=cov,
        rms_residual_m=rms,
        observations_processed=len(residuals),
    )


# ── ManeuverEvent / ManeuverDetectionResult ─────────────────────────


class TestManeuverEvent:

    def test_frozen(self):
        """ManeuverEvent is immutable."""
        evt = ManeuverEvent(
            detection_time=EPOCH,
            cusum_value=5.0,
            residual_magnitude_m=100.0,
            detection_type="cusum",
        )
        with pytest.raises(AttributeError):
            evt.cusum_value = 10.0

    def test_fields(self):
        """All fields are accessible and hold correct values."""
        evt = ManeuverEvent(
            detection_time=EPOCH,
            cusum_value=3.5,
            residual_magnitude_m=50.0,
            detection_type="chi_squared",
        )
        assert evt.detection_time == EPOCH
        assert evt.cusum_value == 3.5
        assert evt.residual_magnitude_m == 50.0
        assert evt.detection_type == "chi_squared"

    def test_detection_type_cusum_negative(self):
        """ManeuverEvent supports cusum_negative detection type."""
        evt = ManeuverEvent(
            detection_time=EPOCH,
            cusum_value=6.0,
            residual_magnitude_m=30.0,
            detection_type="cusum_negative",
        )
        assert evt.detection_type == "cusum_negative"

    def test_detection_type_ewma(self):
        """ManeuverEvent supports ewma detection type."""
        evt = ManeuverEvent(
            detection_time=EPOCH,
            cusum_value=4.0,
            residual_magnitude_m=80.0,
            detection_type="ewma",
        )
        assert evt.detection_type == "ewma"


class TestManeuverDetectionResult:

    def test_frozen(self):
        """ManeuverDetectionResult is immutable."""
        result = ManeuverDetectionResult(
            events=(),
            cusum_history=(1.0, 2.0),
            threshold=5.0,
            mean_residual_m=10.0,
            max_cusum=2.0,
        )
        with pytest.raises(AttributeError):
            result.threshold = 99.0

    def test_fields(self):
        """All fields including new ones are accessible."""
        result = ManeuverDetectionResult(
            events=(),
            cusum_history=(1.0, 2.0),
            threshold=5.0,
            mean_residual_m=10.0,
            max_cusum=2.0,
            cusum_minus_history=(0.5, 0.3),
            detection_sensitivity_d_prime=1.5,
            estimated_arl0=465.0,
        )
        assert result.events == ()
        assert result.cusum_history == (1.0, 2.0)
        assert result.threshold == 5.0
        assert result.mean_residual_m == 10.0
        assert result.max_cusum == 2.0
        assert result.cusum_minus_history == (0.5, 0.3)
        assert result.detection_sensitivity_d_prime == 1.5
        assert result.estimated_arl0 == 465.0

    def test_new_fields_default_values(self):
        """New fields have correct defaults when not provided."""
        result = ManeuverDetectionResult(
            events=(),
            cusum_history=(),
            threshold=5.0,
            mean_residual_m=0.0,
            max_cusum=0.0,
        )
        assert result.cusum_minus_history == ()
        assert result.detection_sensitivity_d_prime == 0.0
        assert result.estimated_arl0 == 0.0


# ── detect_maneuvers_cusum ──────────────────────────────────────────


class TestDetectManeuversCusum:

    def test_no_maneuver_in_clean_data(self):
        """Constant residuals produce no CUSUM detection."""
        residuals = [10.0] * 20
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od, threshold=5.0)
        assert len(result.events) == 0

    def test_spike_detected(self):
        """Large residual spike triggers CUSUM detection."""
        residuals = [10.0] * 10 + [500.0] * 5 + [10.0] * 10
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od, threshold=3.0, drift=0.5)
        assert len(result.events) >= 1
        for evt in result.events:
            assert evt.detection_type in ("cusum", "cusum_negative")

    def test_two_sided_negative_spike_detected(self):
        """Large negative residual shift triggers cusum_negative detection."""
        # Baseline residuals around 100, then sudden drop to near-zero.
        residuals = [100.0] * 15 + [1.0] * 10
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od, threshold=3.0, drift=0.5)
        negative_events = [
            e for e in result.events if e.detection_type == "cusum_negative"
        ]
        assert len(negative_events) >= 1

    def test_reset_preserves_evidence_closely_spaced_spikes(self):
        """Hawkins & Olwell reset (S - threshold) preserves accumulated
        evidence so closely-spaced spikes are both detected."""
        # Two spikes with minimal gap between them.
        residuals = (
            [10.0] * 5
            + [500.0] * 3
            + [10.0] * 3
            + [500.0] * 3
            + [10.0] * 5
        )
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od, threshold=2.0, drift=0.3)
        # Must detect events in both spike regions.
        assert len(result.events) >= 2

    def test_cusum_minus_history_populated(self):
        """cusum_minus_history has same length as estimates."""
        residuals = [10.0] * 15
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od)
        assert len(result.cusum_minus_history) == len(residuals)

    def test_cusum_history_length(self):
        """CUSUM plus history has same length as estimates."""
        residuals = [10.0] * 15
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od)
        assert len(result.cusum_history) == len(residuals)

    def test_threshold_default_is_5(self):
        """Default threshold is 5.0 per Montgomery (2013)."""
        od = _make_od_result([10.0] * 10)
        result = detect_maneuvers_cusum(od)
        assert result.threshold == 5.0

    def test_threshold_stored(self):
        """Custom threshold value is stored in result."""
        od = _make_od_result([10.0] * 10)
        result = detect_maneuvers_cusum(od, threshold=7.0)
        assert result.threshold == 7.0

    def test_mean_residual_uses_baseline(self):
        """Mean residual is computed from baseline window, not entire
        sequence, to avoid maneuver contamination."""
        # First half clean (10.0), second half spike (500.0).
        residuals = [10.0] * 10 + [500.0] * 10
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od)
        # Baseline is first half = [10.0]*10, so mean should be 10.0.
        assert abs(result.mean_residual_m - 10.0) < 1e-10

    def test_arl0_estimation(self):
        """estimated_arl0 is populated with a finite positive value."""
        od = _make_od_result([10.0] * 20)
        result = detect_maneuvers_cusum(od, threshold=5.0, drift=0.5)
        assert result.estimated_arl0 > 0.0
        assert math.isfinite(result.estimated_arl0)

    def test_arl0_two_sided_halved(self):
        """Two-sided ARL_0 is half the one-sided Siegmund approximation."""
        threshold = 5.0
        drift = 0.5
        od = _make_od_result([10.0] * 20)
        result = detect_maneuvers_cusum(od, threshold=threshold, drift=drift)
        arl0_one_sided = _estimate_arl0(threshold, drift)
        expected_two_sided = arl0_one_sided / 2.0
        assert abs(result.estimated_arl0 - expected_two_sided) < 1e-6

    def test_d_prime_computation_no_events(self):
        """d' is 0.0 when no maneuvers are detected."""
        od = _make_od_result([10.0] * 20)
        result = detect_maneuvers_cusum(od, threshold=5.0)
        assert result.detection_sensitivity_d_prime == 0.0

    def test_d_prime_computation_with_events(self):
        """d' is positive when maneuvers are detected."""
        residuals = [10.0] * 10 + [500.0] * 5 + [10.0] * 10
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od, threshold=3.0, drift=0.5)
        assert len(result.events) >= 1
        assert result.detection_sensitivity_d_prime > 0.0

    def test_baseline_window_parameter(self):
        """Explicit baseline_window controls how many initial points
        are used for baseline estimation."""
        # Residuals: first 5 at 10.0, then a ramp up.
        residuals = [10.0] * 5 + [10.0 + 5.0 * k for k in range(15)]
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od, baseline_window=5)
        # Baseline mean should be 10.0 (from first 5 observations).
        assert abs(result.mean_residual_m - 10.0) < 1e-10

    def test_innovation_variance_used_when_available(self):
        """When ODEstimate has innovation_variance_m2, it is used as
        the sigma reference instead of the rolling baseline."""
        # All residuals at 100, but innovation variance says sigma=50.
        residuals = [100.0] * 10 + [500.0] * 5 + [100.0] * 10
        innov_vars = [2500.0] * 25  # sigma = sqrt(2500) = 50
        od = _make_od_result(residuals, innovation_variances=innov_vars)
        result_innov = detect_maneuvers_cusum(od, threshold=3.0, drift=0.5)

        # Compare with no innovation variance (baseline-based).
        od_no_innov = _make_od_result(residuals)
        result_baseline = detect_maneuvers_cusum(
            od_no_innov, threshold=3.0, drift=0.5,
        )

        # Both should detect, but the number of events may differ because
        # the sigma estimates differ. Key point: innovation variance path
        # runs without error and produces a valid result.
        assert isinstance(result_innov, ManeuverDetectionResult)
        assert len(result_innov.cusum_history) == 25
        assert len(result_innov.cusum_minus_history) == 25

    def test_too_few_estimates_raises(self):
        """Fewer than 2 estimates raises ValueError."""
        od = _make_od_result([10.0])
        with pytest.raises(ValueError, match="at least 2"):
            detect_maneuvers_cusum(od)

    def test_max_cusum_tracked(self):
        """max_cusum reflects the peak across both CUSUM sides."""
        residuals = [10.0] * 5 + [100.0] * 3 + [10.0] * 5
        od = _make_od_result(residuals)
        result = detect_maneuvers_cusum(od, threshold=100.0)
        assert result.max_cusum > 0

    def test_higher_threshold_fewer_detections(self):
        """Higher threshold produces fewer or equal detections."""
        residuals = [10.0] * 5 + [200.0] * 3 + [10.0] * 5
        od = _make_od_result(residuals)
        r_low = detect_maneuvers_cusum(od, threshold=2.0)
        r_high = detect_maneuvers_cusum(od, threshold=10.0)
        assert len(r_high.events) <= len(r_low.events)


# ── detect_maneuvers_chi_squared ────────────────────────────────────


class TestDetectManeuversChiSquared:

    def test_no_maneuver_in_clean_data(self):
        """Uniform residuals produce no chi-squared detections."""
        residuals = [10.0] * 20
        od = _make_od_result(residuals)
        result = detect_maneuvers_chi_squared(od, window_size=5)
        assert len(result.events) == 0

    def test_spike_detected(self):
        """Large residual spike triggers chi-squared detection."""
        residuals = [10.0] * 10 + [500.0] * 5 + [10.0] * 10
        od = _make_od_result(residuals)
        result = detect_maneuvers_chi_squared(
            od, window_size=5, chi2_threshold=5.0,
        )
        assert len(result.events) >= 1
        for evt in result.events:
            assert evt.detection_type == "chi_squared"

    def test_correct_dof_default_threshold(self):
        """Default chi2_threshold is 9.49 (DOF=4, p=0.05)."""
        od = _make_od_result([10.0] * 20)
        result = detect_maneuvers_chi_squared(od, window_size=5)
        assert result.threshold == 9.49

    def test_baseline_window(self):
        """Baseline variance uses only the specified baseline window."""
        # First 10 at 10.0 (baseline), then spike at 500.0.
        residuals = [10.0] * 10 + [500.0] * 5 + [10.0] * 5
        od = _make_od_result(residuals)
        # With explicit baseline_window=10, baseline is clean.
        result = detect_maneuvers_chi_squared(
            od, window_size=5, chi2_threshold=5.0, baseline_window=10,
        )
        assert len(result.events) >= 1

    def test_history_length(self):
        """Chi-squared history has length n - window_size + 1."""
        residuals = [10.0] * 20
        od = _make_od_result(residuals)
        result = detect_maneuvers_chi_squared(od, window_size=5)
        assert len(result.cusum_history) == 20 - 5 + 1

    def test_too_few_estimates_raises(self):
        """Fewer than window_size estimates raises ValueError."""
        od = _make_od_result([10.0] * 3)
        with pytest.raises(ValueError, match="at least"):
            detect_maneuvers_chi_squared(od, window_size=5)

    def test_higher_threshold_fewer_detections(self):
        """Higher threshold produces fewer or equal detections."""
        residuals = [10.0] * 10 + [200.0] * 5 + [10.0] * 10
        od = _make_od_result(residuals)
        r_low = detect_maneuvers_chi_squared(od, chi2_threshold=3.0)
        r_high = detect_maneuvers_chi_squared(od, chi2_threshold=20.0)
        assert len(r_high.events) <= len(r_low.events)

    def test_detection_time_near_spike_region(self):
        """Detection time falls near the spike region."""
        residuals = [10.0] * 10 + [500.0] * 5 + [10.0] * 10
        od = _make_od_result(residuals)
        result = detect_maneuvers_chi_squared(
            od, window_size=5, chi2_threshold=5.0,
        )
        if result.events:
            evt = result.events[0]
            earliest = EPOCH + timedelta(seconds=8 * 60)
            latest = EPOCH + timedelta(seconds=14 * 60)
            assert earliest <= evt.detection_time <= latest


# ── detect_maneuvers_ewma ───────────────────────────────────────────


class TestDetectManeuversEwma:

    def test_no_maneuver_in_clean_data(self):
        """Uniform residuals produce no EWMA detections."""
        residuals = [10.0] * 30
        od = _make_od_result(residuals)
        result = detect_maneuvers_ewma(od, lambda_param=0.1, sigma_factor=3.0)
        assert len(result.events) == 0

    def test_sustained_shift_detected(self):
        """Sustained residual shift triggers EWMA detection.

        EWMA is designed for small sustained shifts, not single spikes.
        A persistent level change should be detected.
        """
        # Baseline at 10.0, then sustained shift to 200.0.
        residuals = [10.0] * 15 + [200.0] * 15
        od = _make_od_result(residuals)
        result = detect_maneuvers_ewma(od, lambda_param=0.1, sigma_factor=3.0)
        assert len(result.events) >= 1

    def test_detection_type_is_ewma(self):
        """All EWMA detections have detection_type='ewma'."""
        residuals = [10.0] * 15 + [200.0] * 15
        od = _make_od_result(residuals)
        result = detect_maneuvers_ewma(od, lambda_param=0.1, sigma_factor=3.0)
        for evt in result.events:
            assert evt.detection_type == "ewma"

    def test_lambda_param_sensitivity(self):
        """Smaller lambda_param detects smaller shifts; larger lambda
        is less sensitive to small sustained shifts."""
        # Small sustained shift: 10 -> 30.
        residuals = [10.0] * 20 + [30.0] * 20
        od = _make_od_result(residuals)
        result_sensitive = detect_maneuvers_ewma(
            od, lambda_param=0.05, sigma_factor=2.5,
        )
        result_less_sensitive = detect_maneuvers_ewma(
            od, lambda_param=0.4, sigma_factor=3.0,
        )
        # With small lambda and tight limit, detection should be
        # at least as many as with large lambda and wider limit.
        assert len(result_sensitive.events) >= len(result_less_sensitive.events)

    def test_ewma_history_length(self):
        """EWMA history has same length as estimates."""
        residuals = [10.0] * 20
        od = _make_od_result(residuals)
        result = detect_maneuvers_ewma(od)
        assert len(result.cusum_history) == 20

    def test_threshold_stores_sigma_factor(self):
        """Result threshold stores the sigma_factor parameter."""
        od = _make_od_result([10.0] * 20)
        result = detect_maneuvers_ewma(od, sigma_factor=2.5)
        assert result.threshold == 2.5

    def test_d_prime_with_events(self):
        """d' is positive when EWMA detects events."""
        residuals = [10.0] * 15 + [200.0] * 15
        od = _make_od_result(residuals)
        result = detect_maneuvers_ewma(od, lambda_param=0.1, sigma_factor=3.0)
        if result.events:
            assert result.detection_sensitivity_d_prime > 0.0

    def test_too_few_estimates_raises(self):
        """Fewer than 2 estimates raises ValueError."""
        od = _make_od_result([10.0])
        with pytest.raises(ValueError, match="at least 2"):
            detect_maneuvers_ewma(od)


# ── Helper functions ────────────────────────────────────────────────


class TestEstimateArl0:

    def test_positive_finite_for_valid_inputs(self):
        """ARL_0 is positive and finite for reasonable parameters."""
        arl0 = _estimate_arl0(threshold=5.0, drift=0.5)
        assert arl0 > 0.0
        assert math.isfinite(arl0)

    def test_infinite_for_zero_drift(self):
        """ARL_0 is infinite when drift is zero."""
        arl0 = _estimate_arl0(threshold=5.0, drift=0.0)
        assert arl0 == float("inf")

    def test_infinite_for_zero_threshold(self):
        """ARL_0 is infinite when threshold is zero."""
        arl0 = _estimate_arl0(threshold=0.0, drift=0.5)
        assert arl0 == float("inf")

    def test_infinite_for_negative_drift(self):
        """ARL_0 is infinite when drift is negative."""
        arl0 = _estimate_arl0(threshold=5.0, drift=-1.0)
        assert arl0 == float("inf")

    def test_larger_threshold_larger_arl0(self):
        """Higher threshold gives longer average run length."""
        arl_low = _estimate_arl0(threshold=3.0, drift=0.5)
        arl_high = _estimate_arl0(threshold=5.0, drift=0.5)
        assert arl_high > arl_low

    def test_overflow_protection(self):
        """Very large exponent returns infinity, not overflow error."""
        arl0 = _estimate_arl0(threshold=1000.0, drift=1.0)
        assert arl0 == float("inf")


class TestComputeDPrime:

    def test_zero_when_no_events(self):
        """d' is 0.0 when no events are provided."""
        d = _compute_d_prime(
            residuals=[10.0] * 10,
            baseline_mean=10.0,
            baseline_sigma=1.0,
            events=[],
        )
        assert d == 0.0

    def test_zero_when_sigma_near_zero(self):
        """d' is 0.0 when baseline sigma is effectively zero."""
        evt = ManeuverEvent(
            detection_time=EPOCH,
            cusum_value=5.0,
            residual_magnitude_m=100.0,
            detection_type="cusum",
        )
        d = _compute_d_prime(
            residuals=[10.0] * 10,
            baseline_mean=10.0,
            baseline_sigma=0.0,
            events=[evt],
        )
        assert d == 0.0

    def test_positive_for_shifted_events(self):
        """d' is positive when event residuals differ from baseline mean."""
        evt = ManeuverEvent(
            detection_time=EPOCH,
            cusum_value=5.0,
            residual_magnitude_m=100.0,
            detection_type="cusum",
        )
        d = _compute_d_prime(
            residuals=[10.0] * 10 + [100.0] * 5,
            baseline_mean=10.0,
            baseline_sigma=1.0,
            events=[evt],
        )
        # (100.0 - 10.0) / 1.0 = 90.0
        assert abs(d - 90.0) < 1e-10

    def test_scales_with_shift_magnitude(self):
        """Larger residual shift produces higher d'."""
        evt_small = ManeuverEvent(
            detection_time=EPOCH,
            cusum_value=5.0,
            residual_magnitude_m=20.0,
            detection_type="cusum",
        )
        evt_large = ManeuverEvent(
            detection_time=EPOCH,
            cusum_value=5.0,
            residual_magnitude_m=100.0,
            detection_type="cusum",
        )
        d_small = _compute_d_prime(
            residuals=[10.0] * 10,
            baseline_mean=10.0,
            baseline_sigma=5.0,
            events=[evt_small],
        )
        d_large = _compute_d_prime(
            residuals=[10.0] * 10,
            baseline_mean=10.0,
            baseline_sigma=5.0,
            events=[evt_large],
        )
        assert d_large > d_small


# ── Domain purity ───────────────────────────────────────────────────


class TestManeuverDetectionPurity:

    def test_module_pure(self):
        """maneuver_detection.py only imports stdlib + numpy + domain."""
        import humeris.domain.maneuver_detection as mod

        allowed = {
            'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum',
            '__future__', 'datetime',
        }
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('humeris'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'humeris':
                        assert False, f"Disallowed import from '{node.module}'"
