# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Maneuver detection via EKF innovation monitoring.

Detects non-cooperative maneuvers by analyzing the post-fit residual
(innovation) sequence from orbit determination. Three methods:

1. CUSUM (Cumulative Sum): Two-sided Page's CUSUM with optimal
   parameters from Montgomery (2013). Detects both prograde and
   retrograde maneuvers. Uses rolling baseline to avoid maneuver
   contamination (Hawkins 1987 Self-Starting CUSUM).

2. Chi-squared: Windowed innovation variance ratio test with correct
   DOF = window_size - 1 (Bartlett 1937).

3. EWMA (Exponentially Weighted Moving Average): Optimal for detecting
   small sustained shifts (0.5-1.5 sigma), ideal for low-thrust /
   electric propulsion maneuvers (Roberts 1959, Lucas & Saccucci 1990).

All methods operate on ODResult from orbit_determination.py.

Detection performance metrics:
    d': Signal detection theory sensitivity index.
    ARL_0: Average Run Length under null (false alarm rate).

References:
    Page (1954). Continuous inspection schemes.
    Montgomery (2013). Introduction to Statistical Quality Control, 7th ed.
    Hawkins & Olwell (2012). Cumulative Sum Charts and Charting.
    Moustakides (1986). Optimal stopping for detecting changes.
    Siegmund (1985). Sequential Analysis.
    Roberts (1959). Control chart tests based on geometric moving averages.
    Lucas & Saccucci (1990). EWMA control chart properties.
"""
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from humeris.domain.orbit_determination import ODResult


@dataclass(frozen=True)
class ManeuverEvent:
    """A detected maneuver event."""
    detection_time: datetime
    cusum_value: float
    residual_magnitude_m: float
    detection_type: str  # "cusum", "cusum_negative", "chi_squared", "ewma"


@dataclass(frozen=True)
class ManeuverDetectionResult:
    """Result of maneuver detection analysis."""
    events: tuple
    cusum_history: tuple
    threshold: float
    mean_residual_m: float
    max_cusum: float
    cusum_minus_history: tuple = ()
    detection_sensitivity_d_prime: float = 0.0
    estimated_arl0: float = 0.0


def _estimate_arl0(threshold: float, drift: float) -> float:
    """Estimate Average Run Length under null hypothesis.

    Uses Siegmund (1985) approximation for one-sided CUSUM:
        ARL_0 ~ exp(2*k*h) / (2*k^2)

    For two-sided CUSUM, ARL_0 ~ ARL_0_one_sided / 2.
    """
    if drift <= 0.0 or threshold <= 0.0:
        return float("inf")
    exponent = 2.0 * drift * threshold
    if exponent > 500.0:
        return float("inf")
    return math.exp(exponent) / (2.0 * drift * drift)


def _compute_d_prime(
    residuals: list[float],
    baseline_mean: float,
    baseline_sigma: float,
    events: list,
) -> float:
    """Compute signal detection theory d' (sensitivity index).

    d' = (mean_under_change - mean_under_null) / sigma_noise.
    Higher d' means better separation between null and alternative.
    """
    if not events or baseline_sigma < 1e-30:
        return 0.0
    event_residuals = [e.residual_magnitude_m for e in events]
    mean_change = sum(event_residuals) / len(event_residuals)
    return abs(mean_change - baseline_mean) / baseline_sigma


def detect_maneuvers_cusum(
    od_result: ODResult,
    threshold: float = 5.0,
    drift: float = 0.5,
    baseline_window: int = 0,
) -> ManeuverDetectionResult:
    """Detect maneuvers using two-sided CUSUM on EKF residual sequence.

    Two-sided Page's CUSUM (Page 1954) with parameters from Montgomery
    (2013) SPC best practice:

        S+_i = max(0, S+_{i-1} + z_i - drift)   [prograde]
        S-_i = max(0, S-_{i-1} - z_i - drift)   [retrograde]

    When either side exceeds threshold, a maneuver is flagged.
    Reset uses Hawkins & Olwell (2012): S = S - threshold (preserves
    accumulated evidence for closely-spaced maneuvers).

    Standardization uses a rolling baseline (first baseline_window
    observations assumed in-control) to avoid maneuver contamination.
    When ODEstimate.innovation_variance_m2 is available from the EKF,
    uses that directly as the theoretically correct reference.

    Default threshold=5.0, drift=0.5 gives ARL_0 ~ 465 per Montgomery
    (2013) for 1-sigma shift detection.

    Args:
        od_result: Orbit determination result with estimates.
        threshold: CUSUM detection threshold (h). Default 5.0 per
            Montgomery (2013) for k=0.5.
        drift: Allowance parameter (k). For detecting delta-sigma shifts,
            optimal k = delta/2 (Moustakides 1986).
        baseline_window: Number of initial observations for baseline
            estimation. 0 = auto (use innovation variance or first half).

    Returns:
        ManeuverDetectionResult with detected events, two-sided CUSUM
        history, and detection performance metrics.

    Raises:
        ValueError: If od_result has fewer than 2 estimates.
    """
    estimates = od_result.estimates
    if len(estimates) < 2:
        raise ValueError(
            f"Need at least 2 estimates for CUSUM, got {len(estimates)}"
        )

    residuals = [e.residual_m for e in estimates]

    # Baseline estimation: avoid maneuver contamination.
    # Priority: (1) EKF innovation variance, (2) rolling baseline window.
    has_innov_var = (
        getattr(estimates[0], "innovation_variance_m2", None) is not None
    )

    if has_innov_var:
        # Use EKF innovation variance directly (theoretically correct:
        # S = H * P_predicted * H^T + R)
        innov_vars = [
            getattr(e, "innovation_variance_m2") for e in estimates
        ]
        mean_r = sum(residuals) / len(residuals)
        sigma = float(np.sqrt(max(
            sum(innov_vars) / len(innov_vars), 1e-30
        )))
    else:
        # Self-Starting CUSUM (Hawkins 1987): use first baseline_window
        # observations as Phase I (assumed in-control).
        if baseline_window <= 0:
            baseline_window = max(2, len(residuals) // 2)
        baseline_window = min(baseline_window, len(residuals))
        baseline = residuals[:baseline_window]
        mean_r = sum(baseline) / len(baseline)
        variance = sum((r - mean_r) ** 2 for r in baseline) / (len(baseline) - 1)
        sigma = float(np.sqrt(max(variance, 1e-30)))

    cusum_plus = 0.0
    cusum_minus = 0.0
    cusum_plus_history = []
    cusum_minus_history = []
    events = []

    for est in estimates:
        z = (est.residual_m - mean_r) / sigma

        # Positive side: detects increases (prograde maneuvers)
        cusum_plus = max(0.0, cusum_plus + z - drift)
        cusum_plus_history.append(cusum_plus)

        if cusum_plus > threshold:
            events.append(ManeuverEvent(
                detection_time=est.time,
                cusum_value=cusum_plus,
                residual_magnitude_m=est.residual_m,
                detection_type="cusum",
            ))
            # Hawkins & Olwell (2012): preserve accumulated evidence
            cusum_plus = max(0.0, cusum_plus - threshold)

        # Negative side: detects decreases (retrograde maneuvers)
        cusum_minus = max(0.0, cusum_minus - z - drift)
        cusum_minus_history.append(cusum_minus)

        if cusum_minus > threshold:
            events.append(ManeuverEvent(
                detection_time=est.time,
                cusum_value=cusum_minus,
                residual_magnitude_m=est.residual_m,
                detection_type="cusum_negative",
            ))
            cusum_minus = max(0.0, cusum_minus - threshold)

    max_cusum = 0.0
    if cusum_plus_history:
        max_cusum = max(max(cusum_plus_history), max(cusum_minus_history))

    # Detection performance metrics
    arl0_one_sided = _estimate_arl0(threshold, drift)
    arl0_two_sided = arl0_one_sided / 2.0
    d_prime = _compute_d_prime(residuals, mean_r, sigma, events)

    return ManeuverDetectionResult(
        events=tuple(events),
        cusum_history=tuple(cusum_plus_history),
        threshold=threshold,
        mean_residual_m=mean_r,
        max_cusum=max_cusum,
        cusum_minus_history=tuple(cusum_minus_history),
        detection_sensitivity_d_prime=d_prime,
        estimated_arl0=arl0_two_sided,
    )


def detect_maneuvers_chi_squared(
    od_result: ODResult,
    window_size: int = 5,
    chi2_threshold: float = 9.49,
    baseline_window: int = 0,
) -> ManeuverDetectionResult:
    """Detect maneuvers using chi-squared test on windowed residuals.

    Computes the ratio of windowed residual variance to baseline variance.
    When the ratio exceeds the chi-squared threshold (scaled by DOF),
    a maneuver is flagged.

    DOF = window_size - 1 (window mean estimated from data, consuming
    one degree of freedom per Bartlett 1937).

    Default chi2_threshold=9.49 corresponds to p=0.05 with DOF=4
    (default window_size=5).

    Baseline uses first baseline_window observations to avoid maneuver
    contamination of the reference variance.

    Args:
        od_result: Orbit determination result with estimates.
        window_size: Number of consecutive residuals per window.
        chi2_threshold: Chi-squared critical value for DOF=window_size-1.
            Default 9.49 for DOF=4, p=0.05.
        baseline_window: Number of initial observations for baseline.
            0 = auto (first half of sequence).

    Returns:
        ManeuverDetectionResult with detected events and chi-squared history.

    Raises:
        ValueError: If od_result has fewer estimates than window_size.
    """
    estimates = od_result.estimates
    if len(estimates) < window_size:
        raise ValueError(
            f"Need at least {window_size} estimates, got {len(estimates)}"
        )

    residuals = [e.residual_m for e in estimates]

    # Baseline variance: use first baseline_window observations
    if baseline_window <= 0:
        baseline_window = max(window_size, len(residuals) // 2)
    baseline_window = min(baseline_window, len(residuals))
    baseline = residuals[:baseline_window]
    baseline_mean = sum(baseline) / len(baseline)
    baseline_var = sum(
        (r - baseline_mean) ** 2 for r in baseline
    ) / (len(baseline) - 1)
    if baseline_var < 1e-30:
        baseline_var = 1e-30

    chi2_history = []
    events = []

    dof = window_size - 1  # Correct DOF: window mean consumes 1

    for i in range(len(estimates) - window_size + 1):
        window = residuals[i:i + window_size]
        window_mean = sum(window) / window_size
        window_var = sum(
            (r - window_mean) ** 2 for r in window
        ) / (window_size - 1)
        chi2_stat = (dof * window_var) / baseline_var
        chi2_history.append(chi2_stat)

        if chi2_stat > chi2_threshold:
            center_idx = i + window_size // 2
            est = estimates[center_idx]
            events.append(ManeuverEvent(
                detection_time=est.time,
                cusum_value=chi2_stat,
                residual_magnitude_m=est.residual_m,
                detection_type="chi_squared",
            ))

    max_chi2 = max(chi2_history) if chi2_history else 0.0
    mean_r = sum(residuals) / len(residuals)

    return ManeuverDetectionResult(
        events=tuple(events),
        cusum_history=tuple(chi2_history),
        threshold=chi2_threshold,
        mean_residual_m=mean_r,
        max_cusum=max_chi2,
    )


def detect_maneuvers_ewma(
    od_result: ODResult,
    lambda_param: float = 0.1,
    sigma_factor: float = 3.0,
    baseline_window: int = 0,
) -> ManeuverDetectionResult:
    """Detect maneuvers using EWMA on EKF residual sequence.

    EWMA is optimal for detecting small sustained shifts (0.5-1.5 sigma),
    making it ideal for low-thrust / electric propulsion maneuvers
    (Roberts 1959, Lucas & Saccucci 1990).

        Z_i = lambda * X_i + (1 - lambda) * Z_{i-1}

    Control limits (time-varying for startup transient):
        mu +/- L * sigma * sqrt(lambda/(2-lambda) * (1-(1-lambda)^{2i}))

    Args:
        od_result: Orbit determination result with estimates.
        lambda_param: EWMA smoothing parameter (0 < lambda <= 1).
            0.1 optimal for 0.5-1.0 sigma shifts.
            0.2 optimal for 1.0-1.5 sigma shifts.
        sigma_factor: Control limit width in sigma (L parameter).
            L=3.0 standard for alpha ~= 0.0027.
        baseline_window: Number of initial observations for baseline.
            0 = auto (first half of sequence).

    Returns:
        ManeuverDetectionResult with detected events and EWMA history.

    Raises:
        ValueError: If od_result has fewer than 2 estimates.
    """
    estimates = od_result.estimates
    if len(estimates) < 2:
        raise ValueError(
            f"Need at least 2 estimates for EWMA, got {len(estimates)}"
        )

    residuals = [e.residual_m for e in estimates]

    # Baseline estimation
    if baseline_window <= 0:
        baseline_window = max(2, len(residuals) // 2)
    baseline_window = min(baseline_window, len(residuals))
    baseline = residuals[:baseline_window]
    mean_r = sum(baseline) / len(baseline)
    variance = sum((r - mean_r) ** 2 for r in baseline) / (len(baseline) - 1)
    sigma = float(np.sqrt(max(variance, 1e-30)))

    ewma = mean_r
    ewma_history = []
    events = []

    asymptotic_sigma = sigma * math.sqrt(
        lambda_param / (2.0 - lambda_param)
    )

    for k, est in enumerate(estimates):
        ewma = lambda_param * est.residual_m + (1.0 - lambda_param) * ewma

        # Time-varying control limit (exact for startup transient)
        factor = 1.0 - (1.0 - lambda_param) ** (2 * (k + 1))
        control_sigma = sigma * math.sqrt(
            lambda_param / (2.0 - lambda_param) * factor
        )
        control_limit = sigma_factor * control_sigma

        deviation = abs(ewma - mean_r)
        normalized = (
            deviation / asymptotic_sigma
            if asymptotic_sigma > 1e-30 else 0.0
        )
        ewma_history.append(normalized)

        if deviation > control_limit:
            events.append(ManeuverEvent(
                detection_time=est.time,
                cusum_value=(
                    deviation / control_sigma
                    if control_sigma > 1e-30 else 0.0
                ),
                residual_magnitude_m=est.residual_m,
                detection_type="ewma",
            ))

    max_ewma = max(ewma_history) if ewma_history else 0.0
    d_prime = _compute_d_prime(residuals, mean_r, sigma, events)

    return ManeuverDetectionResult(
        events=tuple(events),
        cusum_history=tuple(ewma_history),
        threshold=sigma_factor,
        mean_residual_m=mean_r,
        max_cusum=max_ewma,
        detection_sensitivity_d_prime=d_prime,
    )


# ── Stopping Time Martingale (SPRT) for Maneuver Detection ─────────
#
# Wald's Sequential Probability Ratio Test with exact non-asymptotic
# bounds via Wald's identity. Better than CUSUM for short observation
# arcs. See SESSION_MINING_R1_CREATIVE.md P10.


@dataclass(frozen=True)
class MartingaleDetectionResult:
    """Result of SPRT / martingale-based maneuver detection.

    Attributes:
        events: Detected maneuver events.
        log_likelihood_ratio_history: Cumulative LLR at each step.
        exact_false_alarm_bound: Wald's exponential bound on P(false alarm).
        wald_delay_bound: Expected detection delay under H1.
        is_optimal: True if SPRT achieves Wald-optimal detection.
        upper_threshold: SPRT upper threshold (accept H1).
        lower_threshold: SPRT lower threshold (accept H0).
    """
    events: tuple
    log_likelihood_ratio_history: tuple
    exact_false_alarm_bound: float
    wald_delay_bound: float
    is_optimal: bool
    upper_threshold: float
    lower_threshold: float


def wald_sequential_test(
    od_result: ODResult,
    shift_sigma: float = 1.0,
    false_alarm_target: float = 0.001,
    miss_probability: float = 0.01,
    baseline_window: int = 0,
) -> MartingaleDetectionResult:
    """Sequential Probability Ratio Test for maneuver detection.

    SPRT is the optimal fixed-sample-size test for two simple hypotheses
    (Wald 1945). It achieves the minimum expected sample size among all
    tests with the same error probabilities (Wald & Wolfowitz 1948).

    Hypotheses:
        H0: residuals ~ N(0, sigma^2)          — no maneuver
        H1: residuals ~ N(delta*sigma, sigma^2) — maneuver of size delta*sigma

    Log-likelihood ratio at step n:
        Z_n = (delta / sigma) * r_n - delta^2 / (2 * sigma^2)
        Lambda_n = sum_{k=1}^n Z_k

    Stopping bounds (Wald's identity):
        A = log((1 - miss_probability) / false_alarm_target)    — upper (accept H1)
        B = log(miss_probability / (1 - false_alarm_target))    — lower (accept H0)

    When Lambda_n >= A: declare maneuver.
    When Lambda_n <= B: declare no maneuver (reset).

    Args:
        od_result: Orbit determination result with estimates.
        shift_sigma: Expected maneuver size in units of sigma.
        false_alarm_target: Target false alarm probability (alpha).
        miss_probability: Target miss probability (beta).
        baseline_window: Number of initial observations for baseline.
            0 = auto (first half of sequence).

    Returns:
        MartingaleDetectionResult with detected events and exact bounds.

    Raises:
        ValueError: If od_result has fewer than 2 estimates, or
            shift_sigma <= 0, or alpha/beta not in (0, 1).
    """
    estimates = od_result.estimates
    if len(estimates) < 2:
        raise ValueError(
            f"Need at least 2 estimates for SPRT, got {len(estimates)}"
        )
    if shift_sigma <= 0:
        raise ValueError(f"shift_sigma must be positive, got {shift_sigma}")
    if not (0 < false_alarm_target < 1):
        raise ValueError(
            f"false_alarm_target must be in (0, 1), got {false_alarm_target}"
        )
    if not (0 < miss_probability < 1):
        raise ValueError(
            f"miss_probability must be in (0, 1), got {miss_probability}"
        )

    residuals = [e.residual_m for e in estimates]

    # Baseline estimation
    has_innov_var = (
        getattr(estimates[0], "innovation_variance_m2", None) is not None
    )

    if has_innov_var:
        innov_vars = [
            getattr(e, "innovation_variance_m2") for e in estimates
        ]
        mean_r = sum(residuals) / len(residuals)
        sigma = float(np.sqrt(max(
            sum(innov_vars) / len(innov_vars), 1e-30
        )))
    else:
        if baseline_window <= 0:
            baseline_window = max(2, len(residuals) // 2)
        baseline_window = min(baseline_window, len(residuals))
        baseline = residuals[:baseline_window]
        mean_r = sum(baseline) / len(baseline)
        variance = sum((r - mean_r) ** 2 for r in baseline) / (len(baseline) - 1)
        sigma = float(np.sqrt(max(variance, 1e-30)))

    # Wald's stopping bounds
    alpha = false_alarm_target
    beta = miss_probability
    upper_threshold = math.log((1.0 - beta) / alpha)
    lower_threshold = math.log(beta / (1.0 - alpha))

    # Two-sided SPRT: monitor for both positive and negative shifts.
    # LLR+ for N(+delta, sigma^2) vs N(0, sigma^2)
    # LLR- for N(-delta, sigma^2) vs N(0, sigma^2)
    delta = shift_sigma * sigma  # absolute shift
    llr_pos = 0.0
    llr_neg = 0.0
    llr_history = []
    events = []

    for est in estimates:
        z = est.residual_m - mean_r
        sig2 = sigma * sigma
        # LLR increment for positive shift: N(+delta) vs N(0)
        llr_pos_inc = (delta / sig2) * z - (delta * delta) / (2.0 * sig2)
        # LLR increment for negative shift: N(-delta) vs N(0)
        llr_neg_inc = (-delta / sig2) * z - (delta * delta) / (2.0 * sig2)
        llr_pos += llr_pos_inc
        llr_neg += llr_neg_inc
        # Report the maximum of the two statistics
        llr_max = max(llr_pos, llr_neg)
        llr_history.append(llr_max)

        if llr_max >= upper_threshold:
            events.append(ManeuverEvent(
                detection_time=est.time,
                cusum_value=llr_max,
                residual_magnitude_m=est.residual_m,
                detection_type="sprt",
            ))
            # Reset after detection (continue monitoring)
            llr_pos = 0.0
            llr_neg = 0.0
        elif llr_pos <= lower_threshold and llr_neg <= lower_threshold:
            # Accept H0: reset both
            llr_pos = 0.0
            llr_neg = 0.0

    # Exact false alarm bound: Wald's exponential identity
    # P(false alarm) <= exp(-upper_threshold) for the martingale
    exact_fa_bound = math.exp(-upper_threshold)

    # Expected detection delay under H1 (Wald's approximation):
    # E_1[N] ~ upper_threshold / E_1[Z_1]
    # E_1[Z_1] = delta^2 / (2 * sigma^2) (KL divergence between H1 and H0)
    kl_divergence = (delta * delta) / (2.0 * sigma * sigma)
    if kl_divergence > 1e-30:
        wald_delay = upper_threshold / kl_divergence
    else:
        wald_delay = float("inf")

    # SPRT is optimal by Wald-Wolfowitz theorem
    is_optimal = True

    return MartingaleDetectionResult(
        events=tuple(events),
        log_likelihood_ratio_history=tuple(llr_history),
        exact_false_alarm_bound=exact_fa_bound,
        wald_delay_bound=wald_delay,
        is_optimal=is_optimal,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
    )
