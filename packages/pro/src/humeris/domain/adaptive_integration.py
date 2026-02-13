# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Dormand-Prince RK4(5) adaptive step-size integrator.

Embedded Runge-Kutta pair with local error estimation, PI step-size
controller, FSAL optimization, and cubic Hermite dense output.

+ domain imports.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import numpy as np

from humeris.domain.numerical_propagation import (
    ForceModel,
    PropagationStep,
    NumericalPropagationResult,
)
from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
)


# --- Dormand-Prince Butcher tableau (7 stages, FSAL) ---

DORMAND_PRINCE_C: tuple[float, ...] = (
    0.0,
    1.0 / 5.0,
    3.0 / 10.0,
    4.0 / 5.0,
    8.0 / 9.0,
    1.0,
    1.0,
)

DORMAND_PRINCE_A: tuple[tuple[float, ...], ...] = (
    (),
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)

# 4th-order solution weights (same as last row of A for FSAL)
DORMAND_PRINCE_B4: tuple[float, ...] = (
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
)

# 5th-order solution weights (for error estimation)
DORMAND_PRINCE_B5: tuple[float, ...] = (
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
)

# Error weights: e_i = b4_i - b5_i (precomputed)
_DORMAND_PRINCE_E: tuple[float, ...] = tuple(
    b4 - b5 for b4, b5 in zip(DORMAND_PRINCE_B4, DORMAND_PRINCE_B5)
)


# --- Types ---

@dataclass(frozen=True)
class AdaptiveStepConfig:
    """Configuration for adaptive step-size integration."""
    rtol: float = 1e-10
    atol: float = 1e-12
    h_init: float = 60.0
    h_min: float = 0.1
    h_max: float = 600.0
    safety_factor: float = 0.9
    max_steps: int = 1_000_000


@dataclass(frozen=True)
class AdaptiveStepResult:
    """Result of an adaptive step-size propagation run."""
    steps: tuple[PropagationStep, ...]
    epoch: datetime
    duration_s: float
    force_model_names: tuple[str, ...]
    total_steps: int
    rejected_steps: int


# --- Dormand-Prince core computational kernel ---

def _dp_full_step(
    t: float,
    y: tuple[float, ...],
    h: float,
    deriv_fn: Callable[[float, tuple[float, ...]], tuple[float, ...]],
    k1_in: tuple[float, ...] | None = None,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Compute all 7 DP stages, the 4th-order solution, and return k_stages.

    Returns:
        (k_stages, y_new, k7) where k_stages is a 7-tuple of derivative
        evaluations, y_new is the 4th-order solution, and k7 is the FSAL
        derivative at y_new.
    """
    n = len(y)
    y_arr = np.array(y)

    # Stage 1 (FSAL: reuse from previous step if available)
    k1 = k1_in if k1_in is not None else deriv_fn(t, y)
    k1_arr = np.array(k1)

    # Stage 2
    a21 = DORMAND_PRINCE_A[1][0]
    y2 = tuple((y_arr + h * a21 * k1_arr).tolist())
    k2 = deriv_fn(t + DORMAND_PRINCE_C[1] * h, y2)
    k2_arr = np.array(k2)

    # Stage 3
    a31, a32 = DORMAND_PRINCE_A[2]
    y3 = tuple((y_arr + h * (a31 * k1_arr + a32 * k2_arr)).tolist())
    k3 = deriv_fn(t + DORMAND_PRINCE_C[2] * h, y3)
    k3_arr = np.array(k3)

    # Stage 4
    a41, a42, a43 = DORMAND_PRINCE_A[3]
    y4 = tuple((y_arr + h * (a41 * k1_arr + a42 * k2_arr + a43 * k3_arr)).tolist())
    k4 = deriv_fn(t + DORMAND_PRINCE_C[3] * h, y4)
    k4_arr = np.array(k4)

    # Stage 5
    a51, a52, a53, a54 = DORMAND_PRINCE_A[4]
    y5 = tuple((y_arr + h * (a51 * k1_arr + a52 * k2_arr + a53 * k3_arr + a54 * k4_arr)).tolist())
    k5 = deriv_fn(t + DORMAND_PRINCE_C[4] * h, y5)
    k5_arr = np.array(k5)

    # Stage 6
    a61, a62, a63, a64, a65 = DORMAND_PRINCE_A[5]
    y6 = tuple((y_arr + h * (a61 * k1_arr + a62 * k2_arr + a63 * k3_arr + a64 * k4_arr + a65 * k5_arr)).tolist())
    k6 = deriv_fn(t + DORMAND_PRINCE_C[5] * h, y6)
    k6_arr = np.array(k6)

    # 4th-order solution (b4 weights; b4[1]=0 and b4[6]=0 so skip those terms)
    b4_0, _, b4_2, b4_3, b4_4, b4_5, _ = DORMAND_PRINCE_B4
    y_new_arr = y_arr + h * (
        b4_0 * k1_arr
        + b4_2 * k3_arr
        + b4_3 * k4_arr
        + b4_4 * k5_arr
        + b4_5 * k6_arr
    )
    y_new = tuple(float(x) for x in y_new_arr)

    # Stage 7 (FSAL: evaluate derivative at the 4th-order solution)
    k7 = deriv_fn(t + h, y_new)

    return ((k1, k2, k3, k4, k5, k6, k7), y_new, k7)


# --- Public single-step API ---

def dormand_prince_step(
    t: float,
    y: tuple[float, ...],
    h: float,
    deriv_fn: Callable[[float, tuple[float, ...]], tuple[float, ...]],
    k1_in: tuple[float, ...] | None = None,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Single Dormand-Prince RK4(5) step with FSAL.

    Args:
        t: Current time (seconds).
        y: Current state vector.
        h: Step size (seconds).
        deriv_fn: Derivative function f(t, y) -> dy/dt.
        k1_in: Reused first stage from previous step (FSAL). If None, computed.

    Returns:
        (t_new, y4_new, k7) where y4_new is the 4th-order solution and
        k7 is the 7th stage evaluation (reusable as k1 of next step via FSAL).
    """
    _, y_new, k7 = _dp_full_step(t, y, h, deriv_fn, k1_in)
    return (t + h, y_new, k7)


# --- Error estimation ---

def _error_norm(
    y: tuple[float, ...],
    y_new: tuple[float, ...],
    k_stages: tuple[tuple[float, ...], ...],
    h: float,
    atol: float,
    rtol: float,
) -> float:
    """Compute weighted RMS error norm for step-size control.

    err = sqrt(1/n * sum_j ((e_j / sc_j)^2))
    where e_j = h * sum_i(e_i * k_i_j) and sc_j = atol + rtol * max(|y_j|, |y_new_j|).
    """
    n = len(y)
    e_weights = np.array(_DORMAND_PRINCE_E)
    k_arr = np.array(k_stages)  # shape (7, n)
    y_arr = np.array(y)
    y_new_arr = np.array(y_new)

    # e_j = h * sum_i(e_i * k_i_j) for each j
    e_vec = h * (e_weights @ k_arr)  # shape (n,)
    sc_vec = atol + rtol * np.maximum(np.abs(y_arr), np.abs(y_new_arr))
    sum_sq = float(np.sum((e_vec / sc_vec) ** 2))

    return float(np.sqrt(sum_sq / n))


# --- Step-size control ---

def _new_step_size(
    h_try: float, err: float, safety: float, h_min: float, h_max: float,
) -> float:
    """Compute new step size from error estimate using PI controller."""
    if err < 1e-30:
        return h_max
    h_new = h_try * min(5.0, max(0.2, safety * err ** (-0.2)))
    return max(h_min, min(h_new, h_max))


# --- Dense output interpolation ---

def _hermite_interpolate(
    t0: float,
    y0: tuple[float, ...],
    f0: tuple[float, ...],
    t1: float,
    y1: tuple[float, ...],
    f1: tuple[float, ...],
    t_eval: float,
) -> tuple[float, ...]:
    """Cubic Hermite interpolation between two integration points."""
    h = t1 - t0
    if abs(h) < 1e-30:
        return y0
    theta = (t_eval - t0) / h
    theta2 = theta * theta
    theta3 = theta2 * theta

    h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0
    h10 = theta3 - 2.0 * theta2 + theta
    h01 = -2.0 * theta3 + 3.0 * theta2
    h11 = theta3 - theta2

    y0_arr = np.array(y0)
    f0_arr = np.array(f0)
    y1_arr = np.array(y1)
    f1_arr = np.array(f1)

    result = h00 * y0_arr + h10 * h * f0_arr + h01 * y1_arr + h11 * h * f1_arr
    return tuple(float(x) for x in result)


def _scale_deriv(k: tuple[float, ...], sign: float) -> tuple[float, ...]:
    """Scale derivative for interpolation in physical time direction."""
    result = sign * np.array(k)
    return tuple(float(x) for x in result)


# --- Main adaptive propagation function ---

def propagate_adaptive(
    initial_state: "object",
    duration: timedelta,
    force_models: list[ForceModel],
    epoch: datetime | None = None,
    config: AdaptiveStepConfig | None = None,
    output_step_s: float | None = None,
) -> AdaptiveStepResult:
    """Adaptive Dormand-Prince RK4(5) numerical propagation.

    Args:
        initial_state: OrbitalState-like object with Keplerian elements.
        duration: Total propagation duration (negative for backward).
        force_models: List of ForceModel instances to sum.
        epoch: Override epoch (defaults to initial_state.reference_epoch).
        config: Adaptive step-size configuration.
        output_step_s: If set, produces evenly-spaced output via dense output.
            If None, outputs at each integration step.

    Returns:
        AdaptiveStepResult with propagation trajectory and metadata.
    """
    if config is None:
        config = AdaptiveStepConfig()

    ref_epoch = epoch if epoch is not None else initial_state.reference_epoch  # type: ignore[attr-defined]
    duration_s = duration.total_seconds()

    # Convert to Cartesian ECI
    pos_list, vel_list = kepler_to_cartesian(
        a=initial_state.semi_major_axis_m,  # type: ignore[attr-defined]
        e=initial_state.eccentricity,  # type: ignore[attr-defined]
        i_rad=initial_state.inclination_rad,  # type: ignore[attr-defined]
        omega_big_rad=initial_state.raan_rad,  # type: ignore[attr-defined]
        omega_small_rad=initial_state.arg_perigee_rad,  # type: ignore[attr-defined]
        nu_rad=initial_state.true_anomaly_rad,  # type: ignore[attr-defined]
    )
    state: tuple[float, ...] = (
        pos_list[0], pos_list[1], pos_list[2],
        vel_list[0], vel_list[1], vel_list[2],
    )

    model_names = tuple(type(fm).__name__ for fm in force_models)

    # Build derivative function
    def deriv_fn(t_s: float, sv: tuple[float, ...]) -> tuple[float, ...]:
        current_epoch = ref_epoch + timedelta(seconds=t_s)
        p = (sv[0], sv[1], sv[2])
        v = (sv[3], sv[4], sv[5])
        ax_total, ay_total, az_total = 0.0, 0.0, 0.0
        for fm in force_models:
            ax, ay, az = fm.acceleration(current_epoch, p, v)
            ax_total += ax
            ay_total += ay
            az_total += az
        return (v[0], v[1], v[2], ax_total, ay_total, az_total)

    # Direction: forward or backward
    sign = 1.0 if duration_s >= 0.0 else -1.0
    t_end = abs(duration_s)

    # Initialize step size
    t = 0.0
    h = min(config.h_init, config.h_max)
    h = max(h, config.h_min)

    total_steps = 0
    rejected_steps = 0
    safety = config.safety_factor

    # FSAL: compute first derivative
    k1 = deriv_fn(sign * t, state)

    steps_list: list[PropagationStep] = []

    if output_step_s is not None:
        # Dense output mode: evenly-spaced output via Hermite interpolation
        output_dt = abs(output_step_s)
        output_times: list[float] = []
        t_out_acc = 0.0
        while t_out_acc <= t_end + 1e-12:
            output_times.append(t_out_acc)
            t_out_acc += output_dt
        if abs(output_times[-1] - t_end) > 1e-6:
            output_times.append(t_end)
        out_idx = 0

        prev_t = t
        prev_state = state
        prev_k = k1

        while t < t_end - 1e-12 and total_steps < config.max_steps:
            h_try = min(h, t_end - t)
            h_try = max(h_try, config.h_min)

            k_stages, y_new, k7 = _dp_full_step(
                sign * t, state, sign * h_try, deriv_fn, k1,
            )
            err = _error_norm(state, y_new, k_stages, sign * h_try, config.atol, config.rtol)

            if err <= 1.0:
                prev_t = t
                prev_state = state
                prev_k = k1

                t += h_try
                state = y_new
                k1 = k7
                total_steps += 1

                while out_idx < len(output_times) and output_times[out_idx] <= t + 1e-12:
                    t_out_val = output_times[out_idx]
                    if abs(t_out_val - t) < 1e-12:
                        interp_state = state
                    elif abs(t_out_val - prev_t) < 1e-12:
                        interp_state = prev_state
                    else:
                        interp_state = _hermite_interpolate(
                            prev_t, prev_state, _scale_deriv(prev_k, sign),
                            t, state, _scale_deriv(k1, sign),
                            t_out_val,
                        )
                    step_time = ref_epoch + timedelta(seconds=sign * t_out_val)
                    steps_list.append(PropagationStep(
                        time=step_time,
                        position_eci=(interp_state[0], interp_state[1], interp_state[2]),
                        velocity_eci=(interp_state[3], interp_state[4], interp_state[5]),
                    ))
                    out_idx += 1

                h = _new_step_size(h_try, err, safety, config.h_min, config.h_max)
            else:
                rejected_steps += 1
                h = _new_step_size(h_try, err, safety, config.h_min, config.h_max)
    else:
        # Natural output mode: output at each accepted integration step
        step_time = ref_epoch + timedelta(seconds=sign * t)
        steps_list.append(PropagationStep(
            time=step_time,
            position_eci=(state[0], state[1], state[2]),
            velocity_eci=(state[3], state[4], state[5]),
        ))

        while t < t_end - 1e-12 and total_steps < config.max_steps:
            h_try = min(h, t_end - t)
            h_try = max(h_try, config.h_min)

            k_stages, y_new, k7 = _dp_full_step(
                sign * t, state, sign * h_try, deriv_fn, k1,
            )
            err = _error_norm(state, y_new, k_stages, sign * h_try, config.atol, config.rtol)

            if err <= 1.0:
                t += h_try
                state = y_new
                k1 = k7
                total_steps += 1

                step_time = ref_epoch + timedelta(seconds=sign * t)
                steps_list.append(PropagationStep(
                    time=step_time,
                    position_eci=(state[0], state[1], state[2]),
                    velocity_eci=(state[3], state[4], state[5]),
                ))

                h = _new_step_size(h_try, err, safety, config.h_min, config.h_max)
            else:
                rejected_steps += 1
                h = _new_step_size(h_try, err, safety, config.h_min, config.h_max)

    return AdaptiveStepResult(
        steps=tuple(steps_list),
        epoch=ref_epoch,
        duration_s=duration_s,
        force_model_names=model_names,
        total_steps=total_steps,
        rejected_steps=rejected_steps,
    )
