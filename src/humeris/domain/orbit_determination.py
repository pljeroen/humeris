# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Extended Kalman Filter (EKF) orbit determination.

Processes position observations to estimate spacecraft state (position
and velocity) using two-body dynamics for prediction and a linear
measurement model for updates.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


_MU = 3.986004418e14  # m³/s² — Earth gravitational parameter


@dataclass(frozen=True)
class ODObservation:
    """A single position observation."""
    time: datetime
    position_m: tuple[float, float, float]
    noise_std_m: float  # 1-sigma measurement noise


@dataclass(frozen=True)
class ODEstimate:
    """State estimate at a single time."""
    time: datetime
    state: tuple[float, float, float, float, float, float]  # x,y,z,vx,vy,vz
    covariance: tuple[tuple[float, ...], ...]  # 6x6
    residual_m: float  # post-fit residual
    innovation_variance_m2: float | None = None  # EKF predicted innovation S


@dataclass(frozen=True)
class ODResult:
    """Complete orbit determination result."""
    estimates: tuple[ODEstimate, ...]
    final_state: tuple[float, float, float, float, float, float]
    final_covariance: tuple[tuple[float, ...], ...]
    rms_residual_m: float
    observations_processed: int


def _mat_zeros(n: int, m: int) -> list[list[float]]:
    return [[0.0] * m for _ in range(n)]


def _mat_identity(n: int) -> list[list[float]]:
    result = _mat_zeros(n, n)
    for i in range(n):
        result[i][i] = 1.0
    return result


def _mat_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = len(b[0])
    k = len(b)
    result = _mat_zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += a[i][p] * b[p][j]
            result[i][j] = s
    return result


def _mat_transpose(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = len(a[0])
    return [[a[i][j] for i in range(n)] for j in range(m)]


def _mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = len(a[0])
    return [[a[i][j] + b[i][j] for j in range(m)] for i in range(n)]


def _mat_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = len(a[0])
    return [[a[i][j] - b[i][j] for j in range(m)] for i in range(n)]


def _mat_scale(a: list[list[float]], s: float) -> list[list[float]]:
    return [[a[i][j] * s for j in range(len(a[0]))] for i in range(len(a))]


def _mat_inverse_3x3(m: list[list[float]]) -> list[list[float]]:
    """Inverse of a 3x3 matrix via cofactors."""
    a, b, c = m[0][0], m[0][1], m[0][2]
    d, e, f = m[1][0], m[1][1], m[1][2]
    g, h, i = m[2][0], m[2][1], m[2][2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-30:
        raise ValueError("Singular 3x3 matrix — filter may have diverged")

    inv_det = 1.0 / det
    return [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


def _two_body_propagate(
    state: list[float],
    dt: float,
) -> list[float]:
    """Simple two-body propagation using RK4.

    State = [x, y, z, vx, vy, vz].
    """
    def deriv(s: list[float]) -> list[float]:
        r = float(np.linalg.norm(s[:3]))
        r3 = r**3
        coeff = -_MU / r3
        return [s[3], s[4], s[5], coeff * s[0], coeff * s[1], coeff * s[2]]

    # RK4
    k1 = deriv(state)
    s1 = [state[i] + 0.5 * dt * k1[i] for i in range(6)]
    k2 = deriv(s1)
    s2 = [state[i] + 0.5 * dt * k2[i] for i in range(6)]
    k3 = deriv(s2)
    s3 = [state[i] + dt * k3[i] for i in range(6)]
    k4 = deriv(s3)

    return [
        state[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(6)
    ]


def _compute_stm(
    state: list[float],
    dt: float,
) -> list[list[float]]:
    """Compute 6x6 state transition matrix via finite differences.

    Perturbs each state element and computes the Jacobian numerically.
    """
    n = 6
    stm = _mat_zeros(n, n)
    eps_pos = 1.0  # 1 m perturbation for position
    eps_vel = 0.001  # 1 mm/s perturbation for velocity

    for j in range(n):
        eps = eps_pos if j < 3 else eps_vel

        state_plus = state[:]
        state_plus[j] += eps
        result_plus = _two_body_propagate(state_plus, dt)

        state_minus = state[:]
        state_minus[j] -= eps
        result_minus = _two_body_propagate(state_minus, dt)

        for i in range(n):
            stm[i][j] = (result_plus[i] - result_minus[i]) / (2.0 * eps)

    return stm


def _kalman_update(
    x_pred: list[float],
    p_pred: list[list[float]],
    z: tuple[float, float, float],
    noise_std: float,
) -> tuple[list[float], list[list[float]], float]:
    """Single EKF measurement update (position-only).

    H = [I_3x3 | 0_3x3] — position-only observation.

    Returns:
        (state_updated, covariance_updated, pre_fit_residual_magnitude)
    """
    # Measurement matrix H (3x6): [I_3 | 0_3]
    # Innovation: y = z - H * x_pred = z - x_pred[:3]
    y = [z[0] - x_pred[0], z[1] - x_pred[1], z[2] - x_pred[2]]

    # S = H * P_pred * H^T + R (3x3)
    # H * P_pred → just rows 0:3 of P_pred columns 0:3
    hp = [[p_pred[i][j] for j in range(6)] for i in range(3)]
    # H * P_pred * H^T → top-left 3x3 of P_pred
    s = [[p_pred[i][j] for j in range(3)] for i in range(3)]
    # Add R = noise_std^2 * I
    r_sq = noise_std * noise_std
    for i in range(3):
        s[i][i] += r_sq

    # K = P_pred * H^T * S^-1 (6x3)
    s_inv = _mat_inverse_3x3(s)
    # P_pred * H^T = columns 0:3 of P_pred
    pht = [[p_pred[i][j] for j in range(3)] for i in range(6)]
    k = _mat_multiply(pht, s_inv)

    # State update: x = x_pred + K * y
    x_new = x_pred[:]
    for i in range(6):
        for j in range(3):
            x_new[i] += k[i][j] * y[j]

    # Joseph stabilized covariance update:
    # P = (I - KH) * P_pred * (I - KH)^T + K * R * K^T
    # Guarantees symmetry and positive semi-definiteness.
    kh = _mat_zeros(6, 6)
    for i in range(6):
        for j in range(3):
            kh[i][j] = k[i][j]  # H has identity in first 3 cols

    i_kh = _mat_sub(_mat_identity(6), kh)
    i_kh_t = _mat_transpose(i_kh)
    p_joseph = _mat_multiply(_mat_multiply(i_kh, p_pred), i_kh_t)

    # K * R * K^T term (R = noise_std^2 * I_3)
    k_t = _mat_transpose(k)
    kk_t = _mat_multiply(k, k_t)
    p_new = _mat_add(p_joseph, _mat_scale(kk_t, r_sq))

    # Post-fit residual
    residual = float(np.linalg.norm(y))

    return x_new, p_new, residual


def run_ekf(
    observations: list[ODObservation],
    initial_state: tuple[float, float, float, float, float, float],
    initial_covariance: list[list[float]],
    process_noise: list[list[float]],
) -> ODResult:
    """Run Extended Kalman Filter for orbit determination.

    Args:
        observations: List of position observations, time-ordered.
        initial_state: Initial state estimate (x,y,z,vx,vy,vz).
        initial_covariance: 6x6 initial covariance matrix.
        process_noise: 6x6 process noise matrix (Q).

    Returns:
        ODResult with state estimates and statistics.

    Raises:
        ValueError: If observations list is empty.
    """
    if not observations:
        raise ValueError("observations list must not be empty")

    state = list(initial_state)
    cov = [row[:] for row in initial_covariance]

    estimates: list[ODEstimate] = []
    residuals_sq_sum = 0.0
    current_time = observations[0].time

    for obs in observations:
        # Predict to observation time
        dt = (obs.time - current_time).total_seconds()
        if abs(dt) > 0.001:
            # Propagate state
            stm = _compute_stm(state, dt)
            state = _two_body_propagate(state, dt)

            # Propagate covariance: P = Phi * P * Phi^T + Q
            phi_p = _mat_multiply(stm, cov)
            stm_t = _mat_transpose(stm)
            cov = _mat_add(_mat_multiply(phi_p, stm_t), process_noise)

        current_time = obs.time

        # Update
        state, cov, residual = _kalman_update(
            state, cov, obs.position_m, obs.noise_std_m,
        )

        residuals_sq_sum += residual ** 2

        cov_tuple = tuple(tuple(row) for row in cov)
        estimates.append(ODEstimate(
            time=obs.time,
            state=tuple(state),
            covariance=cov_tuple,
            residual_m=residual,
        ))

    rms_residual = float(np.sqrt(residuals_sq_sum / len(observations)))
    final_cov = tuple(tuple(row) for row in cov)

    return ODResult(
        estimates=tuple(estimates),
        final_state=tuple(state),
        final_covariance=final_cov,
        rms_residual_m=rms_residual,
        observations_processed=len(observations),
    )
