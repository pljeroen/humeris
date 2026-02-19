# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Extended Kalman Filter (EKF) orbit determination.

Processes position observations to estimate spacecraft state (position
and velocity) using two-body dynamics for prediction and a linear
measurement model for updates.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants

_MU = OrbitalConstants.MU_EARTH


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
        if r < 1e-3:
            return [s[3], s[4], s[5], 0.0, 0.0, 0.0]
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

    Process noise Q should be tuned for the expected observation interval.
    For irregular spacing, scale Q by dt/dt_reference where dt_reference
    is the nominal observation interval used when tuning Q.

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

            # Enforce symmetry: P[i][j] = P[j][i] = (P[i][j] + P[j][i]) / 2
            for _i in range(6):
                for _j in range(_i + 1, 6):
                    avg = (cov[_i][_j] + cov[_j][_i]) / 2.0
                    cov[_i][_j] = avg
                    cov[_j][_i] = avg

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


# ── P22: Bayesian Model Selection for Force Models ─────────────────


@dataclass(frozen=True)
class ForceModelCandidate:
    """A force model configuration to evaluate.

    Attributes:
        name: Descriptive name (e.g., "two-body", "J2", "J2+drag").
        n_params: Number of force parameters enabled (p_k).
        rss: Residual sum of squares from OD under this model.
    """
    name: str
    n_params: int
    rss: float


@dataclass(frozen=True)
class ForceModelRanking:
    """Result of Bayesian model selection for force models.

    Attributes:
        rankings: List of (name, BIC) tuples sorted by BIC ascending (best first).
        best_model_name: Name of the model with lowest BIC.
        bayes_factor_best_vs_full: Bayes factor B_{best,full} comparing
            the best model against the most complex (highest p_k) model.
        computation_savings_percent: Percentage of parameters saved by
            using the best model vs. the full model.
    """
    rankings: tuple[tuple[str, float], ...]
    best_model_name: str
    bayes_factor_best_vs_full: float
    computation_savings_percent: float


def select_force_model(
    candidates: list[ForceModelCandidate],
    n_observations: int,
) -> ForceModelRanking:
    """Bayesian model selection for force model configurations via BIC.

    Computes the Bayesian Information Criterion (BIC) for each force
    model configuration and ranks them. Lower BIC is better, balancing
    fit quality against model complexity.

    BIC_k = n * ln(RSS_k / n) + p_k * ln(n)

    The Bayes factor between models k and j:
        B_kj = exp(-0.5 * (BIC_k - BIC_j))

    Args:
        candidates: List of ForceModelCandidate with RSS from OD runs.
        n_observations: Number of observations used in each OD run.

    Returns:
        ForceModelRanking with sorted rankings and Bayes factors.

    Raises:
        ValueError: If candidates is empty or n_observations < 2.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    if n_observations < 2:
        raise ValueError(f"n_observations must be >= 2, got {n_observations}")

    n = n_observations
    ln_n = math.log(n)

    # Compute BIC for each candidate
    bic_list: list[tuple[str, float]] = []
    for c in candidates:
        if c.rss <= 0:
            # Perfect fit: use a very small RSS to avoid log(0)
            rss_safe = 1e-30
        else:
            rss_safe = c.rss
        bic = n * math.log(rss_safe / n) + c.n_params * ln_n
        bic_list.append((c.name, bic))

    # Sort by BIC ascending (best first)
    bic_list.sort(key=lambda x: x[1])

    best_name = bic_list[0][0]
    best_bic = bic_list[0][1]

    # Find the "full" model (most parameters)
    full_model = max(candidates, key=lambda c: c.n_params)
    full_bic = next(bic for name, bic in bic_list if name == full_model.name)

    # Bayes factor: best vs full
    delta_bic = best_bic - full_bic
    bayes_factor = math.exp(-0.5 * delta_bic)

    # Computation savings: parameter reduction
    best_params = next(c.n_params for c in candidates if c.name == best_name)
    full_params = full_model.n_params
    if full_params > 0:
        savings = (1.0 - best_params / full_params) * 100.0
    else:
        savings = 0.0

    return ForceModelRanking(
        rankings=tuple(bic_list),
        best_model_name=best_name,
        bayes_factor_best_vs_full=bayes_factor,
        computation_savings_percent=savings,
    )


# ── P23: Particle Filter for Non-Gaussian OD ──────────────────────


@dataclass(frozen=True)
class ParticleFilterResult:
    """Result of particle filter orbit determination.

    Attributes:
        estimates: Tuple of ODEstimate at each observation time.
        final_state: Weighted mean state after all observations.
        final_covariance: Weighted covariance after all observations (6x6).
        rms_residual: RMS of post-fit residuals (m).
        mean_effective_particles: Mean N_eff over all update steps.
        resampling_count: Number of times systematic resampling was triggered.
    """
    estimates: tuple[ODEstimate, ...]
    final_state: tuple[float, float, float, float, float, float]
    final_covariance: tuple[tuple[float, ...], ...]
    rms_residual: float
    mean_effective_particles: float
    resampling_count: int


def _rk4_two_body(state: np.ndarray, dt: float) -> np.ndarray:
    """RK4 integration step for two-body dynamics.

    Fourth-order Runge-Kutta for d/dt [pos, vel] = [vel, -mu/r^3 * pos].

    Args:
        state: Array of shape (6,) — [x, y, z, vx, vy, vz].
        dt: Time step in seconds.

    Returns:
        Propagated state array of shape (6,).
    """
    def _deriv(s: np.ndarray) -> np.ndarray:
        r = float(np.linalg.norm(s[:3]))
        if r < 1e-3:
            return np.array([s[3], s[4], s[5], 0.0, 0.0, 0.0])
        coeff = -_MU / (r ** 3)
        return np.array([s[3], s[4], s[5], coeff * s[0], coeff * s[1], coeff * s[2]])

    k1 = _deriv(state)
    k2 = _deriv(state + 0.5 * dt * k1)
    k3 = _deriv(state + 0.5 * dt * k2)
    k4 = _deriv(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _systematic_resample(weights: np.ndarray, n_particles: int) -> np.ndarray:
    """Systematic resampling for particle filter.

    Generates evenly-spaced cumulative threshold points with a single
    random offset, then selects particles by cumulative weight.

    Args:
        weights: Normalized weights of shape (n_particles,).
        n_particles: Number of particles to resample.

    Returns:
        Array of resampled indices.
    """
    cumulative = np.cumsum(weights)
    # Use deterministic spacing with a pseudo-random offset based on weights
    # For reproducibility in a domain module, use a simple hash-based offset
    offset = (weights[0] * 1e6) % 1.0
    positions = (offset + np.arange(n_particles)) / n_particles
    indices = np.searchsorted(cumulative, positions)
    indices = np.clip(indices, 0, n_particles - 1)
    return indices


def run_particle_filter(
    observations: list[ODObservation],
    initial_state: tuple[float, float, float, float, float, float],
    initial_covariance: list[list[float]],
    process_noise_std: float = 10.0,
    n_particles: int = 500,
) -> ParticleFilterResult:
    """Bootstrap particle filter for non-Gaussian orbit determination.

    Represents the state distribution as a set of weighted particles.
    Prediction propagates each particle through two-body dynamics with
    additive Gaussian noise. Update reweights by observation likelihood.
    Systematic resampling when N_eff < N/2.

    Args:
        observations: List of position observations, time-ordered.
        initial_state: Initial state estimate (x,y,z,vx,vy,vz).
        initial_covariance: 6x6 initial covariance matrix.
        process_noise_std: Standard deviation of process noise (m for
            position, m/s for velocity). Applied per Euler step.
        n_particles: Number of particles (default 500).

    Returns:
        ParticleFilterResult with estimates and diagnostics.

    Raises:
        ValueError: If observations is empty or n_particles < 2.
    """
    if not observations:
        raise ValueError("observations list must not be empty")
    if n_particles < 2:
        raise ValueError(f"n_particles must be >= 2, got {n_particles}")

    N = n_particles
    dim = 6

    # Initialize particles from prior distribution
    cov_matrix = np.array(initial_covariance, dtype=np.float64)
    mean_state = np.array(initial_state, dtype=np.float64)

    # Use Cholesky decomposition for sampling
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If covariance is not positive definite, use diagonal
        L = np.diag(np.sqrt(np.maximum(np.diag(cov_matrix), 1e-10)))

    # Generate particles: x_i = mean + L @ z_i, z_i ~ N(0, I)
    # Use a deterministic seed based on initial state for reproducibility
    rng_seed = int(abs(initial_state[0]) * 1000) % (2**31)
    rng = np.random.RandomState(rng_seed)
    z_samples = rng.randn(N, dim)
    particles = mean_state[np.newaxis, :] + z_samples @ L.T

    # Uniform initial weights
    weights = np.ones(N) / N

    estimates: list[ODEstimate] = []
    residuals_sq_sum = 0.0
    current_time = observations[0].time
    n_eff_sum = 0.0
    resampling_count = 0
    n_updates = 0

    for obs in observations:
        dt = (obs.time - current_time).total_seconds()

        # Prediction: propagate each particle
        if abs(dt) > 0.001:
            noise_pos = rng.randn(N, 3) * process_noise_std
            noise_vel = rng.randn(N, 3) * process_noise_std * 0.01
            noise = np.hstack([noise_pos, noise_vel])

            for i in range(N):
                particles[i] = _rk4_two_body(particles[i], dt)
            particles += noise

        current_time = obs.time

        # Update: compute observation likelihood for each particle
        z = np.array(obs.position_m)
        R_inv = 1.0 / (obs.noise_std_m ** 2)

        # p(z|x_i) = exp(-0.5 * (z - H*x_i)^T * R^-1 * (z - H*x_i))
        # H*x_i = x_i[:3] (position only)
        residuals = z[np.newaxis, :] - particles[:, :3]
        mahal_sq = np.sum(residuals ** 2, axis=1) * R_inv
        log_weights = -0.5 * mahal_sq

        # Normalize weights in log space for numerical stability
        max_log_w = np.max(log_weights)
        log_weights -= max_log_w
        raw_weights = weights * np.exp(log_weights)
        w_sum = np.sum(raw_weights)
        if w_sum > 1e-300:
            weights = raw_weights / w_sum
        else:
            weights = np.ones(N) / N

        # Effective sample size
        n_eff = 1.0 / np.sum(weights ** 2)
        n_eff_sum += n_eff
        n_updates += 1

        # Systematic resampling if N_eff < N/2
        if n_eff < N / 2.0:
            indices = _systematic_resample(weights, N)
            particles = particles[indices].copy()
            weights = np.ones(N) / N
            resampling_count += 1

        # Weighted state estimate
        w_state = np.sum(weights[:, np.newaxis] * particles, axis=0)

        # Weighted covariance
        diff = particles - w_state[np.newaxis, :]
        w_cov = np.zeros((dim, dim))
        for i in range(N):
            w_cov += weights[i] * np.outer(diff[i], diff[i])

        # Post-fit residual
        pred_pos = w_state[:3]
        residual_vec = z - pred_pos
        residual_mag = float(np.linalg.norm(residual_vec))
        residuals_sq_sum += residual_mag ** 2

        cov_tuple = tuple(tuple(float(v) for v in row) for row in w_cov)
        estimates.append(ODEstimate(
            time=obs.time,
            state=tuple(float(v) for v in w_state),
            covariance=cov_tuple,
            residual_m=residual_mag,
        ))

    # Final weighted state and covariance
    final_state = np.sum(weights[:, np.newaxis] * particles, axis=0)
    diff = particles - final_state[np.newaxis, :]
    final_cov = np.zeros((dim, dim))
    for i in range(N):
        final_cov += weights[i] * np.outer(diff[i], diff[i])

    rms_residual = math.sqrt(residuals_sq_sum / len(observations))
    mean_n_eff = n_eff_sum / max(n_updates, 1)

    return ParticleFilterResult(
        estimates=tuple(estimates),
        final_state=tuple(float(v) for v in final_state),
        final_covariance=tuple(tuple(float(v) for v in row) for row in final_cov),
        rms_residual=rms_residual,
        mean_effective_particles=mean_n_eff,
        resampling_count=resampling_count,
    )


# ── P29: Compressed Sensing for Sparse OD ──────────────────────────


@dataclass(frozen=True)
class CompressedSensingOD:
    """Result of compressed sensing (ISTA) orbit determination.

    Attributes:
        state_correction: The sparse state correction vector (6 elements).
        sparsity_level: Number of components with |x_i| > epsilon.
        rms_residual: RMS of residuals after correction (m).
        compression_ratio: Ratio of non-zero components to total.
        ista_iterations: Number of ISTA iterations to converge.
    """
    state_correction: tuple[float, float, float, float, float, float]
    sparsity_level: int
    rms_residual: float
    compression_ratio: float
    ista_iterations: int


def _soft_threshold(u: np.ndarray, lam: float) -> np.ndarray:
    """Soft thresholding operator S_lambda(u) = sign(u) * max(|u| - lambda, 0)."""
    return np.sign(u) * np.maximum(np.abs(u) - lam, 0.0)


def compressed_sensing_od(
    predicted_state: tuple[float, float, float, float, float, float],
    observations: list[ODObservation],
    lambda_reg: float = 1.0,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    sparsity_epsilon: float = 1e-6,
) -> CompressedSensingOD:
    """L1-regularized orbit determination via ISTA.

    Computes a sparse state correction to the predicted state using
    observations and the Iterative Shrinkage-Thresholding Algorithm.

    The measurement model is linear: y = H * x + noise
    where H = [I_3 | 0_3] for position-only observations, and
    x is the 6D state correction.

    ISTA iteration:
        x^{k+1} = S_lambda(x^k + H^T(y - H*x^k))

    Args:
        predicted_state: Predicted state (x,y,z,vx,vy,vz) in meters and m/s.
        observations: List of position observations.
        lambda_reg: L1 regularization parameter. Larger values produce
            sparser solutions.
        max_iterations: Maximum ISTA iterations.
        tolerance: Convergence tolerance on ||x^{k+1} - x^k||.
        sparsity_epsilon: Threshold for counting non-zero components.

    Returns:
        CompressedSensingOD with sparse correction and diagnostics.

    Raises:
        ValueError: If observations is empty or lambda_reg < 0.
    """
    if not observations:
        raise ValueError("observations list must not be empty")
    if lambda_reg < 0:
        raise ValueError(f"lambda_reg must be >= 0, got {lambda_reg}")

    n_obs = len(observations)
    dim = 6

    # Build measurement matrix H and observation vector y
    # Each observation gives 3 rows in H and 3 elements in y
    H = np.zeros((3 * n_obs, dim))
    y = np.zeros(3 * n_obs)
    pred = np.array(predicted_state)

    for i, obs in enumerate(observations):
        row = 3 * i
        # H = [I_3 | 0_3] for each observation
        H[row, 0] = 1.0
        H[row + 1, 1] = 1.0
        H[row + 2, 2] = 1.0
        # y = observation - predicted position
        y[row] = obs.position_m[0] - pred[0]
        y[row + 1] = obs.position_m[1] - pred[1]
        y[row + 2] = obs.position_m[2] - pred[2]

    # Step size: 1 / (largest eigenvalue of H^T H)
    HtH = H.T @ H
    eigenvalues = np.linalg.eigvalsh(HtH)
    L_lipschitz = float(np.max(eigenvalues))
    if L_lipschitz < 1e-30:
        L_lipschitz = 1.0
    step_size = 1.0 / L_lipschitz

    # ISTA iterations
    x = np.zeros(dim)
    HtY = H.T @ y  # precompute

    iterations = 0
    for k in range(max_iterations):
        # Gradient step: x_temp = x + step_size * H^T(y - Hx)
        residual = y - H @ x
        gradient = HtY - HtH @ x
        x_temp = x + step_size * gradient

        # Soft thresholding
        x_new = _soft_threshold(x_temp, lambda_reg * step_size)

        # Check convergence
        diff_norm = float(np.linalg.norm(x_new - x))
        x = x_new
        iterations = k + 1

        if diff_norm < tolerance:
            break

    # Compute final residual
    final_residual = y - H @ x
    rms_residual = float(np.sqrt(np.mean(final_residual ** 2)))

    # Sparsity analysis
    sparsity_level = int(np.sum(np.abs(x) > sparsity_epsilon))
    compression_ratio = sparsity_level / dim if dim > 0 else 0.0

    return CompressedSensingOD(
        state_correction=tuple(float(v) for v in x),
        sparsity_level=sparsity_level,
        rms_residual=rms_residual,
        compression_ratio=compression_ratio,
        ista_iterations=iterations,
    )


# ── P59: Optimal OD Scheduling via Fisher Information ──────────────
#
# Greedy D-optimal observation selection: at each step, add the
# observation that maximises det(FIM). Uses the matrix determinant
# lemma for efficient incremental determinant computation.


@dataclass(frozen=True)
class OptimalObservationSchedule:
    """Optimal observation schedule via Fisher information maximisation.

    Attributes:
        selected_indices: Indices of selected observations (in greedy order).
        fisher_information_det: Determinant of FIM for selected observations.
        baseline_det: Determinant of FIM for uniformly spaced baseline.
        information_gain_ratio: Ratio of optimal to baseline FIM determinant.
        condition_number: Condition number of the optimal FIM.
    """
    selected_indices: tuple[int, ...]
    fisher_information_det: float
    baseline_det: float
    information_gain_ratio: float
    condition_number: float


def _compute_observation_jacobian(
    state: list[float],
    dt: float,
    noise_std: float,
) -> np.ndarray:
    """Compute observation Jacobian H_k for a candidate observation.

    For position-only observations at time t_k, the Jacobian is:
        H_k = H_obs @ STM(t_k)
    where H_obs = [I_3 | 0_3] (position extraction) and STM is the
    state transition matrix from epoch to t_k.

    The contribution to the Fisher information is:
        F_k = H_k^T @ R^{-1} @ H_k

    Args:
        state: Reference state [x, y, z, vx, vy, vz].
        dt: Time offset from epoch (seconds).
        noise_std: Measurement noise standard deviation (m).

    Returns:
        6x6 numpy array: Fisher information contribution H_k^T R^{-1} H_k.
    """
    if abs(dt) < 0.001:
        stm = np.eye(6)
    else:
        stm_list = _compute_stm(state, dt)
        stm = np.array(stm_list, dtype=np.float64)

    # H_obs = [I_3 | 0_3], so H_k = STM[:3, :] (first 3 rows)
    h_k = stm[:3, :]

    # Fisher contribution: H_k^T @ R^{-1} @ H_k
    r_inv = 1.0 / (noise_std ** 2)
    return r_inv * (h_k.T @ h_k)


def compute_optimal_observation_schedule(
    reference_state: tuple[float, float, float, float, float, float],
    candidate_times_s: list[float],
    noise_stds_m: list[float],
    n_select: int,
) -> OptimalObservationSchedule:
    """Compute D-optimal observation schedule via greedy Fisher information.

    At each step, selects the candidate observation that maximises
    det(FIM_current + F_candidate) using the matrix determinant lemma:
        det(A + uv^T) = det(A) * (1 + v^T A^{-1} u)

    For rank-3 updates (position observations contribute rank-3 to FIM),
    we use the full determinant computation for the first few observations
    and then the matrix determinant lemma for efficiency.

    Also computes a uniformly-spaced baseline for comparison.

    Args:
        reference_state: Reference state at epoch (x, y, z, vx, vy, vz) in m, m/s.
        candidate_times_s: Time offsets for candidate observations (seconds from epoch).
        noise_stds_m: Measurement noise std for each candidate (m).
        n_select: Number of observations to select.

    Returns:
        OptimalObservationSchedule with selected indices and information metrics.

    Raises:
        ValueError: If inputs are inconsistent or n_select > len(candidates).
    """
    n_candidates = len(candidate_times_s)
    if n_candidates == 0:
        raise ValueError("candidate_times_s must not be empty")
    if len(noise_stds_m) != n_candidates:
        raise ValueError(
            f"noise_stds_m length ({len(noise_stds_m)}) != "
            f"candidate_times_s length ({n_candidates})"
        )
    if n_select <= 0:
        raise ValueError(f"n_select must be positive, got {n_select}")
    if n_select > n_candidates:
        raise ValueError(
            f"n_select ({n_select}) > candidates ({n_candidates})"
        )

    state = list(reference_state)
    dim = 6

    # Precompute Fisher information contributions for all candidates
    fisher_contributions = []
    for i in range(n_candidates):
        f_k = _compute_observation_jacobian(state, candidate_times_s[i], noise_stds_m[i])
        fisher_contributions.append(f_k)

    # Greedy D-optimal selection
    selected = []
    available = set(range(n_candidates))
    fim = np.zeros((dim, dim))

    # Small regularization to make FIM invertible initially
    fim += np.eye(dim) * 1e-20

    for _ in range(n_select):
        best_idx = -1
        best_det = -1.0

        for idx in available:
            fim_test = fim + fisher_contributions[idx]
            try:
                det_val = float(np.linalg.det(fim_test))
            except np.linalg.LinAlgError:
                det_val = 0.0
            if det_val > best_det:
                best_det = det_val
                best_idx = idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        available.remove(best_idx)
        fim = fim + fisher_contributions[best_idx]

    # Final FIM determinant and condition number
    optimal_det = max(float(np.linalg.det(fim)), 0.0)

    try:
        eigenvalues = np.linalg.eigvalsh(fim)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        if eigenvalues[0] > 1e-30:
            condition_number = float(eigenvalues[-1] / eigenvalues[0])
        else:
            condition_number = float('inf')
    except np.linalg.LinAlgError:
        condition_number = float('inf')

    # Baseline: uniformly spaced selection
    if n_candidates >= n_select:
        step = n_candidates / n_select
        baseline_indices = [int(i * step) for i in range(n_select)]
    else:
        baseline_indices = list(range(n_candidates))

    fim_baseline = np.eye(dim) * 1e-20
    for idx in baseline_indices:
        fim_baseline += fisher_contributions[idx]
    baseline_det = max(float(np.linalg.det(fim_baseline)), 0.0)

    # Information gain ratio
    if baseline_det > 1e-300:
        info_gain = optimal_det / baseline_det
    else:
        info_gain = float('inf') if optimal_det > 0 else 1.0

    return OptimalObservationSchedule(
        selected_indices=tuple(selected),
        fisher_information_det=optimal_det,
        baseline_det=baseline_det,
        information_gain_ratio=info_gain,
        condition_number=condition_number,
    )
