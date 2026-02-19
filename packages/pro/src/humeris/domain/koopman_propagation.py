# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Koopman operator for fast long-term orbital propagation via DMD.

Fits a finite-dimensional Koopman operator approximation from propagation
snapshots using Dynamic Mode Decomposition (DMD). The lifted observable
space captures nonlinear dynamics in a linear operator, enabling fast
multi-step prediction without re-evaluating force models.

Uses numpy for matrix operations (SVD, pseudoinverse).
"""
import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KoopmanModel:
    """Fitted Koopman operator model for fast propagation."""
    koopman_matrix: tuple  # Flattened n_obs x n_obs matrix
    n_observables: int
    training_step_s: float
    mean_state: tuple      # Mean of training observables (for centering)
    singular_values: tuple  # SVD singular values (for diagnostics)
    training_error: float   # RMS training error (normalized)
    max_eigenvalue_magnitude: float = 0.0  # Max |lambda| of Koopman matrix (stability diagnostic)
    is_stable: bool = True  # True if max |lambda| <= 1.0 + epsilon


@dataclass(frozen=True)
class KoopmanPrediction:
    """Koopman propagation result."""
    times_s: tuple               # Time offsets from epoch
    positions_eci: tuple         # Tuple of (x,y,z) tuples
    velocities_eci: tuple        # Tuple of (vx,vy,vz) tuples
    model: KoopmanModel
    is_stable: bool = True       # Stability flag from model


def _build_observables(
    pos: tuple[float, float, float],
    vel: tuple[float, float, float],
    n_observables: int,
) -> np.ndarray:
    """Build lifted observable vector from state.

    Observable library:
    [x, y, z, vx, vy, vz, r, v, x^2, y^2, z^2, x*y]
    Takes first n_observables (min 6, max 12).
    """
    x, y, z = pos
    vx, vy, vz = vel

    full = [
        x, y, z, vx, vy, vz,
        math.sqrt(x * x + y * y + z * z),
        math.sqrt(vx * vx + vy * vy + vz * vz),
        x * x, y * y, z * z, x * y,
    ]
    return np.array(full[:n_observables], dtype=np.float64)


def fit_koopman_model(
    positions: list[tuple[float, float, float]],
    velocities: list[tuple[float, float, float]],
    step_s: float,
    n_observables: int = 12,
) -> KoopmanModel:
    """Fit a Koopman operator from propagation snapshots via DMD.

    Observables: [x, y, z, vx, vy, vz, r, v, x^2, y^2, z^2, x*y]
    (first n_observables of these).

    K_approx = G_future @ pinv(G_current) via SVD.

    Args:
        positions: List of (x, y, z) position tuples in meters.
        velocities: List of (vx, vy, vz) velocity tuples in m/s.
        step_s: Time step between consecutive snapshots in seconds.
        n_observables: Number of observables to use (6-12). Default 12.

    Returns:
        Fitted KoopmanModel.
    """
    n_obs = max(6, min(n_observables, 12))
    m = len(positions)

    # Build observable vectors for all snapshots
    g_all = np.zeros((n_obs, m), dtype=np.float64)
    for i in range(m):
        g_all[:, i] = _build_observables(positions[i], velocities[i], n_obs)

    # Compute mean for centering diagnostics (but DMD works on raw data)
    mean_state = np.mean(g_all, axis=1)

    # Build snapshot matrices
    # G_current = snapshots 0..m-2, G_future = snapshots 1..m-1
    g_current = g_all[:, :-1]  # (n_obs, m-1)
    g_future = g_all[:, 1:]    # (n_obs, m-1)

    # DMD via pseudoinverse: K = G_future @ pinv(G_current)
    # SVD of G_current for diagnostics and pseudoinverse
    u, s, vt = np.linalg.svd(g_current, full_matrices=False)
    singular_values = tuple(float(sv) for sv in s)

    # Truncated SVD: discard singular values below s_max * 1e-10
    # to improve numerical stability for rank-deficient cases (F-07)
    s_max = float(s[0]) if len(s) > 0 else 1.0
    threshold = s_max * 1e-10
    keep = s > threshold
    u_trunc = u[:, keep]
    s_trunc = s[keep]
    vt_trunc = vt[keep, :]

    # K = G_future @ V @ S^{-1} @ U^T (using truncated SVD)
    s_inv = np.diag(1.0 / s_trunc)
    k_matrix = g_future @ vt_trunc.T @ s_inv @ u_trunc.T  # (n_obs, n_obs)

    # Compute max eigenvalue magnitude for stability diagnostic (N-03)
    eigvals = np.linalg.eigvals(k_matrix)
    max_eig_mag = float(np.max(np.abs(eigvals)))

    # Training error: RMS of ||G_future - K @ G_current|| / ||G_future||
    residual = g_future - k_matrix @ g_current
    rms_residual = np.sqrt(np.mean(residual ** 2))
    rms_future = np.sqrt(np.mean(g_future ** 2))
    training_error = float(rms_residual / rms_future) if rms_future > 1e-15 else 0.0

    _STABILITY_EPSILON = 1e-6
    stable = max_eig_mag <= 1.0 + _STABILITY_EPSILON

    return KoopmanModel(
        koopman_matrix=tuple(float(v) for v in k_matrix.ravel()),
        n_observables=n_obs,
        training_step_s=step_s,
        mean_state=tuple(float(v) for v in mean_state),
        singular_values=singular_values,
        training_error=training_error,
        max_eigenvalue_magnitude=max_eig_mag,
        is_stable=stable,
    )


def predict_koopman(
    model: KoopmanModel,
    initial_position: tuple[float, float, float],
    initial_velocity: tuple[float, float, float],
    duration_s: float,
    step_s: float,
) -> KoopmanPrediction:
    """Propagate using fitted Koopman operator.

    For each step: obs_{n+1} = K @ obs_n
    Extract position/velocity from first 6 observables.

    Args:
        model: Fitted KoopmanModel from fit_koopman_model.
        initial_position: Initial (x, y, z) in meters.
        initial_velocity: Initial (vx, vy, vz) in m/s.
        duration_s: Total prediction duration in seconds.
        step_s: Output step size in seconds.

    Returns:
        KoopmanPrediction with time series of positions and velocities.
    """
    if not model.is_stable:
        logger.warning(
            "Koopman model is unstable (max eigenvalue magnitude %.4f > 1.0). "
            "Predictions may diverge.",
            model.max_eigenvalue_magnitude,
        )

    n_obs = model.n_observables
    k_matrix = np.array(model.koopman_matrix, dtype=np.float64).reshape(n_obs, n_obs)

    # Compute how many Koopman steps per output step
    # The model was trained at model.training_step_s intervals
    # If step_s matches training_step_s, apply K once per step.
    # If step_s is a multiple, apply K multiple times.
    # For simplicity: compute the number of training steps per output step
    steps_per_output = max(1, round(step_s / model.training_step_s))

    # Pre-compute K^steps_per_output if needed
    if steps_per_output == 1:
        k_step = k_matrix
    else:
        k_step = np.linalg.matrix_power(k_matrix, steps_per_output)

    num_output = int(duration_s / step_s) + 1

    # Build initial observable
    obs = _build_observables(initial_position, initial_velocity, n_obs)

    times = []
    positions = []
    velocities = []

    for i in range(num_output):
        t = i * step_s
        times.append(t)

        # Extract position and velocity from first 6 components
        px, py, pz = float(obs[0]), float(obs[1]), float(obs[2])
        vx, vy, vz = float(obs[3]), float(obs[4]), float(obs[5])
        positions.append((px, py, pz))
        velocities.append((vx, vy, vz))

        # Advance one output step
        if i < num_output - 1:
            obs = k_step @ obs

    return KoopmanPrediction(
        times_s=tuple(times),
        positions_eci=tuple(positions),
        velocities_eci=tuple(velocities),
        model=model,
        is_stable=model.is_stable,
    )
