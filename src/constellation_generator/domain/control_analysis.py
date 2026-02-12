# Copyright (c) 2026 Jeroen. All rights reserved.
"""CW controllability Gramian analysis.

The CW equations form a linear time-invariant system. The state transition
matrix Phi(t) from cw_propagate_state IS the system propagator. The
controllability Gramian eigenvalues reveal fuel cost anisotropy.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass

from constellation_generator.domain.relative_motion import RelativeState, cw_propagate_state
from constellation_generator.domain.linalg import (
    mat_zeros,
    mat_multiply,
    mat_transpose,
    mat_add,
    mat_scale,
    mat_eigenvalues_symmetric,
    mat_trace,
)


@dataclass(frozen=True)
class ControllabilityAnalysis:
    """Result of CW controllability Gramian analysis."""
    gramian_eigenvalues: tuple
    gramian_eigenvectors: tuple
    is_controllable: bool
    condition_number: float
    min_energy_direction: tuple
    max_energy_direction: tuple
    gramian_trace: float


def _state_transition_column(n_rad_s: float, t_s: float, col: int) -> list:
    """Get column `col` of the 6x6 state transition matrix Phi(t).

    Done by propagating a unit initial condition in dimension `col`.
    """
    init = [0.0] * 6
    init[col] = 1.0
    result = cw_propagate_state(
        RelativeState(x=init[0], y=init[1], z=init[2],
                      vx=init[3], vy=init[4], vz=init[5]),
        n_rad_s, t_s,
    )
    return [result.x, result.y, result.z, result.vx, result.vy, result.vz]


def _state_transition_matrix(n_rad_s: float, t_s: float) -> list:
    """Compute 6x6 state transition matrix Phi(t) by propagating unit vectors."""
    cols = [_state_transition_column(n_rad_s, t_s, c) for c in range(6)]
    # cols[c] is column c → transpose to get row-major
    phi = mat_zeros(6, 6)
    for r in range(6):
        for c in range(6):
            phi[r][c] = cols[c][r]
    return phi


def compute_cw_controllability(
    n_rad_s: float,
    duration_s: float,
    step_s: float = 10.0,
) -> ControllabilityAnalysis:
    """Compute controllability Gramian of the CW system.

    W_c(T) = integral_0^T Phi(tau) * Phi(tau)^T dtau
           approx= sum_i Phi(i*dt) * Phi(i*dt)^T * dt

    Eigenvalues of W_c reveal control effort per direction.
    """
    num_steps = max(1, int(duration_s / step_s))
    dt = duration_s / num_steps

    # Integrate Gramian using trapezoidal rule
    wc = mat_zeros(6, 6)
    for i in range(num_steps + 1):
        tau = i * dt
        phi = _state_transition_matrix(n_rad_s, tau)
        phi_t = mat_transpose(phi)
        phi_phit = mat_multiply(phi, phi_t)
        weight = dt if 0 < i < num_steps else dt / 2.0
        scaled = mat_scale(phi_phit, weight)
        wc = mat_add(wc, scaled)

    # Eigendecompose the Gramian
    eig = mat_eigenvalues_symmetric(wc)
    eigenvalues = eig.eigenvalues
    eigenvectors = eig.eigenvectors

    # Controllability check: all eigenvalues > small threshold
    min_eig = min(eigenvalues) if eigenvalues else 0.0
    max_eig = max(eigenvalues) if eigenvalues else 0.0
    is_controllable = min_eig > 1e-6 and len(eigenvalues) == 6

    condition_number = max_eig / min_eig if min_eig > 1e-15 else float('inf')

    min_energy_dir = eigenvectors[0] if eigenvectors else ()
    max_energy_dir = eigenvectors[-1] if eigenvectors else ()

    gramian_tr = mat_trace(wc)

    return ControllabilityAnalysis(
        gramian_eigenvalues=eigenvalues,
        gramian_eigenvectors=eigenvectors,
        is_controllable=is_controllable,
        condition_number=condition_number,
        min_energy_direction=min_energy_dir,
        max_energy_direction=max_energy_dir,
        gramian_trace=gramian_tr,
    )
