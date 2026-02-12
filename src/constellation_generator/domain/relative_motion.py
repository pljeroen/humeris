# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Relative motion using Clohessy-Wiltshire (Hill) linearized equations.

Analytical CW solution for deputy satellite motion relative to chief
in the LVLH frame (x=radial, y=along-track, z=cross-track).

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime

from constellation_generator.domain.propagation import OrbitalState, propagate_to


@dataclass(frozen=True)
class RelativeState:
    """Relative position and velocity in LVLH frame."""
    x: float      # radial (m)
    y: float      # along-track (m)
    z: float      # cross-track (m)
    vx: float     # radial rate (m/s)
    vy: float     # along-track rate (m/s)
    vz: float     # cross-track rate (m/s)


@dataclass(frozen=True)
class CWTrajectory:
    """Time series of CW relative motion."""
    states: tuple[RelativeState, ...]
    is_bounded: bool
    drift_rate_m_per_s: float


def cw_propagate_state(
    rel_state: RelativeState,
    n_rad_s: float,
    t_s: float,
) -> RelativeState:
    """Single-point analytical CW propagation.

    Propagates the linearized relative state forward by t_s seconds
    using the closed-form Clohessy-Wiltshire solution.

    Args:
        rel_state: Initial relative state in LVLH.
        n_rad_s: Chief orbit mean motion (rad/s).
        t_s: Propagation time (seconds).

    Returns:
        Propagated RelativeState.
    """
    x0 = rel_state.x
    y0 = rel_state.y
    z0 = rel_state.z
    vx0 = rel_state.vx
    vy0 = rel_state.vy
    vz0 = rel_state.vz

    n = n_rad_s
    nt = n * t_s
    cos_nt = math.cos(nt)
    sin_nt = math.sin(nt)

    # Position
    x = (4.0 - 3.0 * cos_nt) * x0 + sin_nt / n * vx0 + 2.0 * (1.0 - cos_nt) / n * vy0
    y = (6.0 * (sin_nt - nt) * x0 + y0
         - 2.0 * (1.0 - cos_nt) / n * vx0
         + (4.0 * sin_nt / n - 3.0 * t_s) * vy0)
    z = z0 * cos_nt + vz0 / n * sin_nt

    # Velocity
    vx_new = 3.0 * n * sin_nt * x0 + cos_nt * vx0 + 2.0 * sin_nt * vy0
    vy_new = (6.0 * n * (cos_nt - 1.0) * x0
              - 2.0 * sin_nt * vx0
              + (4.0 * cos_nt - 3.0) * vy0)
    vz_new = -z0 * n * sin_nt + vz0 * cos_nt

    return RelativeState(x=x, y=y, z=z, vx=vx_new, vy=vy_new, vz=vz_new)


def cw_propagate(
    rel_state: RelativeState,
    n_rad_s: float,
    duration_s: float,
    step_s: float,
) -> CWTrajectory:
    """CW time series with bounded orbit detection.

    Args:
        rel_state: Initial relative state in LVLH.
        n_rad_s: Chief orbit mean motion (rad/s).
        duration_s: Total propagation duration (seconds).
        step_s: Time step (seconds).

    Returns:
        CWTrajectory with state snapshots, bounded flag, and drift rate.
    """
    # Along-track drift rate: dy_secular/dt = -(6n·x₀ + 3ẏ₀)
    drift = -(6.0 * n_rad_s * rel_state.x + 3.0 * rel_state.vy)

    # Bounded when drift ≈ 0 (i.e., ẏ₀ = -2nx₀)
    is_bounded = abs(drift) < 1e-6

    states: list[RelativeState] = []
    elapsed = 0.0
    while elapsed <= duration_s + 1e-9:
        states.append(cw_propagate_state(rel_state, n_rad_s, elapsed))
        elapsed += step_s

    return CWTrajectory(
        states=tuple(states),
        is_bounded=is_bounded,
        drift_rate_m_per_s=drift,
    )


def compute_relative_state(
    state_chief: OrbitalState,
    state_deputy: OrbitalState,
    time: datetime,
) -> RelativeState:
    """Convert two absolute OrbitalStates to LVLH relative state.

    Constructs the RSW (radial/along-track/cross-track) frame from the
    chief's position and velocity, then projects the deputy offset into
    that frame.

    Args:
        state_chief: Chief satellite orbital state.
        state_deputy: Deputy satellite orbital state.
        time: Evaluation epoch.

    Returns:
        RelativeState of deputy w.r.t. chief in LVLH.
    """
    pos_c, vel_c = propagate_to(state_chief, time)
    pos_d, vel_d = propagate_to(state_deputy, time)

    # Chief position and velocity as tuples
    rc = (pos_c[0], pos_c[1], pos_c[2])
    vc = (vel_c[0], vel_c[1], vel_c[2])

    # Build RSW frame (R=radial, S=along-track, W=cross-track)
    r_mag = math.sqrt(rc[0]**2 + rc[1]**2 + rc[2]**2)
    r_hat = (rc[0] / r_mag, rc[1] / r_mag, rc[2] / r_mag)

    # W = R × V / |R × V|
    hx = rc[1] * vc[2] - rc[2] * vc[1]
    hy = rc[2] * vc[0] - rc[0] * vc[2]
    hz = rc[0] * vc[1] - rc[1] * vc[0]
    h_mag = math.sqrt(hx**2 + hy**2 + hz**2)
    w_hat = (hx / h_mag, hy / h_mag, hz / h_mag)

    # S = W × R
    s_hat = (
        w_hat[1] * r_hat[2] - w_hat[2] * r_hat[1],
        w_hat[2] * r_hat[0] - w_hat[0] * r_hat[2],
        w_hat[0] * r_hat[1] - w_hat[1] * r_hat[0],
    )

    # Relative position in ECI
    dx = pos_d[0] - pos_c[0]
    dy = pos_d[1] - pos_c[1]
    dz = pos_d[2] - pos_c[2]

    # Relative velocity in ECI
    dvx = vel_d[0] - vel_c[0]
    dvy = vel_d[1] - vel_c[1]
    dvz = vel_d[2] - vel_c[2]

    # Project onto RSW
    x = dx * r_hat[0] + dy * r_hat[1] + dz * r_hat[2]
    y = dx * s_hat[0] + dy * s_hat[1] + dz * s_hat[2]
    z = dx * w_hat[0] + dy * w_hat[1] + dz * w_hat[2]

    vx = dvx * r_hat[0] + dvy * r_hat[1] + dvz * r_hat[2]
    vy = dvx * s_hat[0] + dvy * s_hat[1] + dvz * s_hat[2]
    vz = dvx * w_hat[0] + dvy * w_hat[1] + dvz * w_hat[2]

    return RelativeState(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)


def is_passively_safe(
    rel_state: RelativeState,
    n_rad_s: float,
    min_distance_m: float,
    check_duration_s: float,
    step_s: float = 10.0,
) -> bool:
    """Check if relative orbit maintains minimum distance over check period.

    Args:
        rel_state: Initial relative state in LVLH.
        n_rad_s: Chief orbit mean motion (rad/s).
        min_distance_m: Minimum allowed distance (m).
        check_duration_s: Duration to check (seconds).
        step_s: Check time step (seconds).

    Returns:
        True if minimum distance is never violated.
    """
    elapsed = 0.0
    while elapsed <= check_duration_s + 1e-9:
        s = cw_propagate_state(rel_state, n_rad_s, elapsed)
        dist = math.sqrt(s.x**2 + s.y**2 + s.z**2)
        if dist < min_distance_m:
            return False
        elapsed += step_s
    return True
