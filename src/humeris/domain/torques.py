# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Gravity gradient and aerodynamic torques.

Computes environmental torques on spacecraft for attitude disturbance
analysis. Gravity gradient uses the general 3μ/r⁵ formulation.
Aerodynamic torque uses drag force with center-of-pressure offset.

No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.atmosphere import DragConfig, atmospheric_density

_MU = OrbitalConstants.MU_EARTH
_R_EARTH = OrbitalConstants.R_EARTH
_OMEGA_EARTH = OrbitalConstants.EARTH_ROTATION_RATE


@dataclass(frozen=True)
class InertiaTensor:
    """Spacecraft principal inertia tensor (kg·m²)."""
    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0


@dataclass(frozen=True)
class TorqueResult:
    """Torque vector and scalar magnitude."""
    torque_nm: tuple[float, float, float]
    magnitude_nm: float


@dataclass(frozen=True)
class AerodynamicTorqueResult:
    """Aerodynamic torque with drag force magnitude."""
    torque_nm: tuple[float, float, float]
    magnitude_nm: float
    drag_force_n: float


def compute_gravity_gradient_torque(
    pos_eci: tuple[float, float, float],
    inertia: InertiaTensor,
) -> TorqueResult:
    """Gravity gradient torque for spacecraft at given ECI position.

    T_gg = (3μ/r⁵) · (r × (I · r))

    Args:
        pos_eci: Satellite ECI position (m).
        inertia: Spacecraft inertia tensor.

    Returns:
        TorqueResult with torque vector (Nm) and magnitude.
    """
    r_vec = np.array(pos_eci)
    r_mag = float(np.linalg.norm(r_vec))
    r5 = r_mag ** 5

    coeff = 3.0 * _MU / r5

    # I · r (matrix-vector multiply with full tensor)
    I_mat = np.array([
        [inertia.ixx, inertia.ixy, inertia.ixz],
        [inertia.ixy, inertia.iyy, inertia.iyz],
        [inertia.ixz, inertia.iyz, inertia.izz],
    ])
    ir_vec = np.dot(I_mat, r_vec)

    # r × (I · r)
    cross = np.cross(r_vec, ir_vec)
    torque_vec = coeff * cross

    torque = (float(torque_vec[0]), float(torque_vec[1]), float(torque_vec[2]))
    magnitude = float(np.linalg.norm(torque_vec))

    return TorqueResult(torque_nm=torque, magnitude_nm=magnitude)


def compute_aerodynamic_torque(
    pos_eci: tuple[float, float, float],
    vel_eci: tuple[float, float, float],
    drag_config: DragConfig,
    cp_offset_m: tuple[float, float, float],
) -> AerodynamicTorqueResult:
    """Aerodynamic torque from drag with center-of-pressure offset.

    F_drag = 0.5 · ρ · v² · Cd · A (in velocity direction)
    T_aero = F_drag_vec × cp_offset

    Accounts for co-rotating atmosphere.

    Args:
        pos_eci: Satellite ECI position (m).
        vel_eci: Satellite ECI velocity (m/s).
        drag_config: Satellite drag configuration.
        cp_offset_m: Center-of-pressure offset from center-of-mass (m).

    Returns:
        AerodynamicTorqueResult with torque, magnitude, and drag force.
    """
    r_vec = np.array(pos_eci)
    r_mag = float(np.linalg.norm(r_vec))
    alt_km = (r_mag - _R_EARTH) / 1000.0

    # Co-rotating atmosphere velocity
    v_atm = np.array([-_OMEGA_EARTH * pos_eci[1], _OMEGA_EARTH * pos_eci[0], 0.0])

    # Relative velocity
    v_rel_vec = np.array(vel_eci) - v_atm
    v_rel = float(np.linalg.norm(v_rel_vec))

    if v_rel < 1e-10:
        return AerodynamicTorqueResult(
            torque_nm=(0.0, 0.0, 0.0), magnitude_nm=0.0, drag_force_n=0.0,
        )

    rho = atmospheric_density(alt_km)
    f_drag = 0.5 * rho * v_rel**2 * drag_config.cd * drag_config.area_m2

    # Drag force vector (opposite to relative velocity)
    f_hat = -v_rel_vec / v_rel
    f_vec = f_drag * f_hat

    # Torque = F × cp_offset
    cp_vec = np.array(cp_offset_m)
    torque_vec = np.cross(f_vec, cp_vec)

    magnitude = float(np.linalg.norm(torque_vec))

    return AerodynamicTorqueResult(
        torque_nm=(float(torque_vec[0]), float(torque_vec[1]), float(torque_vec[2])),
        magnitude_nm=magnitude,
        drag_force_n=f_drag,
    )
