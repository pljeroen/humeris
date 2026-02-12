"""Gravity gradient and aerodynamic torques.

Computes environmental torques on spacecraft for attitude disturbance
analysis. Gravity gradient uses the general 3μ/r⁵ formulation.
Aerodynamic torque uses drag force with center-of-pressure offset.

No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.atmosphere import DragConfig, atmospheric_density

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
    rx, ry, rz = pos_eci
    r_mag = math.sqrt(rx**2 + ry**2 + rz**2)
    r5 = r_mag ** 5

    coeff = 3.0 * _MU / r5

    # I · r (matrix-vector multiply with full tensor)
    ir_x = inertia.ixx * rx + inertia.ixy * ry + inertia.ixz * rz
    ir_y = inertia.ixy * rx + inertia.iyy * ry + inertia.iyz * rz
    ir_z = inertia.ixz * rx + inertia.iyz * ry + inertia.izz * rz

    # r × (I · r)
    tx = ry * ir_z - rz * ir_y
    ty = rz * ir_x - rx * ir_z
    tz = rx * ir_y - ry * ir_x

    torque = (coeff * tx, coeff * ty, coeff * tz)
    magnitude = math.sqrt(torque[0]**2 + torque[1]**2 + torque[2]**2)

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
    rx, ry, rz = pos_eci
    r_mag = math.sqrt(rx**2 + ry**2 + rz**2)
    alt_km = (r_mag - _R_EARTH) / 1000.0

    # Co-rotating atmosphere velocity
    v_atm_x = -_OMEGA_EARTH * ry
    v_atm_y = _OMEGA_EARTH * rx
    v_atm_z = 0.0

    # Relative velocity
    vrel_x = vel_eci[0] - v_atm_x
    vrel_y = vel_eci[1] - v_atm_y
    vrel_z = vel_eci[2] - v_atm_z
    v_rel = math.sqrt(vrel_x**2 + vrel_y**2 + vrel_z**2)

    if v_rel < 1e-10:
        return AerodynamicTorqueResult(
            torque_nm=(0.0, 0.0, 0.0), magnitude_nm=0.0, drag_force_n=0.0,
        )

    rho = atmospheric_density(alt_km)
    f_drag = 0.5 * rho * v_rel**2 * drag_config.cd * drag_config.area_m2

    # Drag force vector (opposite to relative velocity)
    f_hat_x = -vrel_x / v_rel
    f_hat_y = -vrel_y / v_rel
    f_hat_z = -vrel_z / v_rel

    fx = f_drag * f_hat_x
    fy = f_drag * f_hat_y
    fz = f_drag * f_hat_z

    # Torque = F × cp_offset
    ox, oy, oz = cp_offset_m
    tx = fy * oz - fz * oy
    ty = fz * ox - fx * oz
    tz = fx * oy - fy * ox

    magnitude = math.sqrt(tx**2 + ty**2 + tz**2)

    return AerodynamicTorqueResult(
        torque_nm=(tx, ty, tz),
        magnitude_nm=magnitude,
        drag_force_n=f_drag,
    )
