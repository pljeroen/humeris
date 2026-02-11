"""
Shared Keplerian + J2 propagation.

Reusable orbital state derivation and propagation functions for
converting satellite objects to time-varying ECI/ECEF positions.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from constellation_generator.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
    j2_raan_rate,
    j2_arg_perigee_rate,
    j2_mean_motion_correction,
)
from constellation_generator.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
)


@dataclass(frozen=True)
class OrbitalState:
    """Frozen orbital state for propagation."""
    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_perigee_rad: float
    true_anomaly_rad: float
    mean_motion_rad_s: float
    reference_epoch: datetime
    j2_raan_rate: float = 0.0
    j2_arg_perigee_rate: float = 0.0
    j2_mean_motion_correction: float = 0.0


def derive_orbital_state(
    satellite,
    reference_epoch: datetime,
    include_j2: bool = False,
) -> OrbitalState:
    """
    Derive an OrbitalState from a Satellite domain object.

    Assumes circular orbit (a = |pos|, e = 0).

    Args:
        satellite: Satellite with position_eci, velocity_eci, raan_deg,
            true_anomaly_deg, and optional epoch.
        reference_epoch: Reference time for the state.
        include_j2: If True, compute J2 secular perturbation rates.

    Returns:
        OrbitalState for use with propagate_to / propagate_ecef_to.
    """
    px, py, pz = satellite.position_eci
    vx, vy, vz = satellite.velocity_eci

    r_mag = math.sqrt(px**2 + py**2 + pz**2)
    a = r_mag
    e = 0.0

    hx = py * vz - pz * vy
    hy = pz * vx - px * vz
    hz = px * vy - py * vx
    h_mag = math.sqrt(hx**2 + hy**2 + hz**2)
    inc_rad = math.acos(hz / h_mag) if h_mag > 0 else 0.0

    n = math.sqrt(OrbitalConstants.MU_EARTH / a**3)

    raan_rad = math.radians(satellite.raan_deg)
    nu_0_rad = math.radians(satellite.true_anomaly_deg)

    if satellite.epoch is not None:
        epoch_offset = (reference_epoch - satellite.epoch).total_seconds()
        nu_at_ref = nu_0_rad + n * epoch_offset
    else:
        nu_at_ref = nu_0_rad

    j2_raan = 0.0
    j2_argp = 0.0
    j2_n_corr = 0.0

    if include_j2:
        j2_raan = j2_raan_rate(n, a, e, inc_rad)
        j2_argp = j2_arg_perigee_rate(n, a, e, inc_rad)
        j2_n_corr = j2_mean_motion_correction(n, a, e, inc_rad)

    return OrbitalState(
        semi_major_axis_m=a,
        eccentricity=e,
        inclination_rad=inc_rad,
        raan_rad=raan_rad,
        arg_perigee_rad=0.0,
        true_anomaly_rad=nu_at_ref,
        mean_motion_rad_s=n,
        reference_epoch=reference_epoch,
        j2_raan_rate=j2_raan,
        j2_arg_perigee_rate=j2_argp,
        j2_mean_motion_correction=j2_n_corr,
    )


def propagate_to(
    state: OrbitalState,
    target_time: datetime,
) -> tuple[list[float], list[float]]:
    """
    Propagate orbital state to target time, returning ECI position/velocity.

    Applies J2 secular corrections to RAAN, argument of perigee, and
    mean motion if the state has nonzero J2 rates.

    Args:
        state: OrbitalState from derive_orbital_state.
        target_time: Target UTC datetime.

    Returns:
        (position_eci [x,y,z] in m, velocity_eci [vx,vy,vz] in m/s)
    """
    dt = (target_time - state.reference_epoch).total_seconds()

    raan = state.raan_rad + state.j2_raan_rate * dt
    arg_perigee = state.arg_perigee_rad + state.j2_arg_perigee_rate * dt

    n_eff = state.j2_mean_motion_correction if state.j2_mean_motion_correction != 0.0 else state.mean_motion_rad_s
    nu = state.true_anomaly_rad + n_eff * dt

    return kepler_to_cartesian(
        a=state.semi_major_axis_m,
        e=state.eccentricity,
        i_rad=state.inclination_rad,
        omega_big_rad=raan,
        omega_small_rad=arg_perigee,
        nu_rad=nu,
    )


def propagate_ecef_to(
    state: OrbitalState,
    target_time: datetime,
) -> tuple[float, float, float]:
    """
    Propagate orbital state to target time, returning ECEF position.

    Composes propagate_to → gmst → eci_to_ecef.

    Args:
        state: OrbitalState from derive_orbital_state.
        target_time: Target UTC datetime.

    Returns:
        (x, y, z) in meters, ECEF frame.
    """
    pos_eci, vel_eci = propagate_to(state, target_time)
    gmst_angle = gmst_rad(target_time)
    pos_ecef, _ = eci_to_ecef(
        (pos_eci[0], pos_eci[1], pos_eci[2]),
        (vel_eci[0], vel_eci[1], vel_eci[2]),
        gmst_angle,
    )
    return pos_ecef
