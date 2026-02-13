# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Derived orbital properties from state vectors and orbital elements.

Computes orbital velocity/period, specific energy/angular momentum,
Sun-synchronous verification, RSW velocity decomposition, LTAN,
state-vector-to-elements conversion, and ground track repeat analysis.

"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    j2_raan_rate,
)
from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.solar import sun_position_eci

_MU = OrbitalConstants.MU_EARTH
_R_E = OrbitalConstants.R_EARTH
_SIDEREAL_DAY_S = 86164.0905
_SSO_RAAN_RATE_DEG_DAY = 360.0 / 365.2422  # ~0.9856 deg/day


# --- Types ---

@dataclass(frozen=True)
class OrbitalVelocity:
    """Orbital velocity, period, and ground speed."""
    circular_velocity_ms: float
    orbital_period_s: float
    ground_speed_kmh: float


@dataclass(frozen=True)
class EnergyMomentum:
    """Specific orbital energy and angular momentum."""
    specific_energy_j_kg: float
    angular_momentum_m2_s: float
    vis_viva_velocity_ms: float


@dataclass(frozen=True)
class SunSyncCheck:
    """Result of Sun-synchronous orbit verification."""
    is_sun_synchronous: bool
    actual_raan_rate_deg_day: float
    required_raan_rate_deg_day: float
    error_deg_day: float


@dataclass(frozen=True)
class GroundTrackRepeat:
    """Ground track repeat analysis result."""
    revs_per_day: float
    near_repeat_revs: int
    near_repeat_days: int
    drift_deg_per_day: float


# --- Functions ---

def compute_orbital_velocity(state: OrbitalState) -> OrbitalVelocity:
    """Compute circular velocity, orbital period, and sub-satellite ground speed.

    Args:
        state: OrbitalState with semi_major_axis_m.

    Returns:
        OrbitalVelocity with velocity (m/s), period (s), ground speed (km/h).
    """
    a = state.semi_major_axis_m
    v_circ = float(np.sqrt(_MU / a))
    T = 2.0 * np.pi * float(np.sqrt(a**3 / _MU))

    # Ground speed: project orbital velocity onto Earth surface
    # v_ground = v_orbital * (R_E / a) - Earth rotation contribution is small
    # Simplified: ground speed ≈ v_orbital * R_E / a
    v_ground_ms = v_circ * _R_E / a
    v_ground_kmh = v_ground_ms * 3.6

    return OrbitalVelocity(
        circular_velocity_ms=v_circ,
        orbital_period_s=T,
        ground_speed_kmh=v_ground_kmh,
    )


def compute_energy_momentum(
    pos_eci: list[float] | tuple[float, float, float],
    vel_eci: list[float] | tuple[float, float, float],
    mu: float | None = None,
) -> EnergyMomentum:
    """Compute specific orbital energy and angular momentum from state vectors.

    Energy: E = v²/2 - mu/r (vis-viva)
    Angular momentum: h = |r × v|

    Args:
        pos_eci: ECI position [x, y, z] in meters.
        vel_eci: ECI velocity [vx, vy, vz] in m/s.
        mu: Gravitational parameter (defaults to Earth).

    Returns:
        EnergyMomentum with energy (J/kg), angular momentum (m²/s), vis-viva velocity.
    """
    if mu is None:
        mu = _MU

    p_vec = np.array([pos_eci[0], pos_eci[1], pos_eci[2]], dtype=np.float64)
    v_vec = np.array([vel_eci[0], vel_eci[1], vel_eci[2]], dtype=np.float64)

    r = float(np.linalg.norm(p_vec))
    if r < 1e-10:
        raise ValueError("position vector has zero magnitude")
    v = float(np.linalg.norm(v_vec))

    # Specific orbital energy (vis-viva)
    energy = v**2 / 2.0 - mu / r

    # Angular momentum vector: h = r × v
    h_vec = np.cross(p_vec, v_vec)
    h_mag = float(np.linalg.norm(h_vec))

    # Vis-viva velocity at current radius
    # v² = mu * (2/r - 1/a), a = -mu/(2*E)
    vis_viva_v = float(np.sqrt(2.0 * (energy + mu / r))) if (energy + mu / r) > 0 else v

    return EnergyMomentum(
        specific_energy_j_kg=energy,
        angular_momentum_m2_s=h_mag,
        vis_viva_velocity_ms=vis_viva_v,
    )


def check_sun_synchronous(state: OrbitalState) -> SunSyncCheck:
    """Verify if an orbit is Sun-synchronous.

    Compares actual J2 RAAN drift rate to the required SSO rate
    (~0.9856 deg/day). Tolerates ±0.01 deg/day.

    Args:
        state: OrbitalState (should have J2 rates if include_j2 was used).

    Returns:
        SunSyncCheck with verification result and rates.
    """
    a = state.semi_major_axis_m
    n = state.mean_motion_rad_s
    e = state.eccentricity
    i = state.inclination_rad

    # Compute actual J2 RAAN rate
    actual_rate_rad_s = j2_raan_rate(n, a, e, i)
    actual_rate_deg_day = float(np.degrees(actual_rate_rad_s)) * 86400.0

    required_rate_deg_day = _SSO_RAAN_RATE_DEG_DAY
    error = actual_rate_deg_day - required_rate_deg_day

    # Tolerance: ±0.01 deg/day
    is_sso = abs(error) < 0.01

    return SunSyncCheck(
        is_sun_synchronous=is_sso,
        actual_raan_rate_deg_day=actual_rate_deg_day,
        required_raan_rate_deg_day=required_rate_deg_day,
        error_deg_day=error,
    )


def compute_rsw_velocity(
    pos_eci: list[float] | tuple[float, float, float],
    vel_eci: list[float] | tuple[float, float, float],
) -> tuple[float, float, float]:
    """Decompose ECI velocity into RSW (radial, along-track, cross-track).

    R = radial (along position vector, positive outward)
    S = along-track (in orbital plane, perpendicular to R, positive in velocity direction)
    W = cross-track (normal to orbital plane, completes right-hand system)

    Ref: Vallado Ch. 3.

    Args:
        pos_eci: ECI position [x, y, z] in meters.
        vel_eci: ECI velocity [vx, vy, vz] in m/s.

    Returns:
        (radial_ms, along_track_ms, cross_track_ms).
    """
    p_vec = np.array([pos_eci[0], pos_eci[1], pos_eci[2]], dtype=np.float64)
    v_vec = np.array([vel_eci[0], vel_eci[1], vel_eci[2]], dtype=np.float64)

    # R unit vector (radial, along position)
    r_mag = float(np.linalg.norm(p_vec))
    if r_mag < 1e-10:
        raise ValueError("position vector has zero magnitude")
    r_hat = p_vec / r_mag

    # W unit vector (cross-track, h = r × v, then normalize)
    w_vec = np.cross(p_vec, v_vec)
    w_mag = float(np.linalg.norm(w_vec))
    w_hat = w_vec / w_mag

    # S unit vector (along-track, S = W × R)
    s_hat = np.cross(w_hat, r_hat)

    # Project velocity onto RSW
    v_r = float(np.dot(v_vec, r_hat))
    v_s = float(np.dot(v_vec, s_hat))
    v_w = float(np.dot(v_vec, w_hat))

    return v_r, v_s, v_w


def compute_ltan(raan_rad: float, epoch: datetime) -> float:
    """Compute Local Time of Ascending Node (LTAN).

    LTAN = 12 + (RAAN - Sun_RA) / 15 (hours)

    Args:
        raan_rad: Right ascension of ascending node (radians).
        epoch: Epoch for Sun position.

    Returns:
        LTAN in hours (0.0 to 24.0).
    """
    sun = sun_position_eci(epoch)
    ra_sun_deg = float(np.degrees(sun.right_ascension_rad))
    raan_deg = float(np.degrees(raan_rad))

    ltan = 12.0 + (raan_deg - ra_sun_deg) / 15.0
    ltan = ltan % 24.0
    return ltan


def state_vector_to_elements(
    pos_eci: list[float] | tuple[float, float, float],
    vel_eci: list[float] | tuple[float, float, float],
    mu: float | None = None,
) -> dict:
    """Convert ECI state vector to classical orbital elements.

    Ref: Vallado Algorithm 9 (rv2coe).

    Args:
        pos_eci: ECI position [x, y, z] in meters.
        vel_eci: ECI velocity [vx, vy, vz] in m/s.
        mu: Gravitational parameter (defaults to Earth).

    Returns:
        Dict with keys: semi_major_axis_m, eccentricity, inclination_deg,
        raan_deg, arg_perigee_deg, true_anomaly_deg.
    """
    if mu is None:
        mu = _MU

    p_vec = np.array([pos_eci[0], pos_eci[1], pos_eci[2]], dtype=np.float64)
    v_vec = np.array([vel_eci[0], vel_eci[1], vel_eci[2]], dtype=np.float64)
    px, py, pz = p_vec[0], p_vec[1], p_vec[2]
    vx, vy, vz = v_vec[0], v_vec[1], v_vec[2]

    r_mag = float(np.linalg.norm(p_vec))
    if r_mag < 1e-10:
        raise ValueError("position vector has zero magnitude")
    v_mag = float(np.linalg.norm(v_vec))

    # Angular momentum h = r × v
    h_vec = np.cross(p_vec, v_vec)
    hx, hy, hz = h_vec[0], h_vec[1], h_vec[2]
    h_mag = float(np.linalg.norm(h_vec))

    # Node vector n = k × h (where k = [0, 0, 1])
    nx = -hy
    ny = hx
    n_mag = float(np.sqrt(nx**2 + ny**2))

    # Eccentricity vector e = ((v² - mu/r)*r - (r·v)*v) / mu
    rdotv = float(np.dot(p_vec, v_vec))
    coeff1 = (v_mag**2 - mu / r_mag)
    coeff2 = rdotv

    e_vec = (coeff1 * p_vec - coeff2 * v_vec) / mu
    ex, ey, ez = e_vec[0], e_vec[1], e_vec[2]
    ecc = float(np.linalg.norm(e_vec))

    # Semi-major axis from vis-viva
    energy = v_mag**2 / 2.0 - mu / r_mag
    if abs(energy) > 1e-10:
        a = -mu / (2.0 * energy)
    else:
        a = float('inf')

    # Inclination
    inc_rad = float(np.arccos(np.clip(hz / h_mag, -1.0, 1.0))) if h_mag > 1e-10 else 0.0

    # RAAN
    if n_mag > 1e-10:
        raan_rad = float(np.arccos(np.clip(nx / n_mag, -1.0, 1.0)))
        if ny < 0:
            raan_rad = 2.0 * np.pi - raan_rad
    else:
        raan_rad = 0.0

    # Argument of perigee
    if n_mag > 1e-10 and ecc > 1e-10:
        n_vec_2d = np.array([nx, ny], dtype=np.float64)
        e_vec_2d = np.array([ex, ey], dtype=np.float64)
        cos_omega = float(np.dot(n_vec_2d, e_vec_2d)) / (n_mag * ecc)
        cos_omega = float(np.clip(cos_omega, -1.0, 1.0))
        omega_rad = float(np.arccos(cos_omega))
        if ez < 0:
            omega_rad = 2.0 * np.pi - omega_rad
    else:
        omega_rad = 0.0

    # True anomaly
    if ecc > 1e-10:
        cos_nu = float(np.dot(e_vec, p_vec)) / (ecc * r_mag)
        cos_nu = float(np.clip(cos_nu, -1.0, 1.0))
        nu_rad = float(np.arccos(cos_nu))
        if rdotv < 0:
            nu_rad = 2.0 * np.pi - nu_rad
    else:
        # Circular orbit: use argument of latitude
        if n_mag > 1e-10:
            n_vec_full = np.array([nx, ny, 0.0], dtype=np.float64)
            cos_u = float(np.dot(n_vec_full, p_vec)) / (n_mag * r_mag)
            cos_u = float(np.clip(cos_u, -1.0, 1.0))
            nu_rad = float(np.arccos(cos_u))
            if pz < 0:
                nu_rad = 2.0 * np.pi - nu_rad
        else:
            nu_rad = 0.0

    return {
        "semi_major_axis_m": a,
        "eccentricity": ecc,
        "inclination_deg": float(np.degrees(inc_rad)),
        "raan_deg": float(np.degrees(raan_rad)),
        "arg_perigee_deg": float(np.degrees(omega_rad)),
        "true_anomaly_deg": float(np.degrees(nu_rad)),
    }


def compute_ground_track_repeat(state: OrbitalState) -> GroundTrackRepeat:
    """Find nearest ground track repeat period using continued fraction approximation.

    Computes revolutions per day and finds the closest rational approximation
    (revs/days) to determine when the ground track approximately repeats.

    Args:
        state: OrbitalState.

    Returns:
        GroundTrackRepeat with revs/day, nearest repeat integers, and drift rate.
    """
    a = state.semi_major_axis_m
    T = 2.0 * np.pi * float(np.sqrt(a**3 / _MU))
    revs_per_day = _SIDEREAL_DAY_S / T

    # Continued fraction approximation to find p/q ≈ revs_per_day
    best_p, best_q = _best_rational_approx(revs_per_day, max_denom=100)

    # Drift per day: how far the ground track shifts
    # Exact repeat requires integer revs in integer days
    drift_deg_per_day = (revs_per_day - best_p / best_q) * 360.0

    return GroundTrackRepeat(
        revs_per_day=revs_per_day,
        near_repeat_revs=best_p,
        near_repeat_days=best_q,
        drift_deg_per_day=drift_deg_per_day,
    )


def _best_rational_approx(x: float, max_denom: int = 100) -> tuple[int, int]:
    """Find best rational approximation p/q to x with q <= max_denom.

    Uses Stern-Brocot tree / mediants for convergents.
    """
    best_p = round(x)
    best_q = 1
    best_err = abs(x - best_p)

    for q in range(1, max_denom + 1):
        p = round(x * q)
        err = abs(x - p / q)
        if err < best_err:
            best_err = err
            best_p = p
            best_q = q
            if err < 1e-10:
                break

    return best_p, best_q


# --- Element History ---

@dataclass(frozen=True)
class ElementSnapshot:
    """Orbital elements at a single point in time."""
    time: datetime
    semi_major_axis_m: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    true_anomaly_deg: float


@dataclass(frozen=True)
class ElementHistory:
    """Time series of orbital elements."""
    snapshots: tuple[ElementSnapshot, ...]
    duration_s: float


def compute_element_history(
    state: OrbitalState,
    epoch: datetime,
    duration_s: float,
    step_s: float,
) -> ElementHistory:
    """Compute orbital element time history by propagating and converting back.

    At each time step, propagates the state to get position/velocity,
    then converts back to classical orbital elements.

    Args:
        state: Initial OrbitalState.
        epoch: Start time.
        duration_s: Total duration in seconds.
        step_s: Time step in seconds.

    Returns:
        ElementHistory with snapshots at each time step.
    """
    from datetime import timedelta

    if step_s <= 0 or duration_s <= 0:
        return ElementHistory(snapshots=(), duration_s=0.0)

    snapshots: list[ElementSnapshot] = []
    elapsed = 0.0

    while elapsed <= duration_s + 1e-9:
        current_time = epoch + timedelta(seconds=elapsed)
        pos, vel = propagate_to(state, current_time)
        elements = state_vector_to_elements(pos, vel)

        snapshots.append(ElementSnapshot(
            time=current_time,
            semi_major_axis_m=elements["semi_major_axis_m"],
            eccentricity=elements["eccentricity"],
            inclination_deg=elements["inclination_deg"],
            raan_deg=elements["raan_deg"],
            arg_perigee_deg=elements["arg_perigee_deg"],
            true_anomaly_deg=elements["true_anomaly_deg"],
        ))
        elapsed += step_s

    return ElementHistory(
        snapshots=tuple(snapshots),
        duration_s=duration_s,
    )
