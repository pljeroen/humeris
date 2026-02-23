# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Numerical orbit propagation with RK4 and pluggable force models.

4th-order Runge-Kutta integrator with composable perturbation forces.
Handles accumulated perturbation effects (drag, SRP, higher-order
gravitational harmonics) that analytical propagation cannot model.

+ domain imports.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
)
from humeris.domain.atmosphere import (
    DragConfig,
    atmospheric_density,
)
from humeris.domain.solar import (
    sun_position_eci,
    AU_METERS,
)
from humeris.domain.eclipse import is_eclipsed, EclipseType


# --- Types ---

@runtime_checkable
class ForceModel(Protocol):
    """Structural typing port for pluggable force models."""

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]: ...


@dataclass(frozen=True)
class PropagationStep:
    """Single step in a numerical propagation trajectory."""
    time: datetime
    position_eci: tuple[float, float, float]
    velocity_eci: tuple[float, float, float]
    specific_energy_j_kg: float = 0.0  # v²/2 - μ/r (J/kg)


@dataclass(frozen=True)
class NumericalPropagationResult:
    """Complete result of a numerical propagation run."""
    steps: tuple[PropagationStep, ...]
    epoch: datetime
    duration_s: float
    force_model_names: tuple[str, ...]
    initial_energy_j_kg: float = 0.0
    final_energy_j_kg: float = 0.0
    max_energy_drift_j_kg: float = 0.0
    relative_energy_drift: float = 0.0  # |dE/E0|


# --- Energy computation ---

def _specific_energy(pos: tuple[float, float, float], vel: tuple[float, float, float]) -> float:
    """Compute specific orbital energy: E = v²/2 - μ/r."""
    r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    if r < 1e-3:
        return 0.0
    v = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
    return 0.5 * v**2 - OrbitalConstants.MU_EARTH / r


# --- Force models ---

class TwoBodyGravity:
    """Central body gravitational acceleration: a = -mu * r / |r|^3."""

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        pos = np.array(position)
        r = float(np.linalg.norm(pos))
        if r < 1e-3:
            return (0.0, 0.0, 0.0)
        r3 = r * r * r
        coeff = -OrbitalConstants.MU_EARTH / r3
        a = coeff * pos
        return (float(a[0]), float(a[1]), float(a[2]))


class J2Perturbation:
    """J2 zonal harmonic perturbation acceleration."""

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        pos = np.array(position)
        r2 = float(np.dot(pos, pos))
        r = float(np.sqrt(r2))
        if r < 1e-3:
            return (0.0, 0.0, 0.0)
        r5 = r2 * r2 * r

        mu = OrbitalConstants.MU_EARTH
        j2 = OrbitalConstants.J2_EARTH
        re = OrbitalConstants.R_EARTH_EQUATORIAL

        coeff = -1.5 * j2 * mu * re * re / r5
        z = position[2]
        z2_r2 = z * z / r2

        ax = coeff * pos[0] * (1.0 - 5.0 * z2_r2)
        ay = coeff * pos[1] * (1.0 - 5.0 * z2_r2)
        az = coeff * z * (3.0 - 5.0 * z2_r2)
        return (float(ax), float(ay), float(az))


class J3Perturbation:
    """J3 zonal harmonic perturbation acceleration.

    Derived from geopotential gradient (Montenbruck & Gill).
    """

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        pos = np.array(position)
        x, y, z = position
        r2 = float(np.dot(pos, pos))
        r = float(np.sqrt(r2))
        if r < 1e-3:
            return (0.0, 0.0, 0.0)
        r7 = r2 * r2 * r2 * r

        mu = OrbitalConstants.MU_EARTH
        j3 = OrbitalConstants.J3_EARTH
        re = OrbitalConstants.R_EARTH_EQUATORIAL

        coeff = -mu * j3 * re * re * re / 2.0

        z3_r2 = z * z * z / r2

        # Gradient of U_J3 = coeff*(5z³/r⁷ - 3z/r⁵):
        #   dU/dx = coeff * (x/r⁷) * (-35z³/r² + 15z)
        #   dU/dz = coeff * (1/r⁷) * (30z² - 35z⁴/r² - 3r²)
        ax = coeff * (x / r7) * (-35.0 * z3_r2 + 15.0 * z)
        ay = coeff * (y / r7) * (-35.0 * z3_r2 + 15.0 * z)
        az = coeff * (1.0 / r7) * (30.0 * z * z - 35.0 * z * z * z * z / r2 - 3.0 * r2)
        return (float(ax), float(ay), float(az))


# --- EGM96/JGM-3 spherical harmonic coefficients to degree 8 ---
# Format: _CS_COEFFS[(n, m)] = (C_nm, S_nm)
# n=degree, m=order. C_n0 = -J_n (zonal harmonics), S_n0 = 0.
_CS_COEFFS: dict[tuple[int, int], tuple[float, float]] = {
    (2, 0): (-1.08263e-3, 0.0),           # -J2
    (2, 1): (-2.414e-10, 1.543e-9),
    (2, 2): (1.5745e-6, -9.0386e-7),
    (3, 0): (2.53241e-6, 0.0),            # -J3 (note sign: C30 = -J3)
    (3, 1): (2.1928e-6, 2.6801e-7),
    (3, 2): (3.0900e-7, -2.1140e-7),
    (3, 3): (1.0055e-7, 1.9720e-7),
    (4, 0): (1.6199e-6, 0.0),             # -J4
    (4, 1): (-5.0872e-7, -4.4945e-7),
    (4, 2): (7.8412e-8, 1.4817e-7),
    (4, 3): (5.9215e-8, -1.2010e-8),
    (4, 4): (-3.9824e-9, 6.5253e-9),
    (5, 0): (2.2772e-7, 0.0),
    (5, 1): (-5.3195e-8, -9.4997e-8),
    (5, 2): (1.0559e-7, -6.5314e-8),
    (5, 3): (-1.4926e-8, -3.2323e-9),
    (5, 4): (6.8629e-10, -7.1622e-10),
    (5, 5): (3.5274e-10, -5.5373e-10),
    (6, 0): (-5.3964e-7, 0.0),
    (6, 1): (-5.9835e-8, 2.1476e-8),
    (6, 2): (6.5256e-9, 3.2827e-8),
    (6, 3): (1.0061e-8, 7.5855e-10),
    (6, 4): (-1.5780e-9, -2.3854e-9),
    (6, 5): (-3.3044e-11, -5.0586e-10),
    (6, 6): (7.8972e-12, -3.1534e-11),
    (7, 0): (3.5136e-7, 0.0),
    (7, 1): (9.2024e-8, 1.2233e-7),
    (7, 2): (4.2963e-8, 1.1847e-8),
    (7, 3): (-2.3283e-9, -1.0209e-8),
    (7, 4): (-4.3310e-10, 2.5795e-10),
    (7, 5): (-2.3503e-10, 1.2345e-10),
    (7, 6): (1.7920e-12, -4.5648e-11),
    (7, 7): (-1.4490e-12, 6.5210e-12),
    (8, 0): (2.0251e-7, 0.0),
    (8, 1): (2.4563e-8, 5.7461e-8),
    (8, 2): (8.2823e-9, 1.6586e-8),
    (8, 3): (-1.9278e-9, -1.2560e-9),
    (8, 4): (-1.9279e-10, 2.3451e-10),
    (8, 5): (-3.0600e-11, -2.2555e-11),
    (8, 6): (4.6040e-12, -5.1100e-12),
    (8, 7): (-1.6310e-13, 7.0560e-13),
    (8, 8): (-2.1880e-14, 1.9640e-14),
}


def _norm_factor(n: int, m: int) -> float:
    """Normalization factor N_nm for fully normalized Legendre polynomials.

    N_nm = sqrt((2 - delta_{m,0}) * (2n+1) * (n-m)! / (n+m)!)

    Used to convert unnormalized harmonic coefficients to normalized form.
    """
    delta = 1 if m == 0 else 0
    # Compute (n-m)! / (n+m)! as product to avoid large factorials
    ratio = 1.0
    for k in range(n - m + 1, n + m + 1):
        ratio /= k
    return float(np.sqrt((2 - delta) * (2 * n + 1) * ratio))


def _gmst_rad(epoch: datetime) -> float:
    """Greenwich Mean Sidereal Time in radians.

    IAU formula: GMST(°) = 280.46061837 + 360.98564736629 * (JD - 2451545.0)
    """
    from humeris.domain.coordinate_frames import gmst_rad as _gmst
    return _gmst(epoch)


class SphericalHarmonicGravity:
    """Spherical harmonic gravity acceleration to degree/order N (max 8).

    Implements the Montenbruck & Gill Ch. 3.2 algorithm:
    1. Rotate ECI to ECEF using GMST
    2. Compute fully normalized associated Legendre polynomials via recursion
    3. Sum gravity acceleration in ECEF using normalized coefficients
    4. Rotate back to ECI

    Coefficients are stored unnormalized in _CS_COEFFS and converted to
    fully normalized form at construction time.
    """

    def __init__(self, max_degree: int = 8) -> None:
        if max_degree < 2 or max_degree > 8:
            raise ValueError(f"max_degree must be 2-8, got {max_degree}")
        self._max_degree = max_degree
        # Pre-normalize coefficients: _CS_COEFFS stores unnormalized C_nm,
        # but _legendre computes fully normalized P̄_nm. Convert C_nm to
        # C̄_nm = C_nm / N_nm so that P̄_nm * C̄_nm = P_nm * C_nm.
        self._norm_coeffs: dict[tuple[int, int], tuple[float, float]] = {}
        for (n, m), (c, s) in _CS_COEFFS.items():
            if n <= max_degree:
                nf = _norm_factor(n, m)
                self._norm_coeffs[(n, m)] = (c / nf, s / nf)

    def _legendre(self, n_max: int, sin_lat: float) -> list[list[float]]:
        """Compute normalized associated Legendre polynomials P_nm(sin_lat).

        Uses standard recursion (Montenbruck & Gill Eq. 3.12-3.14).
        Returns P[n][m] for n=0..n_max, m=0..n.
        """
        cos_lat = float(np.sqrt(max(0.0, 1.0 - sin_lat * sin_lat)))

        p: list[list[float]] = [[0.0] * (i + 1) for i in range(n_max + 1)]
        p[0][0] = 1.0
        if n_max >= 1:
            sqrt3 = float(np.sqrt(3.0))
            p[1][0] = sin_lat * sqrt3
            p[1][1] = cos_lat * sqrt3

        for n in range(2, n_max + 1):
            for m in range(n + 1):
                if m == n:
                    p[n][m] = cos_lat * float(np.sqrt((2.0 * n + 1.0) / (2.0 * n))) * p[n - 1][n - 1]
                elif m == n - 1:
                    p[n][m] = sin_lat * float(np.sqrt(2.0 * n + 1.0)) * p[n - 1][m]
                else:
                    a = float(np.sqrt((2.0 * n + 1.0) * (2.0 * n - 1.0) / ((n - m) * (n + m))))
                    b = float(np.sqrt((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0)
                                  / ((2.0 * n - 3.0) * (n - m) * (n + m))))
                    p[n][m] = a * sin_lat * p[n - 1][m] - b * p[n - 2][m]

        return p

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Compute gravitational acceleration from spherical harmonics."""
        pos_eci = np.array(position)
        r = float(np.linalg.norm(pos_eci))
        if r < 1e-3:
            return (0.0, 0.0, 0.0)

        x_eci, y_eci, z_eci = position

        # Rotate ECI to ECEF
        gmst = _gmst_rad(epoch)
        cos_g = float(np.cos(gmst))
        sin_g = float(np.sin(gmst))
        x_ecef = cos_g * x_eci + sin_g * y_eci
        y_ecef = -sin_g * x_eci + cos_g * y_eci
        z_ecef = z_eci

        # Spherical coordinates
        r_xy = float(np.sqrt(x_ecef**2 + y_ecef**2))
        sin_lat = z_ecef / r
        cos_lat = r_xy / r
        if r_xy > 1e-10:
            lon = float(np.arctan2(y_ecef, x_ecef))
        else:
            lon = 0.0

        # Legendre polynomials
        p = self._legendre(self._max_degree, sin_lat)

        mu = OrbitalConstants.MU_EARTH
        re = OrbitalConstants.R_EARTH_EQUATORIAL

        # Acceleration components in local (r, lat, lon) frame
        ar = -mu / (r * r)  # Central term
        alat = 0.0
        alon = 0.0

        re_r = re / r
        re_r_power = re_r  # Start at (Re/r)^1

        for n in range(2, self._max_degree + 1):
            re_r_power *= re_r  # (Re/r)^n incrementally

            for m in range(n + 1):
                cnm, snm = self._norm_coeffs.get((n, m), (0.0, 0.0))

                cos_m_lon = float(np.cos(m * lon))
                sin_m_lon = float(np.sin(m * lon))

                cs_term = cnm * cos_m_lon + snm * sin_m_lon

                # Radial component
                ar += -mu / (r * r) * re_r_power * (n + 1) * p[n][m] * cs_term

                # Latitudinal component
                # dP_nm/d(lat) = P_{n,m+1} * sqrt((n-m)*(n+m+1)) - m*tan(lat)*P_nm
                # For m < n:
                if m < n:
                    dp = p[n][m + 1] * float(np.sqrt((n - m) * (n + m + 1.0)))
                else:
                    dp = 0.0
                if cos_lat > 1e-12:
                    dp -= m * (sin_lat / cos_lat) * p[n][m]
                alat += mu / (r * r) * re_r_power * dp * cs_term

                # Longitudinal component
                sc_term = -cnm * sin_m_lon + snm * cos_m_lon
                if cos_lat > 1e-12:
                    alon += mu / (r * r) * re_r_power * m * p[n][m] * sc_term / cos_lat

        # Convert local to ECEF Cartesian
        cos_lon = float(np.cos(lon))
        sin_lon = float(np.sin(lon))

        # Unit vectors in ECEF
        # r_hat = (cos_lat*cos_lon, cos_lat*sin_lon, sin_lat)
        # lat_hat = (-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat)
        # lon_hat = (-sin_lon, cos_lon, 0)
        ax_ecef = (ar * cos_lat * cos_lon
                    + alat * (-sin_lat * cos_lon)
                    + alon * (-sin_lon))
        ay_ecef = (ar * cos_lat * sin_lon
                    + alat * (-sin_lat * sin_lon)
                    + alon * cos_lon)
        az_ecef = ar * sin_lat + alat * cos_lat

        # Rotate ECEF back to ECI
        ax_eci = cos_g * ax_ecef - sin_g * ay_ecef
        ay_eci = sin_g * ax_ecef + cos_g * ay_ecef
        az_eci = az_ecef

        return (ax_eci, ay_eci, az_eci)


class AtmosphericDragForce:
    """Atmospheric drag acceleration with co-rotating atmosphere.

    a = -0.5 * rho * Cd * (A/m) * |v_rel| * v_rel
    where v_rel accounts for atmosphere co-rotation.
    """

    def __init__(self, drag_config: DragConfig) -> None:
        self._config = drag_config

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = position
        vx, vy, vz = velocity

        # Altitude check
        r = float(np.linalg.norm(position))
        alt_km = (r - OrbitalConstants.R_EARTH_EQUATORIAL) / 1000.0

        # Return zero if outside atmosphere table range
        try:
            rho = atmospheric_density(alt_km)
        except ValueError:
            return (0.0, 0.0, 0.0)

        # Relative velocity (atmosphere co-rotates with Earth)
        omega_e = OrbitalConstants.EARTH_ROTATION_RATE
        vr = np.array([vx + omega_e * y, vy - omega_e * x, vz])

        v_rel = float(np.linalg.norm(vr))
        if v_rel < 1e-10:
            return (0.0, 0.0, 0.0)

        bc = self._config.ballistic_coefficient
        coeff = -0.5 * rho * bc * v_rel

        a = coeff * vr
        return (float(a[0]), float(a[1]), float(a[2]))


class SolarRadiationPressureForce:
    """Solar radiation pressure (cannonball model, no shadow).

    a = P_sr * Cr * (A/m) * (AU/|d|)^2 * d_hat
    where d = r_sat - r_sun.
    """

    _P_SR: float = 4.56e-6  # N/m² — solar radiation pressure at 1 AU

    def __init__(
        self,
        cr: float,
        area_m2: float,
        mass_kg: float,
        include_shadow: bool = False,
    ) -> None:
        if mass_kg <= 0:
            raise ValueError(f"mass_kg must be positive, got {mass_kg}")
        self._cr = cr
        self._area_m2 = area_m2
        self._mass_kg = mass_kg
        self._include_shadow = include_shadow

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        sun = sun_position_eci(epoch)
        sx, sy, sz = sun.position_eci_m

        if self._include_shadow:
            eclipse = is_eclipsed(position, (sx, sy, sz))
            if eclipse == EclipseType.UMBRA:
                return (0.0, 0.0, 0.0)
            if eclipse == EclipseType.PENUMBRA:
                shadow_factor = 0.5
            else:
                shadow_factor = 1.0
        else:
            shadow_factor = 1.0

        # d = r_sat - r_sun (from Sun toward satellite)
        d = np.array(position) - np.array([sx, sy, sz])

        d_mag = float(np.linalg.norm(d))
        if d_mag < 1e-10:
            return (0.0, 0.0, 0.0)

        am_ratio = self._area_m2 / self._mass_kg
        au_ratio_sq = (AU_METERS / d_mag) ** 2
        coeff = self._P_SR * self._cr * am_ratio * au_ratio_sq / d_mag * shadow_factor

        a = coeff * d
        return (float(a[0]), float(a[1]), float(a[2]))


# --- RK4 integrator ---

def rk4_step(
    t_s: float,
    state: tuple[float, ...],
    h: float,
    deriv_fn: Callable[[float, tuple[float, ...]], tuple[float, ...]],
) -> tuple[float, tuple[float, ...]]:
    """Single 4th-order Runge-Kutta integration step.

    Args:
        t_s: Current time (seconds).
        state: Current state vector.
        h: Step size (seconds).
        deriv_fn: Derivative function f(t, state) -> d(state)/dt.

    Returns:
        (t_new, state_new)
    """
    sv = np.array(state)
    k1 = np.array(deriv_fn(t_s, state))
    s1 = tuple((sv + 0.5 * h * k1).tolist())

    k2 = np.array(deriv_fn(t_s + 0.5 * h, s1))
    s2 = tuple((sv + 0.5 * h * k2).tolist())

    k3 = np.array(deriv_fn(t_s + 0.5 * h, s2))
    s3 = tuple((sv + h * k3).tolist())

    k4 = np.array(deriv_fn(t_s + h, s3))

    state_new_arr = sv + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    state_new = tuple(float(x) for x in state_new_arr)
    return (t_s + h, state_new)


# --- Symplectic integrators ---

def stormer_verlet_step(
    t_s: float,
    state: tuple[float, ...],
    h: float,
    accel_fn: Callable[[float, tuple[float, float, float], tuple[float, float, float]], tuple[float, float, float]],
) -> tuple[float, tuple[float, ...]]:
    """Single Stormer-Verlet (leapfrog) integration step (2nd order symplectic).

    Args:
        t_s: Current time (seconds).
        state: Current state vector (x, y, z, vx, vy, vz).
        h: Step size (seconds).
        accel_fn: Acceleration function a(t, pos, vel) -> (ax, ay, az).

    Returns:
        (t_new, state_new)
    """
    pos_arr = np.array([state[0], state[1], state[2]])
    vel_arr = np.array([state[3], state[4], state[5]])

    pos = (state[0], state[1], state[2])
    vel = (state[3], state[4], state[5])

    # Half-step velocity
    acc = np.array(accel_fn(t_s, pos, vel))
    v_half_arr = vel_arr + 0.5 * h * acc
    v_half = (float(v_half_arr[0]), float(v_half_arr[1]), float(v_half_arr[2]))

    # Full-step position
    pos_new_arr = pos_arr + h * v_half_arr
    pos_new = (float(pos_new_arr[0]), float(pos_new_arr[1]), float(pos_new_arr[2]))

    # Acceleration at new position
    acc_new = np.array(accel_fn(t_s + h, pos_new, v_half))

    # Complete velocity step
    vel_new_arr = v_half_arr + 0.5 * h * acc_new
    vel_new = (float(vel_new_arr[0]), float(vel_new_arr[1]), float(vel_new_arr[2]))

    return (t_s + h, pos_new + vel_new)


def yoshida4_step(
    t_s: float,
    state: tuple[float, ...],
    h: float,
    accel_fn: Callable[[float, tuple[float, float, float], tuple[float, float, float]], tuple[float, float, float]],
) -> tuple[float, tuple[float, ...]]:
    """Single 4th-order Yoshida integration step (symplectic).

    Composition of 3 Stormer-Verlet sub-steps with Yoshida coefficients.

    Args:
        t_s: Current time (seconds).
        state: Current state vector (x, y, z, vx, vy, vz).
        h: Step size (seconds).
        accel_fn: Acceleration function a(t, pos, vel) -> (ax, ay, az).

    Returns:
        (t_new, state_new)
    """
    # Yoshida 4th-order coefficients
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)

    d = (w1, w0, w1)
    c_half = (w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0)

    pos = [state[0], state[1], state[2]]
    vel = [state[3], state[4], state[5]]
    t = t_s

    # c[0] position kick
    for k in range(3):
        pos[k] += c_half[0] * h * vel[k]
    t += c_half[0] * h

    for i in range(3):
        # d[i] velocity kick
        acc = accel_fn(t, (pos[0], pos[1], pos[2]), (vel[0], vel[1], vel[2]))
        for k in range(3):
            vel[k] += d[i] * h * acc[k]

        # c[i+1] position kick
        for k in range(3):
            pos[k] += c_half[i + 1] * h * vel[k]
        t = t_s + sum(c_half[:i + 2]) * h

    return (t_s + h, (pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]))


# --- Main propagation function ---

def propagate_numerical(
    initial_state: "OrbitalState",
    duration: timedelta,
    step: timedelta,
    force_models: list[ForceModel],
    epoch: datetime | None = None,
    integrator: str = "rk4",
) -> NumericalPropagationResult:
    """Numerical integration with summed force model accelerations.

    1. Convert OrbitalState -> Cartesian ECI via kepler_to_cartesian
    2. Build derivative/acceleration function from force models
    3. Step through time using chosen integrator
    4. Returns NumericalPropagationResult

    Args:
        initial_state: OrbitalState from derive_orbital_state.
        duration: Total propagation duration.
        step: Integration time step.
        force_models: List of force models to sum.
        epoch: Override epoch (defaults to initial_state.reference_epoch).
        integrator: Integration method — "rk4", "verlet", "yoshida", "dormand_prince", or "rk89".
    """
    # Import here to avoid circular import at module level
    from humeris.domain.propagation import OrbitalState as _OS

    if integrator not in ("rk4", "verlet", "yoshida", "dormand_prince", "rk89"):
        raise ValueError(f"Unknown integrator: {integrator!r}. Use 'rk4', 'verlet', 'yoshida', 'dormand_prince', or 'rk89'.")

    if integrator == "dormand_prince":
        from humeris.domain.adaptive_integration import (
            propagate_adaptive,
            AdaptiveStepConfig,
        )
        ref_epoch_dp = epoch if epoch is not None else initial_state.reference_epoch
        dp_result = propagate_adaptive(
            initial_state=initial_state,
            duration=duration,
            force_models=force_models,
            epoch=ref_epoch_dp,
            output_step_s=step.total_seconds(),
        )
        dp_steps = dp_result.steps
        if dp_steps:
            dp_positions = np.array([(s.position_eci[0], s.position_eci[1], s.position_eci[2]) for s in dp_steps])
            dp_velocities = np.array([(s.velocity_eci[0], s.velocity_eci[1], s.velocity_eci[2]) for s in dp_steps])
            dp_r = np.sqrt(np.sum(dp_positions**2, axis=1))
            dp_v = np.sqrt(np.sum(dp_velocities**2, axis=1))
            dp_energies = 0.5 * dp_v**2 - OrbitalConstants.MU_EARTH / dp_r

            dp_steps_with_energy = tuple(
                PropagationStep(
                    time=s.time,
                    position_eci=s.position_eci,
                    velocity_eci=s.velocity_eci,
                    specific_energy_j_kg=float(dp_energies[i]),
                )
                for i, s in enumerate(dp_steps)
            )
            dp_initial_energy = float(dp_energies[0])
            dp_final_energy = float(dp_energies[-1])
            dp_drift = np.abs(dp_energies - dp_initial_energy)
            dp_max_drift = float(np.max(dp_drift))
            dp_relative_drift = abs(dp_max_drift / dp_initial_energy) if dp_initial_energy != 0.0 else 0.0
        else:
            dp_steps_with_energy = dp_steps
            dp_initial_energy = 0.0
            dp_final_energy = 0.0
            dp_max_drift = 0.0
            dp_relative_drift = 0.0

        return NumericalPropagationResult(
            steps=dp_steps_with_energy,
            epoch=dp_result.epoch,
            duration_s=dp_result.duration_s,
            force_model_names=dp_result.force_model_names,
            initial_energy_j_kg=dp_initial_energy,
            final_energy_j_kg=dp_final_energy,
            max_energy_drift_j_kg=dp_max_drift,
            relative_energy_drift=dp_relative_drift,
        )

    if integrator == "rk89":
        from humeris.domain.adaptive_integration import (
            propagate_rk89_adaptive,
            AdaptiveStepConfig,
        )
        ref_epoch_rk89 = epoch if epoch is not None else initial_state.reference_epoch
        rk89_result = propagate_rk89_adaptive(
            initial_state=initial_state,
            duration=duration,
            force_models=force_models,
            epoch=ref_epoch_rk89,
            output_step_s=step.total_seconds(),
        )
        rk89_steps = rk89_result.steps
        if rk89_steps:
            rk89_positions = np.array([(s.position_eci[0], s.position_eci[1], s.position_eci[2]) for s in rk89_steps])
            rk89_velocities = np.array([(s.velocity_eci[0], s.velocity_eci[1], s.velocity_eci[2]) for s in rk89_steps])
            rk89_r = np.sqrt(np.sum(rk89_positions**2, axis=1))
            rk89_v = np.sqrt(np.sum(rk89_velocities**2, axis=1))
            rk89_energies = 0.5 * rk89_v**2 - OrbitalConstants.MU_EARTH / rk89_r

            rk89_steps_with_energy = tuple(
                PropagationStep(
                    time=s.time,
                    position_eci=s.position_eci,
                    velocity_eci=s.velocity_eci,
                    specific_energy_j_kg=float(rk89_energies[i]),
                )
                for i, s in enumerate(rk89_steps)
            )
            rk89_initial_energy = float(rk89_energies[0])
            rk89_final_energy = float(rk89_energies[-1])
            rk89_drift = np.abs(rk89_energies - rk89_initial_energy)
            rk89_max_drift = float(np.max(rk89_drift))
            rk89_relative_drift = abs(rk89_max_drift / rk89_initial_energy) if rk89_initial_energy != 0.0 else 0.0
        else:
            rk89_steps_with_energy = rk89_steps
            rk89_initial_energy = 0.0
            rk89_final_energy = 0.0
            rk89_max_drift = 0.0
            rk89_relative_drift = 0.0

        return NumericalPropagationResult(
            steps=rk89_steps_with_energy,
            epoch=rk89_result.epoch,
            duration_s=rk89_result.duration_s,
            force_model_names=rk89_result.force_model_names,
            initial_energy_j_kg=rk89_initial_energy,
            final_energy_j_kg=rk89_final_energy,
            max_energy_drift_j_kg=rk89_max_drift,
            relative_energy_drift=rk89_relative_drift,
        )

    ref_epoch = epoch if epoch is not None else initial_state.reference_epoch
    duration_s = duration.total_seconds()
    step_s = step.total_seconds()

    # Convert to Cartesian ECI
    pos_list, vel_list = kepler_to_cartesian(
        a=initial_state.semi_major_axis_m,
        e=initial_state.eccentricity,
        i_rad=initial_state.inclination_rad,
        omega_big_rad=initial_state.raan_rad,
        omega_small_rad=initial_state.arg_perigee_rad,
        nu_rad=initial_state.true_anomaly_rad,
    )
    pos = (pos_list[0], pos_list[1], pos_list[2])
    vel = (vel_list[0], vel_list[1], vel_list[2])

    # State vector: (x, y, z, vx, vy, vz)
    state_vec: tuple[float, ...] = pos + vel

    # Force model names for result
    model_names = tuple(type(fm).__name__ for fm in force_models)

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

    def accel_fn(
        t_s: float,
        p: tuple[float, float, float],
        v: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        current_epoch = ref_epoch + timedelta(seconds=t_s)
        ax_total, ay_total, az_total = 0.0, 0.0, 0.0
        for fm in force_models:
            ax, ay, az = fm.acceleration(current_epoch, p, v)
            ax_total += ax
            ay_total += ay
            az_total += az
        return (ax_total, ay_total, az_total)

    # Select stepper
    if integrator == "rk4":
        def step_fn(t: float, sv: tuple[float, ...], h: float) -> tuple[float, tuple[float, ...]]:
            return rk4_step(t, sv, h, deriv_fn)
    elif integrator == "verlet":
        def step_fn(t: float, sv: tuple[float, ...], h: float) -> tuple[float, tuple[float, ...]]:
            return stormer_verlet_step(t, sv, h, accel_fn)
    else:  # yoshida
        def step_fn(t: float, sv: tuple[float, ...], h: float) -> tuple[float, tuple[float, ...]]:
            return yoshida4_step(t, sv, h, accel_fn)

    # Collect steps
    steps: list[PropagationStep] = []
    t_current = 0.0
    num_steps = int(duration_s / step_s) + 1

    for i in range(num_steps):
        current_time = ref_epoch + timedelta(seconds=t_current)
        p = (state_vec[0], state_vec[1], state_vec[2])
        v = (state_vec[3], state_vec[4], state_vec[5])
        steps.append(PropagationStep(time=current_time, position_eci=p, velocity_eci=v))

        if i < num_steps - 1:
            t_current, state_vec = step_fn(t_current, state_vec, step_s)

    # Compute energy for each step using numpy for efficiency
    positions = np.array([(s.position_eci[0], s.position_eci[1], s.position_eci[2]) for s in steps])
    velocities = np.array([(s.velocity_eci[0], s.velocity_eci[1], s.velocity_eci[2]) for s in steps])
    r_magnitudes = np.sqrt(np.sum(positions**2, axis=1))
    v_magnitudes = np.sqrt(np.sum(velocities**2, axis=1))
    energies = 0.5 * v_magnitudes**2 - OrbitalConstants.MU_EARTH / r_magnitudes

    # Rebuild steps with energy populated
    steps_with_energy: list[PropagationStep] = []
    for i, s in enumerate(steps):
        steps_with_energy.append(PropagationStep(
            time=s.time,
            position_eci=s.position_eci,
            velocity_eci=s.velocity_eci,
            specific_energy_j_kg=float(energies[i]),
        ))

    initial_energy = float(energies[0])
    final_energy = float(energies[-1])
    drift_from_initial = np.abs(energies - initial_energy)
    max_drift = float(np.max(drift_from_initial))
    relative_drift = abs(max_drift / initial_energy) if initial_energy != 0.0 else 0.0

    return NumericalPropagationResult(
        steps=tuple(steps_with_energy),
        epoch=ref_epoch,
        duration_s=duration_s,
        force_model_names=model_names,
        initial_energy_j_kg=initial_energy,
        final_energy_j_kg=final_energy,
        max_energy_drift_j_kg=max_drift,
        relative_energy_drift=relative_drift,
    )


# ── Jacobi Metric for Geodesic Propagation Stability (P5) ─────────

@dataclass(frozen=True)
class JacobiStability:
    """Riemannian curvature diagnostic on the Jacobi-Maupertuis energy surface.

    The Jacobi metric g_J = 2(E - V) * g_euclidean reformulates Keplerian
    dynamics as geodesic flow on a Riemannian manifold. The scalar curvature
    of this metric provides a coordinate-independent instantaneous stability
    diagnostic:
        - scalar_curvature > 0: stable (nearby geodesics converge, bound orbit)
        - scalar_curvature < 0: unstable (nearby geodesics diverge, hyperbolic)
        - scalar_curvature ~ 0: marginally stable (parabolic)

    References:
        Maupertuis (1744). Principle of least action.
        Jacobi (1837). Note on the geodesic interpretation of dynamics.
        Pin (1975). Curvature and mechanics. Advances in Mathematics.
    """
    scalar_curvature: float      # Sectional curvature of Jacobi metric
    stability_index: float       # Normalized stability: curvature * r^2 (dimensionless)
    is_stable: bool              # True if scalar_curvature > 0
    jacobi_metric_factor: float  # 2(E - V), the conformal factor
    specific_energy_j_kg: float  # Total specific energy E = T + V


def compute_jacobi_stability(
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    mu: float = OrbitalConstants.MU_EARTH,
) -> JacobiStability:
    """Compute Riemannian curvature on the Jacobi-Maupertuis energy surface.

    For the Kepler potential V = -mu/r, the Jacobi metric is:
        g_J = 2(E + mu/r) * (dx^2 + dy^2 + dz^2)

    The sectional curvature of a conformally flat metric g_J = Omega^2 * g_flat
    in 3D is (Pin 1975):
        K = (1/Omega^2) * [ -Delta(ln Omega) + |grad(ln Omega)|^2 ]

    For Omega^2 = 2(E + mu/r):
        ln(Omega) = (1/2) * ln(2(E + mu/r))
        grad(ln Omega) = (mu / (2r^3 * (E + mu/r))) * r_vec
        Delta(ln Omega) = mu*(2E*r - mu) / (2*r^3*(E + mu/r))^2 ... (simplified)

    The scalar curvature in 3D for conformal factor f = 2(E+mu/r) reduces to:
        K = -(1/f) * [ nabla^2(f)/(2*f) - 3|grad(f)|^2/(4*f^2) ]

    For Kepler: f = 2E + 2mu/r
        grad(f) = -2*mu/r^3 * r_vec
        |grad(f)|^2 = 4*mu^2/r^4
        nabla^2(f) = -2*mu * nabla^2(1/r) + 2*mu * 2/r^3 = 4*mu/r^3
        (since nabla^2(1/r) = 0 for r > 0)

    Therefore:
        K = -(1/f) * [ 4*mu/(2*f*r^3) - 3*4*mu^2/(4*f^2*r^4) ]
          = -(1/f) * [ 2*mu/(f*r^3) - 3*mu^2/(f^2*r^4) ]

    Args:
        position: ECI position (x, y, z) in meters.
        velocity: ECI velocity (vx, vy, vz) in m/s.
        mu: Gravitational parameter (m^3/s^2).

    Returns:
        JacobiStability with curvature diagnostic.
    """
    r_vec = np.array(position)
    v_vec = np.array(velocity)
    r = float(np.linalg.norm(r_vec))
    v = float(np.linalg.norm(v_vec))

    if r < 1.0:
        # Degenerate: at or near singularity
        return JacobiStability(
            scalar_curvature=0.0,
            stability_index=0.0,
            is_stable=False,
            jacobi_metric_factor=0.0,
            specific_energy_j_kg=0.0,
        )

    # Specific energy: E = v^2/2 - mu/r
    kinetic = 0.5 * v * v
    potential = -mu / r
    energy = kinetic + potential

    # Jacobi metric conformal factor: f = 2(E - V) = 2(E + mu/r) = 2*T = v^2
    # Equivalently: f = 2 * kinetic energy (must be positive on allowed region)
    f = 2.0 * (energy + mu / r)  # = 2 * kinetic = v^2

    if f <= 0.0:
        # Outside classically allowed region (shouldn't happen for real orbits)
        return JacobiStability(
            scalar_curvature=0.0,
            stability_index=0.0,
            is_stable=False,
            jacobi_metric_factor=f,
            specific_energy_j_kg=energy,
        )

    # For Kepler potential V = -mu/r in 3D:
    # Conformal factor f = 2(E + mu/r)
    # grad(f) = -2*mu*r_hat/r^2  =>  |grad(f)|^2 = 4*mu^2/r^4
    # Laplacian(f) = -2*mu * Laplacian(1/r) + 0 = 0 for r > 0
    #   BUT: Laplacian(1/r) = 0 only for r != 0, AND
    #   f = 2E + 2mu/r => Laplacian(f) = 2mu * Laplacian(1/r) = 0
    #   grad(f) = -2mu/r^3 * r_vec => div(grad(f)) = -2mu * div(r_vec/r^3)
    #   div(r_vec/r^3) = 3/r^3 - 3r^2/r^5 = 0 for r > 0. Confirmed.
    #
    # So Laplacian(f) = 0 and the curvature formula simplifies.
    #
    # For 3D conformal metric g = f * g_flat (note: g_J = f * g_flat):
    # Scalar curvature R = (1/f) * [ -2*(n-1)*Lap(f)/(f) + (n-1)(n-4)*|grad f|^2/f^2 ]
    #   where n = 3 (dimension).
    #
    # More precisely, for g_ij = phi * delta_ij in n dimensions:
    #   R_scalar = (1/phi) * [ -(n-1)(n-2)|grad(phi)|^2/(4*phi^2) - (n-1)*Lap(phi)/phi ]
    #            + correction terms...
    #
    # Standard formula for conformal factor phi (g = phi * g_flat) in n=3:
    #   R = -(1/phi) * [ 2 * Lap(phi)/phi - |grad(phi)|^2/phi^2 ]
    #   (from Pin 1975, with phi = f)
    #
    # With Laplacian(f) = 0:
    #   R = -(1/f) * [ 0 - |grad(f)|^2/f^2 ]
    #     = |grad(f)|^2 / f^3
    #
    # |grad(f)|^2 = 4*mu^2/r^4
    # f = 2(E + mu/r) = v^2
    #
    # R = 4*mu^2 / (r^4 * f^3)

    grad_f_sq = 4.0 * mu * mu / (r ** 4)

    # Scalar curvature: R = |grad(f)|^2 / f^3
    scalar_curvature = grad_f_sq / (f ** 3)

    # For bound orbits (E < 0), f = v^2 > 0 always, and curvature > 0 (stable).
    # For unbound (E > 0), still positive but smaller (less focusing).
    # The sign is always positive for Kepler — instability comes from
    # perturbations that make the effective curvature negative.
    # To capture this, we use a signed stability index that becomes
    # negative for hyperbolic excess energy:
    # stability_index = sign(binding) * |K| * r^2
    # where binding energy is -E (positive for bound orbits)

    stability_index = scalar_curvature * r * r
    if energy > 0.0:
        # Unbound orbit: negate to signal instability
        stability_index = -stability_index

    is_stable = energy < 0.0  # Bound orbit = stable geodesic

    return JacobiStability(
        scalar_curvature=scalar_curvature,
        stability_index=stability_index,
        is_stable=is_stable,
        jacobi_metric_factor=f,
        specific_energy_j_kg=energy,
    )


# ── Shadowing Lemma for Propagation Validation (P31) ──────────────

@dataclass(frozen=True)
class ShadowingValidation:
    """Assessment of whether a numerical trajectory shadows a true orbit.

    The shadowing lemma guarantees that near a hyperbolic trajectory,
    every pseudo-orbit (numerical trajectory with bounded per-step error)
    is shadowed by a true orbit within some distance.

    The shadowing distance grows exponentially with the Lyapunov exponent:
        d_shadow = max_k (delta_k * exp(lambda * k * h))
    where delta_k is the per-step truncation error and lambda is the
    maximum Lyapunov exponent.

    References:
        Bowen (1975). Omega-limit sets for Axiom A diffeomorphisms.
        Palmer (2000). Shadowing in Dynamical Systems.
        Hayes & Jackson (2005). A survey of shadowing methods.
    """
    shadowing_distance_m: float      # Estimated max shadowing distance
    per_step_error_m: float          # Typical per-step truncation error
    max_lyapunov_exponent: float     # Max Lyapunov exponent (1/s)
    propagation_horizon_s: float     # Time until shadowing_distance > tolerance
    is_reliable: bool                # True if shadowing_distance < tolerance
    reliability_margin: float        # tolerance / shadowing_distance (>1 = reliable)


def compute_shadowing_validation(
    steps: list[PropagationStep],
    force_models: list[ForceModel],
    step_size_s: float,
    tolerance_m: float = 1000.0,
    perturbation_m: float = 1.0,
) -> ShadowingValidation:
    """Validate a numerical propagation via the shadowing lemma.

    Estimates per-step truncation error via step doubling and the
    maximum Lyapunov exponent via STM approximation (finite differences).

    The shadowing distance bounds how far the numerical trajectory
    can drift from a true orbit:
        d_shadow = max_k (delta_k * exp(lambda * k * h))

    Args:
        steps: Propagation steps from propagate_numerical.
        force_models: Force models used in the propagation.
        step_size_s: Integration step size in seconds.
        tolerance_m: Position accuracy requirement in meters.
        perturbation_m: Perturbation magnitude for FTLE estimation.

    Returns:
        ShadowingValidation with reliability assessment.
    """
    if len(steps) < 3:
        return ShadowingValidation(
            shadowing_distance_m=0.0,
            per_step_error_m=0.0,
            max_lyapunov_exponent=0.0,
            propagation_horizon_s=float('inf'),
            is_reliable=True,
            reliability_margin=float('inf'),
        )

    # ── Estimate per-step truncation error via step doubling ──
    # Compare full step vs two half-steps at sample points
    ref_epoch = steps[0].time

    def accel_fn(
        t_s: float,
        p: tuple[float, float, float],
        v: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        current_epoch = ref_epoch + timedelta(seconds=t_s)
        ax_total, ay_total, az_total = 0.0, 0.0, 0.0
        for fm in force_models:
            ax, ay, az = fm.acceleration(current_epoch, p, v)
            ax_total += ax
            ay_total += ay
            az_total += az
        return (ax_total, ay_total, az_total)

    def deriv_fn(t_s: float, sv: tuple[float, ...]) -> tuple[float, ...]:
        p = (sv[0], sv[1], sv[2])
        v = (sv[3], sv[4], sv[5])
        ax, ay, az = accel_fn(t_s, p, v)
        return (v[0], v[1], v[2], ax, ay, az)

    # Sample a few points for error estimation (up to 10 evenly spaced)
    n_samples = min(10, len(steps) - 1)
    sample_indices = [int(i * (len(steps) - 1) / n_samples) for i in range(n_samples)]

    per_step_errors: list[float] = []
    for idx in sample_indices:
        s = steps[idx]
        state = s.position_eci + s.velocity_eci
        t = 0.0

        # One full step
        _, state_full = rk4_step(t, state, step_size_s, deriv_fn)

        # Two half steps
        _, state_half1 = rk4_step(t, state, step_size_s / 2.0, deriv_fn)
        _, state_half2 = rk4_step(t + step_size_s / 2.0, state_half1, step_size_s / 2.0, deriv_fn)

        # Position error estimate (Richardson extrapolation factor for RK4: 2^4 - 1 = 15)
        pos_full = np.array(state_full[:3])
        pos_half = np.array(state_half2[:3])
        error = float(np.linalg.norm(pos_half - pos_full)) / 15.0
        per_step_errors.append(error)

    avg_step_error = float(np.mean(per_step_errors)) if per_step_errors else 0.0

    # ── Estimate maximum Lyapunov exponent via STM finite differences ──
    # Perturb initial position in 3 directions, propagate a short window,
    # build 3x3 sensitivity matrix, SVD for max singular value
    s0 = steps[0]
    pos0 = np.array(s0.position_eci)
    vel0 = np.array(s0.velocity_eci)

    # Use a window of min(10 steps, total steps)
    window_steps = min(10, len(steps) - 1)
    window_s = window_steps * step_size_s

    # Nominal propagation over window
    state_nom = s0.position_eci + s0.velocity_eci
    t = 0.0
    for _ in range(window_steps):
        t, state_nom = rk4_step(t, state_nom, step_size_s, deriv_fn)
    pos_nom_end = np.array(state_nom[:3])

    # Build 3x3 position sensitivity matrix
    phi = np.zeros((3, 3))
    for axis in range(3):
        perturbed_pos = list(s0.position_eci)
        perturbed_pos[axis] += perturbation_m

        state_pert = tuple(perturbed_pos) + s0.velocity_eci
        t = 0.0
        for _ in range(window_steps):
            t, state_pert = rk4_step(t, state_pert, step_size_s, deriv_fn)
        pos_pert_end = np.array(state_pert[:3])

        phi[:, axis] = (pos_pert_end - pos_nom_end) / perturbation_m

    # SVD for max singular value
    _, singular_values, _ = np.linalg.svd(phi)
    sigma_max = float(singular_values[0])
    sigma_max = max(sigma_max, 1.0)

    # FTLE = ln(sigma_max) / window_s
    ftle = math.log(sigma_max) / window_s if window_s > 0 else 0.0
    ftle = max(ftle, 0.0)

    # ── Shadowing distance ──
    # d_shadow = max_k (delta_k * exp(ftle * k * h))
    # With constant delta_k = avg_step_error, the maximum is at k = N (last step):
    n_total_steps = len(steps) - 1
    total_time = n_total_steps * step_size_s

    if ftle > 1e-15 and avg_step_error > 0:
        # Geometric sum: sum_{k=0}^{N} delta * exp(ftle * k * h)
        # Maximum term is at k = N
        shadowing_distance = avg_step_error * math.exp(ftle * total_time)
        # But cap at a reasonable value to avoid overflow
        shadowing_distance = min(shadowing_distance, 1e15)
    else:
        # Linear error growth
        shadowing_distance = avg_step_error * n_total_steps

    # ── Propagation horizon ──
    # Time when shadowing_distance exceeds tolerance:
    # delta * exp(ftle * t) = tolerance
    # t = ln(tolerance / delta) / ftle
    if ftle > 1e-15 and avg_step_error > 1e-30:
        ratio = tolerance_m / avg_step_error
        if ratio > 1.0:
            propagation_horizon = math.log(ratio) / ftle
        else:
            propagation_horizon = 0.0
    else:
        propagation_horizon = float('inf')

    is_reliable = shadowing_distance < tolerance_m
    if shadowing_distance > 0:
        reliability_margin = tolerance_m / shadowing_distance
    else:
        reliability_margin = float('inf')

    return ShadowingValidation(
        shadowing_distance_m=shadowing_distance,
        per_step_error_m=avg_step_error,
        max_lyapunov_exponent=ftle,
        propagation_horizon_s=propagation_horizon,
        is_reliable=is_reliable,
        reliability_margin=reliability_margin,
    )


# ── Melnikov Method for Chaos Onset Detection (P34) ───────────────

@dataclass(frozen=True)
class MelnikovChaosAnalysis:
    """Melnikov method analysis for detecting chaos onset near resonances.

    Near mean-motion resonances (p:q where p*n_sat ~ q*n_Earth),
    the Melnikov function measures the transverse splitting of stable
    and unstable manifolds of the unperturbed separatrix.

    If the Melnikov function has simple zeros, the manifolds intersect
    transversally, implying horseshoe chaos (Smale-Birkhoff theorem).

    The chaotic layer width around the resonance is proportional to
    the Melnikov amplitude divided by the action gradient of the
    unperturbed Hamiltonian.

    References:
        Melnikov (1963). On the stability of the center for
            time-periodic perturbations.
        Guckenheimer & Holmes (1983). Nonlinear Oscillations,
            Dynamical Systems, and Bifurcations of Vector Fields.
        Celletti & Chierchia (2007). KAM stability and celestial mechanics.
    """
    melnikov_amplitude: float      # Max |M(t_0)|
    has_zeros: bool                # Melnikov function has simple zeros
    chaos_width_m: float           # Width of chaotic layer in meters
    onset_altitude_km: float       # Altitude where chaos layer first opens
    resonance_order: tuple[int, int]  # (p, q) resonance


def compute_melnikov_chaos(
    semi_major_axis_m: float,
    eccentricity: float = 0.0,
    max_resonance_order: int = 10,
    damping_rate: float = 0.0,
    j2: float = OrbitalConstants.J2_EARTH,
    mu: float = OrbitalConstants.MU_EARTH,
    r_earth: float = OrbitalConstants.R_EARTH_EQUATORIAL,
) -> MelnikovChaosAnalysis:
    """Detect chaos onset near mean-motion resonances via Melnikov method.

    Identifies the nearest p:q resonance and computes the Melnikov
    integral to determine whether chaotic dynamics exist near the
    separatrix.

    The J2 perturbation provides the forcing term for the Melnikov
    integral. The Melnikov amplitude is:
        M_max = A_J2 * I_integral
    where:
        A_J2 = (3/2) * J2 * mu * R_e^2 / a^3  (J2 perturbation amplitude)
        I_integral = integral |sin(n*t)| * exp(-gamma*t) dt  (0 to T_res)

    For gamma = 0 (no damping): I = 2/n (half-period integral of |sin|)
    For gamma > 0: I = n / (n^2 + gamma^2) * (1 - exp(-gamma*pi/n))

    Args:
        semi_major_axis_m: Orbital semi-major axis in meters.
        eccentricity: Orbital eccentricity.
        max_resonance_order: Maximum p+q to search for resonances.
        damping_rate: Exponential damping rate (1/s), e.g. from drag.
        j2: J2 coefficient.
        mu: Gravitational parameter (m^3/s^2).
        r_earth: Earth equatorial radius (m).

    Returns:
        MelnikovChaosAnalysis with amplitude, chaos width, and onset altitude.
    """
    # Mean motion of satellite
    a = semi_major_axis_m
    if a <= 0:
        return MelnikovChaosAnalysis(
            melnikov_amplitude=0.0,
            has_zeros=False,
            chaos_width_m=0.0,
            onset_altitude_km=0.0,
            resonance_order=(0, 0),
        )

    n_sat = math.sqrt(mu / (a ** 3))

    # Earth's mean motion (sidereal rotation rate is not orbital mean motion;
    # for resonance we compare satellite mean motion with Earth's rotation rate
    # which causes tesseral resonance via the gravity field)
    n_earth = OrbitalConstants.EARTH_ROTATION_RATE

    # Find nearest p:q resonance
    # p * n_sat ≈ q * n_earth
    # => p/q ≈ n_earth/n_sat
    ratio = n_earth / n_sat
    best_p, best_q = 1, 1
    best_err = float('inf')

    for q in range(1, max_resonance_order + 1):
        for p in range(1, max_resonance_order + 1 - q):
            if p + q > max_resonance_order:
                continue
            err = abs(float(p) / float(q) - ratio)
            if err < best_err:
                best_err = err
                best_p = p
                best_q = q

    # Resonance semi-major axis: a_res where p * n(a_res) = q * n_earth
    # n = sqrt(mu/a^3) => a_res = (mu * q^2 / (n_earth^2 * p^2))^(1/3)
    a_res = (mu * best_q ** 2 / (n_earth ** 2 * best_p ** 2)) ** (1.0 / 3.0)

    # J2 perturbation amplitude at the actual orbit
    # A_J2 = (3/2) * J2 * n * (R_e/a)^2
    # This is the dominant forcing term from J2 on the resonant angle
    a_j2 = 1.5 * j2 * n_sat * (r_earth / a) ** 2

    # Separatrix half-width in semi-major axis (from pendulum model of resonance):
    # delta_a_sep = 2 * a * sqrt(A_J2 / n_sat) for the pendulum approximation
    # The resonance strength parameter: epsilon = A_J2 * |q|
    epsilon = a_j2 * abs(best_q)

    # Melnikov integral: I = integral_0^{T} |sin(n_res * t)| * exp(-gamma * t) dt
    # where n_res = |p * n_sat - q * n_earth| (near-resonance frequency)
    n_res = abs(best_p * n_sat - best_q * n_earth)
    if n_res < 1e-15:
        n_res = n_sat * 1e-6  # Avoid division by zero; exact resonance

    if damping_rate < 1e-15:
        # Undamped: I = 2/n_res (integral of |sin| over one half-period)
        melnikov_integral = 2.0 / n_res
    else:
        # Damped: I = n_res / (n_res^2 + gamma^2) * (1 - exp(-gamma * pi / n_res))
        gamma = damping_rate
        melnikov_integral = (n_res / (n_res ** 2 + gamma ** 2)
                             * (1.0 - math.exp(-gamma * math.pi / n_res)))

    # Melnikov amplitude
    melnikov_amplitude = epsilon * melnikov_integral

    # Chaos onset criterion: Melnikov has simple zeros when
    # M_max > 0 (which it always is for non-zero J2)
    # More precisely, for the perturbed pendulum, chaos exists when
    # the perturbation exceeds the separatrix energy:
    # We use the criterion that the Melnikov amplitude exceeds a threshold
    # relative to the unperturbed separatrix size.
    #
    # Separatrix half-width in terms of action I:
    # |dH0/dI| = n_sat (frequency of the unperturbed oscillator)
    #
    # Chaotic layer width: W = 2 * M_max / |dH0/dI|
    dh0_di = n_sat
    chaos_width_action = 2.0 * melnikov_amplitude / dh0_di if dh0_di > 0 else 0.0

    # Convert action width to physical width in meters (semi-major axis)
    # da/dI ≈ 2a/(n*a) = 2/n for Keplerian action I = sqrt(mu*a)
    # More precisely: I = sqrt(mu * a), dI/da = sqrt(mu)/(2*sqrt(a))
    # da = dI / (dI/da) = dI * 2*sqrt(a)/sqrt(mu)
    chaos_width_m = chaos_width_action * 2.0 * math.sqrt(a) / math.sqrt(mu)

    # Has zeros: Melnikov function has zeros if amplitude > 0
    has_zeros = melnikov_amplitude > 1e-20

    # Onset altitude: altitude at the resonant semi-major axis
    onset_altitude_km = (a_res - r_earth) / 1000.0

    return MelnikovChaosAnalysis(
        melnikov_amplitude=melnikov_amplitude,
        has_zeros=has_zeros,
        chaos_width_m=chaos_width_m,
        onset_altitude_km=onset_altitude_km,
        resonance_order=(best_p, best_q),
    )
