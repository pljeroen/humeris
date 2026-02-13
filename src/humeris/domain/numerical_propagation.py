# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Numerical orbit propagation with RK4 and pluggable force models.

4th-order Runge-Kutta integrator with composable perturbation forces.
Handles accumulated perturbation effects (drag, SRP, higher-order
gravitational harmonics) that analytical propagation cannot model.

No external dependencies — only stdlib math/dataclasses/typing/datetime
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


@dataclass(frozen=True)
class NumericalPropagationResult:
    """Complete result of a numerical propagation run."""
    steps: tuple[PropagationStep, ...]
    epoch: datetime
    duration_s: float
    force_model_names: tuple[str, ...]


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
        r7 = r2 * r2 * r2 * r

        mu = OrbitalConstants.MU_EARTH
        j3 = OrbitalConstants.J3_EARTH
        re = OrbitalConstants.R_EARTH_EQUATORIAL

        coeff = mu * j3 * re * re * re / 2.0

        z3_r2 = z * z * z / r2

        ax = coeff * (x / r7) * (35.0 * z3_r2 - 15.0 * z)
        ay = coeff * (y / r7) * (35.0 * z3_r2 - 15.0 * z)
        az = -coeff * (1.0 / r7) * (30.0 * z * z - 35.0 * z * z * z * z / r2 - 3.0 * r2)
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
        integrator: Integration method — "rk4", "verlet", "yoshida", or "dormand_prince".
    """
    # Import here to avoid circular import at module level
    from humeris.domain.propagation import OrbitalState as _OS

    if integrator not in ("rk4", "verlet", "yoshida", "dormand_prince"):
        raise ValueError(f"Unknown integrator: {integrator!r}. Use 'rk4', 'verlet', 'yoshida', or 'dormand_prince'.")

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
        return NumericalPropagationResult(
            steps=dp_result.steps,
            epoch=dp_result.epoch,
            duration_s=dp_result.duration_s,
            force_model_names=dp_result.force_model_names,
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

    return NumericalPropagationResult(
        steps=tuple(steps),
        epoch=ref_epoch,
        duration_s=duration_s,
        force_model_names=model_names,
    )
