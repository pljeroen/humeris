# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Tidal force models for high-fidelity orbit propagation.

Two tidal perturbation models implementing the ForceModel protocol:

- SolidTideForce: IERS 2010 Chapter 6 frequency-independent solid Earth tides
  via degree-2 Love numbers. Computes the gradient of the tidal geopotential
  perturbation caused by the Moon and Sun raising tides in the solid Earth.
  Magnitude at LEO: ~1e-9 to 1e-8 m/s².

- OceanTideForce: FES2004 ocean tides using the 8 main tidal constituents
  (M2, S2, N2, K2, K1, O1, P1, Q1). Loads spherical harmonic corrections
  from bundled JSON data and computes the degree-2 geopotential gradient.
  Magnitude at LEO: ~1e-11 to 1e-9 m/s² (roughly 1/10th of solid tides).

All implementations use only stdlib (math, datetime, json, pathlib, dataclasses).
No external dependencies.

References:
    IERS Conventions 2010, Chapter 6
    Montenbruck & Gill, "Satellite Orbits", Ch. 3.2
    FES2004 (Lyard et al., 2006)
"""

import json
import math
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# --- Physical constants ---

_GM_EARTH: float = 3.986004418e14  # m³/s² (Earth gravitational parameter)
_R_EARTH: float = 6378137.0  # m (Earth equatorial radius)
_GM_MOON: float = 4.9028e12  # m³/s² (Moon gravitational parameter)
_GM_SUN: float = 1.32712440041e20  # m³/s² (Sun gravitational parameter)

# Love numbers (IERS 2010, Table 6.3)
_K20: float = 0.30190  # degree 2, order 0
_K21: float = 0.29830  # degree 2, order 1
_K22: float = 0.30102  # degree 2, order 2

# --- Data directory ---

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# --- Helpers ---

_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_AU = 1.495978707e11  # meters


def _datetime_to_jd(dt: datetime) -> float:
    """Convert datetime to Julian Date.

    Uses the standard algorithm: JD = 2451545.0 + seconds_since_J2000 / 86400.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (dt - _J2000).total_seconds()
    return 2451545.0 + delta / 86400.0


def _sun_position_approx(dt: datetime) -> tuple[float, float, float]:
    """Approximate Sun position in ECI (meters).

    Low-precision solar coordinates based on Meeus (1991) truncated series.
    Accuracy ~1 degree in ecliptic longitude, sufficient for tidal calculations.
    """
    jd = _datetime_to_jd(dt)
    T = (jd - 2451545.0) / 36525.0

    # Mean anomaly of Sun (degrees)
    M_deg = (357.5291092 + 35999.0502909 * T) % 360.0
    M_rad = math.radians(M_deg)

    # Equation of center (degrees)
    C_deg = (
        (1.9146 - 0.004817 * T) * math.sin(M_rad)
        + 0.019993 * math.sin(2.0 * M_rad)
    )

    # Sun's mean longitude (degrees)
    L0_deg = (280.46646 + 36000.76983 * T) % 360.0

    # Sun's true longitude (radians)
    sun_lon = math.radians((L0_deg + C_deg) % 360.0)

    # Obliquity of ecliptic (radians)
    eps = math.radians(23.439291 - 0.0130042 * T)

    # Orbital eccentricity
    e = 0.016708634 - 0.000042037 * T

    # True anomaly
    nu = M_rad + math.radians(C_deg)

    # Sun-Earth distance (meters)
    r_au = 1.000001018 * (1.0 - e * e) / (1.0 + e * math.cos(nu))
    r = r_au * _AU

    # ECI coordinates (equatorial frame via obliquity rotation)
    cos_lon = math.cos(sun_lon)
    sin_lon = math.sin(sun_lon)
    cos_eps = math.cos(eps)
    sin_eps = math.sin(eps)

    x = r * cos_lon
    y = r * sin_lon * cos_eps
    z = r * sin_lon * sin_eps

    return (x, y, z)


def _moon_position_approx(dt: datetime) -> tuple[float, float, float]:
    """Low-precision Moon position in ECI (meters).

    Simplified Meeus algorithm giving ~1 degree accuracy in ecliptic longitude.
    Sufficient for tidal force direction computation.
    """
    jd = _datetime_to_jd(dt)
    T = (jd - 2451545.0) / 36525.0

    # Meeus simplified lunar coordinates
    L0 = 218.3165 + 481267.8813 * T  # mean longitude (degrees)
    M_moon = 134.9634 + 477198.8676 * T  # mean anomaly (degrees)
    D = 297.8502 + 445267.1115 * T  # mean elongation (degrees)

    L0_r = math.radians(L0 % 360.0)
    M_r = math.radians(M_moon % 360.0)
    D_r = math.radians(D % 360.0)

    # Ecliptic longitude (simplified, degrees -> radians)
    lon_ecl = L0_r + math.radians(6.289) * math.sin(M_r)

    # Ecliptic latitude
    F = 93.272 + 483202.0175 * T  # argument of latitude (degrees)
    lat_ecl = math.radians(5.128) * math.sin(math.radians(F % 360.0))

    # Distance (meters), with eccentricity correction
    r = 385001e3 * (1.0 - 0.0549 * math.cos(M_r))

    # Ecliptic to equatorial rotation
    eps = math.radians(23.439291 - 0.0130042 * T)

    x_ecl = r * math.cos(lat_ecl) * math.cos(lon_ecl)
    y_ecl = r * math.cos(lat_ecl) * math.sin(lon_ecl)
    z_ecl = r * math.sin(lat_ecl)

    x = x_ecl
    y = y_ecl * math.cos(eps) - z_ecl * math.sin(eps)
    z = y_ecl * math.sin(eps) + z_ecl * math.cos(eps)

    return (x, y, z)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Dot product of two 3-vectors."""
    return float(np.dot(a, b))


def _mag(v: tuple[float, float, float]) -> float:
    """Magnitude of a 3-vector."""
    return float(np.linalg.norm(v))


def _scale(s: float, v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Scalar multiply a 3-vector."""
    result = s * np.array(v)
    return (float(result[0]), float(result[1]), float(result[2]))


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Unit vector."""
    arr = np.array(v)
    m = float(np.linalg.norm(arr))
    if m == 0.0:
        return (0.0, 0.0, 0.0)
    result = arr / m
    return (float(result[0]), float(result[1]), float(result[2]))


# --- Force models ---


@dataclass(frozen=True)
class SolidTideForce:
    """IERS 2010 frequency-independent solid Earth tide acceleration.

    Computes the degree-2 tidal perturbation on a satellite due to Moon
    and Sun raising tides in the solid Earth, parameterized by Love number k2.

    Per IERS 2010 Conventions (Chapter 6, eq. 6.6), the change in the
    fully-normalized geopotential coefficients due to solid tides is:

        Delta_C_bar_nm = k_nm / (2n+1) * (GM_j/GM_E) * (R_E/d_j)^(n+1) * ...

    The 1/(2n+1) factor arises from the conversion between the tide-generating
    potential and the geopotential coefficient perturbation in the fully-normalized
    spherical harmonic framework.

    The resulting acceleration on the satellite (gradient of the perturbed
    geopotential) for degree n=2 is:

        a = k2 / (2n+1) * GM_j * R_E^5 / (d_j^3 * r^4) * [
            3 * cos(psi) * D_hat - (15*cos^2(psi) - 3)/2 * r_hat
        ]

    with 2n+1 = 5 for degree 2, summed over Moon and Sun.

    Magnitude at LEO: ~1e-9 to 1e-8 m/s^2.
    """

    k2: float = _K20

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        r_vec = np.array(position)
        r = float(np.linalg.norm(r_vec))
        if r < 1.0:
            return (0.0, 0.0, 0.0)

        r_hat = r_vec / r

        acc = np.zeros(3)

        # Compute tidal acceleration from each perturbing body
        bodies = [
            (_GM_MOON, _moon_position_approx(epoch)),
            (_GM_SUN, _sun_position_approx(epoch)),
        ]

        # 1/(2n+1) factor from IERS 2010 fully-normalized harmonic formulation
        degree_factor = 1.0 / 5.0  # 1/(2*2+1) for degree n=2

        for gm_j, d_vec in bodies:
            d_arr = np.array(d_vec)
            d = float(np.linalg.norm(d_arr))
            if d < 1.0:
                continue

            d_hat = d_arr / d
            cos_psi = float(np.dot(r_hat, d_hat))

            # Coefficient: k2/(2n+1) * GM_j * R_E^5 / (d^3 * r^4)
            factor = (
                self.k2 * degree_factor * gm_j * _R_EARTH**5 / (d**3 * r**4)
            )

            # Acceleration components (gradient of degree-2 tidal geopotential):
            # a = factor * [3*cos_psi * d_hat - (15*cos_psi^2 - 3)/2 * r_hat]
            radial_coeff = -(15.0 * cos_psi * cos_psi - 3.0) / 2.0
            cross_coeff = 3.0 * cos_psi

            acc += factor * (radial_coeff * r_hat + cross_coeff * d_hat)

        return (float(acc[0]), float(acc[1]), float(acc[2]))


class OceanTideForce:
    """FES2004 ocean tide acceleration from 8 main tidal constituents.

    Loads spherical harmonic corrections (degree-2) from bundled JSON data
    and computes the geopotential gradient perturbation.

    For each constituent k with period T_k, the tidal argument is:
        theta_k(t) = 2*pi * t / T_k

    where t is hours since J2000.

    The corrections modify the C_nm and S_nm coefficients:
        dC_nm = sum_k [Cp_k * cos(theta_k) + Cm_k * sin(theta_k)]
        dS_nm = sum_k [Sp_k * cos(theta_k) + Sm_k * sin(theta_k)]

    The acceleration is computed from the degree-2 geopotential gradient
    of the modified harmonics, rotated between ECI and ECEF frames.

    Magnitude at LEO: ~1e-11 to 1e-9 m/s^2.
    """

    def __init__(self) -> None:
        data_path = _DATA_DIR / "ocean_tide_coefficients.json"
        with open(data_path) as f:
            raw = json.load(f)

        self._constituents: list[dict] = raw["constituents"]
        self._n_constituents: int = len(self._constituents)

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        r_vec = np.array(position)
        r = float(np.linalg.norm(r_vec))
        if r < 1.0:
            return (0.0, 0.0, 0.0)

        # Hours since J2000 for tidal arguments
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        t_hours = (epoch - _J2000).total_seconds() / 3600.0

        # Compute GMST for ECI <-> ECEF rotation
        jd = _datetime_to_jd(epoch)
        T = (jd - 2451545.0) / 36525.0
        # GMST in degrees (IAU 1982 simplified)
        gmst_deg = (
            280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * T * T
        ) % 360.0
        gmst = math.radians(gmst_deg)
        cos_g = math.cos(gmst)
        sin_g = math.sin(gmst)

        # Rotate ECI -> ECEF using rotation matrix
        rot_eci_to_ecef = np.array([
            [cos_g, sin_g, 0.0],
            [-sin_g, cos_g, 0.0],
            [0.0, 0.0, 1.0],
        ])
        r_ecef = rot_eci_to_ecef @ r_vec
        x_ecef = float(r_ecef[0])
        y_ecef = float(r_ecef[1])
        z_ecef = float(r_ecef[2])

        # Spherical coordinates from ECEF
        r_xy = math.sqrt(x_ecef * x_ecef + y_ecef * y_ecef)
        phi = math.atan2(z_ecef, r_xy)  # geocentric latitude
        lam = math.atan2(y_ecef, x_ecef)  # longitude

        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)

        # Accumulate degree-2 harmonic corrections from all constituents
        dC = np.zeros((3, 3))  # dC[n][m] for n=0,1,2
        dS = np.zeros((3, 3))

        for constituent in self._constituents:
            period_hours = constituent["period_hours"]
            theta = 2.0 * math.pi * t_hours / period_hours
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            for corr in constituent["corrections_nm"]:
                n = corr["n"]
                m = corr["m"]
                Cp = corr["Cp"]
                Sp = corr["Sp"]
                Cm = corr["Cm"]
                Sm = corr["Sm"]

                dC[n, m] += Cp * cos_theta + Cm * sin_theta
                dS[n, m] += Sp * cos_theta + Sm * sin_theta

        # Compute degree-2 geopotential gradient in spherical coordinates
        # The perturbed potential (degree 2 only):
        #   dPhi = (GM/r) * sum_{m=0}^{2} (R_E/r)^2 *
        #          [dC_2m * cos(m*lam) + dS_2m * sin(m*lam)] * P_2m(sin phi)
        #
        # Acceleration in spherical (r, phi, lambda):
        #   a_r     = -d(dPhi)/dr
        #   a_phi   = -(1/r) * d(dPhi)/d(phi)
        #   a_lam   = -(1/(r*cos(phi))) * d(dPhi)/d(lambda)

        n = 2
        re_r = _R_EARTH / r
        re_r_n = re_r * re_r  # (R_E/r)^2
        gm_r = _GM_EARTH / r

        # Associated Legendre functions P_2m(sin phi) and their derivatives
        # P_20 = (3*sin^2(phi) - 1) / 2
        # P_21 = 3*sin(phi)*cos(phi)
        # P_22 = 3*cos^2(phi)
        P = np.array([
            (3.0 * sin_phi * sin_phi - 1.0) / 2.0,
            3.0 * sin_phi * cos_phi,
            3.0 * cos_phi * cos_phi,
        ])

        # dP/dphi derivatives
        # dP20/dphi = 3*sin(phi)*cos(phi)
        # dP21/dphi = 3*(cos^2(phi) - sin^2(phi)) = 3*cos(2*phi)
        # dP22/dphi = -6*cos(phi)*sin(phi) = -3*sin(2*phi)
        dP = np.array([
            3.0 * sin_phi * cos_phi,
            3.0 * (cos_phi * cos_phi - sin_phi * sin_phi),
            -6.0 * cos_phi * sin_phi,
        ])

        a_r = 0.0
        a_phi = 0.0
        a_lam = 0.0

        for m in range(3):
            cos_mlam = math.cos(m * lam)
            sin_mlam = math.sin(m * lam)

            Hnm = dC[2, m] * cos_mlam + dS[2, m] * sin_mlam
            Hnm_lam = m * (-dC[2, m] * sin_mlam + dS[2, m] * cos_mlam)

            # Radial: a_r = -(n+1) * (GM/r^2) * (R_E/r)^n * Hnm * Pnm
            a_r += (n + 1) * Hnm * P[m]

            # Latitudinal: a_phi = -(1/r) * GM/r * (R_E/r)^n * Hnm * dPnm/dphi
            a_phi += Hnm * dP[m]

            # Longitudinal: a_lam = -(1/(r*cos_phi)) * GM/r * (R_E/r)^n * Hnm_lam * Pnm
            a_lam += Hnm_lam * P[m]

        # Apply common factors and signs
        # The potential is dPhi = (GM/r) * (R_E/r)^2 * sum(...)
        # a_r = d(dPhi)/dr = -(n+1)/r * dPhi_content = -(n+1) * GM/r^2 * (R_E/r)^n * sum
        # But since we want the acceleration FROM the potential (a = +grad(Phi)):
        # a_r = +(n+1) * (GM/r^2) * (R_E/r)^n * sum   [actually -d/dr of -GM/r gives +]
        # For tidal perturbation: a = grad(dPhi), and dPhi increases the potential
        # d(dPhi)/dr = -(n+1)/r * (GM/r) * (R_E/r)^n * Hnm*Pnm
        # So a_r (outward positive) = -d(dPhi)/dr is not right for acceleration...
        #
        # The gravitational acceleration from potential Phi is a = -grad(Phi) for
        # Phi = -GM/r convention. But for Phi = GM/r (positive) convention:
        # a = +grad(Phi) doesn't work either. Standard: a = -grad(V) where V = -GM/r.
        # For the perturbation dV (additional potential), the perturbation acceleration
        # is a_pert = -grad(dV). But dV here is a positive quantity (adds to potential).
        #
        # Actually in celestial mechanics, the potential U = GM/r + corrections, and
        # the acceleration is a = grad(U). So for a positive perturbation dU:
        # a = grad(dU)
        #
        # dU = (GM/r) * (R_E/r)^2 * F(phi, lam)
        # d(dU)/dr = -(GM/r^2) * (1 + n) * (R_E/r)^n * F  [since d/dr(1/r * (1/r)^n) = -(n+1)/r^(n+2)]
        # But in gradient convention, a_r = d(dU)/dr, so:
        # a_r = -(n+1) * (GM/r^2) * (R_E/r)^n * F
        # This means the radial component is inward (negative) for positive F.

        common = gm_r * re_r_n / r  # = GM / r^2 * (R_E/r)^n

        # Radial acceleration (positive outward in spherical coords)
        a_r_final = -(n + 1) * common * a_r

        # Latitudinal acceleration
        a_phi_final = common * a_phi

        # Longitudinal acceleration (avoid division by zero at poles)
        if abs(cos_phi) > 1e-15:
            a_lam_final = common * a_lam / cos_phi
        else:
            a_lam_final = 0.0

        # Convert spherical acceleration to ECEF Cartesian
        # Unit vectors in spherical:
        # r_hat = (cos_phi*cos_lam, cos_phi*sin_lam, sin_phi)
        # phi_hat = (-sin_phi*cos_lam, -sin_phi*sin_lam, cos_phi)
        # lam_hat = (-sin_lam, cos_lam, 0)
        cos_lam = math.cos(lam)
        sin_lam = math.sin(lam)

        ax_ecef = (
            a_r_final * cos_phi * cos_lam
            + a_phi_final * (-sin_phi) * cos_lam
            + a_lam_final * (-sin_lam)
        )
        ay_ecef = (
            a_r_final * cos_phi * sin_lam
            + a_phi_final * (-sin_phi) * sin_lam
            + a_lam_final * cos_lam
        )
        az_ecef = (
            a_r_final * sin_phi
            + a_phi_final * cos_phi
        )

        # Rotate ECEF -> ECI
        ax_eci = cos_g * ax_ecef - sin_g * ay_ecef
        ay_eci = sin_g * ax_ecef + cos_g * ay_ecef
        az_eci = az_ecef

        return (ax_eci, ay_eci, az_eci)
