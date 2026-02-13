# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Third-body perturbations (solar and lunar).

Analytical lunar ephemeris (Meeus Ch. 47 simplified) and third-body
gravitational perturbation force models for numerical propagation.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from humeris.domain.solar import sun_position_eci, AU_METERS

# J2000.0 reference epoch
_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Gravitational parameters
_MU_SUN = 1.32712440041e20   # m³/s² (IAU 2015 / IERS 2010)
_MU_MOON = 4.9028e12         # m³/s² (IAU 2015 / DE440)

# Obliquity of ecliptic (J2000, degrees)
_OBLIQUITY_DEG = 23.4393


@dataclass(frozen=True)
class MoonPosition:
    """Moon position at a given epoch."""
    position_eci_m: tuple[float, float, float]
    right_ascension_rad: float
    declination_rad: float
    distance_m: float


def _julian_centuries_j2000(epoch: datetime) -> float:
    """Julian centuries since J2000.0."""
    dt_seconds = (epoch - _J2000).total_seconds()
    return dt_seconds / (36525.0 * 86400.0)


def moon_position_eci(epoch: datetime) -> MoonPosition:
    """Analytical lunar ephemeris (Meeus Ch. 47 simplified).

    Computes geocentric ecliptic coordinates of the Moon, then converts
    to equatorial ECI. Accuracy ~0.5° in position, sufficient for
    perturbation modeling.

    Args:
        epoch: UTC datetime for Moon position.

    Returns:
        MoonPosition with ECI coordinates, RA, Dec, and distance.
    """
    T = _julian_centuries_j2000(epoch)

    # Fundamental arguments (degrees)
    L_prime = (218.3165 + 481267.8813 * T) % 360.0    # mean longitude
    D = (297.8502 + 445267.1115 * T) % 360.0          # mean elongation
    M = (357.5291 + 35999.0503 * T) % 360.0           # Sun mean anomaly
    M_prime = (134.9634 + 477198.8676 * T) % 360.0    # Moon mean anomaly
    F = (93.2721 + 483202.0175 * T) % 360.0           # argument of latitude

    # Convert to radians
    D_r = float(np.radians(D))
    M_r = float(np.radians(M))
    Mp_r = float(np.radians(M_prime))
    F_r = float(np.radians(F))

    # Ecliptic longitude (degrees)
    lam = L_prime + (
        6.289 * np.sin(Mp_r)
        - 1.274 * np.sin(2 * D_r - Mp_r)
        + 0.658 * np.sin(2 * D_r)
        - 0.214 * np.sin(2 * Mp_r)
        - 0.186 * np.sin(M_r)
        + 0.114 * np.sin(2 * F_r)
    )

    # Ecliptic latitude (degrees)
    beta = (
        5.128 * np.sin(F_r)
        + 0.281 * np.sin(Mp_r + F_r)
        - 0.278 * np.sin(Mp_r - F_r)
        - 0.173 * np.sin(2 * D_r - F_r)
    )

    # Distance (km)
    r_km = (
        385001.0
        - 20905.0 * np.cos(Mp_r)
        - 3699.0 * np.cos(2 * D_r - Mp_r)
        - 2956.0 * np.cos(2 * D_r)
        + 570.0 * np.cos(2 * Mp_r)
    )

    distance_m = r_km * 1000.0

    # Ecliptic → equatorial
    lam_r = float(np.radians(lam))
    beta_r = float(np.radians(beta))
    eps_r = float(np.radians(_OBLIQUITY_DEG))

    cos_beta = float(np.cos(beta_r))
    sin_beta = float(np.sin(beta_r))
    cos_lam = float(np.cos(lam_r))
    sin_lam = float(np.sin(lam_r))
    cos_eps = float(np.cos(eps_r))
    sin_eps = float(np.sin(eps_r))

    # Right ascension
    ra_rad = float(np.arctan2(
        sin_lam * cos_eps - np.tan(beta_r) * sin_eps,
        cos_lam,
    ))

    # Declination
    sin_dec = sin_beta * cos_eps + cos_beta * sin_eps * sin_lam
    sin_dec = float(np.clip(sin_dec, -1.0, 1.0))
    dec_rad = float(np.arcsin(sin_dec))

    # ECI Cartesian
    cos_ra = float(np.cos(ra_rad))
    sin_ra = float(np.sin(ra_rad))
    cos_dec = float(np.cos(dec_rad))
    sin_dec_val = float(np.sin(dec_rad))

    x = distance_m * cos_ra * cos_dec
    y = distance_m * sin_ra * cos_dec
    z = distance_m * sin_dec_val

    return MoonPosition(
        position_eci_m=(x, y, z),
        right_ascension_rad=ra_rad,
        declination_rad=dec_rad,
        distance_m=distance_m,
    )


def _third_body_acceleration(
    mu_body: float,
    r_body: tuple[float, float, float],
    r_sat: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Third-body tidal acceleration.

    a = μ · (d/|d|³ − r_body/|r_body|³)
    where d = r_body − r_sat

    Args:
        mu_body: Gravitational parameter of perturbing body (m³/s²).
        r_body: ECI position of perturbing body (m).
        r_sat: ECI position of satellite (m).

    Returns:
        Acceleration vector (m/s²).
    """
    rb = np.array(r_body, dtype=np.float64)
    rs = np.array(r_sat, dtype=np.float64)
    d_vec = rb - rs

    d_mag = float(np.linalg.norm(d_vec))
    d_mag3 = d_mag * d_mag * d_mag

    rb_mag = float(np.linalg.norm(rb))
    rb_mag3 = rb_mag * rb_mag * rb_mag

    a_vec = mu_body * (d_vec / d_mag3 - rb / rb_mag3)

    return (float(a_vec[0]), float(a_vec[1]), float(a_vec[2]))


class SolarThirdBodyForce:
    """Solar third-body gravitational perturbation.

    Implements ForceModel protocol for use with numerical propagator.
    """

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Solar third-body acceleration at given epoch and position."""
        sun = sun_position_eci(epoch)
        return _third_body_acceleration(_MU_SUN, sun.position_eci_m, position)


class LunarThirdBodyForce:
    """Lunar third-body gravitational perturbation.

    Implements ForceModel protocol for use with numerical propagator.
    """

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Lunar third-body acceleration at given epoch and position."""
        moon = moon_position_eci(epoch)
        return _third_body_acceleration(_MU_MOON, moon.position_eci_m, position)
