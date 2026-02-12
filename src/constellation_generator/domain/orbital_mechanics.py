"""
Orbital mechanics functions.

Pure mathematical conversions for orbital element transformations.
No external dependencies — only stdlib math.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _OrbitalConstants:
    """Standard orbital constants (IAU/WGS84 values)."""
    MU_EARTH: float = 3.986004418e14   # m³/s² — gravitational parameter
    R_EARTH: float = 6_371_000          # m — mean radius
    J2_EARTH: float = 1.08263e-3        # J2 perturbation coefficient
    J3_EARTH: float = -2.53215e-6       # J3 zonal harmonic coefficient
    EARTH_OMEGA: float = 1.99e-7        # rad/s — rotation rate for SSO
    EARTH_ROTATION_RATE: float = 7.2921159e-5  # rad/s — sidereal rotation rate
    # WGS84 ellipsoid
    R_EARTH_EQUATORIAL: float = 6_378_137.0       # m — semi-major axis
    R_EARTH_POLAR: float = 6_356_752.3142         # m — semi-minor axis
    FLATTENING: float = 1.0 / 298.257223563       # WGS84 flattening
    E_SQUARED: float = 0.00669437999014           # first eccentricity squared


OrbitalConstants: _OrbitalConstants = _OrbitalConstants()


def kepler_to_cartesian(
    a: float,
    e: float,
    i_rad: float,
    omega_big_rad: float,
    omega_small_rad: float,
    nu_rad: float,
) -> tuple[list[float], list[float]]:
    """
    Convert Keplerian orbital elements to ECI Cartesian position/velocity.

    Args:
        a: Semi-major axis (m)
        e: Eccentricity (0 for circular)
        i_rad: Inclination (radians)
        omega_big_rad: RAAN / longitude of ascending node (radians)
        omega_small_rad: Argument of perigee (radians)
        nu_rad: True anomaly (radians)

    Returns:
        (position_eci [x,y,z] in m, velocity_eci [vx,vy,vz] in m/s)
    """
    mu = OrbitalConstants.MU_EARTH

    r = a * (1 - e**2) / (1 + e * math.cos(nu_rad))

    p_factor = math.sqrt(mu / (a * (1 - e**2)))
    pos_pqw = [r * math.cos(nu_rad), r * math.sin(nu_rad), 0.0]
    vel_pqw = [
        -p_factor * math.sin(nu_rad),
        p_factor * (e + math.cos(nu_rad)),
        0.0,
    ]

    cO = math.cos(omega_big_rad)
    sO = math.sin(omega_big_rad)
    co = math.cos(omega_small_rad)
    so = math.sin(omega_small_rad)
    ci = math.cos(i_rad)
    si = math.sin(i_rad)

    rotation = [
        [cO * co - sO * so * ci, -cO * so - sO * co * ci, sO * si],
        [sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
        [so * si, co * si, ci],
    ]

    pos_eci = [
        sum(rotation[j][k] * pos_pqw[k] for k in range(3)) for j in range(3)
    ]
    vel_eci = [
        sum(rotation[j][k] * vel_pqw[k] for k in range(3)) for j in range(3)
    ]

    return pos_eci, vel_eci


def sso_inclination_deg(altitude_km: float) -> float:
    """
    Calculate Sun-synchronous orbit inclination for a given altitude.

    Uses the J2 perturbation-based SSO condition:
        cos(i) = -(2 * ω_earth / (3 * J2 * R²)) * ((alt + R)^3.5 / √μ)

    Args:
        altitude_km: Orbital altitude above Earth surface (km)

    Returns:
        Inclination in degrees (retrograde, > 90°)
    """
    c = OrbitalConstants
    r = altitude_km * 1000 + c.R_EARTH
    cos_i = -(2 * c.EARTH_OMEGA / (3 * c.J2_EARTH * c.R_EARTH**2)) * (
        r**3.5 / math.sqrt(c.MU_EARTH)
    )
    return math.degrees(math.acos(cos_i))


def j2_raan_rate(n: float, a: float, e: float, i_rad: float) -> float:
    """
    J2 secular rate of RAAN (longitude of ascending node).

    dΩ/dt = -3/2 · n · J2 · (R_E/a)² · cos(i) / (1-e²)²

    Args:
        n: Mean motion (rad/s).
        a: Semi-major axis (m).
        e: Eccentricity.
        i_rad: Inclination (radians).

    Returns:
        RAAN rate in rad/s. Negative for prograde, positive for retrograde.
    """
    c = OrbitalConstants
    p_ratio = (c.R_EARTH / a) ** 2
    return -1.5 * n * c.J2_EARTH * p_ratio * math.cos(i_rad) / (1 - e**2) ** 2


def j2_arg_perigee_rate(n: float, a: float, e: float, i_rad: float) -> float:
    """
    J2 secular rate of argument of perigee.

    dω/dt = 3/2 · n · J2 · (R_E/a)² · (2 - 5/2·sin²i) / (1-e²)²

    Zero at the critical inclination (~63.4°).

    Args:
        n: Mean motion (rad/s).
        a: Semi-major axis (m).
        e: Eccentricity.
        i_rad: Inclination (radians).

    Returns:
        Argument of perigee rate in rad/s.
    """
    c = OrbitalConstants
    p_ratio = (c.R_EARTH / a) ** 2
    return 1.5 * n * c.J2_EARTH * p_ratio * (2 - 2.5 * math.sin(i_rad) ** 2) / (1 - e**2) ** 2


def j2_mean_motion_correction(n: float, a: float, e: float, i_rad: float) -> float:
    """
    J2-corrected mean motion.

    n_corrected = n · (1 + 3/2 · J2 · (R_E/a)² · √(1-e²) · (1 - 3/2·sin²i))

    Args:
        n: Unperturbed mean motion (rad/s).
        a: Semi-major axis (m).
        e: Eccentricity.
        i_rad: Inclination (radians).

    Returns:
        Corrected mean motion in rad/s.
    """
    c = OrbitalConstants
    p_ratio = (c.R_EARTH / a) ** 2
    return n * (1 + 1.5 * c.J2_EARTH * p_ratio * math.sqrt(1 - e**2) * (1 - 1.5 * math.sin(i_rad) ** 2))
