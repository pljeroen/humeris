# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see LICENSE-COMMERCIAL.md.
"""IAU 2006 precession, IAU 2000B nutation, and GCRS↔ITRS transformations.

Implements the IAU 2006/2000A precession-nutation framework with the
truncated 77-term IAU 2000B nutation series (~1 mas accuracy).

References:
    IERS Conventions 2010, Chapters 5 and 6.
    Capitaine, N., Wallace, P.T., Chapront, J. (2003).
    Fukushima, T. (2003).
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from constellation_generator.domain.time_systems import AstroTime

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_ARCSEC_TO_RAD: float = math.pi / (180.0 * 3600.0)
_MAS_TO_RAD: float = _ARCSEC_TO_RAD / 1000.0

# Frame bias (IERS 2010, Eq. 5.4)
_DA0: float = -14.6e-3 * _ARCSEC_TO_RAD   # dα₀ in radians
_XI0: float = -16.617e-3 * _ARCSEC_TO_RAD  # ξ₀
_ETA0: float = -6.819e-3 * _ARCSEC_TO_RAD  # η₀

# --------------------------------------------------------------------------- #
# Nutation data loading
# --------------------------------------------------------------------------- #

_NUTATION_TERMS: Optional[list] = None


def _load_nutation_terms() -> list:
    """Load IAU 2000B nutation coefficients from bundled JSON."""
    global _NUTATION_TERMS
    if _NUTATION_TERMS is not None:
        return _NUTATION_TERMS

    data_path = Path(__file__).parent.parent / "data" / "iau2000b_nutation.json"
    with open(data_path) as f:
        data = json.load(f)

    _NUTATION_TERMS = data["terms"]
    return _NUTATION_TERMS


# --------------------------------------------------------------------------- #
# 3x3 matrix utilities (tuples of tuples, no numpy)
# --------------------------------------------------------------------------- #

def _r1(angle: float) -> tuple[tuple[float, ...], ...]:
    """Rotation matrix about X-axis."""
    c, s = math.cos(angle), math.sin(angle)
    return (
        (1.0, 0.0, 0.0),
        (0.0, c, s),
        (0.0, -s, c),
    )


def _r3(angle: float) -> tuple[tuple[float, ...], ...]:
    """Rotation matrix about Z-axis."""
    c, s = math.cos(angle), math.sin(angle)
    return (
        (c, s, 0.0),
        (-s, c, 0.0),
        (0.0, 0.0, 1.0),
    )


def _mat_mul(A: tuple, B: tuple) -> tuple[tuple[float, ...], ...]:
    """Multiply two 3x3 matrices."""
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return tuple(tuple(row) for row in result)


def _mat_vec(M: tuple, v: tuple) -> tuple[float, float, float]:
    """Multiply 3x3 matrix by 3-vector."""
    return (
        M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
        M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
        M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
    )


# --------------------------------------------------------------------------- #
# Fundamental arguments (Delaunay)
# --------------------------------------------------------------------------- #

def fundamental_arguments(t_tt: float) -> tuple[float, float, float, float, float]:
    """Compute Delaunay fundamental arguments at TT Julian centuries from J2000.

    Returns (l, l', F, D, Ω) in radians.

    l  = Mean anomaly of the Moon
    l' = Mean anomaly of the Sun
    F  = Mean argument of latitude of the Moon
    D  = Mean elongation of the Moon from the Sun
    Ω  = Mean longitude of the ascending node of the Moon

    IERS Conventions 2010, Table 5.2.
    """
    t = t_tt

    # l: Mean anomaly of the Moon (arcseconds)
    l = (485868.249036
         + t * (1717915923.2178
                + t * (31.8792
                       + t * (0.051635
                              + t * (-0.00024470)))))

    # l': Mean anomaly of the Sun
    lp = (1287104.79305
          + t * (129596581.0481
                 + t * (-0.5532
                        + t * (0.000136
                               + t * (-0.00001149)))))

    # F: Mean argument of latitude of the Moon
    F = (335779.526232
         + t * (1739527262.8478
                + t * (-12.7512
                       + t * (-0.001037
                              + t * (0.00000417)))))

    # D: Mean elongation of the Moon from the Sun
    D = (1072260.70369
         + t * (1602961601.2090
                + t * (-6.3706
                       + t * (0.006593
                              + t * (-0.00003169)))))

    # Ω: Mean longitude of the ascending node of the Moon
    Om = (450160.398036
          + t * (-6962890.5431
                 + t * (7.4722
                        + t * (0.007702
                               + t * (-0.00005939)))))

    # Convert arcseconds to radians, reduce modulo 2π
    to_rad = _ARCSEC_TO_RAD
    l_rad = (l * to_rad) % (2 * math.pi)
    lp_rad = (lp * to_rad) % (2 * math.pi)
    F_rad = (F * to_rad) % (2 * math.pi)
    D_rad = (D * to_rad) % (2 * math.pi)
    Om_rad = (Om * to_rad) % (2 * math.pi)

    return l_rad, lp_rad, F_rad, D_rad, Om_rad


# --------------------------------------------------------------------------- #
# IAU 2006 Precession (Fukushima-Williams angles)
# --------------------------------------------------------------------------- #

def precession_matrix(t_tt: float) -> tuple[tuple[float, ...], ...]:
    """IAU 2006 precession matrix using Fukushima-Williams angles.

    Parameters
    ----------
    t_tt : float
        Julian centuries of TT from J2000.0.

    Returns
    -------
    P : 3x3 rotation matrix (tuple of tuples)
        Precession matrix such that r_mean = P · r_GCRS_no_bias.

    Reference: IERS Conventions 2010, Eqs. 5.39-5.40.
    """
    t = t_tt

    # Fukushima-Williams angles (arcseconds)
    # γ̄ (gamb): IERS 2010 Eq. 5.39
    gamb = (-0.052928
            + t * (10.556378
                   + t * (0.4932044
                          + t * (-0.00031238
                                 + t * (-0.000002788
                                        + t * 0.0000000260)))))

    # φ̄ (phib): IERS 2010 Eq. 5.39
    phib = (84381.412819
            + t * (-46.811016
                   + t * (0.0511268
                          + t * (0.00053289
                                 + t * (-0.000000440
                                        + t * (-0.0000000176))))))

    # ψ̄ (psib): IERS 2010 Eq. 5.39
    psib = (-0.041775
            + t * (5038.481484
                   + t * (1.5584175
                          + t * (-0.00018522
                                 + t * (-0.000026452
                                        + t * (-0.0000000148))))))

    # ε_A (mean obliquity): IERS 2010 Eq. 5.40
    epsa = (84381.406
            + t * (-46.836769
                   + t * (-0.0001831
                          + t * (0.00200340
                                 + t * (-0.000000576
                                        + t * (-0.0000000434))))))

    # Convert to radians
    gamb_rad = gamb * _ARCSEC_TO_RAD
    phib_rad = phib * _ARCSEC_TO_RAD
    psib_rad = psib * _ARCSEC_TO_RAD
    epsa_rad = epsa * _ARCSEC_TO_RAD

    # P = R1(-ε_A) · R3(-ψ̄) · R1(φ̄) · R3(γ̄)
    return _mat_mul(
        _r1(-epsa_rad),
        _mat_mul(
            _r3(-psib_rad),
            _mat_mul(_r1(phib_rad), _r3(gamb_rad))
        )
    )


def mean_obliquity(t_tt: float) -> float:
    """Mean obliquity of the ecliptic ε_A in radians (IAU 2006).

    IERS Conventions 2010, Eq. 5.40.
    """
    t = t_tt
    epsa = (84381.406
            + t * (-46.836769
                   + t * (-0.0001831
                          + t * (0.00200340
                                 + t * (-0.000000576
                                        + t * (-0.0000000434))))))
    return epsa * _ARCSEC_TO_RAD


# --------------------------------------------------------------------------- #
# IAU 2000B Nutation
# --------------------------------------------------------------------------- #

def nutation_angles(t_tt: float) -> tuple[float, float]:
    """Compute nutation in longitude (Δψ) and obliquity (Δε).

    Uses the IAU 2000B truncated 77-term series.
    Accuracy: ~1 mas (sufficient for satellite work).

    Parameters
    ----------
    t_tt : float
        Julian centuries of TT from J2000.0.

    Returns
    -------
    (dpsi, deps) : tuple[float, float]
        Nutation in longitude and obliquity, in radians.
    """
    terms = _load_nutation_terms()
    l, lp, F, D, Om = fundamental_arguments(t_tt)
    args = (l, lp, F, D, Om)

    unit = 1e-7 * _ARCSEC_TO_RAD  # 0.1 μas → radians

    dpsi = 0.0
    deps = 0.0

    for term in terms:
        # Linear combination of fundamental arguments
        arg = (term["l"] * args[0]
               + term["lp"] * args[1]
               + term["F"] * args[2]
               + term["D"] * args[3]
               + term["Om"] * args[4])

        sin_arg = math.sin(arg)
        cos_arg = math.cos(arg)

        dpsi += (term["Sp"] + term["Spp"] * t_tt) * sin_arg + term["Cp"] * cos_arg
        deps += (term["Ce"] + term["Cep"] * t_tt) * cos_arg + term["Se"] * sin_arg

    return dpsi * unit, deps * unit


def nutation_matrix(
    dpsi: float,
    deps: float,
    eps0: float,
) -> tuple[tuple[float, ...], ...]:
    """Construct the nutation rotation matrix N.

    N = R1(-ε₀ - Δε) · R3(-Δψ) · R1(ε₀)

    Parameters
    ----------
    dpsi : float
        Nutation in longitude (radians).
    deps : float
        Nutation in obliquity (radians).
    eps0 : float
        Mean obliquity of the ecliptic (radians).
    """
    return _mat_mul(
        _r1(-(eps0 + deps)),
        _mat_mul(_r3(-dpsi), _r1(eps0))
    )


# --------------------------------------------------------------------------- #
# Frame Bias
# --------------------------------------------------------------------------- #

def frame_bias_matrix() -> tuple[tuple[float, ...], ...]:
    """Constant GCRS frame bias matrix B.

    Accounts for the offset between the GCRS pole and the mean J2000 pole:
        dα₀ = -14.6 mas, ξ₀ = -16.617 mas, η₀ = -6.819 mas

    B = R1(-η₀) · R2(ξ₀) · R3(dα₀)

    IERS Conventions 2010, Eq. 5.5.
    """
    # For such small angles, we can use the exact rotation matrices
    # R2 for Y-axis rotation
    c_xi, s_xi = math.cos(_XI0), math.sin(_XI0)
    c_eta, s_eta = math.cos(_ETA0), math.sin(_ETA0)
    c_da, s_da = math.cos(_DA0), math.sin(_DA0)

    # R2(ξ₀)
    R2_xi = (
        (c_xi, 0.0, -s_xi),
        (0.0, 1.0, 0.0),
        (s_xi, 0.0, c_xi),
    )

    # R1(-η₀) · R2(ξ₀) · R3(dα₀)
    return _mat_mul(_r1(-_ETA0), _mat_mul(R2_xi, _r3(_DA0)))


# --------------------------------------------------------------------------- #
# Earth Rotation Angle
# --------------------------------------------------------------------------- #

def earth_rotation_angle(ut1_jd: float) -> float:
    """Compute Earth Rotation Angle from UT1 Julian Date.

    ERA = 2π(0.7790572732640 + 1.00273781191135448 · D_u)

    where D_u is the UT1 Julian Date minus 2451545.0 (J2000.0).

    IERS Conventions 2010, Eq. 5.15.

    Parameters
    ----------
    ut1_jd : float
        UT1 Julian Date.

    Returns
    -------
    ERA in radians, normalized to [0, 2π).
    """
    Du = ut1_jd - 2451545.0
    theta = 2 * math.pi * (0.7790572732640 + 1.00273781191135448 * Du)
    return theta % (2 * math.pi)


# --------------------------------------------------------------------------- #
# Full GCRS → ITRS transformation
# --------------------------------------------------------------------------- #

def gcrs_to_itrs_matrix(
    t: AstroTime,
    ut1_utc: float = 0.0,
) -> tuple[tuple[float, ...], ...]:
    """Compute the full GCRS → ITRS rotation matrix.

    Without EOP data:
        M = R3(ERA) · N · P · B

    With EOP data (ut1_utc provided):
        M = W · R3(ERA) · N · P · B
    where W is the polar motion matrix (identity without EOP).

    Parameters
    ----------
    t : AstroTime
        Epoch for the transformation.
    ut1_utc : float
        UT1-UTC correction in seconds. Default 0.0 (no EOP).

    Returns
    -------
    M : 3x3 rotation matrix (tuple of tuples)
    """
    t_tt = t.to_julian_centuries_tt()

    # Frame bias
    B = frame_bias_matrix()

    # Precession
    P = precession_matrix(t_tt)

    # Nutation
    dpsi, deps = nutation_angles(t_tt)
    eps0 = mean_obliquity(t_tt)
    N = nutation_matrix(dpsi, deps, eps0)

    # Earth rotation angle: UT1 JD = UTC JD + ut1_utc/86400
    from constellation_generator.domain.time_systems import datetime_to_jd
    dt_utc = t.to_utc_datetime()
    jd_utc = datetime_to_jd(dt_utc)
    jd_ut1 = jd_utc + ut1_utc / 86400.0

    era = earth_rotation_angle(jd_ut1)
    R_era = _r3(era)

    # Combine: M = R3(ERA) · N · P · B
    NPB = _mat_mul(N, _mat_mul(P, B))
    return _mat_mul(R_era, NPB)


def eci_to_ecef_precise(
    pos_eci: tuple[float, float, float],
    t: AstroTime,
    ut1_utc: float = 0.0,
) -> tuple[float, float, float]:
    """High-fidelity GCRS → ITRS position transformation.

    Parameters
    ----------
    pos_eci : tuple[float, float, float]
        Position in GCRS/ECI frame (meters).
    t : AstroTime
        Epoch.
    ut1_utc : float
        UT1-UTC correction in seconds.

    Returns
    -------
    Position in ITRS/ECEF frame (meters).
    """
    M = gcrs_to_itrs_matrix(t, ut1_utc)
    return _mat_vec(M, pos_eci)
