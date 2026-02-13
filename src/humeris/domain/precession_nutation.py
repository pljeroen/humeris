# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""IAU 2006 precession, IAU 2000A/B nutation, and GCRS-ITRS transformations.

Implements the IAU 2006/2000A precession-nutation framework. Default uses the
full 1320-term IAU 2000A nutation series (~0.001 mas accuracy). Falls back to
the 77-term IAU 2000B series (~1 mas) if requested.

NumPy vectorized: the nutation series evaluation uses array operations for
all 1320 terms in a single pass (vs per-term Python loop).

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

import numpy as np

from humeris.domain.time_systems import AstroTime

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_ARCSEC_TO_RAD: float = math.pi / (180.0 * 3600.0)
_MAS_TO_RAD: float = _ARCSEC_TO_RAD / 1000.0
_TWO_PI: float = 2.0 * math.pi

# Frame bias (IERS 2010, Eq. 5.4)
_DA0: float = -14.6e-3 * _ARCSEC_TO_RAD   # dα₀ in radians
_XI0: float = -16.617e-3 * _ARCSEC_TO_RAD  # ξ₀
_ETA0: float = -6.819e-3 * _ARCSEC_TO_RAD  # η₀

# --------------------------------------------------------------------------- #
# Nutation data loading (cached NumPy arrays)
# --------------------------------------------------------------------------- #

_NUT_CACHE: dict[str, Optional[dict]] = {"2000A": None, "2000B": None}


def _load_nutation_arrays(model: str = "2000A") -> dict:
    """Load nutation coefficients into pre-built NumPy arrays.

    Returns dict with keys:
        mults: (N, num_args) int array of argument multipliers
        Sp, Spp, Cp, Ce, Cep, Se: (N,) float arrays of coefficients
        unit: float conversion factor (coefficient units -> arcseconds)
        num_args: 5 for 2000B, 14 for 2000A
    """
    if _NUT_CACHE[model] is not None:
        return _NUT_CACHE[model]

    if model == "2000A":
        data_path = Path(__file__).parent.parent / "data" / "iau2000a_nutation.json"
    else:
        data_path = Path(__file__).parent.parent / "data" / "iau2000b_nutation.json"

    with open(data_path) as f:
        data = json.load(f)

    terms = data["terms"]
    unit = data["unit_factor_arcsec"]  # 1e-6 for 2000A, 1e-7 for 2000B
    n = len(terms)

    if model == "2000A":
        # 14 fundamental arguments: 5 Delaunay + 9 planetary
        arg_keys = ["l", "lp", "F", "D", "Om",
                     "L_Me", "L_Ve", "L_E", "L_Ma",
                     "L_J", "L_Sa", "L_U", "L_Ne", "p_A"]
        coeff_keys_extra = ["Cpp", "Sep"]
    else:
        arg_keys = ["l", "lp", "F", "D", "Om"]
        coeff_keys_extra = []

    mults = np.zeros((n, len(arg_keys)), dtype=np.float64)
    Sp = np.zeros(n)
    Spp = np.zeros(n)
    Cp = np.zeros(n)
    Ce = np.zeros(n)
    Cep = np.zeros(n)
    Se = np.zeros(n)
    Cpp = np.zeros(n)
    Sep = np.zeros(n)

    for i, term in enumerate(terms):
        for j, key in enumerate(arg_keys):
            mults[i, j] = term[key]
        Sp[i] = term["Sp"]
        Spp[i] = term.get("Spp", 0.0)
        Cp[i] = term.get("Cp", 0.0)
        Ce[i] = term.get("Ce", 0.0)
        Cep[i] = term.get("Cep", 0.0)
        Se[i] = term.get("Se", 0.0)
        if model == "2000A":
            Cpp[i] = term.get("Cpp", 0.0)
            Sep[i] = term.get("Sep", 0.0)

    result = {
        "mults": mults,
        "Sp": Sp, "Spp": Spp, "Cp": Cp, "Cpp": Cpp,
        "Ce": Ce, "Cep": Cep, "Se": Se, "Sep": Sep,
        "unit": unit,
        "num_args": len(arg_keys),
        "model": model,
    }
    _NUT_CACHE[model] = result
    return result


# Backward compat: keep the old dict-based loader for tests that import it
_NUTATION_TERMS: Optional[list] = None


def _load_nutation_terms() -> list:
    """Load IAU 2000B nutation coefficients from bundled JSON (legacy)."""
    global _NUTATION_TERMS
    if _NUTATION_TERMS is not None:
        return _NUTATION_TERMS

    data_path = Path(__file__).parent.parent / "data" / "iau2000b_nutation.json"
    with open(data_path) as f:
        data = json.load(f)

    _NUTATION_TERMS = data["terms"]
    return _NUTATION_TERMS


# --------------------------------------------------------------------------- #
# 3x3 matrix utilities (tuples of tuples for API compatibility)
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

    Returns (l, l', F, D, Omega) in radians.

    l  = Mean anomaly of the Moon
    l' = Mean anomaly of the Sun
    F  = Mean argument of latitude of the Moon
    D  = Mean elongation of the Moon from the Sun
    Omega  = Mean longitude of the ascending node of the Moon

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

    # Omega: Mean longitude of the ascending node of the Moon
    Om = (450160.398036
          + t * (-6962890.5431
                 + t * (7.4722
                        + t * (0.007702
                               + t * (-0.00005939)))))

    # Convert arcseconds to radians, reduce modulo 2pi
    to_rad = _ARCSEC_TO_RAD
    l_rad = (l * to_rad) % _TWO_PI
    lp_rad = (lp * to_rad) % _TWO_PI
    F_rad = (F * to_rad) % _TWO_PI
    D_rad = (D * to_rad) % _TWO_PI
    Om_rad = (Om * to_rad) % _TWO_PI

    return l_rad, lp_rad, F_rad, D_rad, Om_rad


def planetary_arguments(t_tt: float) -> tuple[float, ...]:
    """Compute 9 planetary fundamental arguments at TT Julian centuries.

    Returns (L_Me, L_Ve, L_E, L_Ma, L_J, L_Sa, L_U, L_Ne, p_A) in radians.

    IERS Conventions 2003 (Simon et al. 1994).
    """
    t = t_tt

    # Mean longitude of Mercury (radians)
    L_Me = (4.402608842 + 2608.7903141574 * t) % _TWO_PI
    # Mean longitude of Venus
    L_Ve = (3.176146697 + 1021.3285546211 * t) % _TWO_PI
    # Mean longitude of Earth
    L_E = (1.753470314 + 628.3075849991 * t) % _TWO_PI
    # Mean longitude of Mars
    L_Ma = (6.203480913 + 334.0612426700 * t) % _TWO_PI
    # Mean longitude of Jupiter
    L_J = (0.599546497 + 52.9690962641 * t) % _TWO_PI
    # Mean longitude of Saturn
    L_Sa = (0.874016757 + 21.3299104960 * t) % _TWO_PI
    # Mean longitude of Uranus
    L_U = (5.481293872 + 7.4781598567 * t) % _TWO_PI
    # Mean longitude of Neptune
    L_Ne = (5.311886287 + 3.8133035638 * t) % _TWO_PI
    # General accumulated precession in longitude
    p_A = (0.02438175 + 0.00000538691 * t) * t

    return L_Me, L_Ve, L_E, L_Ma, L_J, L_Sa, L_U, L_Ne, p_A


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
        Precession matrix such that r_mean = P * r_GCRS_no_bias.

    Reference: IERS Conventions 2010, Eqs. 5.39-5.40.
    """
    t = t_tt

    # Fukushima-Williams angles (arcseconds)
    gamb = (-0.052928
            + t * (10.556378
                   + t * (0.4932044
                          + t * (-0.00031238
                                 + t * (-0.000002788
                                        + t * 0.0000000260)))))

    phib = (84381.412819
            + t * (-46.811016
                   + t * (0.0511268
                          + t * (0.00053289
                                 + t * (-0.000000440
                                        + t * (-0.0000000176))))))

    psib = (-0.041775
            + t * (5038.481484
                   + t * (1.5584175
                          + t * (-0.00018522
                                 + t * (-0.000026452
                                        + t * (-0.0000000148))))))

    epsa = (84381.406
            + t * (-46.836769
                   + t * (-0.0001831
                          + t * (0.00200340
                                 + t * (-0.000000576
                                        + t * (-0.0000000434))))))

    gamb_rad = gamb * _ARCSEC_TO_RAD
    phib_rad = phib * _ARCSEC_TO_RAD
    psib_rad = psib * _ARCSEC_TO_RAD
    epsa_rad = epsa * _ARCSEC_TO_RAD

    # P = R1(-eps_A) * R3(-psi_bar) * R1(phi_bar) * R3(gamma_bar)
    return _mat_mul(
        _r1(-epsa_rad),
        _mat_mul(
            _r3(-psib_rad),
            _mat_mul(_r1(phib_rad), _r3(gamb_rad))
        )
    )


def mean_obliquity(t_tt: float) -> float:
    """Mean obliquity of the ecliptic eps_A in radians (IAU 2006).

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
# IAU 2000A/B Nutation (NumPy vectorized)
# --------------------------------------------------------------------------- #

def nutation_angles(
    t_tt: float,
    model: str = "2000A",
) -> tuple[float, float]:
    """Compute nutation in longitude (dpsi) and obliquity (deps).

    Parameters
    ----------
    t_tt : float
        Julian centuries of TT from J2000.0.
    model : str
        "2000A" (default, 1320 terms, ~0.001 mas) or
        "2000B" (77 terms, ~1 mas).

    Returns
    -------
    (dpsi, deps) : tuple[float, float]
        Nutation in longitude and obliquity, in radians.
    """
    nut = _load_nutation_arrays(model)
    mults = nut["mults"]
    unit = nut["unit"] * _ARCSEC_TO_RAD  # -> radians

    # Build fundamental arguments array
    l, lp, F, D, Om = fundamental_arguments(t_tt)

    if nut["num_args"] == 14:
        # IAU 2000A: 5 Delaunay + 9 planetary
        L_Me, L_Ve, L_E, L_Ma, L_J, L_Sa, L_U, L_Ne, p_A = planetary_arguments(t_tt)
        args = np.array([l, lp, F, D, Om, L_Me, L_Ve, L_E, L_Ma, L_J, L_Sa, L_U, L_Ne, p_A])
    else:
        # IAU 2000B: 5 Delaunay only
        args = np.array([l, lp, F, D, Om])

    # Vectorized argument computation: mults (N, num_args) @ args (num_args,) -> (N,)
    phi = mults @ args

    # Vectorized trig
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Vectorized summation
    Sp = nut["Sp"]
    Spp = nut["Spp"]
    Cp = nut["Cp"]
    Ce = nut["Ce"]
    Cep = nut["Cep"]
    Se = nut["Se"]

    dpsi = np.sum((Sp + Spp * t_tt) * sin_phi + Cp * cos_phi)
    deps = np.sum((Ce + Cep * t_tt) * cos_phi + Se * sin_phi)

    # Add time-dependent cosine terms for 2000A (Cpp, Sep)
    if model == "2000A":
        Cpp = nut["Cpp"]
        Sep = nut["Sep"]
        dpsi += np.sum(Cpp * t_tt * cos_phi)
        deps += np.sum(Sep * t_tt * sin_phi)

    return float(dpsi * unit), float(deps * unit)


def nutation_matrix(
    dpsi: float,
    deps: float,
    eps0: float,
) -> tuple[tuple[float, ...], ...]:
    """Construct the nutation rotation matrix N.

    N = R1(-(eps0 + deps)) * R3(-dpsi) * R1(eps0)

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
        da0 = -14.6 mas, xi0 = -16.617 mas, eta0 = -6.819 mas

    B = R1(-eta0) * R2(xi0) * R3(da0)

    IERS Conventions 2010, Eq. 5.5.
    """
    c_xi, s_xi = math.cos(_XI0), math.sin(_XI0)
    c_eta, s_eta = math.cos(_ETA0), math.sin(_ETA0)
    c_da, s_da = math.cos(_DA0), math.sin(_DA0)

    R2_xi = (
        (c_xi, 0.0, -s_xi),
        (0.0, 1.0, 0.0),
        (s_xi, 0.0, c_xi),
    )

    return _mat_mul(_r1(-_ETA0), _mat_mul(R2_xi, _r3(_DA0)))


# --------------------------------------------------------------------------- #
# Earth Rotation Angle
# --------------------------------------------------------------------------- #

def earth_rotation_angle(ut1_jd: float) -> float:
    """Compute Earth Rotation Angle from UT1 Julian Date.

    ERA = 2pi(0.7790572732640 + 1.00273781191135448 * D_u)

    where D_u is the UT1 Julian Date minus 2451545.0 (J2000.0).

    IERS Conventions 2010, Eq. 5.15.

    Parameters
    ----------
    ut1_jd : float
        UT1 Julian Date.

    Returns
    -------
    ERA in radians, normalized to [0, 2pi).
    """
    Du = ut1_jd - 2451545.0
    theta = _TWO_PI * (0.7790572732640 + 1.00273781191135448 * Du)
    return theta % _TWO_PI


# --------------------------------------------------------------------------- #
# Full GCRS -> ITRS transformation
# --------------------------------------------------------------------------- #

def gcrs_to_itrs_matrix(
    t: AstroTime,
    ut1_utc: float = 0.0,
    nutation_model: str = "2000A",
) -> tuple[tuple[float, ...], ...]:
    """Compute the full GCRS -> ITRS rotation matrix.

    Without EOP data:
        M = R3(ERA) * N * P * B

    With EOP data (ut1_utc provided):
        M = W * R3(ERA) * N * P * B
    where W is the polar motion matrix (identity without EOP).

    Parameters
    ----------
    t : AstroTime
        Epoch for the transformation.
    ut1_utc : float
        UT1-UTC correction in seconds. Default 0.0 (no EOP).
    nutation_model : str
        "2000A" (default) or "2000B".

    Returns
    -------
    M : 3x3 rotation matrix (tuple of tuples)
    """
    t_tt = t.to_julian_centuries_tt()

    B = frame_bias_matrix()
    P = precession_matrix(t_tt)

    dpsi, deps = nutation_angles(t_tt, model=nutation_model)
    eps0 = mean_obliquity(t_tt)
    N = nutation_matrix(dpsi, deps, eps0)

    from humeris.domain.time_systems import datetime_to_jd
    dt_utc = t.to_utc_datetime()
    jd_utc = datetime_to_jd(dt_utc)
    jd_ut1 = jd_utc + ut1_utc / 86400.0

    era = earth_rotation_angle(jd_ut1)
    R_era = _r3(era)

    NPB = _mat_mul(N, _mat_mul(P, B))
    return _mat_mul(R_era, NPB)


def eci_to_ecef_precise(
    pos_eci: tuple[float, float, float],
    t: AstroTime,
    ut1_utc: float = 0.0,
    nutation_model: str = "2000A",
) -> tuple[float, float, float]:
    """High-fidelity GCRS -> ITRS position transformation.

    Parameters
    ----------
    pos_eci : tuple[float, float, float]
        Position in GCRS/ECI frame (meters).
    t : AstroTime
        Epoch.
    ut1_utc : float
        UT1-UTC correction in seconds.
    nutation_model : str
        "2000A" (default) or "2000B".

    Returns
    -------
    Position in ITRS/ECEF frame (meters).
    """
    M = gcrs_to_itrs_matrix(t, ut1_utc, nutation_model=nutation_model)
    return _mat_vec(M, pos_eci)
