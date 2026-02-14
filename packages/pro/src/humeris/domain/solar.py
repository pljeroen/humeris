# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Analytical solar ephemeris.

Low-precision Sun position using Meeus "Astronomical Algorithms" Ch. 25 /
Vallado simplified algorithm. Accuracy ~1 arcminute, sufficient for
eclipse/illumination analysis.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

AU_METERS: float = 1.495978707e11  # Astronomical unit in meters

# J2000.0 reference epoch
_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class SunPosition:
    """Sun position at a given epoch."""
    position_eci_m: tuple[float, float, float]  # ECI position in meters
    right_ascension_rad: float
    declination_rad: float
    distance_m: float  # Earth-Sun distance in meters


def julian_centuries_j2000(epoch: datetime) -> float:
    """Julian centuries since J2000.0 (2000-01-01 12:00:00 UTC)."""
    dt_seconds = (epoch - _J2000).total_seconds()
    return dt_seconds / (36525.0 * 86400.0)


def sun_position_eci(epoch: datetime) -> SunPosition:
    """Low-precision analytical solar ephemeris.

    Meeus "Astronomical Algorithms" Ch. 25 / Vallado simplified algorithm.
    Accuracy: ~1 arcminute, sufficient for eclipse/illumination analysis.

    Args:
        epoch: UTC datetime for Sun position computation.

    Returns:
        SunPosition with ECI coordinates, RA, Dec, and distance.
    """
    T = julian_centuries_j2000(epoch)

    # Mean anomaly (degrees)
    M_deg = (357.5291 + 35999.0503 * T) % 360.0
    M_rad = float(np.radians(M_deg))

    # Ecliptic longitude (degrees)
    L_deg = (280.4665 + 36000.7698 * T + 1.9146 * float(np.sin(M_rad))
             + 0.0200 * float(np.sin(2.0 * M_rad))) % 360.0
    L_rad = float(np.radians(L_deg))

    # Obliquity of the ecliptic (degrees)
    eps_deg = 23.4393 - 0.01300 * T
    eps_rad = float(np.radians(eps_deg))

    # Right ascension and declination
    ra_rad = float(np.arctan2(np.cos(eps_rad) * np.sin(L_rad), np.cos(L_rad)))
    dec_rad = float(np.arcsin(np.sin(eps_rad) * np.sin(L_rad)))

    # Distance in AU
    r_au = 1.00014 - 0.01671 * float(np.cos(M_rad)) - 0.00014 * float(np.cos(2.0 * M_rad))
    distance_m = r_au * AU_METERS

    # ECI position
    cos_ra = float(np.cos(ra_rad))
    sin_ra = float(np.sin(ra_rad))
    cos_dec = float(np.cos(dec_rad))
    sin_dec = float(np.sin(dec_rad))

    x = distance_m * cos_ra * cos_dec
    y = distance_m * sin_ra * cos_dec
    z = distance_m * sin_dec

    return SunPosition(
        position_eci_m=(x, y, z),
        right_ascension_rad=ra_rad,
        declination_rad=dec_rad,
        distance_m=distance_m,
    )


def solar_declination_rad(epoch: datetime) -> float:
    """Solar declination at given epoch. Convenience wrapper."""
    return sun_position_eci(epoch).declination_rad


# --------------------------------------------------------------------------- #
# Solar Cycle Prediction (Hathaway Model)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SolarCyclePrediction:
    """Predicted solar activity at an epoch.

    f107_predicted: Predicted F10.7 solar radio flux (SFU).
    f107_81day: 81-day centered average of F10.7 (smoothed).
    ap_predicted: Predicted planetary geomagnetic index.
    cycle_number: Solar cycle number (23, 24, 25, ...).
    cycle_phase: Phase within cycle [0, 1] (0=minimum, ~0.3-0.4=maximum).
    f107_upper: Upper bound (+1σ) of F10.7 prediction.
    f107_lower: Lower bound (-1σ) of F10.7 prediction.
    """
    f107_predicted: float
    f107_81day: float
    ap_predicted: float
    cycle_number: int
    cycle_phase: float
    f107_upper: float = 0.0
    f107_lower: float = 0.0


@dataclass(frozen=True)
class _SolarCycleParams:
    """Parameters for a single solar cycle (Hathaway model)."""
    number: int
    start_year: float  # Decimal year of cycle minimum
    amplitude: float   # Peak SSN amplitude
    rise_time: float   # Years from minimum to maximum
    duration: float    # Approximate cycle length in years
    asymmetry: float = 0.8  # Hathaway (2015) asymmetry parameter c
    amplitude_upper: float = 0.0  # +1σ amplitude (from SWPC prediction panels)
    amplitude_lower: float = 0.0  # -1σ amplitude


# Known solar cycle parameters (Hathaway 2015 parametric form)
# Start years, amplitudes, and rise times from NOAA/SWPC observations.
# Uncertainty ranges: ±1σ from SWPC prediction panels.
SOLAR_CYCLES: tuple[_SolarCycleParams, ...] = (
    _SolarCycleParams(number=23, start_year=1996.4, amplitude=175.0,
                      rise_time=4.0, duration=12.5,
                      amplitude_upper=190.0, amplitude_lower=160.0),
    _SolarCycleParams(number=24, start_year=2008.9, amplitude=116.0,
                      rise_time=5.4, duration=11.0,
                      amplitude_upper=128.0, amplitude_lower=104.0),
    _SolarCycleParams(number=25, start_year=2019.9, amplitude=155.0,
                      rise_time=4.6, duration=11.0,
                      amplitude_upper=185.0, amplitude_lower=125.0),
    _SolarCycleParams(number=26, start_year=2030.9, amplitude=130.0,
                      rise_time=4.5, duration=11.0,
                      amplitude_upper=170.0, amplitude_lower=90.0),
)


def _epoch_to_decimal_year(epoch: datetime) -> float:
    """Convert datetime to decimal year."""
    year = epoch.year
    start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
    start_of_next = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    year_fraction = (
        (epoch - start_of_year).total_seconds()
        / (start_of_next - start_of_year).total_seconds()
    )
    return year + year_fraction


def _hathaway_peak_x(c: float) -> float:
    """Find x_peak where f(x) = x^3 / (exp(x^2) - c) is maximized.

    Solves exp(x^2)(3 - 2x^2) = 3c via bisection.
    For c=0, peak is at sqrt(3/2). For c>0, peak shifts left.
    """
    if c <= 0.0:
        return math.sqrt(1.5)

    # Bisection: h(x) = exp(x^2)*(3 - 2*x^2) - 3*c
    # h is positive for small x, negative for x > sqrt(3/2)
    lo, hi = 0.5, math.sqrt(1.5)
    for _ in range(60):  # 60 iterations → ~18 digits of precision
        mid = 0.5 * (lo + hi)
        h = math.exp(mid * mid) * (3.0 - 2.0 * mid * mid) - 3.0 * c
        if h > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _hathaway_ssn(
    t_years: float, amplitude: float, rise_time: float, c: float = 0.8,
) -> float:
    """Hathaway (2015) solar cycle shape function.

    Full form: R(t) = A * x^3 / (exp(x^2) - c) / peak_norm
    where x = t / tau, c is the asymmetry parameter (~0.8 per Hathaway 2015),
    and tau is chosen so the peak occurs at t = rise_time.

    The asymmetry parameter c > 0 produces a heavier declining-phase tail
    compared to the c=0 simplification, matching observed cycle behavior.

    References:
        Hathaway, Wilson, Reichmann (1994) Solar Physics 151, 177-190
        Hathaway (2015) Living Reviews in Solar Physics 12, 4
    """
    if t_years <= 0:
        return 0.0

    # Find peak location of f(x) = x^3 / (exp(x^2) - c)
    x_peak = _hathaway_peak_x(c)

    # tau places peak at t = rise_time
    tau = rise_time / x_peak
    x = t_years / tau

    # Evaluate shape function
    exp_x2 = math.exp(x * x)
    denom = exp_x2 - c
    if denom <= 0.0:
        return 0.0

    f_x = (x ** 3) / denom

    # Normalize: peak value of f(x) at x_peak
    exp_peak = math.exp(x_peak * x_peak)
    peak_norm = (x_peak ** 3) / (exp_peak - c)

    ssn = amplitude * f_x / peak_norm
    return max(0.0, ssn)


def _ssn_to_f107(ssn: float) -> float:
    """Convert sunspot number to F10.7 using ITU-R P.371 quadratic proxy.

    F10.7 = 63.7 + 0.728 * R12 + 0.00089 * R12^2

    References:
        ITU-R Recommendation P.371-8 (Choice of indices for long-term
        ionospheric predictions). Also adopted by IRI and ECSS-E-ST-10-04C.
    """
    return 63.7 + 0.728 * ssn + 0.00089 * ssn * ssn


def _f107_to_ap(f107: float, cycle_phase: float) -> float:
    """Estimate Ap from F10.7 with ~2-year lag from solar max.

    Empirical correlation: Ap ~ 5 + 0.1 * F10.7, with enhanced
    geomagnetic activity during the declining phase of the cycle.
    """
    base_ap = 5.0 + 0.1 * f107
    # Enhanced Ap during declining phase (phase 0.4-0.8)
    if 0.35 < cycle_phase < 0.85:
        decline_factor = 1.0 + 0.3 * math.sin(
            math.pi * (cycle_phase - 0.35) / 0.5
        )
        base_ap *= decline_factor
    return base_ap


def _find_cycle(decimal_year: float) -> _SolarCycleParams:
    """Find the solar cycle containing the given decimal year."""
    for i in range(len(SOLAR_CYCLES) - 1, -1, -1):
        if decimal_year >= SOLAR_CYCLES[i].start_year:
            return SOLAR_CYCLES[i]
    return SOLAR_CYCLES[0]


def predict_solar_activity(epoch: datetime) -> SolarCyclePrediction:
    """Predict solar activity (F10.7, Ap) at a given epoch.

    Uses the Hathaway (2015) parametric solar cycle model with asymmetry
    parameter c=0.8 for cycles 23-26. Converts SSN to F10.7 via ITU-R P.371.

    Args:
        epoch: UTC datetime for prediction.

    Returns:
        SolarCyclePrediction with F10.7, Ap, cycle number, and phase.
    """
    decimal_year = _epoch_to_decimal_year(epoch)
    cycle = _find_cycle(decimal_year)
    t_years = decimal_year - cycle.start_year

    # Compute SSN from Hathaway model (full form with asymmetry)
    ssn = _hathaway_ssn(t_years, cycle.amplitude, cycle.rise_time, cycle.asymmetry)

    # Also sum contributions from adjacent cycles (overlap during minima)
    for other in SOLAR_CYCLES:
        if other.number == cycle.number:
            continue
        t_other = decimal_year - other.start_year
        if 0 < t_other < other.duration + 3.0:
            ssn += _hathaway_ssn(t_other, other.amplitude, other.rise_time, other.asymmetry)

    # Convert SSN to F10.7
    f107 = _ssn_to_f107(ssn)
    f107 = max(65.0, f107)  # Physical floor: quiet Sun baseline

    # Cycle phase [0, 1]
    phase = min(1.0, max(0.0, t_years / cycle.duration))

    # 81-day average: use slightly smoothed value (model is already smooth)
    f107_81 = f107  # Model output is inherently smoothed

    # Ap estimate
    ap = _f107_to_ap(f107, phase)

    # Uncertainty bounds: run Hathaway model at ±1σ amplitudes
    ssn_upper = _hathaway_ssn(t_years, cycle.amplitude_upper, cycle.rise_time, cycle.asymmetry)
    ssn_lower = _hathaway_ssn(t_years, cycle.amplitude_lower, cycle.rise_time, cycle.asymmetry)
    for other in SOLAR_CYCLES:
        if other.number == cycle.number:
            continue
        t_other = decimal_year - other.start_year
        if 0 < t_other < other.duration + 3.0:
            ssn_upper += _hathaway_ssn(t_other, other.amplitude_upper, other.rise_time, other.asymmetry)
            ssn_lower += _hathaway_ssn(t_other, other.amplitude_lower, other.rise_time, other.asymmetry)

    f107_upper = max(65.0, _ssn_to_f107(ssn_upper))
    f107_lower = max(65.0, _ssn_to_f107(ssn_lower))

    return SolarCyclePrediction(
        f107_predicted=f107,
        f107_81day=f107_81,
        ap_predicted=ap,
        cycle_number=cycle.number,
        cycle_phase=phase,
        f107_upper=f107_upper,
        f107_lower=f107_lower,
    )
