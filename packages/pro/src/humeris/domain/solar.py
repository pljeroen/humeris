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
    """
    f107_predicted: float
    f107_81day: float
    ap_predicted: float
    cycle_number: int
    cycle_phase: float


@dataclass(frozen=True)
class _SolarCycleParams:
    """Parameters for a single solar cycle (Hathaway model)."""
    number: int
    start_year: float  # Decimal year of cycle minimum
    amplitude: float   # Peak SSN amplitude
    rise_time: float   # Years from minimum to maximum
    duration: float    # Approximate cycle length in years


# Known solar cycle parameters (Hathaway 2015 parametric form)
# Start years, amplitudes, and rise times from NOAA/SWPC observations.
SOLAR_CYCLES: tuple[_SolarCycleParams, ...] = (
    _SolarCycleParams(number=23, start_year=1996.4, amplitude=175.0,
                      rise_time=4.0, duration=12.5),
    _SolarCycleParams(number=24, start_year=2008.9, amplitude=116.0,
                      rise_time=5.4, duration=11.0),
    _SolarCycleParams(number=25, start_year=2019.9, amplitude=155.0,
                      rise_time=4.6, duration=11.0),
    _SolarCycleParams(number=26, start_year=2030.9, amplitude=130.0,
                      rise_time=4.5, duration=11.0),
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


def _hathaway_ssn(t_years: float, amplitude: float, rise_time: float) -> float:
    """Hathaway (2015) solar cycle shape function.

    Uses a skewed Gaussian profile: SSN(t) = A * x^3 * exp(-x^2) / peak_norm
    where x = t / tau and tau is chosen so the peak occurs at t = rise_time.

    Peak of x^3 * exp(-x^2) occurs at x = sqrt(3/2), so tau = rise_time / sqrt(3/2).
    The peak is then normalized to equal amplitude.
    """
    if t_years <= 0:
        return 0.0

    tau = rise_time / math.sqrt(1.5)  # Width placing peak at rise_time
    x = t_years / tau

    # Normalization: peak of x^3 * exp(-x^2) is (3/2)^(3/2) * exp(-3/2)
    peak_norm = 1.5 ** 1.5 * math.exp(-1.5)  # ~0.40987

    ssn = amplitude * (x ** 3 * math.exp(-(x ** 2))) / peak_norm
    return max(0.0, ssn)


def _ssn_to_f107(ssn: float) -> float:
    """Convert sunspot number to F10.7 using Tapping (2013) proxy.

    F10.7 = 63.7 + 0.727 * SSN + 0.00089 * SSN^2
    """
    return 63.7 + 0.727 * ssn + 0.00089 * ssn * ssn


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

    Uses the Hathaway (2015) parametric solar cycle model with known
    parameters for cycles 23-26. Converts SSN to F10.7 via Tapping (2013).

    Args:
        epoch: UTC datetime for prediction.

    Returns:
        SolarCyclePrediction with F10.7, Ap, cycle number, and phase.
    """
    decimal_year = _epoch_to_decimal_year(epoch)
    cycle = _find_cycle(decimal_year)
    t_years = decimal_year - cycle.start_year

    # Compute SSN from Hathaway model
    ssn = _hathaway_ssn(t_years, cycle.amplitude, cycle.rise_time)

    # Also sum contributions from adjacent cycles (overlap during minima)
    for other in SOLAR_CYCLES:
        if other.number == cycle.number:
            continue
        t_other = decimal_year - other.start_year
        if 0 < t_other < other.duration + 3.0:
            ssn += _hathaway_ssn(t_other, other.amplitude, other.rise_time)

    # Convert SSN to F10.7
    f107 = _ssn_to_f107(ssn)
    f107 = max(65.0, f107)  # Physical floor: quiet Sun baseline

    # Cycle phase [0, 1]
    phase = min(1.0, max(0.0, t_years / cycle.duration))

    # 81-day average: use slightly smoothed value (model is already smooth)
    f107_81 = f107  # Model output is inherently smoothed

    # Ap estimate
    ap = _f107_to_ap(f107, phase)

    return SolarCyclePrediction(
        f107_predicted=f107,
        f107_81day=f107_81,
        ap_predicted=ap,
        cycle_number=cycle.number,
        cycle_phase=phase,
    )
