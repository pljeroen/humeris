# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Simplified NRLMSISE-00 atmosphere model with solar activity dependence.

Implements a physically realistic approximation of the NRLMSISE-00 empirical
atmosphere model (Picone et al. 2002). Provides species-resolved number
densities, temperature profiles, and total mass density as functions of
altitude, latitude, longitude, time, and solar/geomagnetic activity.

Includes a drop-in NRLMSISE00DragForce class compatible with the ForceModel
protocol for use with the numerical propagation framework.

"""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_BOLTZMANN: float = 1.380649e-23  # J/K
_AVOGADRO: float = 6.02214076e23  # mol^-1
_G0: float = 9.80665  # m/s^2 standard gravity

# Species molecular masses in kg
_MASS_N2: float = 28.0134e-3 / _AVOGADRO
_MASS_O2: float = 31.9988e-3 / _AVOGADRO
_MASS_O: float = 15.999e-3 / _AVOGADRO
_MASS_HE: float = 4.0026e-3 / _AVOGADRO
_MASS_AR: float = 39.948e-3 / _AVOGADRO
_MASS_H: float = 1.008e-3 / _AVOGADRO
_MASS_N: float = 14.007e-3 / _AVOGADRO

# Species molecular masses in amu (for scale height calculation)
_AMU_N2: float = 28.0134
_AMU_O2: float = 31.9988
_AMU_O: float = 15.999
_AMU_HE: float = 4.0026
_AMU_AR: float = 39.948
_AMU_H: float = 1.008
_AMU_N: float = 14.007

_AMU_KG: float = 1.66053906660e-27  # kg per amu

# Reference number densities at 120 km (m^-3) for moderate solar activity
# Tuned to produce realistic densities across 200-1000 km altitude range
_N120_N2: float = 3.8e17
_N120_O2: float = 7.0e16
_N120_O: float = 2.5e17
_N120_HE: float = 3.4e13
_N120_AR: float = 1.8e15
_N120_H: float = 2.6e11
_N120_N: float = 5.0e10

# Alpha (thermal diffusion coefficients) for each species
_ALPHA_N2: float = 0.0
_ALPHA_O2: float = 0.0
_ALPHA_O: float = 0.0
_ALPHA_HE: float = -0.38
_ALPHA_AR: float = 0.0
_ALPHA_H: float = -0.38
_ALPHA_N: float = 0.0

# Solar activity scaling exponents (gamma)
# These control how reference densities at 120 km scale with F10.7
_GAMMA_N2: float = 0.6
_GAMMA_O2: float = 0.5
_GAMMA_O: float = 0.45
_GAMMA_HE: float = -0.4
_GAMMA_AR: float = 0.5
_GAMMA_H: float = -0.25
_GAMMA_N: float = 0.45

# Temperature at 120 km base
_T120_BASE: float = 355.0  # K

# Exospheric temperature baseline
_T_INF_BASE: float = 750.0  # K


# --------------------------------------------------------------------------- #
# Kp↔Ap Conversion (Bartels 1957)
# --------------------------------------------------------------------------- #

# Standard 28-entry quasi-logarithmic table mapping Kp to Ap.
_KP_TO_AP_TABLE: tuple[tuple[float, float], ...] = (
    (0.0, 0), (0.33, 2), (0.67, 3), (1.0, 4), (1.33, 5), (1.67, 6),
    (2.0, 7), (2.33, 9), (2.67, 12), (3.0, 15), (3.33, 18), (3.67, 22),
    (4.0, 27), (4.33, 32), (4.67, 39), (5.0, 48), (5.33, 56), (5.67, 67),
    (6.0, 80), (6.33, 94), (6.67, 111), (7.0, 132), (7.33, 154), (7.67, 179),
    (8.0, 207), (8.33, 236), (8.67, 300), (9.0, 400),
)


def kp_to_ap(kp: float) -> float:
    """Convert Kp index to Ap using Bartels (1957) table with interpolation.

    Clamps to [0, 9] range. For exact table entries returns exact Ap;
    between entries uses linear interpolation.
    """
    if kp <= 0.0:
        return 0.0
    if kp >= 9.0:
        return 400.0

    for i in range(len(_KP_TO_AP_TABLE) - 1):
        kp_lo, ap_lo = _KP_TO_AP_TABLE[i]
        kp_hi, ap_hi = _KP_TO_AP_TABLE[i + 1]
        if kp_lo <= kp <= kp_hi:
            if kp_hi == kp_lo:
                return ap_lo
            frac = (kp - kp_lo) / (kp_hi - kp_lo)
            return ap_lo + frac * (ap_hi - ap_lo)

    return 400.0


def ap_to_kp(ap: float) -> float:
    """Convert Ap index to Kp using reverse Bartels (1957) table with interpolation.

    Clamps to [0, 400] range.
    """
    if ap <= 0.0:
        return 0.0
    if ap >= 400.0:
        return 9.0

    for i in range(len(_KP_TO_AP_TABLE) - 1):
        _, ap_lo = _KP_TO_AP_TABLE[i]
        _, ap_hi = _KP_TO_AP_TABLE[i + 1]
        kp_lo = _KP_TO_AP_TABLE[i][0]
        kp_hi = _KP_TO_AP_TABLE[i + 1][0]
        if ap_lo <= ap <= ap_hi:
            if ap_hi == ap_lo:
                return kp_lo
            frac = (ap - ap_lo) / (ap_hi - ap_lo)
            return kp_lo + frac * (kp_hi - kp_lo)

    return 9.0


# --------------------------------------------------------------------------- #
# NRLMSISE-00 Nonlinear Geomagnetic Activity (Picone et al. 2002)
# --------------------------------------------------------------------------- #

# g0 parameters from Picone et al. (2002) reference implementation (pt array).
# Structure: saturating nonlinearity centered at quiet-day Ap=4.
# alpha controls saturation rate; beta controls large-disturbance slope.
# At small Ap: g0 ≈ (Ap-4) (slope 1). At large Ap: slope → beta (saturation).
_G0_ALPHA: float = 0.00513  # Saturation rate from NRLMSISE-00 pt[24]
_G0_BETA: float = 0.0867    # Large-disturbance slope from NRLMSISE-00 pt[25]

# Temporal decay for 3-hourly ap weighting.
# exp1 = decay factor per 3-hour step. 0.39 ≈ 6.5h e-folding, consistent
# with upper thermosphere thermal inertia after geomagnetic forcing.
_AP_DECAY: float = 0.39


def _g0(ap_val: float, alpha: float = _G0_ALPHA, beta: float = _G0_BETA) -> float:
    """Nonlinear geomagnetic activity transform (Picone et al. 2002).

    Maps raw Ap (centered at quiet-day value of 4) through a saturating
    nonlinearity that prevents unrealistic temperatures during extreme storms.

    g0(a) = beta*(a-4) + (beta-1)*(exp(-alpha*(a-4)) - 1)/alpha

    Linear for small disturbances, saturates for large Ap.

    References:
        Picone, Hedin, Drob, Aikin (2002) J. Geophys. Res. 107(A12), 1468
    """
    x = ap_val - 4.0
    if abs(alpha) < 1e-12:
        return beta * x

    exp_term = math.exp(-abs(alpha) * x)
    return beta * x + (beta - 1.0) * (exp_term - 1.0) / abs(alpha)


def _sumex(ex: float) -> float:
    """Normalization denominator for sg0 time-weighted sum.

    sumex(ex) = 1 + ex^0.5 * (1 - ex^19) / (1 - ex)

    The geometric series (1 - ex^19)/(1 - ex) = 1 + ex + ex^2 + ... + ex^18,
    giving 19 exponentially decaying weights when multiplied by ex^0.5.

    References:
        NRLMSISE-00 reference implementation (Picone et al. 2002)
    """
    if ex < 1e-12:
        return 2.0
    if ex > 0.99999:
        ex = 0.99999
    return 1.0 + (ex ** 0.5) * (1.0 - ex ** 19) / (1.0 - ex)


def _sg0(
    ap_array: tuple[float, ...],
    ex: float = _AP_DECAY,
    alpha: float = _G0_ALPHA,
    beta: float = _G0_BETA,
) -> float:
    """Time-weighted geomagnetic activity from 7-element Ap array.

    Applies g0 nonlinearity to each Ap element, then weights with
    exponential decay (older activity has less influence). Structure
    matches Picone (2002): recent 3h windows at ex^k, averaged historical
    windows expanded over their sub-intervals.

    ap_array elements:
        [0] daily Ap, [1] 3h current, [2] 3h ago, [3] 6h ago,
        [4] 9h ago, [5] avg 12-33h, [6] avg 36-57h

    References:
        Picone, Hedin, Drob, Aikin (2002) J. Geophys. Res. 107(A12), 1468
    """
    g = [_g0(a, alpha, beta) for a in ap_array[:7]]

    # Numerator: exponentially decaying weighted sum
    # ap[1]: current 3h (weight 1), ap[2]: 3h ago (weight ex), etc.
    # ap[5]: 12-33h avg expanded over 8 sub-intervals (ex^4 to ex^11)
    # ap[6]: 36-57h avg expanded over 8 sub-intervals (ex^12 to ex^19)
    geo_sum = (1.0 - ex ** 8) / (1.0 - ex) if ex > 1e-12 else 8.0

    numerator = (
        g[1]
        + g[2] * ex
        + g[3] * ex ** 2
        + g[4] * ex ** 3
        + (g[5] * ex ** 4 + g[6] * ex ** 12) * geo_sum
    )

    return numerator / _sumex(ex)


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SpaceWeather:
    """Solar and geomagnetic activity indices for atmosphere model.

    f107_daily: 10.7 cm solar radio flux (SFU) for the day.
    f107_average: 81-day centered average of F10.7.
    ap_daily: Planetary geomagnetic index (0-400).
    ap_array: Optional 7-element tuple for time-weighted geomagnetic history.
        Elements: daily Ap, 3h current, 6h, 9h, 12-33h avg, 36-57h avg, magnetic.
    """
    f107_daily: float
    f107_average: float
    ap_daily: float
    ap_array: tuple[float, ...] | None = None


@dataclass(frozen=True)
class AtmosphereState:
    """Complete atmosphere state at a point.

    Includes total mass density, temperature, exospheric temperature,
    and individual species number densities.
    """
    total_density_kg_m3: float
    temperature_k: float
    temperature_exospheric_k: float
    n2_density_m3: float
    o2_density_m3: float
    o_density_m3: float
    he_density_m3: float
    ar_density_m3: float
    h_density_m3: float
    n_density_m3: float


# Default moderate solar activity
_DEFAULT_SPACE_WEATHER = SpaceWeather(
    f107_daily=150.0, f107_average=150.0, ap_daily=15.0
)


# --------------------------------------------------------------------------- #
# NRLMSISE-00 Model
# --------------------------------------------------------------------------- #

class NRLMSISE00Model:
    """Simplified NRLMSISE-00 atmosphere model.

    Computes atmospheric density and temperature as a function of altitude,
    geographic position, time, and solar/geomagnetic activity. Based on
    Picone et al. 2002 with simplified parameterization.
    """

    def evaluate(
        self,
        altitude_km: float,
        latitude_deg: float,
        longitude_deg: float,
        year: int,
        day_of_year: int,
        ut_seconds: float,
        space_weather: Optional[SpaceWeather] = None,
    ) -> AtmosphereState:
        """Evaluate the atmosphere model at a given point.

        Parameters
        ----------
        altitude_km : float
            Geodetic altitude in km.
        latitude_deg : float
            Geodetic latitude in degrees (-90 to 90).
        longitude_deg : float
            Geodetic longitude in degrees (-180 to 360).
        year : int
            Calendar year.
        day_of_year : int
            Day of year (1-366).
        ut_seconds : float
            UT seconds of day (0-86400).
        space_weather : SpaceWeather | None
            Solar and geomagnetic indices. Uses moderate defaults if None.

        Returns
        -------
        AtmosphereState
            Complete atmosphere state.
        """
        if space_weather is None:
            space_weather = _DEFAULT_SPACE_WEATHER

        f107 = space_weather.f107_daily
        f107a = space_weather.f107_average
        ap = space_weather.ap_daily
        ap_array = space_weather.ap_array

        # Clamp altitude to minimum of 80 km for computation
        z = max(80.0, altitude_km)

        # --- Exospheric temperature ---
        t_inf = self._exospheric_temperature(f107, f107a, ap, ap_array)

        # --- Temperature at 120 km (weakly dependent on F10.7) ---
        t120 = _T120_BASE + 0.03 * (f107a - 150.0)

        # --- Temperature profile (Bates-Walker) ---
        # s controls how quickly T approaches T_inf with altitude
        # Typical values: 0.008-0.020 km^-1
        s = 0.010 + 0.000015 * (t_inf - _T_INF_BASE)
        if s < 0.008:
            s = 0.008

        if z <= 120.0:
            # Below 120 km: linear approximation from ~200K at 80km to T120
            frac = (z - 80.0) / 40.0
            temperature = 200.0 + frac * (t120 - 200.0)
            temperature = max(180.0, temperature)
        else:
            temperature = t_inf - (t_inf - t120) * float(np.exp(-s * (z - 120.0)))

        # --- Local solar time ---
        lst_hours = (ut_seconds / 3600.0) + (longitude_deg / 15.0)
        lst_hours = lst_hours % 24.0

        # --- Diurnal variation factor ---
        # Peak at ~14h LST (2 PM), minimum at ~4h LST
        hour_angle = 2.0 * math.pi * (lst_hours - 14.0) / 24.0
        diurnal_factor = 1.0 + 0.35 * float(np.cos(hour_angle))

        # --- Latitude factor ---
        # Slight bulge at equator due to solar heating
        lat_rad = float(np.radians(latitude_deg))
        latitude_factor = 1.0 - 0.08 * abs(float(np.sin(lat_rad)))

        # --- Seasonal/annual variation ---
        # Approximate: higher density in summer hemisphere at solstices
        # Day 172 ~ June 21, day 355 ~ Dec 21
        annual_phase = 2.0 * math.pi * (day_of_year - 172.0) / 365.25
        seasonal_factor = 1.0 + 0.05 * float(np.cos(annual_phase)) * float(np.sin(lat_rad))

        # --- Magnetic activity factor ---
        # Stronger effect at higher altitudes
        # Apply g0 nonlinear saturation to prevent unrealistic density
        # enhancement at extreme Ap. Use 3h current Ap when available.
        ap_mag = ap_array[1] if (ap_array is not None and len(ap_array) >= 7) else ap
        g0_mag = _g0(ap_mag)
        if z > 200.0:
            ap_scale = 1.0 + 0.008 * g0_mag * (1.0 + 0.002 * (z - 200.0))
        else:
            ap_scale = 1.0 + 0.003 * g0_mag

        # Combined atmospheric factor (applied to densities)
        atm_factor = diurnal_factor * latitude_factor * seasonal_factor * ap_scale

        # --- Species densities above 120 km ---
        if z >= 120.0:
            densities = self._compute_species_densities(
                z, temperature, t120, t_inf, f107, f107a, atm_factor
            )
        else:
            # Below 120 km: exponential extrapolation from 120 km values
            densities_120 = self._compute_species_densities(
                120.0, t120, t120, t_inf, f107, f107a, atm_factor
            )
            # Scale exponentially down from 120 km using a mean scale height
            mean_scale_height = 8.0 + 0.1 * (z - 80.0)  # km, ~8 km at 80km
            scale = float(np.exp(-(120.0 - z) / mean_scale_height))
            densities = tuple(d * scale for d in densities_120)

        n_n2, n_o2, n_o, n_he, n_ar, n_h, n_n = densities

        # --- Total mass density ---
        total_density = (
            n_n2 * _MASS_N2
            + n_o2 * _MASS_O2
            + n_o * _MASS_O
            + n_he * _MASS_HE
            + n_ar * _MASS_AR
            + n_h * _MASS_H
            + n_n * _MASS_N
        )

        return AtmosphereState(
            total_density_kg_m3=total_density,
            temperature_k=temperature,
            temperature_exospheric_k=t_inf,
            n2_density_m3=n_n2,
            o2_density_m3=n_o2,
            o_density_m3=n_o,
            he_density_m3=n_he,
            ar_density_m3=n_ar,
            h_density_m3=n_h,
            n_density_m3=n_n,
        )

    def _exospheric_temperature(
        self,
        f107: float,
        f107a: float,
        ap: float,
        ap_array: tuple[float, ...] | None = None,
    ) -> float:
        """Compute exospheric temperature T_inf.

        T_inf = T_inf_0 + dT_F107 + dT_Ap

        Uses the g0 nonlinear saturation (Picone et al. 2002) for the
        geomagnetic contribution. When ap_array is provided with >= 7
        elements, uses sg0 time-weighted exponential decay; otherwise
        uses scalar Ap through g0.
        """
        dt_f107 = 2.75 * (f107a - 70.0) + 0.8 * (f107 - f107a)
        if ap_array is not None and len(ap_array) >= 7:
            # 3-hourly mode: time-weighted nonlinear sum
            effective_ap = _sg0(ap_array)
            dt_ap = 1.5 * effective_ap
        else:
            # Daily mode: scalar nonlinear transform
            dt_ap = 1.5 * _g0(ap)
        return _T_INF_BASE + dt_f107 + dt_ap

    def _compute_species_densities(
        self,
        z: float,
        temperature: float,
        t120: float,
        t_inf: float,
        f107: float,
        f107a: float,
        atm_factor: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        """Compute number densities for all species at altitude z >= 120 km.

        Uses diffusive equilibrium with species-specific scale heights.
        """
        # Gravity at altitude (simple inverse-square approximation)
        r_earth_km = OrbitalConstants.R_EARTH_EQUATORIAL / 1000.0
        g_z = _G0 * (r_earth_km / (r_earth_km + z)) ** 2

        # F10.7 scaling of 120 km reference densities
        f107_ratio = f107a / 150.0

        n120_n2 = _N120_N2 * f107_ratio ** _GAMMA_N2
        n120_o2 = _N120_O2 * f107_ratio ** _GAMMA_O2
        n120_o = _N120_O * f107_ratio ** _GAMMA_O
        n120_he = _N120_HE * max(0.01, f107_ratio) ** _GAMMA_HE
        n120_ar = _N120_AR * f107_ratio ** _GAMMA_AR
        n120_h = _N120_H * max(0.01, f107_ratio) ** _GAMMA_H
        n120_n = _N120_N * f107_ratio ** _GAMMA_N

        # Also apply daily F10.7 correction (smaller effect)
        f107_daily_ratio = f107 / 150.0
        daily_corr_n2 = f107_daily_ratio ** (_GAMMA_N2 * 0.3)
        daily_corr_o2 = f107_daily_ratio ** (_GAMMA_O2 * 0.3)
        daily_corr_o = f107_daily_ratio ** (_GAMMA_O * 0.3)
        daily_corr_he = max(0.01, f107_daily_ratio) ** (_GAMMA_HE * 0.3)
        daily_corr_ar = f107_daily_ratio ** (_GAMMA_AR * 0.3)
        daily_corr_h = max(0.01, f107_daily_ratio) ** (_GAMMA_H * 0.3)
        daily_corr_n = f107_daily_ratio ** (_GAMMA_N * 0.3)

        n120_n2 *= daily_corr_n2
        n120_o2 *= daily_corr_o2
        n120_o *= daily_corr_o
        n120_he *= daily_corr_he
        n120_ar *= daily_corr_ar
        n120_h *= daily_corr_h
        n120_n *= daily_corr_n

        dz = z - 120.0

        def _species_density(
            n120: float, amu: float, alpha: float
        ) -> float:
            """Compute density of a single species using diffusive equilibrium."""
            # Scale height H = kT / (m*g) in km
            m_kg = amu * _AMU_KG
            # Use weighted average temperature for integration
            # Weight toward upper altitude for large dz (better approximation
            # of the integral of 1/H through the temperature profile)
            w = min(0.7, 0.3 + 0.001 * dz)
            t_avg = (1.0 - w) * t120 + w * temperature
            h_km = (_BOLTZMANN * t_avg) / (m_kg * g_z) / 1000.0  # convert m to km

            # Diffusive equilibrium with thermal diffusion
            temp_ratio = t120 / temperature if temperature > 1.0 else 1.0
            density = n120 * temp_ratio ** (1.0 + alpha) * float(np.exp(-dz / h_km))
            return density * atm_factor

        n_n2 = _species_density(n120_n2, _AMU_N2, _ALPHA_N2)
        n_o2 = _species_density(n120_o2, _AMU_O2, _ALPHA_O2)
        n_o = _species_density(n120_o, _AMU_O, _ALPHA_O)
        n_he = _species_density(n120_he, _AMU_HE, _ALPHA_HE)
        n_ar = _species_density(n120_ar, _AMU_AR, _ALPHA_AR)
        n_h = _species_density(n120_h, _AMU_H, _ALPHA_H)
        n_n = _species_density(n120_n, _AMU_N, _ALPHA_N)

        return (n_n2, n_o2, n_o, n_he, n_ar, n_h, n_n)


# --------------------------------------------------------------------------- #
# Space Weather History
# --------------------------------------------------------------------------- #

class SpaceWeatherHistory:
    """Historical space weather data for NRLMSISE-00.

    Loads synthetic F10.7 and Ap data from a bundled JSON file and provides
    date-based lookup with linear interpolation.
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        if data_path is None:
            data_path = str(
                Path(__file__).parent.parent / "data" / "space_weather_historical.json"
            )
        with open(data_path) as f:
            raw = json.load(f)

        self._entries: list[tuple[float, float, float, float]] = []
        for entry in raw["entries"]:
            parts = entry["date"].split("-")
            dt = datetime(
                int(parts[0]), int(parts[1]), int(parts[2]),
                tzinfo=timezone.utc,
            )
            epoch_days = (dt - datetime(2000, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0
            self._entries.append((
                epoch_days,
                float(entry["f107"]),
                float(entry["f107a"]),
                float(entry["ap"]),
            ))

    def lookup(self, dt: datetime) -> SpaceWeather:
        """Look up space weather for a given UTC datetime.

        Interpolates linearly between the two nearest data points.
        Clamps to edge values if outside the data range.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        epoch_days = (dt - datetime(2000, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0

        entries = self._entries

        if epoch_days <= entries[0][0]:
            return SpaceWeather(
                f107_daily=entries[0][1],
                f107_average=entries[0][2],
                ap_daily=entries[0][3],
            )

        if epoch_days >= entries[-1][0]:
            return SpaceWeather(
                f107_daily=entries[-1][1],
                f107_average=entries[-1][2],
                ap_daily=entries[-1][3],
            )

        # Binary search for bracket
        lo, hi = 0, len(entries) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if entries[mid][0] <= epoch_days:
                lo = mid
            else:
                hi = mid

        # Linear interpolation
        t0, f107_0, f107a_0, ap_0 = entries[lo]
        t1, f107_1, f107a_1, ap_1 = entries[hi]
        dt_frac = (epoch_days - t0) / (t1 - t0) if t1 != t0 else 0.0

        return SpaceWeather(
            f107_daily=f107_0 + dt_frac * (f107_1 - f107_0),
            f107_average=f107a_0 + dt_frac * (f107a_1 - f107a_0),
            ap_daily=ap_0 + dt_frac * (ap_1 - ap_0),
        )


# --------------------------------------------------------------------------- #
# SpaceWeatherProvider Protocol
# --------------------------------------------------------------------------- #


class SpaceWeatherProvider(Protocol):
    """Protocol for any source of space weather data."""

    def lookup(self, dt: datetime) -> SpaceWeather: ...


class PredictedSpaceWeatherProvider:
    """Space weather provider using the Hathaway solar cycle prediction model."""

    def lookup(self, dt: datetime) -> SpaceWeather:
        from humeris.domain.solar import predict_solar_activity

        pred = predict_solar_activity(dt)
        return SpaceWeather(
            f107_daily=pred.f107_predicted,
            f107_average=pred.f107_81day,
            ap_daily=pred.ap_predicted,
        )


class CompositeSpaceWeatherProvider:
    """Uses historical data before crossover date, predicted data after.

    Provides seamless transition from historical to predicted space weather.
    """

    def __init__(
        self,
        historical: SpaceWeatherProvider,
        predicted: SpaceWeatherProvider,
        crossover_date: datetime,
    ) -> None:
        self._historical = historical
        self._predicted = predicted
        self._crossover = crossover_date

    def lookup(self, dt: datetime) -> SpaceWeather:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt < self._crossover:
            return self._historical.lookup(dt)
        return self._predicted.lookup(dt)


_default_provider: CompositeSpaceWeatherProvider | None = None


def get_default_provider() -> CompositeSpaceWeatherProvider:
    """Return a composite provider with bundled historical + predicted fallback.

    Uses historical data through 2030, predicted Hathaway model after.
    The provider is cached as a module-level singleton.
    """
    global _default_provider
    if _default_provider is None:
        historical = SpaceWeatherHistory()
        predicted = PredictedSpaceWeatherProvider()
        crossover = datetime(2030, 1, 1, tzinfo=timezone.utc)
        _default_provider = CompositeSpaceWeatherProvider(
            historical, predicted, crossover,
        )
    return _default_provider


# --------------------------------------------------------------------------- #
# NRLMSISE-00 Convenience Function
# --------------------------------------------------------------------------- #


_cached_model: NRLMSISE00Model | None = None


def _get_cached_model() -> NRLMSISE00Model:
    """Return a cached NRLMSISE00Model singleton."""
    global _cached_model
    if _cached_model is None:
        _cached_model = NRLMSISE00Model()
    return _cached_model


def atmospheric_density_nrlmsise00(
    altitude_km: float,
    epoch: datetime,
    latitude_deg: float = 0.0,
    longitude_deg: float = 0.0,
    provider: Optional[SpaceWeatherProvider] = None,
) -> float:
    """One-call atmospheric density using NRLMSISE-00 with auto space weather.

    Convenience function that handles all SpaceWeather plumbing internally.

    Args:
        altitude_km: Geodetic altitude in km.
        epoch: UTC datetime for density computation.
        latitude_deg: Geodetic latitude (degrees). Default 0.
        longitude_deg: Geodetic longitude (degrees). Default 0.
        provider: Space weather provider. Uses default composite if None.

    Returns:
        Total atmospheric density in kg/m3.
    """
    if provider is None:
        provider = get_default_provider()

    sw = provider.lookup(epoch)

    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    tt = epoch.timetuple()

    model = _get_cached_model()
    state = model.evaluate(
        altitude_km=altitude_km,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        year=tt.tm_year,
        day_of_year=tt.tm_yday,
        ut_seconds=tt.tm_hour * 3600.0 + tt.tm_min * 60.0 + tt.tm_sec,
        space_weather=sw,
    )
    return state.total_density_kg_m3


# --------------------------------------------------------------------------- #
# NRLMSISE-00 Drag Force (ForceModel compatible)
# --------------------------------------------------------------------------- #

class NRLMSISE00DragForce:
    """Atmospheric drag using NRLMSISE-00 density model.

    Drop-in replacement for AtmosphericDragForce from numerical_propagation.
    Computes drag acceleration using species-resolved NRLMSISE-00 density
    instead of the static exponential model.

    Conforms to the ForceModel protocol:
        acceleration(epoch, position, velocity) -> (ax, ay, az)
    """

    def __init__(
        self,
        cd: float,
        area_m2: float,
        mass_kg: float,
        space_weather: Optional[SpaceWeather] = None,
        historical_data: Optional[SpaceWeatherHistory] = None,
    ) -> None:
        self._cd = cd
        self._area_m2 = area_m2
        self._mass_kg = mass_kg
        self._bc = cd * area_m2 / mass_kg  # ballistic coefficient
        self._space_weather = space_weather
        self._historical_data = historical_data
        self._model = NRLMSISE00Model()

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Compute drag acceleration in ECI frame.

        Parameters
        ----------
        epoch : datetime
            UTC epoch.
        position : tuple[float, float, float]
            ECI position in meters.
        velocity : tuple[float, float, float]
            ECI velocity in m/s.

        Returns
        -------
        tuple[float, float, float]
            Drag acceleration (ax, ay, az) in m/s^2.
        """
        x, y, z = position
        vx, vy, vz = velocity

        # Position magnitude and altitude
        r = float(np.linalg.norm(position))
        alt_km = (r - OrbitalConstants.R_EARTH_EQUATORIAL) / 1000.0

        if alt_km < 0.0:
            return (0.0, 0.0, 0.0)

        # Convert ECI position to geodetic
        gmst = gmst_rad(epoch)
        pos_ecef, _ = eci_to_ecef(position, velocity, gmst)
        lat_deg, lon_deg, _ = ecef_to_geodetic(pos_ecef)

        # Time parameters
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        tt = epoch.timetuple()
        year = tt.tm_year
        doy = tt.tm_yday
        ut_s = tt.tm_hour * 3600.0 + tt.tm_min * 60.0 + tt.tm_sec

        # Space weather lookup
        sw = self._space_weather
        if sw is None and self._historical_data is not None:
            sw = self._historical_data.lookup(epoch)

        # Evaluate NRLMSISE-00
        state = self._model.evaluate(
            altitude_km=alt_km,
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            year=year,
            day_of_year=doy,
            ut_seconds=ut_s,
            space_weather=sw,
        )

        rho = state.total_density_kg_m3

        # Relative velocity (atmosphere co-rotates with Earth)
        omega_e = OrbitalConstants.EARTH_ROTATION_RATE
        vr = np.array([vx + omega_e * y, vy - omega_e * x, vz])

        v_rel = float(np.linalg.norm(vr))
        if v_rel < 1e-10:
            return (0.0, 0.0, 0.0)

        # Drag acceleration: a = -0.5 * rho * Cd * (A/m) * v_rel * v_hat
        coeff = -0.5 * rho * self._bc * v_rel

        a = coeff * vr
        return (float(a[0]), float(a[1]), float(a[2]))
