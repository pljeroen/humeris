# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see LICENSE-COMMERCIAL.md.
"""Simplified NRLMSISE-00 atmosphere model with solar activity dependence.

Implements a physically realistic approximation of the NRLMSISE-00 empirical
atmosphere model (Picone et al. 2002). Provides species-resolved number
densities, temperature profiles, and total mass density as functions of
altitude, latitude, longitude, time, and solar/geomagnetic activity.

Includes a drop-in NRLMSISE00DragForce class compatible with the ForceModel
protocol for use with the numerical propagation framework.

No external dependencies — stdlib only + domain imports.
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.coordinate_frames import (
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
# Value objects
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SpaceWeather:
    """Solar and geomagnetic activity indices for atmosphere model.

    f107_daily: 10.7 cm solar radio flux (SFU) for the day.
    f107_average: 81-day centered average of F10.7.
    ap_daily: Planetary geomagnetic index (0-400).
    """
    f107_daily: float
    f107_average: float
    ap_daily: float


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

        # Clamp altitude to minimum of 80 km for computation
        z = max(80.0, altitude_km)

        # --- Exospheric temperature ---
        t_inf = self._exospheric_temperature(f107, f107a, ap)

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
            temperature = t_inf - (t_inf - t120) * math.exp(-s * (z - 120.0))

        # --- Local solar time ---
        lst_hours = (ut_seconds / 3600.0) + (longitude_deg / 15.0)
        lst_hours = lst_hours % 24.0

        # --- Diurnal variation factor ---
        # Peak at ~14h LST (2 PM), minimum at ~4h LST
        hour_angle = 2.0 * math.pi * (lst_hours - 14.0) / 24.0
        diurnal_factor = 1.0 + 0.35 * math.cos(hour_angle)

        # --- Latitude factor ---
        # Slight bulge at equator due to solar heating
        lat_rad = math.radians(latitude_deg)
        latitude_factor = 1.0 - 0.08 * abs(math.sin(lat_rad))

        # --- Seasonal/annual variation ---
        # Approximate: higher density in summer hemisphere at solstices
        # Day 172 ~ June 21, day 355 ~ Dec 21
        annual_phase = 2.0 * math.pi * (day_of_year - 172.0) / 365.25
        seasonal_factor = 1.0 + 0.05 * math.cos(annual_phase) * math.sin(lat_rad)

        # --- Magnetic activity factor ---
        # Stronger effect at higher altitudes
        if z > 200.0:
            ap_scale = 1.0 + 0.008 * ap * (1.0 + 0.002 * (z - 200.0))
        else:
            ap_scale = 1.0 + 0.003 * ap

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
            scale = math.exp(-(120.0 - z) / mean_scale_height)
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
        self, f107: float, f107a: float, ap: float
    ) -> float:
        """Compute exospheric temperature T_inf.

        T_inf = T_inf_0 + dT_F107 + dT_Ap
        """
        dt_f107 = 2.75 * (f107a - 70.0) + 0.8 * (f107 - f107a)
        dt_ap = 1.5 * ap
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
            density = n120 * temp_ratio ** (1.0 + alpha) * math.exp(-dz / h_km)
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
        r = math.sqrt(x * x + y * y + z * z)
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
        vr_x = vx + omega_e * y
        vr_y = vy - omega_e * x
        vr_z = vz

        v_rel = math.sqrt(vr_x * vr_x + vr_y * vr_y + vr_z * vr_z)
        if v_rel < 1e-10:
            return (0.0, 0.0, 0.0)

        # Drag acceleration: a = -0.5 * rho * Cd * (A/m) * v_rel * v_hat
        coeff = -0.5 * rho * self._bc * v_rel

        return (coeff * vr_x, coeff * vr_y, coeff * vr_z)
