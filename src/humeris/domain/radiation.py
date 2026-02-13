# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Radiation environment modeling.

Parametric radiation belt model using dipole L-shell approximation,
Gaussian proton/electron flux profiles, and SAA detection.
Provides point assessments and orbit-averaged radiation summaries.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)

_R_EARTH_KM = OrbitalConstants.R_EARTH / 1000.0

# Dipole axis: geographic ~79°N, 72°W (tilted ~11.5° from rotation axis)
_DIPOLE_TILT_DEG = 11.5
_DIPOLE_LON_DEG = -72.0

# Reference flux amplitudes (particles/cm²/s)
_F_PROTON_0 = 1e5    # inner belt peak
_F_ELECTRON_0 = 1e6  # outer belt peak

# Dose conversion coefficients (rad·cm²/particle)
_C_PROTON = 1e-9
_C_ELECTRON = 1e-10

# SAA geographic bounds (approximate)
_SAA_LAT_MIN = -45.0
_SAA_LAT_MAX = -15.0
_SAA_LON_MIN = -90.0
_SAA_LON_MAX = 0.0
_SAA_ALT_MIN_KM = 200.0
_SAA_ALT_MAX_KM = 600.0


@dataclass(frozen=True)
class RadiationEnvironment:
    """Point radiation assessment."""
    proton_flux_cm2_s: float
    electron_flux_cm2_s: float
    total_dose_rate_rad_s: float
    l_shell: float
    is_in_saa: bool


@dataclass(frozen=True)
class OrbitRadiationSummary:
    """Orbit-averaged radiation summary."""
    mean_dose_rate_rad_s: float
    max_dose_rate_rad_s: float
    annual_dose_rad: float
    saa_fraction: float
    max_l_shell: float


def compute_l_shell(lat_deg: float, lon_deg: float, alt_km: float) -> float:
    """McIlwain L-shell parameter (dipole approximation).

    L = r / (R_E · cos²(λ_m))

    where λ_m is geomagnetic latitude from offset dipole.

    Args:
        lat_deg: Geographic latitude (degrees).
        lon_deg: Geographic longitude (degrees).
        alt_km: Altitude above Earth surface (km).

    Returns:
        L-shell parameter (dimensionless).
    """
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    tilt_r = np.radians(_DIPOLE_TILT_DEG)
    ref_lon_r = np.radians(_DIPOLE_LON_DEG)

    # Geomagnetic latitude
    sin_lam_m = float(
        np.sin(lat_r) * np.cos(tilt_r)
        + np.cos(lat_r) * np.sin(tilt_r) * np.cos(lon_r - ref_lon_r)
    )
    sin_lam_m = max(-1.0, min(1.0, sin_lam_m))
    lam_m = float(np.arcsin(sin_lam_m))

    cos_lam_m = float(np.cos(lam_m))
    if abs(cos_lam_m) < 1e-10:
        cos_lam_m = 1e-10

    r = _R_EARTH_KM + alt_km
    return r / (_R_EARTH_KM * cos_lam_m * cos_lam_m)


def compute_radiation_environment(
    lat_deg: float,
    lon_deg: float,
    alt_km: float,
) -> RadiationEnvironment:
    """Point radiation assessment at given location.

    Uses parametric Gaussian flux profiles peaked at L≈1.5 (protons)
    and L≈4.5 (electrons).

    Args:
        lat_deg: Geographic latitude (degrees).
        lon_deg: Geographic longitude (degrees).
        alt_km: Altitude above Earth surface (km).

    Returns:
        RadiationEnvironment with fluxes, dose rate, L-shell, SAA flag.
    """
    l_val = compute_l_shell(lat_deg, lon_deg, alt_km)

    # Below 200 km: atmosphere absorbs
    if alt_km < 200.0:
        return RadiationEnvironment(
            proton_flux_cm2_s=0.0, electron_flux_cm2_s=0.0,
            total_dose_rate_rad_s=0.0, l_shell=l_val, is_in_saa=False,
        )

    # Above 60,000 km: negligible
    if alt_km > 60_000.0:
        return RadiationEnvironment(
            proton_flux_cm2_s=0.0, electron_flux_cm2_s=0.0,
            total_dose_rate_rad_s=0.0, l_shell=l_val, is_in_saa=False,
        )

    # Proton flux: inner belt, peaks at L≈1.5
    f_proton = float(_F_PROTON_0 * np.exp(-((l_val - 1.5) / 0.4) ** 2))

    # Electron flux: outer belt, peaks at L≈4.5
    f_electron = float(_F_ELECTRON_0 * np.exp(-((l_val - 4.5) / 1.5) ** 2))

    # Total dose rate
    dose_rate = _C_PROTON * f_proton + _C_ELECTRON * f_electron

    # SAA detection: inner belt dip at specific geographic region
    in_saa = (
        _SAA_LAT_MIN <= lat_deg <= _SAA_LAT_MAX
        and _SAA_LON_MIN <= lon_deg <= _SAA_LON_MAX
        and _SAA_ALT_MIN_KM <= alt_km <= _SAA_ALT_MAX_KM
        and l_val < 2.0
    )

    return RadiationEnvironment(
        proton_flux_cm2_s=f_proton,
        electron_flux_cm2_s=f_electron,
        total_dose_rate_rad_s=dose_rate,
        l_shell=l_val,
        is_in_saa=in_saa,
    )


def compute_orbit_radiation_summary(
    state: OrbitalState,
    epoch: datetime,
    num_points: int = 360,
) -> OrbitRadiationSummary:
    """Orbit-averaged radiation summary over one orbital period.

    Sweeps one period at num_points intervals, converts each position
    to geodetic, and evaluates the radiation environment.

    Args:
        state: Satellite orbital state.
        epoch: Reference epoch.
        num_points: Number of sample points around orbit.

    Returns:
        OrbitRadiationSummary with mean/max dose, annual dose, SAA fraction.
    """
    T = 2.0 * np.pi / state.mean_motion_rad_s
    dt = T / num_points

    dose_rates: list[float] = []
    saa_count = 0
    max_l = 0.0

    for i in range(num_points):
        t = epoch + timedelta(seconds=i * dt)
        pos_eci, vel_eci = propagate_to(state, t)

        gmst_angle = gmst_rad(t)
        pos_ecef, _ = eci_to_ecef(
            (pos_eci[0], pos_eci[1], pos_eci[2]),
            (vel_eci[0], vel_eci[1], vel_eci[2]),
            gmst_angle,
        )
        lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)
        alt_km = alt_m / 1000.0

        env = compute_radiation_environment(lat_deg, lon_deg, alt_km)
        dose_rates.append(env.total_dose_rate_rad_s)
        if env.is_in_saa:
            saa_count += 1
        if env.l_shell > max_l:
            max_l = env.l_shell

    mean_dose = sum(dose_rates) / len(dose_rates)
    max_dose = max(dose_rates)
    annual_dose = mean_dose * 365.25 * 86400.0
    saa_frac = saa_count / num_points

    return OrbitRadiationSummary(
        mean_dose_rate_rad_s=mean_dose,
        max_dose_rate_rad_s=max_dose,
        annual_dose_rad=annual_dose,
        saa_fraction=saa_frac,
        max_l_shell=max_l,
    )
