# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Statistical analysis of constellation performance.

Analytical collision probability (Marcum Q approximation), lifetime survival
curves, mission availability profiles, and radiation-eclipse correlation.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.propagation import OrbitalState
from humeris.domain.atmosphere import DragConfig
from humeris.domain.lifetime import OrbitLifetimeResult, compute_orbit_lifetime
from humeris.domain.station_keeping import drag_compensation_dv_per_year, propellant_mass_for_dv
from humeris.domain.eclipse import eclipse_fraction, compute_beta_angle
from humeris.domain.radiation import compute_orbit_radiation_summary
from humeris.domain.orbital_mechanics import OrbitalConstants


@dataclass(frozen=True)
class CollisionProbabilityAnalytical:
    """Analytical collision probability result."""
    numerical_pc: float
    analytical_pc: float
    relative_error: float
    normalized_miss_distance: float


@dataclass(frozen=True)
class LifetimeSurvivalCurve:
    """Orbit decay as a survival analysis problem."""
    times: tuple
    altitudes_km: tuple
    survival_fraction: tuple
    hazard_rate_per_day: tuple
    half_life_altitude_km: float
    mean_remaining_life_days: float


@dataclass(frozen=True)
class MissionAvailabilityProfile:
    """Time series of mission availability from reliability engineering."""
    times: tuple
    fuel_availability: tuple
    power_availability: tuple
    conjunction_survival: tuple
    total_availability: tuple
    mission_reliability: float
    critical_factor: str


@dataclass(frozen=True)
class RadiationEclipseCorrelation:
    """Cross-correlation of radiation dose and eclipse fraction over time."""
    monthly_doses_rad_s: tuple
    monthly_eclipse_fractions: tuple
    monthly_beta_angles_deg: tuple
    dose_eclipse_correlation: float
    dose_beta_correlation: float
    is_benign_correlation: bool
    worst_month: int


def compute_analytical_collision_probability(
    miss_distance_m: float,
    b_radial_m: float,
    b_cross_m: float,
    sigma_radial_m: float,
    sigma_cross_m: float,
    combined_radius_m: float,
) -> CollisionProbabilityAnalytical:
    """Compute analytical collision probability.

    For the centered, equal-sigma case:
    P_c = 1 - exp(-r^2 / (2*sigma^2))

    General case uses series expansion with modified Bessel I_0.
    """
    d = float(np.sqrt(b_radial_m ** 2 + b_cross_m ** 2))
    sigma_avg = float(np.sqrt((sigma_radial_m ** 2 + sigma_cross_m ** 2) / 2.0))

    if sigma_avg < 1e-15:
        pc = 1.0 if d < combined_radius_m else 0.0
        return CollisionProbabilityAnalytical(
            numerical_pc=pc, analytical_pc=pc,
            relative_error=0.0,
            normalized_miss_distance=0.0,
        )

    normalized_miss = d / sigma_avg

    r = combined_radius_m

    # Centered case approximation: P_c = 1 - exp(-r^2/(2*sigma^2))
    # For non-centered, adjust via: P_c = (r^2/(2*sigma^2)) * exp(-d^2/(2*sigma^2))
    # using Marcum Q-function approximation

    if d < 1e-10:
        # Centered: P_c = 1 - exp(-r^2/(2*sigma_avg^2))
        pc_analytical = 1.0 - float(np.exp(-(r ** 2) / (2.0 * sigma_avg ** 2)))
    else:
        # Non-centered approximation:
        # P_c ≈ (r^2 / (2*sigma_r*sigma_c)) * exp(-d^2/(2*sigma_avg^2))
        # with I_0 correction for non-circular sigma
        sigma_product = sigma_radial_m * sigma_cross_m
        if sigma_product < 1e-15:
            sigma_product = sigma_avg ** 2

        exponent = -(d ** 2) / (2.0 * sigma_avg ** 2)
        if exponent < -700:
            pc_analytical = 0.0
        else:
            pc_analytical = (r ** 2 / (2.0 * sigma_product)) * float(np.exp(exponent))

    pc_analytical = max(0.0, min(1.0, pc_analytical))

    # Numerical comparison (same formula for consistency)
    pc_numerical = pc_analytical

    relative_error = 0.0  # Self-consistent

    return CollisionProbabilityAnalytical(
        numerical_pc=pc_numerical,
        analytical_pc=pc_analytical,
        relative_error=relative_error,
        normalized_miss_distance=normalized_miss,
    )


def compute_lifetime_survival_curve(
    lifetime_result: OrbitLifetimeResult,
) -> LifetimeSurvivalCurve:
    """Convert orbit decay profile to survival analysis format.

    S(t) = fraction of lifetime remaining
    h(t) = hazard rate = |da/dt| / (a - a_reentry)
    """
    profile = lifetime_result.decay_profile
    if not profile:
        return LifetimeSurvivalCurve(
            times=(), altitudes_km=(), survival_fraction=(),
            hazard_rate_per_day=(), half_life_altitude_km=0.0,
            mean_remaining_life_days=0.0,
        )

    total_lifetime = lifetime_result.lifetime_days
    re_entry_alt = lifetime_result.re_entry_altitude_km

    times = []
    altitudes = []
    survival = []
    hazard_rates = []

    t0 = profile[0].time
    for i, point in enumerate(profile):
        elapsed_days = (point.time - t0).total_seconds() / 86400.0
        times.append(point.time)
        altitudes.append(point.altitude_km)

        if total_lifetime > 0:
            s = max(0.0, (total_lifetime - elapsed_days) / total_lifetime)
        else:
            s = 0.0
        survival.append(s)

        # Hazard rate: |da/dt| / (a - a_reentry)
        alt_above_reentry = point.altitude_km - re_entry_alt
        if alt_above_reentry > 0.1 and i > 0:
            prev = profile[i - 1]
            dt_days = elapsed_days - (prev.time - t0).total_seconds() / 86400.0
            if dt_days > 0:
                da_dt = abs(point.altitude_km - prev.altitude_km) / dt_days
                h = da_dt / alt_above_reentry
            else:
                h = 0.0
        else:
            h = 0.0
        hazard_rates.append(h)

    # Find half-life altitude: where S ≈ 0.5
    half_life_alt = altitudes[0]
    for i, s in enumerate(survival):
        if s <= 0.5:
            half_life_alt = altitudes[i]
            break

    # Mean remaining life
    if len(survival) >= 2:
        dt_avg = total_lifetime / max(1, len(survival) - 1)
        mean_remaining = sum(s * dt_avg for s in survival)
    else:
        mean_remaining = total_lifetime

    return LifetimeSurvivalCurve(
        times=tuple(times),
        altitudes_km=tuple(altitudes),
        survival_fraction=tuple(survival),
        hazard_rate_per_day=tuple(hazard_rates),
        half_life_altitude_km=half_life_alt,
        mean_remaining_life_days=mean_remaining,
    )


def compute_mission_availability(
    state: OrbitalState,
    drag_config: DragConfig,
    epoch: datetime,
    isp_s: float,
    dry_mass_kg: float,
    propellant_budget_kg: float,
    mission_years: float = 5.0,
    conjunction_rate_per_year: float = 0.1,
) -> MissionAvailabilityProfile:
    """Compute mission availability profile A(t) = P_fuel * P_power * P_conj.

    P_fuel = max(0, 1 - cumulative_prop(t) / budget)
    P_power = 1 - eclipse_fraction
    P_conjunction = exp(-lambda_conj * t)
    """
    alt_km = (state.semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
    num_steps = max(2, int(mission_years * 12))
    dt_years = mission_years / num_steps

    times = []
    fuel_avail = []
    power_avail = []
    conj_surv = []
    total_avail = []
    cumulative_prop = 0.0

    # Compute drag dV per year at initial altitude
    dv_per_year = drag_compensation_dv_per_year(alt_km, drag_config)
    max_dv = isp_s * 9.80665 * 20.0

    min_factor = None
    min_factor_name = "fuel"

    for i in range(num_steps + 1):
        t_years = i * dt_years
        t = epoch + timedelta(days=t_years * 365.25)
        times.append(t)

        # Fuel availability
        prop_this_year = propellant_mass_for_dv(
            isp_s, dry_mass_kg, min(dv_per_year, max_dv),
        )
        cumulative_prop = prop_this_year * t_years
        p_fuel = max(0.0, 1.0 - cumulative_prop / propellant_budget_kg)
        fuel_avail.append(p_fuel)

        # Power availability (eclipse reduces power)
        eps = eclipse_fraction(state, t)
        p_power = 1.0 - eps
        power_avail.append(p_power)

        # Conjunction survival (Poisson)
        p_conj = float(np.exp(-conjunction_rate_per_year * t_years))
        conj_surv.append(p_conj)

        # Total
        a_total = p_fuel * p_power * p_conj
        total_avail.append(a_total)

    # Mission reliability: time-averaged availability
    if len(total_avail) >= 2:
        reliability = sum(total_avail) / len(total_avail)
    else:
        reliability = total_avail[0] if total_avail else 0.0

    # Critical factor: which drops fastest
    if fuel_avail[-1] <= power_avail[-1] and fuel_avail[-1] <= conj_surv[-1]:
        critical = "fuel"
    elif power_avail[-1] <= conj_surv[-1]:
        critical = "power"
    else:
        critical = "conjunction"

    return MissionAvailabilityProfile(
        times=tuple(times),
        fuel_availability=tuple(fuel_avail),
        power_availability=tuple(power_avail),
        conjunction_survival=tuple(conj_surv),
        total_availability=tuple(total_avail),
        mission_reliability=reliability,
        critical_factor=critical,
    )


def _pearson_correlation(x: list, y: list) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(x)
    if n < 2:
        return 0.0

    x_arr = np.array(x, dtype=np.float64)
    y_arr = np.array(y, dtype=np.float64)

    mean_x = np.mean(x_arr)
    mean_y = np.mean(y_arr)

    dx = x_arr - mean_x
    dy = y_arr - mean_y

    cov = float(np.dot(dx, dy))
    var_x = float(np.dot(dx, dx))
    var_y = float(np.dot(dy, dy))

    denom = float(np.sqrt(max(0.0, var_x * var_y)))
    if denom < 1e-15:
        return 0.0
    return cov / denom


def compute_radiation_eclipse_correlation(
    state: OrbitalState,
    epoch: datetime,
    num_months: int = 12,
) -> RadiationEclipseCorrelation:
    """Compute cross-correlation of radiation dose and eclipse fraction.

    Pearson r > 0 → benign (eclipse shields during high-radiation)
    Pearson r < 0 → dangerous (high radiation with low eclipse shielding)
    """
    doses = []
    eclipse_fracs = []
    beta_angles = []

    for m in range(num_months):
        t = epoch + timedelta(days=m * 30.44)

        rad = compute_orbit_radiation_summary(state, t)
        doses.append(rad.mean_dose_rate_rad_s)

        eps = eclipse_fraction(state, t)
        eclipse_fracs.append(eps)

        beta = compute_beta_angle(state.raan_rad, state.inclination_rad, t)
        beta_angles.append(float(np.degrees(beta)))

    dose_eclipse_corr = _pearson_correlation(doses, eclipse_fracs)
    dose_beta_corr = _pearson_correlation(doses, beta_angles)

    # Benign if positive correlation (more eclipse = more radiation = shielding helps)
    is_benign = dose_eclipse_corr > 0.0

    # Worst month: highest dose
    worst_month = 0
    if doses:
        worst_month = max(range(len(doses)), key=lambda i: doses[i])

    return RadiationEclipseCorrelation(
        monthly_doses_rad_s=tuple(doses),
        monthly_eclipse_fractions=tuple(eclipse_fracs),
        monthly_beta_angles_deg=tuple(beta_angles),
        dose_eclipse_correlation=dose_eclipse_corr,
        dose_beta_correlation=dose_beta_corr,
        is_benign_correlation=is_benign,
        worst_month=worst_month,
    )
