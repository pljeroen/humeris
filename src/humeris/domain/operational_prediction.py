# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Operational predictions composing survival/propellant/scheduling outputs.

End-of-life mode prediction (fuel vs reentry race) and maneuver-contact
feasibility assessment.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from humeris.domain.statistical_analysis import (
    LifetimeSurvivalCurve,
    MissionAvailabilityProfile,
)
from humeris.domain.mission_analysis import PropellantProfile
from humeris.domain.maintenance_planning import MaintenanceSchedule
from humeris.domain.propagation import OrbitalState, propagate_ecef_to
from humeris.domain.observation import GroundStation, compute_observation
from humeris.domain.access_windows import compute_access_windows
from humeris.domain.nrlmsise00 import (
    NRLMSISE00Model,
    SpaceWeather,
    SpaceWeatherHistory,
)
from humeris.domain.atmosphere import DragConfig, atmospheric_density
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.maneuver_detection import ManeuverDetectionResult
from humeris.domain.kessler_heatmap import KesslerHeatMap
from humeris.domain.station_keeping import propellant_mass_for_dv


@dataclass(frozen=True)
class EndOfLifePrediction:
    """Prediction of satellite end-of-life mode."""
    fuel_depletion_time: datetime | None
    reentry_time: datetime | None
    end_of_life_mode: str
    hazard_ratio_at_eol: float
    crossover_time: datetime | None
    controlled_deorbit_feasible: bool


@dataclass(frozen=True)
class ManeuverContactWindow:
    """Assessment of ground contact during a maintenance burn."""
    burn_time: datetime
    burn_description: str
    has_ground_contact: bool
    contact_margin_s: float


@dataclass(frozen=True)
class ManeuverContactFeasibility:
    """Overall maneuver-contact feasibility for a maintenance schedule."""
    windows: tuple
    feasible_count: int
    infeasible_count: int
    feasibility_fraction: float


def compute_end_of_life_mode(
    survival_curve: LifetimeSurvivalCurve,
    propellant_profile: PropellantProfile,
    availability: MissionAvailabilityProfile,
) -> EndOfLifePrediction:
    """Determine end-of-life mode from survival/propellant/availability data.

    Compares fuel depletion time vs reentry time to determine EOL mode.
    controlled_deorbit_feasible = True when fuel depletion comes after reentry.

    Returns one of: "fuel_depletion", "reentry", "conjunction", "indeterminate".
    """
    fuel_time = propellant_profile.depletion_time

    # Reentry time: when survival fraction drops to ~0
    reentry_time = None
    if survival_curve.times and survival_curve.survival_fraction:
        for i, s in enumerate(survival_curve.survival_fraction):
            if s <= 0.01:
                reentry_time = survival_curve.times[i]
                break
        if reentry_time is None and len(survival_curve.times) > 1:
            # Extrapolate: use mean remaining life
            if survival_curve.mean_remaining_life_days > 0:
                reentry_time = survival_curve.times[0] + timedelta(
                    days=survival_curve.mean_remaining_life_days,
                )

    # Hazard ratio at EOL: use last available hazard rate
    hazard_at_eol = 0.0
    if survival_curve.hazard_rate_per_day:
        non_zero = [h for h in survival_curve.hazard_rate_per_day if h > 0]
        if non_zero:
            hazard_at_eol = non_zero[-1]

    # Crossover: where fuel depletion and reentry times intersect
    crossover_time = None
    if fuel_time and reentry_time:
        crossover_time = min(fuel_time, reentry_time)

    # Determine mode
    # Check conjunction first (from availability critical factor)
    if availability.critical_factor == "conjunction":
        # Check if conjunction dominates
        if availability.conjunction_survival and availability.conjunction_survival[-1] < 0.5:
            mode = "conjunction"
        elif fuel_time is None and reentry_time is None:
            mode = "conjunction"
        elif fuel_time and reentry_time:
            if fuel_time < reentry_time:
                mode = "fuel_depletion"
            else:
                mode = "reentry"
        elif fuel_time:
            mode = "fuel_depletion"
        elif reentry_time:
            mode = "reentry"
        else:
            mode = "conjunction"
    elif fuel_time and reentry_time:
        if fuel_time < reentry_time:
            mode = "fuel_depletion"
        else:
            mode = "reentry"
    elif fuel_time:
        mode = "fuel_depletion"
    elif reentry_time:
        mode = "reentry"
    else:
        mode = "indeterminate"

    # Controlled deorbit feasible: fuel remains at reentry time
    # i.e., fuel depletion comes after reentry
    controlled_deorbit_feasible = False
    if fuel_time and reentry_time:
        controlled_deorbit_feasible = fuel_time > reentry_time
    elif reentry_time and fuel_time is None:
        # Never runs out of fuel
        controlled_deorbit_feasible = True

    return EndOfLifePrediction(
        fuel_depletion_time=fuel_time,
        reentry_time=reentry_time,
        end_of_life_mode=mode,
        hazard_ratio_at_eol=hazard_at_eol,
        crossover_time=crossover_time,
        controlled_deorbit_feasible=controlled_deorbit_feasible,
    )


def compute_maneuver_contact_feasibility(
    schedule: MaintenanceSchedule,
    states: list,
    stations: list,
    epoch: datetime,
    contact_margin_s: float = 300.0,
) -> ManeuverContactFeasibility:
    """Assess whether maintenance burns have ground contact.

    For each burn in the schedule, checks if any ground station has
    visibility within contact_margin_s of the burn time.
    """
    windows: list[ManeuverContactWindow] = []

    for burn in schedule.burns:
        has_contact = False
        best_margin = float('inf')

        # Check each station for visibility around burn time
        for station in stations:
            for state in states:
                # Check if satellite is visible from station at burn time
                search_start = burn.time - timedelta(seconds=contact_margin_s)
                search_duration = timedelta(seconds=2 * contact_margin_s)
                search_step = timedelta(seconds=30)

                access = compute_access_windows(
                    station, state, search_start, search_duration, search_step,
                    min_elevation_deg=10.0,
                )

                for aw in access:
                    # Check if burn time falls within access window (with margin)
                    margin_start = (burn.time - aw.rise_time).total_seconds()
                    margin_end = (aw.set_time - burn.time).total_seconds()

                    if margin_start >= -contact_margin_s and margin_end >= -contact_margin_s:
                        has_contact = True
                        actual_margin = min(abs(margin_start), abs(margin_end))
                        if actual_margin < best_margin:
                            best_margin = actual_margin

        if not has_contact:
            best_margin = 0.0

        windows.append(ManeuverContactWindow(
            burn_time=burn.time,
            burn_description=burn.description,
            has_ground_contact=has_contact,
            contact_margin_s=best_margin,
        ))

    feasible = sum(1 for w in windows if w.has_ground_contact)
    infeasible = len(windows) - feasible
    fraction = feasible / len(windows) if windows else 0.0

    return ManeuverContactFeasibility(
        windows=tuple(windows),
        feasible_count=feasible,
        infeasible_count=infeasible,
        feasibility_fraction=fraction,
    )


# --------------------------------------------------------------------------- #
# Solar-Aware End-of-Life Prediction
# --------------------------------------------------------------------------- #

_SECONDS_PER_YEAR = 365.25 * 86400.0
_G0 = 9.80665
_REENTRY_ALTITUDE_KM = 100.0
# Typical dV per avoidance maneuver (m/s) for LEO conjunction avoidance
_TYPICAL_MANEUVER_DV_MS = 0.5


@dataclass(frozen=True)
class SolarDecayPoint:
    """Point in solar-cycle-aware decay profile."""
    time: datetime
    altitude_km: float
    semi_major_axis_m: float
    density_kg_m3: float
    f107: float
    dv_per_year_ms: float
    propellant_remaining_kg: float


@dataclass(frozen=True)
class SolarAwareEOL:
    """EOL prediction incorporating solar cycle and operational data."""
    end_of_life_time: datetime | None
    end_of_life_mode: str
    fuel_depletion_time: datetime | None
    reentry_time: datetime | None
    controlled_deorbit_feasible: bool
    solar_cycle_phase: str
    mean_f107: float
    density_ratio_vs_static: float
    maneuver_adjusted_dv_per_year_ms: float
    observed_maneuver_rate_per_year: float
    cascade_k_eff: float
    environment_risk_level: str
    decay_profile: tuple


def _orbit_averaged_density(
    altitude_km: float,
    epoch: datetime,
    model: NRLMSISE00Model,
    space_weather: SpaceWeather,
) -> float:
    """Compute orbit-averaged NRLMSISE-00 density.

    Uses orbit-averaged parameters: lat=0, UT=28800s, lon=0 giving
    LST=8h and diurnal_factor=1.0, latitude_factor=1.0, seasonal_factor=1.0.
    The only time-varying drivers are F10.7 and Ap from the space weather.
    """
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    tt = epoch.timetuple()
    state = model.evaluate(
        altitude_km=altitude_km,
        latitude_deg=0.0,
        longitude_deg=0.0,
        year=tt.tm_year,
        day_of_year=tt.tm_yday,
        ut_seconds=28800.0,
        space_weather=space_weather,
    )
    return state.total_density_kg_m3


def _classify_solar_phase(mean_f107: float, f107_values: list[float]) -> str:
    """Classify solar cycle phase from F10.7 statistics."""
    if mean_f107 < 90.0:
        return "minimum"
    if mean_f107 >= 130.0:
        return "maximum"
    # Ascending vs descending: check trend
    if len(f107_values) >= 2:
        first_half = sum(f107_values[:len(f107_values) // 2]) / max(len(f107_values) // 2, 1)
        second_half = sum(f107_values[len(f107_values) // 2:]) / max(len(f107_values) - len(f107_values) // 2, 1)
        if second_half > first_half:
            return "ascending"
        return "descending"
    return "ascending"


def _classify_environment_risk(
    kessler_heatmap: KesslerHeatMap | None,
    conjunction_weather_index: float,
) -> str:
    """Classify combined environment risk from cascade and CWI data."""
    # Cascade risk
    cascade_risk = "low"
    if kessler_heatmap is not None:
        if kessler_heatmap.is_supercritical:
            cascade_risk = "critical"
        elif kessler_heatmap.cascade_k_eff > 0.7:
            cascade_risk = "high"
        elif kessler_heatmap.cascade_k_eff > 0.3:
            cascade_risk = "moderate"

    # CWI risk
    cwi_risk = "low"
    if conjunction_weather_index >= 0.7:
        cwi_risk = "critical"
    elif conjunction_weather_index >= 0.5:
        cwi_risk = "high"
    elif conjunction_weather_index >= 0.3:
        cwi_risk = "moderate"

    # Worst of the two
    level_order = {"low": 0, "moderate": 1, "high": 2, "critical": 3}
    if level_order.get(cascade_risk, 0) >= level_order.get(cwi_risk, 0):
        return cascade_risk
    return cwi_risk


def compute_solar_aware_eol(
    state: OrbitalState,
    drag_config: DragConfig,
    epoch: datetime,
    isp_s: float,
    dry_mass_kg: float,
    propellant_budget_kg: float,
    space_weather_history: SpaceWeatherHistory | None = None,
    maneuver_result: ManeuverDetectionResult | None = None,
    kessler_heatmap: KesslerHeatMap | None = None,
    conjunction_weather_index: float = 0.0,
    mission_years: float = 25.0,
    step_days: float = 30.0,
    static_space_weather: SpaceWeather | None = None,
) -> SolarAwareEOL:
    """Predict end-of-life using NRLMSISE-00 solar-varying density.

    Integrates orbit decay with solar-activity-dependent atmospheric density,
    tracks propellant consumption with maneuver frequency adjustments, and
    factors in Kessler cascade / CWI environment risk.

    Args:
        state: Initial orbital state.
        drag_config: Satellite drag configuration.
        epoch: Simulation start time.
        isp_s: Engine specific impulse (seconds).
        dry_mass_kg: Satellite dry mass (kg).
        propellant_budget_kg: Total propellant budget (kg).
        space_weather_history: Historical F10.7/Ap data for time-varying lookup.
        maneuver_result: Detected maneuver events (optional).
        kessler_heatmap: Kessler cascade data (optional).
        conjunction_weather_index: CWI score 0-1 (optional).
        mission_years: Simulation horizon (years).
        step_days: Integration step size (days).
        static_space_weather: Override space weather (for testing). When set,
            used instead of space_weather_history for all steps.

    Returns:
        SolarAwareEOL with decay profile and EOL determination.
    """
    nrlmsise = NRLMSISE00Model()
    bc = drag_config.ballistic_coefficient

    # Default space weather if neither history nor static provided
    default_sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)

    # Maneuver rate computation
    maneuver_rate_per_year = 0.0
    if maneuver_result is not None and len(maneuver_result.events) >= 1:
        events = maneuver_result.events
        if len(events) >= 2:
            span_s = (events[-1].detection_time - events[0].detection_time).total_seconds()
            span_years = span_s / _SECONDS_PER_YEAR
            if span_years > 0:
                maneuver_rate_per_year = len(events) / span_years
        else:
            maneuver_rate_per_year = 1.0  # single event, assume ~1/year

    extra_dv_per_year = maneuver_rate_per_year * _TYPICAL_MANEUVER_DV_MS

    # Integration state
    a = state.semi_major_axis_m
    propellant_remaining = propellant_budget_kg
    dt_s = step_days * 86400.0
    current_time = epoch
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    end_time = current_time + timedelta(seconds=mission_years * _SECONDS_PER_YEAR)

    profile: list[SolarDecayPoint] = []
    f107_values: list[float] = []
    density_sum = 0.0
    density_count = 0

    fuel_depletion_time: datetime | None = None
    reentry_time: datetime | None = None

    while current_time < end_time:
        alt_km = (a - OrbitalConstants.R_EARTH) / 1000.0

        # Check reentry
        if alt_km <= _REENTRY_ALTITUDE_KM:
            if reentry_time is None:
                reentry_time = current_time
            break

        # Get space weather for this epoch
        if static_space_weather is not None:
            sw = static_space_weather
        elif space_weather_history is not None:
            sw = space_weather_history.lookup(current_time)
        else:
            sw = default_sw

        f107_values.append(sw.f107_average)

        # NRLMSISE-00 orbit-averaged density
        rho = _orbit_averaged_density(alt_km, current_time, nrlmsise, sw)
        density_sum += rho
        density_count += 1

        # Orbital velocity and mean motion
        v = float(np.sqrt(OrbitalConstants.MU_EARTH / a))
        n = float(np.sqrt(OrbitalConstants.MU_EARTH / a ** 3))

        # Decay rate: da/dt = -rho * v * Bc * a
        da_dt = -rho * v * bc * a

        # dV/year from drag compensation: |da/dt| * n/2 * seconds_per_year
        dv_per_year_drag = abs(da_dt) * n / 2.0 * _SECONDS_PER_YEAR
        total_dv_per_year = dv_per_year_drag + extra_dv_per_year

        # Propellant consumption for this step
        dv_this_step = total_dv_per_year * (dt_s / _SECONDS_PER_YEAR)
        if dv_this_step > 0 and isp_s > 0:
            prop_consumed = propellant_mass_for_dv(isp_s, dry_mass_kg, dv_this_step)
        else:
            prop_consumed = 0.0

        profile.append(SolarDecayPoint(
            time=current_time,
            altitude_km=alt_km,
            semi_major_axis_m=a,
            density_kg_m3=rho,
            f107=sw.f107_average,
            dv_per_year_ms=total_dv_per_year,
            propellant_remaining_kg=propellant_remaining,
        ))

        # Update propellant
        propellant_remaining -= prop_consumed
        if propellant_remaining <= 0.0 and fuel_depletion_time is None:
            fuel_depletion_time = current_time
            propellant_remaining = 0.0

        # Update semi-major axis (forward Euler)
        a += da_dt * dt_s
        current_time += timedelta(seconds=dt_s)

    # Final point if we exited due to time limit
    if reentry_time is None and current_time >= end_time:
        alt_km = (a - OrbitalConstants.R_EARTH) / 1000.0
        if alt_km <= _REENTRY_ALTITUDE_KM:
            reentry_time = current_time

    # Compute mean F10.7
    mean_f107 = sum(f107_values) / len(f107_values) if f107_values else 150.0

    # Solar cycle phase
    solar_phase = _classify_solar_phase(mean_f107, f107_values)

    # Density ratio vs static exponential model (at initial altitude)
    initial_alt_km = (state.semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
    initial_nrlmsise_density = profile[0].density_kg_m3 if profile else 0.0
    try:
        static_density = atmospheric_density(initial_alt_km)
    except ValueError:
        static_density = initial_nrlmsise_density  # fallback
    density_ratio = initial_nrlmsise_density / static_density if static_density > 0 else 1.0

    # Mean adjusted dV/year from first profile point (representative)
    if profile:
        adjusted_dv = profile[0].dv_per_year_ms
    else:
        adjusted_dv = extra_dv_per_year

    # Cascade k_eff
    cascade_k_eff = kessler_heatmap.cascade_k_eff if kessler_heatmap is not None else 0.0

    # Environment risk
    env_risk = _classify_environment_risk(kessler_heatmap, conjunction_weather_index)

    # EOL mode determination: whichever comes first
    cascade_critical_time: datetime | None = None
    if cascade_k_eff > 1.0:
        # Supercritical environment — flag immediately
        cascade_critical_time = epoch

    candidates: list[tuple[datetime, str]] = []
    if fuel_depletion_time is not None:
        candidates.append((fuel_depletion_time, "fuel_depletion"))
    if reentry_time is not None:
        candidates.append((reentry_time, "reentry"))
    if cascade_critical_time is not None:
        candidates.append((cascade_critical_time, "cascade_critical"))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        eol_time = candidates[0][0]
        eol_mode = candidates[0][1]
    else:
        eol_time = None
        eol_mode = "indeterminate"

    # Controlled deorbit feasible: fuel remains at reentry
    controlled_deorbit_feasible = False
    if reentry_time is not None:
        if fuel_depletion_time is None:
            controlled_deorbit_feasible = True
        elif fuel_depletion_time > reentry_time:
            controlled_deorbit_feasible = True

    return SolarAwareEOL(
        end_of_life_time=eol_time,
        end_of_life_mode=eol_mode,
        fuel_depletion_time=fuel_depletion_time,
        reentry_time=reentry_time,
        controlled_deorbit_feasible=controlled_deorbit_feasible,
        solar_cycle_phase=solar_phase,
        mean_f107=mean_f107,
        density_ratio_vs_static=density_ratio,
        maneuver_adjusted_dv_per_year_ms=adjusted_dv,
        observed_maneuver_rate_per_year=maneuver_rate_per_year,
        cascade_k_eff=cascade_k_eff,
        environment_risk_level=env_risk,
        decay_profile=tuple(profile),
    )
