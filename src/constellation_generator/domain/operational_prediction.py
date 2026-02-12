# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Operational predictions composing survival/propellant/scheduling outputs.

End-of-life mode prediction (fuel vs reentry race) and maneuver-contact
feasibility assessment.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.statistical_analysis import (
    LifetimeSurvivalCurve,
    MissionAvailabilityProfile,
)
from constellation_generator.domain.mission_analysis import PropellantProfile
from constellation_generator.domain.maintenance_planning import MaintenanceSchedule
from constellation_generator.domain.propagation import OrbitalState, propagate_ecef_to
from constellation_generator.domain.observation import GroundStation, compute_observation
from constellation_generator.domain.access_windows import compute_access_windows


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
