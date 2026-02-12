# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Environment analysis compositions.

Composes eclipse, radiation, beta angle, and torque modules to produce
seasonal thermal profiles, dose profiles, radiation-optimized LTAN,
eclipse-free windows, and worst-case torque timing.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.propagation import OrbitalState, propagate_to
from constellation_generator.domain.eclipse import (
    compute_beta_angle,
    eclipse_fraction,
    is_eclipsed,
    EclipseType,
    predict_eclipse_seasons,
)
from constellation_generator.domain.solar import sun_position_eci
from constellation_generator.domain.radiation import (
    compute_orbit_radiation_summary,
)
from constellation_generator.domain.torques import (
    InertiaTensor,
    TorqueResult,
    AerodynamicTorqueResult,
    compute_gravity_gradient_torque,
    compute_aerodynamic_torque,
)
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.orbital_mechanics import OrbitalConstants

_R_EARTH = OrbitalConstants.R_EARTH
_MU = OrbitalConstants.MU_EARTH


@dataclass(frozen=True)
class ThermalMonth:
    """Thermal environment for one month."""
    month: int
    mean_beta_deg: float
    cycle_count: int
    mean_eclipse_duration_s: float
    mean_sunlit_duration_s: float


@dataclass(frozen=True)
class SeasonalThermalProfile:
    """12-month thermal environment profile."""
    months: tuple[ThermalMonth, ...]
    total_annual_cycles: int
    max_cycle_month: int
    min_cycle_month: int


@dataclass(frozen=True)
class DoseSnapshot:
    """Radiation dose at a single point in time."""
    time: datetime
    dose_rate_rad_s: float
    cumulative_dose_rad: float
    eclipse_fraction: float
    beta_deg: float


@dataclass(frozen=True)
class SeasonalDoseProfile:
    """Seasonal radiation dose profile."""
    snapshots: tuple[DoseSnapshot, ...]
    annual_dose_rad: float
    max_dose_rate_rad_s: float
    mean_eclipse_fraction: float


@dataclass(frozen=True)
class RadiationLTANPoint:
    """Radiation assessment at a specific LTAN."""
    ltan_hours: float
    annual_dose_rad: float
    saa_fraction: float


@dataclass(frozen=True)
class RadiationLTANResult:
    """Radiation-optimized LTAN result."""
    points: tuple[RadiationLTANPoint, ...]
    optimal_ltan_hours: float
    min_annual_dose_rad: float


@dataclass(frozen=True)
class EclipseFreeLTANPoint:
    """Eclipse-free assessment at a specific LTAN."""
    ltan_hours: float
    eclipse_free_days: float
    total_eclipse_season_days: float


@dataclass(frozen=True)
class EclipseFreeResult:
    """Eclipse-free window optimization result."""
    points: tuple[EclipseFreeLTANPoint, ...]
    optimal_ltan_hours: float
    max_eclipse_free_days: float


@dataclass(frozen=True)
class TorqueBoundary:
    """Torque discontinuity at eclipse boundary."""
    time: datetime
    is_eclipse_entry: bool
    gg_torque_before_nm: float
    gg_torque_after_nm: float
    aero_torque_before_nm: float
    aero_torque_after_nm: float
    total_discontinuity_nm: float


@dataclass(frozen=True)
class TorqueTimingResult:
    """Worst-case torque timing result."""
    boundaries: tuple[TorqueBoundary, ...]
    max_discontinuity_nm: float
    worst_case_time: datetime


def compute_seasonal_thermal_profile(
    state: OrbitalState,
    epoch: datetime,
    raan_drift_rad_s: float = 0.0,
) -> SeasonalThermalProfile:
    """12-month thermal environment profile.

    For each month, computes mean beta angle, eclipse fraction, and
    thermal cycle count.

    Args:
        state: Satellite orbital state.
        epoch: Start epoch.
        raan_drift_rad_s: RAAN precession rate (rad/s).

    Returns:
        SeasonalThermalProfile with monthly thermal data.
    """
    T_orbit = 2.0 * math.pi / state.mean_motion_rad_s
    orbits_per_day = 86400.0 / T_orbit

    months: list[ThermalMonth] = []
    total_cycles = 0

    for m in range(12):
        # Middle of month
        month_offset_days = m * 30.44
        month_time = epoch + timedelta(days=month_offset_days)

        # RAAN at this time
        raan = state.raan_rad + raan_drift_rad_s * month_offset_days * 86400.0

        # Beta angle
        beta = compute_beta_angle(raan, state.inclination_rad, month_time)

        # Eclipse fraction at this time
        ecl_frac = eclipse_fraction(state, month_time, num_points=72)

        # Thermal cycles: one per orbit if in eclipse
        days_in_month = 30.44
        cycles = int(orbits_per_day * days_in_month * ecl_frac) if ecl_frac > 0.01 else 0

        # Eclipse and sunlit durations per orbit
        eclipse_duration = T_orbit * ecl_frac
        sunlit_duration = T_orbit * (1.0 - ecl_frac)

        months.append(ThermalMonth(
            month=m + 1,
            mean_beta_deg=beta,
            cycle_count=cycles,
            mean_eclipse_duration_s=eclipse_duration,
            mean_sunlit_duration_s=sunlit_duration,
        ))
        total_cycles += cycles

    # Find max and min cycle months
    max_month = max(months, key=lambda m: m.cycle_count).month
    min_month = min(months, key=lambda m: m.cycle_count).month

    return SeasonalThermalProfile(
        months=tuple(months),
        total_annual_cycles=total_cycles,
        max_cycle_month=max_month,
        min_cycle_month=min_month,
    )


def compute_seasonal_dose_profile(
    state: OrbitalState,
    epoch: datetime,
    duration_days: float = 365.0,
    step_days: float = 30.0,
) -> SeasonalDoseProfile:
    """Seasonal radiation dose profile with eclipse correlation.

    At each step, computes orbit-averaged dose rate, cumulative dose,
    eclipse fraction, and beta angle.

    Args:
        state: Satellite orbital state.
        epoch: Start epoch.
        duration_days: Profile duration (days).
        step_days: Step interval (days).

    Returns:
        SeasonalDoseProfile with dose snapshots.
    """
    snapshots: list[DoseSnapshot] = []
    cum_dose = 0.0
    max_rate = 0.0
    ecl_frac_sum = 0.0
    count = 0
    prev_time = epoch

    t = 0.0
    while t <= duration_days + 1e-9:
        current_time = epoch + timedelta(days=t)

        # Radiation summary
        rad = compute_orbit_radiation_summary(state, current_time, num_points=72)
        dose_rate = rad.mean_dose_rate_rad_s

        # Eclipse fraction
        ecl_frac = eclipse_fraction(state, current_time, num_points=72)

        # Beta angle
        beta = compute_beta_angle(state.raan_rad, state.inclination_rad, current_time)

        # Accumulate dose
        if count > 0:
            dt_s = (current_time - prev_time).total_seconds()
            cum_dose += dose_rate * dt_s

        if dose_rate > max_rate:
            max_rate = dose_rate

        ecl_frac_sum += ecl_frac
        count += 1

        snapshots.append(DoseSnapshot(
            time=current_time,
            dose_rate_rad_s=dose_rate,
            cumulative_dose_rad=cum_dose,
            eclipse_fraction=ecl_frac,
            beta_deg=beta,
        ))

        prev_time = current_time
        t += step_days

    annual_dose = cum_dose * (365.0 / max(duration_days, 1.0))
    mean_ecl = ecl_frac_sum / count if count > 0 else 0.0

    return SeasonalDoseProfile(
        snapshots=tuple(snapshots),
        annual_dose_rad=annual_dose,
        max_dose_rate_rad_s=max_rate,
        mean_eclipse_fraction=mean_ecl,
    )


def compute_radiation_optimized_ltan(
    altitude_km: float,
    epoch: datetime,
    ltan_values: list[float] | None = None,
) -> RadiationLTANResult:
    """Find LTAN that minimizes radiation dose.

    Evaluates each LTAN for orbit-averaged radiation dose and SAA fraction.

    Args:
        altitude_km: Orbital altitude (km).
        epoch: Reference epoch.
        ltan_values: List of LTAN values to evaluate (hours).

    Returns:
        RadiationLTANResult with optimal LTAN.
    """
    if ltan_values is None:
        ltan_values = [6.0, 10.5, 14.0, 18.0, 22.0]

    from constellation_generator.domain.solar import sun_position_eci as _sun

    sun = _sun(epoch)
    ra_sun = sun.right_ascension_rad

    a = _R_EARTH + altitude_km * 1000.0
    n = math.sqrt(_MU / a**3)
    inc_rad = math.radians(97.4)

    points: list[RadiationLTANPoint] = []

    for ltan in ltan_values:
        raan = ra_sun + math.radians((ltan - 12.0) * 15.0)
        state = OrbitalState(
            semi_major_axis_m=a, eccentricity=0.0,
            inclination_rad=inc_rad, raan_rad=raan,
            arg_perigee_rad=0.0, true_anomaly_rad=0.0,
            mean_motion_rad_s=n, reference_epoch=epoch,
        )

        rad = compute_orbit_radiation_summary(state, epoch, num_points=72)

        points.append(RadiationLTANPoint(
            ltan_hours=ltan,
            annual_dose_rad=rad.annual_dose_rad,
            saa_fraction=rad.saa_fraction,
        ))

    optimal = min(points, key=lambda p: p.annual_dose_rad)

    return RadiationLTANResult(
        points=tuple(points),
        optimal_ltan_hours=optimal.ltan_hours,
        min_annual_dose_rad=optimal.annual_dose_rad,
    )


def compute_eclipse_free_windows(
    altitude_km: float,
    epoch: datetime,
    ltan_values: list[float] | None = None,
) -> EclipseFreeResult:
    """Find LTAN that maximizes eclipse-free days.

    For each LTAN, predicts eclipse seasons over one year and computes
    total eclipse-free days.

    Args:
        altitude_km: Orbital altitude (km).
        epoch: Reference epoch.
        ltan_values: List of LTAN values to evaluate (hours).

    Returns:
        EclipseFreeResult with optimal LTAN.
    """
    if ltan_values is None:
        ltan_values = [6.0, 10.5, 14.0, 18.0, 22.0]

    from constellation_generator.domain.solar import sun_position_eci as _sun

    sun = _sun(epoch)
    ra_sun = sun.right_ascension_rad

    a = _R_EARTH + altitude_km * 1000.0
    inc_rad = math.radians(97.4)

    points: list[EclipseFreeLTANPoint] = []

    for ltan in ltan_values:
        raan = ra_sun + math.radians((ltan - 12.0) * 15.0)

        seasons = predict_eclipse_seasons(
            raan, inc_rad, epoch, 365.0, step_days=1.0,
        )

        total_eclipse_days = sum(
            (end - start).total_seconds() / 86400.0
            for start, end in seasons
        )
        eclipse_free = 365.0 - total_eclipse_days

        points.append(EclipseFreeLTANPoint(
            ltan_hours=ltan,
            eclipse_free_days=max(0.0, eclipse_free),
            total_eclipse_season_days=total_eclipse_days,
        ))

    optimal = max(points, key=lambda p: p.eclipse_free_days)

    return EclipseFreeResult(
        points=tuple(points),
        optimal_ltan_hours=optimal.ltan_hours,
        max_eclipse_free_days=optimal.eclipse_free_days,
    )


def compute_worst_case_torque_timing(
    state: OrbitalState,
    inertia: InertiaTensor,
    drag_config: DragConfig,
    cp_offset_m: tuple[float, float, float],
    epoch: datetime,
    duration_s: float,
    step_s: float,
) -> TorqueTimingResult:
    """Find worst-case torque discontinuities at eclipse boundaries.

    Steps through the orbit, detects eclipse entry/exit transitions,
    and computes torque change at each boundary.

    Args:
        state: Satellite orbital state.
        inertia: Spacecraft inertia tensor.
        drag_config: Drag configuration.
        cp_offset_m: Center-of-pressure offset from COM (m).
        epoch: Start epoch.
        duration_s: Analysis duration (seconds).
        step_s: Time step (seconds).

    Returns:
        TorqueTimingResult with boundary discontinuities.
    """
    boundaries: list[TorqueBoundary] = []
    max_disc = 0.0
    worst_time = epoch

    prev_eclipsed = False
    prev_gg_mag = 0.0
    prev_aero_mag = 0.0
    prev_time = epoch

    t = 0.0
    while t <= duration_s + 1e-9:
        current_time = epoch + timedelta(seconds=t)
        pos_eci, vel_eci = propagate_to(state, current_time)
        pos_t = (pos_eci[0], pos_eci[1], pos_eci[2])
        vel_t = (vel_eci[0], vel_eci[1], vel_eci[2])

        sun = sun_position_eci(current_time)
        ecl = is_eclipsed(pos_t, sun.position_eci_m)
        is_dark = ecl != EclipseType.NONE

        # Compute torques
        gg = compute_gravity_gradient_torque(pos_t, inertia)
        aero = compute_aerodynamic_torque(pos_t, vel_t, drag_config, cp_offset_m)

        # Detect transition
        if t > 0 and is_dark != prev_eclipsed:
            disc = abs(gg.magnitude_nm - prev_gg_mag) + abs(aero.magnitude_nm - prev_aero_mag)

            boundary = TorqueBoundary(
                time=current_time,
                is_eclipse_entry=is_dark,
                gg_torque_before_nm=prev_gg_mag,
                gg_torque_after_nm=gg.magnitude_nm,
                aero_torque_before_nm=prev_aero_mag,
                aero_torque_after_nm=aero.magnitude_nm,
                total_discontinuity_nm=disc,
            )
            boundaries.append(boundary)

            if disc > max_disc:
                max_disc = disc
                worst_time = current_time

        prev_eclipsed = is_dark
        prev_gg_mag = gg.magnitude_nm
        prev_aero_mag = aero.magnitude_nm
        prev_time = current_time
        t += step_s

    # If no boundaries detected, create a synthetic one
    if not boundaries:
        boundaries.append(TorqueBoundary(
            time=epoch,
            is_eclipse_entry=False,
            gg_torque_before_nm=prev_gg_mag,
            gg_torque_after_nm=prev_gg_mag,
            aero_torque_before_nm=prev_aero_mag,
            aero_torque_after_nm=prev_aero_mag,
            total_discontinuity_nm=max_disc,
        ))

    return TorqueTimingResult(
        boundaries=tuple(boundaries),
        max_discontinuity_nm=max_disc,
        worst_case_time=worst_time,
    )
