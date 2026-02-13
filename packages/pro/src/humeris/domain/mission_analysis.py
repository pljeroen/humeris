# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Mission-level analysis compositions.

Composes lifetime, station-keeping, radiation, eclipse, and maneuver
modules to produce propellant profiles, health timelines, altitude
trade studies, and mission cost metrics.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.lifetime import DecayPoint
from humeris.domain.atmosphere import DragConfig
from humeris.domain.station_keeping import (
    drag_compensation_dv_per_year,
    propellant_mass_for_dv,
)
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.propagation import OrbitalState
from humeris.domain.radiation import compute_orbit_radiation_summary
from humeris.domain.eclipse import eclipse_fraction
from humeris.domain.maneuvers import hohmann_transfer
from humeris.domain.revisit import CoverageResult
from humeris.domain.trade_study import WalkerConfig


_R_EARTH = OrbitalConstants.R_EARTH
_MU = OrbitalConstants.MU_EARTH
_G0 = 9.80665
_SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass(frozen=True)
class PropellantPoint:
    """Single point in propellant consumption profile."""
    time: datetime
    altitude_km: float
    dv_per_year_ms: float
    propellant_per_year_kg: float
    cumulative_propellant_kg: float


@dataclass(frozen=True)
class PropellantProfile:
    """Propellant consumption over orbit decay."""
    points: tuple[PropellantPoint, ...]
    total_propellant_kg: float
    depletion_time: datetime | None


@dataclass(frozen=True)
class HealthSnapshot:
    """Satellite health at a single point in time."""
    time: datetime
    altitude_km: float
    cumulative_dose_rad: float
    cumulative_thermal_cycles: float
    cumulative_propellant_kg: float


@dataclass(frozen=True)
class HealthTimeline:
    """Multi-parameter health timeline."""
    snapshots: tuple[HealthSnapshot, ...]
    limiting_factor: str
    end_of_life_time: datetime | None


@dataclass(frozen=True)
class AltitudeTradePoint:
    """Single point in altitude trade study."""
    altitude_km: float
    raising_dv_ms: float
    sk_dv_ms: float
    total_dv_ms: float


@dataclass(frozen=True)
class AltitudeOptimization:
    """Altitude optimization result."""
    points: tuple[AltitudeTradePoint, ...]
    optimal_altitude_km: float
    minimum_dv_ms: float


@dataclass(frozen=True)
class MissionCostMetric:
    """Mission cost metric combining all factors."""
    total_dv_ms: float
    wet_mass_per_sat_kg: float
    total_constellation_mass_kg: float
    coverage_pct: float
    cost_per_coverage_point: float


def compute_propellant_profile(
    decay_profile: tuple[DecayPoint, ...],
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    propellant_budget_kg: float,
) -> PropellantProfile:
    """Propellant consumption curve over orbit decay.

    At each decay point, computes the station-keeping dV and propellant
    consumption rate. Detects when propellant budget is exceeded.

    Args:
        decay_profile: Tuple of DecayPoint from compute_orbit_lifetime.
        drag_config: Satellite drag configuration.
        isp_s: Specific impulse (seconds).
        dry_mass_kg: Dry mass (kg).
        propellant_budget_kg: Total propellant budget (kg).

    Returns:
        PropellantProfile with points and depletion time.
    """
    points: list[PropellantPoint] = []
    cumulative = 0.0
    depletion_time = None

    for dp in decay_profile:
        alt_km = dp.altitude_km
        if alt_km < 100.0:
            alt_km = 100.0

        dv_year = drag_compensation_dv_per_year(alt_km, drag_config)

        # Guard against overflow in Tsiolkovsky exponential at extreme dV
        max_dv = isp_s * 9.80665 * 20.0  # exp(20) ≈ 5e8, reasonable cap
        capped_dv = min(dv_year, max_dv)
        prop_year = propellant_mass_for_dv(isp_s, dry_mass_kg, capped_dv)

        # Estimate propellant consumed since last point
        if len(points) > 0:
            dt_s = (dp.time - points[-1].time).total_seconds()
            dt_years = dt_s / _SECONDS_PER_YEAR
            cumulative += prop_year * dt_years

        points.append(PropellantPoint(
            time=dp.time,
            altitude_km=dp.altitude_km,
            dv_per_year_ms=dv_year,
            propellant_per_year_kg=prop_year,
            cumulative_propellant_kg=cumulative,
        ))

        if cumulative >= propellant_budget_kg and depletion_time is None:
            depletion_time = dp.time

    total = cumulative
    return PropellantProfile(
        points=tuple(points),
        total_propellant_kg=total,
        depletion_time=depletion_time,
    )


def compute_health_timeline(
    state: OrbitalState,
    drag_config: DragConfig,
    epoch: datetime,
    mission_years: float = 5.0,
    radiation_limit_rad: float = 10000.0,
    thermal_limit_cycles: int = 50000,
) -> HealthTimeline:
    """Multi-parameter health timeline over mission duration.

    Tracks cumulative radiation dose, thermal cycles, and propellant
    consumption. Identifies limiting factor.

    Args:
        state: Satellite orbital state.
        drag_config: Satellite drag configuration.
        epoch: Mission start epoch.
        mission_years: Mission duration (years).
        radiation_limit_rad: Radiation limit (rad).
        thermal_limit_cycles: Thermal cycle limit.

    Returns:
        HealthTimeline with snapshots and limiting factor.
    """
    step_days = 30.0
    duration_days = mission_years * 365.25
    num_steps = max(1, int(duration_days / step_days))

    alt_km = (state.semi_major_axis_m - _R_EARTH) / 1000.0

    # Get orbit radiation summary for dose rate
    rad_summary = compute_orbit_radiation_summary(state, epoch)
    dose_rate = rad_summary.mean_dose_rate_rad_s

    # Get eclipse fraction for thermal cycles
    ecl_frac = eclipse_fraction(state, epoch, num_points=72)
    T_orbit = 2.0 * np.pi / state.mean_motion_rad_s
    cycles_per_day = 86400.0 / T_orbit

    # Station-keeping propellant rate
    dv_year = drag_compensation_dv_per_year(alt_km, drag_config)

    snapshots: list[HealthSnapshot] = []
    cum_dose = 0.0
    cum_cycles = 0.0
    cum_prop = 0.0
    eol_time = None
    limiting = "none"

    for i in range(num_steps + 1):
        t = epoch + timedelta(days=i * step_days)
        dt_s = i * step_days * 86400.0

        cum_dose = dose_rate * dt_s
        cum_cycles = cycles_per_day * i * step_days * ecl_frac
        cum_prop = propellant_mass_for_dv(220.0, drag_config.mass_kg, dv_year) * (dt_s / _SECONDS_PER_YEAR)

        snapshots.append(HealthSnapshot(
            time=t,
            altitude_km=alt_km,
            cumulative_dose_rad=cum_dose,
            cumulative_thermal_cycles=cum_cycles,
            cumulative_propellant_kg=cum_prop,
        ))

        if eol_time is None:
            if cum_dose >= radiation_limit_rad:
                eol_time = t
                limiting = "radiation"
            elif cum_cycles >= thermal_limit_cycles:
                eol_time = t
                limiting = "thermal"

    return HealthTimeline(
        snapshots=tuple(snapshots),
        limiting_factor=limiting,
        end_of_life_time=eol_time,
    )


def compute_optimal_altitude(
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    injection_altitude_km: float,
    mission_years: float,
    alt_min_km: float = 300.0,
    alt_max_km: float = 800.0,
    alt_step_km: float = 25.0,
) -> AltitudeOptimization:
    """Find optimal operational altitude balancing raising dV and SK dV.

    Higher altitude → more raising dV from injection orbit, less SK dV.
    Lower altitude → less raising dV, more SK dV (atmospheric drag).

    Args:
        drag_config: Satellite drag configuration.
        isp_s: Specific impulse (seconds).
        dry_mass_kg: Dry mass (kg).
        injection_altitude_km: Launch injection altitude (km).
        mission_years: Mission duration (years).
        alt_min_km: Minimum altitude to evaluate (km).
        alt_max_km: Maximum altitude to evaluate (km).
        alt_step_km: Altitude step (km).

    Returns:
        AltitudeOptimization with trade points and optimal.
    """
    points: list[AltitudeTradePoint] = []

    alt = alt_min_km
    while alt <= alt_max_km + 1e-9:
        # Raising dV: Hohmann from injection to operational altitude
        r1 = _R_EARTH + injection_altitude_km * 1000.0
        r2 = _R_EARTH + alt * 1000.0
        transfer = hohmann_transfer(r1, r2)
        raising_dv = transfer.total_delta_v_ms

        # Station-keeping dV per year × mission years
        sk_dv_year = drag_compensation_dv_per_year(alt, drag_config)
        sk_dv = sk_dv_year * mission_years

        total_dv = raising_dv + sk_dv

        points.append(AltitudeTradePoint(
            altitude_km=alt,
            raising_dv_ms=raising_dv,
            sk_dv_ms=sk_dv,
            total_dv_ms=total_dv,
        ))
        alt += alt_step_km

    # Find minimum
    min_point = min(points, key=lambda p: p.total_dv_ms)

    return AltitudeOptimization(
        points=tuple(points),
        optimal_altitude_km=min_point.altitude_km,
        minimum_dv_ms=min_point.total_dv_ms,
    )


def compute_mission_cost_metric(
    config: WalkerConfig,
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    injection_altitude_km: float,
    mission_years: float,
    coverage_result: CoverageResult,
) -> MissionCostMetric:
    """Mission cost metric combining all factors.

    Computes total dV (raising + SK), wet mass, and cost/coverage ratio.

    Args:
        config: Walker constellation configuration.
        drag_config: Satellite drag configuration.
        isp_s: Specific impulse (seconds).
        dry_mass_kg: Dry mass (kg).
        injection_altitude_km: Launch injection altitude (km).
        mission_years: Mission duration (years).
        coverage_result: Coverage analysis result.

    Returns:
        MissionCostMetric with mass and cost metrics.
    """
    # Raising dV
    r1 = _R_EARTH + injection_altitude_km * 1000.0
    r2 = _R_EARTH + config.altitude_km * 1000.0
    transfer = hohmann_transfer(r1, r2)
    raising_dv = transfer.total_delta_v_ms

    # SK dV
    sk_dv_year = drag_compensation_dv_per_year(config.altitude_km, drag_config)
    sk_dv = sk_dv_year * mission_years

    total_dv = raising_dv + sk_dv

    # Propellant mass per satellite
    prop_mass = propellant_mass_for_dv(isp_s, dry_mass_kg, total_dv)
    wet_mass = dry_mass_kg + prop_mass

    # Total constellation
    num_sats = config.num_planes * config.sats_per_plane
    total_mass = wet_mass * num_sats

    # Coverage metric
    coverage_pct = coverage_result.mean_coverage_fraction * 100.0
    cost_per_cov = total_mass / max(coverage_pct, 0.01)

    return MissionCostMetric(
        total_dv_ms=total_dv,
        wet_mass_per_sat_kg=wet_mass,
        total_constellation_mass_kg=total_mass,
        coverage_pct=coverage_pct,
        cost_per_coverage_point=cost_per_cov,
    )
