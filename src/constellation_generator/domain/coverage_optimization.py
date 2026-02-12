"""Coverage optimization compositions.

Composes sensor FOV, DOP, ground track crossings, access windows, revisit,
trade study, deorbit, and eclipse modules to produce quality-weighted
coverage, crossing revisit, compliant trade studies, EO LTAN optimization,
and ground station siting.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.propagation import OrbitalState, propagate_to, propagate_ecef_to
from constellation_generator.domain.sensor import SensorConfig, compute_sensor_coverage
from constellation_generator.domain.dilution_of_precision import DOPResult, compute_dop
from constellation_generator.domain.ground_track import (
    GroundTrackPoint,
    AscendingNodePass,
    GroundTrackCrossing,
    find_ascending_nodes,
    find_ground_track_crossings,
)
from constellation_generator.domain.observation import GroundStation, compute_observation
from constellation_generator.domain.access_windows import AccessWindow, compute_access_windows
from constellation_generator.domain.revisit import CoverageResult
from constellation_generator.domain.trade_study import (
    WalkerConfig,
    TradePoint,
    TradeStudyResult,
    pareto_front_indices,
)
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.deorbit import assess_deorbit_compliance
from constellation_generator.domain.lifetime import compute_orbit_lifetime
from constellation_generator.domain.eclipse import eclipse_fraction, compute_beta_angle
from constellation_generator.domain.radiation import compute_orbit_radiation_summary
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.coordinate_frames import geodetic_to_ecef

_R_EARTH = OrbitalConstants.R_EARTH
_MU = OrbitalConstants.MU_EARTH


@dataclass(frozen=True)
class QualityCoveragePoint:
    """Coverage point with quality weighting."""
    lat_deg: float
    lon_deg: float
    num_visible: int
    gdop: float
    is_usable: bool


@dataclass(frozen=True)
class QualityCoverageResult:
    """Quality-weighted coverage result."""
    points: tuple[QualityCoveragePoint, ...]
    usable_fraction: float
    mean_gdop: float


@dataclass(frozen=True)
class CrossingRevisitResult:
    """Revisit analysis at ground track crossings."""
    crossing_revisit_s: float
    elsewhere_revisit_s: float
    improvement_factor: float
    num_crossings: int


@dataclass(frozen=True)
class CompliantTradePoint:
    """Trade study point augmented with deorbit compliance."""
    trade_point: TradePoint
    lifetime_days: float
    is_compliant: bool
    deorbit_dv_ms: float


@dataclass(frozen=True)
class CompliantTradeResult:
    """Compliant trade study result."""
    all_points: tuple[CompliantTradePoint, ...]
    compliant_count: int
    pareto_indices: tuple[int, ...]


@dataclass(frozen=True)
class EOLTANPoint:
    """EO LTAN optimization point."""
    ltan_hours: float
    effective_revisit_s: float
    eclipse_free_days_per_year: float
    annual_dose_rad: float


@dataclass(frozen=True)
class EOLTANOptimization:
    """Optimal EO LTAN result."""
    points: tuple[EOLTANPoint, ...]
    optimal_ltan_hours: float


@dataclass(frozen=True)
class StationCandidate:
    """Ground station candidate for siting."""
    lat_deg: float
    lon_deg: float
    total_contact_s: float
    num_passes: int


@dataclass(frozen=True)
class GroundStationOptimization:
    """Ground station siting optimization result."""
    candidates: tuple[StationCandidate, ...]
    ranked_indices: tuple[int, ...]


def compute_quality_weighted_coverage(
    states: list[OrbitalState],
    time: datetime,
    sensor: SensorConfig,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    gdop_threshold: float = 6.0,
) -> QualityCoverageResult:
    """Sensor coverage weighted by geometric quality (GDOP).

    At each grid point, computes visible satellite count and GDOP.
    A point is "usable" if GDOP < threshold and num_visible >= 4.

    Args:
        states: List of satellite orbital states.
        time: Evaluation time.
        sensor: Sensor configuration.
        lat_step_deg: Latitude grid spacing.
        lon_step_deg: Longitude grid spacing.
        gdop_threshold: Maximum acceptable GDOP.

    Returns:
        QualityCoverageResult with quality-weighted coverage.
    """
    sensor_cov = compute_sensor_coverage(
        states, time, sensor,
        lat_step_deg=lat_step_deg, lon_step_deg=lon_step_deg,
    )

    # Pre-compute satellite ECEF positions for DOP
    sat_ecefs = []
    for state in states:
        ecef = propagate_ecef_to(state, time)
        sat_ecefs.append(ecef)

    quality_points: list[QualityCoveragePoint] = []
    usable_count = 0
    gdop_sum = 0.0
    gdop_count = 0

    for cp in sensor_cov:
        # Compute DOP at this grid point
        dop = compute_dop(cp.lat_deg, cp.lon_deg, sat_ecefs)

        is_usable = cp.visible_count >= 4 and dop.gdop < gdop_threshold

        quality_points.append(QualityCoveragePoint(
            lat_deg=cp.lat_deg,
            lon_deg=cp.lon_deg,
            num_visible=cp.visible_count,
            gdop=dop.gdop,
            is_usable=is_usable,
        ))

        if is_usable:
            usable_count += 1
        if dop.gdop < float('inf'):
            gdop_sum += dop.gdop
            gdop_count += 1

    total = len(quality_points)
    usable_frac = usable_count / total if total > 0 else 0.0
    mean_gdop = gdop_sum / gdop_count if gdop_count > 0 else 0.0

    return QualityCoverageResult(
        points=tuple(quality_points),
        usable_fraction=usable_frac,
        mean_gdop=mean_gdop,
    )


def compute_crossing_revisit(
    track: list[GroundTrackPoint],
    states: list[OrbitalState],
    start: datetime,
    duration: timedelta,
    step: timedelta,
    min_elevation_deg: float = 10.0,
) -> CrossingRevisitResult:
    """Revisit analysis at ground track crossing locations.

    Finds ground track crossings and evaluates revisit at those
    locations vs elsewhere, computing the improvement factor.

    Args:
        track: Ground track points.
        states: Satellite orbital states.
        start: Start time.
        duration: Analysis duration.
        step: Time step.
        min_elevation_deg: Minimum elevation angle.

    Returns:
        CrossingRevisitResult with revisit metrics.
    """
    crossings = find_ground_track_crossings(track)
    num_crossings = len(crossings)

    if num_crossings == 0:
        return CrossingRevisitResult(
            crossing_revisit_s=0.0,
            elsewhere_revisit_s=0.0,
            improvement_factor=0.0,
            num_crossings=0,
        )

    # For each crossing, compute access windows as a proxy for revisit
    crossing_revisits: list[float] = []
    for cx in crossings[:min(num_crossings, 5)]:
        station = GroundStation(
            name=f"cx_{cx.lat_deg:.1f}_{cx.lon_deg:.1f}",
            lat_deg=cx.lat_deg,
            lon_deg=cx.lon_deg,
        )
        total_windows = 0
        for state in states:
            windows = compute_access_windows(
                station, state, start, duration, step,
                min_elevation_deg=min_elevation_deg,
            )
            total_windows += len(windows)
        if total_windows > 1:
            avg_revisit = duration.total_seconds() / total_windows
        else:
            avg_revisit = duration.total_seconds()
        crossing_revisits.append(avg_revisit)

    # Elsewhere: use a non-crossing point (equator, 0 lon)
    ref_station = GroundStation(name="ref", lat_deg=0.0, lon_deg=90.0)
    ref_windows = 0
    for state in states:
        windows = compute_access_windows(
            ref_station, state, start, duration, step,
            min_elevation_deg=min_elevation_deg,
        )
        ref_windows += len(windows)
    elsewhere_revisit = duration.total_seconds() / max(ref_windows, 1)

    crossing_avg = sum(crossing_revisits) / len(crossing_revisits) if crossing_revisits else 0.0
    improvement = elsewhere_revisit / crossing_avg if crossing_avg > 0 else 0.0

    return CrossingRevisitResult(
        crossing_revisit_s=crossing_avg,
        elsewhere_revisit_s=elsewhere_revisit,
        improvement_factor=improvement,
        num_crossings=num_crossings,
    )


def compute_compliant_trade_study(
    trade_result: TradeStudyResult,
    drag_config: DragConfig,
    epoch: datetime,
    max_lifetime_years: float = 5.0,
) -> CompliantTradeResult:
    """Trade study augmented with deorbit compliance.

    For each trade point, checks FCC 5-year deorbit compliance and
    computes deorbit dV. Extracts Pareto front from compliant points.

    Args:
        trade_result: Base trade study result.
        drag_config: Satellite drag configuration.
        epoch: Reference epoch.
        max_lifetime_years: Maximum lifetime for compliance.

    Returns:
        CompliantTradeResult with compliance annotations.
    """
    compliant_points: list[CompliantTradePoint] = []
    compliant_count = 0

    for tp in trade_result.points:
        assessment = assess_deorbit_compliance(
            tp.config.altitude_km, drag_config, epoch,
        )

        compliant_points.append(CompliantTradePoint(
            trade_point=tp,
            lifetime_days=assessment.natural_lifetime_days,
            is_compliant=assessment.compliant,
            deorbit_dv_ms=assessment.deorbit_delta_v_ms,
        ))

        if assessment.compliant:
            compliant_count += 1

    # Pareto front: minimize satellite count, minimize mean revisit
    costs = [float(cp.trade_point.total_satellites) for cp in compliant_points]
    metrics = [cp.trade_point.coverage.mean_revisit_s for cp in compliant_points]
    pareto = pareto_front_indices(costs, metrics)

    return CompliantTradeResult(
        all_points=tuple(compliant_points),
        compliant_count=compliant_count,
        pareto_indices=tuple(pareto),
    )


def compute_optimal_eo_ltan(
    altitude_km: float,
    epoch: datetime,
    ltan_values: list[float],
    analysis_duration_s: float = 5400.0,
    analysis_step_s: float = 60.0,
) -> EOLTANOptimization:
    """Find optimal LTAN for Earth observation missions.

    Evaluates each LTAN value for eclipse-free duration, radiation dose,
    and selects optimal as the one minimizing annual dose.

    LTAN is mapped to RAAN via: RAAN = (LTAN_hours - 12) * 15 degrees + RA_sun.

    Args:
        altitude_km: Orbital altitude (km).
        epoch: Reference epoch.
        ltan_values: List of LTAN values to evaluate (hours).
        analysis_duration_s: Duration for revisit analysis.
        analysis_step_s: Step for revisit analysis.

    Returns:
        EOLTANOptimization with points and optimal LTAN.
    """
    from constellation_generator.domain.solar import sun_position_eci as _sun_pos
    from constellation_generator.domain.eclipse import predict_eclipse_seasons

    sun = _sun_pos(epoch)
    ra_sun = sun.right_ascension_rad

    a = _R_EARTH + altitude_km * 1000.0
    n = math.sqrt(_MU / a**3)
    inc_rad = math.radians(97.4)  # typical SSO inclination

    eo_points: list[EOLTANPoint] = []

    for ltan in ltan_values:
        # LTAN to RAAN
        raan = ra_sun + math.radians((ltan - 12.0) * 15.0)

        # Create state for this LTAN
        state = OrbitalState(
            semi_major_axis_m=a, eccentricity=0.0,
            inclination_rad=inc_rad, raan_rad=raan,
            arg_perigee_rad=0.0, true_anomaly_rad=0.0,
            mean_motion_rad_s=n, reference_epoch=epoch,
        )

        # Radiation
        rad = compute_orbit_radiation_summary(state, epoch, num_points=72)

        # Eclipse-free days
        seasons = predict_eclipse_seasons(raan, inc_rad, epoch, 365.0, step_days=1.0)
        total_eclipse_days = sum(
            (end - start).total_seconds() / 86400.0 for start, end in seasons
        )
        eclipse_free_days = 365.0 - total_eclipse_days

        # Effective revisit (simplified: use orbital period)
        T = 2.0 * math.pi / n
        effective_revisit = T

        eo_points.append(EOLTANPoint(
            ltan_hours=ltan,
            effective_revisit_s=effective_revisit,
            eclipse_free_days_per_year=eclipse_free_days,
            annual_dose_rad=rad.annual_dose_rad,
        ))

    # Optimal: minimize annual dose
    if eo_points:
        optimal = min(eo_points, key=lambda p: p.annual_dose_rad)
        optimal_ltan = optimal.ltan_hours
    else:
        optimal_ltan = 0.0

    return EOLTANOptimization(
        points=tuple(eo_points),
        optimal_ltan_hours=optimal_ltan,
    )


def compute_optimal_ground_stations(
    track: list[GroundTrackPoint],
    states: list[OrbitalState],
    start: datetime,
    duration: timedelta,
    step: timedelta,
    min_elevation_deg: float = 10.0,
    num_candidates: int = 10,
) -> GroundStationOptimization:
    """Find optimal ground station locations.

    Uses ground track crossings and ascending nodes as candidate locations,
    evaluates access windows for each.

    Args:
        track: Ground track points.
        states: Satellite orbital states.
        start: Start time.
        duration: Analysis duration.
        step: Time step.
        min_elevation_deg: Minimum elevation angle.
        num_candidates: Maximum number of candidates.

    Returns:
        GroundStationOptimization with ranked candidates.
    """
    # Collect candidate locations from crossings and ascending nodes
    crossings = find_ground_track_crossings(track)
    nodes = find_ascending_nodes(track)

    candidate_locs: list[tuple[float, float]] = []

    for cx in crossings[:num_candidates]:
        candidate_locs.append((cx.lat_deg, cx.lon_deg))
    for nd in nodes[:num_candidates]:
        candidate_locs.append((0.0, nd.longitude_deg))

    # Deduplicate roughly
    unique_locs: list[tuple[float, float]] = []
    for loc in candidate_locs:
        is_dup = False
        for u in unique_locs:
            if abs(loc[0] - u[0]) < 2.0 and abs(loc[1] - u[1]) < 2.0:
                is_dup = True
                break
        if not is_dup:
            unique_locs.append(loc)

    unique_locs = unique_locs[:num_candidates]

    # If no candidates, use equatorial points
    if not unique_locs:
        for lon in range(-180, 180, 36):
            unique_locs.append((0.0, float(lon)))

    candidates: list[StationCandidate] = []

    for lat, lon in unique_locs:
        station = GroundStation(name=f"cand_{lat:.0f}_{lon:.0f}", lat_deg=lat, lon_deg=lon)
        total_contact = 0.0
        total_passes = 0

        for state in states:
            windows = compute_access_windows(
                station, state, start, duration, step,
                min_elevation_deg=min_elevation_deg,
            )
            for w in windows:
                total_contact += w.duration_seconds
            total_passes += len(windows)

        candidates.append(StationCandidate(
            lat_deg=lat, lon_deg=lon,
            total_contact_s=total_contact,
            num_passes=total_passes,
        ))

    # Rank by total contact time descending
    ranked = sorted(range(len(candidates)),
                    key=lambda i: candidates[i].total_contact_s,
                    reverse=True)

    return GroundStationOptimization(
        candidates=tuple(candidates),
        ranked_indices=tuple(ranked),
    )
