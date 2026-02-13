# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Coverage optimization compositions.

Composes sensor FOV, DOP, ground track crossings, access windows, revisit,
trade study, deorbit, and eclipse modules to produce quality-weighted
coverage, crossing revisit, compliant trade studies, EO LTAN optimization,
and ground station siting.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_to, propagate_ecef_to
from humeris.domain.sensor import SensorConfig, compute_sensor_coverage
from humeris.domain.dilution_of_precision import DOPResult, compute_dop
from humeris.domain.ground_track import (
    GroundTrackPoint,
    AscendingNodePass,
    GroundTrackCrossing,
    find_ascending_nodes,
    find_ground_track_crossings,
)
from humeris.domain.observation import GroundStation, compute_observation
from humeris.domain.access_windows import AccessWindow, compute_access_windows
from humeris.domain.revisit import CoverageResult
from humeris.domain.trade_study import (
    WalkerConfig,
    TradePoint,
    TradeStudyResult,
    pareto_front_indices,
)
from humeris.domain.atmosphere import DragConfig
from humeris.domain.deorbit import assess_deorbit_compliance
from humeris.domain.lifetime import compute_orbit_lifetime
from humeris.domain.eclipse import eclipse_fraction, compute_beta_angle
from humeris.domain.radiation import compute_orbit_radiation_summary
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.coordinate_frames import geodetic_to_ecef

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
    from humeris.domain.solar import sun_position_eci as _sun_pos
    from humeris.domain.eclipse import predict_eclipse_seasons

    sun = _sun_pos(epoch)
    ra_sun = sun.right_ascension_rad

    a = _R_EARTH + altitude_km * 1000.0
    n = float(np.sqrt(_MU / a**3))
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


# ---------------------------------------------------------------------------
# P4: Rate-Distortion Optimal Coverage Grid via Blahut-Arimoto
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimalCoverageGrid:
    """Rate-distortion optimal grid resolution for coverage analysis.

    The Blahut-Arimoto algorithm computes the R(D) curve for the coverage
    channel: what is the minimum information rate (grid resolution) needed
    to represent coverage within a given distortion tolerance.
    """
    optimal_lat_step_deg: float
    optimal_lon_step_deg: float
    rate_bits: float
    distortion: float
    compression_ratio: float
    blahut_arimoto_iterations: int


def compute_optimal_grid_resolution(
    coverage_probability: list,
    max_distortion: float = 0.1,
    initial_lat_step_deg: float = 5.0,
    initial_lon_step_deg: float = 5.0,
    max_iterations: int = 50,
    convergence_tol: float = 1e-6,
) -> OptimalCoverageGrid:
    """Compute R(D) optimal coverage grid resolution via Blahut-Arimoto.

    The coverage grid is treated as a discrete source. The source alphabet
    is the set of distinct coverage values. The reproduction alphabet is
    a quantized version. The Blahut-Arimoto algorithm iterates between:

    Step 1: Update marginal q(c_hat) = sum_c p(c) * p(c_hat|c)
    Step 2: Update conditional p(c_hat|c) = q(c_hat) * exp(-beta * d(c, c_hat)) / Z(c)

    where beta is the Lagrange multiplier controlling the rate-distortion
    tradeoff, and d(c, c_hat) = (c - c_hat)^2 is MSE distortion.

    Args:
        coverage_probability: List of coverage fractions at grid points
            (values in [0, 1] or coverage counts). This is the source
            distribution to be compressed.
        max_distortion: Maximum acceptable MSE distortion (0-1 scale).
        initial_lat_step_deg: Initial latitude grid step (reference).
        initial_lon_step_deg: Initial longitude grid step (reference).
        max_iterations: Maximum Blahut-Arimoto iterations.
        convergence_tol: Convergence tolerance for rate.

    Returns:
        OptimalCoverageGrid with optimal grid step and R(D) metrics.
    """
    if not coverage_probability:
        return OptimalCoverageGrid(
            optimal_lat_step_deg=initial_lat_step_deg,
            optimal_lon_step_deg=initial_lon_step_deg,
            rate_bits=0.0,
            distortion=0.0,
            compression_ratio=1.0,
            blahut_arimoto_iterations=0,
        )

    # Normalize to [0, 1] range
    values = np.array(coverage_probability, dtype=np.float64)
    v_min = float(np.min(values))
    v_max = float(np.max(values))

    if v_max - v_min < 1e-12:
        # Constant coverage — no information, any grid suffices
        return OptimalCoverageGrid(
            optimal_lat_step_deg=180.0,  # Coarsest possible
            optimal_lon_step_deg=360.0,
            rate_bits=0.0,
            distortion=0.0,
            compression_ratio=180.0 / initial_lat_step_deg,
            blahut_arimoto_iterations=0,
        )

    normalized = (values - v_min) / (v_max - v_min)

    # Quantize source into bins
    n_source_bins = min(20, max(2, int(np.sqrt(len(normalized)))))
    bin_edges = np.linspace(0.0, 1.0 + 1e-10, n_source_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compute source distribution p(c)
    hist, _ = np.histogram(normalized, bins=bin_edges)
    p_source = hist.astype(np.float64) / len(normalized)
    # Avoid zeros
    p_source = np.maximum(p_source, 1e-15)
    p_source /= np.sum(p_source)

    # Reproduction alphabet: start with same bins, but can be coarser
    n_repr = n_source_bins
    repr_centers = bin_centers.copy()

    # Distortion matrix: d[i][j] = (source_center_i - repr_center_j)^2
    d_matrix = np.zeros((n_source_bins, n_repr))
    for i in range(n_source_bins):
        for j in range(n_repr):
            d_matrix[i, j] = (bin_centers[i] - repr_centers[j]) ** 2

    # Binary search for beta that achieves target distortion
    beta_low = 0.01
    beta_high = 100.0
    best_rate = 0.0
    best_distortion = 0.0
    best_iterations = 0

    for _ in range(20):  # binary search iterations
        beta = 0.5 * (beta_low + beta_high)

        # Blahut-Arimoto iterations
        # Initialize q(c_hat) = uniform
        q_repr = np.ones(n_repr) / n_repr

        rate = 0.0
        distortion_val = 0.0

        for iteration in range(1, max_iterations + 1):
            # Step 2: Update conditional p(c_hat|c)
            # p(c_hat|c) = q(c_hat) * exp(-beta * d(c, c_hat)) / Z(c)
            log_q = np.log(np.maximum(q_repr, 1e-300))
            p_cond = np.zeros((n_source_bins, n_repr))
            for i in range(n_source_bins):
                log_unnorm = log_q - beta * d_matrix[i, :]
                # Log-sum-exp for numerical stability
                max_log = np.max(log_unnorm)
                p_cond[i, :] = np.exp(log_unnorm - max_log)
                z = np.sum(p_cond[i, :])
                if z > 0:
                    p_cond[i, :] /= z

            # Step 1: Update marginal q(c_hat)
            q_repr_new = np.zeros(n_repr)
            for j in range(n_repr):
                q_repr_new[j] = np.sum(p_source * p_cond[:, j])
            q_repr_new = np.maximum(q_repr_new, 1e-15)
            q_repr_new /= np.sum(q_repr_new)

            # Check convergence
            diff = np.max(np.abs(q_repr_new - q_repr))
            q_repr = q_repr_new

            if diff < convergence_tol:
                best_iterations = iteration
                break
        else:
            best_iterations = max_iterations

        # Compute rate I(C; C_hat) and distortion E[d]
        rate = 0.0
        distortion_val = 0.0
        for i in range(n_source_bins):
            for j in range(n_repr):
                if p_cond[i, j] > 1e-15 and q_repr[j] > 1e-15:
                    rate += p_source[i] * p_cond[i, j] * math.log2(
                        p_cond[i, j] / q_repr[j]
                    )
                distortion_val += p_source[i] * p_cond[i, j] * d_matrix[i, j]

        rate = max(0.0, rate)

        if distortion_val > max_distortion:
            beta_low = beta   # Need higher beta to reduce distortion
        else:
            beta_high = beta  # Can lower beta (allow more distortion)

        best_rate = rate
        best_distortion = distortion_val

    # Convert rate to optimal grid resolution
    # Rate in bits measures the information content per grid point.
    # Current grid has log2(n_source_bins) bits per point.
    # Optimal grid needs best_rate bits per point.
    # Compression ratio = log2(n_source_bins) / best_rate
    max_rate = math.log2(n_source_bins) if n_source_bins > 1 else 1.0

    if best_rate > 1e-12:
        compression = max_rate / best_rate
    else:
        compression = max_rate * 10.0  # Very compressible

    compression = max(1.0, compression)

    # Optimal grid: coarsen by sqrt of compression ratio in each dimension
    coarsening = math.sqrt(compression)
    opt_lat = min(180.0, initial_lat_step_deg * coarsening)
    opt_lon = min(360.0, initial_lon_step_deg * coarsening)

    return OptimalCoverageGrid(
        optimal_lat_step_deg=opt_lat,
        optimal_lon_step_deg=opt_lon,
        rate_bits=best_rate,
        distortion=best_distortion,
        compression_ratio=compression,
        blahut_arimoto_iterations=best_iterations,
    )


# ---------------------------------------------------------------------------
# P47: Potential Flow Coverage via Conformal Mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PotentialFlowCoverage:
    """Potential flow model of coverage via stereographic conformal mapping.

    Maps sub-satellite points to the complex plane via stereographic
    projection, computes the complex potential w(z) = sum Q_k log(z - z_k),
    and finds stagnation points where dw/dz = 0.
    """
    stagnation_points_latlon: tuple   # tuple of (lat, lon) pairs
    coverage_potential_grid: np.ndarray  # Re(w) on lat/lon grid
    stream_function_grid: np.ndarray     # Im(w) on lat/lon grid
    uniformity_metric: float             # 1 - std/max of coverage potential
    num_stagnation_points: int


def _latlon_to_stereo(lat_deg: np.ndarray, lon_deg: np.ndarray):
    """Stereographic projection from lat/lon to complex plane.

    Projects from south pole: z = (x + iy) / (1 - z_sphere)
    where (x, y, z_sphere) is the unit sphere point.
    """
    lat_r = np.radians(lat_deg)
    lon_r = np.radians(lon_deg)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z_s = np.sin(lat_r)
    # Avoid division by zero at south pole
    denom = np.maximum(1.0 - z_s, 1e-12)
    return (x + 1j * y) / denom


def _stereo_to_latlon(z: np.ndarray):
    """Inverse stereographic projection: complex plane to lat/lon."""
    r2 = np.abs(z)**2
    x = 2.0 * np.real(z) / (1.0 + r2)
    y = 2.0 * np.imag(z) / (1.0 + r2)
    z_s = (r2 - 1.0) / (r2 + 1.0)
    lat = np.degrees(np.arcsin(np.clip(z_s, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def compute_potential_flow_coverage(
    source_lats: np.ndarray,
    source_lons: np.ndarray,
    source_strengths: np.ndarray,
    grid_resolution: int = 18,
    newton_max_iter: int = 50,
    newton_tol: float = 1e-8,
) -> PotentialFlowCoverage:
    """Compute potential flow coverage via conformal mapping.

    Maps sub-satellite points to the complex plane using stereographic
    projection, computes the complex potential w(z) = sum Q_k * log(z - z_k),
    finds stagnation points via Newton's method on dw/dz = 0, and projects
    back to lat/lon.

    Args:
        source_lats: Latitude of source (sub-satellite) points in degrees.
        source_lons: Longitude of source points in degrees.
        source_strengths: Strength Q_k of each source.
        grid_resolution: Number of latitude grid points (lon = 2x).
        newton_max_iter: Maximum Newton iterations for stagnation points.
        newton_tol: Newton convergence tolerance.

    Returns:
        PotentialFlowCoverage with stagnation points and potential grids.
    """
    source_lats = np.asarray(source_lats, dtype=np.float64)
    source_lons = np.asarray(source_lons, dtype=np.float64)
    source_strengths = np.asarray(source_strengths, dtype=np.float64)

    n_lat = grid_resolution
    n_lon = 2 * grid_resolution
    lat_grid = np.linspace(-89.0, 89.0, n_lat)
    lon_grid = np.linspace(-180.0, 179.0, n_lon)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    n_sources = len(source_lats)

    if n_sources == 0:
        return PotentialFlowCoverage(
            stagnation_points_latlon=(),
            coverage_potential_grid=np.zeros((n_lat, n_lon)),
            stream_function_grid=np.zeros((n_lat, n_lon)),
            uniformity_metric=1.0,
            num_stagnation_points=0,
        )

    # Project sources to complex plane
    z_sources = _latlon_to_stereo(source_lats, source_lons)

    # Project grid to complex plane
    z_grid = _latlon_to_stereo(lat_mesh.ravel(), lon_mesh.ravel())
    z_grid = z_grid.reshape(n_lat, n_lon)

    # Complex potential w(z) = sum Q_k * log(z - z_k)
    w_grid = np.zeros((n_lat, n_lon), dtype=np.complex128)
    for k in range(n_sources):
        diff = z_grid - z_sources[k]
        # Avoid log(0) by clamping magnitude
        safe_diff = np.where(np.abs(diff) < 1e-15, 1e-15 + 0j, diff)
        w_grid += source_strengths[k] * np.log(safe_diff)

    coverage_potential = np.real(w_grid)
    stream_function = np.imag(w_grid)

    # Uniformity metric: 1 - normalized_std
    pot_range = np.max(coverage_potential) - np.min(coverage_potential)
    if pot_range > 1e-15:
        normalized_std = np.std(coverage_potential) / pot_range
        uniformity = max(0.0, 1.0 - normalized_std)
    else:
        uniformity = 1.0

    # Find stagnation points: dw/dz = sum Q_k / (z - z_k) = 0
    # Use Newton's method on f(z) = dw/dz with f'(z) = d2w/dz2
    stagnation_points = []

    # Generate candidate starting points between pairs of sources
    candidates = []
    for i in range(n_sources):
        for j in range(i + 1, n_sources):
            mid = 0.5 * (z_sources[i] + z_sources[j])
            candidates.append(mid)
    # Also add some grid-based candidates
    sample_idx = np.linspace(0, n_lat * n_lon - 1, min(20, n_lat * n_lon), dtype=int)
    for idx in sample_idx:
        ri = idx // n_lon
        ci = idx % n_lon
        candidates.append(z_grid[ri, ci])

    for z0 in candidates:
        z_n = complex(z0)
        for _ in range(newton_max_iter):
            # f(z) = dw/dz = sum Q_k / (z - z_k)
            f_val = 0.0 + 0.0j
            fp_val = 0.0 + 0.0j
            for k in range(n_sources):
                d = z_n - z_sources[k]
                if abs(d) < 1e-15:
                    break
                f_val += source_strengths[k] / d
                fp_val -= source_strengths[k] / d**2
            else:
                if abs(fp_val) < 1e-30:
                    break
                step = f_val / fp_val
                z_n -= step
                if abs(step) < newton_tol:
                    # Check that it's actually a zero
                    check = 0.0 + 0.0j
                    for k in range(n_sources):
                        d = z_n - z_sources[k]
                        if abs(d) < 1e-15:
                            check = float('inf')
                            break
                        check += source_strengths[k] / d
                    if abs(check) < 1e-4:
                        # Deduplicate
                        is_dup = False
                        for existing in stagnation_points:
                            if abs(z_n - existing) < 1e-4:
                                is_dup = True
                                break
                        if not is_dup:
                            stagnation_points.append(z_n)
                    break
                continue
            break

    # Inverse project stagnation points to lat/lon
    stag_latlon = []
    if stagnation_points:
        stag_z = np.array(stagnation_points)
        stag_lats, stag_lons = _stereo_to_latlon(stag_z)
        for slat, slon in zip(stag_lats, stag_lons):
            if -90.0 <= slat <= 90.0:
                stag_latlon.append((float(slat), float(slon)))

    return PotentialFlowCoverage(
        stagnation_points_latlon=tuple(stag_latlon),
        coverage_potential_grid=coverage_potential,
        stream_function_grid=stream_function,
        uniformity_metric=float(uniformity),
        num_stagnation_points=len(stag_latlon),
    )


# ---------------------------------------------------------------------------
# P50: Matroid Independence for Minimum Coverage Sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageMatroid:
    """Matroid-theoretic analysis of minimum satellite coverage sets.

    Greedy algorithm on the coverage matroid identifies the minimum
    independent set (basis) that achieves full coverage.
    """
    rank: int                        # size of minimum covering set
    minimum_basis: tuple             # satellite indices in greedy order
    redundant_satellites: tuple      # satellite indices not in basis
    redundancy_ratio: float          # fraction of satellites that are redundant
    greedy_weight_basis: tuple       # marginal coverage gain at each greedy step


def compute_coverage_matroid(
    coverage_matrix: np.ndarray,
) -> CoverageMatroid:
    """Compute matroid independence for minimum coverage sets.

    Uses a greedy algorithm: at each step, add the satellite that covers
    the most currently-uncovered grid points. A satellite is redundant
    if removing it from the final set does not reduce total coverage.

    Args:
        coverage_matrix: Boolean/integer matrix of shape (n_sats, n_points).
            coverage_matrix[i, j] = 1 if satellite i covers grid point j.

    Returns:
        CoverageMatroid with rank, basis, and redundancy metrics.
    """
    coverage_matrix = np.asarray(coverage_matrix, dtype=np.float64)

    if coverage_matrix.size == 0 or coverage_matrix.shape[0] == 0:
        return CoverageMatroid(
            rank=0,
            minimum_basis=(),
            redundant_satellites=(),
            redundancy_ratio=0.0,
            greedy_weight_basis=(),
        )

    n_sats, n_points = coverage_matrix.shape
    covered = np.zeros(n_points, dtype=bool)
    total_points = n_points
    remaining = set(range(n_sats))
    basis = []
    weights = []

    # Greedy: pick satellite maximizing marginal coverage
    while remaining:
        best_sat = -1
        best_gain = 0
        for s in remaining:
            uncovered_by_s = np.sum(coverage_matrix[s, :] > 0.5)
            marginal = int(np.sum((coverage_matrix[s, :] > 0.5) & ~covered))
            if marginal > best_gain:
                best_gain = marginal
                best_sat = s
            elif marginal == best_gain and best_sat == -1:
                best_sat = s

        if best_gain == 0 or best_sat == -1:
            break

        basis.append(best_sat)
        weights.append(best_gain / max(total_points, 1))
        covered |= coverage_matrix[best_sat, :] > 0.5
        remaining.discard(best_sat)

    # Independence oracle: check each basis member for non-redundancy
    # A satellite is redundant if removing it does not reduce coverage
    final_covered = np.zeros(n_points, dtype=bool)
    for s in basis:
        final_covered |= coverage_matrix[s, :] > 0.5

    # Determine which satellites outside basis are redundant (all of them)
    # and which basis members could be removed
    true_basis = []
    for i, s in enumerate(basis):
        others_covered = np.zeros(n_points, dtype=bool)
        for j, s2 in enumerate(basis):
            if j != i:
                others_covered |= coverage_matrix[s2, :] > 0.5
        if np.sum(others_covered) < np.sum(final_covered):
            true_basis.append(s)

    # If true_basis is empty but we had coverage, at least first is needed
    if not true_basis and basis:
        true_basis = basis[:]

    redundant = tuple(s for s in range(n_sats) if s not in true_basis)
    n_total = n_sats
    redundancy_ratio = len(redundant) / n_total if n_total > 0 else 0.0

    return CoverageMatroid(
        rank=len(true_basis),
        minimum_basis=tuple(basis),
        redundant_satellites=redundant,
        redundancy_ratio=float(redundancy_ratio),
        greedy_weight_basis=tuple(weights),
    )


# ---------------------------------------------------------------------------
# P54: Neural Field Equation for Coverage Demand
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemandFieldEquilibrium:
    """Equilibrium of Wilson-Cowan neural field equation for coverage demand.

    Models coverage demand as a neural field with Mexican-hat lateral
    connectivity, iterated to equilibrium. The deficit field shows
    where demand exceeds supply.
    """
    demand_field: np.ndarray          # equilibrium demand field u(x)
    supply_field: np.ndarray          # supply field (input)
    deficit_field: np.ndarray         # max(0, demand - supply)
    peak_deficit_latlon: tuple        # (lat, lon) of peak deficit
    total_deficit: float              # sum of deficit field
    demand_concentration_index: float # Gini-like concentration


def _mexican_hat_kernel(
    n_lat: int, n_lon: int,
    a_excitatory: float = 1.5,
    sigma_excitatory: float = 3.0,
    a_inhibitory: float = 0.75,
    sigma_inhibitory: float = 6.0,
) -> np.ndarray:
    """Mexican hat connectivity kernel.

    w(r) = A_e * exp(-r^2 / 2*sigma_e^2) - A_i * exp(-r^2 / 2*sigma_i^2)
    """
    cy, cx = n_lat // 2, n_lon // 2
    y = np.arange(n_lat) - cy
    x = np.arange(n_lon) - cx
    xx, yy = np.meshgrid(x, y)
    r2 = xx**2.0 + yy**2.0
    kernel = (
        a_excitatory * np.exp(-r2 / (2.0 * sigma_excitatory**2))
        - a_inhibitory * np.exp(-r2 / (2.0 * sigma_inhibitory**2))
    )
    return kernel


def _sigmoid(u: np.ndarray, beta: float = 5.0, theta: float = 0.5) -> np.ndarray:
    """Sigmoid activation: S(u) = 1 / (1 + exp(-beta*(u - theta)))."""
    arg = -beta * (u - theta)
    # Clip to avoid overflow
    arg = np.clip(arg, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(arg))


def compute_demand_field_equilibrium(
    supply: np.ndarray,
    external_demand: np.ndarray,
    grid_resolution: int = 18,
    tau: float = 1.0,
    dt: float = 0.1,
    max_steps: int = 200,
    convergence_tol: float = 1e-4,
    sigmoid_beta: float = 5.0,
    sigmoid_theta: float = 0.5,
) -> DemandFieldEquilibrium:
    """Compute equilibrium demand field via Wilson-Cowan neural field equation.

    Iterates: tau * du/dt = -u + conv(w, S(u)) + I_ext
    where w is a Mexican-hat kernel and S is a sigmoid.

    Args:
        supply: Supply field (n_lat, n_lon).
        external_demand: External demand input I_ext (n_lat, n_lon).
        grid_resolution: Grid resolution (used for lat/lon mapping).
        tau: Time constant.
        dt: Integration time step.
        max_steps: Maximum Euler steps.
        convergence_tol: Convergence tolerance (max |du|).
        sigmoid_beta: Sigmoid steepness.
        sigmoid_theta: Sigmoid threshold.

    Returns:
        DemandFieldEquilibrium with demand, supply, and deficit fields.
    """
    supply = np.asarray(supply, dtype=np.float64)
    external_demand = np.asarray(external_demand, dtype=np.float64)

    n_lat, n_lon = supply.shape

    # Build Mexican-hat kernel (smaller than field, for convolution)
    k_size = min(n_lat, 7)
    k_lon_size = min(n_lon, 2 * k_size)
    kernel = _mexican_hat_kernel(k_size, k_lon_size)

    # Normalize kernel
    kernel /= max(np.sum(np.abs(kernel)), 1e-15)

    # Initialize demand field
    u = np.copy(external_demand)

    # Euler integration of Wilson-Cowan
    for _ in range(max_steps):
        # Sigmoid of current field
        s_u = _sigmoid(u, beta=sigmoid_beta, theta=sigmoid_theta)

        # Spatial convolution with kernel (using zero-padding)
        # Manual convolution via numpy (no scipy dependency)
        ky, kx = kernel.shape
        py, px = ky // 2, kx // 2
        padded = np.pad(s_u, ((py, py), (px, px)), mode='wrap')
        conv = np.zeros_like(u)
        for iy in range(ky):
            for ix in range(kx):
                conv += kernel[iy, ix] * padded[iy:iy + n_lat, ix:ix + n_lon]

        # Euler step: du/dt = (-u + conv + I_ext) / tau
        du = dt * (-u + conv + external_demand) / tau
        u += du

        if np.max(np.abs(du)) < convergence_tol:
            break

    # Deficit field = max(0, demand - supply)
    deficit = np.maximum(0.0, u - supply)

    # Peak deficit location
    flat_idx = int(np.argmax(deficit))
    peak_lat_idx = flat_idx // n_lon
    peak_lon_idx = flat_idx % n_lon

    # Map grid indices to lat/lon
    lat_vals = np.linspace(-90.0, 90.0, n_lat)
    lon_vals = np.linspace(-180.0, 180.0, n_lon)
    peak_lat = float(lat_vals[min(peak_lat_idx, n_lat - 1)])
    peak_lon = float(lon_vals[min(peak_lon_idx, n_lon - 1)])

    total_deficit = float(np.sum(deficit))

    # Demand concentration index: normalized HHI (Herfindahl)
    # Measures how concentrated the deficit is
    if total_deficit > 1e-15:
        deficit_flat = deficit.ravel()
        shares = deficit_flat / total_deficit
        hhi = float(np.sum(shares**2))
        n_cells = len(shares)
        # Normalized HHI: (HHI - 1/N) / (1 - 1/N)
        if n_cells > 1:
            concentration = (hhi - 1.0 / n_cells) / (1.0 - 1.0 / n_cells)
        else:
            concentration = 1.0
        concentration = max(0.0, min(1.0, concentration))
    else:
        concentration = 0.0

    return DemandFieldEquilibrium(
        demand_field=u,
        supply_field=supply,
        deficit_field=deficit,
        peak_deficit_latlon=(peak_lat, peak_lon),
        total_deficit=total_deficit,
        demand_concentration_index=float(concentration),
    )


# ---------------------------------------------------------------------------
# P64: Ergodic Coverage Partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErgodicPartition:
    """Ergodic partition of coverage over orbit periods.

    Computes time-averaged coverage at each grid point, measures
    ergodicity defect, identifies ergodic components, and estimates
    mixing time.
    """
    coverage_time_averages: np.ndarray  # time-average at each grid point
    ergodicity_defect: float            # max(f_g) - min(f_g)
    num_ergodic_components: int         # number of distinct components
    component_boundaries: np.ndarray    # component label per grid cell
    mixing_time_orbits: float           # orbits until running avg converges
    is_ergodic: bool                    # True if defect < threshold


def compute_ergodic_partition(
    coverage_timeseries: np.ndarray,
    orbit_period_steps: int,
    ergodicity_threshold: float = 0.2,
    num_clusters: int = 0,
) -> ErgodicPartition:
    """Compute ergodic partition of coverage time series.

    Computes time-averaged coverage at each grid point, measures
    the ergodicity defect (max - min of time averages), clusters
    grid points into ergodic components, and estimates mixing time.

    Args:
        coverage_timeseries: Array of shape (n_time, n_lat, n_lon) with
            coverage values at each time step.
        orbit_period_steps: Number of time steps per orbit period.
        ergodicity_threshold: Defect threshold below which system is ergodic.
        num_clusters: Number of ergodic components to cluster into.
            If 0, auto-detect.

    Returns:
        ErgodicPartition with time averages, defect, and components.
    """
    coverage_timeseries = np.asarray(coverage_timeseries, dtype=np.float64)
    n_time, n_lat, n_lon = coverage_timeseries.shape

    # Time-average at each grid point
    time_avg = np.mean(coverage_timeseries, axis=0)

    # Ergodicity defect: max(f_g) - min(f_g)
    defect = float(np.max(time_avg) - np.min(time_avg))

    # Cluster grid points by time-average into ergodic components
    # Simple 1D k-means on the time-average values
    flat_avg = time_avg.ravel()
    n_cells = len(flat_avg)

    if num_clusters <= 0:
        # Auto-detect: use the number of distinct value clusters
        # Heuristic: number of clusters = max(1, ceil(defect / ergodicity_threshold))
        num_clusters = max(1, min(10, int(np.ceil(defect / max(ergodicity_threshold, 1e-10)))))
        # Also limit by unique values
        n_unique = len(np.unique(np.round(flat_avg, 4)))
        num_clusters = min(num_clusters, max(1, n_unique))

    # 1D k-means clustering
    if n_cells == 0:
        labels = np.array([], dtype=int).reshape(n_lat, n_lon)
        n_components = 0
    else:
        # Initialize centroids evenly spaced
        sorted_vals = np.sort(flat_avg)
        step = max(1, n_cells // num_clusters)
        centroids = np.array([sorted_vals[min(i * step, n_cells - 1)]
                              for i in range(num_clusters)])

        labels_flat = np.zeros(n_cells, dtype=int)
        for _ in range(50):  # k-means iterations
            # Assign each point to nearest centroid
            dists = np.abs(flat_avg[:, None] - centroids[None, :])
            new_labels = np.argmin(dists, axis=1)

            if np.array_equal(new_labels, labels_flat):
                break
            labels_flat = new_labels

            # Update centroids
            for c in range(num_clusters):
                mask = labels_flat == c
                if np.any(mask):
                    centroids[c] = np.mean(flat_avg[mask])

        labels = labels_flat.reshape(n_lat, n_lon)
        n_components = len(np.unique(labels_flat))

    # Estimate mixing time: number of orbits until running average converges
    # Convergence criterion: running avg within 5% of final avg for all points
    mixing_time = float(n_time) / max(orbit_period_steps, 1)  # default: all time

    if n_time > 1 and n_cells > 0:
        # Check convergence of running average at each step
        running_sum = np.zeros((n_lat, n_lon), dtype=np.float64)
        final_avg = time_avg
        tol = max(0.05 * defect, 1e-6)
        converged_step = n_time

        for t in range(n_time):
            running_sum += coverage_timeseries[t]
            running_avg = running_sum / (t + 1)
            max_dev = float(np.max(np.abs(running_avg - final_avg)))
            if max_dev < tol and t >= orbit_period_steps:
                converged_step = t + 1
                break

        mixing_time = float(converged_step) / max(orbit_period_steps, 1)

    is_ergodic = defect < ergodicity_threshold

    return ErgodicPartition(
        coverage_time_averages=time_avg,
        ergodicity_defect=defect,
        num_ergodic_components=n_components,
        component_boundaries=labels,
        mixing_time_orbits=max(1.0, mixing_time),
        is_ergodic=is_ergodic,
    )
