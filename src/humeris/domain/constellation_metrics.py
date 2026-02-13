# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Constellation-level derived metrics.

Coverage statistics, revisit statistics, eclipse statistics, N-1 redundancy,
constellation score, thermal cycling, ground network metrics, deployment phasing.

No external dependencies — only stdlib math/dataclasses/datetime.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.coverage import CoveragePoint, compute_coverage_snapshot
from humeris.domain.observation import GroundStation
from humeris.domain.access_windows import AccessWindow, compute_access_windows
from humeris.domain.eclipse import (
    EclipseType,
    is_eclipsed,
)
from humeris.domain.solar import sun_position_eci

_MU = OrbitalConstants.MU_EARTH
_R_E = OrbitalConstants.R_EARTH


# --- Types ---

@dataclass(frozen=True)
class CoverageStatistics:
    """Aggregate statistics from a coverage grid."""
    mean_visible: float
    max_visible: int
    min_visible: int
    std_visible: float
    percent_covered: float
    n_fold_coverage: dict[int, float]


@dataclass(frozen=True)
class RevisitStatistics:
    """Aggregate revisit time statistics."""
    mean_revisit_s: float
    max_revisit_s: float
    percentile_95_s: float
    percent_within_threshold: float


@dataclass(frozen=True)
class EclipseStatistics:
    """Eclipse statistics over a time window."""
    eclipse_fraction: float
    num_eclipses: int
    mean_duration_s: float
    max_duration_s: float
    total_eclipse_s: float


@dataclass(frozen=True)
class RedundancyResult:
    """N-1 redundancy analysis result."""
    baseline_coverage_pct: float
    worst_degraded_coverage_pct: float
    worst_case_satellite_idx: int
    degradation_pct: float


@dataclass(frozen=True)
class ConstellationScore:
    """Weighted constellation performance score."""
    coverage_score: float
    revisit_score: float
    redundancy_score: float
    overall_score: float


@dataclass(frozen=True)
class ThermalCycling:
    """Thermal cycling count and durations."""
    num_cycles: int
    eclipse_durations_s: tuple[float, ...]
    sunlit_durations_s: tuple[float, ...]


@dataclass(frozen=True)
class GroundNetworkMetrics:
    """Aggregate ground network contact metrics."""
    total_contact_s: float
    max_gap_s: float
    num_passes: int
    contact_per_station: dict[str, float]
    best_station: str


@dataclass(frozen=True)
class DeploymentPhasing:
    """Deployment phasing plan for constellation build-up."""
    total_delta_v_ms: float
    total_time_s: float
    per_satellite: tuple[tuple[int, float, float], ...]


# --- Functions ---

def compute_coverage_statistics(
    grid: list[CoveragePoint],
    n_fold_levels: list[int] | None = None,
) -> CoverageStatistics:
    """Compute aggregate statistics from a coverage grid.

    Args:
        grid: List of CoveragePoint from compute_coverage_snapshot.
        n_fold_levels: List of N values for N-fold coverage percentages.

    Returns:
        CoverageStatistics with mean, max, min, std, percent covered, N-fold.
    """
    if not grid:
        return CoverageStatistics(
            mean_visible=0.0, max_visible=0, min_visible=0,
            std_visible=0.0, percent_covered=0.0, n_fold_coverage={},
        )

    counts_list = [p.visible_count for p in grid]
    counts = np.array(counts_list)
    n = len(counts)

    mean_v = float(np.mean(counts))
    max_v = int(np.max(counts))
    min_v = int(np.min(counts))

    # Population standard deviation
    std_v = float(np.std(counts))

    # Percent of grid with at least 1 visible
    covered = int(np.sum(counts > 0))
    pct_covered = 100.0 * covered / n

    # N-fold coverage
    n_fold: dict[int, float] = {}
    if n_fold_levels:
        for level in n_fold_levels:
            above = int(np.sum(counts >= level))
            n_fold[level] = 100.0 * above / n

    return CoverageStatistics(
        mean_visible=mean_v,
        max_visible=max_v,
        min_visible=min_v,
        std_visible=std_v,
        percent_covered=pct_covered,
        n_fold_coverage=n_fold,
    )


def compute_revisit_statistics(
    revisit_times_s: list[float],
    threshold_s: float = 3600.0,
) -> RevisitStatistics:
    """Compute aggregate revisit time statistics.

    Args:
        revisit_times_s: List of revisit times in seconds.
        threshold_s: Threshold for compliance percentage.

    Returns:
        RevisitStatistics with mean, max, p95, and compliance percentage.
    """
    if not revisit_times_s:
        return RevisitStatistics(
            mean_revisit_s=0.0, max_revisit_s=0.0,
            percentile_95_s=0.0, percent_within_threshold=0.0,
        )

    sorted_times = sorted(revisit_times_s)
    n = len(sorted_times)

    mean_r = sum(sorted_times) / n
    max_r = sorted_times[-1]

    # 95th percentile (linear interpolation)
    idx_95 = 0.95 * (n - 1)
    lower = int(idx_95)
    upper = min(lower + 1, n - 1)
    frac = idx_95 - lower
    p95 = sorted_times[lower] + frac * (sorted_times[upper] - sorted_times[lower])

    # Percent within threshold
    within = sum(1 for t in sorted_times if t <= threshold_s)
    pct_within = 100.0 * within / n

    return RevisitStatistics(
        mean_revisit_s=mean_r,
        max_revisit_s=max_r,
        percentile_95_s=p95,
        percent_within_threshold=pct_within,
    )


def compute_eclipse_statistics(
    state: OrbitalState,
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
) -> EclipseStatistics:
    """Compute eclipse statistics over a time window.

    Sweeps the orbit and counts eclipse entry/exit transitions.

    Args:
        state: Satellite orbital state.
        epoch: Start time.
        duration: Analysis duration.
        step: Time step for sweep.

    Returns:
        EclipseStatistics with fraction, count, durations.
    """
    step_s = step.total_seconds()
    duration_s = duration.total_seconds()
    if step_s <= 0 or duration_s <= 0:
        return EclipseStatistics(
            eclipse_fraction=0.0, num_eclipses=0,
            mean_duration_s=0.0, max_duration_s=0.0, total_eclipse_s=0.0,
        )

    eclipsed_steps = 0
    total_steps = 0
    in_eclipse = False
    eclipse_start_s = 0.0
    eclipse_durations: list[float] = []

    elapsed = 0.0
    while elapsed <= duration_s + 1e-9:
        current_time = epoch + timedelta(seconds=elapsed)
        pos_eci, _ = propagate_to(state, current_time)
        sun = sun_position_eci(current_time)
        sat_pos = (pos_eci[0], pos_eci[1], pos_eci[2])
        is_dark = is_eclipsed(sat_pos, sun.position_eci_m) != EclipseType.NONE

        total_steps += 1
        if is_dark:
            eclipsed_steps += 1
            if not in_eclipse:
                eclipse_start_s = elapsed
                in_eclipse = True
        else:
            if in_eclipse:
                eclipse_durations.append(elapsed - eclipse_start_s)
                in_eclipse = False

        elapsed += step_s

    # Close trailing eclipse
    if in_eclipse:
        eclipse_durations.append(min(elapsed, duration_s) - eclipse_start_s)

    fraction = eclipsed_steps / total_steps if total_steps > 0 else 0.0
    num_eclipses = len(eclipse_durations)
    total_eclipse_s = sum(eclipse_durations)
    mean_dur = total_eclipse_s / num_eclipses if num_eclipses > 0 else 0.0
    max_dur = max(eclipse_durations) if eclipse_durations else 0.0

    return EclipseStatistics(
        eclipse_fraction=fraction,
        num_eclipses=num_eclipses,
        mean_duration_s=mean_dur,
        max_duration_s=max_dur,
        total_eclipse_s=total_eclipse_s,
    )


def compute_n_minus_1_redundancy(
    states: list[OrbitalState],
    time: datetime,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    min_elevation_deg: float = 10.0,
) -> RedundancyResult:
    """Compute N-1 redundancy: coverage degradation when removing each satellite.

    For each satellite, compute coverage without it. Report the worst case.

    Args:
        states: Full constellation orbital states.
        time: Analysis time.
        lat_step_deg: Coverage grid latitude step.
        lon_step_deg: Coverage grid longitude step.
        min_elevation_deg: Minimum elevation for visibility.

    Returns:
        RedundancyResult with baseline and worst-case coverage.
    """
    if not states:
        return RedundancyResult(
            baseline_coverage_pct=0.0, worst_degraded_coverage_pct=0.0,
            worst_case_satellite_idx=0, degradation_pct=0.0,
        )

    # Baseline coverage
    baseline_grid = compute_coverage_snapshot(
        states, time, lat_step_deg, lon_step_deg, min_elevation_deg,
    )
    baseline_pct = _coverage_percent(baseline_grid)

    if len(states) == 1:
        return RedundancyResult(
            baseline_coverage_pct=baseline_pct,
            worst_degraded_coverage_pct=0.0,
            worst_case_satellite_idx=0,
            degradation_pct=baseline_pct,
        )

    worst_pct = 100.0
    worst_idx = 0

    for i in range(len(states)):
        reduced = states[:i] + states[i + 1:]
        reduced_grid = compute_coverage_snapshot(
            reduced, time, lat_step_deg, lon_step_deg, min_elevation_deg,
        )
        pct = _coverage_percent(reduced_grid)
        if pct < worst_pct:
            worst_pct = pct
            worst_idx = i

    return RedundancyResult(
        baseline_coverage_pct=baseline_pct,
        worst_degraded_coverage_pct=worst_pct,
        worst_case_satellite_idx=worst_idx,
        degradation_pct=baseline_pct - worst_pct,
    )


def _coverage_percent(grid: list[CoveragePoint]) -> float:
    """Percentage of grid points with at least 1 visible satellite."""
    if not grid:
        return 0.0
    covered = sum(1 for p in grid if p.visible_count > 0)
    return 100.0 * covered / len(grid)


def compute_constellation_score(
    coverage_pct: float,
    max_revisit_s: float,
    target_revisit_s: float,
    degradation_pct: float,
) -> ConstellationScore:
    """Compute weighted constellation performance score.

    Scores:
        coverage_score = coverage_pct / 100 (capped at 1.0)
        revisit_score = min(1.0, target_revisit / max_revisit) if max_revisit > 0
        redundancy_score = max(0, 1.0 - degradation_pct / 100)
        overall = 0.4*coverage + 0.35*revisit + 0.25*redundancy

    Args:
        coverage_pct: Spatial coverage percentage (0-100).
        max_revisit_s: Maximum revisit gap in seconds.
        target_revisit_s: Target revisit time in seconds.
        degradation_pct: Coverage degradation from N-1 analysis (0-100).

    Returns:
        ConstellationScore with component and overall scores.
    """
    cov_score = min(1.0, max(0.0, coverage_pct / 100.0))

    if max_revisit_s > 0:
        rev_score = min(1.0, max(0.0, target_revisit_s / max_revisit_s))
    else:
        rev_score = 1.0

    red_score = max(0.0, min(1.0, 1.0 - degradation_pct / 100.0))

    overall = 0.4 * cov_score + 0.35 * rev_score + 0.25 * red_score

    return ConstellationScore(
        coverage_score=cov_score,
        revisit_score=rev_score,
        redundancy_score=red_score,
        overall_score=overall,
    )


def compute_thermal_cycling(
    state: OrbitalState,
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
) -> ThermalCycling:
    """Count thermal cycles (sunlight-to-eclipse transitions) over a time window.

    Each eclipse entry is one thermal cycle.

    Args:
        state: Satellite orbital state.
        epoch: Start time.
        duration: Analysis duration.
        step: Time step.

    Returns:
        ThermalCycling with cycle count and duration lists.
    """
    step_s = step.total_seconds()
    duration_s = duration.total_seconds()
    if step_s <= 0 or duration_s <= 0:
        return ThermalCycling(
            num_cycles=0, eclipse_durations_s=(), sunlit_durations_s=(),
        )

    eclipse_durs: list[float] = []
    sunlit_durs: list[float] = []

    in_eclipse = False
    phase_start_s = 0.0

    elapsed = 0.0
    while elapsed <= duration_s + 1e-9:
        current_time = epoch + timedelta(seconds=elapsed)
        pos_eci, _ = propagate_to(state, current_time)
        sun = sun_position_eci(current_time)
        sat_pos = (pos_eci[0], pos_eci[1], pos_eci[2])
        is_dark = is_eclipsed(sat_pos, sun.position_eci_m) != EclipseType.NONE

        if is_dark and not in_eclipse:
            # Entering eclipse
            if elapsed > 0:
                sunlit_durs.append(elapsed - phase_start_s)
            phase_start_s = elapsed
            in_eclipse = True
        elif not is_dark and in_eclipse:
            # Exiting eclipse
            eclipse_durs.append(elapsed - phase_start_s)
            phase_start_s = elapsed
            in_eclipse = False

        elapsed += step_s

    # Close trailing phase
    final_s = min(elapsed, duration_s)
    if in_eclipse:
        eclipse_durs.append(final_s - phase_start_s)
    elif final_s > phase_start_s:
        sunlit_durs.append(final_s - phase_start_s)

    return ThermalCycling(
        num_cycles=len(eclipse_durs),
        eclipse_durations_s=tuple(eclipse_durs),
        sunlit_durations_s=tuple(sunlit_durs),
    )


def compute_ground_network_metrics(
    stations: list[GroundStation],
    states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    min_elevation_deg: float = 10.0,
) -> GroundNetworkMetrics:
    """Compute aggregate metrics for a ground station network.

    For each station × satellite pair, computes access windows and
    aggregates contact time, gaps, and per-station metrics.

    Args:
        stations: List of ground stations.
        states: List of satellite orbital states.
        epoch: Start time.
        duration: Analysis duration.
        step: Time step.
        min_elevation_deg: Minimum elevation for visibility.

    Returns:
        GroundNetworkMetrics with aggregate and per-station data.
    """
    station_contact: dict[str, float] = {}
    all_events: list[tuple[datetime, datetime]] = []  # (rise, set) across all

    for station in stations:
        station_total = 0.0
        for state in states:
            windows = compute_access_windows(
                station, state, epoch, duration, step, min_elevation_deg,
            )
            for w in windows:
                station_total += w.duration_seconds
                all_events.append((w.rise_time, w.set_time))
        station_contact[station.name] = station_total

    # Merge overlapping events for total unique contact
    total_contact, max_gap, num_passes = _merge_contact_events(all_events, epoch, duration)

    # Best station
    if station_contact:
        best = max(station_contact, key=station_contact.get)
    else:
        best = ""

    return GroundNetworkMetrics(
        total_contact_s=total_contact,
        max_gap_s=max_gap,
        num_passes=num_passes,
        contact_per_station=station_contact,
        best_station=best,
    )


def _merge_contact_events(
    events: list[tuple[datetime, datetime]],
    epoch: datetime,
    duration: timedelta,
) -> tuple[float, float, int]:
    """Merge overlapping contact events and compute total/gap/count."""
    if not events:
        return 0.0, duration.total_seconds(), 0

    # Sort by rise time
    sorted_events = sorted(events, key=lambda e: e[0])

    merged: list[tuple[datetime, datetime]] = []
    current_rise, current_set = sorted_events[0]

    for rise, set_time in sorted_events[1:]:
        if rise <= current_set:
            # Overlapping — extend
            current_set = max(current_set, set_time)
        else:
            merged.append((current_rise, current_set))
            current_rise, current_set = rise, set_time
    merged.append((current_rise, current_set))

    total_contact = sum((s - r).total_seconds() for r, s in merged)

    # Gaps between merged windows
    gaps: list[float] = []
    for i in range(1, len(merged)):
        gap = (merged[i][0] - merged[i - 1][1]).total_seconds()
        if gap > 0:
            gaps.append(gap)

    # Gap before first contact and after last
    pre_gap = (merged[0][0] - epoch).total_seconds()
    post_gap = ((epoch + duration) - merged[-1][1]).total_seconds()
    if pre_gap > 0:
        gaps.append(pre_gap)
    if post_gap > 0:
        gaps.append(post_gap)

    max_gap = max(gaps) if gaps else 0.0

    return total_contact, max_gap, len(merged)


def compute_deployment_phasing(
    num_planes: int,
    sats_per_plane: int,
    orbit_radius_m: float,
) -> DeploymentPhasing:
    """Compute deployment phasing plan for constellation build-up.

    Assumes all satellites are deployed from a single launch vehicle into one
    plane, then use phasing maneuvers to reach their target slots.

    In-plane phasing: satellites drift to different true anomaly slots.
    Inter-plane: RAAN drift via altitude offset (natural J2 differential drift).

    Simplified model: each satellite needs a phasing maneuver proportional
    to its angular offset from the deployment point.

    Args:
        num_planes: Number of orbital planes.
        sats_per_plane: Satellites per plane.
        orbit_radius_m: Target orbit radius (m).

    Returns:
        DeploymentPhasing with total delta-V, time, and per-satellite breakdown.
    """
    if orbit_radius_m <= 0:
        raise ValueError(f"orbit_radius_m must be positive, got {orbit_radius_m}")

    total_sats = num_planes * sats_per_plane
    T = 2.0 * np.pi * float(np.sqrt(orbit_radius_m**3 / _MU))
    v_circ = float(np.sqrt(_MU / orbit_radius_m))

    per_sat: list[tuple[int, float, float]] = []
    total_dv = 0.0
    max_time = 0.0

    for plane in range(num_planes):
        for slot in range(sats_per_plane):
            sat_idx = plane * sats_per_plane + slot

            # In-plane phase angle offset
            phase_offset_rad = 2.0 * np.pi * slot / sats_per_plane

            if plane == 0 and slot == 0:
                # Reference satellite: no maneuver needed
                per_sat.append((sat_idx, 0.0, 0.0))
                continue

            # In-plane phasing delta-V
            # Use multi-orbit phasing to keep altitude change modest
            if phase_offset_rad > 0 and slot > 0:
                # Choose enough phasing orbits so period change < 25%
                n_phase_orbits = max(1, int(np.ceil(phase_offset_rad / (0.5 * np.pi))))
                T_phase = T * (1.0 - phase_offset_rad / (2.0 * np.pi * n_phase_orbits))
                a_phase = (_MU * (T_phase / (2.0 * np.pi))**2)**(1.0 / 3.0)
                v_phase = float(np.sqrt(_MU * (2.0 / orbit_radius_m - 1.0 / a_phase)))
                dv_in_plane = 2.0 * abs(v_phase - v_circ)
                t_in_plane = T_phase * n_phase_orbits
            else:
                dv_in_plane = 0.0
                t_in_plane = 0.0

            # Inter-plane RAAN drift via altitude offset
            # RAAN separation per plane
            if num_planes > 1 and plane > 0:
                raan_offset_rad = 2.0 * np.pi * plane / num_planes
                # Time to drift RAAN by differential altitude
                # Simplified: ~30 km altitude offset gives ~0.05 deg/day RAAN drift
                # For larger offsets, proportional
                drift_rate_deg_day = 0.05  # per 30 km offset
                raan_offset_deg = float(np.degrees(raan_offset_rad))
                t_drift_days = raan_offset_deg / drift_rate_deg_day
                t_drift_s = t_drift_days * 86400.0

                # Delta-V to raise/lower altitude by 30 km and return
                delta_alt = 30_000.0  # 30 km in meters
                r_drift = orbit_radius_m + delta_alt
                v_drift = float(np.sqrt(_MU * (2.0 / orbit_radius_m - 1.0 / ((orbit_radius_m + r_drift) / 2.0))))
                dv_plane = 2.0 * abs(v_drift - v_circ)
            else:
                dv_plane = 0.0
                t_drift_s = 0.0

            dv_total_sat = dv_in_plane + dv_plane
            t_total_sat = max(t_in_plane, t_drift_s)
            per_sat.append((sat_idx, dv_total_sat, t_total_sat))
            total_dv += dv_total_sat
            max_time = max(max_time, t_total_sat)

    return DeploymentPhasing(
        total_delta_v_ms=total_dv,
        total_time_s=max_time,
        per_satellite=tuple(per_sat),
    )
