# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Information-theoretic analysis of constellation performance.

Eclipse as Binary Erasure Channel, coverage spectral analysis via DFT,
and marginal satellite value via Shannon entropy and Fisher information.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_ecef_to
from humeris.domain.link_budget import LinkConfig, compute_link_budget
from humeris.domain.eclipse import eclipse_fraction
from humeris.domain.observation import GroundStation, compute_observation
from humeris.domain.dilution_of_precision import compute_dop
from humeris.domain.linalg import naive_dft, mat_determinant


@dataclass(frozen=True)
class EclipseChannelCapacity:
    """Eclipse-degraded communication capacity modeled as BEC."""
    awgn_capacity_bps: float
    erasure_fraction: float
    bec_capacity_bps: float
    scheduled_throughput_bps: float
    scheduling_gain: float


@dataclass(frozen=True)
class CoverageSpectrum:
    """Frequency-domain analysis of coverage time series."""
    lat_deg: float
    lon_deg: float
    frequencies_hz: tuple
    power_density: tuple
    dominant_frequency_hz: float
    dominant_period_s: float
    orbital_frequency_hz: float
    resonance_ratio: float


@dataclass(frozen=True)
class MarginalSatelliteValue:
    """Information gain from adding one satellite to constellation."""
    coverage_entropy_before: float
    coverage_entropy_after: float
    coverage_information_gain: float
    positioning_info_gain: float
    total_information_value: float


def compute_eclipse_channel_capacity(
    state: OrbitalState,
    epoch: datetime,
    link_config: LinkConfig,
    distance_m: float,
) -> EclipseChannelCapacity:
    """Model ISL through eclipse as a Binary Erasure Channel.

    C_bec = (1 - epsilon) * C_awgn
    where epsilon is the eclipse fraction.
    Scheduling gain comes from knowing eclipse windows in advance.
    """
    eps = eclipse_fraction(state, epoch)
    budget = compute_link_budget(link_config, distance_m)
    c_awgn = budget.max_data_rate_bps

    c_bec = (1.0 - eps) * c_awgn

    # Scheduling gain: with perfect eclipse foreknowledge, can schedule
    # all data transfers during sunlit portions, achieving higher throughput
    # than blind transmission. Gain = 1 / (1 - epsilon) for the scheduled portion,
    # but total throughput is still bounded by c_bec. Effective scheduled throughput
    # concentrates transmission during (1-eps) fraction.
    if eps < 1.0:
        scheduled = c_bec  # Same total throughput, but concentrated
        gain = 1.0 / (1.0 - eps) if eps > 0.0 else 1.0
    else:
        scheduled = 0.0
        gain = 1.0

    return EclipseChannelCapacity(
        awgn_capacity_bps=c_awgn,
        erasure_fraction=eps,
        bec_capacity_bps=c_bec,
        scheduled_throughput_bps=scheduled,
        scheduling_gain=gain,
    )


def _binary_coverage_signal(
    states: list,
    start: datetime,
    duration_s: float,
    step_s: float,
    lat_deg: float,
    lon_deg: float,
    min_elevation_deg: float,
) -> list:
    """Generate binary coverage time series at a ground point."""
    station = GroundStation(name='cov', lat_deg=lat_deg, lon_deg=lon_deg)
    num_steps = int(duration_s / step_s) + 1
    signal = []

    for i in range(num_steps):
        t = start + timedelta(seconds=i * step_s)
        covered = False
        for s in states:
            ecef = propagate_ecef_to(s, t)
            obs = compute_observation(station, ecef)
            if obs.elevation_deg >= min_elevation_deg:
                covered = True
                break
        signal.append(1.0 if covered else 0.0)

    return signal


def compute_coverage_spectrum(
    states: list,
    start: datetime,
    duration_s: float,
    step_s: float,
    lat_deg: float,
    lon_deg: float,
    min_elevation_deg: float = 10.0,
) -> CoverageSpectrum:
    """Frequency-domain analysis of coverage at a ground point.

    Computes binary coverage signal, applies DFT, and identifies
    dominant frequency (orbital resonance).
    """
    signal = _binary_coverage_signal(
        states, start, duration_s, step_s,
        lat_deg, lon_deg, min_elevation_deg,
    )

    sample_rate = 1.0 / step_s  # Hz
    dft_result = naive_dft(signal, sample_rate)

    # Power spectral density
    n = len(signal)
    psd = tuple(m * m for m in dft_result.magnitudes)

    # Find dominant frequency (excluding DC, only first half)
    half_n = max(1, n // 2)
    if half_n > 1:
        peak_idx = max(range(1, half_n), key=lambda k: psd[k])
        dominant_freq = dft_result.frequencies_hz[peak_idx]
    else:
        peak_idx = 0
        dominant_freq = 0.0

    dominant_period = 1.0 / dominant_freq if dominant_freq > 0.0 else float('inf')

    # Orbital frequency of first satellite
    if states:
        orbital_freq = states[0].mean_motion_rad_s / (2.0 * math.pi)
    else:
        orbital_freq = 0.0

    resonance = dominant_freq / orbital_freq if orbital_freq > 0.0 else 0.0

    return CoverageSpectrum(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        frequencies_hz=dft_result.frequencies_hz,
        power_density=psd,
        dominant_frequency_hz=dominant_freq,
        dominant_period_s=dominant_period,
        orbital_frequency_hz=orbital_freq,
        resonance_ratio=resonance,
    )


def _shannon_entropy(values: list) -> float:
    """Shannon entropy of a discrete distribution (list of probabilities)."""
    total = sum(values)
    if total <= 0.0:
        return 0.0
    h = 0.0
    for v in values:
        p = v / total
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def _coverage_distribution(
    states: list,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    min_elevation_deg: float,
    lat_step_deg: float,
    lon_step_deg: float,
) -> list:
    """Compute coverage fraction at grid points."""
    fractions = []
    lat = -90.0 + lat_step_deg / 2.0
    while lat < 90.0:
        lon = -180.0 + lon_step_deg / 2.0
        while lon < 180.0:
            station = GroundStation(name='grid', lat_deg=lat, lon_deg=lon)
            num_steps = int(duration_s / step_s) + 1
            covered_count = 0
            for i in range(num_steps):
                t = epoch + timedelta(seconds=i * step_s)
                for s in states:
                    ecef = propagate_ecef_to(s, t)
                    obs = compute_observation(station, ecef)
                    if obs.elevation_deg >= min_elevation_deg:
                        covered_count += 1
                        break
            fractions.append(covered_count / max(1, num_steps))
            lon += lon_step_deg
        lat += lat_step_deg
    return fractions


def _fisher_log_det(
    states: list,
    epoch: datetime,
    lat_step_deg: float,
    lon_step_deg: float,
) -> float:
    """Compute sum of log(det(FIM)) over grid points for positioning info."""
    total_log_det = 0.0
    count = 0

    # Get satellite ECEF positions
    sat_positions = [propagate_ecef_to(s, epoch) for s in states]

    lat = -60.0 + lat_step_deg / 2.0
    while lat <= 60.0:
        lon = -180.0 + lon_step_deg / 2.0
        while lon < 180.0:
            dop = compute_dop(lat, lon, sat_positions, min_elevation_deg=10.0)
            if dop.num_visible >= 4 and dop.gdop < 100.0:
                # FIM = (H^T H), det(FIM) = 1/det((H^T H)^-1) = 1/GDOP^4 (approx)
                # More precisely, det(Q) = GDOP^2 product decomposition
                # Use GDOP directly: det(FIM) ~ 1/(GDOP^4) since Q = FIM^-1 is 4x4
                gdop_sq = dop.gdop * dop.gdop
                if gdop_sq > 0.0:
                    total_log_det += math.log2(1.0 / (gdop_sq * gdop_sq))
                    count += 1
            lon += lon_step_deg
        lat += lat_step_deg

    return total_log_det / max(1, count)


def compute_marginal_satellite_value(
    states: list,
    candidate: OrbitalState,
    epoch: datetime,
    duration_s: float = 5400.0,
    step_s: float = 60.0,
    min_elevation_deg: float = 10.0,
    lat_step_deg: float = 30.0,
    lon_step_deg: float = 30.0,
) -> MarginalSatelliteValue:
    """Compute information gain from adding one satellite.

    I_coverage = H(coverage_{N+1}) - H(coverage_N)
    I_positioning = avg[log2(det(FIM_{N+1}))] - avg[log2(det(FIM_N))]
    """
    # Coverage entropy before
    dist_before = _coverage_distribution(
        states, epoch, duration_s, step_s,
        min_elevation_deg, lat_step_deg, lon_step_deg,
    )
    h_before = _shannon_entropy(dist_before)

    # Coverage entropy after
    states_after = list(states) + [candidate]
    dist_after = _coverage_distribution(
        states_after, epoch, duration_s, step_s,
        min_elevation_deg, lat_step_deg, lon_step_deg,
    )
    h_after = _shannon_entropy(dist_after)

    coverage_gain = max(0.0, h_after - h_before)

    # Positioning information gain
    fisher_before = _fisher_log_det(states, epoch, lat_step_deg, lon_step_deg)
    fisher_after = _fisher_log_det(states_after, epoch, lat_step_deg, lon_step_deg)
    positioning_gain = max(0.0, fisher_after - fisher_before)

    # Combined (equal weighting)
    total = 0.5 * coverage_gain + 0.5 * positioning_gain

    return MarginalSatelliteValue(
        coverage_entropy_before=h_before,
        coverage_entropy_after=h_after,
        coverage_information_gain=coverage_gain,
        positioning_info_gain=positioning_gain,
        total_information_value=total,
    )


# ---------------------------------------------------------------------------
# P16: KL Divergence for Coverage Quality
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageKLDivergence:
    """KL divergence of coverage distribution from ideal uniform.

    D_KL(actual || uniform) = 0 means perfectly uniform coverage.
    D_KL > 0 means coverage deviates from uniform (higher = worse).
    coverage_efficiency = 1 - D_KL/H_max is a 0-1 metric (1 = perfect).
    """
    kl_from_uniform: float
    kl_from_demand: float
    coverage_efficiency: float
    worst_deficit_lat_deg: float
    worst_deficit_lon_deg: float


def compute_coverage_kl_divergence(
    coverage_grid: list,
    demand_weights: list = None,
) -> CoverageKLDivergence:
    """Compute KL divergence of coverage distribution from uniform.

    Measures how far the actual satellite visibility distribution is from
    the ideal uniform distribution (or a demand-weighted target distribution).

    D_KL(actual || uniform) = sum_k p(k) * log2(p(k) / q(k))
    where p(k) is the fraction of grid points with coverage count k
    and q(k) = 1/K for uniform.

    Args:
        coverage_grid: List of CoveragePoint objects (or any object with
            lat_deg, lon_deg, visible_count attributes).
        demand_weights: Optional list of demand weights per grid point.
            If provided, computes D_KL(actual || demand) as well.
            Must have same length as coverage_grid.

    Returns:
        CoverageKLDivergence with divergence metrics and worst-deficit location.
    """
    if not coverage_grid:
        return CoverageKLDivergence(
            kl_from_uniform=0.0,
            kl_from_demand=0.0,
            coverage_efficiency=1.0,
            worst_deficit_lat_deg=0.0,
            worst_deficit_lon_deg=0.0,
        )

    # Extract visible counts
    counts = [pt.visible_count for pt in coverage_grid]
    n_points = len(counts)

    # Build empirical distribution over coverage levels
    max_count = max(counts) if counts else 0
    if max_count == 0:
        # No coverage anywhere
        return CoverageKLDivergence(
            kl_from_uniform=0.0,
            kl_from_demand=0.0,
            coverage_efficiency=0.0,
            worst_deficit_lat_deg=coverage_grid[0].lat_deg,
            worst_deficit_lon_deg=coverage_grid[0].lon_deg,
        )

    # p_actual[k] = fraction of grid points with coverage count k
    num_levels = max_count + 1
    histogram = [0] * num_levels
    for c in counts:
        histogram[c] += 1

    p_actual = [h / n_points for h in histogram]

    # KL from uniform: D_KL(actual || uniform)
    # uniform: q[k] = 1/num_levels for all k
    q_uniform = 1.0 / num_levels
    kl_uniform = 0.0
    for k in range(num_levels):
        if p_actual[k] > 0:
            kl_uniform += p_actual[k] * math.log2(p_actual[k] / q_uniform)

    # Maximum entropy for this alphabet size
    h_max = math.log2(num_levels) if num_levels > 1 else 1.0

    # Coverage efficiency: 1 means perfect uniform, 0 means maximally skewed
    efficiency = max(0.0, 1.0 - kl_uniform / h_max) if h_max > 0 else 1.0

    # KL from demand-weighted distribution
    kl_demand = 0.0
    if demand_weights is not None and len(demand_weights) == n_points:
        # Build demand distribution over coverage levels
        demand_total = sum(demand_weights)
        if demand_total > 0:
            demand_histogram = [0.0] * num_levels
            for i, c in enumerate(counts):
                demand_histogram[c] += demand_weights[i]
            q_demand = [d / demand_total for d in demand_histogram]

            for k in range(num_levels):
                if p_actual[k] > 0 and q_demand[k] > 0:
                    kl_demand += p_actual[k] * math.log2(p_actual[k] / q_demand[k])
                elif p_actual[k] > 0 and q_demand[k] == 0:
                    kl_demand += p_actual[k] * 20.0  # Large penalty for uncovered demand

    # Find worst deficit: grid point with lowest coverage (or lowest
    # coverage relative to demand)
    worst_idx = 0
    worst_val = float('inf')
    for i, pt in enumerate(coverage_grid):
        val = pt.visible_count
        if demand_weights is not None and len(demand_weights) == n_points:
            # Deficit = demand - coverage (higher demand with lower coverage = worse)
            val = pt.visible_count - demand_weights[i]
        if val < worst_val:
            worst_val = val
            worst_idx = i

    return CoverageKLDivergence(
        kl_from_uniform=kl_uniform,
        kl_from_demand=kl_demand,
        coverage_efficiency=efficiency,
        worst_deficit_lat_deg=coverage_grid[worst_idx].lat_deg,
        worst_deficit_lon_deg=coverage_grid[worst_idx].lon_deg,
    )
