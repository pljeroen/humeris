# Copyright (c) 2026 Jeroen. All rights reserved.
"""Spectral analysis of ISL topology dynamics.

DFT of Fiedler value time series, eclipse spectral invariant,
and proximity spectral collision prediction.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.propagation import OrbitalState, propagate_to
from constellation_generator.domain.link_budget import LinkConfig
from constellation_generator.domain.graph_analysis import (
    compute_fragmentation_timeline,
    compute_topology_resilience,
)
from constellation_generator.domain.linalg import naive_dft


@dataclass(frozen=True)
class FragmentationSpectralAnalysis:
    """Frequency-domain analysis of ISL topology fragmentation."""
    fiedler_frequencies_hz: tuple
    fiedler_magnitudes: tuple
    eclipse_frequencies_hz: tuple
    eclipse_magnitudes: tuple
    coherence_squared: tuple
    fiedler_dominant_freq_hz: float
    eclipse_dominant_freq_hz: float
    orbital_frequency_hz: float
    fiedler_resonance_ratio: float


@dataclass(frozen=True)
class EclipseSpectralInvariant:
    """Ratio of eclipse-degraded to nominal mean Fiedler value."""
    nominal_mean_fiedler: float
    eclipse_mean_fiedler: float
    eta_eclipse: float
    eclipse_fraction_mean: float


@dataclass(frozen=True)
class ProximitySpectralResult:
    """Frequency-domain analysis of inter-satellite distance."""
    pair_indices: tuple
    distance_frequencies_hz: tuple
    distance_magnitudes: tuple
    dominant_frequency_hz: float
    orbital_frequency_hz: float
    resonance_ratio: float


def compute_fragmentation_spectrum(
    states: list,
    link_config: LinkConfig,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    max_range_km: float = 5000.0,
    eclipse_power_fraction: float = 0.5,
) -> FragmentationSpectralAnalysis:
    """DFT of Fiedler value and eclipse count time series.

    Cross-spectral coherence reveals whether eclipse drives connectivity loss.
    """
    timeline = compute_fragmentation_timeline(
        states, link_config, epoch, duration_s, step_s,
        max_range_km=max_range_km,
        eclipse_power_fraction=eclipse_power_fraction,
    )

    fiedler_signal = [e.fiedler_value for e in timeline.events]
    eclipse_signal = [float(e.eclipsed_count) for e in timeline.events]

    sample_rate = 1.0 / step_s
    dft_fiedler = naive_dft(fiedler_signal, sample_rate)
    dft_eclipse = naive_dft(eclipse_signal, sample_rate)

    n = len(fiedler_signal)
    half_n = max(1, n // 2)

    # Compute cross-spectral coherence
    coherence = []
    for k in range(n):
        mag_f = dft_fiedler.magnitudes[k]
        mag_e = dft_eclipse.magnitudes[k]
        psd_f = mag_f * mag_f
        psd_e = mag_e * mag_e

        # Cross-spectral magnitude from magnitudes and phases
        phase_diff = dft_fiedler.phases_rad[k] - dft_eclipse.phases_rad[k]
        cross_re = mag_f * mag_e * math.cos(phase_diff)
        cross_im = mag_f * mag_e * math.sin(phase_diff)
        cross_mag_sq = cross_re * cross_re + cross_im * cross_im

        denom = psd_f * psd_e
        if denom > 1e-30:
            coh = cross_mag_sq / denom
        else:
            coh = 0.0
        coherence.append(min(1.0, coh))

    # Find dominant frequencies (excluding DC, first half only)
    if half_n > 1:
        fiedler_peak = max(range(1, half_n),
                           key=lambda k: dft_fiedler.magnitudes[k])
        eclipse_peak = max(range(1, half_n),
                           key=lambda k: dft_eclipse.magnitudes[k])
        fiedler_dom = dft_fiedler.frequencies_hz[fiedler_peak]
        eclipse_dom = dft_eclipse.frequencies_hz[eclipse_peak]
    else:
        fiedler_dom = 0.0
        eclipse_dom = 0.0

    orbital_freq = 0.0
    if states:
        orbital_freq = states[0].mean_motion_rad_s / (2.0 * math.pi)

    resonance = fiedler_dom / orbital_freq if orbital_freq > 0.0 else 0.0

    return FragmentationSpectralAnalysis(
        fiedler_frequencies_hz=dft_fiedler.frequencies_hz,
        fiedler_magnitudes=dft_fiedler.magnitudes,
        eclipse_frequencies_hz=dft_eclipse.frequencies_hz,
        eclipse_magnitudes=dft_eclipse.magnitudes,
        coherence_squared=tuple(coherence),
        fiedler_dominant_freq_hz=fiedler_dom,
        eclipse_dominant_freq_hz=eclipse_dom,
        orbital_frequency_hz=orbital_freq,
        fiedler_resonance_ratio=resonance,
    )


def compute_eclipse_spectral_invariant(
    states: list,
    link_config: LinkConfig,
    epoch: datetime,
    orbital_period_s: float,
    num_samples: int,
    eclipse_power_fraction: float = 0.5,
    max_range_km: float = 5000.0,
) -> EclipseSpectralInvariant:
    """Compute orbit-averaged ratio of eclipse-degraded to nominal Fiedler value.

    eta_eclipse = mean(lambda_2_eclipse) / mean(lambda_2_nominal)
    """
    dt = orbital_period_s / max(1, num_samples)
    eclipse_fiedler_sum = 0.0
    nominal_fiedler_sum = 0.0
    eclipse_frac_sum = 0.0
    count = 0

    for i in range(num_samples):
        t = epoch + timedelta(seconds=i * dt)

        res_eclipse = compute_topology_resilience(
            states, t, link_config,
            max_range_km=max_range_km,
            eclipse_power_fraction=eclipse_power_fraction,
        )
        res_nominal = compute_topology_resilience(
            states, t, link_config,
            max_range_km=max_range_km,
            eclipse_power_fraction=1.0,
        )

        eclipse_fiedler_sum += res_eclipse.fiedler_value
        nominal_fiedler_sum += res_nominal.fiedler_value
        count += 1

        # Estimate eclipse fraction from eclipsed count ratio
        from constellation_generator.domain.eclipse import eclipse_fraction as ef
        if states:
            eclipse_frac_sum += ef(states[0], t)

    if count == 0:
        return EclipseSpectralInvariant(
            nominal_mean_fiedler=0.0, eclipse_mean_fiedler=0.0,
            eta_eclipse=1.0, eclipse_fraction_mean=0.0,
        )

    nominal_mean = nominal_fiedler_sum / count
    eclipse_mean = eclipse_fiedler_sum / count
    eclipse_frac_mean = eclipse_frac_sum / count

    if nominal_mean > 1e-15:
        eta = eclipse_mean / nominal_mean
    else:
        eta = 1.0

    return EclipseSpectralInvariant(
        nominal_mean_fiedler=nominal_mean,
        eclipse_mean_fiedler=eclipse_mean,
        eta_eclipse=min(1.0, max(0.0, eta)),
        eclipse_fraction_mean=eclipse_frac_mean,
    )


def compute_proximity_spectrum(
    states: list,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    pair_indices: tuple,
) -> ProximitySpectralResult:
    """DFT of inter-satellite distance time series for a specific pair.

    Spectral peaks at harmonics of orbital frequency indicate periodic
    close approaches and potential collision risk windows.
    """
    idx_a, idx_b = pair_indices
    num_steps = int(duration_s / step_s) + 1
    distances = []

    for i in range(num_steps):
        t = epoch + timedelta(seconds=i * step_s)
        pos_a, _ = propagate_to(states[idx_a], t)
        pos_b, _ = propagate_to(states[idx_b], t)
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        dz = pos_a[2] - pos_b[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        distances.append(dist)

    sample_rate = 1.0 / step_s
    dft_result = naive_dft(distances, sample_rate)

    n = len(distances)
    half_n = max(1, n // 2)

    if half_n > 1:
        peak_idx = max(range(1, half_n),
                       key=lambda k: dft_result.magnitudes[k])
        dominant_freq = dft_result.frequencies_hz[peak_idx]
    else:
        dominant_freq = 0.0

    orbital_freq = 0.0
    if states:
        orbital_freq = states[0].mean_motion_rad_s / (2.0 * math.pi)

    resonance = dominant_freq / orbital_freq if orbital_freq > 0.0 else 0.0

    return ProximitySpectralResult(
        pair_indices=pair_indices,
        distance_frequencies_hz=dft_result.frequencies_hz,
        distance_magnitudes=dft_result.magnitudes,
        dominant_frequency_hz=dominant_freq,
        orbital_frequency_hz=orbital_freq,
        resonance_ratio=resonance,
    )
