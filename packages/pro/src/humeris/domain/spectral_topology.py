# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Spectral analysis of ISL topology dynamics.

DFT of Fiedler value time series, eclipse spectral invariant,
and proximity spectral collision prediction.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.link_budget import LinkConfig
from humeris.domain.graph_analysis import (
    compute_fragmentation_timeline,
    compute_topology_resilience,
)
from humeris.domain.linalg import naive_dft


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

    # Compute cross-spectral coherence using Welch's method (segment averaging)
    # Split signals into overlapping segments for meaningful coherence estimates
    seg_count = max(1, min(4, n // 8))
    seg_len = (2 * n) // (seg_count + 1) if seg_count > 1 else n
    seg_len = max(4, seg_len)
    hop = seg_len // 2 if seg_count > 1 else seg_len

    # Collect segments
    segs_f = []
    segs_e = []
    start = 0
    while start + seg_len <= n:
        segs_f.append(fiedler_signal[start:start + seg_len])
        segs_e.append(eclipse_signal[start:start + seg_len])
        start += hop
    if not segs_f:
        segs_f.append(fiedler_signal)
        segs_e.append(eclipse_signal)
        seg_len = n

    half_seg = seg_len // 2
    num_coh_bins = max(1, half_seg)

    # Accumulate averaged spectra
    psd_ff = np.zeros(num_coh_bins)
    psd_ee = np.zeros(num_coh_bins)
    csd_re = np.zeros(num_coh_bins)
    csd_im = np.zeros(num_coh_bins)

    for sf, se in zip(segs_f, segs_e):
        dft_sf = naive_dft(sf, sample_rate)
        dft_se = naive_dft(se, sample_rate)
        for k in range(num_coh_bins):
            mf = dft_sf.magnitudes[k]
            me = dft_se.magnitudes[k]
            psd_ff[k] += mf * mf
            psd_ee[k] += me * me
            pd = dft_sf.phases_rad[k] - dft_se.phases_rad[k]
            csd_re[k] += mf * me * math.cos(pd)
            csd_im[k] += mf * me * math.sin(pd)

    n_segs = len(segs_f)
    psd_ff /= n_segs
    psd_ee /= n_segs
    csd_re /= n_segs
    csd_im /= n_segs

    coherence = []
    for k in range(n):
        if k < num_coh_bins:
            denom = float(psd_ff[k] * psd_ee[k])
            if denom > 1e-30:
                cross_mag_sq = float(csd_re[k] ** 2 + csd_im[k] ** 2)
                coh = cross_mag_sq / denom
            else:
                coh = 0.0
            coherence.append(min(1.0, coh))
        else:
            coherence.append(0.0)

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
        from humeris.domain.eclipse import eclipse_fraction as ef
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
        dp = np.array(pos_a) - np.array(pos_b)
        dist = float(np.linalg.norm(dp))
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
