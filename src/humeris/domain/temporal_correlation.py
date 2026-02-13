# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Cross-spectral coherence engine for time series analysis.

Applies DFT to availability components, network capacity, and arbitrary
signal pairs. Detects common-cause oscillations and eclipse synchronization.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass

import numpy as np

from humeris.domain.linalg import naive_dft
from humeris.domain.statistical_analysis import MissionAvailabilityProfile
from humeris.domain.communication_analysis import NetworkCapacityTimeline


@dataclass(frozen=True)
class SignalCoherence:
    """Spectral coherence of a single time series."""
    total_time_energy: float
    total_freq_energy: float
    parseval_ratio: float
    dominant_frequency_hz: float
    dominant_power_fraction: float
    is_narrowband: bool


@dataclass(frozen=True)
class SpectralCrossCorrelation:
    """Cross-spectral coherence between two signals."""
    coherence_at_dominant: float
    phase_lag_rad: float
    dominant_frequency_hz: float
    mean_coherence: float


@dataclass(frozen=True)
class AvailabilitySpectralDecomposition:
    """Spectral decomposition of mission availability components."""
    fuel_power_coherence: float
    fuel_conjunction_coherence: float
    power_conjunction_coherence: float
    dominant_failure_frequency_hz: float
    common_cause_detected: bool


@dataclass(frozen=True)
class CapacitySpectrum:
    """Spectral analysis of network capacity timeline."""
    frequencies_hz: tuple
    power_density: tuple
    dominant_frequency_hz: float
    orbital_frequency_hz: float
    is_eclipse_synchronized: bool


def compute_signal_coherence(signal: list, sample_rate_hz: float) -> SignalCoherence:
    """Compute spectral coherence of a single time series.

    Uses naive DFT to decompose signal, checks Parseval's theorem,
    and identifies dominant frequency.
    """
    n = len(signal)
    if n == 0:
        return SignalCoherence(
            total_time_energy=0.0, total_freq_energy=0.0,
            parseval_ratio=1.0, dominant_frequency_hz=0.0,
            dominant_power_fraction=0.0, is_narrowband=False,
        )

    # Time-domain energy
    signal_arr = np.array(signal)
    time_energy = float(np.dot(signal_arr, signal_arr)) / n

    # Frequency-domain via DFT
    dft = naive_dft(signal, sample_rate_hz)

    # Frequency-domain energy (Parseval's: sum of |X[k]|^2 = sum of |x[j]|^2 / N)
    mags_arr = np.array(dft.magnitudes)
    freq_energy = float(np.dot(mags_arr, mags_arr))

    parseval_ratio = freq_energy / time_energy if time_energy > 1e-15 else 1.0

    # Find dominant frequency (skip DC component k=0, and use only first half)
    half_n = n // 2
    if half_n < 2:
        return SignalCoherence(
            total_time_energy=time_energy, total_freq_energy=freq_energy,
            parseval_ratio=parseval_ratio, dominant_frequency_hz=0.0,
            dominant_power_fraction=0.0, is_narrowband=False,
        )

    mags = list(dft.magnitudes[1:half_n])
    freqs = list(dft.frequencies_hz[1:half_n])

    if not mags:
        return SignalCoherence(
            total_time_energy=time_energy, total_freq_energy=freq_energy,
            parseval_ratio=parseval_ratio, dominant_frequency_hz=0.0,
            dominant_power_fraction=0.0, is_narrowband=False,
        )

    mags_np = np.array(mags)
    max_idx = int(np.argmax(mags_np))
    dominant_freq = freqs[max_idx]
    dominant_power = mags[max_idx] ** 2
    total_power = float(np.dot(mags_np, mags_np))
    power_fraction = dominant_power / total_power if total_power > 1e-15 else 0.0

    is_narrowband = power_fraction > 0.5

    return SignalCoherence(
        total_time_energy=time_energy,
        total_freq_energy=freq_energy,
        parseval_ratio=parseval_ratio,
        dominant_frequency_hz=dominant_freq,
        dominant_power_fraction=power_fraction,
        is_narrowband=is_narrowband,
    )


def compute_spectral_cross_correlation(
    signal_a: list, signal_b: list, sample_rate_hz: float,
) -> SpectralCrossCorrelation:
    """Compute cross-spectral coherence between two signals.

    Coherence at dominant frequency: |S_ab(f)|^2 / (S_aa(f) * S_bb(f))
    Phase lag: angle of cross-spectrum at dominant frequency.
    """
    n = min(len(signal_a), len(signal_b))
    if n == 0:
        return SpectralCrossCorrelation(
            coherence_at_dominant=0.0, phase_lag_rad=0.0,
            dominant_frequency_hz=0.0, mean_coherence=0.0,
        )

    a = signal_a[:n]
    b = signal_b[:n]

    dft_a = naive_dft(a, sample_rate_hz)
    dft_b = naive_dft(b, sample_rate_hz)

    half_n = n // 2
    if half_n < 2:
        return SpectralCrossCorrelation(
            coherence_at_dominant=0.0, phase_lag_rad=0.0,
            dominant_frequency_hz=0.0, mean_coherence=0.0,
        )

    # Compute cross-spectral density components
    # For each frequency bin, compute coherence
    coherences = []
    cross_phases = []
    powers_a = []

    for k in range(1, half_n):
        mag_a = dft_a.magnitudes[k]
        mag_b = dft_b.magnitudes[k]
        phase_a = dft_a.phases_rad[k]
        phase_b = dft_b.phases_rad[k]

        power_a = mag_a ** 2
        power_b = mag_b ** 2
        cross_mag = mag_a * mag_b
        cross_phase = phase_b - phase_a

        powers_a.append(power_a)

        if power_a > 1e-20 and power_b > 1e-20:
            coh = cross_mag ** 2 / (power_a * power_b)
            coherences.append(min(1.0, coh))
        else:
            coherences.append(0.0)
        cross_phases.append(cross_phase)

    if not powers_a:
        return SpectralCrossCorrelation(
            coherence_at_dominant=0.0, phase_lag_rad=0.0,
            dominant_frequency_hz=0.0, mean_coherence=0.0,
        )

    # Dominant frequency = where signal_a has max power
    dom_idx = int(np.argmax(powers_a))
    dom_freq = dft_a.frequencies_hz[dom_idx + 1]

    mean_coh = sum(coherences) / len(coherences) if coherences else 0.0

    return SpectralCrossCorrelation(
        coherence_at_dominant=coherences[dom_idx],
        phase_lag_rad=cross_phases[dom_idx],
        dominant_frequency_hz=dom_freq,
        mean_coherence=mean_coh,
    )


def compute_availability_spectral_decomposition(
    profile: MissionAvailabilityProfile,
) -> AvailabilitySpectralDecomposition:
    """Spectral decomposition of mission availability components.

    Computes pairwise cross-spectral coherence between fuel, power,
    and conjunction availability time series.
    """
    n = len(profile.times)
    if n < 4:
        return AvailabilitySpectralDecomposition(
            fuel_power_coherence=0.0,
            fuel_conjunction_coherence=0.0,
            power_conjunction_coherence=0.0,
            dominant_failure_frequency_hz=0.0,
            common_cause_detected=False,
        )

    # Compute sample rate from time series
    dt_s = (profile.times[-1] - profile.times[0]).total_seconds() / (n - 1)
    if dt_s <= 0:
        dt_s = 1.0
    sample_rate = 1.0 / dt_s

    fuel = list(profile.fuel_availability)
    power = list(profile.power_availability)
    conj = list(profile.conjunction_survival)

    fp = compute_spectral_cross_correlation(fuel, power, sample_rate)
    fc = compute_spectral_cross_correlation(fuel, conj, sample_rate)
    pc = compute_spectral_cross_correlation(power, conj, sample_rate)

    # Dominant failure frequency: from total availability
    total = list(profile.total_availability)
    total_coh = compute_signal_coherence(total, sample_rate)
    dom_freq = total_coh.dominant_frequency_hz

    # Common cause: high coherence between at least two components
    threshold = 0.5
    common_cause = (
        fp.coherence_at_dominant > threshold
        or fc.coherence_at_dominant > threshold
        or pc.coherence_at_dominant > threshold
    )

    return AvailabilitySpectralDecomposition(
        fuel_power_coherence=fp.coherence_at_dominant,
        fuel_conjunction_coherence=fc.coherence_at_dominant,
        power_conjunction_coherence=pc.coherence_at_dominant,
        dominant_failure_frequency_hz=dom_freq,
        common_cause_detected=common_cause,
    )


def compute_network_capacity_spectrum(
    timeline: NetworkCapacityTimeline,
    orbital_period_s: float,
) -> CapacitySpectrum:
    """Spectral analysis of network capacity timeline.

    Checks if capacity fluctuations are synchronized with orbital period
    (eclipse-driven).
    """
    n = len(timeline.snapshots)
    if n < 4:
        orbital_freq = 1.0 / orbital_period_s if orbital_period_s > 0 else 0.0
        return CapacitySpectrum(
            frequencies_hz=(), power_density=(),
            dominant_frequency_hz=0.0,
            orbital_frequency_hz=orbital_freq,
            is_eclipse_synchronized=False,
        )

    # Extract ISL count as signal
    signal = [float(s.active_isl_count) for s in timeline.snapshots]

    # Compute sample rate
    dt_s = (timeline.snapshots[-1].time - timeline.snapshots[0].time).total_seconds() / (n - 1)
    if dt_s <= 0:
        dt_s = 1.0
    sample_rate = 1.0 / dt_s

    dft = naive_dft(signal, sample_rate)

    half_n = n // 2
    freqs = list(dft.frequencies_hz[:half_n])
    mags_half = np.array(dft.magnitudes[:half_n])
    powers = list(mags_half ** 2)

    # Skip DC, find dominant
    if half_n > 1:
        non_dc_powers = powers[1:]
        non_dc_freqs = freqs[1:]
        if non_dc_powers:
            dom_idx = int(np.argmax(non_dc_powers))
            dom_freq = non_dc_freqs[dom_idx]
        else:
            dom_freq = 0.0
    else:
        dom_freq = 0.0

    orbital_freq = 1.0 / orbital_period_s if orbital_period_s > 0 else 0.0

    # Check eclipse synchronization: dominant frequency near orbital frequency
    freq_tolerance = orbital_freq * 0.3 if orbital_freq > 0 else 1e-10
    is_eclipse_sync = abs(dom_freq - orbital_freq) < freq_tolerance if orbital_freq > 0 else False

    return CapacitySpectrum(
        frequencies_hz=tuple(freqs),
        power_density=tuple(powers),
        dominant_frequency_hz=dom_freq,
        orbital_frequency_hz=orbital_freq,
        is_eclipse_synchronized=is_eclipse_sync,
    )
