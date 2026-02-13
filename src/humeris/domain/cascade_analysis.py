# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Failure cascade prediction.

Composes DFT of Fiedler value time series with hazard rate peakiness
from orbit lifetime survival curves to predict cascading failures.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from humeris.domain.propagation import OrbitalState
from humeris.domain.link_budget import LinkConfig
from humeris.domain.atmosphere import DragConfig
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.graph_analysis import compute_fragmentation_timeline
from humeris.domain.linalg import naive_dft
from humeris.domain.lifetime import compute_orbit_lifetime
from humeris.domain.statistical_analysis import compute_lifetime_survival_curve


@dataclass(frozen=True)
class CascadeIndicator:
    """Failure cascade risk indicator."""
    spectral_power_at_orbital: float
    hazard_peakiness: float
    cascade_indicator: float
    fiedler_dominant_freq_hz: float
    orbital_frequency_hz: float
    is_cascade_risk: bool


def compute_cascade_indicator(
    states: list,
    link_config: LinkConfig,
    epoch: datetime,
    drag_config: DragConfig,
    fragmentation_duration_s: float = 5400.0,
    fragmentation_step_s: float = 300.0,
    max_range_km: float = 5000.0,
    eclipse_power_fraction: float = 0.5,
) -> CascadeIndicator:
    """Compute failure cascade indicator.

    CI = spectral_power(fiedler, f_orbital) * max(hazard) / mean(hazard)

    High CI means eclipse-driven connectivity oscillations coincide
    with high hazard-rate regimes.
    """
    if not states:
        return CascadeIndicator(
            spectral_power_at_orbital=0.0, hazard_peakiness=1.0,
            cascade_indicator=0.0, fiedler_dominant_freq_hz=0.0,
            orbital_frequency_hz=0.0, is_cascade_risk=False,
        )

    # Fragmentation timeline → DFT
    timeline = compute_fragmentation_timeline(
        states, link_config, epoch,
        fragmentation_duration_s, fragmentation_step_s,
        max_range_km=max_range_km,
        eclipse_power_fraction=eclipse_power_fraction,
    )

    fiedler_signal = [e.fiedler_value for e in timeline.events]
    sample_rate = 1.0 / fragmentation_step_s
    dft_result = naive_dft(fiedler_signal, sample_rate)

    # Orbital frequency
    orbital_freq = states[0].mean_motion_rad_s / (2.0 * np.pi)

    # Find spectral power at/near orbital frequency
    spectral_at_orbital = 0.0
    half_n = max(1, len(fiedler_signal) // 2)
    for k in range(1, half_n):
        freq = dft_result.frequencies_hz[k]
        if abs(freq - orbital_freq) < orbital_freq * 0.2:
            mag = dft_result.magnitudes[k]
            spectral_at_orbital = max(spectral_at_orbital, mag * mag)

    # Find dominant frequency
    if half_n > 1:
        peak_idx = max(range(1, half_n),
                       key=lambda k: dft_result.magnitudes[k])
        dom_freq = dft_result.frequencies_hz[peak_idx]
    else:
        dom_freq = 0.0

    # Hazard rate peakiness from survival curve
    ref_state = states[0]
    a_m = ref_state.semi_major_axis_m

    try:
        lifetime = compute_orbit_lifetime(
            a_m, 0.0, drag_config, epoch,
            step_days=10.0, max_years=25.0,
        )
        survival = compute_lifetime_survival_curve(lifetime)
        hazard_rates = [h for h in survival.hazard_rate_per_day if h > 0]
    except (ValueError, ZeroDivisionError):
        hazard_rates = []

    if hazard_rates:
        mean_hazard = sum(hazard_rates) / len(hazard_rates)
        max_hazard = max(hazard_rates)
        if mean_hazard > 1e-20:
            peakiness = max_hazard / mean_hazard
        else:
            peakiness = 1.0
    else:
        peakiness = 1.0

    ci = spectral_at_orbital * peakiness
    is_risk = ci > 0.01

    return CascadeIndicator(
        spectral_power_at_orbital=spectral_at_orbital,
        hazard_peakiness=peakiness,
        cascade_indicator=ci,
        fiedler_dominant_freq_hz=dom_freq,
        orbital_frequency_hz=orbital_freq,
        is_cascade_risk=is_risk,
    )
