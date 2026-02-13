# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Link budget computation for inter-satellite and ground links.

Free-space path loss, received power, noise floor, SNR, link margin,
and Shannon capacity. Integration with ISL topology for constellation-wide
link budget assessment.

No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.inter_satellite_links import (
    ISLLink,
    compute_isl_link,
)

_C = 299792458.0           # speed of light (m/s)
_K_BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K)


@dataclass(frozen=True)
class LinkConfig:
    """RF link configuration."""
    frequency_hz: float
    transmit_power_w: float
    tx_antenna_gain_dbi: float
    rx_antenna_gain_dbi: float
    system_noise_temp_k: float
    bandwidth_hz: float
    additional_losses_db: float = 0.0
    required_snr_db: float = 10.0


@dataclass(frozen=True)
class LinkBudgetResult:
    """Complete link budget result."""
    fspl_db: float
    received_power_dbw: float
    noise_floor_dbw: float
    snr_db: float
    link_margin_db: float
    max_data_rate_bps: float


def free_space_path_loss_db(distance_m: float, frequency_hz: float) -> float:
    """Free-space path loss in dB.

    FSPL(dB) = 20·log10(4π·d·f/c)

    Args:
        distance_m: Link distance (m).
        frequency_hz: Carrier frequency (Hz).

    Returns:
        FSPL in dB.
    """
    return 20.0 * math.log10(4.0 * math.pi * distance_m * frequency_hz / _C)


def compute_link_budget(config: LinkConfig, distance_m: float) -> LinkBudgetResult:
    """Compute full link budget at given distance.

    Args:
        config: RF link configuration.
        distance_m: Link distance (m).

    Returns:
        LinkBudgetResult with all budget components.
    """
    fspl = free_space_path_loss_db(distance_m, config.frequency_hz)

    p_tx_dbw = 10.0 * math.log10(config.transmit_power_w)
    p_rx_dbw = (p_tx_dbw + config.tx_antenna_gain_dbi + config.rx_antenna_gain_dbi
                - fspl - config.additional_losses_db)

    noise_floor = 10.0 * math.log10(
        _K_BOLTZMANN * config.system_noise_temp_k * config.bandwidth_hz
    )

    snr = p_rx_dbw - noise_floor
    margin = snr - config.required_snr_db

    # Shannon capacity: C = B·log2(1 + SNR_linear)
    snr_linear = 10.0 ** (snr / 10.0)
    capacity = config.bandwidth_hz * math.log2(1.0 + snr_linear)

    return LinkBudgetResult(
        fspl_db=fspl,
        received_power_dbw=p_rx_dbw,
        noise_floor_dbw=noise_floor,
        snr_db=snr,
        link_margin_db=margin,
        max_data_rate_bps=capacity,
    )


def compute_isl_link_budgets(
    config: LinkConfig,
    states: list[OrbitalState],
    time,
    max_range_km: float = 5000.0,
) -> list[tuple[ISLLink, LinkBudgetResult]]:
    """Link budget for each active (unblocked, in-range) ISL.

    Args:
        config: RF link configuration.
        states: List of OrbitalState objects.
        time: Target datetime for evaluation.
        max_range_km: Maximum ISL range in km.

    Returns:
        List of (ISLLink, LinkBudgetResult) for active links.
    """
    n = len(states)
    if n < 2:
        return []

    max_range_m = max_range_km * 1000.0

    positions = []
    for state in states:
        pos, _vel = propagate_to(state, time)
        positions.append((pos[0], pos[1], pos[2]))

    results: list[tuple[ISLLink, LinkBudgetResult]] = []
    for i in range(n):
        for j in range(i + 1, n):
            link = compute_isl_link(positions[i], positions[j], i, j)
            if not link.is_blocked and link.distance_m <= max_range_m:
                budget = compute_link_budget(config, link.distance_m)
                results.append((link, budget))

    return results
