# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Beta-angle thermal equilibrium analysis.

Computes spacecraft thermal equilibrium temperatures using the
Stefan-Boltzmann energy balance equation, driven by eclipse fraction
and orbital beta angle.

No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass

import numpy as np


_SOLAR_FLUX_1AU = 1361.0          # W/m² — solar constant at 1 AU
_EARTH_ALBEDO = 0.3               # Earth average albedo fraction
_EARTH_IR = 237.0                 # W/m² — Earth IR emission
_STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
_DANGER_BETA_THRESHOLD = 70.0    # degrees — thermal danger zone threshold


@dataclass(frozen=True)
class ThermalConfig:
    """Spacecraft thermal properties."""
    absorptivity: float          # alpha — solar absorptance (0-1)
    emissivity: float            # epsilon — IR emittance (0-1)
    solar_area_m2: float         # cross-section area exposed to Sun
    radiator_area_m2: float      # radiator area
    internal_power_w: float = 0.0  # internal dissipation


@dataclass(frozen=True)
class ThermalEquilibrium:
    """Thermal equilibrium result at a single condition."""
    temperature_k: float
    absorbed_power_w: float
    emitted_power_w: float
    is_sunlit: bool
    beta_deg: float


@dataclass(frozen=True)
class ThermalDangerZone:
    """Interval where thermal conditions are extreme."""
    start_beta_deg: float
    end_beta_deg: float
    peak_temperature_k: float
    reason: str


@dataclass(frozen=True)
class ThermalProfile:
    """Hot/cold case thermal bounds for a spacecraft."""
    hot_case: ThermalEquilibrium
    cold_case: ThermalEquilibrium
    danger_zones: tuple[ThermalDangerZone, ...]


def _validate_config(config: ThermalConfig) -> None:
    """Validate thermal config parameters."""
    if not 0.0 <= config.absorptivity <= 1.0:
        raise ValueError(f"absorptivity must be 0-1, got {config.absorptivity}")
    if not 0.0 < config.emissivity <= 1.0:
        raise ValueError(f"emissivity must be (0,1], got {config.emissivity}")
    if config.solar_area_m2 <= 0:
        raise ValueError(f"solar_area_m2 must be > 0, got {config.solar_area_m2}")
    if config.radiator_area_m2 <= 0:
        raise ValueError(f"radiator_area_m2 must be > 0, got {config.radiator_area_m2}")


def compute_thermal_equilibrium(
    config: ThermalConfig,
    solar_flux_w_m2: float = _SOLAR_FLUX_1AU,
    albedo_flux_w_m2: float = _EARTH_ALBEDO * _SOLAR_FLUX_1AU * 0.25,
    earth_ir_w_m2: float = _EARTH_IR,
    eclipse_fraction: float = 0.0,
    beta_deg: float = 0.0,
) -> ThermalEquilibrium:
    """Compute thermal equilibrium temperature.

    Energy balance: Q_absorbed = Q_emitted
    T = (Q_absorbed / (epsilon * sigma * A_rad))^0.25

    Args:
        config: Spacecraft thermal properties.
        solar_flux_w_m2: Direct solar flux (W/m²).
        albedo_flux_w_m2: Earth albedo flux on spacecraft (W/m²).
        earth_ir_w_m2: Earth IR flux on spacecraft (W/m²).
        eclipse_fraction: Fraction of orbit in eclipse (0-1).
        beta_deg: Beta angle (degrees) for reporting.

    Returns:
        ThermalEquilibrium with temperature and power values.
    """
    _validate_config(config)

    sunlit_fraction = 1.0 - eclipse_fraction

    q_solar = config.absorptivity * solar_flux_w_m2 * config.solar_area_m2 * sunlit_fraction
    q_albedo = config.absorptivity * albedo_flux_w_m2 * config.solar_area_m2 * sunlit_fraction
    q_earth_ir = config.emissivity * earth_ir_w_m2 * config.solar_area_m2
    q_internal = config.internal_power_w

    q_total = q_solar + q_albedo + q_earth_ir + q_internal

    denominator = config.emissivity * _STEFAN_BOLTZMANN * config.radiator_area_m2
    temperature = (q_total / denominator) ** 0.25

    q_emitted = config.emissivity * _STEFAN_BOLTZMANN * config.radiator_area_m2 * temperature ** 4

    return ThermalEquilibrium(
        temperature_k=temperature,
        absorbed_power_w=q_total,
        emitted_power_w=q_emitted,
        is_sunlit=eclipse_fraction < 1.0,
        beta_deg=beta_deg,
    )


def compute_thermal_extremes(
    config: ThermalConfig,
) -> ThermalProfile:
    """Compute hot and cold case thermal equilibrium.

    Hot case: beta = 90° (no eclipse, continuous sunlight).
    Cold case: maximum eclipse fraction (beta = 0°, ~35% eclipse).

    Args:
        config: Spacecraft thermal properties.

    Returns:
        ThermalProfile with hot/cold cases and danger zones.
    """
    _validate_config(config)

    # Hot case: no eclipse (beta ≈ 90°)
    hot = compute_thermal_equilibrium(
        config, eclipse_fraction=0.0, beta_deg=90.0,
    )

    # Cold case: maximum eclipse (~35% for typical LEO)
    cold = compute_thermal_equilibrium(
        config, eclipse_fraction=0.35, beta_deg=0.0,
    )

    # Danger zones: |beta| > threshold
    danger_zones = flag_thermal_danger_zones_from_range(config)

    return ThermalProfile(
        hot_case=hot,
        cold_case=cold,
        danger_zones=danger_zones,
    )


def flag_thermal_danger_zones(
    beta_history: list[tuple[float, float]],
) -> list[ThermalDangerZone]:
    """Scan beta angle history and flag intervals where |beta| > threshold.

    Args:
        beta_history: List of (time_s, beta_deg) tuples.

    Returns:
        List of ThermalDangerZone for continuous high-beta intervals.
    """
    zones: list[ThermalDangerZone] = []
    in_zone = False
    start_beta = 0.0
    peak_beta = 0.0

    for _, beta in beta_history:
        if abs(beta) > _DANGER_BETA_THRESHOLD:
            if not in_zone:
                start_beta = beta
                peak_beta = abs(beta)
                in_zone = True
            else:
                peak_beta = max(peak_beta, abs(beta))
        elif in_zone:
            zones.append(ThermalDangerZone(
                start_beta_deg=start_beta,
                end_beta_deg=beta,
                peak_temperature_k=0.0,  # computed separately if needed
                reason="continuous_sunlight" if peak_beta > 80 else "no_eclipse",
            ))
            in_zone = False

    if in_zone:
        zones.append(ThermalDangerZone(
            start_beta_deg=start_beta,
            end_beta_deg=beta_history[-1][1],
            peak_temperature_k=0.0,
            reason="continuous_sunlight" if peak_beta > 80 else "no_eclipse",
        ))

    return zones


def flag_thermal_danger_zones_from_range(
    config: ThermalConfig,
) -> tuple[ThermalDangerZone, ...]:
    """Flag danger zones based on high-beta thermal analysis.

    Scans beta from 0° to 90° and identifies where temperature exceeds
    the hot-case threshold.

    Returns:
        Tuple of ThermalDangerZone objects.
    """
    zones: list[ThermalDangerZone] = []
    in_zone = False
    start_beta = 0.0
    peak_temp = 0.0

    for beta_deg_i in range(0, 91):
        beta = float(beta_deg_i)
        # Eclipse fraction approximation: decreases with |beta|
        # At beta=0: ~35%, at beta=70: ~0%
        if beta < _DANGER_BETA_THRESHOLD:
            eclipse_frac = 0.35 * (1.0 - beta / _DANGER_BETA_THRESHOLD)
        else:
            eclipse_frac = 0.0

        eq = compute_thermal_equilibrium(config, eclipse_fraction=eclipse_frac, beta_deg=beta)

        if beta >= _DANGER_BETA_THRESHOLD:
            if not in_zone:
                start_beta = beta
                peak_temp = eq.temperature_k
                in_zone = True
            else:
                peak_temp = max(peak_temp, eq.temperature_k)
        elif in_zone:
            zones.append(ThermalDangerZone(
                start_beta_deg=start_beta,
                end_beta_deg=beta,
                peak_temperature_k=peak_temp,
                reason="continuous_sunlight",
            ))
            in_zone = False

    if in_zone:
        zones.append(ThermalDangerZone(
            start_beta_deg=start_beta,
            end_beta_deg=90.0,
            peak_temperature_k=peak_temp,
            reason="continuous_sunlight",
        ))

    return tuple(zones)
