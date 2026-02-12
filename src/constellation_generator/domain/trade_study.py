# Copyright (c) 2026 Jeroen. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""
Parametric Walker constellation trade studies.

Sweeps constellation parameters (altitude, inclination, planes, sats/plane,
phase factor) and evaluates coverage FoMs for each configuration. Supports
Pareto frontier extraction for cost-vs-performance analysis.

No external dependencies — only stdlib dataclasses/datetime.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.constellation import (
    ShellConfig,
    generate_walker_shell,
)
from constellation_generator.domain.propagation import (
    OrbitalState,
    derive_orbital_state,
)
from constellation_generator.domain.revisit import (
    CoverageResult,
    compute_revisit,
)


@dataclass(frozen=True)
class WalkerConfig:
    """Walker constellation configuration for parametric trade study."""
    altitude_km: float
    inclination_deg: float
    num_planes: int
    sats_per_plane: int
    phase_factor: int


@dataclass(frozen=True)
class TradePoint:
    """Single evaluated configuration in a trade study."""
    config: WalkerConfig
    total_satellites: int
    coverage: CoverageResult


@dataclass(frozen=True)
class TradeStudyResult:
    """Complete trade study results."""
    points: tuple[TradePoint, ...]
    analysis_duration_s: float
    min_elevation_deg: float


def generate_walker_configs(
    altitude_range: tuple[float, ...],
    inclination_range: tuple[float, ...],
    planes_range: tuple[int, ...],
    sats_per_plane_range: tuple[int, ...],
    phase_factor_range: tuple[int, ...] | None = None,
) -> list[WalkerConfig]:
    """Generate Cartesian product of parameter ranges as WalkerConfigs.

    If phase_factor_range is None, uses 0 for all configs.
    Validates: planes >= 1, sats_per_plane >= 1, altitude > 0.
    """
    for alt in altitude_range:
        if alt <= 0:
            raise ValueError(f"altitude must be > 0, got {alt}")
    for p in planes_range:
        if p < 1:
            raise ValueError(f"num_planes must be >= 1, got {p}")
    for s in sats_per_plane_range:
        if s < 1:
            raise ValueError(f"sats_per_plane must be >= 1, got {s}")

    if phase_factor_range is None:
        phase_factor_range = (0,)

    configs: list[WalkerConfig] = []
    for alt in altitude_range:
        for inc in inclination_range:
            for planes in planes_range:
                for spp in sats_per_plane_range:
                    for pf in phase_factor_range:
                        configs.append(WalkerConfig(
                            altitude_km=alt,
                            inclination_deg=inc,
                            num_planes=planes,
                            sats_per_plane=spp,
                            phase_factor=pf,
                        ))
    return configs


def run_walker_trade_study(
    configs: list[WalkerConfig],
    analysis_epoch: datetime,
    analysis_duration: timedelta,
    analysis_step: timedelta,
    min_elevation_deg: float = 10.0,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
) -> TradeStudyResult:
    """Run coverage analysis for each Walker configuration.

    For each config:
    1. Generate Walker shell via generate_walker_shell
    2. Derive orbital states for all satellites
    3. Run compute_revisit for the full analysis window
    4. Collect results as TradePoint
    """
    points: list[TradePoint] = []

    for config in configs:
        shell = ShellConfig(
            altitude_km=config.altitude_km,
            inclination_deg=config.inclination_deg,
            num_planes=config.num_planes,
            sats_per_plane=config.sats_per_plane,
            phase_factor=config.phase_factor,
            raan_offset_deg=0.0,
            shell_name='TradeStudy',
        )
        sats = generate_walker_shell(shell)
        states = [derive_orbital_state(s, analysis_epoch) for s in sats]

        coverage = compute_revisit(
            states, analysis_epoch, analysis_duration, analysis_step,
            min_elevation_deg=min_elevation_deg,
            lat_step_deg=lat_step_deg, lon_step_deg=lon_step_deg,
            lat_range=lat_range, lon_range=lon_range,
        )

        points.append(TradePoint(
            config=config,
            total_satellites=config.num_planes * config.sats_per_plane,
            coverage=coverage,
        ))

    return TradeStudyResult(
        points=tuple(points),
        analysis_duration_s=analysis_duration.total_seconds(),
        min_elevation_deg=min_elevation_deg,
    )


def pareto_front_indices(
    costs: list[float],
    metrics: list[float],
) -> list[int]:
    """Return indices of Pareto-optimal points (minimizing both cost and metric).

    A point (c_i, m_i) is non-dominated if no other point (c_j, m_j) has
    c_j <= c_i AND m_j <= m_i (with at least one strict inequality).

    Returns sorted list of indices into the input arrays.
    """
    n = len(costs)
    if n == 0:
        return []

    front: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if costs[j] <= costs[i] and metrics[j] <= metrics[i]:
                if costs[j] < costs[i] or metrics[j] < metrics[i]:
                    dominated = True
                    break
        if not dominated:
            front.append(i)

    return sorted(front)
