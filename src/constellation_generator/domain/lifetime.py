# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""
Orbit lifetime and decay profile computation.

Numerical integration (forward Euler) of semi-major axis decay
until re-entry altitude, using atmospheric drag model.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.atmosphere import (
    AtmosphereModel,
    DragConfig,
    semi_major_axis_decay_rate,
)


@dataclass(frozen=True)
class DecayPoint:
    """Single point in an orbit decay profile."""
    time: datetime
    altitude_km: float
    semi_major_axis_m: float


@dataclass(frozen=True)
class OrbitLifetimeResult:
    """Result of orbit lifetime computation."""
    initial_altitude_km: float
    re_entry_altitude_km: float
    lifetime_days: float
    re_entry_time: datetime | None
    decay_profile: tuple[DecayPoint, ...]
    converged: bool


def compute_orbit_lifetime(
    semi_major_axis_m: float,
    eccentricity: float,
    drag_config: DragConfig,
    epoch: datetime,
    re_entry_altitude_km: float = 100.0,
    step_days: float = 1.0,
    max_years: float = 50.0,
    atmosphere_model: AtmosphereModel | None = None,
) -> OrbitLifetimeResult:
    """Compute orbit lifetime by integrating semi-major axis decay.

    Forward Euler integration: a += da/dt * dt_step until altitude
    drops to re-entry or max_years is exceeded.

    Args:
        semi_major_axis_m: Initial semi-major axis (m).
        eccentricity: Orbital eccentricity.
        drag_config: Satellite drag configuration.
        epoch: Start time.
        re_entry_altitude_km: Re-entry altitude threshold (km).
        step_days: Integration step size (days).
        max_years: Maximum simulation duration (years).

    Returns:
        OrbitLifetimeResult with decay profile and convergence status.

    Raises:
        ValueError: If already below re-entry altitude or step <= 0.
    """
    if step_days <= 0:
        raise ValueError(f"step_days must be positive, got {step_days}")

    initial_alt_km = (semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
    if initial_alt_km <= re_entry_altitude_km:
        raise ValueError(
            f"Initial altitude {initial_alt_km:.1f} km already at or below "
            f"re-entry altitude {re_entry_altitude_km} km"
        )

    dt_seconds = step_days * 86400.0
    max_seconds = max_years * 365.25 * 86400.0

    a = semi_major_axis_m
    t_elapsed = 0.0
    profile: list[DecayPoint] = []

    # Record initial point
    profile.append(DecayPoint(
        time=epoch,
        altitude_km=initial_alt_km,
        semi_major_axis_m=a,
    ))

    converged = False
    re_entry_time = None

    decay_kwargs: dict = {}
    if atmosphere_model is not None:
        decay_kwargs["model"] = atmosphere_model

    while t_elapsed < max_seconds:
        da_dt = semi_major_axis_decay_rate(a, eccentricity, drag_config, **decay_kwargs)
        a += da_dt * dt_seconds
        t_elapsed += dt_seconds

        alt_km = (a - OrbitalConstants.R_EARTH) / 1000.0
        current_time = epoch + timedelta(seconds=t_elapsed)

        profile.append(DecayPoint(
            time=current_time,
            altitude_km=alt_km,
            semi_major_axis_m=a,
        ))

        if alt_km <= re_entry_altitude_km:
            converged = True
            re_entry_time = current_time
            break

    lifetime_days = t_elapsed / 86400.0

    return OrbitLifetimeResult(
        initial_altitude_km=initial_alt_km,
        re_entry_altitude_km=re_entry_altitude_km,
        lifetime_days=lifetime_days,
        re_entry_time=re_entry_time,
        decay_profile=tuple(profile),
        converged=converged,
    )


def compute_altitude_at_time(
    semi_major_axis_m: float,
    eccentricity: float,
    drag_config: DragConfig,
    epoch: datetime,
    target_time: datetime,
    step_days: float = 1.0,
    atmosphere_model: AtmosphereModel | None = None,
) -> float:
    """Compute altitude at a specific future time by integrating decay.

    Args:
        semi_major_axis_m: Initial semi-major axis (m).
        eccentricity: Orbital eccentricity.
        drag_config: Satellite drag configuration.
        epoch: Start time.
        target_time: Time at which to compute altitude.
        step_days: Integration step size (days).

    Returns:
        Altitude in km at target_time.

    Raises:
        ValueError: If target_time < epoch.
    """
    if target_time < epoch:
        raise ValueError("target_time must be >= epoch")

    dt_seconds = step_days * 86400.0
    target_elapsed = (target_time - epoch).total_seconds()

    a = semi_major_axis_m
    t_elapsed = 0.0

    alt_decay_kwargs: dict = {}
    if atmosphere_model is not None:
        alt_decay_kwargs["model"] = atmosphere_model

    while t_elapsed < target_elapsed:
        remaining = target_elapsed - t_elapsed
        step = min(dt_seconds, remaining)
        da_dt = semi_major_axis_decay_rate(a, eccentricity, drag_config, **alt_decay_kwargs)
        a += da_dt * step
        t_elapsed += step
        if step < dt_seconds:
            break

    return (a - OrbitalConstants.R_EARTH) / 1000.0
