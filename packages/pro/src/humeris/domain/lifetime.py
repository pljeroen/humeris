# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Orbit lifetime and decay profile computation.

Numerical integration (forward Euler) of semi-major axis decay
until re-entry altitude, using atmospheric drag model.

"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.atmosphere import (
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
    density_func: Optional[Callable[[float, datetime], float]] = None,
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
        current_time = epoch + timedelta(seconds=t_elapsed + dt_seconds)

        if density_func is not None:
            # Use NRLMSISE-00 or custom density function
            alt_km_now = (a - OrbitalConstants.R_EARTH) / 1000.0
            rho = density_func(alt_km_now, current_time)
            v = float(np.sqrt(OrbitalConstants.MU_EARTH / a))
            da_dt = -rho * v * drag_config.ballistic_coefficient * a
        else:
            da_dt = semi_major_axis_decay_rate(a, eccentricity, drag_config, **decay_kwargs)

        a += da_dt * dt_seconds
        t_elapsed += dt_seconds

        alt_km = (a - OrbitalConstants.R_EARTH) / 1000.0

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
    density_func: Optional[Callable[[float, datetime], float]] = None,
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
        current_time = epoch + timedelta(seconds=t_elapsed + step)

        if density_func is not None:
            alt_km_now = (a - OrbitalConstants.R_EARTH) / 1000.0
            rho = density_func(alt_km_now, current_time)
            v = float(np.sqrt(OrbitalConstants.MU_EARTH / a))
            da_dt = -rho * v * drag_config.ballistic_coefficient * a
        else:
            da_dt = semi_major_axis_decay_rate(a, eccentricity, drag_config, **alt_decay_kwargs)

        a += da_dt * step
        t_elapsed += step
        if step < dt_seconds:
            break

    return (a - OrbitalConstants.R_EARTH) / 1000.0


# ── Ornstein-Uhlenbeck Process for Stochastic Lifetime ──────────────
#
# Models atmospheric density as a mean-reverting stochastic process
# (OU process) and propagates orbit decay with Monte Carlo sampling.
# Returns lifetime distribution instead of point estimate.
# See SESSION_MINING_R1_CREATIVE.md P20.


@dataclass(frozen=True)
class StochasticLifetime:
    """Stochastic lifetime result with uncertainty quantification.

    Attributes:
        mean_lifetime_days: Mean lifetime across Monte Carlo samples.
        std_lifetime_days: Standard deviation of lifetime.
        percentile_5_days: 5th percentile (90% chance lifetime exceeds this).
        percentile_50_days: Median lifetime.
        percentile_95_days: 95th percentile (90% chance lifetime is less).
        deterministic_lifetime_days: Point estimate from deterministic model.
        uncertainty_ratio: std / mean (coefficient of variation).
        num_samples: Number of Monte Carlo samples used.
        num_converged: Number of samples that reached re-entry.
    """
    mean_lifetime_days: float
    std_lifetime_days: float
    percentile_5_days: float
    percentile_50_days: float
    percentile_95_days: float
    deterministic_lifetime_days: float
    uncertainty_ratio: float
    num_samples: int
    num_converged: int


def compute_stochastic_lifetime(
    semi_major_axis_m: float,
    eccentricity: float,
    drag_config: DragConfig,
    epoch: datetime,
    density_variability_fraction: float = 0.2,
    mean_reversion_days: float = 27.0,
    num_samples: int = 100,
    re_entry_altitude_km: float = 100.0,
    step_days: float = 1.0,
    max_years: float = 50.0,
    rng_seed: int | None = None,
    density_func: Optional[Callable[[float, datetime], float]] = None,
) -> StochasticLifetime:
    """Compute stochastic lifetime via Ornstein-Uhlenbeck density model.

    The atmospheric density rho(t) is modelled as a mean-reverting
    stochastic process (Ornstein-Uhlenbeck):

        d(rho) = theta * (rho_bar - rho) * dt + sigma_rho * dW

    where:
        theta = 1 / mean_reversion_days (reversion rate, ~solar rotation)
        rho_bar = mean density from the deterministic atmosphere model
        sigma_rho = density_variability_fraction * rho_bar

    We implement this as a multiplicative density factor f(t) around 1.0:
        df = theta * (1.0 - f) * dt + sigma_f * dW
    where sigma_f = density_variability_fraction * sqrt(2 * theta).

    Each Monte Carlo sample generates a different OU trajectory for f(t),
    then integrates semi-major axis decay with da/dt scaled by f(t).

    Args:
        semi_major_axis_m: Initial semi-major axis (m).
        eccentricity: Orbital eccentricity.
        drag_config: Satellite drag configuration.
        epoch: Start time.
        density_variability_fraction: Fractional density variability (~0.2).
        mean_reversion_days: Mean reversion timescale (days, ~27 for solar rotation).
        num_samples: Number of Monte Carlo samples.
        re_entry_altitude_km: Re-entry altitude threshold (km).
        step_days: Integration step size (days).
        max_years: Maximum simulation duration (years).
        rng_seed: Random number generator seed for reproducibility.

    Returns:
        StochasticLifetime with distribution statistics.

    Raises:
        ValueError: If already below re-entry altitude or invalid parameters.
    """
    initial_alt_km = (semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
    if initial_alt_km <= re_entry_altitude_km:
        raise ValueError(
            f"Initial altitude {initial_alt_km:.1f} km already at or below "
            f"re-entry altitude {re_entry_altitude_km} km"
        )
    if step_days <= 0:
        raise ValueError(f"step_days must be positive, got {step_days}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    # Deterministic baseline
    det_result = compute_orbit_lifetime(
        semi_major_axis_m, eccentricity, drag_config, epoch,
        re_entry_altitude_km=re_entry_altitude_km,
        step_days=step_days,
        max_years=max_years,
    )
    det_lifetime_days = det_result.lifetime_days

    # OU process parameters
    theta = 1.0 / mean_reversion_days  # reversion rate
    dt = step_days
    dt_seconds = step_days * 86400.0
    max_steps = int(max_years * 365.25 / step_days) + 1

    # OU diffusion coefficient for the density multiplier
    # Stationary variance of OU process: sigma_f^2 / (2*theta)
    # We want stationary std = density_variability_fraction
    # So sigma_f = density_variability_fraction * sqrt(2 * theta)
    sigma_f = density_variability_fraction * math.sqrt(2.0 * theta)

    rng = np.random.default_rng(rng_seed)
    re_entry_a = OrbitalConstants.R_EARTH + re_entry_altitude_km * 1000.0

    lifetimes = []
    num_converged = 0

    for _ in range(num_samples):
        a = semi_major_axis_m
        f = 1.0
        t_elapsed_days = 0.0
        converged = False

        for _ in range(max_steps):
            dw = rng.normal(0.0, math.sqrt(dt))
            f += theta * (1.0 - f) * dt + sigma_f * dw
            f = max(f, 0.01)

            current_time = epoch + timedelta(days=t_elapsed_days + step_days)
            if density_func is not None:
                alt_km = (a - OrbitalConstants.R_EARTH) / 1000.0
                rho = density_func(alt_km, current_time)
                v = float(np.sqrt(OrbitalConstants.MU_EARTH / a))
                da_dt = -rho * v * drag_config.ballistic_coefficient * a
            else:
                da_dt = semi_major_axis_decay_rate(a, eccentricity, drag_config)
            a += da_dt * f * dt_seconds
            t_elapsed_days += step_days

            if a <= re_entry_a:
                converged = True
                break

        if converged:
            num_converged += 1
        lifetimes.append(t_elapsed_days)

    lifetimes_arr = np.array(lifetimes)
    mean_lt = float(np.mean(lifetimes_arr))
    std_lt = float(np.std(lifetimes_arr))
    p5 = float(np.percentile(lifetimes_arr, 5))
    p50 = float(np.percentile(lifetimes_arr, 50))
    p95 = float(np.percentile(lifetimes_arr, 95))

    uncertainty_ratio = std_lt / mean_lt if mean_lt > 0 else 0.0

    return StochasticLifetime(
        mean_lifetime_days=mean_lt,
        std_lifetime_days=std_lt,
        percentile_5_days=p5,
        percentile_50_days=p50,
        percentile_95_days=p95,
        deterministic_lifetime_days=det_lifetime_days,
        uncertainty_ratio=uncertainty_ratio,
        num_samples=num_samples,
        num_converged=num_converged,
    )
