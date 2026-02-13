# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Conjunction management compositions.

Composes relative motion, conjunction assessment, and maneuver planning
to provide conjunction triage, avoidance maneuvers, and decision pipelines.

"""
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.relative_motion import (
    RelativeState,
    compute_relative_state,
    is_passively_safe,
)
from humeris.domain.conjunction import ConjunctionEvent
from humeris.domain.orbital_mechanics import OrbitalConstants


_MU = 3.986004418e14
_R_EARTH = OrbitalConstants.R_EARTH


class ConjunctionAction(Enum):
    NO_ACTION = "no_action"
    MANEUVER = "maneuver"
    ACCEPT_RISK = "accept_risk"


@dataclass(frozen=True)
class ConjunctionTriage:
    """Result of conjunction triage assessment."""
    action: ConjunctionAction
    is_passively_safe: bool
    relative_state: RelativeState
    maneuver_dv_ms: float


@dataclass(frozen=True)
class AvoidanceManeuver:
    """Computed avoidance maneuver."""
    delta_v_ms: float
    along_track_dv_ms: float
    lead_time_s: float
    post_maneuver_miss_m: float


@dataclass(frozen=True)
class ConjunctionDecision:
    """Full decision for a conjunction event."""
    event: ConjunctionEvent
    triage: ConjunctionTriage
    maneuver: AvoidanceManeuver | None
    fuel_sufficient: bool


def triage_conjunction(
    state1: OrbitalState,
    state2: OrbitalState,
    tca: datetime,
    min_safe_distance_m: float = 1000.0,
    check_periods: int = 2,
) -> ConjunctionTriage:
    """Triage a conjunction using relative motion analysis.

    Computes relative state, checks passive safety over check_periods
    orbital periods, and recommends action.

    Args:
        state1: First satellite state.
        state2: Second satellite state.
        tca: Time of closest approach.
        min_safe_distance_m: Minimum safe distance (m).
        check_periods: Number of orbital periods to check passive safety.

    Returns:
        ConjunctionTriage with action recommendation.
    """
    rel = compute_relative_state(state1, state2, tca)
    n = state1.mean_motion_rad_s
    T = 2.0 * math.pi / n
    check_duration = T * check_periods

    safe = is_passively_safe(rel, n, min_safe_distance_m, check_duration)

    # Compute distance at TCA
    dist = float(np.linalg.norm([rel.x, rel.y, rel.z]))

    if dist > min_safe_distance_m and safe:
        action = ConjunctionAction.NO_ACTION
        dv = 0.0
    else:
        action = ConjunctionAction.MANEUVER
        # Estimate dV: along-track impulse to shift by min_safe_distance_m
        # dV ≈ n * miss_distance / 2 for along-track separation
        dv = n * min_safe_distance_m / 2.0

    return ConjunctionTriage(
        action=action,
        is_passively_safe=safe,
        relative_state=rel,
        maneuver_dv_ms=dv,
    )


def compute_avoidance_maneuver(
    state1: OrbitalState,
    state2: OrbitalState,
    tca: datetime,
    target_miss_m: float = 1000.0,
) -> AvoidanceManeuver:
    """Compute an along-track avoidance maneuver.

    Uses CW dynamics: an along-track impulse at lead_time before TCA
    produces a radial/along-track displacement at TCA.

    For an along-track impulse dVy applied at time T before TCA:
    dx(T) ≈ 2*dVy*(1 - cos(nT))/n
    dy(T) ≈ dVy*(4*sin(nT)/n - 3T)

    We use a half-period lead time for near-optimal separation.

    Args:
        state1: Primary satellite state.
        state2: Secondary satellite state.
        tca: Time of closest approach.
        target_miss_m: Desired miss distance (m).

    Returns:
        AvoidanceManeuver with dV and post-maneuver miss.
    """
    rel = compute_relative_state(state1, state2, tca)
    n = state1.mean_motion_rad_s
    T = 2.0 * math.pi / n

    # Use half-period lead time
    lead_time = T / 2.0

    # At half period: sin(nT/2) = sin(π) = 0, cos(nT/2) = cos(π) = -1
    # dx = 2*dVy*(1-(-1))/n = 4*dVy/n
    # dy = dVy*(4*0/n - 3*T/2) = -3*dVy*T/2
    # |displacement| ≈ sqrt((4/n)^2 + (3T/2)^2) * |dVy|
    factor = float(np.sqrt((4.0 / n)**2 + (1.5 * T)**2))

    if factor < 1e-10:
        factor = 1.0

    dv_along = target_miss_m / factor

    # Post-maneuver miss estimate
    post_miss = factor * dv_along

    return AvoidanceManeuver(
        delta_v_ms=abs(dv_along),
        along_track_dv_ms=dv_along,
        lead_time_s=lead_time,
        post_maneuver_miss_m=post_miss,
    )


def run_conjunction_decision_pipeline(
    events: list[ConjunctionEvent],
    states: dict[str, OrbitalState],
    fuel_budgets_ms: dict[str, float],
    min_safe_distance_m: float = 1000.0,
) -> list[ConjunctionDecision]:
    """Run full conjunction decision pipeline for a list of events.

    For each event:
    1. Look up states by satellite name
    2. Triage the conjunction
    3. If maneuver needed, compute avoidance maneuver
    4. Check fuel budget sufficiency

    Args:
        events: List of ConjunctionEvent.
        states: Dict mapping satellite name to OrbitalState.
        fuel_budgets_ms: Dict mapping satellite name to remaining dV budget (m/s).
        min_safe_distance_m: Minimum safe distance (m).

    Returns:
        List of ConjunctionDecision for each event.
    """
    decisions: list[ConjunctionDecision] = []

    for event in events:
        s1 = states.get(event.sat1_name)
        s2 = states.get(event.sat2_name)
        if s1 is None or s2 is None:
            continue

        triage = triage_conjunction(
            s1, s2, event.tca,
            min_safe_distance_m=min_safe_distance_m,
        )

        maneuver = None
        fuel_ok = True

        if triage.action == ConjunctionAction.MANEUVER:
            maneuver = compute_avoidance_maneuver(
                s1, s2, event.tca,
                target_miss_m=min_safe_distance_m,
            )
            budget = fuel_budgets_ms.get(event.sat1_name, 0.0)
            fuel_ok = budget >= maneuver.delta_v_ms

        decisions.append(ConjunctionDecision(
            event=event,
            triage=triage,
            maneuver=maneuver,
            fuel_sufficient=fuel_ok,
        ))

    return decisions


# ── M/G/1 Queue for Maneuver Scheduling ─────────────────────────────
#
# Models conjunction avoidance maneuvers as arrivals and satellite
# propulsion as the server. Pollaczek-Khinchine formula for mean
# waiting time and dV budget competition between avoidance and
# station-keeping. See SESSION_MINING_R1_CREATIVE.md P9.

_SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass(frozen=True)
class ManeuverQueueModel:
    """M/G/1 queueing model for maneuver scheduling.

    Attributes:
        conjunction_arrival_rate_per_year: Lambda — Poisson arrival rate.
        mean_maneuver_dv_ms: E[S] — mean service (dV per avoidance maneuver).
        utilization_factor: rho = lambda * E[S] / capacity_rate.
        mean_queue_length: L_q from Pollaczek-Khinchine.
        mean_waiting_time_days: W_q — mean time a maneuver waits in queue.
        is_stable: True if rho < 1 (queue does not grow unbounded).
        overflow_probability: P(cumulative dV demand > budget) over mission.
        sk_fraction: Fraction of dV budget consumed by station-keeping.
        avoidance_fraction: Fraction of dV budget consumed by avoidance.
        total_dv_demand_per_year_ms: Combined annual dV demand (SK + avoidance).
    """
    conjunction_arrival_rate_per_year: float
    mean_maneuver_dv_ms: float
    utilization_factor: float
    mean_queue_length: float
    mean_waiting_time_days: float
    is_stable: bool
    overflow_probability: float
    sk_fraction: float
    avoidance_fraction: float
    total_dv_demand_per_year_ms: float


def compute_maneuver_queue(
    conjunction_rate_per_year: float,
    mean_avoidance_dv_ms: float,
    variance_avoidance_dv_ms2: float,
    sk_dv_per_year_ms: float,
    total_dv_budget_ms: float,
    mission_years: float,
) -> ManeuverQueueModel:
    """Compute M/G/1 queue model for maneuver scheduling.

    Models the competition between conjunction avoidance maneuvers
    (random arrivals) and station-keeping maneuvers (deterministic
    service) for the finite delta-V budget.

    The "server" is the satellite propulsion system. Conjunction
    warnings arrive as a Poisson process with rate lambda. Each
    avoidance maneuver consumes a random dV drawn from a general
    distribution with known mean and variance.

    Pollaczek-Khinchine formula:
        L_q = (lambda^2 * E[S^2]) / (2 * (1 - rho))
        where rho = lambda * E[S] / (dV_budget / mission_time - SK_rate)

    The "service rate" is the available dV rate after station-keeping:
        mu = (total_dV / mission_years - sk_dV/year) per year

    Args:
        conjunction_rate_per_year: Expected conjunctions per year (lambda).
        mean_avoidance_dv_ms: Mean dV per avoidance maneuver (m/s).
        variance_avoidance_dv_ms2: Variance of avoidance dV (m^2/s^2).
        sk_dv_per_year_ms: Station-keeping dV per year (m/s/year).
        total_dv_budget_ms: Total mission dV budget (m/s).
        mission_years: Mission duration (years).

    Returns:
        ManeuverQueueModel with queue statistics and budget analysis.

    Raises:
        ValueError: If mission_years <= 0 or total_dv_budget <= 0.
    """
    if mission_years <= 0:
        raise ValueError(f"mission_years must be positive, got {mission_years}")
    if total_dv_budget_ms <= 0:
        raise ValueError(
            f"total_dv_budget_ms must be positive, got {total_dv_budget_ms}"
        )

    # Available dV rate for avoidance (after station-keeping)
    total_dv_rate = total_dv_budget_ms / mission_years  # m/s per year
    avoidance_dv_rate = total_dv_rate - sk_dv_per_year_ms  # available for CA

    # Total annual dV demand
    avoidance_demand = conjunction_rate_per_year * mean_avoidance_dv_ms
    total_demand = sk_dv_per_year_ms + avoidance_demand

    # Utilization factor: fraction of available CA budget consumed
    if avoidance_dv_rate > 0:
        rho = avoidance_demand / avoidance_dv_rate
    else:
        rho = float("inf")

    is_stable = rho < 1.0

    # Pollaczek-Khinchine formula for mean queue length
    # E[S^2] = Var[S] + (E[S])^2
    e_s2 = variance_avoidance_dv_ms2 + mean_avoidance_dv_ms ** 2
    lam = conjunction_rate_per_year

    if is_stable and avoidance_dv_rate > 0:
        # L_q = (lambda^2 * E[S^2]) / (2 * mu^2 * (1 - rho))
        l_q = (lam ** 2 * e_s2) / (2.0 * avoidance_dv_rate ** 2 * (1.0 - rho))
        # W_q = L_q / lambda (Little's Law)
        if lam > 0:
            w_q_years = l_q / lam
            w_q_days = w_q_years * 365.25
        else:
            w_q_days = 0.0
    else:
        l_q = float("inf")
        w_q_days = float("inf")

    # Budget fractions
    if total_dv_rate > 0:
        sk_frac = sk_dv_per_year_ms / total_dv_rate
        avoid_frac = avoidance_demand / total_dv_rate
    else:
        sk_frac = 0.0
        avoid_frac = 0.0

    # Overflow probability: P(total dV demand > budget)
    # Over mission_years, total avoidance dV ~ N(lambda*E[S]*T, lambda*E[S^2]*T)
    # by CLT for compound Poisson process.
    total_avoidance_mean = avoidance_demand * mission_years
    total_avoidance_var = lam * e_s2 * mission_years
    total_sk = sk_dv_per_year_ms * mission_years
    total_mean = total_sk + total_avoidance_mean
    total_std = math.sqrt(total_avoidance_var) if total_avoidance_var > 0 else 0.0

    if total_std > 1e-30:
        # P(demand > budget) via normal approximation
        z = (total_dv_budget_ms - total_mean) / total_std
        # Use complementary error function: P(Z > z) = 0.5 * erfc(z / sqrt(2))
        overflow_p = 0.5 * math.erfc(z / math.sqrt(2.0))
    else:
        overflow_p = 1.0 if total_mean > total_dv_budget_ms else 0.0

    return ManeuverQueueModel(
        conjunction_arrival_rate_per_year=conjunction_rate_per_year,
        mean_maneuver_dv_ms=mean_avoidance_dv_ms,
        utilization_factor=rho,
        mean_queue_length=l_q,
        mean_waiting_time_days=w_q_days,
        is_stable=is_stable,
        overflow_probability=overflow_p,
        sk_fraction=sk_frac,
        avoidance_fraction=avoid_frac,
        total_dv_demand_per_year_ms=total_demand,
    )


# ── P27: Black-Scholes Maneuver Option Pricing ────────────────────


@dataclass(frozen=True)
class ManeuverOption:
    """Black-Scholes option pricing model for conjunction maneuver decisions.

    Treats a collision avoidance maneuver as an American put option on
    miss distance. The "asset" is the miss distance (geometric Brownian
    motion), and the "strike" is the safe miss distance threshold.

    Attributes:
        option_value_ms: Option value in equivalent delta-V cost (m/s).
        optimal_threshold_m: Critical miss distance below which
            immediate maneuver is optimal (American put early exercise).
        should_maneuver_now: True if current miss distance <= optimal_threshold.
        time_value_ms: Time value component of the option (m/s).
        intrinsic_value_ms: Intrinsic value component (m/s).
    """
    option_value_ms: float
    optimal_threshold_m: float
    should_maneuver_now: bool
    time_value_ms: float
    intrinsic_value_ms: float


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via error function: N(x) = 0.5*(1 + erf(x/sqrt(2)))."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_maneuver_option(
    miss_distance_m: float,
    miss_distance_sigma_m: float,
    safe_distance_m: float,
    time_to_tca_s: float,
    maneuver_cost_ms_per_m: float,
    risk_free_rate: float = 0.0,
) -> ManeuverOption:
    """Black-Scholes option pricing for conjunction avoidance maneuvers.

    Models miss distance as geometric Brownian motion:
        dS = mu*S*dt + sigma*S*dW
    where S = miss distance, sigma = relative uncertainty.

    A collision avoidance maneuver is an American put option: the right
    to "sell" (execute maneuver) at strike K (safe distance). If miss
    distance falls below K, the maneuver has positive intrinsic value.

    The Barone-Adesi and Whaley approximation for the American put
    critical exercise boundary:
        S* = K * beta1 / (beta1 - 1)
    where beta1 = (0.5 - mu/sigma^2) + sqrt((mu/sigma^2 - 0.5)^2 + 2*r/sigma^2)

    Args:
        miss_distance_m: Current predicted miss distance (m), the "asset price" S.
        miss_distance_sigma_m: 1-sigma uncertainty in miss distance (m).
        safe_distance_m: Safe miss distance threshold (m), the "strike price" K.
        time_to_tca_s: Time to TCA (seconds). Normalized internally to
            orbital periods (~5400s for LEO).
        maneuver_cost_ms_per_m: Cost of maneuver in delta-V per meter of
            miss distance change (m/s per m). Converts option value to dV.
        risk_free_rate: Risk-free rate (dimensionless, per normalized time unit).
            Typically 0 for astrodynamics (no discounting). Default 0.

    Returns:
        ManeuverOption with option valuation and decision.

    Raises:
        ValueError: If miss_distance_m <= 0, miss_distance_sigma_m <= 0,
            safe_distance_m <= 0, time_to_tca_s <= 0, or
            maneuver_cost_ms_per_m <= 0.
    """
    if miss_distance_m <= 0:
        raise ValueError(f"miss_distance_m must be positive, got {miss_distance_m}")
    if miss_distance_sigma_m <= 0:
        raise ValueError(
            f"miss_distance_sigma_m must be positive, got {miss_distance_sigma_m}"
        )
    if safe_distance_m <= 0:
        raise ValueError(f"safe_distance_m must be positive, got {safe_distance_m}")
    if time_to_tca_s <= 0:
        raise ValueError(f"time_to_tca_s must be positive, got {time_to_tca_s}")
    if maneuver_cost_ms_per_m <= 0:
        raise ValueError(
            f"maneuver_cost_ms_per_m must be positive, got {maneuver_cost_ms_per_m}"
        )

    S = miss_distance_m
    K = safe_distance_m
    sigma = miss_distance_sigma_m / miss_distance_m  # relative uncertainty
    r = risk_free_rate

    # Normalize time: use ~1 orbital period as unit (~5400s for LEO)
    T_norm = 5400.0
    T = time_to_tca_s / T_norm

    # Drift: assume zero drift (miss distance is a martingale under
    # equal probability of approach/recession). For conservative analysis,
    # mu < 0 would model approaching objects.
    mu = 0.0

    # Black-Scholes European put: P = K*exp(-r*T)*N(-d2) - S*N(-d1)
    sigma_sqrt_T = sigma * math.sqrt(T)

    if sigma_sqrt_T < 1e-15:
        # Degenerate: no uncertainty or no time left
        intrinsic = max(K - S, 0.0)
        return ManeuverOption(
            option_value_ms=intrinsic * maneuver_cost_ms_per_m,
            optimal_threshold_m=K,
            should_maneuver_now=S <= K,
            time_value_ms=0.0,
            intrinsic_value_ms=intrinsic * maneuver_cost_ms_per_m,
        )

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    put_value = K * math.exp(-r * T) * _normal_cdf(-d2) - S * _normal_cdf(-d1)
    put_value = max(put_value, 0.0)

    # Intrinsic value
    intrinsic = max(K - S, 0.0)
    time_value = put_value - intrinsic

    # American put approximation: critical exercise boundary
    # beta1 = (0.5 - mu/sigma^2) + sqrt((mu/sigma^2 - 0.5)^2 + 2*r/sigma^2)
    sigma2 = sigma ** 2
    if sigma2 > 1e-30:
        half_minus_mu_over_s2 = 0.5 - mu / sigma2
        discriminant = half_minus_mu_over_s2 ** 2 + 2.0 * r / sigma2
        # discriminant is always >= 0.25 when mu=0, r>=0
        beta1 = half_minus_mu_over_s2 + math.sqrt(max(0.0, discriminant))
    else:
        beta1 = float('inf')

    # Critical miss distance: S* = K * beta1 / (beta1 - 1)
    if beta1 > 1.0:
        optimal_threshold = K * beta1 / (beta1 - 1.0)
    else:
        # beta1 <= 1: immediate exercise always optimal
        optimal_threshold = float('inf')

    should_maneuver = S <= optimal_threshold

    return ManeuverOption(
        option_value_ms=put_value * maneuver_cost_ms_per_m,
        optimal_threshold_m=optimal_threshold,
        should_maneuver_now=should_maneuver,
        time_value_ms=max(time_value, 0.0) * maneuver_cost_ms_per_m,
        intrinsic_value_ms=intrinsic * maneuver_cost_ms_per_m,
    )
