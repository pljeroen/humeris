# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Conjunction management compositions.

Composes relative motion, conjunction assessment, and maneuver planning
to provide conjunction triage, avoidance maneuvers, and decision pipelines.

No external dependencies — only stdlib math/dataclasses/datetime/enum.
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


_MU = 3.986004418e14
_R_EARTH = 6_371_000.0


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
