# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Hazard reporting with NASA CARA levels, provenance, and composite risk.

Classifies conjunction events into ROUTINE / WARNING / CRITICAL levels
and signs hazard reports with SHA-256 provenance hashes linking the
physical models used in the assessment.

Cross-disciplinary enhancements:
    Covariance-unavailable handling (STPA, Aerospace FMEA):
        Missing covariance defaults to miss-distance-only classification
        with tighter thresholds, not silent ROUTINE.
    Hysteresis (Software Engineering circuit breaker):
        Once escalated, require sustained ROUTINE period before
        de-escalation to prevent dangerous boundary oscillation.
    Breakup potential (Physics):
        Flag catastrophic breakup when V_rel > 283 m/s (40 kJ/kg).
    Secondary maneuver detection (ESA/18th SDS practice):
        Escalate hazard when secondary object detected maneuvering.
    Conjunction Weather Index (Meteorology FWI):
        Composite scalar from Pc, miss, covariance quality, cascade
        context, and secondary maneuver detection.
    HMAC-SHA256 (Forensic Science):
        Optional keyed hash for non-repudiation.

References:
    NASA CARA (Conjunction Assessment Risk Analysis) operational thresholds.
    IERS 2010 Conventions Chapter 10.
    Forestry Canada Fire Weather Index (FWI) composition structure.
"""
import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from humeris.domain.conjunction import ConjunctionEvent


class HazardLevel(Enum):
    """NASA CARA hazard classification levels.

    ROUTINE:  Pc < 1e-6  -- logged in standard telemetry.
    WARNING:  Pc >= 1e-6 and < 1e-4, or miss < 5 km -- triggers COLA report.
    CRITICAL: Pc >= 1e-4, or miss < 1 km with no maneuver plan -- triggers
              Hazard Alert + Adversarial Assessment.
    """
    ROUTINE = "routine"
    WARNING = "warning"
    CRITICAL = "critical"


# Specific kinetic energy threshold for catastrophic breakup (40 kJ/kg).
# V_rel > sqrt(2 * 40000) ~ 283 m/s.
_CATASTROPHIC_BREAKUP_VELOCITY_MS = math.sqrt(2.0 * 40000.0)


@dataclass(frozen=True)
class HazardReport:
    """Signed hazard report with full provenance chain."""
    level: HazardLevel
    conjunction_event: ConjunctionEvent
    force_models_used: tuple
    nutation_model: str
    timestamp: datetime
    provenance_hash: str
    has_maneuver_plan: bool
    recommended_action: str
    is_catastrophic_breakup: bool = False
    breakup_specific_energy_j_per_kg: float = 0.0
    conjunction_weather_index: float = 0.0
    covariance_available: bool = True
    secondary_maneuver_detected: bool = False


@dataclass(frozen=True)
class HazardHysteresis:
    """Hysteresis state for hazard level de-escalation.

    Once escalated to WARNING or CRITICAL, requires consecutive_routine
    >= required_routine_count before de-escalation is permitted.
    Prevents dangerous oscillation at classification boundaries
    (Software Engineering circuit breaker pattern).
    """
    current_level: HazardLevel
    consecutive_routine: int
    required_routine_count: int = 3


def classify_hazard(
    event: ConjunctionEvent,
    has_maneuver_plan: bool = False,
    secondary_maneuver_detected: bool = False,
    covariance_available: bool = True,
) -> HazardLevel:
    """Classify a conjunction event per NASA CARA thresholds.

    When covariance is unavailable (Pc = 0 from missing covariance data),
    classification falls back to miss-distance-only with tighter thresholds
    (STPA finding: covariance-unavailable must not default to ROUTINE).

    When secondary object has a detected maneuver, escalate by one level
    because orbital prediction is unreliable (ESA/18th SDS practice).

    Args:
        event: Conjunction event with miss distance and collision probability.
        has_maneuver_plan: Whether an avoidance maneuver plan exists.
        secondary_maneuver_detected: Whether the secondary object has a
            detected non-cooperative maneuver (increases uncertainty).
        covariance_available: Whether covariance data was available for
            Pc computation. False triggers miss-distance-only path.

    Returns:
        HazardLevel classification.
    """
    pc = max(event.collision_probability, event.max_collision_probability)
    miss = event.miss_distance_m

    if not covariance_available:
        # STPA/FMEA: covariance-unavailable path.
        # Pc=0 from missing covariance is NOT "safe" -- classify on
        # miss distance alone with tighter thresholds.
        level = _classify_miss_distance_only(miss, has_maneuver_plan)
    else:
        level = _classify_standard(pc, miss, has_maneuver_plan)

    # Secondary maneuver escalation (ESA/18th SDS practice):
    # non-cooperative maneuver on secondary makes prediction unreliable.
    if secondary_maneuver_detected and miss < 10000.0:
        if level == HazardLevel.ROUTINE:
            level = HazardLevel.WARNING
        elif level == HazardLevel.WARNING:
            level = HazardLevel.CRITICAL

    return level


def _classify_standard(
    pc: float,
    miss: float,
    has_maneuver_plan: bool,
) -> HazardLevel:
    """Standard classification with Pc and miss distance."""
    # CRITICAL
    if pc >= 1e-4:
        return HazardLevel.CRITICAL
    if miss < 1000.0 and not has_maneuver_plan:
        return HazardLevel.CRITICAL

    # WARNING
    if pc >= 1e-6:
        return HazardLevel.WARNING
    if miss < 5000.0:
        return HazardLevel.WARNING

    # ROUTINE
    return HazardLevel.ROUTINE


def _classify_miss_distance_only(
    miss: float,
    has_maneuver_plan: bool,
) -> HazardLevel:
    """Classification when covariance unavailable (Pc not computable).

    Tighter thresholds than standard classification because uncertainty
    is higher without covariance data.

    CRITICAL: miss < 2 km (tighter than standard 1 km)
    WARNING: miss < 10 km (tighter than standard 5 km)
    """
    if miss < 2000.0 and not has_maneuver_plan:
        return HazardLevel.CRITICAL
    if miss < 10000.0:
        return HazardLevel.WARNING
    return HazardLevel.ROUTINE


def apply_hysteresis(
    raw_level: HazardLevel,
    state: HazardHysteresis | None,
) -> tuple[HazardLevel, HazardHysteresis]:
    """Apply hysteresis to prevent hazard level oscillation.

    Escalation is immediate. De-escalation requires sustained ROUTINE
    period (circuit breaker half-open -> closed transition).

    Args:
        raw_level: Current classification from classify_hazard.
        state: Previous hysteresis state (None if first evaluation).

    Returns:
        Tuple of (effective_level, updated_hysteresis_state).
    """
    if state is None:
        return raw_level, HazardHysteresis(
            current_level=raw_level,
            consecutive_routine=0,
        )

    required = state.required_routine_count

    # Escalation is always immediate
    level_order = {
        HazardLevel.ROUTINE: 0,
        HazardLevel.WARNING: 1,
        HazardLevel.CRITICAL: 2,
    }

    if level_order[raw_level] >= level_order[state.current_level]:
        # Escalation or same level
        new_consecutive = (
            state.consecutive_routine + 1
            if raw_level == HazardLevel.ROUTINE else 0
        )
        return raw_level, HazardHysteresis(
            current_level=raw_level,
            consecutive_routine=new_consecutive,
            required_routine_count=required,
        )

    # De-escalation requested
    if raw_level == HazardLevel.ROUTINE:
        new_consecutive = state.consecutive_routine + 1
        if new_consecutive >= required:
            # Sustained ROUTINE: permit de-escalation
            return HazardLevel.ROUTINE, HazardHysteresis(
                current_level=HazardLevel.ROUTINE,
                consecutive_routine=new_consecutive,
                required_routine_count=required,
            )
        # Not enough sustained ROUTINE: hold previous level
        return state.current_level, HazardHysteresis(
            current_level=state.current_level,
            consecutive_routine=new_consecutive,
            required_routine_count=required,
        )

    # Partial de-escalation (CRITICAL -> WARNING)
    return state.current_level, HazardHysteresis(
        current_level=state.current_level,
        consecutive_routine=0,
        required_routine_count=required,
    )


def compute_breakup_potential(relative_velocity_ms: float) -> tuple[bool, float]:
    """Determine if collision would cause catastrophic breakup.

    Catastrophic breakup occurs when specific kinetic energy exceeds
    40 J/g = 40,000 J/kg, corresponding to V_rel > ~283 m/s.

    At typical LEO encounter velocities (7-15 km/s), virtually all
    collisions are catastrophic. This metric is relevant for nearly
    co-orbital objects with low relative velocities.

    Args:
        relative_velocity_ms: Relative velocity at TCA (m/s).

    Returns:
        Tuple of (is_catastrophic, specific_energy_j_per_kg).
    """
    e_specific = 0.5 * relative_velocity_ms ** 2
    is_catastrophic = relative_velocity_ms > _CATASTROPHIC_BREAKUP_VELOCITY_MS
    return is_catastrophic, e_specific


def compute_conjunction_weather_index(
    pc: float,
    miss_distance_m: float,
    relative_velocity_ms: float,
    covariance_available: bool = True,
    secondary_maneuver_detected: bool = False,
    cascade_k_eff: float = 0.0,
) -> float:
    """Compute composite Conjunction Weather Index (CWI).

    Modeled on the Canadian Fire Weather Index (FWI) composition
    structure: multiple sub-indices combine into a single scalar.

    CWI components:
        Pc_score: Normalized collision probability (0-1).
        Miss_score: Normalized miss distance risk (0-1).
        Uncertainty_score: Covariance + maneuver detection (0-1).
        Cascade_score: Environmental cascade context (0-1).

    CWI = weighted geometric mean of sub-scores, range [0, 1].
    Higher CWI = higher overall conjunction risk.

    Args:
        pc: Collision probability (max of computed and upper bound).
        miss_distance_m: Miss distance at TCA (meters).
        relative_velocity_ms: Relative velocity at TCA (m/s).
        covariance_available: Whether covariance data is available.
        secondary_maneuver_detected: Whether secondary is maneuvering.
        cascade_k_eff: Cascade multiplication factor from kessler heatmap.

    Returns:
        CWI score in range [0, 1].
    """
    # Pc sub-index: logistic mapping
    # Pc=1e-7 -> 0.1, Pc=1e-5 -> 0.5, Pc=1e-3 -> 0.9
    if pc > 0:
        pc_score = 1.0 / (1.0 + math.exp(-1.5 * (math.log10(pc) + 5.0)))
    else:
        pc_score = 0.0

    # Miss distance sub-index: inverse logistic
    # 100km -> 0.05, 10km -> 0.3, 1km -> 0.8, 0.1km -> 0.95
    miss_km = miss_distance_m / 1000.0
    miss_score = 1.0 / (1.0 + (miss_km / 5.0) ** 1.5)

    # Uncertainty sub-index
    uncertainty_score = 0.0
    if not covariance_available:
        uncertainty_score += 0.5
    if secondary_maneuver_detected:
        uncertainty_score += 0.5

    # Cascade context sub-index
    # k_eff > 1 is supercritical
    cascade_score = min(cascade_k_eff, 2.0) / 2.0

    # Weighted combination (FWI-style)
    # Weights: Pc=0.35, Miss=0.25, Uncertainty=0.25, Cascade=0.15
    cwi = (
        0.35 * pc_score
        + 0.25 * miss_score
        + 0.25 * uncertainty_score
        + 0.15 * cascade_score
    )
    return min(max(cwi, 0.0), 1.0)


def _compute_provenance_hash(
    event: ConjunctionEvent,
    force_models: tuple,
    nutation_model: str,
    timestamp: datetime,
    hmac_key: bytes | None = None,
) -> str:
    """Compute SHA-256 provenance hash over the physics chain.

    The hash covers: satellite names, TCA, miss distance, Pc,
    force model names, nutation model, and timestamp.

    If hmac_key is provided, uses HMAC-SHA256 for non-repudiation
    (Forensic Science: chain-of-custody integrity).
    """
    data = {
        "sat1": event.sat1_name,
        "sat2": event.sat2_name,
        "tca_iso": event.tca.isoformat(),
        "miss_distance_m": event.miss_distance_m,
        "collision_probability": event.collision_probability,
        "max_collision_probability": event.max_collision_probability,
        "relative_velocity_ms": event.relative_velocity_ms,
        "force_models": list(force_models),
        "nutation_model": nutation_model,
        "timestamp_iso": timestamp.isoformat(),
    }
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    payload = canonical.encode("utf-8")

    if hmac_key is not None:
        return hmac.new(hmac_key, payload, hashlib.sha256).hexdigest()

    return hashlib.sha256(payload).hexdigest()


def _recommended_action(
    level: HazardLevel,
    has_maneuver_plan: bool,
    covariance_available: bool = True,
    secondary_maneuver_detected: bool = False,
) -> str:
    """Determine recommended action based on hazard level and context."""
    suffix = ""
    if not covariance_available:
        suffix = " Covariance unavailable — request updated OD."
    if secondary_maneuver_detected:
        suffix += " Secondary maneuver detected — prediction unreliable."

    if level == HazardLevel.ROUTINE:
        return f"Log to telemetry. No action required.{suffix}".strip()
    if level == HazardLevel.WARNING:
        if has_maneuver_plan:
            return (
                f"COLA report generated. Maneuver plan on standby.{suffix}"
            ).strip()
        return (
            f"COLA report generated. Prepare avoidance maneuver.{suffix}"
        ).strip()
    # CRITICAL
    if has_maneuver_plan:
        return (
            "HAZARD ALERT. Execute avoidance maneuver. "
            f"Adversarial assessment required.{suffix}"
        ).strip()
    return (
        "HAZARD ALERT. Immediate maneuver planning required. "
        f"Adversarial assessment required.{suffix}"
    ).strip()


def sign_report(
    event: ConjunctionEvent,
    force_models: tuple,
    nutation_model: str = "IAU_2000A",
    has_maneuver_plan: bool = False,
    timestamp: datetime | None = None,
    secondary_maneuver_detected: bool = False,
    covariance_available: bool = True,
    cascade_k_eff: float = 0.0,
    hmac_key: bytes | None = None,
) -> HazardReport:
    """Create a signed hazard report with full provenance chain.

    Args:
        event: Conjunction event from conjunction assessment.
        force_models: Tuple of force model names used in propagation.
        nutation_model: Nutation model identifier (e.g., "IAU_2000A").
        has_maneuver_plan: Whether an avoidance maneuver plan exists.
        timestamp: Report timestamp (defaults to event TCA).
        secondary_maneuver_detected: Whether secondary is maneuvering.
        covariance_available: Whether covariance was available for Pc.
        cascade_k_eff: Cascade multiplication factor from kessler heatmap.
        hmac_key: Optional HMAC key for non-repudiation signing.

    Returns:
        HazardReport with SHA-256 provenance hash and composite metrics.
    """
    if timestamp is None:
        timestamp = event.tca

    level = classify_hazard(
        event, has_maneuver_plan,
        secondary_maneuver_detected, covariance_available,
    )
    provenance = _compute_provenance_hash(
        event, force_models, nutation_model, timestamp, hmac_key,
    )
    action = _recommended_action(
        level, has_maneuver_plan,
        covariance_available, secondary_maneuver_detected,
    )

    is_catastrophic, e_specific = compute_breakup_potential(
        event.relative_velocity_ms,
    )

    pc = max(event.collision_probability, event.max_collision_probability)
    cwi = compute_conjunction_weather_index(
        pc, event.miss_distance_m, event.relative_velocity_ms,
        covariance_available, secondary_maneuver_detected, cascade_k_eff,
    )

    return HazardReport(
        level=level,
        conjunction_event=event,
        force_models_used=force_models,
        nutation_model=nutation_model,
        timestamp=timestamp,
        provenance_hash=provenance,
        has_maneuver_plan=has_maneuver_plan,
        recommended_action=action,
        is_catastrophic_breakup=is_catastrophic,
        breakup_specific_energy_j_per_kg=e_specific,
        conjunction_weather_index=cwi,
        covariance_available=covariance_available,
        secondary_maneuver_detected=secondary_maneuver_detected,
    )
