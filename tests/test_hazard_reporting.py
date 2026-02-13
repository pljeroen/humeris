# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for hazard reporting with NASA-STD-8719.14 levels and provenance."""
import ast
import hashlib
import json
import math
from datetime import datetime, timezone

import pytest

from humeris.domain.conjunction import ConjunctionEvent
from humeris.domain.hazard_reporting import (
    HazardHysteresis,
    HazardLevel,
    HazardReport,
    apply_hysteresis,
    classify_hazard,
    compute_breakup_potential,
    compute_conjunction_weather_index,
    sign_report,
    _CATASTROPHIC_BREAKUP_VELOCITY_MS,
)


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_event(
    pc=0.0,
    max_pc=0.0,
    miss_m=50000.0,
    rel_vel=14000.0,
) -> ConjunctionEvent:
    """Create a ConjunctionEvent with given parameters."""
    return ConjunctionEvent(
        sat1_name="SAT-A",
        sat2_name="SAT-B",
        tca=EPOCH,
        miss_distance_m=miss_m,
        relative_velocity_ms=rel_vel,
        collision_probability=pc,
        max_collision_probability=max_pc,
        b_plane_radial_m=100.0,
        b_plane_cross_track_m=200.0,
    )


# -- HazardLevel enum -------------------------------------------------------


class TestHazardLevel:

    def test_three_levels(self):
        """HazardLevel has exactly three values."""
        assert len(HazardLevel) == 3

    def test_values(self):
        assert HazardLevel.ROUTINE.value == "routine"
        assert HazardLevel.WARNING.value == "warning"
        assert HazardLevel.CRITICAL.value == "critical"


# -- classify_hazard: standard thresholds ------------------------------------


class TestClassifyHazard:

    def test_routine_low_pc_far_miss(self):
        """Pc < 1e-6 and miss > 5 km -> ROUTINE."""
        event = _make_event(pc=1e-8, miss_m=50000.0)
        assert classify_hazard(event) == HazardLevel.ROUTINE

    def test_warning_pc_above_1e6(self):
        """Pc >= 1e-6 -> WARNING."""
        event = _make_event(pc=5e-6, miss_m=10000.0)
        assert classify_hazard(event) == HazardLevel.WARNING

    def test_warning_miss_below_5km(self):
        """Miss < 5 km but Pc < 1e-4 -> WARNING."""
        event = _make_event(pc=1e-8, miss_m=3000.0)
        assert classify_hazard(event) == HazardLevel.WARNING

    def test_critical_pc_above_1e4(self):
        """Pc >= 1e-4 -> CRITICAL."""
        event = _make_event(pc=2e-4, miss_m=50000.0)
        assert classify_hazard(event) == HazardLevel.CRITICAL

    def test_critical_sub_km_no_maneuver(self):
        """Miss < 1 km and no maneuver plan -> CRITICAL."""
        event = _make_event(pc=1e-8, miss_m=500.0)
        assert classify_hazard(event, has_maneuver_plan=False) == HazardLevel.CRITICAL

    def test_warning_sub_km_with_maneuver(self):
        """Miss < 1 km but with maneuver plan -> WARNING (not CRITICAL)."""
        event = _make_event(pc=1e-8, miss_m=500.0)
        assert classify_hazard(event, has_maneuver_plan=True) == HazardLevel.WARNING

    def test_max_pc_used_in_classification(self):
        """max_collision_probability is also considered."""
        event = _make_event(pc=1e-8, max_pc=5e-4, miss_m=50000.0)
        assert classify_hazard(event) == HazardLevel.CRITICAL

    def test_boundary_pc_1e6(self):
        """Pc exactly 1e-6 -> WARNING."""
        event = _make_event(pc=1e-6, miss_m=50000.0)
        assert classify_hazard(event) == HazardLevel.WARNING

    def test_boundary_pc_1e4(self):
        """Pc exactly 1e-4 -> CRITICAL."""
        event = _make_event(pc=1e-4, miss_m=50000.0)
        assert classify_hazard(event) == HazardLevel.CRITICAL

    def test_boundary_miss_exactly_5km(self):
        """Miss exactly 5000 m is not < 5000 -> ROUTINE (with low Pc)."""
        event = _make_event(pc=1e-8, miss_m=5000.0)
        assert classify_hazard(event) == HazardLevel.ROUTINE

    def test_boundary_miss_exactly_1km(self):
        """Miss exactly 1000 m is not < 1000 -> not CRITICAL from miss alone."""
        event = _make_event(pc=1e-8, miss_m=1000.0)
        # 1000 is not < 1000, but is < 5000 -> WARNING
        assert classify_hazard(event) == HazardLevel.WARNING


# -- classify_hazard: covariance-unavailable path ----------------------------


class TestClassifyHazardCovarianceUnavailable:

    def test_miss_below_2km_critical(self):
        """Covariance unavailable: miss < 2 km -> CRITICAL."""
        event = _make_event(pc=0.0, miss_m=1500.0)
        assert classify_hazard(event, covariance_available=False) == HazardLevel.CRITICAL

    def test_miss_below_10km_warning(self):
        """Covariance unavailable: miss < 10 km (but >= 2 km) -> WARNING."""
        event = _make_event(pc=0.0, miss_m=5000.0)
        assert classify_hazard(event, covariance_available=False) == HazardLevel.WARNING

    def test_miss_above_10km_routine(self):
        """Covariance unavailable: miss >= 10 km -> ROUTINE."""
        event = _make_event(pc=0.0, miss_m=15000.0)
        assert classify_hazard(event, covariance_available=False) == HazardLevel.ROUTINE

    def test_boundary_miss_exactly_2km(self):
        """Covariance unavailable: miss exactly 2000 m is not < 2000 -> WARNING."""
        event = _make_event(pc=0.0, miss_m=2000.0)
        assert classify_hazard(event, covariance_available=False) == HazardLevel.WARNING

    def test_boundary_miss_exactly_10km(self):
        """Covariance unavailable: miss exactly 10000 m is not < 10000 -> ROUTINE."""
        event = _make_event(pc=0.0, miss_m=10000.0)
        assert classify_hazard(event, covariance_available=False) == HazardLevel.ROUTINE

    def test_miss_below_2km_with_maneuver_not_critical(self):
        """Covariance unavailable: miss < 2 km WITH maneuver plan -> WARNING."""
        event = _make_event(pc=0.0, miss_m=1500.0)
        result = classify_hazard(
            event, has_maneuver_plan=True, covariance_available=False,
        )
        assert result == HazardLevel.WARNING

    def test_pc_ignored_when_covariance_unavailable(self):
        """Covariance unavailable: high Pc is irrelevant, miss-only path used."""
        event = _make_event(pc=1e-3, miss_m=50000.0)
        assert classify_hazard(event, covariance_available=False) == HazardLevel.ROUTINE


# -- classify_hazard: secondary maneuver escalation --------------------------


class TestClassifyHazardSecondaryManeuver:

    def test_escalation_routine_to_warning(self):
        """Secondary maneuver detected: ROUTINE -> WARNING when miss < 10 km."""
        event = _make_event(pc=1e-8, miss_m=8000.0)
        # Without secondary: miss < 5 km is WARNING, but 8 km with low Pc -> ROUTINE?
        # Actually standard: miss 8000 >= 5000, Pc 1e-8 < 1e-6, so ROUTINE.
        assert classify_hazard(event) == HazardLevel.ROUTINE
        # With secondary: escalate ROUTINE -> WARNING since miss < 10 km
        assert classify_hazard(
            event, secondary_maneuver_detected=True,
        ) == HazardLevel.WARNING

    def test_escalation_warning_to_critical(self):
        """Secondary maneuver detected: WARNING -> CRITICAL when miss < 10 km."""
        event = _make_event(pc=5e-6, miss_m=8000.0)
        # Standard: Pc >= 1e-6 -> WARNING
        assert classify_hazard(event) == HazardLevel.WARNING
        # With secondary: escalate WARNING -> CRITICAL since miss < 10 km
        assert classify_hazard(
            event, secondary_maneuver_detected=True,
        ) == HazardLevel.CRITICAL

    def test_no_escalation_when_miss_above_10km(self):
        """Secondary maneuver detected: no escalation when miss >= 10 km."""
        event = _make_event(pc=1e-8, miss_m=15000.0)
        assert classify_hazard(event) == HazardLevel.ROUTINE
        assert classify_hazard(
            event, secondary_maneuver_detected=True,
        ) == HazardLevel.ROUTINE

    def test_critical_stays_critical(self):
        """Secondary maneuver detected: CRITICAL stays CRITICAL (no super-critical)."""
        event = _make_event(pc=2e-4, miss_m=5000.0)
        assert classify_hazard(event) == HazardLevel.CRITICAL
        assert classify_hazard(
            event, secondary_maneuver_detected=True,
        ) == HazardLevel.CRITICAL

    def test_no_escalation_at_exactly_10km(self):
        """Secondary maneuver: miss exactly 10000 m is not < 10000 -> no escalation."""
        event = _make_event(pc=1e-8, miss_m=10000.0)
        assert classify_hazard(
            event, secondary_maneuver_detected=True,
        ) == HazardLevel.ROUTINE


# -- classify_hazard: combined covariance + secondary maneuver ---------------


class TestClassifyHazardCombined:

    def test_covariance_unavailable_plus_secondary_maneuver(self):
        """No covariance + secondary: tighter threshold + escalation stack."""
        # miss 5000 m, covariance unavailable -> WARNING (< 10 km)
        # secondary maneuver + miss < 10 km -> escalate WARNING -> CRITICAL
        event = _make_event(pc=0.0, miss_m=5000.0)
        result = classify_hazard(
            event,
            covariance_available=False,
            secondary_maneuver_detected=True,
        )
        assert result == HazardLevel.CRITICAL

    def test_covariance_unavailable_plus_secondary_far_miss(self):
        """No covariance + secondary: miss >= 10 km -> ROUTINE, no escalation."""
        event = _make_event(pc=0.0, miss_m=15000.0)
        result = classify_hazard(
            event,
            covariance_available=False,
            secondary_maneuver_detected=True,
        )
        assert result == HazardLevel.ROUTINE

    def test_covariance_unavailable_close_miss_secondary(self):
        """No covariance + secondary + very close miss -> CRITICAL (already CRITICAL)."""
        event = _make_event(pc=0.0, miss_m=1000.0)
        result = classify_hazard(
            event,
            covariance_available=False,
            secondary_maneuver_detected=True,
        )
        assert result == HazardLevel.CRITICAL


# -- apply_hysteresis --------------------------------------------------------


class TestApplyHysteresis:

    def test_initial_state_none(self):
        """First evaluation: state is None -> raw level returned, state created."""
        level, state = apply_hysteresis(HazardLevel.WARNING, None)
        assert level == HazardLevel.WARNING
        assert state.current_level == HazardLevel.WARNING
        assert state.consecutive_routine == 0

    def test_initial_state_routine(self):
        """First evaluation with ROUTINE -> ROUTINE, consecutive_routine=0."""
        level, state = apply_hysteresis(HazardLevel.ROUTINE, None)
        assert level == HazardLevel.ROUTINE
        assert state.current_level == HazardLevel.ROUTINE
        assert state.consecutive_routine == 0

    def test_immediate_escalation(self):
        """Escalation is always immediate, no delay."""
        state = HazardHysteresis(
            current_level=HazardLevel.ROUTINE,
            consecutive_routine=5,
        )
        level, new_state = apply_hysteresis(HazardLevel.WARNING, state)
        assert level == HazardLevel.WARNING
        assert new_state.current_level == HazardLevel.WARNING
        assert new_state.consecutive_routine == 0

    def test_escalation_warning_to_critical(self):
        """Escalation from WARNING to CRITICAL is immediate."""
        state = HazardHysteresis(
            current_level=HazardLevel.WARNING,
            consecutive_routine=0,
        )
        level, new_state = apply_hysteresis(HazardLevel.CRITICAL, state)
        assert level == HazardLevel.CRITICAL
        assert new_state.current_level == HazardLevel.CRITICAL

    def test_de_escalation_blocked_insufficient_routine(self):
        """De-escalation blocked when consecutive ROUTINE < required."""
        state = HazardHysteresis(
            current_level=HazardLevel.WARNING,
            consecutive_routine=1,
        )
        level, new_state = apply_hysteresis(HazardLevel.ROUTINE, state)
        # Held at WARNING (not enough sustained ROUTINE)
        assert level == HazardLevel.WARNING
        assert new_state.current_level == HazardLevel.WARNING
        assert new_state.consecutive_routine == 2

    def test_de_escalation_permitted_after_sustained_routine(self):
        """De-escalation permitted after required_routine_count consecutive ROUTINE."""
        state = HazardHysteresis(
            current_level=HazardLevel.WARNING,
            consecutive_routine=2,  # Need 3 (default)
            required_routine_count=3,
        )
        level, new_state = apply_hysteresis(HazardLevel.ROUTINE, state)
        # 2 + 1 = 3 >= 3 -> de-escalation permitted
        assert level == HazardLevel.ROUTINE
        assert new_state.current_level == HazardLevel.ROUTINE
        assert new_state.consecutive_routine == 3

    def test_sustained_routine_increments_counter(self):
        """Multiple ROUTINE evaluations increment the counter."""
        state = HazardHysteresis(
            current_level=HazardLevel.ROUTINE,
            consecutive_routine=5,
        )
        level, new_state = apply_hysteresis(HazardLevel.ROUTINE, state)
        assert level == HazardLevel.ROUTINE
        assert new_state.consecutive_routine == 6

    def test_partial_de_escalation_blocked(self):
        """CRITICAL -> WARNING de-escalation blocked (not via ROUTINE path)."""
        state = HazardHysteresis(
            current_level=HazardLevel.CRITICAL,
            consecutive_routine=0,
        )
        level, new_state = apply_hysteresis(HazardLevel.WARNING, state)
        # Partial de-escalation: CRITICAL -> WARNING is blocked
        assert level == HazardLevel.CRITICAL
        assert new_state.current_level == HazardLevel.CRITICAL
        assert new_state.consecutive_routine == 0

    def test_partial_de_escalation_resets_counter(self):
        """Partial de-escalation (non-ROUTINE) resets consecutive counter."""
        state = HazardHysteresis(
            current_level=HazardLevel.CRITICAL,
            consecutive_routine=2,
        )
        # WARNING (not ROUTINE) -> resets counter
        level, new_state = apply_hysteresis(HazardLevel.WARNING, state)
        assert new_state.consecutive_routine == 0

    def test_custom_required_routine_count(self):
        """Custom required_routine_count is respected."""
        state = HazardHysteresis(
            current_level=HazardLevel.WARNING,
            consecutive_routine=4,
            required_routine_count=5,
        )
        # 4 + 1 = 5 >= 5 -> de-escalation permitted
        level, new_state = apply_hysteresis(HazardLevel.ROUTINE, state)
        assert level == HazardLevel.ROUTINE
        assert new_state.required_routine_count == 5

    def test_custom_required_routine_count_not_met(self):
        """Custom required_routine_count blocks de-escalation when not met."""
        state = HazardHysteresis(
            current_level=HazardLevel.WARNING,
            consecutive_routine=3,
            required_routine_count=5,
        )
        level, new_state = apply_hysteresis(HazardLevel.ROUTINE, state)
        assert level == HazardLevel.WARNING
        assert new_state.consecutive_routine == 4
        assert new_state.required_routine_count == 5

    def test_de_escalation_critical_to_routine_sustained(self):
        """CRITICAL -> ROUTINE after sustained ROUTINE period."""
        state = HazardHysteresis(
            current_level=HazardLevel.CRITICAL,
            consecutive_routine=2,
            required_routine_count=3,
        )
        level, new_state = apply_hysteresis(HazardLevel.ROUTINE, state)
        assert level == HazardLevel.ROUTINE
        assert new_state.current_level == HazardLevel.ROUTINE

    def test_same_level_stays(self):
        """Same level as current: no change."""
        state = HazardHysteresis(
            current_level=HazardLevel.WARNING,
            consecutive_routine=0,
        )
        level, new_state = apply_hysteresis(HazardLevel.WARNING, state)
        assert level == HazardLevel.WARNING
        assert new_state.current_level == HazardLevel.WARNING
        assert new_state.consecutive_routine == 0


# -- compute_breakup_potential -----------------------------------------------


class TestComputeBreakupPotential:

    def test_leo_velocity_catastrophic(self):
        """Typical LEO encounter (14 km/s) is catastrophic."""
        is_cat, energy = compute_breakup_potential(14000.0)
        assert is_cat is True
        assert energy == pytest.approx(0.5 * 14000.0 ** 2)

    def test_low_velocity_not_catastrophic(self):
        """Low relative velocity (100 m/s) is not catastrophic."""
        is_cat, energy = compute_breakup_potential(100.0)
        assert is_cat is False
        assert energy == pytest.approx(0.5 * 100.0 ** 2)

    def test_boundary_exactly_at_threshold(self):
        """Velocity exactly at threshold: V = sqrt(2*40000) is not > threshold."""
        threshold = math.sqrt(2.0 * 40000.0)
        is_cat, energy = compute_breakup_potential(threshold)
        assert is_cat is False  # > not >=
        assert energy == pytest.approx(40000.0)

    def test_boundary_just_above_threshold(self):
        """Velocity just above threshold is catastrophic."""
        threshold = math.sqrt(2.0 * 40000.0)
        is_cat, energy = compute_breakup_potential(threshold + 0.01)
        assert is_cat is True

    def test_threshold_value(self):
        """Catastrophic breakup velocity threshold is ~283 m/s."""
        assert _CATASTROPHIC_BREAKUP_VELOCITY_MS == pytest.approx(
            math.sqrt(80000.0), rel=1e-10,
        )

    def test_energy_formula(self):
        """Specific energy is 0.5 * v^2."""
        _, energy = compute_breakup_potential(500.0)
        assert energy == pytest.approx(125000.0)

    def test_zero_velocity(self):
        """Zero relative velocity: not catastrophic, zero energy."""
        is_cat, energy = compute_breakup_potential(0.0)
        assert is_cat is False
        assert energy == 0.0


# -- compute_conjunction_weather_index ---------------------------------------


class TestComputeConjunctionWeatherIndex:

    def test_high_pc_high_cwi(self):
        """High Pc (1e-3) produces high CWI."""
        cwi = compute_conjunction_weather_index(
            pc=1e-3, miss_distance_m=500.0, relative_velocity_ms=14000.0,
        )
        assert cwi > 0.5

    def test_low_pc_low_cwi(self):
        """Low Pc (1e-9) with far miss produces low CWI."""
        cwi = compute_conjunction_weather_index(
            pc=1e-9, miss_distance_m=100000.0, relative_velocity_ms=14000.0,
        )
        assert cwi < 0.2

    def test_zero_pc(self):
        """Zero Pc produces zero Pc sub-index contribution."""
        cwi = compute_conjunction_weather_index(
            pc=0.0, miss_distance_m=100000.0, relative_velocity_ms=14000.0,
        )
        # Only miss_score contributes (small for 100 km), rest is zero
        assert cwi < 0.1

    def test_no_covariance_increases_cwi(self):
        """Missing covariance increases the CWI."""
        cwi_with = compute_conjunction_weather_index(
            pc=1e-5, miss_distance_m=5000.0, relative_velocity_ms=14000.0,
            covariance_available=True,
        )
        cwi_without = compute_conjunction_weather_index(
            pc=1e-5, miss_distance_m=5000.0, relative_velocity_ms=14000.0,
            covariance_available=False,
        )
        assert cwi_without > cwi_with

    def test_secondary_maneuver_increases_cwi(self):
        """Secondary maneuver detection increases the CWI."""
        cwi_no_maneuver = compute_conjunction_weather_index(
            pc=1e-5, miss_distance_m=5000.0, relative_velocity_ms=14000.0,
            secondary_maneuver_detected=False,
        )
        cwi_maneuver = compute_conjunction_weather_index(
            pc=1e-5, miss_distance_m=5000.0, relative_velocity_ms=14000.0,
            secondary_maneuver_detected=True,
        )
        assert cwi_maneuver > cwi_no_maneuver

    def test_cwi_range_zero_to_one(self):
        """CWI is always in range [0, 1]."""
        # Extreme high-risk scenario
        cwi_max = compute_conjunction_weather_index(
            pc=1.0, miss_distance_m=1.0, relative_velocity_ms=14000.0,
            covariance_available=False,
            secondary_maneuver_detected=True,
            cascade_k_eff=10.0,
        )
        assert 0.0 <= cwi_max <= 1.0

        # Extreme low-risk scenario
        cwi_min = compute_conjunction_weather_index(
            pc=0.0, miss_distance_m=1e6, relative_velocity_ms=0.0,
            covariance_available=True,
            secondary_maneuver_detected=False,
            cascade_k_eff=0.0,
        )
        assert 0.0 <= cwi_min <= 1.0

    def test_cascade_k_eff_increases_cwi(self):
        """Higher cascade k_eff increases the CWI."""
        cwi_low = compute_conjunction_weather_index(
            pc=1e-5, miss_distance_m=5000.0, relative_velocity_ms=14000.0,
            cascade_k_eff=0.0,
        )
        cwi_high = compute_conjunction_weather_index(
            pc=1e-5, miss_distance_m=5000.0, relative_velocity_ms=14000.0,
            cascade_k_eff=1.5,
        )
        assert cwi_high > cwi_low

    def test_cascade_k_eff_capped_at_2(self):
        """Cascade score saturates at k_eff=2.0."""
        cwi_at_2 = compute_conjunction_weather_index(
            pc=1e-5, miss_distance_m=5000.0, relative_velocity_ms=14000.0,
            cascade_k_eff=2.0,
        )
        cwi_at_100 = compute_conjunction_weather_index(
            pc=1e-5, miss_distance_m=5000.0, relative_velocity_ms=14000.0,
            cascade_k_eff=100.0,
        )
        assert cwi_at_2 == pytest.approx(cwi_at_100)

    def test_combined_no_covariance_and_maneuver(self):
        """No covariance + secondary maneuver -> uncertainty_score = 1.0."""
        cwi = compute_conjunction_weather_index(
            pc=0.0, miss_distance_m=100000.0, relative_velocity_ms=14000.0,
            covariance_available=False,
            secondary_maneuver_detected=True,
        )
        # uncertainty_score = 0.5 + 0.5 = 1.0, contributes 0.25 * 1.0
        assert cwi >= 0.25


# -- sign_report -------------------------------------------------------------


class TestSignReport:

    def test_basic_report_structure(self):
        """sign_report returns a valid HazardReport."""
        event = _make_event(pc=1e-8, miss_m=50000.0)
        report = sign_report(
            event,
            force_models=("TwoBodyGravity", "J2Perturbation"),
            nutation_model="IAU_2000A",
        )
        assert isinstance(report, HazardReport)
        assert report.level == HazardLevel.ROUTINE
        assert report.conjunction_event is event
        assert report.force_models_used == ("TwoBodyGravity", "J2Perturbation")
        assert report.nutation_model == "IAU_2000A"
        assert len(report.provenance_hash) == 64  # SHA-256 hex

    def test_new_fields_populated(self):
        """sign_report populates all new fields correctly."""
        event = _make_event(pc=1e-3, miss_m=500.0, rel_vel=14000.0)
        report = sign_report(
            event,
            force_models=("TwoBodyGravity",),
            secondary_maneuver_detected=True,
            covariance_available=False,
            cascade_k_eff=1.2,
        )
        assert report.is_catastrophic_breakup is True
        assert report.breakup_specific_energy_j_per_kg == pytest.approx(
            0.5 * 14000.0 ** 2,
        )
        assert 0.0 <= report.conjunction_weather_index <= 1.0
        assert report.covariance_available is False
        assert report.secondary_maneuver_detected is True

    def test_provenance_hash_deterministic(self):
        """Same inputs produce the same provenance hash."""
        event = _make_event(pc=1e-5, miss_m=3000.0)
        r1 = sign_report(event, ("TwoBodyGravity",), timestamp=EPOCH)
        r2 = sign_report(event, ("TwoBodyGravity",), timestamp=EPOCH)
        assert r1.provenance_hash == r2.provenance_hash

    def test_provenance_hash_changes_with_models(self):
        """Different force models produce different hashes."""
        event = _make_event(pc=1e-5, miss_m=3000.0)
        r1 = sign_report(event, ("TwoBodyGravity",), timestamp=EPOCH)
        r2 = sign_report(event, ("TwoBodyGravity", "J2Perturbation"), timestamp=EPOCH)
        assert r1.provenance_hash != r2.provenance_hash

    def test_hmac_key_changes_hash(self):
        """HMAC key produces a different hash than plain SHA-256."""
        event = _make_event(pc=1e-5, miss_m=3000.0)
        r_plain = sign_report(event, ("TwoBodyGravity",), timestamp=EPOCH)
        r_hmac = sign_report(
            event, ("TwoBodyGravity",), timestamp=EPOCH,
            hmac_key=b"secret-key",
        )
        assert r_plain.provenance_hash != r_hmac.provenance_hash
        assert len(r_hmac.provenance_hash) == 64  # HMAC-SHA256 is also 64 hex

    def test_different_hmac_keys_produce_different_hashes(self):
        """Different HMAC keys produce different hashes."""
        event = _make_event(pc=1e-5, miss_m=3000.0)
        r1 = sign_report(
            event, ("TwoBodyGravity",), timestamp=EPOCH,
            hmac_key=b"key-alpha",
        )
        r2 = sign_report(
            event, ("TwoBodyGravity",), timestamp=EPOCH,
            hmac_key=b"key-beta",
        )
        assert r1.provenance_hash != r2.provenance_hash

    def test_hmac_deterministic(self):
        """HMAC hash is deterministic for same inputs and key."""
        event = _make_event(pc=1e-5, miss_m=3000.0)
        r1 = sign_report(
            event, ("TwoBodyGravity",), timestamp=EPOCH,
            hmac_key=b"my-key",
        )
        r2 = sign_report(
            event, ("TwoBodyGravity",), timestamp=EPOCH,
            hmac_key=b"my-key",
        )
        assert r1.provenance_hash == r2.provenance_hash

    def test_report_is_frozen(self):
        """HazardReport is immutable."""
        event = _make_event()
        report = sign_report(event, ("TwoBodyGravity",))
        with pytest.raises(AttributeError):
            report.level = HazardLevel.CRITICAL

    def test_recommended_action_routine(self):
        """ROUTINE level gets telemetry log action."""
        event = _make_event(pc=1e-8, miss_m=50000.0)
        report = sign_report(event, ("TwoBodyGravity",))
        assert "telemetry" in report.recommended_action.lower()

    def test_recommended_action_warning(self):
        """WARNING level gets COLA report action."""
        event = _make_event(pc=5e-6, miss_m=10000.0)
        report = sign_report(event, ("TwoBodyGravity",))
        assert "COLA" in report.recommended_action

    def test_recommended_action_critical(self):
        """CRITICAL level gets HAZARD ALERT action."""
        event = _make_event(pc=2e-4, miss_m=50000.0)
        report = sign_report(event, ("TwoBodyGravity",))
        assert "HAZARD ALERT" in report.recommended_action

    def test_recommended_action_critical_with_maneuver(self):
        """CRITICAL with maneuver plan says to execute it."""
        event = _make_event(pc=2e-4, miss_m=50000.0)
        report = sign_report(event, ("TwoBodyGravity",), has_maneuver_plan=True)
        assert "Execute" in report.recommended_action or "execute" in report.recommended_action

    def test_recommended_action_covariance_unavailable(self):
        """Covariance unavailable appears in recommended action."""
        event = _make_event(pc=0.0, miss_m=5000.0)
        report = sign_report(
            event, ("TwoBodyGravity",), covariance_available=False,
        )
        assert "Covariance unavailable" in report.recommended_action
        assert "OD" in report.recommended_action

    def test_recommended_action_secondary_maneuver(self):
        """Secondary maneuver detection appears in recommended action."""
        event = _make_event(pc=5e-6, miss_m=8000.0)
        report = sign_report(
            event, ("TwoBodyGravity",),
            secondary_maneuver_detected=True,
        )
        assert "Secondary maneuver" in report.recommended_action
        assert "prediction unreliable" in report.recommended_action.lower()

    def test_recommended_action_both_context_flags(self):
        """Both covariance and maneuver context appear together."""
        event = _make_event(pc=0.0, miss_m=5000.0)
        report = sign_report(
            event, ("TwoBodyGravity",),
            covariance_available=False,
            secondary_maneuver_detected=True,
        )
        assert "Covariance unavailable" in report.recommended_action
        assert "Secondary maneuver" in report.recommended_action

    def test_default_timestamp_is_tca(self):
        """Default timestamp is the event TCA."""
        event = _make_event()
        report = sign_report(event, ("TwoBodyGravity",))
        assert report.timestamp == event.tca

    def test_custom_timestamp(self):
        """Custom timestamp is used when provided."""
        event = _make_event()
        ts = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        report = sign_report(event, ("TwoBodyGravity",), timestamp=ts)
        assert report.timestamp == ts

    def test_breakup_fields_populated(self):
        """Breakup fields are set from compute_breakup_potential."""
        event = _make_event(rel_vel=14000.0)
        report = sign_report(event, ("TwoBodyGravity",))
        assert report.is_catastrophic_breakup is True
        assert report.breakup_specific_energy_j_per_kg > 0.0

    def test_cwi_field_populated(self):
        """Conjunction weather index is computed and stored."""
        event = _make_event(pc=1e-3, miss_m=500.0)
        report = sign_report(event, ("TwoBodyGravity",))
        assert 0.0 <= report.conjunction_weather_index <= 1.0
        assert report.conjunction_weather_index > 0.0

    def test_cascade_k_eff_affects_cwi(self):
        """cascade_k_eff parameter affects the CWI in the report."""
        event = _make_event(pc=1e-5, miss_m=5000.0)
        r_no_cascade = sign_report(
            event, ("TwoBodyGravity",), cascade_k_eff=0.0, timestamp=EPOCH,
        )
        r_cascade = sign_report(
            event, ("TwoBodyGravity",), cascade_k_eff=1.5, timestamp=EPOCH,
        )
        assert r_cascade.conjunction_weather_index > r_no_cascade.conjunction_weather_index


# -- Domain purity -----------------------------------------------------------


class TestHazardReportingPurity:

    def test_module_pure(self):
        """hazard_reporting.py only imports stdlib + numpy + domain."""
        import humeris.domain.hazard_reporting as mod

        allowed = {
            'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum',
            '__future__', 'datetime', 'hashlib', 'json', 'hmac',
        }
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('humeris'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'humeris':
                        assert False, f"Disallowed import from '{node.module}'"
