# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for conjunction management compositions."""
import ast
import math
from datetime import datetime, timezone

import pytest

from humeris.domain.conjunction_management import (
    ConjunctionAction,
    ConjunctionTriage,
    AvoidanceManeuver,
    ConjunctionDecision,
    triage_conjunction,
    compute_avoidance_maneuver,
    run_conjunction_decision_pipeline,
)
from humeris.domain.propagation import OrbitalState


_MU = 3.986004418e14
_R_E = 6371000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _state(alt_km=500.0, inc_deg=53.0, raan_rad=0.0, ta_rad=0.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a**3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(inc_deg), raan_rad=raan_rad,
        arg_perigee_rad=0.0, true_anomaly_rad=ta_rad,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


class TestTriageConjunction:
    def test_returns_type(self):
        s1 = _state(500.0, 53.0, 0.0, 0.0)
        s2 = _state(500.0, 53.0, 0.01, 0.0)
        result = triage_conjunction(s1, s2, _EPOCH)
        assert isinstance(result, ConjunctionTriage)

    def test_safe_no_action(self):
        s1 = _state(500.0, 53.0, 0.0, 0.0)
        s2 = _state(500.0, 53.0, 0.5, 0.0)
        result = triage_conjunction(s1, s2, _EPOCH, min_safe_distance_m=100.0)
        assert result.action == ConjunctionAction.NO_ACTION

    def test_close_maneuver(self):
        s1 = _state(500.0, 53.0, 0.0, 0.0)
        s2 = _state(500.0, 53.0, 0.0, 0.0001)
        result = triage_conjunction(s1, s2, _EPOCH, min_safe_distance_m=100_000.0)
        assert result.action == ConjunctionAction.MANEUVER


class TestAvoidanceManeuver:
    def test_returns_type(self):
        s1 = _state(500.0, 53.0, 0.0, 0.0)
        s2 = _state(500.0, 53.0, 0.0, 0.001)
        result = compute_avoidance_maneuver(s1, s2, _EPOCH)
        assert isinstance(result, AvoidanceManeuver)

    def test_dv_positive(self):
        s1 = _state(500.0, 53.0, 0.0, 0.0)
        s2 = _state(500.0, 53.0, 0.0, 0.001)
        result = compute_avoidance_maneuver(s1, s2, _EPOCH)
        assert result.delta_v_ms > 0

    def test_larger_miss_more_dv(self):
        s1 = _state(500.0, 53.0, 0.0, 0.0)
        s2 = _state(500.0, 53.0, 0.0, 0.001)
        r_small = compute_avoidance_maneuver(s1, s2, _EPOCH, target_miss_m=500.0)
        r_large = compute_avoidance_maneuver(s1, s2, _EPOCH, target_miss_m=5000.0)
        assert r_large.delta_v_ms >= r_small.delta_v_ms


class TestDecisionPipeline:
    def test_returns_list(self):
        result = run_conjunction_decision_pipeline([], {}, {})
        assert isinstance(result, list)

    def test_empty_events(self):
        result = run_conjunction_decision_pipeline([], {}, {})
        assert len(result) == 0


class TestConjunctionManagementPurity:
    def test_module_pure(self):
        import humeris.domain.conjunction_management as mod

        allowed = {'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
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
