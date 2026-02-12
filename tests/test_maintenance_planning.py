# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for maintenance planning compositions."""
import ast
import math
from datetime import datetime, timezone

import pytest

from constellation_generator.domain.maintenance_planning import (
    ElementPerturbation,
    PerturbationBudget,
    MaintenanceBurn,
    MaintenanceSchedule,
    compute_perturbation_budget,
    compute_maintenance_schedule,
)
from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.atmosphere import DragConfig


_MU = 3.986004418e14
_R_E = 6371000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_DRAG = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)


def _state(alt_km=500.0, inc_deg=53.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a**3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(inc_deg), raan_rad=0.0,
        arg_perigee_rad=0.0, true_anomaly_rad=0.0,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


class TestPerturbationBudget:
    def test_returns_type(self):
        result = compute_perturbation_budget(_state(), _DRAG)
        assert isinstance(result, PerturbationBudget)

    def test_has_elements(self):
        result = compute_perturbation_budget(_state(), _DRAG)
        names = [e.element_name for e in result.elements]
        assert "raan" in names
        assert "arg_perigee" in names
        assert "sma" in names

    def test_dominant_identified(self):
        result = compute_perturbation_budget(_state(), _DRAG)
        for e in result.elements:
            assert e.dominant_source in ("j2", "drag", "j3", "other")


class TestMaintenanceSchedule:
    def test_returns_type(self):
        result = compute_maintenance_schedule(
            _state(400.0), _DRAG, _EPOCH,
            altitude_tolerance_km=5.0, mission_duration_days=365.0,
        )
        assert isinstance(result, MaintenanceSchedule)

    def test_burns_exist(self):
        result = compute_maintenance_schedule(
            _state(400.0), _DRAG, _EPOCH,
            altitude_tolerance_km=5.0, mission_duration_days=365.0,
        )
        assert len(result.burns) > 0

    def test_dv_positive(self):
        result = compute_maintenance_schedule(
            _state(400.0), _DRAG, _EPOCH,
            altitude_tolerance_km=5.0, mission_duration_days=365.0,
        )
        assert result.total_dv_per_year_ms > 0


class TestMaintenancePlanningPurity:
    def test_module_pure(self):
        import constellation_generator.domain.maintenance_planning as mod

        allowed = {'math', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('constellation_generator'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'constellation_generator':
                        assert False, f"Disallowed import from '{node.module}'"
