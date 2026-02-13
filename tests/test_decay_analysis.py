# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for domain/decay_analysis.py — exponential scale map extraction."""
import ast
import math
from datetime import datetime, timezone

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.atmosphere import DragConfig

from constellation_generator.domain.decay_analysis import (
    ExponentialProcess,
    ExponentialScaleMap,
    compute_exponential_scale_map,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _state(alt_km=400.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(53.0),
        raan_rad=0.0, arg_perigee_rad=0.0, true_anomaly_rad=0.0,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


def _drag():
    return DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)


class TestComputeExponentialScaleMap:
    def test_returns_type(self):
        result = compute_exponential_scale_map(
            _state(), _drag(), _EPOCH,
            isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert isinstance(result, ExponentialScaleMap)

    def test_scale_height_positive(self):
        result = compute_exponential_scale_map(
            _state(), _drag(), _EPOCH,
            isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        for p in result.processes:
            assert p.scale_parameter > 0.0, f"Scale parameter for {p.name} must be positive"

    def test_half_life_positive(self):
        result = compute_exponential_scale_map(
            _state(), _drag(), _EPOCH,
            isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        for p in result.processes:
            assert p.half_life > 0.0, f"Half-life for {p.name} must be positive"

    def test_ordering(self):
        result = compute_exponential_scale_map(
            _state(), _drag(), _EPOCH,
            isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert result.fastest_process is not None
        assert result.slowest_process is not None
        fast = next(p for p in result.processes if p.name == result.fastest_process)
        slow = next(p for p in result.processes if p.name == result.slowest_process)
        assert fast.half_life <= slow.half_life

    def test_scale_ratio_positive(self):
        result = compute_exponential_scale_map(
            _state(), _drag(), _EPOCH,
            isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert result.scale_ratio > 0.0

    def test_e_folding_equals_scale(self):
        """e-folding time = scale parameter for exp(-t/tau)."""
        result = compute_exponential_scale_map(
            _state(), _drag(), _EPOCH,
            isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        for p in result.processes:
            assert p.e_folding == p.scale_parameter


class TestDecayAnalysisPurity:
    def test_no_external_deps(self):
        import constellation_generator.domain.decay_analysis as mod
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())
        allowed = {"math", "numpy", "dataclasses", "datetime", "constellation_generator"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed, f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed, f"Forbidden import from: {node.module}"
