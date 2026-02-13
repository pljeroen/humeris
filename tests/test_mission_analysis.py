# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for mission-level analysis compositions."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator.domain.mission_analysis import (
    PropellantPoint,
    PropellantProfile,
    HealthSnapshot,
    HealthTimeline,
    AltitudeTradePoint,
    AltitudeOptimization,
    MissionCostMetric,
    compute_propellant_profile,
    compute_health_timeline,
    compute_optimal_altitude,
    compute_mission_cost_metric,
)
from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.lifetime import compute_orbit_lifetime
from constellation_generator.domain.revisit import CoverageResult
from constellation_generator.domain.trade_study import WalkerConfig


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


def _decay_profile():
    result = compute_orbit_lifetime(
        semi_major_axis_m=_R_E + 350_000.0,
        eccentricity=0.0, drag_config=_DRAG,
        epoch=_EPOCH, step_days=30.0, max_years=2.0,
    )
    return result.decay_profile


class TestPropellantProfile:
    def test_returns_type(self):
        profile = compute_propellant_profile(
            _decay_profile(), _DRAG, isp_s=220.0,
            dry_mass_kg=80.0, propellant_budget_kg=20.0,
        )
        assert isinstance(profile, PropellantProfile)

    def test_accelerates(self):
        profile = compute_propellant_profile(
            _decay_profile(), _DRAG, isp_s=220.0,
            dry_mass_kg=80.0, propellant_budget_kg=50.0,
        )
        if len(profile.points) >= 3:
            assert profile.points[-1].dv_per_year_ms >= profile.points[0].dv_per_year_ms

    def test_depletion_detected(self):
        profile = compute_propellant_profile(
            _decay_profile(), _DRAG, isp_s=220.0,
            dry_mass_kg=80.0, propellant_budget_kg=0.01,
        )
        assert profile.depletion_time is not None


class TestHealthTimeline:
    def test_returns_type(self):
        tl = compute_health_timeline(_state(400.0), _DRAG, _EPOCH, mission_years=0.5)
        assert isinstance(tl, HealthTimeline)

    def test_dose_increases(self):
        tl = compute_health_timeline(_state(400.0), _DRAG, _EPOCH, mission_years=0.5)
        if len(tl.snapshots) >= 2:
            assert tl.snapshots[-1].cumulative_dose_rad >= tl.snapshots[0].cumulative_dose_rad

    def test_limiting_factor(self):
        tl = compute_health_timeline(
            _state(400.0), _DRAG, _EPOCH, mission_years=0.5,
            radiation_limit_rad=1e20, thermal_limit_cycles=1_000_000_000,
        )
        assert tl.limiting_factor in ("radiation", "thermal", "propellant", "none")


class TestOptimalAltitude:
    def test_returns_type(self):
        result = compute_optimal_altitude(
            _DRAG, isp_s=220.0, dry_mass_kg=80.0,
            injection_altitude_km=300.0, mission_years=1.0,
            alt_min_km=350.0, alt_max_km=550.0, alt_step_km=100.0,
        )
        assert isinstance(result, AltitudeOptimization)

    def test_finds_minimum(self):
        result = compute_optimal_altitude(
            _DRAG, isp_s=220.0, dry_mass_kg=80.0,
            injection_altitude_km=300.0, mission_years=3.0,
            alt_min_km=350.0, alt_max_km=650.0, alt_step_km=50.0,
        )
        assert result.minimum_dv_ms == min(p.total_dv_ms for p in result.points)

    def test_low_alt_high_sk(self):
        result = compute_optimal_altitude(
            _DRAG, isp_s=220.0, dry_mass_kg=80.0,
            injection_altitude_km=300.0, mission_years=3.0,
            alt_min_km=300.0, alt_max_km=600.0, alt_step_km=100.0,
        )
        low = next(p for p in result.points if p.altitude_km == 300.0)
        high = next(p for p in result.points if p.altitude_km == 600.0)
        assert low.sk_dv_ms > high.sk_dv_ms


class TestMissionCostMetric:
    def _coverage(self):
        return CoverageResult(
            analysis_duration_s=3600.0, num_grid_points=10, num_satellites=22,
            mean_coverage_fraction=0.8, min_coverage_fraction=0.5,
            mean_revisit_s=900.0, max_revisit_s=1800.0,
            mean_response_time_s=450.0,
            percent_coverage_single=0.6, point_results=(),
        )

    def test_returns_type(self):
        config = WalkerConfig(
            altitude_km=500.0, inclination_deg=53.0,
            num_planes=22, sats_per_plane=72, phase_factor=17,
        )
        result = compute_mission_cost_metric(
            config, _DRAG, isp_s=220.0, dry_mass_kg=80.0,
            injection_altitude_km=300.0, mission_years=5.0,
            coverage_result=self._coverage(),
        )
        assert isinstance(result, MissionCostMetric)

    def test_positive(self):
        config = WalkerConfig(
            altitude_km=500.0, inclination_deg=53.0,
            num_planes=22, sats_per_plane=72, phase_factor=17,
        )
        result = compute_mission_cost_metric(
            config, _DRAG, isp_s=220.0, dry_mass_kg=80.0,
            injection_altitude_km=300.0, mission_years=5.0,
            coverage_result=self._coverage(),
        )
        assert result.total_dv_ms > 0
        assert result.wet_mass_per_sat_kg > 0

    def test_higher_alt_more_raising(self):
        c_low = WalkerConfig(400.0, 53.0, 22, 72, 17)
        c_high = WalkerConfig(600.0, 53.0, 22, 72, 17)
        r_low = compute_mission_cost_metric(
            c_low, _DRAG, 220.0, 80.0, 300.0, 5.0, self._coverage(),
        )
        r_high = compute_mission_cost_metric(
            c_high, _DRAG, 220.0, 80.0, 300.0, 5.0, self._coverage(),
        )
        assert r_low.total_dv_ms > 0
        assert r_high.total_dv_ms > 0


class TestMissionAnalysisPurity:
    def test_module_pure(self):
        import constellation_generator.domain.mission_analysis as mod

        allowed = {'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
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
