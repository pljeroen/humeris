"""Tests for environment analysis compositions."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator.domain.environment_analysis import (
    ThermalMonth,
    SeasonalThermalProfile,
    DoseSnapshot,
    SeasonalDoseProfile,
    RadiationLTANPoint,
    RadiationLTANResult,
    EclipseFreeLTANPoint,
    EclipseFreeResult,
    TorqueBoundary,
    TorqueTimingResult,
    compute_seasonal_thermal_profile,
    compute_seasonal_dose_profile,
    compute_radiation_optimized_ltan,
    compute_eclipse_free_windows,
    compute_worst_case_torque_timing,
)
from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.torques import InertiaTensor


_MU = 3.986004418e14
_R_E = 6371000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _state(alt_km=500.0, inc_deg=53.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a**3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(inc_deg), raan_rad=0.0,
        arg_perigee_rad=0.0, true_anomaly_rad=0.0,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


class TestSeasonalThermalProfile:
    def test_returns_type(self):
        result = compute_seasonal_thermal_profile(_state(), _EPOCH)
        assert isinstance(result, SeasonalThermalProfile)

    def test_12_months(self):
        result = compute_seasonal_thermal_profile(_state(), _EPOCH)
        assert len(result.months) == 12

    def test_cycles_nonnegative(self):
        result = compute_seasonal_thermal_profile(_state(), _EPOCH)
        for m in result.months:
            assert m.cycle_count >= 0


class TestSeasonalDoseProfile:
    def test_returns_type(self):
        result = compute_seasonal_dose_profile(
            _state(), _EPOCH, duration_days=90.0, step_days=30.0,
        )
        assert isinstance(result, SeasonalDoseProfile)

    def test_dose_positive(self):
        result = compute_seasonal_dose_profile(
            _state(), _EPOCH, duration_days=90.0, step_days=30.0,
        )
        assert result.annual_dose_rad > 0

    def test_cumulative_increases(self):
        result = compute_seasonal_dose_profile(
            _state(), _EPOCH, duration_days=180.0, step_days=30.0,
        )
        doses = [s.cumulative_dose_rad for s in result.snapshots]
        for i in range(1, len(doses)):
            assert doses[i] >= doses[i - 1]


class TestRadiationLTAN:
    def test_returns_type(self):
        result = compute_radiation_optimized_ltan(
            500.0, _EPOCH, ltan_values=[6.0, 10.5, 18.0],
        )
        assert isinstance(result, RadiationLTANResult)

    def test_ltan_in_range(self):
        ltans = [6.0, 10.5, 18.0]
        result = compute_radiation_optimized_ltan(500.0, _EPOCH, ltan_values=ltans)
        assert result.optimal_ltan_hours in ltans


class TestEclipseFreeWindows:
    def test_returns_type(self):
        result = compute_eclipse_free_windows(
            500.0, _EPOCH, ltan_values=[6.0, 10.5, 18.0],
        )
        assert isinstance(result, EclipseFreeResult)

    def test_days_range(self):
        result = compute_eclipse_free_windows(
            500.0, _EPOCH, ltan_values=[6.0, 10.5, 18.0],
        )
        for p in result.points:
            assert 0.0 <= p.eclipse_free_days <= 366.0


class TestTorqueTiming:
    def test_returns_type(self):
        inertia = InertiaTensor(ixx=10.0, iyy=12.0, izz=8.0)
        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        result = compute_worst_case_torque_timing(
            _state(), inertia, drag,
            cp_offset_m=(0.01, 0.0, 0.0),
            epoch=_EPOCH, duration_s=5700.0, step_s=60.0,
        )
        assert isinstance(result, TorqueTimingResult)

    def test_discontinuity_nonnegative(self):
        inertia = InertiaTensor(ixx=10.0, iyy=12.0, izz=8.0)
        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        result = compute_worst_case_torque_timing(
            _state(), inertia, drag,
            cp_offset_m=(0.01, 0.0, 0.0),
            epoch=_EPOCH, duration_s=5700.0, step_s=60.0,
        )
        assert result.max_discontinuity_nm >= 0

    def test_boundaries_tuple(self):
        inertia = InertiaTensor(ixx=10.0, iyy=12.0, izz=8.0)
        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        result = compute_worst_case_torque_timing(
            _state(), inertia, drag,
            cp_offset_m=(0.01, 0.0, 0.0),
            epoch=_EPOCH, duration_s=5700.0, step_s=60.0,
        )
        assert isinstance(result.boundaries, tuple)


class TestEnvironmentAnalysisPurity:
    def test_module_pure(self):
        import constellation_generator.domain.environment_analysis as mod

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
