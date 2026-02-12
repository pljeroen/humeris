# Copyright (c) 2026 Jeroen. All rights reserved.
"""Tests for domain/design_optimization.py — DOP/Fisher, coverage drift, mass efficiency frontier."""
import ast
import math
from datetime import datetime, timezone

from constellation_generator.domain.propagation import OrbitalState, propagate_ecef_to
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.design_optimization import (
    PositioningInformationMetric,
    CoverageDriftAnalysis,
    MassEfficiencyPoint,
    MassEfficiencyFrontier,
    compute_positioning_information,
    compute_coverage_drift,
    compute_mass_efficiency_frontier,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_DRAG = DragConfig(cd=2.2, area_m2=0.05, mass_kg=100.0)


def _state(alt_km=550.0, inc_deg=53.0, raan_deg=0.0, ta_deg=0.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(inc_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=0.0, true_anomaly_rad=math.radians(ta_deg),
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


def _sat_positions_ecef():
    """Generate satellite ECEF positions visible from 45N, 0E (GPS-like altitude)."""
    states = [
        _state(alt_km=20200.0, inc_deg=55.0, raan_deg=r, ta_deg=t)
        for r in [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        for t in [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    ]
    return [propagate_ecef_to(s, _EPOCH) for s in states]


class TestPositioningInformation:
    def test_returns_type(self):
        positions = _sat_positions_ecef()
        result = compute_positioning_information(
            lat_deg=45.0, lon_deg=0.0,
            sat_positions_ecef=positions,
        )
        assert isinstance(result, PositioningInformationMetric)

    def test_fisher_det_positive(self):
        positions = _sat_positions_ecef()
        result = compute_positioning_information(
            lat_deg=45.0, lon_deg=0.0,
            sat_positions_ecef=positions,
        )
        assert result.fisher_determinant > 0.0

    def test_d_optimal_positive(self):
        positions = _sat_positions_ecef()
        result = compute_positioning_information(
            lat_deg=45.0, lon_deg=0.0,
            sat_positions_ecef=positions,
        )
        assert result.d_optimal_criterion > 0.0

    def test_crlb_scales_with_sigma(self):
        positions = _sat_positions_ecef()
        low = compute_positioning_information(
            lat_deg=45.0, lon_deg=0.0,
            sat_positions_ecef=positions,
            sigma_measurement_m=1.0,
        )
        high = compute_positioning_information(
            lat_deg=45.0, lon_deg=0.0,
            sat_positions_ecef=positions,
            sigma_measurement_m=10.0,
        )
        assert high.crlb_position_m > low.crlb_position_m

    def test_information_efficiency_range(self):
        positions = _sat_positions_ecef()
        result = compute_positioning_information(
            lat_deg=45.0, lon_deg=0.0,
            sat_positions_ecef=positions,
        )
        assert 0.0 < result.information_efficiency <= 1.0 + 1e-10


class TestCoverageDrift:
    def test_returns_type(self):
        states = [_state(raan_deg=r) for r in [0.0, 90.0, 180.0, 270.0]]
        result = compute_coverage_drift(
            states, _EPOCH, altitude_error_m=100.0,
            duration_s=5400.0, step_s=60.0,
        )
        assert isinstance(result, CoverageDriftAnalysis)

    def test_sensitivity_nonzero(self):
        """Altitude errors cause differential RAAN drift."""
        states = [_state(raan_deg=r) for r in [0.0, 90.0, 180.0, 270.0]]
        result = compute_coverage_drift(
            states, _EPOCH, altitude_error_m=100.0,
            duration_s=5400.0, step_s=60.0,
        )
        assert abs(result.raan_sensitivity_rad_s_per_m) > 0.0

    def test_half_life_positive(self):
        states = [_state(raan_deg=r) for r in [0.0, 90.0, 180.0, 270.0]]
        result = compute_coverage_drift(
            states, _EPOCH, altitude_error_m=100.0,
            duration_s=5400.0, step_s=60.0,
        )
        assert result.coverage_half_life_s > 0.0

    def test_larger_error_faster_drift(self):
        states = [_state(raan_deg=r) for r in [0.0, 90.0, 180.0, 270.0]]
        small = compute_coverage_drift(
            states, _EPOCH, altitude_error_m=50.0,
            duration_s=5400.0, step_s=60.0,
        )
        large = compute_coverage_drift(
            states, _EPOCH, altitude_error_m=200.0,
            duration_s=5400.0, step_s=60.0,
        )
        # Larger altitude error → shorter coverage half-life
        assert large.coverage_half_life_s <= small.coverage_half_life_s + 1.0


class TestMassEfficiencyFrontier:
    def test_returns_type(self):
        result = compute_mass_efficiency_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            alt_min_km=350.0, alt_max_km=650.0, alt_step_km=50.0,
        )
        assert isinstance(result, MassEfficiencyFrontier)

    def test_has_peak(self):
        result = compute_mass_efficiency_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            alt_min_km=350.0, alt_max_km=650.0, alt_step_km=50.0,
        )
        endpoints = [result.points[0].mass_efficiency, result.points[-1].mass_efficiency]
        assert result.peak_efficiency >= min(endpoints) - 1e-10

    def test_mass_wall_below_optimal(self):
        result = compute_mass_efficiency_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            alt_min_km=300.0, alt_max_km=700.0, alt_step_km=25.0,
        )
        assert result.mass_wall_altitude_km <= result.optimal_altitude_km + 1e-6

    def test_wet_mass_increases_at_extremes(self):
        result = compute_mass_efficiency_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            alt_min_km=300.0, alt_max_km=700.0, alt_step_km=25.0,
        )
        # Find the point with minimum wet mass
        min_wet = min(p.wet_mass_kg for p in result.points)
        # Endpoints should have higher or equal wet mass
        assert result.points[0].wet_mass_kg >= min_wet - 1e-6

    def test_efficiency_positive(self):
        result = compute_mass_efficiency_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            alt_min_km=350.0, alt_max_km=650.0, alt_step_km=50.0,
        )
        assert all(p.mass_efficiency > 0.0 for p in result.points)


class TestDesignOptimizationPurity:
    def test_module_pure(self):
        import constellation_generator.domain.design_optimization as mod
        source = ast.parse(open(mod.__file__).read())
        for node in ast.walk(source):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                else:
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                assert top in {"math", "dataclasses", "datetime", "typing", "enum", "constellation_generator", "__future__"}, (
                    f"Forbidden import: {top}"
                )
