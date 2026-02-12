"""Tests for coverage optimization compositions."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator.domain.coverage_optimization import (
    QualityCoveragePoint,
    QualityCoverageResult,
    CrossingRevisitResult,
    CompliantTradePoint,
    CompliantTradeResult,
    EOLTANPoint,
    EOLTANOptimization,
    StationCandidate,
    GroundStationOptimization,
    compute_quality_weighted_coverage,
    compute_crossing_revisit,
    compute_compliant_trade_study,
    compute_optimal_eo_ltan,
    compute_optimal_ground_stations,
)
from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.sensor import SensorConfig, SensorType
from constellation_generator.domain.trade_study import TradeStudyResult, TradePoint, WalkerConfig
from constellation_generator.domain.revisit import CoverageResult
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.ground_track import GroundTrackPoint


_MU = 3.986004418e14
_R_E = 6371000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_DRAG = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)


def _state(alt_km=500.0, inc_deg=53.0, raan_rad=0.0, ta_rad=0.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a**3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(inc_deg), raan_rad=raan_rad,
        arg_perigee_rad=0.0, true_anomaly_rad=ta_rad,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


_SENSOR = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)


class TestQualityWeightedCoverage:
    def test_returns_type(self):
        states = [_state(500.0, 53.0, i * 1.0, 0.0) for i in range(4)]
        result = compute_quality_weighted_coverage(
            states, _EPOCH, _SENSOR, lat_step_deg=45.0, lon_step_deg=90.0,
        )
        assert isinstance(result, QualityCoverageResult)

    def test_usable_fraction_range(self):
        states = [_state(500.0, 53.0, i * 1.0, 0.0) for i in range(4)]
        result = compute_quality_weighted_coverage(
            states, _EPOCH, _SENSOR, lat_step_deg=45.0, lon_step_deg=90.0,
        )
        assert 0.0 <= result.usable_fraction <= 1.0

    def test_gdop_nonnegative(self):
        states = [_state(500.0, 53.0, i * 1.0, 0.0) for i in range(4)]
        result = compute_quality_weighted_coverage(
            states, _EPOCH, _SENSOR, lat_step_deg=45.0, lon_step_deg=90.0,
        )
        assert result.mean_gdop >= 0.0


class TestCrossingRevisit:
    def _make_track(self):
        track = []
        for i in range(30):
            t = _EPOCH + timedelta(minutes=i * 6)
            lat = 53.0 * math.sin(2 * math.pi * i / 15)
            lon = -180.0 + (i * 12.0) % 360.0
            if lon > 180.0:
                lon -= 360.0
            track.append(GroundTrackPoint(time=t, lat_deg=lat, lon_deg=lon, alt_km=500.0))
        return track

    def test_returns_type(self):
        result = compute_crossing_revisit(
            self._make_track(), [_state()],
            _EPOCH, timedelta(hours=3), timedelta(minutes=10),
        )
        assert isinstance(result, CrossingRevisitResult)

    def test_improvement_nonnegative(self):
        result = compute_crossing_revisit(
            self._make_track(), [_state()],
            _EPOCH, timedelta(hours=3), timedelta(minutes=10),
        )
        assert result.improvement_factor >= 0.0

    def test_count_nonnegative(self):
        track = [GroundTrackPoint(time=_EPOCH, lat_deg=0.0, lon_deg=0.0, alt_km=500.0)]
        result = compute_crossing_revisit(
            track, [_state()], _EPOCH, timedelta(hours=1), timedelta(minutes=10),
        )
        assert result.num_crossings >= 0


class TestCompliantTradeStudy:
    def _trade_result(self):
        coverage = CoverageResult(
            analysis_duration_s=3600.0, num_grid_points=4, num_satellites=6,
            mean_coverage_fraction=0.7, min_coverage_fraction=0.3,
            mean_revisit_s=1200.0, max_revisit_s=2400.0,
            mean_response_time_s=600.0,
            percent_coverage_single=0.5, point_results=(),
        )
        config = WalkerConfig(
            altitude_km=500.0, inclination_deg=53.0,
            num_planes=3, sats_per_plane=2, phase_factor=1,
        )
        tp = TradePoint(config=config, total_satellites=6, coverage=coverage)
        return TradeStudyResult(
            points=(tp,), analysis_duration_s=3600.0, min_elevation_deg=10.0,
        )

    def test_returns_type(self):
        result = compute_compliant_trade_study(self._trade_result(), _DRAG, _EPOCH)
        assert isinstance(result, CompliantTradeResult)

    def test_compliant_leq_all(self):
        result = compute_compliant_trade_study(self._trade_result(), _DRAG, _EPOCH)
        assert result.compliant_count <= len(result.all_points)

    def test_pareto_valid(self):
        result = compute_compliant_trade_study(self._trade_result(), _DRAG, _EPOCH)
        for idx in result.pareto_indices:
            assert 0 <= idx < len(result.all_points)


class TestEOLTAN:
    def test_returns_type(self):
        result = compute_optimal_eo_ltan(
            altitude_km=500.0, epoch=_EPOCH, ltan_values=[6.0, 10.5, 14.0],
        )
        assert isinstance(result, EOLTANOptimization)

    def test_ltan_in_range(self):
        ltans = [6.0, 10.5, 14.0]
        result = compute_optimal_eo_ltan(altitude_km=500.0, epoch=_EPOCH, ltan_values=ltans)
        assert result.optimal_ltan_hours in ltans


class TestOptimalGroundStations:
    def _track(self):
        return [
            GroundTrackPoint(
                time=_EPOCH + timedelta(minutes=i),
                lat_deg=53.0 * math.sin(2 * math.pi * i / 90),
                lon_deg=((i * 4.0) % 360.0) - 180.0,
                alt_km=500.0,
            )
            for i in range(100)
        ]

    def test_returns_type(self):
        result = compute_optimal_ground_stations(
            self._track(), [_state()],
            _EPOCH, timedelta(hours=3), timedelta(minutes=10),
        )
        assert isinstance(result, GroundStationOptimization)

    def test_ranked_valid(self):
        result = compute_optimal_ground_stations(
            self._track(), [_state()],
            _EPOCH, timedelta(hours=3), timedelta(minutes=10),
        )
        for idx in result.ranked_indices:
            assert 0 <= idx < len(result.candidates)

    def test_contact_nonnegative(self):
        result = compute_optimal_ground_stations(
            self._track(), [_state()],
            _EPOCH, timedelta(hours=3), timedelta(minutes=10),
        )
        for c in result.candidates:
            assert c.total_contact_s >= 0


class TestCoverageOptimizationPurity:
    def test_module_pure(self):
        import constellation_generator.domain.coverage_optimization as mod

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
