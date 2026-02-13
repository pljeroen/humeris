# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for communication analysis compositions."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.communication_analysis import (
    DegradedLink,
    EclipseDegradedTopology,
    PassDataPoint,
    PassThroughput,
    ISLDistancePrediction,
    NetworkCapacitySnapshot,
    NetworkCapacityTimeline,
    compute_eclipse_degraded_topology,
    compute_pass_data_throughput,
    predict_isl_distances,
    compute_network_capacity_timeline,
)
from humeris.domain.propagation import OrbitalState
from humeris.domain.observation import GroundStation
from humeris.domain.link_budget import LinkConfig
from humeris.domain.access_windows import AccessWindow


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


_LINK_CONFIG = LinkConfig(
    frequency_hz=26.5e9, transmit_power_w=10.0,
    tx_antenna_gain_dbi=35.0, rx_antenna_gain_dbi=35.0,
    system_noise_temp_k=500.0, bandwidth_hz=100e6,
    additional_losses_db=2.0, required_snr_db=10.0,
)

_STATION = GroundStation(name="Test", lat_deg=52.0, lon_deg=5.0, alt_m=0.0)


class TestEclipseDegradedTopology:
    def test_returns_type(self):
        states = [_state(500.0, 53.0, i * 0.5, 0.0) for i in range(4)]
        result = compute_eclipse_degraded_topology(states, _EPOCH, _LINK_CONFIG)
        assert isinstance(result, EclipseDegradedTopology)

    def test_capacity_leq_full(self):
        states = [_state(500.0, 53.0, i * 0.5, 0.0) for i in range(4)]
        result = compute_eclipse_degraded_topology(states, _EPOCH, _LINK_CONFIG)
        assert result.capacity_fraction <= 1.001

    def test_counts_consistent(self):
        states = [_state(500.0, 53.0, i * 0.5, 0.0) for i in range(4)]
        result = compute_eclipse_degraded_topology(states, _EPOCH, _LINK_CONFIG)
        assert result.active_link_count + result.degraded_link_count <= result.total_link_count


class TestPassDataThroughput:
    def test_returns_type(self):
        window = AccessWindow(
            rise_time=_EPOCH,
            set_time=_EPOCH + timedelta(minutes=6),
            max_elevation_deg=45.0,
            duration_seconds=360.0,
        )
        result = compute_pass_data_throughput(
            _STATION, _state(), window, _LINK_CONFIG, freq_hz=2.2e9, step_s=60.0,
        )
        assert isinstance(result, PassThroughput)

    def test_bytes_positive(self):
        window = AccessWindow(
            rise_time=_EPOCH,
            set_time=_EPOCH + timedelta(minutes=6),
            max_elevation_deg=45.0,
            duration_seconds=360.0,
        )
        result = compute_pass_data_throughput(
            _STATION, _state(), window, _LINK_CONFIG, freq_hz=2.2e9, step_s=60.0,
        )
        assert result.total_bytes >= 0

    def test_effective_rate_leq_one(self):
        window = AccessWindow(
            rise_time=_EPOCH,
            set_time=_EPOCH + timedelta(minutes=6),
            max_elevation_deg=45.0,
            duration_seconds=360.0,
        )
        result = compute_pass_data_throughput(
            _STATION, _state(), window, _LINK_CONFIG, freq_hz=2.2e9, step_s=60.0,
        )
        assert result.effective_rate_fraction <= 1.01


class TestISLDistances:
    def test_returns_type(self):
        result = predict_isl_distances([0.0, 5.0, 10.0, 15.0], altitude_km=550.0)
        assert isinstance(result, ISLDistancePrediction)

    def test_distances_positive(self):
        result = predict_isl_distances([0.0, 5.0, 10.0, 15.0], altitude_km=550.0)
        assert all(d > 0 for d in result.predicted_distances_m)

    def test_spacing_matches(self):
        result = predict_isl_distances([0.0, 5.0, 10.0, 15.0], altitude_km=550.0)
        assert abs(result.node_spacing_deg - 5.0) < 0.01


class TestNetworkCapacityTimeline:
    def test_returns_type(self):
        states = [_state(500.0, 53.0, i * 0.5, 0.0) for i in range(3)]
        result = compute_network_capacity_timeline(
            states, [_STATION], _LINK_CONFIG, _EPOCH,
            duration_s=600.0, step_s=300.0,
        )
        assert isinstance(result, NetworkCapacityTimeline)

    def test_snapshot_count(self):
        states = [_state(500.0, 53.0, i * 0.5, 0.0) for i in range(3)]
        result = compute_network_capacity_timeline(
            states, [_STATION], _LINK_CONFIG, _EPOCH,
            duration_s=600.0, step_s=300.0,
        )
        assert len(result.snapshots) >= 2


class TestCommunicationAnalysisPurity:
    def test_module_pure(self):
        import humeris.domain.communication_analysis as mod

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
