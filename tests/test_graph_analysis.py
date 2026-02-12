# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for domain/graph_analysis.py — ISL algebraic connectivity and fragmentation."""
import ast
import math
from datetime import datetime, timezone

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.link_budget import LinkConfig
from constellation_generator.domain.graph_analysis import (
    TopologyResilience,
    FragmentationTimeline,
    compute_topology_resilience,
    compute_fragmentation_timeline,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_LINK_CONFIG = LinkConfig(
    frequency_hz=26.5e9, transmit_power_w=10.0,
    tx_antenna_gain_dbi=35.0, rx_antenna_gain_dbi=35.0,
    system_noise_temp_k=500.0, bandwidth_hz=100e6,
    additional_losses_db=2.0, required_snr_db=10.0,
)


def _state(raan_deg=0.0, alt_km=550.0, inc_deg=53.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(inc_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=0.0, true_anomaly_rad=0.0,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


def _two_sat_states():
    """Two satellites in nearby planes — should be in range."""
    return [_state(raan_deg=0.0), _state(raan_deg=10.0)]


def _four_sat_states():
    """Four satellites spread across orbital planes."""
    return [_state(raan_deg=r) for r in [0.0, 30.0, 60.0, 90.0]]


class TestTopologyResilience:
    def test_returns_type(self):
        result = compute_topology_resilience(
            _two_sat_states(), _EPOCH, _LINK_CONFIG,
        )
        assert isinstance(result, TopologyResilience)

    def test_two_connected_sats_positive_fiedler(self):
        result = compute_topology_resilience(
            _two_sat_states(), _EPOCH, _LINK_CONFIG,
        )
        assert result.fiedler_value > 0.0

    def test_single_sat_zero_fiedler(self):
        result = compute_topology_resilience(
            [_state()], _EPOCH, _LINK_CONFIG,
        )
        assert result.fiedler_value == 0.0

    def test_fiedler_vector_length(self):
        states = _four_sat_states()
        result = compute_topology_resilience(
            states, _EPOCH, _LINK_CONFIG,
        )
        assert len(result.fiedler_vector) == len(states)

    def test_connected_graph_flag(self):
        result = compute_topology_resilience(
            _two_sat_states(), _EPOCH, _LINK_CONFIG,
        )
        assert result.is_connected is True

    def test_spectral_gap_nonneg(self):
        result = compute_topology_resilience(
            _four_sat_states(), _EPOCH, _LINK_CONFIG,
        )
        assert result.spectral_gap >= -1e-10


class TestFragmentationTimeline:
    def test_returns_type(self):
        result = compute_fragmentation_timeline(
            _two_sat_states(), _LINK_CONFIG, _EPOCH,
            duration_s=600.0, step_s=300.0,
        )
        assert isinstance(result, FragmentationTimeline)

    def test_event_count(self):
        result = compute_fragmentation_timeline(
            _two_sat_states(), _LINK_CONFIG, _EPOCH,
            duration_s=600.0, step_s=300.0,
        )
        expected = int(600.0 / 300.0) + 1
        assert len(result.events) == expected

    def test_min_fiedler_leq_mean(self):
        result = compute_fragmentation_timeline(
            _two_sat_states(), _LINK_CONFIG, _EPOCH,
            duration_s=600.0, step_s=300.0,
        )
        assert result.min_fiedler_value <= result.mean_fiedler_value + 1e-10

    def test_resilience_margin_range(self):
        result = compute_fragmentation_timeline(
            _two_sat_states(), _LINK_CONFIG, _EPOCH,
            duration_s=600.0, step_s=300.0,
        )
        assert result.resilience_margin <= 1.0 + 1e-10

    def test_fragmentation_time_set(self):
        result = compute_fragmentation_timeline(
            _two_sat_states(), _LINK_CONFIG, _EPOCH,
            duration_s=600.0, step_s=300.0,
        )
        assert isinstance(result.min_fiedler_time, datetime)


class TestGraphAnalysisPurity:
    def test_module_pure(self):
        import constellation_generator.domain.graph_analysis as mod
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
