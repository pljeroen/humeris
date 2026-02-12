# Copyright (c) 2026 Jeroen. All rights reserved.
"""Tests for domain/cascade_analysis.py — failure cascade prediction."""
import ast
import math
from datetime import datetime, timezone

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.link_budget import LinkConfig
from constellation_generator.domain.atmosphere import DragConfig

from constellation_generator.domain.cascade_analysis import (
    CascadeIndicator,
    compute_cascade_indicator,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_LINK_CONFIG = LinkConfig(
    frequency_hz=26e9, transmit_power_w=1.0,
    tx_antenna_gain_dbi=30.0, rx_antenna_gain_dbi=30.0,
    system_noise_temp_k=300.0, bandwidth_hz=100e6,
)
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


def _small_constellation():
    return [
        _state(raan_deg=0.0, ta_deg=0.0),
        _state(raan_deg=0.0, ta_deg=180.0),
        _state(raan_deg=90.0, ta_deg=0.0),
        _state(raan_deg=90.0, ta_deg=180.0),
    ]


class TestCascadeIndicator:
    def test_returns_type(self):
        states = _small_constellation()
        result = compute_cascade_indicator(
            states, _LINK_CONFIG, _EPOCH, _DRAG,
            fragmentation_duration_s=5400.0,
            fragmentation_step_s=300.0,
        )
        assert isinstance(result, CascadeIndicator)

    def test_cascade_nonnegative(self):
        states = _small_constellation()
        result = compute_cascade_indicator(
            states, _LINK_CONFIG, _EPOCH, _DRAG,
            fragmentation_duration_s=5400.0,
            fragmentation_step_s=300.0,
        )
        assert result.cascade_indicator >= 0.0

    def test_orbital_frequency_positive(self):
        states = _small_constellation()
        result = compute_cascade_indicator(
            states, _LINK_CONFIG, _EPOCH, _DRAG,
            fragmentation_duration_s=5400.0,
            fragmentation_step_s=300.0,
        )
        assert result.orbital_frequency_hz > 0.0

    def test_hazard_peakiness_geq_one(self):
        """max/mean >= 1 by definition."""
        states = _small_constellation()
        result = compute_cascade_indicator(
            states, _LINK_CONFIG, _EPOCH, _DRAG,
            fragmentation_duration_s=5400.0,
            fragmentation_step_s=300.0,
        )
        assert result.hazard_peakiness >= 1.0 - 1e-10

    def test_spectral_power_nonnegative(self):
        states = _small_constellation()
        result = compute_cascade_indicator(
            states, _LINK_CONFIG, _EPOCH, _DRAG,
            fragmentation_duration_s=5400.0,
            fragmentation_step_s=300.0,
        )
        assert result.spectral_power_at_orbital >= 0.0


class TestCascadeAnalysisPurity:
    def test_module_pure(self):
        import constellation_generator.domain.cascade_analysis as mod
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
