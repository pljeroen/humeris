# Copyright (c) 2026 Jeroen. All rights reserved.
"""Tests for domain/constellation_operability.py — COI and common-cause failure detection."""
import ast
import math
from datetime import datetime, timezone

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.link_budget import LinkConfig
from constellation_generator.domain.atmosphere import DragConfig

from constellation_generator.domain.constellation_operability import (
    ConstellationOperabilityIndex,
    CommonCauseFailureResult,
    compute_operability_index,
    compute_common_cause_failure,
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


class TestConstellationOperabilityIndex:
    def test_returns_type(self):
        states = _small_constellation()
        result = compute_operability_index(
            states, _EPOCH, _LINK_CONFIG,
        )
        assert isinstance(result, ConstellationOperabilityIndex)

    def test_coi_bounded(self):
        states = _small_constellation()
        result = compute_operability_index(
            states, _EPOCH, _LINK_CONFIG,
        )
        assert 0.0 <= result.coi <= 1.0 + 1e-10

    def test_single_sat_not_operable(self):
        """Single satellite has no ISL graph → COI=0."""
        states = [_state()]
        result = compute_operability_index(
            states, _EPOCH, _LINK_CONFIG,
        )
        assert result.coi < 1e-10
        assert result.is_operable is False

    def test_factors_positive(self):
        states = _small_constellation()
        result = compute_operability_index(
            states, _EPOCH, _LINK_CONFIG,
        )
        assert result.connectivity_factor >= 0.0
        assert result.communication_factor >= 0.0
        assert result.controllability_factor >= 0.0


class TestCommonCauseFailure:
    def test_returns_type(self):
        states = _small_constellation()
        result = compute_common_cause_failure(
            states, _LINK_CONFIG, _EPOCH,
            _DRAG, isp_s=300.0, dry_mass_kg=100.0,
            propellant_budget_kg=20.0,
            duration_s=5400.0, step_s=600.0,
        )
        assert isinstance(result, CommonCauseFailureResult)

    def test_correlations_bounded(self):
        states = _small_constellation()
        result = compute_common_cause_failure(
            states, _LINK_CONFIG, _EPOCH,
            _DRAG, isp_s=300.0, dry_mass_kg=100.0,
            propellant_budget_kg=20.0,
            duration_s=5400.0, step_s=600.0,
        )
        assert -1.0 - 1e-10 <= result.fiedler_bec_correlation <= 1.0 + 1e-10
        assert -1.0 - 1e-10 <= result.fiedler_availability_correlation <= 1.0 + 1e-10

    def test_degradation_correlation_bounded(self):
        states = _small_constellation()
        result = compute_common_cause_failure(
            states, _LINK_CONFIG, _EPOCH,
            _DRAG, isp_s=300.0, dry_mass_kg=100.0,
            propellant_budget_kg=20.0,
            duration_s=5400.0, step_s=600.0,
        )
        assert -1.0 - 1e-10 <= result.degradation_correlation <= 1.0 + 1e-10


class TestConstellationOperabilityPurity:
    def test_module_pure(self):
        import constellation_generator.domain.constellation_operability as mod
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
