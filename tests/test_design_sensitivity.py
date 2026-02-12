# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for domain/design_sensitivity.py — spectral fragility, coverage-connectivity, altitude sensitivity."""
import ast
import math
from datetime import datetime, timezone, timedelta

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.link_budget import LinkConfig

from constellation_generator.domain.design_sensitivity import (
    SpectralFragility,
    CoverageConnectivityPoint,
    CoverageConnectivityCrossover,
    AltitudeSensitivity,
    compute_spectral_fragility,
    compute_coverage_connectivity_crossover,
    compute_altitude_sensitivity,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_LINK_CONFIG = LinkConfig(
    frequency_hz=26e9, transmit_power_w=1.0,
    tx_antenna_gain_dbi=30.0, rx_antenna_gain_dbi=30.0,
    system_noise_temp_k=300.0, bandwidth_hz=100e6,
)


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


def _drag():
    return DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)


class TestComputeSpectralFragility:
    def test_returns_type(self):
        states = _small_constellation()
        n_rad_s = states[0].mean_motion_rad_s
        result = compute_spectral_fragility(
            states, _EPOCH, _LINK_CONFIG, n_rad_s,
            control_duration_s=5400.0, lat_deg=0.0, lon_deg=0.0,
        )
        assert isinstance(result, SpectralFragility)

    def test_composite_positive(self):
        states = _small_constellation()
        n_rad_s = states[0].mean_motion_rad_s
        result = compute_spectral_fragility(
            states, _EPOCH, _LINK_CONFIG, n_rad_s,
            control_duration_s=5400.0, lat_deg=0.0, lon_deg=0.0,
        )
        assert result.composite_fragility >= 0.0

    def test_limiting_dimension_valid(self):
        states = _small_constellation()
        n_rad_s = states[0].mean_motion_rad_s
        result = compute_spectral_fragility(
            states, _EPOCH, _LINK_CONFIG, n_rad_s,
            control_duration_s=5400.0, lat_deg=0.0, lon_deg=0.0,
        )
        valid = {"topology", "controllability", "positioning"}
        assert result.limiting_dimension in valid

    def test_isotropy_bounded(self):
        states = _small_constellation()
        n_rad_s = states[0].mean_motion_rad_s
        result = compute_spectral_fragility(
            states, _EPOCH, _LINK_CONFIG, n_rad_s,
            control_duration_s=5400.0, lat_deg=0.0, lon_deg=0.0,
        )
        assert result.topology_isotropy >= 0.0
        assert result.controllability_isotropy >= 0.0
        assert result.positioning_efficiency >= 0.0


class TestComputeCoverageConnectivityCrossover:
    def test_returns_type(self):
        result = compute_coverage_connectivity_crossover(
            inclination_deg=53.0, num_planes=2, sats_per_plane=2,
            link_config=_LINK_CONFIG, epoch=_EPOCH,
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=100.0,
        )
        assert isinstance(result, CoverageConnectivityCrossover)

    def test_points_present(self):
        result = compute_coverage_connectivity_crossover(
            inclination_deg=53.0, num_planes=2, sats_per_plane=2,
            link_config=_LINK_CONFIG, epoch=_EPOCH,
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=100.0,
        )
        assert len(result.points) > 0

    def test_peak_product_positive(self):
        result = compute_coverage_connectivity_crossover(
            inclination_deg=53.0, num_planes=2, sats_per_plane=2,
            link_config=_LINK_CONFIG, epoch=_EPOCH,
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=100.0,
        )
        assert result.peak_product >= 0.0

    def test_peak_altitude_in_range(self):
        result = compute_coverage_connectivity_crossover(
            inclination_deg=53.0, num_planes=2, sats_per_plane=2,
            link_config=_LINK_CONFIG, epoch=_EPOCH,
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=100.0,
        )
        assert 400.0 <= result.peak_product_altitude_km <= 600.0


class TestComputeAltitudeSensitivity:
    def test_returns_type(self):
        state = _state()
        states = _small_constellation()
        result = compute_altitude_sensitivity(
            state, _drag(), _EPOCH, _LINK_CONFIG, states,
        )
        assert isinstance(result, AltitudeSensitivity)

    def test_dominant_sensitivity_valid(self):
        state = _state()
        states = _small_constellation()
        result = compute_altitude_sensitivity(
            state, _drag(), _EPOCH, _LINK_CONFIG, states,
        )
        valid = {"raan_drift", "drag_decay", "dose_rate", "thermal_cycle", "fiedler"}
        assert result.dominant_sensitivity in valid

    def test_sensitivities_computed(self):
        state = _state()
        states = _small_constellation()
        result = compute_altitude_sensitivity(
            state, _drag(), _EPOCH, _LINK_CONFIG, states,
        )
        # All sensitivities should be finite numbers
        assert math.isfinite(result.raan_drift_sensitivity)
        assert math.isfinite(result.drag_decay_sensitivity)
        assert math.isfinite(result.dose_rate_sensitivity)

    def test_delta_altitude_matches(self):
        state = _state()
        states = _small_constellation()
        result = compute_altitude_sensitivity(
            state, _drag(), _EPOCH, _LINK_CONFIG, states, delta_altitude_m=200.0,
        )
        assert result.delta_altitude_m == 200.0


class TestDesignSensitivityPurity:
    def test_no_external_deps(self):
        import constellation_generator.domain.design_sensitivity as mod
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())
        allowed = {"math", "dataclasses", "datetime", "constellation_generator"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed, f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed, f"Forbidden import from: {node.module}"
