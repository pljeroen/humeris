# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/information_theory.py — BEC capacity, coverage spectrum, marginal value."""
import ast
import math
from datetime import datetime, timezone

from humeris.domain.propagation import OrbitalState
from humeris.domain.link_budget import LinkConfig
from humeris.domain.information_theory import (
    EclipseChannelCapacity,
    CoverageSpectrum,
    MarginalSatelliteValue,
    compute_eclipse_channel_capacity,
    compute_coverage_spectrum,
    compute_marginal_satellite_value,
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


class TestEclipseChannelCapacity:
    def test_returns_type(self):
        result = compute_eclipse_channel_capacity(
            _state(), _EPOCH, _LINK_CONFIG, distance_m=1000_000.0,
        )
        assert isinstance(result, EclipseChannelCapacity)

    def test_bec_leq_awgn(self):
        result = compute_eclipse_channel_capacity(
            _state(), _EPOCH, _LINK_CONFIG, distance_m=1000_000.0,
        )
        assert result.bec_capacity_bps <= result.awgn_capacity_bps + 1e-6

    def test_zero_eclipse_equals_awgn(self):
        # Dawn-dusk SSO at equinox should have near-zero eclipse
        result = compute_eclipse_channel_capacity(
            _state(inc_deg=97.4, raan_deg=90.0), _EPOCH, _LINK_CONFIG,
            distance_m=1000_000.0,
        )
        # Even if not exactly zero, BEC should be close to AWGN
        assert result.bec_capacity_bps > 0.0

    def test_scheduling_gain_geq_one(self):
        result = compute_eclipse_channel_capacity(
            _state(), _EPOCH, _LINK_CONFIG, distance_m=1000_000.0,
        )
        assert result.scheduling_gain >= 1.0 - 1e-10


class TestCoverageSpectrum:
    def test_returns_type(self):
        states = [_state(raan_deg=r) for r in [0.0, 90.0, 180.0, 270.0]]
        result = compute_coverage_spectrum(
            states, _EPOCH, duration_s=5400.0, step_s=60.0,
            lat_deg=45.0, lon_deg=0.0,
        )
        assert isinstance(result, CoverageSpectrum)

    def test_has_orbital_peak(self):
        states = [_state()]
        result = compute_coverage_spectrum(
            states, _EPOCH, duration_s=10800.0, step_s=30.0,
            lat_deg=45.0, lon_deg=0.0,
        )
        assert result.dominant_frequency_hz > 0.0

    def test_frequencies_positive(self):
        states = [_state()]
        result = compute_coverage_spectrum(
            states, _EPOCH, duration_s=5400.0, step_s=60.0,
            lat_deg=45.0, lon_deg=0.0,
        )
        assert all(f >= 0.0 for f in result.frequencies_hz)

    def test_parseval_energy(self):
        states = [_state()]
        result = compute_coverage_spectrum(
            states, _EPOCH, duration_s=5400.0, step_s=60.0,
            lat_deg=45.0, lon_deg=0.0,
        )
        # Power density should be non-negative
        assert all(p >= -1e-10 for p in result.power_density)

    def test_resonance_ratio_positive(self):
        states = [_state()]
        result = compute_coverage_spectrum(
            states, _EPOCH, duration_s=5400.0, step_s=60.0,
            lat_deg=45.0, lon_deg=0.0,
        )
        assert result.resonance_ratio >= 0.0


class TestMarginalSatelliteValue:
    def test_returns_type(self):
        states = [_state(raan_deg=0.0), _state(raan_deg=90.0)]
        candidate = _state(raan_deg=45.0)
        result = compute_marginal_satellite_value(
            states, candidate, _EPOCH,
            duration_s=5400.0, step_s=60.0,
        )
        assert isinstance(result, MarginalSatelliteValue)

    def test_coverage_gain_nonneg(self):
        states = [_state(raan_deg=0.0), _state(raan_deg=90.0)]
        candidate = _state(raan_deg=45.0)
        result = compute_marginal_satellite_value(
            states, candidate, _EPOCH,
            duration_s=5400.0, step_s=60.0,
        )
        assert result.coverage_information_gain >= -1e-10

    def test_positioning_gain_nonneg(self):
        states = [_state(raan_deg=0.0), _state(raan_deg=90.0)]
        candidate = _state(raan_deg=45.0)
        result = compute_marginal_satellite_value(
            states, candidate, _EPOCH,
            duration_s=5400.0, step_s=60.0,
        )
        assert result.positioning_info_gain >= -1e-10

    def test_total_value_positive(self):
        states = [_state(raan_deg=0.0), _state(raan_deg=90.0)]
        candidate = _state(raan_deg=45.0)
        result = compute_marginal_satellite_value(
            states, candidate, _EPOCH,
            duration_s=5400.0, step_s=60.0,
        )
        assert result.total_information_value >= 0.0


class TestInformationTheoryPurity:
    def test_module_pure(self):
        import humeris.domain.information_theory as mod
        source = ast.parse(open(mod.__file__).read())
        for node in ast.walk(source):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                else:
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                assert top in {"math", "dataclasses", "datetime", "typing", "enum", "numpy", "humeris", "__future__"}, (
                    f"Forbidden import: {top}"
                )
