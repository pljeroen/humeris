# Copyright (c) 2026 Jeroen. All rights reserved.
"""Tests for domain/spectral_topology.py — Fiedler DFT, eclipse invariant, proximity spectrum."""
import ast
import math
from datetime import datetime, timezone, timedelta

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.link_budget import LinkConfig

from constellation_generator.domain.spectral_topology import (
    FragmentationSpectralAnalysis,
    EclipseSpectralInvariant,
    ProximitySpectralResult,
    compute_fragmentation_spectrum,
    compute_eclipse_spectral_invariant,
    compute_proximity_spectrum,
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
    """4-sat constellation in 2 planes."""
    return [
        _state(raan_deg=0.0, ta_deg=0.0),
        _state(raan_deg=0.0, ta_deg=180.0),
        _state(raan_deg=90.0, ta_deg=0.0),
        _state(raan_deg=90.0, ta_deg=180.0),
    ]


class TestFragmentationSpectrum:
    def test_returns_type(self):
        states = _small_constellation()
        result = compute_fragmentation_spectrum(
            states, _LINK_CONFIG, _EPOCH,
            duration_s=5400.0, step_s=300.0,
        )
        assert isinstance(result, FragmentationSpectralAnalysis)

    def test_frequencies_match_sample_rate(self):
        states = _small_constellation()
        step_s = 300.0
        result = compute_fragmentation_spectrum(
            states, _LINK_CONFIG, _EPOCH,
            duration_s=5400.0, step_s=step_s,
        )
        assert len(result.fiedler_frequencies_hz) > 0
        # First frequency should be 0 (DC)
        assert abs(result.fiedler_frequencies_hz[0]) < 1e-10

    def test_orbital_frequency_positive(self):
        states = _small_constellation()
        result = compute_fragmentation_spectrum(
            states, _LINK_CONFIG, _EPOCH,
            duration_s=5400.0, step_s=300.0,
        )
        assert result.orbital_frequency_hz > 0.0

    def test_fiedler_magnitudes_nonnegative(self):
        states = _small_constellation()
        result = compute_fragmentation_spectrum(
            states, _LINK_CONFIG, _EPOCH,
            duration_s=5400.0, step_s=300.0,
        )
        assert all(m >= -1e-15 for m in result.fiedler_magnitudes)

    def test_coherence_bounded(self):
        states = _small_constellation()
        result = compute_fragmentation_spectrum(
            states, _LINK_CONFIG, _EPOCH,
            duration_s=5400.0, step_s=300.0,
        )
        assert all(0.0 <= c <= 1.0 + 1e-10 for c in result.coherence_squared)


class TestEclipseSpectralInvariant:
    def test_returns_type(self):
        states = _small_constellation()
        result = compute_eclipse_spectral_invariant(
            states, _LINK_CONFIG, _EPOCH,
            orbital_period_s=5400.0, num_samples=10,
            eclipse_power_fraction=0.5,
        )
        assert isinstance(result, EclipseSpectralInvariant)

    def test_eta_between_zero_and_one(self):
        states = _small_constellation()
        result = compute_eclipse_spectral_invariant(
            states, _LINK_CONFIG, _EPOCH,
            orbital_period_s=5400.0, num_samples=10,
            eclipse_power_fraction=0.5,
        )
        assert 0.0 <= result.eta_eclipse <= 1.0 + 1e-10

    def test_no_eclipse_eta_is_one(self):
        """With eclipse_power_fraction=1.0, eclipse has no effect → eta=1."""
        states = _small_constellation()
        result = compute_eclipse_spectral_invariant(
            states, _LINK_CONFIG, _EPOCH,
            orbital_period_s=5400.0, num_samples=10,
            eclipse_power_fraction=1.0,
        )
        assert abs(result.eta_eclipse - 1.0) < 1e-6


class TestProximitySpectrum:
    def test_returns_type(self):
        states = _small_constellation()
        result = compute_proximity_spectrum(
            states, _EPOCH,
            duration_s=5400.0, step_s=300.0,
            pair_indices=(0, 1),
        )
        assert isinstance(result, ProximitySpectralResult)

    def test_dominant_frequency_positive(self):
        states = _small_constellation()
        result = compute_proximity_spectrum(
            states, _EPOCH,
            duration_s=5400.0, step_s=300.0,
            pair_indices=(0, 2),
        )
        assert result.dominant_frequency_hz >= 0.0

    def test_resonance_ratio_defined(self):
        states = _small_constellation()
        result = compute_proximity_spectrum(
            states, _EPOCH,
            duration_s=5400.0, step_s=300.0,
            pair_indices=(0, 2),
        )
        assert result.resonance_ratio >= 0.0


class TestSpectralTopologyPurity:
    def test_module_pure(self):
        import constellation_generator.domain.spectral_topology as mod
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
