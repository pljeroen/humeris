# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for domain/temporal_correlation.py — cross-spectral coherence engine."""
import ast
import math
from datetime import datetime, timezone, timedelta

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.atmosphere import DragConfig

from constellation_generator.domain.temporal_correlation import (
    SignalCoherence,
    SpectralCrossCorrelation,
    AvailabilitySpectralDecomposition,
    CapacitySpectrum,
    compute_signal_coherence,
    compute_spectral_cross_correlation,
    compute_availability_spectral_decomposition,
    compute_network_capacity_spectrum,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _state(alt_km=400.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(53.0),
        raan_rad=0.0, arg_perigee_rad=0.0, true_anomaly_rad=0.0,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


def _drag():
    return DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)


class TestComputeSignalCoherence:
    def test_returns_type(self):
        signal = [math.sin(2 * math.pi * i / 16) for i in range(64)]
        result = compute_signal_coherence(signal, 1.0)
        assert isinstance(result, SignalCoherence)

    def test_parseval_ratio_near_one(self):
        """Parseval's theorem: energy in time ≈ energy in frequency."""
        signal = [math.sin(2 * math.pi * i / 16) for i in range(64)]
        result = compute_signal_coherence(signal, 1.0)
        assert 0.8 <= result.parseval_ratio <= 1.2

    def test_pure_sine_is_narrowband(self):
        signal = [math.sin(2 * math.pi * i / 16) for i in range(64)]
        result = compute_signal_coherence(signal, 1.0)
        assert result.dominant_power_fraction > 0.3


class TestComputeSpectralCrossCorrelation:
    def test_returns_type(self):
        a = [math.sin(2 * math.pi * i / 16) for i in range(64)]
        b = [math.sin(2 * math.pi * i / 16 + 0.5) for i in range(64)]
        result = compute_spectral_cross_correlation(a, b, 1.0)
        assert isinstance(result, SpectralCrossCorrelation)

    def test_coherence_bounds(self):
        a = [math.sin(2 * math.pi * i / 16) for i in range(64)]
        b = [math.sin(2 * math.pi * i / 16 + 0.5) for i in range(64)]
        result = compute_spectral_cross_correlation(a, b, 1.0)
        assert 0.0 <= result.coherence_at_dominant <= 1.0

    def test_identical_signals_high_coherence(self):
        a = [math.sin(2 * math.pi * i / 16) for i in range(64)]
        result = compute_spectral_cross_correlation(a, a, 1.0)
        assert result.coherence_at_dominant > 0.9


class TestComputeAvailabilitySpectralDecomposition:
    def test_returns_type(self):
        from constellation_generator.domain.statistical_analysis import (
            compute_mission_availability,
        )
        state = _state()
        profile = compute_mission_availability(
            state, _drag(), _EPOCH,
            isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
            mission_years=5.0,
        )
        result = compute_availability_spectral_decomposition(profile)
        assert isinstance(result, AvailabilitySpectralDecomposition)

    def test_coherence_bounded(self):
        from constellation_generator.domain.statistical_analysis import (
            compute_mission_availability,
        )
        state = _state()
        profile = compute_mission_availability(
            state, _drag(), _EPOCH,
            isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
            mission_years=5.0,
        )
        result = compute_availability_spectral_decomposition(profile)
        assert -1.0 <= result.fuel_power_coherence <= 1.0
        assert -1.0 <= result.fuel_conjunction_coherence <= 1.0
        assert -1.0 <= result.power_conjunction_coherence <= 1.0

    def test_dominant_frequency_non_negative(self):
        from constellation_generator.domain.statistical_analysis import (
            compute_mission_availability,
        )
        state = _state()
        profile = compute_mission_availability(
            state, _drag(), _EPOCH,
            isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
            mission_years=5.0,
        )
        result = compute_availability_spectral_decomposition(profile)
        assert result.dominant_failure_frequency_hz >= 0.0


class TestComputeNetworkCapacitySpectrum:
    def test_returns_type(self):
        from constellation_generator.domain.communication_analysis import (
            NetworkCapacityTimeline,
            NetworkCapacitySnapshot,
        )
        snapshots = []
        for i in range(32):
            t = _EPOCH + timedelta(seconds=i * 300)
            snapshots.append(NetworkCapacitySnapshot(
                time=t, active_isl_count=5 + int(3 * math.sin(2 * math.pi * i / 16)),
                degraded_isl_count=1, ground_contact_count=2, eclipsed_sat_count=0,
            ))
        timeline = NetworkCapacityTimeline(
            snapshots=tuple(snapshots),
            min_active_isl_count=2,
            min_capacity_time=_EPOCH,
            mean_active_isl_count=5.0,
        )
        orbital_period_s = 2 * math.pi / _state().mean_motion_rad_s
        result = compute_network_capacity_spectrum(timeline, orbital_period_s)
        assert isinstance(result, CapacitySpectrum)

    def test_frequencies_present(self):
        from constellation_generator.domain.communication_analysis import (
            NetworkCapacityTimeline,
            NetworkCapacitySnapshot,
        )
        snapshots = []
        for i in range(32):
            t = _EPOCH + timedelta(seconds=i * 300)
            snapshots.append(NetworkCapacitySnapshot(
                time=t, active_isl_count=5 + int(3 * math.sin(2 * math.pi * i / 16)),
                degraded_isl_count=1, ground_contact_count=2, eclipsed_sat_count=0,
            ))
        timeline = NetworkCapacityTimeline(
            snapshots=tuple(snapshots),
            min_active_isl_count=2,
            min_capacity_time=_EPOCH,
            mean_active_isl_count=5.0,
        )
        orbital_period_s = 2 * math.pi / _state().mean_motion_rad_s
        result = compute_network_capacity_spectrum(timeline, orbital_period_s)
        assert len(result.frequencies_hz) > 0
        assert len(result.power_density) > 0

    def test_orbital_frequency_detected(self):
        from constellation_generator.domain.communication_analysis import (
            NetworkCapacityTimeline,
            NetworkCapacitySnapshot,
        )
        orbital_period_s = 2 * math.pi / _state().mean_motion_rad_s
        snapshots = []
        for i in range(64):
            t = _EPOCH + timedelta(seconds=i * (orbital_period_s / 16))
            snapshots.append(NetworkCapacitySnapshot(
                time=t, active_isl_count=5 + int(3 * math.sin(2 * math.pi * i / 16)),
                degraded_isl_count=1, ground_contact_count=2, eclipsed_sat_count=0,
            ))
        timeline = NetworkCapacityTimeline(
            snapshots=tuple(snapshots),
            min_active_isl_count=2,
            min_capacity_time=_EPOCH,
            mean_active_isl_count=5.0,
        )
        result = compute_network_capacity_spectrum(timeline, orbital_period_s)
        assert result.orbital_frequency_hz > 0


class TestTemporalCorrelationPurity:
    def test_no_external_deps(self):
        import constellation_generator.domain.temporal_correlation as mod
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())
        allowed = {"math", "numpy", "dataclasses", "datetime", "constellation_generator"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed, f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed, f"Forbidden import from: {node.module}"
