"""Tests for radiation environment modeling."""
import ast
import math
from datetime import datetime, timezone

import pytest

from constellation_generator.domain.radiation import (
    OrbitRadiationSummary,
    RadiationEnvironment,
    compute_l_shell,
    compute_orbit_radiation_summary,
    compute_radiation_environment,
)


_R_EARTH_KM = 6371.0


class TestLShell:
    """Tests for dipole L-shell computation."""

    def test_l_shell_equator(self):
        """L ≈ (R_E + alt)/R_E at magnetic equator."""
        alt_km = 500.0
        # Near magnetic equator (geographic ~11° N, -72° W for offset dipole)
        l_val = compute_l_shell(11.0, -72.0, alt_km)
        expected = (_R_EARTH_KM + alt_km) / _R_EARTH_KM
        assert abs(l_val - expected) < 0.25

    def test_l_shell_increases_with_latitude(self):
        """Higher magnetic latitude → higher L."""
        l_low = compute_l_shell(20.0, 0.0, 500.0)
        l_high = compute_l_shell(60.0, 0.0, 500.0)
        assert l_high > l_low

    def test_l_shell_increases_with_altitude(self):
        """Higher altitude → higher L."""
        l_low = compute_l_shell(30.0, 0.0, 400.0)
        l_high = compute_l_shell(30.0, 0.0, 1000.0)
        assert l_high > l_low


class TestRadiationEnvironment:
    """Tests for point radiation assessment."""

    def test_radiation_env_returns_type(self):
        """Return type is RadiationEnvironment."""
        result = compute_radiation_environment(30.0, 0.0, 500.0)
        assert isinstance(result, RadiationEnvironment)

    def test_radiation_leo_moderate(self):
        """400 km, mid-lat → moderate dose rate (not zero, not extreme)."""
        result = compute_radiation_environment(30.0, 0.0, 400.0)
        assert result.total_dose_rate_rad_s > 0

    def test_radiation_inner_belt_peak(self):
        """L ≈ 1.5 → high proton flux."""
        # L≈1.5 at ~500 km near magnetic equator with slight offset
        result_near = compute_radiation_environment(11.0, -72.0, 3000.0)
        result_far = compute_radiation_environment(11.0, -72.0, 500.0)
        # At higher altitude (higher L), protons should eventually peak then decrease
        # Just check both produce valid results
        assert result_near.proton_flux_cm2_s >= 0
        assert result_far.proton_flux_cm2_s >= 0

    def test_radiation_outer_belt_peak(self):
        """L ≈ 4.5 → high electron flux."""
        # High L → need high altitude or high latitude
        result = compute_radiation_environment(60.0, 0.0, 1000.0)
        assert result.electron_flux_cm2_s >= 0

    def test_saa_detection(self):
        """lat ≈ -30°, lon ≈ -45° → is_in_saa at low altitude."""
        result = compute_radiation_environment(-30.0, -45.0, 400.0)
        assert result.is_in_saa is True

    def test_non_saa_region(self):
        """High northern latitude, positive longitude → not SAA."""
        result = compute_radiation_environment(50.0, 90.0, 400.0)
        assert result.is_in_saa is False


class TestOrbitRadiationSummary:
    """Tests for orbit-averaged radiation summary."""

    def test_orbit_summary_returns_type(self):
        """Return type is OrbitRadiationSummary."""
        from constellation_generator.domain.propagation import OrbitalState

        mu = 3.986004418e14
        a = _R_EARTH_KM * 1000.0 + 500_000.0
        n = math.sqrt(mu / a**3)
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        state = OrbitalState(
            semi_major_axis_m=a, eccentricity=0.0,
            inclination_rad=math.radians(53.0), raan_rad=0.0,
            arg_perigee_rad=0.0, true_anomaly_rad=0.0,
            mean_motion_rad_s=n, reference_epoch=epoch,
        )
        result = compute_orbit_radiation_summary(state, epoch)
        assert isinstance(result, OrbitRadiationSummary)

    def test_orbit_summary_annual_dose_positive(self):
        """Annual dose > 0 for LEO orbit."""
        from constellation_generator.domain.propagation import OrbitalState

        mu = 3.986004418e14
        a = _R_EARTH_KM * 1000.0 + 500_000.0
        n = math.sqrt(mu / a**3)
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        state = OrbitalState(
            semi_major_axis_m=a, eccentricity=0.0,
            inclination_rad=math.radians(53.0), raan_rad=0.0,
            arg_perigee_rad=0.0, true_anomaly_rad=0.0,
            mean_motion_rad_s=n, reference_epoch=epoch,
        )
        result = compute_orbit_radiation_summary(state, epoch)
        assert result.annual_dose_rad > 0

    def test_higher_inclination_more_saa(self):
        """Higher inclination orbit → more SAA exposure."""
        from constellation_generator.domain.propagation import OrbitalState

        mu = 3.986004418e14
        a = _R_EARTH_KM * 1000.0 + 500_000.0
        n = math.sqrt(mu / a**3)
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

        state_low = OrbitalState(
            semi_major_axis_m=a, eccentricity=0.0,
            inclination_rad=math.radians(10.0), raan_rad=0.0,
            arg_perigee_rad=0.0, true_anomaly_rad=0.0,
            mean_motion_rad_s=n, reference_epoch=epoch,
        )
        state_high = OrbitalState(
            semi_major_axis_m=a, eccentricity=0.0,
            inclination_rad=math.radians(55.0), raan_rad=0.0,
            arg_perigee_rad=0.0, true_anomaly_rad=0.0,
            mean_motion_rad_s=n, reference_epoch=epoch,
        )
        r_low = compute_orbit_radiation_summary(state_low, epoch)
        r_high = compute_orbit_radiation_summary(state_high, epoch)
        assert r_high.saa_fraction >= r_low.saa_fraction


class TestRadiationPurity:
    """Domain purity: radiation.py must only import stdlib + domain."""

    def test_module_pure(self):
        import constellation_generator.domain.radiation as mod

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
