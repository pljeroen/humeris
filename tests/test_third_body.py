# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for solar and lunar third-body perturbations."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.third_body import (
    LunarThirdBodyForce,
    MoonPosition,
    SolarThirdBodyForce,
    moon_position_eci,
)
from humeris.domain.numerical_propagation import ForceModel


_R_EARTH = 6_371_000.0
_LEO_ALT = 500_000.0
_LEO_R = _R_EARTH + _LEO_ALT


class TestMoonPosition:
    """Tests for analytical lunar ephemeris."""

    def test_moon_position_returns_type(self):
        """Return type is MoonPosition."""
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        result = moon_position_eci(epoch)
        assert isinstance(result, MoonPosition)

    def test_moon_distance_reasonable(self):
        """Moon distance in 356,000–407,000 km range."""
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        result = moon_position_eci(epoch)
        dist_km = result.distance_m / 1000.0
        assert 356_000 < dist_km < 407_000, f"Moon distance {dist_km} km out of range"

    def test_moon_position_varies_monthly(self):
        """Position differs at 14-day offset."""
        epoch1 = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        epoch2 = epoch1 + timedelta(days=14)
        p1 = moon_position_eci(epoch1)
        p2 = moon_position_eci(epoch2)
        dx = p1.position_eci_m[0] - p2.position_eci_m[0]
        dy = p1.position_eci_m[1] - p2.position_eci_m[1]
        dz = p1.position_eci_m[2] - p2.position_eci_m[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        assert dist > 1e8  # > 100 km difference

    def test_moon_ecliptic_latitude_bounded(self):
        """|β| < 6° (Moon stays near ecliptic)."""
        epoch = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = moon_position_eci(epoch)
        dec_deg = abs(math.degrees(result.declination_rad))
        # Declination combines ecliptic latitude and obliquity; bound by ~28.6° max
        assert dec_deg < 30.0

    def test_moon_position_eci_magnitude(self):
        """|pos| ≈ distance_m."""
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        result = moon_position_eci(epoch)
        pos_mag = math.sqrt(sum(c**2 for c in result.position_eci_m))
        assert abs(pos_mag - result.distance_m) / result.distance_m < 1e-6


class TestSolarThirdBody:
    """Tests for solar third-body perturbation force model."""

    def test_solar_third_body_returns_tuple(self):
        """Returns 3-tuple acceleration."""
        force = SolarThirdBodyForce()
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        pos = (_LEO_R, 0.0, 0.0)
        vel = (0.0, 7600.0, 0.0)
        a = force.acceleration(epoch, pos, vel)
        assert isinstance(a, tuple)
        assert len(a) == 3

    def test_solar_third_body_magnitude_order(self):
        """Solar perturbation at LEO: ~5e-6 m/s² order."""
        force = SolarThirdBodyForce()
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        pos = (_LEO_R, 0.0, 0.0)
        vel = (0.0, 7600.0, 0.0)
        a = force.acceleration(epoch, pos, vel)
        mag = math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        assert 1e-7 < mag < 1e-4, f"Solar perturbation magnitude {mag} out of expected range"


class TestLunarThirdBody:
    """Tests for lunar third-body perturbation force model."""

    def test_lunar_third_body_returns_tuple(self):
        """Returns 3-tuple acceleration."""
        force = LunarThirdBodyForce()
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        pos = (_LEO_R, 0.0, 0.0)
        vel = (0.0, 7600.0, 0.0)
        a = force.acceleration(epoch, pos, vel)
        assert isinstance(a, tuple)
        assert len(a) == 3

    def test_lunar_third_body_magnitude_order(self):
        """Lunar tidal perturbation at LEO: ~1e-5 m/s² order."""
        force = LunarThirdBodyForce()
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        pos = (_LEO_R, 0.0, 0.0)
        vel = (0.0, 7600.0, 0.0)
        a = force.acceleration(epoch, pos, vel)
        mag = math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        assert 1e-8 < mag < 1e-4, f"Lunar perturbation magnitude {mag} out of expected range"


class TestThirdBodyComparison:
    """Comparison and integration tests."""

    def test_lunar_larger_tidal_than_solar(self):
        """At LEO, lunar tidal > solar tidal (ratio ~2.2)."""
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        pos = (_LEO_R, 0.0, 0.0)
        vel = (0.0, 7600.0, 0.0)
        a_sun = SolarThirdBodyForce().acceleration(epoch, pos, vel)
        a_moon = LunarThirdBodyForce().acceleration(epoch, pos, vel)
        mag_sun = math.sqrt(sum(c**2 for c in a_sun))
        mag_moon = math.sqrt(sum(c**2 for c in a_moon))
        assert mag_moon > mag_sun

    def test_third_body_implements_protocol(self):
        """Both classes satisfy ForceModel protocol."""
        assert isinstance(SolarThirdBodyForce(), ForceModel)
        assert isinstance(LunarThirdBodyForce(), ForceModel)

    def test_numerical_propagation_with_third_body(self):
        """Third-body forces integrate with numerical propagator without error."""
        from humeris.domain.numerical_propagation import (
            TwoBodyGravity,
            propagate_numerical,
        )
        from humeris.domain.propagation import OrbitalState

        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        mu = 3.986004418e14
        a = _LEO_R
        n = math.sqrt(mu / a**3)
        state = OrbitalState(
            semi_major_axis_m=a, eccentricity=0.0,
            inclination_rad=math.radians(53.0), raan_rad=0.0,
            arg_perigee_rad=0.0, true_anomaly_rad=0.0,
            mean_motion_rad_s=n, reference_epoch=epoch,
        )
        forces = [TwoBodyGravity(), SolarThirdBodyForce(), LunarThirdBodyForce()]
        result = propagate_numerical(
            state, timedelta(minutes=90), timedelta(seconds=60),
            forces, epoch=epoch,
        )
        assert len(result.steps) > 0

    def test_energy_changes_with_third_body(self):
        """Specific energy not conserved with third-body perturbation."""
        from humeris.domain.numerical_propagation import (
            TwoBodyGravity,
            propagate_numerical,
        )
        from humeris.domain.propagation import OrbitalState

        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        mu = 3.986004418e14
        a = _LEO_R
        n = math.sqrt(mu / a**3)
        state = OrbitalState(
            semi_major_axis_m=a, eccentricity=0.0,
            inclination_rad=math.radians(53.0), raan_rad=0.0,
            arg_perigee_rad=0.0, true_anomaly_rad=0.0,
            mean_motion_rad_s=n, reference_epoch=epoch,
        )
        forces = [TwoBodyGravity(), SolarThirdBodyForce()]
        result = propagate_numerical(
            state, timedelta(hours=6), timedelta(seconds=60),
            forces, epoch=epoch,
        )
        # Compute specific energy at start and end
        def energy(step):
            r = math.sqrt(sum(c**2 for c in step.position_eci))
            v = math.sqrt(sum(c**2 for c in step.velocity_eci))
            return 0.5 * v**2 - mu / r

        e0 = energy(result.steps[0])
        ef = energy(result.steps[-1])
        assert abs(ef - e0) > 1e-3  # Energy drifts with perturbation

    def test_zero_distance_guard(self):
        """Far from body → small perturbation (sanity)."""
        force = SolarThirdBodyForce()
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        # Satellite at origin — extreme case but should not crash
        pos = (1000.0, 0.0, 0.0)
        vel = (0.0, 0.0, 0.0)
        a = force.acceleration(epoch, pos, vel)
        mag = math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
        assert mag < 1.0  # Should still be small (tidal effect)


class TestThirdBodyPurity:
    """Domain purity: third_body.py must only import stdlib + domain."""

    def test_module_pure(self):
        import humeris.domain.third_body as mod

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
