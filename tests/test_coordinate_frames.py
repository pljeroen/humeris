"""Tests for coordinate frame conversions (ECI→ECEF→Geodetic)."""
import math
from datetime import datetime, timezone

import pytest

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from constellation_generator.domain.constellation import Satellite


# ── GMST computation ──────────────────────────────────────────────

class TestGMST:

    def test_gmst_returns_radians_in_range(self):
        """GMST must be in [0, 2π)."""
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        theta = gmst_rad(epoch)
        assert 0 <= theta < 2 * math.pi

    def test_gmst_j2000_epoch_reference(self):
        """At J2000.0 (2000-01-01 12:00 UTC), GMST ≈ 4.8949 rad (280.46°)."""
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        theta = gmst_rad(j2000)
        expected_rad = math.radians(280.46)
        assert abs(theta - expected_rad) < math.radians(0.1)

    def test_gmst_advances_with_time(self):
        """GMST increases over sidereal day (~23h56m)."""
        t1 = datetime(2026, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 15, 6, 0, 0, tzinfo=timezone.utc)
        theta1 = gmst_rad(t1)
        theta2 = gmst_rad(t2)
        # 6 hours ≈ π/2 radians of Earth rotation
        diff = (theta2 - theta1) % (2 * math.pi)
        assert abs(diff - math.pi / 2) < math.radians(2)


# ── ECI → ECEF ────────────────────────────────────────────────────

class TestECItoECEF:

    def test_identity_at_zero_gmst(self):
        """At GMST=0, ECEF equals ECI."""
        pos_eci = (7_000_000.0, 0.0, 0.0)
        vel_eci = (0.0, 7_500.0, 0.0)
        # J2000 epoch is close enough; use explicit gmst=0 test
        # by testing the rotation directly
        pos_ecef, vel_ecef = eci_to_ecef(pos_eci, vel_eci, gmst_angle_rad=0.0)
        for i in range(3):
            assert abs(pos_ecef[i] - pos_eci[i]) < 1.0
            assert abs(vel_ecef[i] - vel_eci[i]) < 0.01

    def test_rotation_preserves_position_magnitude(self):
        """‖ECEF(pos)‖ = ‖pos_ECI‖ for any GMST angle."""
        pos_eci = (6_778_000.0, 1_234_000.0, 3_456_000.0)
        vel_eci = (1_000.0, 7_000.0, 2_000.0)
        for angle_deg in [0, 45, 90, 135, 180, 270]:
            angle_rad = math.radians(angle_deg)
            pos_ecef, _ = eci_to_ecef(pos_eci, vel_eci, gmst_angle_rad=angle_rad)
            r_eci = math.sqrt(sum(p**2 for p in pos_eci))
            r_ecef = math.sqrt(sum(p**2 for p in pos_ecef))
            assert abs(r_ecef - r_eci) < 1.0

    def test_rotation_preserves_velocity_magnitude(self):
        """‖ECEF(vel)‖ = ‖vel_ECI‖ for any GMST angle."""
        pos_eci = (6_778_000.0, 0.0, 0.0)
        vel_eci = (0.0, 7_500.0, 1_000.0)
        pos_ecef, vel_ecef = eci_to_ecef(pos_eci, vel_eci, gmst_angle_rad=math.radians(90))
        v_eci = math.sqrt(sum(v**2 for v in vel_eci))
        v_ecef = math.sqrt(sum(v**2 for v in vel_ecef))
        assert abs(v_ecef - v_eci) < 0.1

    def test_90deg_rotation(self):
        """At GMST=90°, ECI x-axis maps to ECEF -y-axis (approximately)."""
        pos_eci = (7_000_000.0, 0.0, 0.0)
        vel_eci = (0.0, 0.0, 0.0)
        pos_ecef, _ = eci_to_ecef(pos_eci, vel_eci, gmst_angle_rad=math.radians(90))
        # x rotated 90° → (0, -7M, 0)
        assert abs(pos_ecef[0]) < 1.0
        assert abs(pos_ecef[1] + 7_000_000.0) < 1.0
        assert abs(pos_ecef[2]) < 1.0

    def test_z_component_unchanged(self):
        """Z-axis rotation leaves Z unchanged."""
        pos_eci = (1_000_000.0, 2_000_000.0, 5_000_000.0)
        vel_eci = (100.0, 200.0, 500.0)
        pos_ecef, vel_ecef = eci_to_ecef(pos_eci, vel_eci, gmst_angle_rad=math.radians(37))
        assert abs(pos_ecef[2] - pos_eci[2]) < 1e-6
        assert abs(vel_ecef[2] - vel_eci[2]) < 1e-6

    def test_with_epoch(self):
        """Convenience: can pass epoch datetime instead of raw angle."""
        pos_eci = (7_000_000.0, 0.0, 0.0)
        vel_eci = (0.0, 7_500.0, 0.0)
        epoch = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        gmst = gmst_rad(epoch)
        pos_ecef, vel_ecef = eci_to_ecef(pos_eci, vel_eci, gmst_angle_rad=gmst)
        r = math.sqrt(sum(p**2 for p in pos_ecef))
        assert abs(r - 7_000_000.0) < 1.0


# ── ECEF → Geodetic ───────────────────────────────────────────────

class TestECEFtoGeodetic:

    def test_equator_prime_meridian(self):
        """Point on equator at prime meridian → (0°, 0°, 0m)."""
        r_eq = 6_378_137.0  # WGS84 equatorial radius
        lat, lon, alt = ecef_to_geodetic((r_eq, 0.0, 0.0))
        assert abs(lat) < 0.001
        assert abs(lon) < 0.001
        assert abs(alt) < 1.0  # within 1 meter

    def test_north_pole(self):
        """Point at north pole → (90°, *, 0m)."""
        r_polar = 6_356_752.3142  # WGS84 polar radius
        lat, lon, alt = ecef_to_geodetic((0.0, 0.0, r_polar))
        assert abs(lat - 90.0) < 0.001
        assert abs(alt) < 1.0

    def test_south_pole(self):
        """Point at south pole → (-90°, *, 0m)."""
        r_polar = 6_356_752.3142
        lat, lon, alt = ecef_to_geodetic((0.0, 0.0, -r_polar))
        assert abs(lat + 90.0) < 0.001
        assert abs(alt) < 1.0

    def test_longitude_90_east(self):
        """Point on equator at 90°E → (0°, 90°, 0m)."""
        r_eq = 6_378_137.0
        lat, lon, alt = ecef_to_geodetic((0.0, r_eq, 0.0))
        assert abs(lat) < 0.001
        assert abs(lon - 90.0) < 0.001
        assert abs(alt) < 1.0

    def test_longitude_180(self):
        """Point on equator at 180° → (0°, 180° or -180°, 0m)."""
        r_eq = 6_378_137.0
        lat, lon, alt = ecef_to_geodetic((-r_eq, 0.0, 0.0))
        assert abs(lat) < 0.001
        assert abs(abs(lon) - 180.0) < 0.001
        assert abs(alt) < 1.0

    def test_altitude_iss_orbit(self):
        """ISS-altitude point has ~420 km altitude."""
        r_eq = 6_378_137.0
        iss_r = r_eq + 420_000  # ~420 km above equator
        lat, lon, alt = ecef_to_geodetic((iss_r, 0.0, 0.0))
        assert 410_000 < alt < 430_000

    def test_latitude_range(self):
        """Latitude always in [-90, 90]."""
        for z_frac in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            r = 7_000_000.0
            x = r * math.sqrt(1 - z_frac**2)
            z = r * z_frac
            lat, lon, alt = ecef_to_geodetic((x, 0.0, z))
            assert -90.0 <= lat <= 90.0

    def test_longitude_range(self):
        """Longitude always in (-180, 180]."""
        for angle_deg in [0, 45, 90, 135, 180, -135, -90, -45]:
            r = 7_000_000.0
            x = r * math.cos(math.radians(angle_deg))
            y = r * math.sin(math.radians(angle_deg))
            lat, lon, alt = ecef_to_geodetic((x, y, 0.0))
            assert -180.0 < lon <= 180.0


# ── Geodetic → ECEF ──────────────────────────────────────────────

class TestGeodeticToECEF:

    def test_equator_prime_meridian(self):
        """(0°, 0°, 0m) → equatorial radius on x-axis."""
        x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
        r_eq = OrbitalConstants.R_EARTH_EQUATORIAL
        assert abs(x - r_eq) < 1.0
        assert abs(y) < 1.0
        assert abs(z) < 1.0

    def test_north_pole(self):
        """(90°, 0°, 0m) → polar radius on z-axis."""
        x, y, z = geodetic_to_ecef(90.0, 0.0, 0.0)
        r_polar = OrbitalConstants.R_EARTH_POLAR
        assert abs(x) < 1.0
        assert abs(y) < 1.0
        assert abs(z - r_polar) < 1.0

    def test_south_pole(self):
        """(-90°, 0°, 0m) → negative polar radius on z-axis."""
        x, y, z = geodetic_to_ecef(-90.0, 0.0, 0.0)
        r_polar = OrbitalConstants.R_EARTH_POLAR
        assert abs(x) < 1.0
        assert abs(y) < 1.0
        assert abs(z + r_polar) < 1.0

    def test_longitude_90_east(self):
        """(0°, 90°, 0m) → equatorial radius on y-axis."""
        x, y, z = geodetic_to_ecef(0.0, 90.0, 0.0)
        r_eq = OrbitalConstants.R_EARTH_EQUATORIAL
        assert abs(x) < 1.0
        assert abs(y - r_eq) < 1.0
        assert abs(z) < 1.0

    def test_longitude_180(self):
        """(0°, 180°, 0m) → negative equatorial radius on x-axis."""
        x, y, z = geodetic_to_ecef(0.0, 180.0, 0.0)
        r_eq = OrbitalConstants.R_EARTH_EQUATORIAL
        assert abs(x + r_eq) < 1.0
        assert abs(y) < 1.0
        assert abs(z) < 1.0

    def test_round_trip_geodetic_ecef_geodetic(self):
        """geodetic→ECEF→geodetic round-trip preserves coordinates."""
        for lat, lon, alt in [(52.0, 4.4, 0.0), (-33.9, 18.4, 100.0), (0.0, -75.0, 5000.0)]:
            ecef = geodetic_to_ecef(lat, lon, alt)
            lat2, lon2, alt2 = ecef_to_geodetic(ecef)
            assert abs(lat2 - lat) < 0.001, f"Lat mismatch: {lat} → {lat2}"
            assert abs(lon2 - lon) < 0.001, f"Lon mismatch: {lon} → {lon2}"
            assert abs(alt2 - alt) < 1.0, f"Alt mismatch: {alt} → {alt2}"

    def test_round_trip_ecef_geodetic_ecef(self):
        """ECEF→geodetic→ECEF round-trip preserves coordinates."""
        for ecef in [(6_378_137.0, 0.0, 0.0), (0.0, 0.0, 6_356_752.3142),
                     (4_000_000.0, 3_000_000.0, 4_500_000.0)]:
            lat, lon, alt = ecef_to_geodetic(ecef)
            x2, y2, z2 = geodetic_to_ecef(lat, lon, alt)
            assert abs(x2 - ecef[0]) < 1.0, f"X mismatch: {ecef[0]} → {x2}"
            assert abs(y2 - ecef[1]) < 1.0, f"Y mismatch: {ecef[1]} → {y2}"
            assert abs(z2 - ecef[2]) < 1.0, f"Z mismatch: {ecef[2]} → {z2}"

    def test_altitude_effect(self):
        """Higher altitude → larger ECEF radius."""
        x0, y0, z0 = geodetic_to_ecef(0.0, 0.0, 0.0)
        x1, y1, z1 = geodetic_to_ecef(0.0, 0.0, 100_000.0)
        r0 = math.sqrt(x0**2 + y0**2 + z0**2)
        r1 = math.sqrt(x1**2 + y1**2 + z1**2)
        assert r1 - r0 == pytest.approx(100_000.0, abs=1.0)


# ── Satellite epoch field ─────────────────────────────────────────

class TestSatelliteEpoch:

    def test_satellite_epoch_default_none(self):
        """Synthetic satellites have epoch=None by default."""
        sat = Satellite(
            name="Test",
            position_eci=(7_000_000.0, 0.0, 0.0),
            velocity_eci=(0.0, 7_500.0, 0.0),
            plane_index=0,
            sat_index=0,
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        assert sat.epoch is None

    def test_satellite_epoch_can_be_set(self):
        """Live satellites can have epoch set."""
        epoch = datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc)
        sat = Satellite(
            name="Test",
            position_eci=(7_000_000.0, 0.0, 0.0),
            velocity_eci=(0.0, 7_500.0, 0.0),
            plane_index=0,
            sat_index=0,
            raan_deg=0.0,
            true_anomaly_deg=0.0,
            epoch=epoch,
        )
        assert sat.epoch == epoch

    def test_satellite_epoch_immutable(self):
        """Epoch field is frozen with the rest of Satellite."""
        sat = Satellite(
            name="Test",
            position_eci=(7_000_000.0, 0.0, 0.0),
            velocity_eci=(0.0, 7_500.0, 0.0),
            plane_index=0,
            sat_index=0,
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            sat.epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ── Domain purity (extended) ──────────────────────────────────────

class TestCoordinateFramesPurity:

    def test_coordinate_frames_imports_only_stdlib(self):
        """coordinate_frames.py must only import allowed stdlib modules."""
        import ast
        import os
        allowed = {'math', 'dataclasses', 'typing', 'abc', 'enum',
                   '__future__', 'datetime'}
        path = os.path.join(os.path.dirname(__file__), '..', 'src',
                            'constellation_generator', 'domain',
                            'coordinate_frames.py')
        with open(path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('constellation_generator'):
                        assert False, f"Disallowed import '{alias.name}' in coordinate_frames.py"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'constellation_generator':
                        assert False, f"Disallowed import from '{node.module}' in coordinate_frames.py"
