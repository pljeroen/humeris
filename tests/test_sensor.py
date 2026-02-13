# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for sensor/payload FOV modeling."""

import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


# --- Type tests ---

class TestSensorType:

    def test_enum_values(self):
        from humeris.domain.sensor import SensorType
        assert SensorType.CIRCULAR.value == "circular"
        assert SensorType.RECTANGULAR.value == "rectangular"


class TestSensorConfig:

    def test_frozen(self):
        from humeris.domain.sensor import SensorType, SensorConfig
        cfg = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        with pytest.raises(AttributeError):
            cfg.half_angle_deg = 45.0

    def test_circular_fields(self):
        from humeris.domain.sensor import SensorType, SensorConfig
        cfg = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        assert cfg.sensor_type == SensorType.CIRCULAR
        assert cfg.half_angle_deg == 30.0

    def test_rectangular_fields(self):
        from humeris.domain.sensor import SensorType, SensorConfig
        cfg = SensorConfig(
            sensor_type=SensorType.RECTANGULAR,
            half_angle_deg=0.0,
            cross_track_half_angle_deg=20.0,
            along_track_half_angle_deg=10.0,
        )
        assert cfg.cross_track_half_angle_deg == 20.0
        assert cfg.along_track_half_angle_deg == 10.0


class TestGroundFootprint:

    def test_frozen(self):
        from humeris.domain.sensor import GroundFootprint
        fp = GroundFootprint(
            center_lat_deg=0.0, center_lon_deg=0.0,
            swath_width_km=100.0, along_track_extent_km=50.0,
        )
        with pytest.raises(AttributeError):
            fp.swath_width_km = 200.0

    def test_fields(self):
        from humeris.domain.sensor import GroundFootprint
        fp = GroundFootprint(
            center_lat_deg=10.0, center_lon_deg=20.0,
            swath_width_km=100.0, along_track_extent_km=50.0,
        )
        assert fp.center_lat_deg == 10.0
        assert fp.center_lon_deg == 20.0
        assert fp.swath_width_km == 100.0
        assert fp.along_track_extent_km == 50.0


class TestSensorAccessResult:

    def test_frozen(self):
        from humeris.domain.sensor import SensorAccessResult
        r = SensorAccessResult(is_visible=True, off_nadir_angle_deg=5.0, ground_range_km=100.0)
        with pytest.raises(AttributeError):
            r.is_visible = False

    def test_fields(self):
        from humeris.domain.sensor import SensorAccessResult
        r = SensorAccessResult(is_visible=True, off_nadir_angle_deg=5.0, ground_range_km=100.0)
        assert r.is_visible is True
        assert r.off_nadir_angle_deg == 5.0
        assert r.ground_range_km == 100.0


# --- compute_swath_width tests ---

class TestSwathWidth:

    def test_known_geometry(self):
        """Computed swath width matches formula for 500 km, 30 deg half-angle."""
        from humeris.domain.sensor import compute_swath_width
        swath = compute_swath_width(500.0, 30.0)
        # Manual check: should be several hundred km
        assert 400 < swath < 1200

    def test_increases_with_altitude(self):
        from humeris.domain.sensor import compute_swath_width
        sw_low = compute_swath_width(300.0, 30.0)
        sw_high = compute_swath_width(800.0, 30.0)
        assert sw_high > sw_low

    def test_increases_with_angle(self):
        from humeris.domain.sensor import compute_swath_width
        sw_narrow = compute_swath_width(500.0, 10.0)
        sw_wide = compute_swath_width(500.0, 40.0)
        assert sw_wide > sw_narrow

    def test_small_angle_near_flat_earth(self):
        """Small angle approximation: swath ≈ 2*h*tan(alpha)."""
        from humeris.domain.sensor import compute_swath_width
        h = 500.0
        alpha = 1.0  # 1 degree
        swath = compute_swath_width(h, alpha)
        flat_approx = 2 * h * math.tan(math.radians(alpha))
        # Within 5% for small angles at low altitude
        assert abs(swath - flat_approx) / flat_approx < 0.05

    def test_zero_altitude_raises(self):
        from humeris.domain.sensor import compute_swath_width
        with pytest.raises(ValueError):
            compute_swath_width(0.0, 30.0)

    def test_negative_angle_raises(self):
        from humeris.domain.sensor import compute_swath_width
        with pytest.raises(ValueError):
            compute_swath_width(500.0, -5.0)

    def test_angle_90_raises(self):
        from humeris.domain.sensor import compute_swath_width
        with pytest.raises(ValueError):
            compute_swath_width(500.0, 90.0)


# --- compute_nadir_footprint tests ---

class TestNadirFootprint:

    def test_circular_symmetric(self):
        """Circular sensor → swath_width_km == along_track_extent_km."""
        from humeris.domain.sensor import (
            SensorType, SensorConfig, compute_nadir_footprint,
        )
        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        fp = compute_nadir_footprint(500.0, sensor)
        assert abs(fp.swath_width_km - fp.along_track_extent_km) < 0.01

    def test_rectangular_asymmetric(self):
        """Rectangular sensor → swath != along_track (different half-angles)."""
        from humeris.domain.sensor import (
            SensorType, SensorConfig, compute_nadir_footprint,
        )
        sensor = SensorConfig(
            sensor_type=SensorType.RECTANGULAR,
            half_angle_deg=0.0,
            cross_track_half_angle_deg=30.0,
            along_track_half_angle_deg=15.0,
        )
        fp = compute_nadir_footprint(500.0, sensor)
        assert abs(fp.swath_width_km - fp.along_track_extent_km) > 10

    def test_zero_altitude_raises(self):
        from humeris.domain.sensor import (
            SensorType, SensorConfig, compute_nadir_footprint,
        )
        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        with pytest.raises(ValueError):
            compute_nadir_footprint(0.0, sensor)

    def test_center_coordinates(self):
        from humeris.domain.sensor import (
            SensorType, SensorConfig, compute_nadir_footprint,
        )
        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        fp = compute_nadir_footprint(500.0, sensor, center_lat_deg=45.0, center_lon_deg=-90.0)
        assert fp.center_lat_deg == 45.0
        assert fp.center_lon_deg == -90.0

    def test_footprint_size_reasonable(self):
        """30 deg half-angle at 500 km → footprint 400-1200 km."""
        from humeris.domain.sensor import (
            SensorType, SensorConfig, compute_nadir_footprint,
        )
        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        fp = compute_nadir_footprint(500.0, sensor)
        assert 400 < fp.swath_width_km < 1200


# --- is_in_sensor_fov tests ---

class TestIsInSensorFov:

    def test_nadir_point_visible(self):
        """Point directly below satellite → visible."""
        from humeris.domain.sensor import (
            SensorType, SensorConfig, is_in_sensor_fov,
        )
        from humeris.domain.coordinate_frames import (
            gmst_rad, geodetic_to_ecef,
        )
        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        r = (OrbitalConstants.R_EARTH + 500_000)
        sat_eci = (r, 0.0, 0.0)
        gmst = gmst_rad(EPOCH)
        # Sub-satellite point: transform sat to ECEF, then to geodetic, then back
        from humeris.domain.coordinate_frames import (
            eci_to_ecef, ecef_to_geodetic,
        )
        sat_ecef, _ = eci_to_ecef(sat_eci, (0.0, 7500.0, 0.0), gmst)
        lat, lon, _ = ecef_to_geodetic(sat_ecef)
        ground_ecef = geodetic_to_ecef(lat, lon, 0.0)

        result = is_in_sensor_fov(sat_eci, ground_ecef, sensor, gmst)
        assert result.is_visible is True

    def test_far_point_not_visible(self):
        """Point on opposite side of Earth → not visible."""
        from humeris.domain.sensor import (
            SensorType, SensorConfig, is_in_sensor_fov,
        )
        from humeris.domain.coordinate_frames import (
            gmst_rad, geodetic_to_ecef, eci_to_ecef, ecef_to_geodetic,
        )
        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        r = (OrbitalConstants.R_EARTH + 500_000)
        sat_eci = (r, 0.0, 0.0)
        gmst = gmst_rad(EPOCH)
        # Compute sub-satellite point, then go to opposite side
        sat_ecef, _ = eci_to_ecef(sat_eci, (0.0, 7500.0, 0.0), gmst)
        lat, lon, _ = ecef_to_geodetic(sat_ecef)
        opposite_lon = lon + 180.0
        if opposite_lon > 180.0:
            opposite_lon -= 360.0
        ground_ecef = geodetic_to_ecef(-lat, opposite_lon, 0.0)

        result = is_in_sensor_fov(sat_eci, ground_ecef, sensor, gmst)
        assert result.is_visible is False

    def test_off_nadir_angle_correct(self):
        """Off-nadir angle approximately matches expected geometry."""
        from humeris.domain.sensor import (
            SensorType, SensorConfig, is_in_sensor_fov,
        )
        from humeris.domain.coordinate_frames import (
            gmst_rad, geodetic_to_ecef, eci_to_ecef, ecef_to_geodetic,
        )
        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=45.0)
        r = (OrbitalConstants.R_EARTH + 500_000)
        sat_eci = (r, 0.0, 0.0)
        gmst = gmst_rad(EPOCH)
        # Sub-satellite point
        sat_ecef, _ = eci_to_ecef(sat_eci, (0.0, 7500.0, 0.0), gmst)
        lat, lon, _ = ecef_to_geodetic(sat_ecef)
        ground_ecef = geodetic_to_ecef(lat, lon, 0.0)

        result = is_in_sensor_fov(sat_eci, ground_ecef, sensor, gmst)
        # Nadir point should have near-zero off-nadir angle
        assert result.off_nadir_angle_deg < 5.0

    def test_circular_fov_boundary(self):
        """Point just outside FOV → not visible."""
        from humeris.domain.sensor import (
            SensorType, SensorConfig, is_in_sensor_fov,
        )
        from humeris.domain.coordinate_frames import (
            gmst_rad, geodetic_to_ecef, eci_to_ecef, ecef_to_geodetic,
        )
        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=5.0)
        r = (OrbitalConstants.R_EARTH + 500_000)
        sat_eci = (r, 0.0, 0.0)
        gmst = gmst_rad(EPOCH)
        # A point far enough away to be outside 5 deg FOV
        sat_ecef, _ = eci_to_ecef(sat_eci, (0.0, 7500.0, 0.0), gmst)
        lat, lon, _ = ecef_to_geodetic(sat_ecef)
        # Offset by ~10 degrees lat → should be outside 5 deg half-angle
        ground_ecef = geodetic_to_ecef(lat + 10.0, lon, 0.0)
        result = is_in_sensor_fov(sat_eci, ground_ecef, sensor, gmst)
        assert result.is_visible is False

    def test_rectangular_fov_within(self):
        """Point within rectangular FOV → visible."""
        from humeris.domain.sensor import (
            SensorType, SensorConfig, is_in_sensor_fov,
        )
        from humeris.domain.coordinate_frames import (
            gmst_rad, geodetic_to_ecef, eci_to_ecef, ecef_to_geodetic,
        )
        sensor = SensorConfig(
            sensor_type=SensorType.RECTANGULAR,
            half_angle_deg=0.0,
            cross_track_half_angle_deg=30.0,
            along_track_half_angle_deg=30.0,
        )
        r = (OrbitalConstants.R_EARTH + 500_000)
        sat_eci = (r, 0.0, 0.0)
        gmst = gmst_rad(EPOCH)
        sat_ecef, _ = eci_to_ecef(sat_eci, (0.0, 7500.0, 0.0), gmst)
        lat, lon, _ = ecef_to_geodetic(sat_ecef)
        ground_ecef = geodetic_to_ecef(lat, lon, 0.0)

        result = is_in_sensor_fov(sat_eci, ground_ecef, sensor, gmst)
        assert result.is_visible is True

    def test_ground_range_positive(self):
        """Ground range > 0 for non-nadir point."""
        from humeris.domain.sensor import (
            SensorType, SensorConfig, is_in_sensor_fov,
        )
        from humeris.domain.coordinate_frames import (
            gmst_rad, geodetic_to_ecef, eci_to_ecef, ecef_to_geodetic,
        )
        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=45.0)
        r = (OrbitalConstants.R_EARTH + 500_000)
        sat_eci = (r, 0.0, 0.0)
        gmst = gmst_rad(EPOCH)
        sat_ecef, _ = eci_to_ecef(sat_eci, (0.0, 7500.0, 0.0), gmst)
        lat, lon, _ = ecef_to_geodetic(sat_ecef)
        # Slight offset from nadir
        ground_ecef = geodetic_to_ecef(lat + 2.0, lon, 0.0)
        result = is_in_sensor_fov(sat_eci, ground_ecef, sensor, gmst)
        assert result.ground_range_km > 0


# --- compute_sensor_coverage tests ---

class TestSensorCoverage:

    def test_returns_coverage_points(self):
        from humeris.domain.sensor import (
            SensorType, SensorConfig, compute_sensor_coverage,
        )
        from humeris.domain.constellation import ShellConfig, generate_walker_shell
        from humeris.domain.propagation import derive_orbital_state
        from humeris.domain.coverage import CoveragePoint

        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        shell = ShellConfig(
            altitude_km=500, inclination_deg=53,
            num_planes=2, sats_per_plane=2,
            phase_factor=1, raan_offset_deg=0, shell_name="Test",
        )
        sats = generate_walker_shell(shell)
        states = [derive_orbital_state(s, EPOCH) for s in sats]
        grid = compute_sensor_coverage(states, EPOCH, sensor, lat_step_deg=30, lon_step_deg=30)
        assert all(isinstance(p, CoveragePoint) for p in grid)

    def test_grid_size_matches_parameters(self):
        from humeris.domain.sensor import (
            SensorType, SensorConfig, compute_sensor_coverage,
        )
        from humeris.domain.constellation import ShellConfig, generate_walker_shell
        from humeris.domain.propagation import derive_orbital_state

        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        shell = ShellConfig(
            altitude_km=500, inclination_deg=53,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0, shell_name="Test",
        )
        sats = generate_walker_shell(shell)
        states = [derive_orbital_state(s, EPOCH) for s in sats]
        grid = compute_sensor_coverage(states, EPOCH, sensor, lat_step_deg=30, lon_step_deg=30)
        # lat: -90 to 90 in 30° steps = 7 rows. lon: -180 to 150 in 30° = 12 cols.
        expected = 7 * 12
        assert len(grid) == expected

    def test_narrow_fov_fewer_visible(self):
        from humeris.domain.sensor import (
            SensorType, SensorConfig, compute_sensor_coverage,
        )
        from humeris.domain.constellation import ShellConfig, generate_walker_shell
        from humeris.domain.propagation import derive_orbital_state

        shell = ShellConfig(
            altitude_km=500, inclination_deg=53,
            num_planes=6, sats_per_plane=10,
            phase_factor=1, raan_offset_deg=0, shell_name="Test",
        )
        sats = generate_walker_shell(shell)
        states = [derive_orbital_state(s, EPOCH) for s in sats]

        sensor_wide = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=45.0)
        sensor_narrow = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=5.0)

        grid_wide = compute_sensor_coverage(states, EPOCH, sensor_wide, lat_step_deg=30, lon_step_deg=30)
        grid_narrow = compute_sensor_coverage(states, EPOCH, sensor_narrow, lat_step_deg=30, lon_step_deg=30)

        visible_wide = sum(1 for p in grid_wide if p.visible_count > 0)
        visible_narrow = sum(1 for p in grid_narrow if p.visible_count > 0)
        assert visible_narrow <= visible_wide

    def test_empty_states_all_zero(self):
        from humeris.domain.sensor import (
            SensorType, SensorConfig, compute_sensor_coverage,
        )

        sensor = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
        grid = compute_sensor_coverage([], EPOCH, sensor, lat_step_deg=30, lon_step_deg=30)
        assert all(p.visible_count == 0 for p in grid)


# --- Domain purity ---

class TestSensorPurity:

    def test_sensor_module_pure(self):
        """sensor.py must only import from stdlib and domain."""
        import humeris.domain.sensor as mod

        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_top = {"math", "numpy", "dataclasses", "typing", "enum", "datetime"}
        allowed_internal_prefix = "humeris"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed_top or alias.name.startswith(allowed_internal_prefix), \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed_top or node.module.startswith(allowed_internal_prefix), \
                        f"Forbidden import from: {node.module}"
