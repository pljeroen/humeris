# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Invariant tests for geodetic / ECEF coordinate conversions.

These verify round-trip consistency and output bounds for the
WGS84 geodetic transformations.

Invariants C1-C5 from the formal invariant specification.
"""

import math

import pytest

from constellation_generator.domain.coordinate_frames import (
    geodetic_to_ecef,
    ecef_to_geodetic,
)
from constellation_generator.domain.orbital_mechanics import OrbitalConstants


# Deterministic parameter grid (lat_deg, lon_deg, alt_m)
_MID_LATITUDE_POINTS = [
    (0.0, 0.0, 0.0, "Equator prime meridian sea level"),
    (45.0, 90.0, 0.0, "Mid-lat 45N 90E"),
    (-45.0, -90.0, 0.0, "Mid-lat 45S 90W"),
    (30.0, 120.0, 500_000.0, "30N 120E 500km"),
    (-30.0, -120.0, 300_000.0, "-30 -120 300km"),
    (60.0, 0.0, 800_000.0, "60N prime meridian 800km"),
    (0.0, 180.0, 0.0, "Equator antimeridian"),
    (0.0, -180.0, 0.0, "Equator -180"),
    (10.0, 45.0, 35_786_000.0, "GEO altitude"),
]

_POLE_POINTS = [
    (89.0, 0.0, 0.0, "Near north pole"),
    (-89.0, 0.0, 0.0, "Near south pole"),
    (90.0, 0.0, 0.0, "North pole"),
    (-90.0, 0.0, 0.0, "South pole"),
    (90.0, 45.0, 500_000.0, "North pole lon=45 alt=500km"),
    (-90.0, -90.0, 300_000.0, "South pole lon=-90 alt=300km"),
]


class TestC1RoundTripMidLatitude:
    """C1: geodetic -> ECEF -> geodetic round-trip (mid-latitude, tight tolerance)."""

    @pytest.mark.parametrize("lat,lon,alt,label", _MID_LATITUDE_POINTS)
    def test_roundtrip(self, lat, lon, alt, label):
        ecef = geodetic_to_ecef(lat, lon, alt)
        lat_r, lon_r, alt_r = ecef_to_geodetic(ecef)
        assert abs(lat_r - lat) < 1e-6, f"{label}: lat {lat_r} vs {lat}"
        # Handle antimeridian wrap
        lon_diff = abs(lon_r - lon)
        if lon_diff > 180:
            lon_diff = 360 - lon_diff
        assert lon_diff < 1e-6, f"{label}: lon {lon_r} vs {lon}"
        assert abs(alt_r - alt) < 0.01, f"{label}: alt {alt_r} vs {alt}"  # 1cm


class TestC2RoundTripNearPoles:
    """C2: Round-trip near poles (looser tolerance for lon/alt)."""

    @pytest.mark.parametrize("lat,lon,alt,label", _POLE_POINTS)
    def test_roundtrip_pole(self, lat, lon, alt, label):
        ecef = geodetic_to_ecef(lat, lon, alt)
        lat_r, lon_r, alt_r = ecef_to_geodetic(ecef)
        # Latitude: tight
        assert abs(lat_r - lat) < 1e-4, f"{label}: lat {lat_r} vs {lat}"
        # At exact poles, longitude is undefined — skip lon check
        if abs(lat) < 89.99:
            lon_diff = abs(lon_r - lon)
            if lon_diff > 180:
                lon_diff = 360 - lon_diff
            assert lon_diff < 1e-3, f"{label}: lon {lon_r} vs {lon}"
        # Altitude: looser near poles
        assert abs(alt_r - alt) < 1.0, f"{label}: alt {alt_r} vs {alt}"  # 1m


class TestC3LatitudeBounds:
    """C3: Returned latitude always in [-90, +90]."""

    def test_latitude_bounds_grid(self):
        """Scan ECEF points on a sphere and verify lat bounds."""
        r = OrbitalConstants.R_EARTH_EQUATORIAL
        for theta in range(0, 360, 15):
            for phi in range(-89, 90, 15):
                theta_rad = math.radians(theta)
                phi_rad = math.radians(phi)
                x = r * math.cos(phi_rad) * math.cos(theta_rad)
                y = r * math.cos(phi_rad) * math.sin(theta_rad)
                z = r * math.sin(phi_rad)
                lat, lon, alt = ecef_to_geodetic((x, y, z))
                assert -90.0 <= lat <= 90.0, f"lat={lat} out of bounds"

    def test_latitude_bounds_at_altitude(self):
        """Points at LEO altitude."""
        r = OrbitalConstants.R_EARTH + 500_000
        for angle_deg in range(0, 360, 30):
            angle = math.radians(angle_deg)
            x = r * math.cos(angle)
            y = 0.0
            z = r * math.sin(angle)
            lat, lon, alt = ecef_to_geodetic((x, y, z))
            assert -90.0 <= lat <= 90.0


class TestC4LongitudeNormalization:
    """C4: Longitude output convention is (-180, 180]."""

    def test_longitude_convention(self):
        r = OrbitalConstants.R_EARTH_EQUATORIAL
        for theta in range(0, 360, 10):
            theta_rad = math.radians(theta)
            x = r * math.cos(theta_rad)
            y = r * math.sin(theta_rad)
            z = 0.0
            lat, lon, alt = ecef_to_geodetic((x, y, z))
            assert -180.0 <= lon <= 180.0, f"lon={lon} violates [-180, 180] convention"


class TestC5EquatorSanity:
    """C5: At equator, alt=0, geodetic_to_ecef should be (R_equatorial, 0, 0)."""

    def test_equator_prime_meridian(self):
        x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
        r_eq = OrbitalConstants.R_EARTH_EQUATORIAL
        assert abs(x - r_eq) < 1.0, f"x={x}, expected {r_eq}"
        assert abs(y) < 1.0, f"y={y}, expected 0"
        assert abs(z) < 1.0, f"z={z}, expected 0"

    def test_equator_90_east(self):
        x, y, z = geodetic_to_ecef(0.0, 90.0, 0.0)
        r_eq = OrbitalConstants.R_EARTH_EQUATORIAL
        assert abs(x) < 1.0, f"x={x}"
        assert abs(y - r_eq) < 1.0, f"y={y}, expected {r_eq}"
        assert abs(z) < 1.0, f"z={z}"

    def test_north_pole(self):
        x, y, z = geodetic_to_ecef(90.0, 0.0, 0.0)
        r_polar = OrbitalConstants.R_EARTH_POLAR
        assert abs(x) < 1.0, f"x={x}"
        assert abs(y) < 1.0, f"y={y}"
        assert abs(z - r_polar) < 1.0, f"z={z}, expected {r_polar}"
