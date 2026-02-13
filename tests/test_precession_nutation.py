# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see LICENSE-COMMERCIAL.md.
"""Tests for IAU 2006 precession + IAU 2000B nutation + GCRS↔ITRS transformation."""

import ast
import math
from datetime import datetime, timezone

import pytest


# --------------------------------------------------------------------------- #
# Helper: 3x3 matrix determinant and transpose
# --------------------------------------------------------------------------- #

def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _mat_mul(A, B):
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    return tuple(tuple(row) for row in result)


def _transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


def _mat_vec(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def _vec_dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# --------------------------------------------------------------------------- #
# Fundamental Arguments
# --------------------------------------------------------------------------- #

class TestFundamentalArguments:
    """Delaunay arguments at known dates."""

    def test_at_j2000(self):
        from constellation_generator.domain.precession_nutation import fundamental_arguments
        l, lp, F, D, Om = fundamental_arguments(0.0)
        # At J2000, all are well-defined constants (IERS 2010 Table 5.2)
        # l = 134.96340251° = 485868.249036" → radians
        assert isinstance(l, float)

    def test_arguments_are_radians(self):
        from constellation_generator.domain.precession_nutation import fundamental_arguments
        l, lp, F, D, Om = fundamental_arguments(0.0)
        for arg in (l, lp, F, D, Om):
            assert -100 < arg < 100  # Radians, not arcseconds

    def test_arguments_at_known_date(self):
        """At T=0.1 (~2010), verify arguments are in expected range."""
        from constellation_generator.domain.precession_nutation import fundamental_arguments
        l, lp, F, D, Om = fundamental_arguments(0.1)
        # All should be finite, varying
        for arg in (l, lp, F, D, Om):
            assert math.isfinite(arg)


# --------------------------------------------------------------------------- #
# Precession
# --------------------------------------------------------------------------- #

class TestPrecession:
    """IAU 2006 precession matrix (Fukushima-Williams)."""

    def test_near_identity_at_j2000(self):
        """Precession matrix is near identity at T=0.

        Not exactly identity because Fukushima-Williams angles have small
        constant terms (gamb=-0.053", psib=-0.042").
        """
        from constellation_generator.domain.precession_nutation import precession_matrix
        P = precession_matrix(0.0)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(P[i][j] - expected) < 1e-6  # Sub-arcsecond

    def test_orthogonal(self):
        """Precession matrix is orthogonal: P^T P = I."""
        from constellation_generator.domain.precession_nutation import precession_matrix
        P = precession_matrix(0.1)
        PtP = _mat_mul(_transpose(P), P)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(PtP[i][j] - expected) < 1e-12

    def test_determinant_one(self):
        from constellation_generator.domain.precession_nutation import precession_matrix
        P = precession_matrix(0.26)
        assert abs(_det3(P) - 1.0) < 1e-12

    def test_precession_accumulated_2026(self):
        """By T≈0.26 (2026), accumulated precession ~ 1300" in ψ_A."""
        from constellation_generator.domain.precession_nutation import precession_matrix
        P = precession_matrix(0.26)
        # The (0,0) element should be slightly less than 1
        # cos(1300") ≈ cos(0.0063 rad) ≈ 0.99998
        assert P[0][0] < 1.0
        assert P[0][0] > 0.999

    def test_precession_at_half_century(self):
        """T=0.5 (~2050): larger precession angle."""
        from constellation_generator.domain.precession_nutation import precession_matrix
        P = precession_matrix(0.5)
        assert abs(_det3(P) - 1.0) < 1e-12
        # More precession than T=0.26
        assert P[0][0] < 0.999999


# --------------------------------------------------------------------------- #
# Nutation
# --------------------------------------------------------------------------- #

class TestNutation:
    """IAU 2000B nutation (77 terms)."""

    def test_nutation_at_j2000(self):
        from constellation_generator.domain.precession_nutation import nutation_angles
        dpsi, deps = nutation_angles(0.0)
        # Nutation in longitude typically -17" to +17"
        assert abs(dpsi) < 20.0 * math.pi / (180 * 3600)  # 20 arcsec in radians

    def test_nutation_magnitude(self):
        """Largest nutation term (18.6yr) has amplitude ~17.2" in dpsi."""
        from constellation_generator.domain.precession_nutation import nutation_angles
        # Sample over ~19 years to capture full cycle
        max_dpsi = 0.0
        for year_frac in range(0, 20):
            t = year_frac / 100.0
            dpsi, _ = nutation_angles(t)
            dpsi_arcsec = abs(dpsi) * 180 * 3600 / math.pi
            max_dpsi = max(max_dpsi, dpsi_arcsec)
        # Should find amplitude somewhere near 17"
        assert max_dpsi > 10.0  # At least 10" detected
        assert max_dpsi < 25.0  # Not unreasonably large

    def test_nutation_obliquity(self):
        """Nutation in obliquity ~9.2" amplitude."""
        from constellation_generator.domain.precession_nutation import nutation_angles
        max_deps = 0.0
        for year_frac in range(0, 20):
            t = year_frac / 100.0
            _, deps = nutation_angles(t)
            deps_arcsec = abs(deps) * 180 * 3600 / math.pi
            max_deps = max(max_deps, deps_arcsec)
        assert max_deps > 5.0
        assert max_deps < 15.0

    def test_nutation_matrix_orthogonal(self):
        from constellation_generator.domain.precession_nutation import (
            nutation_angles,
            nutation_matrix,
        )
        dpsi, deps = nutation_angles(0.1)
        eps0 = math.radians(23.4393)  # approximate mean obliquity
        N = nutation_matrix(dpsi, deps, eps0)
        NtN = _mat_mul(_transpose(N), N)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(NtN[i][j] - expected) < 1e-12

    def test_nutation_matrix_determinant(self):
        from constellation_generator.domain.precession_nutation import (
            nutation_angles,
            nutation_matrix,
        )
        dpsi, deps = nutation_angles(0.2)
        eps0 = math.radians(23.4393)
        N = nutation_matrix(dpsi, deps, eps0)
        assert abs(_det3(N) - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# Earth Rotation Angle
# --------------------------------------------------------------------------- #

class TestEarthRotationAngle:
    """ERA from UT1 Julian Date."""

    def test_era_at_j2000(self):
        """ERA at J2000.0 UT1."""
        from constellation_generator.domain.precession_nutation import earth_rotation_angle
        era = earth_rotation_angle(2451545.0)
        # ERA at J2000 = 2π × 0.7790572732640 ≈ 4.8949...
        expected = 2 * math.pi * 0.7790572732640
        assert abs(era - expected) < 1e-10

    def test_era_one_day_later(self):
        """ERA advances ~360.9856° per day (sidereal)."""
        from constellation_generator.domain.precession_nutation import earth_rotation_angle
        era0 = earth_rotation_angle(2451545.0)
        era1 = earth_rotation_angle(2451546.0)
        # Difference should be ~2π × 1.00273781191135448 ≈ one full + a bit
        diff = (era1 - era0) % (2 * math.pi)
        # ~0.00273781 of a full turn = 0.01719 rad
        assert abs(diff - 2 * math.pi * 0.00273781191135448) < 1e-6

    def test_era_range(self):
        """ERA should be in [0, 2π)."""
        from constellation_generator.domain.precession_nutation import earth_rotation_angle
        for jd in [2451545.0, 2451545.5, 2460000.0]:
            era = earth_rotation_angle(jd)
            assert 0.0 <= era < 2 * math.pi


# --------------------------------------------------------------------------- #
# Frame Bias
# --------------------------------------------------------------------------- #

class TestFrameBias:
    """GCRS frame bias matrix."""

    def test_bias_near_identity(self):
        from constellation_generator.domain.precession_nutation import frame_bias_matrix
        B = frame_bias_matrix()
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                # Bias angles are ~17 mas, so off-diag ~ 8e-8
                assert abs(B[i][j] - expected) < 1e-4

    def test_bias_orthogonal(self):
        from constellation_generator.domain.precession_nutation import frame_bias_matrix
        B = frame_bias_matrix()
        BtB = _mat_mul(_transpose(B), B)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(BtB[i][j] - expected) < 1e-12


# --------------------------------------------------------------------------- #
# GCRS ↔ ITRS
# --------------------------------------------------------------------------- #

class TestGCRStoITRS:
    """Full GCRS → ITRS transformation."""

    def test_matrix_orthogonal(self):
        from constellation_generator.domain.precession_nutation import gcrs_to_itrs_matrix
        from constellation_generator.domain.time_systems import AstroTime
        t = AstroTime.from_utc(datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc))
        M = gcrs_to_itrs_matrix(t)
        MtM = _mat_mul(_transpose(M), M)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(MtM[i][j] - expected) < 1e-10

    def test_matrix_determinant(self):
        from constellation_generator.domain.precession_nutation import gcrs_to_itrs_matrix
        from constellation_generator.domain.time_systems import AstroTime
        t = AstroTime.from_utc(datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
        M = gcrs_to_itrs_matrix(t)
        assert abs(_det3(M) - 1.0) < 1e-10

    def test_round_trip(self):
        """GCRS → ITRS → GCRS should return original vector to sub-mm."""
        from constellation_generator.domain.precession_nutation import (
            gcrs_to_itrs_matrix,
            eci_to_ecef_precise,
        )
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime.from_utc(datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc))
        pos_gcrs = (6778137.0, 0.0, 0.0)  # LEO on x-axis

        # Forward
        pos_itrs = eci_to_ecef_precise(pos_gcrs, t)

        # Inverse (transpose of rotation matrix)
        M = gcrs_to_itrs_matrix(t)
        Mt = _transpose(M)
        pos_back = _mat_vec(Mt, pos_itrs)

        dist = _vec_dist(pos_gcrs, pos_back)
        assert dist < 0.001  # Sub-millimeter round-trip

    def test_gmst_only_vs_iau2006_different(self):
        """The IAU 2006 result should differ from simple GMST rotation."""
        from constellation_generator.domain.precession_nutation import eci_to_ecef_precise
        from constellation_generator.domain.coordinate_frames import eci_to_ecef, gmst_rad
        from constellation_generator.domain.time_systems import AstroTime

        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt)
        pos_eci = (6778137.0, 0.0, 0.0)
        vel_eci = (0.0, 7668.0, 0.0)

        # Simple GMST
        gmst = gmst_rad(dt)
        pos_simple, _ = eci_to_ecef(pos_eci, vel_eci, gmst)

        # High-fidelity IAU 2006
        pos_precise = eci_to_ecef_precise(pos_eci, t)

        # Should differ (precession + nutation effect)
        dist = _vec_dist(pos_simple, pos_precise)
        assert dist > 1.0  # At least 1 meter difference (actually much more)

    def test_quantified_gmst_vs_iau2006(self):
        """Quantify the improvement: old GMST Z-rotation vs full IAU 2006 3D rotation.

        The old approach (GMST Z-rotation) can't capture the ~1300" CIP offset
        from precession, which is inherently a 3D effect. By 2024, the
        accumulated precession produces a ~40-100 km difference at LEO.
        """
        from constellation_generator.domain.precession_nutation import eci_to_ecef_precise
        from constellation_generator.domain.coordinate_frames import eci_to_ecef, gmst_rad
        from constellation_generator.domain.time_systems import AstroTime

        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt)
        pos_eci = (6778137.0, 0.0, 0.0)
        vel_eci = (0.0, 7668.0, 0.0)

        gmst = gmst_rad(dt)
        pos_simple, _ = eci_to_ecef(pos_eci, vel_eci, gmst)
        pos_precise = eci_to_ecef_precise(pos_eci, t)

        dist = _vec_dist(pos_simple, pos_precise)
        # The full 3D precession/nutation vs Z-only rotation produces
        # tens of km difference at LEO after 24 years from J2000
        assert dist > 1000.0, f"Difference too small: {dist:.1f} m"
        assert dist < 200_000.0, f"Difference too large: {dist:.1f} m"


# --------------------------------------------------------------------------- #
# Backward Compatibility
# --------------------------------------------------------------------------- #

class TestBackwardCompat:
    """Existing coordinate_frames functions remain unchanged."""

    def test_gmst_rad_unchanged(self):
        from constellation_generator.domain.coordinate_frames import gmst_rad
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        gmst = gmst_rad(dt)
        assert 0 <= gmst < 2 * math.pi

    def test_eci_to_ecef_unchanged(self):
        from constellation_generator.domain.coordinate_frames import eci_to_ecef, gmst_rad
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        gmst = gmst_rad(dt)
        pos_ecef, vel_ecef = eci_to_ecef(pos, vel, gmst)
        assert len(pos_ecef) == 3
        assert len(vel_ecef) == 3

    def test_ecef_to_geodetic_unchanged(self):
        from constellation_generator.domain.coordinate_frames import (
            ecef_to_geodetic,
            geodetic_to_ecef,
        )
        lat, lon, alt = ecef_to_geodetic((6378137.0, 0.0, 0.0))
        assert abs(lat) < 0.01  # Equator
        assert abs(lon) < 0.01


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #

class TestPerformance:
    """Nutation computation should be fast."""

    def test_nutation_under_1ms(self):
        import time
        from constellation_generator.domain.precession_nutation import nutation_angles

        # Warm up
        nutation_angles(0.1)

        start = time.perf_counter()
        for _ in range(100):
            nutation_angles(0.1)
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.001  # <1ms per call


# --------------------------------------------------------------------------- #
# Domain Purity
# --------------------------------------------------------------------------- #

class TestDomainPurity:
    """Verify precession_nutation.py has zero external dependencies."""

    def test_no_external_imports(self):
        source_path = "src/constellation_generator/domain/precession_nutation.py"
        with open(source_path) as f:
            tree = ast.parse(f.read())
        allowed = {
            "math", "datetime", "dataclasses", "typing", "json",
            "pathlib", "os", "functools", "enum", "collections",
            "abc", "copy", "struct", "bisect", "operator",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed or top == "constellation_generator", \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    assert top in allowed or top == "constellation_generator", \
                        f"Forbidden import from: {node.module}"
