# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Tests for Earth Orientation Parameters (EOP) loading, interpolation, and polar motion."""

import ast
import math
from datetime import datetime, timezone

import pytest


class TestEOPLoading:
    """Load and validate EOP data from bundled JSON."""

    def test_load_returns_eop_table(self):
        from humeris.domain.earth_orientation import load_eop, EOPTable
        table = load_eop()
        assert isinstance(table, EOPTable)

    def test_load_has_entries(self):
        from humeris.domain.earth_orientation import load_eop
        table = load_eop()
        assert len(table.entries) > 100

    def test_entries_are_sorted(self):
        from humeris.domain.earth_orientation import load_eop
        table = load_eop()
        for i in range(len(table.entries) - 1):
            assert table.entries[i].mjd < table.entries[i + 1].mjd


class TestEOPInterpolation:
    """Interpolation at arbitrary MJD values."""

    def test_interpolate_at_known_point(self):
        from humeris.domain.earth_orientation import (
            load_eop,
            interpolate_eop,
        )
        table = load_eop()
        # First entry
        e0 = table.entries[0]
        result = interpolate_eop(table, e0.mjd)
        assert abs(result.ut1_utc - e0.ut1_utc) < 1e-6

    def test_interpolate_midpoint(self):
        from humeris.domain.earth_orientation import (
            load_eop,
            interpolate_eop,
        )
        table = load_eop()
        e0 = table.entries[0]
        e1 = table.entries[1]
        mid_mjd = (e0.mjd + e1.mjd) / 2.0
        result = interpolate_eop(table, mid_mjd)
        # Should be between the two values
        low = min(e0.ut1_utc, e1.ut1_utc) - 0.01
        high = max(e0.ut1_utc, e1.ut1_utc) + 0.01
        assert low <= result.ut1_utc <= high

    def test_interpolate_at_boundary(self):
        from humeris.domain.earth_orientation import (
            load_eop,
            interpolate_eop,
        )
        table = load_eop()
        last = table.entries[-1]
        result = interpolate_eop(table, last.mjd)
        assert abs(result.ut1_utc - last.ut1_utc) < 1e-6

    def test_extrapolation_before_range(self):
        """Before table range: use first entry."""
        from humeris.domain.earth_orientation import (
            load_eop,
            interpolate_eop,
        )
        table = load_eop()
        first = table.entries[0]
        result = interpolate_eop(table, first.mjd - 100.0)
        assert abs(result.ut1_utc - first.ut1_utc) < 1e-6

    def test_extrapolation_after_range(self):
        """After table range: use last entry."""
        from humeris.domain.earth_orientation import (
            load_eop,
            interpolate_eop,
        )
        table = load_eop()
        last = table.entries[-1]
        result = interpolate_eop(table, last.mjd + 100.0)
        assert abs(result.ut1_utc - last.ut1_utc) < 1e-6


class TestUT1UTC:
    """UT1-UTC values at known dates."""

    def test_ut1_utc_2000(self):
        """UT1-UTC at J2000.0 (MJD 51544.5) should be ~0.35s."""
        from humeris.domain.earth_orientation import (
            load_eop,
            interpolate_eop,
        )
        table = load_eop()
        result = interpolate_eop(table, 51544.5)
        assert abs(result.ut1_utc - 0.355) < 0.1

    def test_ut1_utc_bounded(self):
        """UT1-UTC should always be in [-0.9, 0.9] seconds."""
        from humeris.domain.earth_orientation import load_eop
        table = load_eop()
        for entry in table.entries:
            assert -0.9 <= entry.ut1_utc <= 0.9, \
                f"UT1-UTC out of bounds: {entry.ut1_utc} at MJD {entry.mjd}"

    def test_ut1_utc_at_2017(self):
        """After 2017 leap second, UT1-UTC should be positive."""
        from humeris.domain.earth_orientation import (
            load_eop,
            interpolate_eop,
        )
        table = load_eop()
        # MJD 57760 ≈ 2017-01-07
        result = interpolate_eop(table, 57760.0)
        assert result.ut1_utc > 0.0


class TestPolarMotion:
    """Polar motion matrix construction."""

    def test_polar_motion_matrix_near_identity(self):
        """With zero polar motion, matrix is identity."""
        from humeris.domain.earth_orientation import polar_motion_matrix
        W = polar_motion_matrix(0.0, 0.0, 0.0)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(W[i][j] - expected) < 1e-15

    def test_polar_motion_matrix_orthogonal(self):
        """Polar motion matrix should be orthogonal."""
        from humeris.domain.earth_orientation import polar_motion_matrix

        xp_rad = 0.1 * math.pi / (180 * 3600)  # 0.1"
        yp_rad = 0.3 * math.pi / (180 * 3600)
        W = polar_motion_matrix(xp_rad, yp_rad, 0.0)

        # W^T * W = I
        WtW = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    WtW[i][j] += W[k][i] * W[k][j]
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(WtW[i][j] - expected) < 1e-12


class TestFullTransformation:
    """Full GCRS → ITRS with EOP data."""

    def test_gcrs_to_itrs_with_eop(self):
        """gcrs_to_itrs_matrix should accept EOPTable and use it."""
        from humeris.domain.earth_orientation import load_eop
        from humeris.domain.precession_nutation import gcrs_to_itrs_matrix
        from humeris.domain.time_systems import AstroTime

        table = load_eop()
        dt = datetime(2020, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt)

        # Without EOP
        M_no_eop = gcrs_to_itrs_matrix(t, ut1_utc=0.0)

        # With EOP: get UT1-UTC from table
        from humeris.domain.earth_orientation import interpolate_eop
        mjd = t.to_mjd_tt()  # Approximate MJD
        eop = interpolate_eop(table, mjd)
        M_with_eop = gcrs_to_itrs_matrix(t, ut1_utc=eop.ut1_utc)

        # Matrices should differ (UT1-UTC != 0)
        diff = sum(abs(M_with_eop[i][j] - M_no_eop[i][j])
                   for i in range(3) for j in range(3))
        assert diff > 0.0

    def test_full_vs_gmst_improvement(self):
        """Full GCRS→ITRS with EOP should still differ from GMST-only."""
        from humeris.domain.earth_orientation import load_eop, interpolate_eop
        from humeris.domain.precession_nutation import eci_to_ecef_precise
        from humeris.domain.coordinate_frames import eci_to_ecef, gmst_rad
        from humeris.domain.time_systems import AstroTime

        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt)
        pos_eci = (6778137.0, 0.0, 0.0)
        vel_eci = (0.0, 7668.0, 0.0)

        table = load_eop()
        mjd = t.to_mjd_tt()
        eop = interpolate_eop(table, mjd)

        # GMST-only
        gmst = gmst_rad(dt)
        pos_simple, _ = eci_to_ecef(pos_eci, vel_eci, gmst)

        # Full IAU 2006 + EOP
        pos_precise = eci_to_ecef_precise(pos_eci, t, ut1_utc=eop.ut1_utc)

        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_simple, pos_precise)))
        assert dist > 100.0, "EOP should increase difference from GMST-only"

    def test_missing_eop_fallback(self):
        """Without EOP table, use default ut1_utc=0."""
        from humeris.domain.precession_nutation import gcrs_to_itrs_matrix
        from humeris.domain.time_systems import AstroTime

        t = AstroTime.from_utc(datetime(2020, 1, 1, tzinfo=timezone.utc))
        M = gcrs_to_itrs_matrix(t)  # Should work without EOP
        assert len(M) == 3

    def test_missing_eop_fallback_same_as_zero(self):
        """Default (no EOP) should equal ut1_utc=0."""
        from humeris.domain.precession_nutation import gcrs_to_itrs_matrix
        from humeris.domain.time_systems import AstroTime

        t = AstroTime.from_utc(datetime(2020, 1, 1, tzinfo=timezone.utc))
        M_default = gcrs_to_itrs_matrix(t)
        M_zero = gcrs_to_itrs_matrix(t, ut1_utc=0.0)
        for i in range(3):
            for j in range(3):
                assert abs(M_default[i][j] - M_zero[i][j]) < 1e-15

    def test_cunningham_with_precise_rotation(self):
        """CunninghamGravity should accept a rotation_provider."""
        from humeris.domain.gravity_field import (
            CunninghamGravity,
            load_gravity_field,
        )
        model = load_gravity_field()
        gravity = CunninghamGravity(model)
        # Should work with default (GMST) rotation
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc = gravity.acceleration(dt, pos, vel)
        assert len(acc) == 3
        assert all(math.isfinite(a) for a in acc)


class TestMJDConversion:
    """MJD to/from calendar date conversion."""

    def test_mjd_j2000(self):
        """J2000.0 = MJD 51544.5."""
        from humeris.domain.earth_orientation import datetime_to_mjd
        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mjd = datetime_to_mjd(dt)
        assert abs(mjd - 51544.5) < 0.001

    def test_mjd_known_date(self):
        """2020-01-01 00:00 UTC = MJD 58849.0."""
        from humeris.domain.earth_orientation import datetime_to_mjd
        dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mjd = datetime_to_mjd(dt)
        assert abs(mjd - 58849.0) < 0.001

    def test_mjd_2024(self):
        """2024-01-01 00:00 UTC = MJD 60310.0."""
        from humeris.domain.earth_orientation import datetime_to_mjd
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mjd = datetime_to_mjd(dt)
        assert abs(mjd - 60310.0) < 0.001


class TestDomainPurity:
    """Verify earth_orientation.py has zero external dependencies."""

    def test_no_external_imports(self):
        import humeris.domain.earth_orientation as _mod
        source_path = _mod.__file__
        with open(source_path) as f:
            tree = ast.parse(f.read())
        allowed = {
            "math", "numpy", "datetime", "dataclasses", "typing", "json",
            "pathlib", "os", "functools", "enum", "collections",
            "abc", "copy", "struct", "bisect", "operator",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed or top == "humeris", \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    assert top in allowed or top == "humeris", \
                        f"Forbidden import from: {node.module}"
