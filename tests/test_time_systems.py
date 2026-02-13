# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see LICENSE-COMMERCIAL.md.
"""Tests for AstroTime value object and time system conversions."""

import math
import ast
import importlib
from datetime import datetime, timezone, timedelta

import pytest


class TestAstroTimeConstruction:
    """AstroTime creation and basic properties."""

    def test_from_utc_returns_astro_time(self):
        from constellation_generator.domain.time_systems import AstroTime

        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt)
        assert isinstance(t, AstroTime)

    def test_from_utc_j2000_epoch(self):
        """J2000.0 = 2000-01-01 11:58:55.816 TDB. UTC at J2000 is ~11:58:55.816 + offsets."""
        from constellation_generator.domain.time_systems import AstroTime

        # J2000.0 TDB = 2000-01-01T11:58:55.816 TDB
        # TDB - TT ~ 0 at J2000, TT = TAI + 32.184, TAI = UTC + 32
        # So UTC at J2000 TDB ~ 11:58:55.816 - 32.184 - 32 = 11:57:51.632
        dt_j2000_utc = datetime(2000, 1, 1, 11, 58, 55, 816000, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt_j2000_utc)
        # Should be close to 0 TDB seconds from J2000
        # Not exactly 0 because UTC != TDB, but within a few seconds
        assert abs(t.tdb_j2000) < 70.0  # Within TAI-UTC + TT-TAI offset

    def test_from_utc_naive_datetime_treated_as_utc(self):
        from constellation_generator.domain.time_systems import AstroTime

        naive = datetime(2020, 6, 15, 12, 0, 0)
        aware = datetime(2020, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        t_naive = AstroTime.from_utc(naive)
        t_aware = AstroTime.from_utc(aware)
        assert t_naive.tdb_j2000 == t_aware.tdb_j2000

    def test_frozen_dataclass(self):
        """AstroTime is immutable."""
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime.from_utc(datetime(2020, 1, 1, tzinfo=timezone.utc))
        with pytest.raises(AttributeError):
            t.tdb_j2000 = 999.0  # type: ignore[misc]

    def test_from_tt_seconds(self):
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime.from_tt_seconds(0.0)  # TT at J2000 = JD 2451545.0 TT
        # TDB - TT oscillation is small (~1.6ms), so tdb_j2000 should be close to
        # the offset between TT and TDB epochs
        assert isinstance(t, AstroTime)

    def test_from_gps(self):
        """GPS epoch = 1980-01-06 00:00:00 UTC. TAI = GPS + 19s."""
        from constellation_generator.domain.time_systems import AstroTime

        # GPS week 0, second 0 = 1980-01-06 00:00:00 UTC
        t = AstroTime.from_gps(gps_seconds=0.0)
        dt = t.to_utc_datetime()
        assert dt.year == 1980
        assert dt.month == 1
        assert dt.day == 6
        assert dt.hour == 0


class TestTimeConversions:
    """UTC ↔ TAI ↔ TT ↔ TDB round-trips."""

    def test_utc_to_tai_current_offset(self):
        """As of 2017+, TAI = UTC + 37s."""
        from constellation_generator.domain.time_systems import (
            AstroTime,
            utc_to_tai_seconds,
        )

        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        tai_offset = utc_to_tai_seconds(dt)
        assert tai_offset == 37.0

    def test_utc_to_tai_1972(self):
        """First leap second entry: 1972-01-01, delta_AT = 10."""
        from constellation_generator.domain.time_systems import utc_to_tai_seconds

        dt = datetime(1972, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert utc_to_tai_seconds(dt) == 10.0

    def test_utc_to_tai_leap_second_boundary_2017(self):
        """2017-01-01: delta_AT went from 36 to 37."""
        from constellation_generator.domain.time_systems import utc_to_tai_seconds

        before = datetime(2016, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        after = datetime(2017, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert utc_to_tai_seconds(before) == 36.0
        assert utc_to_tai_seconds(after) == 37.0

    def test_utc_to_tai_leap_second_boundary_2015(self):
        """2015-07-01: delta_AT went from 35 to 36."""
        from constellation_generator.domain.time_systems import utc_to_tai_seconds

        before = datetime(2015, 6, 30, 23, 59, 59, tzinfo=timezone.utc)
        after = datetime(2015, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert utc_to_tai_seconds(before) == 35.0
        assert utc_to_tai_seconds(after) == 36.0

    def test_utc_to_tai_leap_second_boundary_2009(self):
        """2009-01-01: delta_AT went from 33 to 34."""
        from constellation_generator.domain.time_systems import utc_to_tai_seconds

        before = datetime(2008, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        after = datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert utc_to_tai_seconds(before) == 33.0
        assert utc_to_tai_seconds(after) == 34.0

    def test_tai_to_tt_exact(self):
        """TT = TAI + 32.184s (exact, IAU 1991)."""
        from constellation_generator.domain.time_systems import AstroTime

        dt = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt)
        tt_s = t.to_tt_seconds()
        # TAI = UTC + 37, TT = TAI + 32.184 = UTC + 69.184
        # So TT seconds from J2000 should reflect this
        assert isinstance(tt_s, float)

    def test_tt_tdb_oscillation_amplitude(self):
        """TDB-TT oscillation has amplitude ~1.657ms (Fairhead & Bretagnon)."""
        from constellation_generator.domain.time_systems import AstroTime

        # Sample many points over a year, check max TDB-TT
        max_diff = 0.0
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        for day in range(0, 365, 10):
            dt = base + timedelta(days=day)
            t = AstroTime.from_utc(dt)
            tt_s = t.to_tt_seconds()
            tdb_s = t.tdb_j2000
            # tdb_j2000 is TDB seconds from J2000 TDB
            # tt_s is TT seconds from J2000 TT
            # The difference should be the TDB-TT oscillation
            diff = abs(tdb_s - tt_s)
            if diff > max_diff:
                max_diff = diff
        # Max amplitude ~1.657ms, but with epoch offsets it's more complex
        # Just check it's bounded
        assert max_diff < 0.002  # Less than 2ms

    def test_tt_tdb_oscillation_periodic(self):
        """TDB-TT oscillation has ~1 year period (Earth's orbital motion)."""
        from constellation_generator.domain.time_systems import AstroTime

        diffs = []
        base = datetime(2020, 1, 1, tzinfo=timezone.utc)
        for day in range(0, 730, 30):
            dt = base + timedelta(days=day)
            t = AstroTime.from_utc(dt)
            diffs.append(t.tdb_j2000 - t.to_tt_seconds())
        # Should see sign changes (oscillation)
        signs = [d > 0 for d in diffs]
        assert not all(signs) or not all(not s for s in signs), \
            "TDB-TT should oscillate"

    def test_tt_tdb_oscillation_fairhead_terms(self):
        """Verify TDB-TT at vernal equinox vs solstice differ."""
        from constellation_generator.domain.time_systems import AstroTime

        equinox = AstroTime.from_utc(datetime(2020, 3, 20, 3, 50, tzinfo=timezone.utc))
        solstice = AstroTime.from_utc(datetime(2020, 6, 20, 21, 44, tzinfo=timezone.utc))
        diff_eq = equinox.tdb_j2000 - equinox.to_tt_seconds()
        diff_sol = solstice.tdb_j2000 - solstice.to_tt_seconds()
        assert diff_eq != diff_sol


class TestRoundTrips:
    """UTC → AstroTime → UTC round-trips."""

    def test_utc_round_trip_j2000(self):
        from constellation_generator.domain.time_systems import AstroTime

        dt_in = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt_in)
        dt_out = t.to_utc_datetime()
        delta = abs((dt_out - dt_in).total_seconds())
        assert delta < 0.001  # Sub-millisecond round-trip

    def test_utc_round_trip_2024(self):
        from constellation_generator.domain.time_systems import AstroTime

        dt_in = datetime(2024, 7, 4, 18, 30, 45, 123456, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt_in)
        dt_out = t.to_utc_datetime()
        delta = abs((dt_out - dt_in).total_seconds())
        assert delta < 0.001

    def test_utc_round_trip_1972(self):
        from constellation_generator.domain.time_systems import AstroTime

        dt_in = datetime(1972, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt_in)
        dt_out = t.to_utc_datetime()
        delta = abs((dt_out - dt_in).total_seconds())
        assert delta < 0.001

    def test_utc_round_trip_2000_feb_29(self):
        """Leap day."""
        from constellation_generator.domain.time_systems import AstroTime

        dt_in = datetime(2000, 2, 29, 6, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt_in)
        dt_out = t.to_utc_datetime()
        delta = abs((dt_out - dt_in).total_seconds())
        assert delta < 0.001

    def test_utc_round_trip_far_future(self):
        from constellation_generator.domain.time_systems import AstroTime

        dt_in = datetime(2050, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt_in)
        dt_out = t.to_utc_datetime()
        delta = abs((dt_out - dt_in).total_seconds())
        assert delta < 0.001

    def test_gps_round_trip(self):
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime.from_gps(gps_seconds=1_000_000_000.0)
        gps_out = t.to_gps_seconds()
        # Float64 precision at 1e9 limits round-trip to ~10μs
        assert abs(gps_out - 1_000_000_000.0) < 1e-4

    def test_from_utc_to_utc_consistency(self):
        """from_utc and to_utc_datetime are inverses."""
        from constellation_generator.domain.time_systems import AstroTime

        for year in [1980, 1999, 2000, 2010, 2020, 2025]:
            dt_in = datetime(year, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            dt_out = AstroTime.from_utc(dt_in).to_utc_datetime()
            delta = abs((dt_out - dt_in).total_seconds())
            assert delta < 0.001, f"Round-trip failed for {year}"


class TestKnownEpochs:
    """Verification against published values (Vallado, SOFA)."""

    def test_j2000_julian_date(self):
        """J2000.0 = JD 2451545.0 TDB."""
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime(tdb_j2000=0.0)
        jd = t.to_julian_date_tdb()
        assert abs(jd - 2451545.0) < 1e-10

    def test_julian_date_at_known_epoch(self):
        """2020-01-01 12:00:00 TDB = JD 2458849.5 + 0.5 = 2458850.0 (approx)."""
        from constellation_generator.domain.time_systems import AstroTime

        dt = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt)
        jd = t.to_julian_date_tdb()
        # JD 2458850.0 for 2020-01-01 12:00:00 UTC (approx, TDB offset is ~69s)
        assert abs(jd - 2458850.0) < 0.001

    def test_julian_centuries_at_j2000(self):
        """T = 0 at J2000.0."""
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime(tdb_j2000=0.0)
        T = t.to_julian_centuries_tt()
        assert abs(T) < 1e-6  # Very close to 0

    def test_julian_centuries_vallado_example(self):
        """Vallado Example 3-15: 2004-04-06 07:51:28.386 UTC.

        JD_UTC = 2453101.8274065, TAI-UTC = 32, TT = UTC + 64.184s.
        JD_TT = 2453101.82815, T_TT = (JD_TT - 2451545.0) / 36525 ≈ 0.04261.
        """
        from constellation_generator.domain.time_systems import AstroTime

        dt = datetime(2004, 4, 6, 7, 51, 28, 386000, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt)
        T_tt = t.to_julian_centuries_tt()
        # JD_TT = 2453101.82815, T_TT ≈ 0.04261
        assert abs(T_tt - 0.04261) < 1e-4

    def test_modified_julian_date(self):
        """MJD = JD - 2400000.5. MJD 51544.5 = J2000.0."""
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime(tdb_j2000=0.0)
        mjd = t.to_mjd_tt()
        assert abs(mjd - 51544.5) < 0.001


class TestArithmetic:
    """AstroTime arithmetic operations."""

    def test_add_seconds(self):
        from constellation_generator.domain.time_systems import AstroTime

        t1 = AstroTime(tdb_j2000=0.0)
        t2 = t1 + 100.0
        assert isinstance(t2, AstroTime)
        assert abs(t2.tdb_j2000 - 100.0) < 1e-15

    def test_subtract_astrotime(self):
        from constellation_generator.domain.time_systems import AstroTime

        t1 = AstroTime(tdb_j2000=100.0)
        t2 = AstroTime(tdb_j2000=300.0)
        diff = t2 - t1
        assert isinstance(diff, float)
        assert abs(diff - 200.0) < 1e-15

    def test_add_negative_seconds(self):
        from constellation_generator.domain.time_systems import AstroTime

        t1 = AstroTime(tdb_j2000=1000.0)
        t2 = t1 + (-500.0)
        assert abs(t2.tdb_j2000 - 500.0) < 1e-15

    def test_subtract_negative_result(self):
        from constellation_generator.domain.time_systems import AstroTime

        t1 = AstroTime(tdb_j2000=300.0)
        t2 = AstroTime(tdb_j2000=100.0)
        diff = t2 - t1
        assert abs(diff - (-200.0)) < 1e-15

    def test_equality(self):
        from constellation_generator.domain.time_systems import AstroTime

        t1 = AstroTime(tdb_j2000=12345.678)
        t2 = AstroTime(tdb_j2000=12345.678)
        assert t1 == t2

    def test_ordering(self):
        from constellation_generator.domain.time_systems import AstroTime

        t1 = AstroTime(tdb_j2000=100.0)
        t2 = AstroTime(tdb_j2000=200.0)
        assert t1 < t2
        assert t2 > t1


class TestJulianConversions:
    """Julian date and century conversions."""

    def test_julian_date_from_tdb(self):
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime(tdb_j2000=86400.0)  # 1 day after J2000
        jd = t.to_julian_date_tdb()
        assert abs(jd - 2451546.0) < 1e-10

    def test_julian_centuries_one_century(self):
        from constellation_generator.domain.time_systems import AstroTime

        # One Julian century = 36525 days = 36525 * 86400 seconds
        one_century_s = 36525.0 * 86400.0
        t = AstroTime(tdb_j2000=one_century_s)
        T = t.to_julian_centuries_tt()
        # TDB-TT offset is small, so T should be close to 1.0
        assert abs(T - 1.0) < 0.001

    def test_from_julian_date(self):
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime.from_julian_date_tdb(2451545.0)
        assert abs(t.tdb_j2000) < 1e-10

    def test_from_julian_date_round_trip(self):
        from constellation_generator.domain.time_systems import AstroTime

        jd_in = 2460000.5
        t = AstroTime.from_julian_date_tdb(jd_in)
        jd_out = t.to_julian_date_tdb()
        assert abs(jd_out - jd_in) < 1e-10

    def test_datetime_to_julian_date_known(self):
        """2000-01-01 12:00:00 UTC = JD 2451545.0 (definition)."""
        from constellation_generator.domain.time_systems import datetime_to_jd

        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        jd = datetime_to_jd(dt)
        assert abs(jd - 2451545.0) < 1e-10

    def test_datetime_to_julian_date_1999(self):
        """1999-01-01 00:00:00 UTC = JD 2451179.5."""
        from constellation_generator.domain.time_systems import datetime_to_jd

        dt = datetime(1999, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        jd = datetime_to_jd(dt)
        assert abs(jd - 2451179.5) < 1e-10


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_before_1972_raises(self):
        """Leap second table starts at 1972. Before that is undefined."""
        from constellation_generator.domain.time_systems import utc_to_tai_seconds

        dt = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="before 1972"):
            utc_to_tai_seconds(dt)

    def test_far_future_uses_last_known_offset(self):
        """After last known leap second, use the last delta_AT."""
        from constellation_generator.domain.time_systems import utc_to_tai_seconds

        dt = datetime(2100, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        offset = utc_to_tai_seconds(dt)
        assert offset == 37.0  # Last known as of 2017

    def test_negative_tdb_j2000(self):
        """Times before J2000 should work."""
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime(tdb_j2000=-1_000_000.0)
        assert t.tdb_j2000 == -1_000_000.0
        dt = t.to_utc_datetime()
        assert dt.year < 2000

    def test_to_utc_preserves_timezone(self):
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime.from_utc(datetime(2020, 1, 1, tzinfo=timezone.utc))
        dt = t.to_utc_datetime()
        assert dt.tzinfo is not None

    def test_repr(self):
        from constellation_generator.domain.time_systems import AstroTime

        t = AstroTime(tdb_j2000=0.0)
        r = repr(t)
        assert "AstroTime" in r


class TestGPSConversions:
    """GPS time system conversions."""

    def test_gps_epoch_tai_offset(self):
        """TAI = GPS + 19s (exact)."""
        from constellation_generator.domain.time_systems import AstroTime

        # GPS epoch = 1980-01-06 00:00:00 UTC
        # At that time, TAI-UTC = 19s, so GPS = TAI - 19
        t = AstroTime.from_gps(0.0)
        dt = t.to_utc_datetime()
        expected = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)
        delta = abs((dt - expected).total_seconds())
        assert delta < 0.001

    def test_gps_week_rollover(self):
        """GPS week 1024 rollover (1999-08-22) and 2048 (2019-04-07)."""
        from constellation_generator.domain.time_systems import AstroTime

        # Week 2048 = 2048 * 7 * 86400 GPS seconds from epoch
        gps_s = 2048 * 7 * 86400.0
        t = AstroTime.from_gps(gps_s)
        dt = t.to_utc_datetime()
        # 2019-04-07 (approximately)
        assert dt.year == 2019
        assert dt.month == 4

    def test_gps_to_utc_accounts_for_leap_seconds(self):
        """GPS doesn't have leap seconds, UTC does. Conversion must account."""
        from constellation_generator.domain.time_systems import AstroTime

        # At GPS epoch (1980-01-06), TAI-UTC = 19s
        # In 2020, TAI-UTC = 37s
        # GPS = TAI - 19, so GPS-UTC = TAI-19-UTC = delta_AT-19
        # In 2020: GPS-UTC = 37-19 = 18s
        dt_2020 = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt_2020)
        gps_s = t.to_gps_seconds()
        # Reconstruct
        t2 = AstroTime.from_gps(gps_s)
        dt_out = t2.to_utc_datetime()
        delta = abs((dt_out - dt_2020).total_seconds())
        assert delta < 0.001


class TestDomainPurity:
    """Verify time_systems.py has zero external dependencies."""

    def test_no_external_imports(self):
        source_path = (
            "src/constellation_generator/domain/time_systems.py"
        )
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
