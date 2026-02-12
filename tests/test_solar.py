# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for analytical solar ephemeris."""
import ast
import math
from datetime import datetime, timezone

import pytest

from constellation_generator.domain.solar import (
    AU_METERS,
    SunPosition,
    julian_centuries_j2000,
    solar_declination_rad,
    sun_position_eci,
)


# ── SunPosition dataclass ─────────────────────────────────────────

class TestSunPosition:

    def test_frozen(self):
        """SunPosition is immutable."""
        sp = SunPosition(
            position_eci_m=(1.0, 0.0, 0.0),
            right_ascension_rad=0.0,
            declination_rad=0.0,
            distance_m=AU_METERS,
        )
        with pytest.raises(AttributeError):
            sp.distance_m = 0.0

    def test_fields(self):
        """SunPosition exposes expected fields."""
        sp = SunPosition(
            position_eci_m=(1.0, 2.0, 3.0),
            right_ascension_rad=0.1,
            declination_rad=0.2,
            distance_m=1e11,
        )
        assert sp.position_eci_m == (1.0, 2.0, 3.0)
        assert sp.right_ascension_rad == 0.1
        assert sp.declination_rad == 0.2
        assert sp.distance_m == 1e11


# ── Julian centuries ───────────────────────────────────────────────

class TestJulianCenturies:

    def test_j2000_epoch_is_zero(self):
        """J2000.0 (2000-01-01 12:00:00 UTC) → T = 0.0."""
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert julian_centuries_j2000(j2000) == pytest.approx(0.0, abs=1e-12)

    def test_one_century_later(self):
        """2100-01-01 12:00:00 UTC → T ≈ 1.0 (Julian century = 36525 days)."""
        epoch = datetime(2100, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        T = julian_centuries_j2000(epoch)
        assert T == pytest.approx(1.0, abs=0.003)


# ── Sun position ──────────────────────────────────────────────────

class TestSunPositionECI:

    def test_vernal_equinox_ra_near_zero(self):
        """At vernal equinox (~March 20), Sun RA ≈ 0°."""
        equinox = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sp = sun_position_eci(equinox)
        # RA should be near 0 (mod 2*pi), within ~2 degrees
        ra_deg = math.degrees(sp.right_ascension_rad) % 360
        assert ra_deg < 3.0 or ra_deg > 357.0

    def test_vernal_equinox_dec_near_zero(self):
        """At vernal equinox, Sun declination ≈ 0°."""
        equinox = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sp = sun_position_eci(equinox)
        assert abs(math.degrees(sp.declination_rad)) < 2.0

    def test_summer_solstice_dec_positive(self):
        """At summer solstice (~June 21), Sun declination ≈ +23.44°."""
        solstice = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        sp = sun_position_eci(solstice)
        dec_deg = math.degrees(sp.declination_rad)
        assert abs(dec_deg - 23.44) < 1.0

    def test_winter_solstice_dec_negative(self):
        """At winter solstice (~Dec 21), Sun declination ≈ -23.44°."""
        solstice = datetime(2026, 12, 21, 12, 0, 0, tzinfo=timezone.utc)
        sp = sun_position_eci(solstice)
        dec_deg = math.degrees(sp.declination_rad)
        assert abs(dec_deg + 23.44) < 1.0

    def test_distance_about_one_au(self):
        """Sun distance ≈ 1 AU (within 2%)."""
        epoch = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        sp = sun_position_eci(epoch)
        assert abs(sp.distance_m - AU_METERS) / AU_METERS < 0.02

    def test_position_vector_magnitude_matches_distance(self):
        """ECI position vector magnitude matches reported distance."""
        epoch = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        sp = sun_position_eci(epoch)
        x, y, z = sp.position_eci_m
        mag = math.sqrt(x**2 + y**2 + z**2)
        assert mag == pytest.approx(sp.distance_m, rel=1e-10)

    def test_eci_signs_at_summer_solstice(self):
        """At summer solstice, Sun is in +y hemisphere (RA ≈ 90°), z > 0."""
        solstice = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        sp = sun_position_eci(solstice)
        _, y, z = sp.position_eci_m
        assert y > 0, "Sun y should be positive at summer solstice"
        assert z > 0, "Sun z should be positive at summer solstice"

    def test_monotonic_ra_progression(self):
        """Sun RA increases monotonically over a year (mod 2pi)."""
        epochs = [
            datetime(2026, m, 15, 12, 0, 0, tzinfo=timezone.utc)
            for m in range(1, 12)
        ]
        ras = [sun_position_eci(e).right_ascension_rad for e in epochs]
        # Unwrap RA to handle 2pi crossover
        unwrapped = [ras[0]]
        for i in range(1, len(ras)):
            diff = ras[i] - ras[i - 1]
            if diff < -math.pi:
                diff += 2 * math.pi
            unwrapped.append(unwrapped[-1] + diff)
        for i in range(len(unwrapped) - 1):
            assert unwrapped[i + 1] > unwrapped[i], (
                f"RA not monotonic between month {i+1} and {i+2}"
            )


# ── Solar declination convenience ─────────────────────────────────

class TestSolarDeclination:

    def test_matches_sun_position(self):
        """solar_declination_rad matches sun_position_eci declination."""
        epoch = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)
        sp = sun_position_eci(epoch)
        dec = solar_declination_rad(epoch)
        assert dec == pytest.approx(sp.declination_rad, abs=1e-12)


# ── Domain purity ─────────────────────────────────────────────────

class TestSolarPurity:

    def test_solar_module_pure(self):
        """solar.py must only import stdlib modules."""
        import constellation_generator.domain.solar as mod

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
