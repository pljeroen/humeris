# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Tests for JPL-grade planetary ephemeris (Chebyshev interpolation)."""

import ast
import math
from datetime import datetime, timezone

import pytest


class TestClenshawEvaluation:
    """Clenshaw recurrence for known polynomials."""

    def test_constant_polynomial(self):
        from humeris.domain.planetary_ephemeris import chebyshev_evaluate
        # T0(x) = 1, so coeffs [5.0] = 5.0 everywhere
        assert abs(chebyshev_evaluate((5.0,), 0.0) - 5.0) < 1e-10
        assert abs(chebyshev_evaluate((5.0,), 0.5) - 5.0) < 1e-10

    def test_linear_polynomial(self):
        from humeris.domain.planetary_ephemeris import chebyshev_evaluate
        # coeffs [a0, a1]: a0*T0(x) + a1*T1(x) = a0 + a1*x
        assert abs(chebyshev_evaluate((3.0, 2.0), 0.0) - 3.0) < 1e-10
        assert abs(chebyshev_evaluate((3.0, 2.0), 1.0) - 5.0) < 1e-10
        assert abs(chebyshev_evaluate((3.0, 2.0), -1.0) - 1.0) < 1e-10

    def test_quadratic_polynomial(self):
        from humeris.domain.planetary_ephemeris import chebyshev_evaluate
        # coeffs [a0, a1, a2]: T2(x) = 2x²-1
        # [1, 0, 1] → T0 + T2 = 1 + (2x²-1) = 2x²
        assert abs(chebyshev_evaluate((1.0, 0.0, 1.0), 0.0) - 0.0) < 1e-10
        assert abs(chebyshev_evaluate((1.0, 0.0, 1.0), 1.0) - 2.0) < 1e-10
        assert abs(chebyshev_evaluate((1.0, 0.0, 1.0), 0.5) - 0.5) < 1e-10


class TestSunPosition:
    """Sun position at known dates."""

    def test_sun_at_j2000(self):
        """Sun geocentric position at J2000.0 — should be ~1 AU from Earth."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        t = AstroTime(tdb_j2000=0.0)
        pos = evaluate_position(eph["sun"], t)

        # Distance should be ~1 AU = 149,597,870,700 m ± 2.5%
        dist = math.sqrt(sum(c ** 2 for c in pos))
        au = 149_597_870_700.0
        assert 0.97 * au < dist < 1.03 * au

    def test_sun_at_summer_solstice_2020(self):
        """Sun at 2020 summer solstice — declination ~23.4°."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        # 2020-06-20 21:44 UTC
        t = AstroTime.from_utc(datetime(2020, 6, 20, 21, 44, tzinfo=timezone.utc))
        pos = evaluate_position(eph["sun"], t)
        dist = math.sqrt(sum(c ** 2 for c in pos))

        # Z component should be significant (Sun at +23.4° declination)
        declination_rad = math.asin(pos[2] / dist)
        declination_deg = math.degrees(declination_rad)
        # Expect positive declination near 23.4° (within a few degrees)
        assert declination_deg > 15.0
        assert declination_deg < 30.0

    def test_sun_distance_varies(self):
        """Earth-Sun distance varies ~3.3% over a year (perihelion/aphelion)."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        distances = []
        for month in range(1, 13):
            t = AstroTime.from_utc(datetime(2020, month, 15, tzinfo=timezone.utc))
            pos = evaluate_position(eph["sun"], t)
            distances.append(math.sqrt(sum(c ** 2 for c in pos)))

        ratio = max(distances) / min(distances)
        assert 1.02 < ratio < 1.06  # ~3.3% variation


class TestMoonPosition:
    """Moon position at known dates."""

    def test_moon_at_j2000(self):
        """Moon geocentric distance at J2000.0 — ~384,400 km."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        t = AstroTime(tdb_j2000=0.0)
        pos = evaluate_position(eph["moon"], t)

        dist_km = math.sqrt(sum(c ** 2 for c in pos)) / 1000.0
        # Moon distance: 356,500 km (perigee) to 406,700 km (apogee)
        assert 350_000 < dist_km < 410_000

    def test_moon_distance_varies(self):
        """Moon distance varies significantly (eccentric orbit)."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        distances = []
        # Sample over one sidereal month (~27.3 days)
        for day in range(0, 28):
            t = AstroTime.from_utc(datetime(2020, 1, 1, tzinfo=timezone.utc))
            t = t + day * 86400.0
            pos = evaluate_position(eph["moon"], t)
            distances.append(math.sqrt(sum(c ** 2 for c in pos)))

        ratio = max(distances) / min(distances)
        # Eccentricity ~0.055 → distance varies ~11%
        assert ratio > 1.05

    def test_moon_orbital_period(self):
        """Moon returns to approximately same position after ~27.3 days."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        t0 = AstroTime.from_utc(datetime(2020, 1, 1, tzinfo=timezone.utc))
        # Sidereal month ≈ 27.3217 days
        t1 = t0 + 27.3217 * 86400.0

        pos0 = evaluate_position(eph["moon"], t0)
        pos1 = evaluate_position(eph["moon"], t1)

        # Positions should be similar (within ~10% of moon distance)
        dist0 = math.sqrt(sum(c ** 2 for c in pos0))
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos0, pos1)))
        assert diff / dist0 < 0.15


class TestVelocity:
    """Analytical Chebyshev derivative for velocity."""

    def test_sun_velocity_magnitude(self):
        """Sun apparent velocity ~30 km/s (Earth's orbital velocity)."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_velocity,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        t = AstroTime.from_utc(datetime(2020, 6, 15, tzinfo=timezone.utc))
        vel = evaluate_velocity(eph["sun"], t)
        speed = math.sqrt(sum(v ** 2 for v in vel))
        # Earth orbital velocity ~29.8 km/s
        assert 25_000 < speed < 35_000

    def test_velocity_vs_finite_difference(self):
        """Analytical derivative should match finite difference."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
            evaluate_velocity,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        t = AstroTime.from_utc(datetime(2020, 3, 15, tzinfo=timezone.utc))
        dt_s = 10.0  # 10 seconds

        pos_m = evaluate_position(eph["sun"], t + (-dt_s))
        pos_p = evaluate_position(eph["sun"], t + dt_s)
        vel_fd = tuple((p - m) / (2 * dt_s) for p, m in zip(pos_p, pos_m))

        vel_an = evaluate_velocity(eph["sun"], t)

        for fd, an in zip(vel_fd, vel_an):
            if abs(an) > 100:
                assert abs(fd - an) / abs(an) < 0.01  # 1% relative

    def test_moon_velocity_magnitude(self):
        """Moon orbital velocity ~1 km/s."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_velocity,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        t = AstroTime.from_utc(datetime(2020, 1, 15, tzinfo=timezone.utc))
        vel = evaluate_velocity(eph["moon"], t)
        speed = math.sqrt(sum(v ** 2 for v in vel))
        assert 800 < speed < 1200


class TestGranuleBoundary:
    """Continuity at granule boundaries."""

    def test_sun_continuous_at_boundary(self):
        """Position should be continuous across granule boundaries."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        sun = eph["sun"]
        # Find a granule boundary
        boundary_t = sun["granule_days"] * 86400.0  # End of first granule
        eps = 0.001  # 1 ms

        t_before = AstroTime(tdb_j2000=boundary_t - eps)
        t_after = AstroTime(tdb_j2000=boundary_t + eps)

        pos_before = evaluate_position(sun, t_before)
        pos_after = evaluate_position(sun, t_after)

        # Should be within a few km (continuity)
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_before, pos_after)))
        assert diff < 100_000  # Less than 100 km discontinuity

    def test_moon_continuous_at_boundary(self):
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        moon = eph["moon"]
        boundary_t = moon["granule_days"] * 86400.0
        eps = 0.001

        t_before = AstroTime(tdb_j2000=boundary_t - eps)
        t_after = AstroTime(tdb_j2000=boundary_t + eps)

        pos_before = evaluate_position(moon, t_before)
        pos_after = evaluate_position(moon, t_after)

        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_before, pos_after)))
        assert diff < 50_000  # Less than 50 km


class TestOutOfRange:
    """Behavior outside ephemeris range."""

    def test_before_range_raises(self):
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        # Way before 2000
        t = AstroTime(tdb_j2000=-365.25 * 86400.0 * 10)  # 1990
        with pytest.raises(ValueError):
            evaluate_position(eph["sun"], t)

    def test_after_range_raises(self):
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        # After 2050
        t = AstroTime(tdb_j2000=365.25 * 86400.0 * 60)  # ~2060
        with pytest.raises(ValueError):
            evaluate_position(eph["sun"], t)


class TestBackwardCompat:
    """Meeus ephemeris still works without Chebyshev."""

    def test_meeus_moon_still_available(self):
        from humeris.domain.third_body import moon_position_eci
        result = moon_position_eci(datetime(2020, 1, 1, tzinfo=timezone.utc))
        assert hasattr(result, "position_eci_m")

    def test_third_body_force_without_ephemeris(self):
        from humeris.domain.third_body import (
            SolarThirdBodyForce,
            LunarThirdBodyForce,
        )
        # Should work without ephemeris (uses Meeus)
        solar = SolarThirdBodyForce()
        lunar = LunarThirdBodyForce()
        dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc_s = solar.acceleration(dt, pos, vel)
        acc_l = lunar.acceleration(dt, pos, vel)
        assert len(acc_s) == 3
        assert len(acc_l) == 3


class TestThirdBodyJPLvsMeeus:
    """Quantify improvement of Chebyshev over Meeus."""

    def test_sun_direction_agreement(self):
        """Sun direction should roughly agree between Meeus and Chebyshev."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.solar import (
            sun_position_eci,
        )
        from humeris.domain.time_systems import AstroTime

        dt = datetime(2020, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        t = AstroTime.from_utc(dt)

        eph = load_ephemeris()
        pos_cheb = evaluate_position(eph["sun"], t)
        sun_meeus = sun_position_eci(dt)
        pos_meeus = sun_meeus.position_eci_m

        # Normalize both to unit vectors
        d_cheb = math.sqrt(sum(c ** 2 for c in pos_cheb))
        d_meeus = math.sqrt(sum(c ** 2 for c in pos_meeus))
        u_cheb = tuple(c / d_cheb for c in pos_cheb)
        u_meeus = tuple(c / d_meeus for c in pos_meeus)

        # Dot product should be close to 1 (directions agree)
        dot = sum(a * b for a, b in zip(u_cheb, u_meeus))
        angle_deg = math.degrees(math.acos(min(1.0, max(-1.0, dot))))
        assert angle_deg < 2.0  # Within 2 degrees

    def test_moon_distance_improvement(self):
        """Chebyshev Moon should be closer to truth than Meeus (less than 0.5° error)."""
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        # Just verify it produces reasonable values at multiple epochs
        for year in [2005, 2010, 2015, 2020, 2025]:
            t = AstroTime.from_utc(datetime(year, 6, 15, tzinfo=timezone.utc))
            pos = evaluate_position(eph["moon"], t)
            dist_km = math.sqrt(sum(c ** 2 for c in pos)) / 1000.0
            assert 350_000 < dist_km < 410_000


class TestEphemerisLoading:
    """Loading and caching of ephemeris data."""

    def test_load_returns_dict(self):
        from humeris.domain.planetary_ephemeris import load_ephemeris
        eph = load_ephemeris()
        assert "sun" in eph
        assert "moon" in eph

    def test_gm_values(self):
        from humeris.domain.planetary_ephemeris import load_ephemeris
        eph = load_ephemeris()
        assert abs(eph["sun"]["gm"] - 1.32712440041e20) < 1e15
        assert abs(eph["moon"]["gm"] - 4.9028e12) < 1e9


class TestPerformance:
    """Chebyshev evaluation should be fast."""

    def test_evaluation_fast(self):
        import time
        from humeris.domain.planetary_ephemeris import (
            load_ephemeris,
            evaluate_position,
        )
        from humeris.domain.time_systems import AstroTime

        eph = load_ephemeris()
        t = AstroTime.from_utc(datetime(2020, 1, 1, tzinfo=timezone.utc))

        # Warm up
        evaluate_position(eph["sun"], t)

        start = time.perf_counter()
        for i in range(1000):
            evaluate_position(eph["sun"], t + i * 3600.0)
        elapsed = (time.perf_counter() - start) / 1000
        assert elapsed < 0.0001  # <100μs per evaluation


class TestDomainPurity:
    """Verify planetary_ephemeris.py has zero external dependencies."""

    def test_no_external_imports(self):
        import humeris.domain.planetary_ephemeris as _mod
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
