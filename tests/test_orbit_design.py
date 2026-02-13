# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for orbit design utilities (SSO, frozen, repeat ground track)."""
import ast
import math
from datetime import datetime, timezone

import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.orbit_design import (
    FrozenOrbitDesign,
    RepeatGroundTrackDesign,
    SSODesign,
    design_frozen_orbit,
    design_repeat_ground_track,
    design_sso_orbit,
)


# ── Dataclass tests ───────────────────────────────────────────────

class TestSSODesign:

    def test_frozen(self):
        """SSODesign is immutable."""
        d = SSODesign(altitude_km=500, inclination_deg=97.4, raan_deg=0.0, ltan_hours=10.5)
        with pytest.raises(AttributeError):
            d.altitude_km = 600

    def test_fields(self):
        """SSODesign exposes expected fields."""
        d = SSODesign(altitude_km=500, inclination_deg=97.4, raan_deg=45.0, ltan_hours=10.5)
        assert d.altitude_km == 500
        assert d.inclination_deg == 97.4
        assert d.raan_deg == 45.0
        assert d.ltan_hours == 10.5


class TestFrozenOrbitDesign:

    def test_frozen(self):
        """FrozenOrbitDesign is immutable."""
        d = FrozenOrbitDesign(
            altitude_km=800, inclination_deg=98.6,
            eccentricity=0.001, arg_perigee_deg=90.0,
        )
        with pytest.raises(AttributeError):
            d.eccentricity = 0.0

    def test_fields(self):
        """FrozenOrbitDesign exposes expected fields."""
        d = FrozenOrbitDesign(
            altitude_km=800, inclination_deg=98.6,
            eccentricity=0.001, arg_perigee_deg=90.0,
        )
        assert d.altitude_km == 800
        assert d.arg_perigee_deg == 90.0


class TestRepeatGroundTrackDesign:

    def test_frozen(self):
        """RepeatGroundTrackDesign is immutable."""
        d = RepeatGroundTrackDesign(
            semi_major_axis_m=6.9e6, altitude_km=550,
            inclination_deg=97.0, repeat_days=1,
            repeat_revolutions=15, revolutions_per_day=15.0,
        )
        with pytest.raises(AttributeError):
            d.repeat_days = 2


# ── SSO design ────────────────────────────────────────────────────

class TestDesignSSO:

    def test_500km_inclination(self):
        """SSO at 500km → inclination ≈ 97.4° (matches sso_inclination_deg)."""
        from humeris.domain.orbital_mechanics import sso_inclination_deg

        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sso = design_sso_orbit(500, 10.5, epoch)
        expected_inc = sso_inclination_deg(500)
        assert sso.inclination_deg == pytest.approx(expected_inc, abs=0.01)

    def test_ltan_preserved(self):
        """LTAN in output matches requested LTAN."""
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sso = design_sso_orbit(500, 10.5, epoch)
        assert sso.ltan_hours == 10.5

    def test_raan_in_valid_range(self):
        """RAAN is in [0, 360) range."""
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sso = design_sso_orbit(500, 10.5, epoch)
        assert 0 <= sso.raan_deg < 360

    def test_dawn_dusk_raan(self):
        """LTAN 6:00 (dawn-dusk) → RAAN ≈ sun_RA - 90°."""
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sso = design_sso_orbit(500, 6.0, epoch)
        from humeris.domain.solar import sun_position_eci
        sun = sun_position_eci(epoch)
        expected_raan = (math.degrees(sun.right_ascension_rad) - 90.0) % 360
        # Allow ~5° tolerance due to simplified algorithm
        diff = abs(sso.raan_deg - expected_raan) % 360
        if diff > 180:
            diff = 360 - diff
        assert diff < 5.0


# ── Frozen orbit ──────────────────────────────────────────────────

class TestDesignFrozenOrbit:

    def test_800km_98deg_eccentricity(self):
        """Frozen orbit 800km, 98° → small positive eccentricity."""
        f = design_frozen_orbit(800, 98.0)
        assert f.eccentricity > 0
        assert f.eccentricity < 0.01

    def test_arg_perigee_90_or_270(self):
        """Frozen orbit argument of perigee is 90° or 270°."""
        f = design_frozen_orbit(800, 98.0)
        assert f.arg_perigee_deg in (90.0, 270.0)

    def test_equatorial_eccentricity_near_zero(self):
        """Equatorial orbit → eccentricity ≈ 0 (sin(0)=0)."""
        f = design_frozen_orbit(800, 0.0)
        assert abs(f.eccentricity) < 1e-10


# ── Repeat ground track ──────────────────────────────────────────

class TestDesignRepeatGroundTrack:

    def test_1day_15rev_altitude(self):
        """1-day, 15 revolutions → altitude ≈ 560-590 km."""
        rgt = design_repeat_ground_track(97.0, 1, 15)
        assert 540 < rgt.altitude_km < 610

    def test_higher_revolutions_lower_altitude(self):
        """More revolutions per day → lower altitude."""
        rgt_15 = design_repeat_ground_track(97.0, 1, 15)
        rgt_16 = design_repeat_ground_track(97.0, 1, 16)
        assert rgt_16.altitude_km < rgt_15.altitude_km

    def test_raises_on_zero_days(self):
        """Raises ValueError for 0 days."""
        with pytest.raises(ValueError):
            design_repeat_ground_track(97.0, 0, 15)

    def test_raises_on_zero_revolutions(self):
        """Raises ValueError for 0 revolutions."""
        with pytest.raises(ValueError):
            design_repeat_ground_track(97.0, 1, 0)


# ── Domain purity ─────────────────────────────────────────────────

class TestOrbitDesignPurity:

    def test_orbit_design_module_pure(self):
        """orbit_design.py must only import stdlib modules."""
        import humeris.domain.orbit_design as mod

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
