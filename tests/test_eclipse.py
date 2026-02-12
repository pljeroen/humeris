"""Tests for eclipse/shadow prediction."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.solar import AU_METERS, sun_position_eci
from constellation_generator.domain.eclipse import (
    EclipseEvent,
    EclipseType,
    compute_beta_angle,
    compute_eclipse_windows,
    eclipse_fraction,
    is_eclipsed,
)


R_EARTH = OrbitalConstants.R_EARTH


# ── EclipseType enum ──────────────────────────────────────────────

class TestEclipseType:

    def test_values(self):
        """EclipseType has NONE, PENUMBRA, UMBRA values."""
        assert EclipseType.NONE.value == "none"
        assert EclipseType.PENUMBRA.value == "penumbra"
        assert EclipseType.UMBRA.value == "umbra"


# ── EclipseEvent dataclass ────────────────────────────────────────

class TestEclipseEvent:

    def test_frozen(self):
        """EclipseEvent is immutable."""
        e = EclipseEvent(
            entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            exit_time=datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc),
            eclipse_type=EclipseType.UMBRA,
            duration_seconds=1800.0,
        )
        with pytest.raises(AttributeError):
            e.duration_seconds = 0.0

    def test_fields(self):
        """EclipseEvent exposes expected fields."""
        entry = datetime(2026, 1, 1, tzinfo=timezone.utc)
        exit_ = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
        e = EclipseEvent(
            entry_time=entry, exit_time=exit_,
            eclipse_type=EclipseType.UMBRA, duration_seconds=1800.0,
        )
        assert e.entry_time == entry
        assert e.exit_time == exit_
        assert e.eclipse_type == EclipseType.UMBRA
        assert e.duration_seconds == 1800.0


# ── is_eclipsed ───────────────────────────────────────────────────

class TestIsEclipsed:

    def test_satellite_behind_earth_umbra(self):
        """Satellite directly behind Earth (opposite Sun) → UMBRA."""
        # Sun at +x, satellite at -x (behind Earth)
        sun_pos = (AU_METERS, 0.0, 0.0)
        sat_pos = (-(R_EARTH + 400_000), 0.0, 0.0)
        result = is_eclipsed(sat_pos, sun_pos)
        assert result == EclipseType.UMBRA

    def test_satellite_sunlit_side(self):
        """Satellite on same side as Sun → NONE."""
        sun_pos = (AU_METERS, 0.0, 0.0)
        sat_pos = (R_EARTH + 400_000, 0.0, 0.0)
        result = is_eclipsed(sat_pos, sun_pos)
        assert result == EclipseType.NONE

    def test_satellite_behind_earth_off_axis(self):
        """Satellite behind Earth but far off-axis → NONE."""
        sun_pos = (AU_METERS, 0.0, 0.0)
        # Satellite behind Earth but displaced far in y
        sat_pos = (-(R_EARTH + 400_000), R_EARTH * 2, 0.0)
        result = is_eclipsed(sat_pos, sun_pos)
        assert result == EclipseType.NONE


# ── Beta angle ────────────────────────────────────────────────────

class TestBetaAngle:

    def test_equatorial_at_equinox(self):
        """Equatorial orbit at equinox → beta ≈ 0°."""
        equinox = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        sun = sun_position_eci(equinox)
        beta = compute_beta_angle(
            raan_rad=sun.right_ascension_rad,
            inclination_rad=0.0,
            epoch=equinox,
        )
        assert abs(beta) < 3.0  # degrees

    def test_polar_orbit_beta_at_solstice(self):
        """Polar orbit (90° inc), RAAN ⊥ Sun → |beta| ≈ 90° - |dec|."""
        solstice = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        sun = sun_position_eci(solstice)
        # RAAN perpendicular to Sun direction gives maximum beta
        beta = compute_beta_angle(
            raan_rad=sun.right_ascension_rad + math.pi / 2,
            inclination_rad=math.radians(90.0),
            epoch=solstice,
        )
        solar_dec_deg = abs(math.degrees(sun.declination_rad))
        # For polar orbit with RAAN-RA=90°: beta = arcsin(cos(dec)) ≈ 90-dec
        expected = 90.0 - solar_dec_deg
        assert abs(abs(beta) - expected) < 2.0


# ── Eclipse windows ──────────────────────────────────────────────

class TestEclipseWindows:

    def _make_leo_state(self):
        """Create LEO orbital state for testing."""
        from constellation_generator.domain.constellation import (
            ShellConfig,
            generate_walker_shell,
        )
        from constellation_generator.domain.propagation import derive_orbital_state

        shell = ShellConfig(
            altitude_km=500, inclination_deg=53, num_planes=1,
            sats_per_plane=1, phase_factor=0, raan_offset_deg=0,
            shell_name='Test',
        )
        sat = generate_walker_shell(shell)[0]
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        return derive_orbital_state(sat, epoch), epoch

    def test_leo_returns_eclipse_events(self):
        """LEO orbit returns non-empty eclipse window list over 2 orbits."""
        state, epoch = self._make_leo_state()
        windows = compute_eclipse_windows(
            state, epoch,
            duration=timedelta(hours=3),
            step=timedelta(seconds=30),
        )
        assert len(windows) > 0

    def test_eclipse_durations_positive_and_bounded(self):
        """Eclipse window durations are positive and < orbital period."""
        state, epoch = self._make_leo_state()
        windows = compute_eclipse_windows(
            state, epoch,
            duration=timedelta(hours=3),
            step=timedelta(seconds=30),
        )
        T_orbital = 2 * math.pi / state.mean_motion_rad_s
        for w in windows:
            assert w.duration_seconds > 0
            assert w.duration_seconds < T_orbital


# ── Eclipse fraction ─────────────────────────────────────────────

class TestEclipseFraction:

    def _make_leo_state(self):
        """Create LEO orbital state for testing."""
        from constellation_generator.domain.constellation import (
            ShellConfig,
            generate_walker_shell,
        )
        from constellation_generator.domain.propagation import derive_orbital_state

        shell = ShellConfig(
            altitude_km=500, inclination_deg=53, num_planes=1,
            sats_per_plane=1, phase_factor=0, raan_offset_deg=0,
            shell_name='Test',
        )
        sat = generate_walker_shell(shell)[0]
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        return derive_orbital_state(sat, epoch), epoch

    def test_leo_eclipse_fraction_range(self):
        """LEO eclipse fraction ≈ 30-40% typical."""
        state, epoch = self._make_leo_state()
        frac = eclipse_fraction(state, epoch)
        assert 0.2 < frac < 0.5

    def test_fraction_between_zero_and_one(self):
        """Eclipse fraction is in [0, 1]."""
        state, epoch = self._make_leo_state()
        frac = eclipse_fraction(state, epoch)
        assert 0.0 <= frac <= 1.0


# ── Domain purity ─────────────────────────────────────────────────

class TestEclipsePurity:

    def test_eclipse_module_pure(self):
        """eclipse.py must only import stdlib modules."""
        import constellation_generator.domain.eclipse as mod

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
