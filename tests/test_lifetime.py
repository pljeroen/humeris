# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for orbit lifetime and decay profile computation."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.atmosphere import DragConfig


# ── Helpers ──────────────────────────────────────────────────────────

EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
STARLINK_DRAG = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)


def _sma(altitude_km: float) -> float:
    return OrbitalConstants.R_EARTH + altitude_km * 1000


# ── DecayPoint / OrbitLifetimeResult ─────────────────────────────────

class TestDecayPoint:

    def test_frozen(self):
        """DecayPoint is immutable."""
        from humeris.domain.lifetime import DecayPoint

        dp = DecayPoint(time=EPOCH, altitude_km=550.0, semi_major_axis_m=_sma(550))
        with pytest.raises(AttributeError):
            dp.altitude_km = 400.0

    def test_fields(self):
        """DecayPoint exposes time, altitude_km, semi_major_axis_m."""
        from humeris.domain.lifetime import DecayPoint

        dp = DecayPoint(time=EPOCH, altitude_km=550.0, semi_major_axis_m=_sma(550))
        assert dp.time == EPOCH
        assert dp.altitude_km == 550.0
        assert dp.semi_major_axis_m == _sma(550)


class TestOrbitLifetimeResult:

    def test_frozen(self):
        """OrbitLifetimeResult is immutable."""
        from humeris.domain.lifetime import OrbitLifetimeResult

        result = OrbitLifetimeResult(
            initial_altitude_km=550.0,
            re_entry_altitude_km=100.0,
            lifetime_days=1000.0,
            re_entry_time=None,
            decay_profile=(),
            converged=True,
        )
        with pytest.raises(AttributeError):
            result.lifetime_days = 500.0

    def test_profile_is_tuple(self):
        """decay_profile is a tuple."""
        from humeris.domain.lifetime import OrbitLifetimeResult

        result = OrbitLifetimeResult(
            initial_altitude_km=550.0,
            re_entry_altitude_km=100.0,
            lifetime_days=0.0,
            re_entry_time=None,
            decay_profile=(),
            converged=False,
        )
        assert isinstance(result.decay_profile, tuple)


# ── compute_orbit_lifetime ───────────────────────────────────────────

class TestComputeOrbitLifetime:

    def test_300km_converges_quickly(self):
        """300 km orbit with Starlink-like B_c converges in < 365 days."""
        from humeris.domain.lifetime import compute_orbit_lifetime

        result = compute_orbit_lifetime(_sma(300), 0.0, STARLINK_DRAG, EPOCH)
        assert result.converged
        assert result.lifetime_days < 365

    def test_800km_lifetime_over_4_years(self):
        """800 km orbit lifetime > 4 years (forward Euler model)."""
        from humeris.domain.lifetime import compute_orbit_lifetime

        result = compute_orbit_lifetime(_sma(800), 0.0, STARLINK_DRAG, EPOCH)
        assert result.lifetime_days > 4 * 365.25

    def test_higher_bc_shorter_life(self):
        """Higher ballistic coefficient → shorter lifetime."""
        from humeris.domain.lifetime import compute_orbit_lifetime

        low_bc = DragConfig(cd=2.2, area_m2=5.0, mass_kg=260.0)
        high_bc = DragConfig(cd=2.2, area_m2=20.0, mass_kg=260.0)
        life_low = compute_orbit_lifetime(_sma(400), 0.0, low_bc, EPOCH)
        life_high = compute_orbit_lifetime(_sma(400), 0.0, high_bc, EPOCH)
        assert life_high.lifetime_days < life_low.lifetime_days

    def test_profile_altitudes_monotonically_decreasing(self):
        """Decay profile altitudes decrease monotonically."""
        from humeris.domain.lifetime import compute_orbit_lifetime

        result = compute_orbit_lifetime(_sma(400), 0.0, STARLINK_DRAG, EPOCH, step_days=1.0)
        altitudes = [p.altitude_km for p in result.decay_profile]
        for i in range(len(altitudes) - 1):
            assert altitudes[i] >= altitudes[i + 1], (
                f"Altitude increased at step {i}: {altitudes[i]} -> {altitudes[i+1]}"
            )

    def test_first_point_matches_input(self):
        """First decay profile point matches initial altitude."""
        from humeris.domain.lifetime import compute_orbit_lifetime

        result = compute_orbit_lifetime(_sma(500), 0.0, STARLINK_DRAG, EPOCH)
        assert len(result.decay_profile) > 0
        first = result.decay_profile[0]
        assert abs(first.altitude_km - 500.0) < 0.1

    def test_last_point_near_reentry(self):
        """Last decay profile point is near re-entry altitude."""
        from humeris.domain.lifetime import compute_orbit_lifetime

        result = compute_orbit_lifetime(_sma(300), 0.0, STARLINK_DRAG, EPOCH)
        assert result.converged
        last = result.decay_profile[-1]
        assert last.altitude_km <= 100.0 + 5.0  # within 5 km tolerance

    def test_reentry_time_set_when_converged(self):
        """re_entry_time is set when orbit converges."""
        from humeris.domain.lifetime import compute_orbit_lifetime

        result = compute_orbit_lifetime(_sma(300), 0.0, STARLINK_DRAG, EPOCH)
        assert result.converged
        assert result.re_entry_time is not None
        assert result.re_entry_time > EPOCH

    def test_below_reentry_raises(self):
        """Starting below re-entry altitude raises ValueError."""
        from humeris.domain.lifetime import compute_orbit_lifetime

        with pytest.raises(ValueError):
            compute_orbit_lifetime(_sma(90), 0.0, STARLINK_DRAG, EPOCH)

    def test_negative_step_raises(self):
        """Negative step_days raises ValueError."""
        from humeris.domain.lifetime import compute_orbit_lifetime

        with pytest.raises(ValueError):
            compute_orbit_lifetime(_sma(400), 0.0, STARLINK_DRAG, EPOCH, step_days=-1.0)


# ── compute_altitude_at_time ─────────────────────────────────────────

class TestComputeAltitudeAtTime:

    def test_altitude_decreases_over_time(self):
        """Altitude decreases after some propagation time."""
        from humeris.domain.lifetime import compute_altitude_at_time

        alt_1day = compute_altitude_at_time(
            _sma(400), 0.0, STARLINK_DRAG, EPOCH,
            EPOCH + timedelta(days=1),
        )
        alt_30day = compute_altitude_at_time(
            _sma(400), 0.0, STARLINK_DRAG, EPOCH,
            EPOCH + timedelta(days=30),
        )
        assert alt_30day < alt_1day

    def test_target_before_epoch_raises(self):
        """Target time before epoch raises ValueError."""
        from humeris.domain.lifetime import compute_altitude_at_time

        with pytest.raises(ValueError):
            compute_altitude_at_time(
                _sma(400), 0.0, STARLINK_DRAG, EPOCH,
                EPOCH - timedelta(days=1),
            )


# ── Domain purity ───────────────────────────────────────────────────

class TestLifetimePurity:

    def test_lifetime_module_pure(self):
        """lifetime.py must only import stdlib modules."""
        import humeris.domain.lifetime as mod

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
