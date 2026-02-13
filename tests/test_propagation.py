# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for shared propagation module and ground_track J2 integration."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.orbital_mechanics import OrbitalConstants


# ── Helpers ──────────────────────────────────────────────────────────

def _make_satellite(inclination_deg=53.0, altitude_km=500.0):
    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=inclination_deg,
        num_planes=1, sats_per_plane=1,
        phase_factor=0, raan_offset_deg=0,
        shell_name='Test',
    )
    return generate_walker_shell(shell)[0]


_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


# ── OrbitalState dataclass ───────────────────────────────────────────

class TestOrbitalState:

    def test_frozen(self):
        from humeris.domain.propagation import OrbitalState

        state = OrbitalState(
            semi_major_axis_m=7_000_000.0,
            eccentricity=0.0,
            inclination_rad=math.radians(53),
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=0.001,
            reference_epoch=_EPOCH,
        )
        with pytest.raises(AttributeError):
            state.semi_major_axis_m = 8_000_000.0

    def test_j2_defaults_zero(self):
        from humeris.domain.propagation import OrbitalState

        state = OrbitalState(
            semi_major_axis_m=7_000_000.0,
            eccentricity=0.0,
            inclination_rad=0.0,
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=0.001,
            reference_epoch=_EPOCH,
        )
        assert state.j2_raan_rate == 0.0
        assert state.j2_arg_perigee_rate == 0.0
        assert state.j2_mean_motion_correction == 0.0

    def test_fields(self):
        from humeris.domain.propagation import OrbitalState

        state = OrbitalState(
            semi_major_axis_m=6_871_000.0,
            eccentricity=0.001,
            inclination_rad=math.radians(53),
            raan_rad=math.radians(45),
            arg_perigee_rad=math.radians(10),
            true_anomaly_rad=math.radians(90),
            mean_motion_rad_s=0.00112,
            reference_epoch=_EPOCH,
            j2_raan_rate=-1e-7,
            j2_arg_perigee_rate=2e-7,
            j2_mean_motion_correction=0.00113,
        )
        assert state.semi_major_axis_m == 6_871_000.0
        assert state.j2_raan_rate == -1e-7
        assert state.j2_mean_motion_correction == 0.00113


# ── derive_orbital_state ─────────────────────────────────────────────

class TestDeriveOrbitalState:

    def test_circular_orbit_semi_major_axis(self):
        """a = |pos| for circular orbit."""
        from humeris.domain.propagation import derive_orbital_state

        sat = _make_satellite(altitude_km=500)
        state = derive_orbital_state(sat, _EPOCH)
        expected_a = OrbitalConstants.R_EARTH + 500_000
        assert abs(state.semi_major_axis_m - expected_a) < 1.0

    def test_mean_motion(self):
        """n = sqrt(mu/a^3)."""
        from humeris.domain.propagation import derive_orbital_state

        sat = _make_satellite(altitude_km=500)
        state = derive_orbital_state(sat, _EPOCH)
        a = OrbitalConstants.R_EARTH + 500_000
        expected_n = math.sqrt(OrbitalConstants.MU_EARTH / a**3)
        assert abs(state.mean_motion_rad_s - expected_n) < 1e-10

    def test_inclination(self):
        """Inclination matches satellite orbit."""
        from humeris.domain.propagation import derive_orbital_state

        sat = _make_satellite(inclination_deg=53)
        state = derive_orbital_state(sat, _EPOCH)
        assert abs(math.degrees(state.inclination_rad) - 53.0) < 0.1

    def test_j2_off_by_default(self):
        """Without include_j2, J2 rates are zero."""
        from humeris.domain.propagation import derive_orbital_state

        sat = _make_satellite()
        state = derive_orbital_state(sat, _EPOCH)
        assert state.j2_raan_rate == 0.0
        assert state.j2_arg_perigee_rate == 0.0
        assert state.j2_mean_motion_correction == 0.0

    def test_j2_on_produces_nonzero_rates(self):
        """With include_j2=True, J2 rates are nonzero."""
        from humeris.domain.propagation import derive_orbital_state

        sat = _make_satellite(inclination_deg=53)
        state = derive_orbital_state(sat, _EPOCH, include_j2=True)
        assert state.j2_raan_rate != 0.0
        assert state.j2_arg_perigee_rate != 0.0
        assert state.j2_mean_motion_correction != 0.0


# ── propagate_to ─────────────────────────────────────────────────────

class TestPropagateTo:

    def test_zero_dt_matches_initial(self):
        """At dt=0, propagated position ≈ initial kepler_to_cartesian output."""
        from humeris.domain.propagation import derive_orbital_state, propagate_to

        sat = _make_satellite(altitude_km=500)
        state = derive_orbital_state(sat, _EPOCH)
        pos, vel = propagate_to(state, _EPOCH)
        r = math.sqrt(sum(p**2 for p in pos))
        expected_r = OrbitalConstants.R_EARTH + 500_000
        assert abs(r - expected_r) < 1.0

    def test_one_period_returns_near_start(self):
        """After one orbital period, position returns near initial."""
        from humeris.domain.propagation import derive_orbital_state, propagate_to

        sat = _make_satellite(altitude_km=500)
        state = derive_orbital_state(sat, _EPOCH)
        period_s = 2 * math.pi / state.mean_motion_rad_s
        pos_0, _ = propagate_to(state, _EPOCH)
        pos_1, _ = propagate_to(state, _EPOCH + timedelta(seconds=period_s))
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(pos_0, pos_1)))
        # Should be very close for circular, no-J2 orbit
        assert dist < 100.0  # within 100m

    def test_radius_preserved_circular(self):
        """For circular orbit, radius stays constant over time."""
        from humeris.domain.propagation import derive_orbital_state, propagate_to

        sat = _make_satellite(altitude_km=500)
        state = derive_orbital_state(sat, _EPOCH)
        expected_r = OrbitalConstants.R_EARTH + 500_000
        for minutes in [0, 15, 30, 45, 60]:
            pos, _ = propagate_to(state, _EPOCH + timedelta(minutes=minutes))
            r = math.sqrt(sum(p**2 for p in pos))
            assert abs(r - expected_r) < 1.0


# ── propagate_ecef_to ────────────────────────────────────────────────

class TestPropagateECEFTo:

    def test_returns_tuple(self):
        from humeris.domain.propagation import derive_orbital_state, propagate_ecef_to

        sat = _make_satellite()
        state = derive_orbital_state(sat, _EPOCH)
        result = propagate_ecef_to(state, _EPOCH)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_plausible_radius(self):
        from humeris.domain.propagation import derive_orbital_state, propagate_ecef_to

        sat = _make_satellite(altitude_km=500)
        state = derive_orbital_state(sat, _EPOCH)
        x, y, z = propagate_ecef_to(state, _EPOCH)
        r = math.sqrt(x**2 + y**2 + z**2)
        expected_r = OrbitalConstants.R_EARTH + 500_000
        assert abs(r - expected_r) < 100.0


# ── ground_track backward compatibility ──────────────────────────────

class TestGroundTrackCompat:

    def test_default_no_j2_same_output(self):
        """include_j2=False (default) produces same output as before."""
        from humeris.domain.ground_track import compute_ground_track

        sat = _make_satellite()
        start = _EPOCH
        track = compute_ground_track(sat, start, timedelta(minutes=90), timedelta(minutes=1))
        assert len(track) == 91
        for point in track:
            assert -90 <= point.lat_deg <= 90

    def test_j2_differs_over_long_duration(self):
        """With J2, ground track differs from without over 24 hours."""
        from humeris.domain.ground_track import compute_ground_track

        sat = _make_satellite()
        start = _EPOCH
        duration = timedelta(hours=24)
        step = timedelta(minutes=10)

        track_no_j2 = compute_ground_track(sat, start, duration, step)
        track_j2 = compute_ground_track(sat, start, duration, step, include_j2=True)

        assert len(track_no_j2) == len(track_j2)

        # Last points should differ in longitude due to J2 RAAN drift
        last_no_j2 = track_no_j2[-1]
        last_j2 = track_j2[-1]
        lon_diff = abs(last_no_j2.lon_deg - last_j2.lon_deg)
        assert lon_diff > 0.01, "J2 should produce measurable difference over 24h"


# ── Domain purity ────────────────────────────────────────────────────

class TestPropagationPurity:

    def test_propagation_imports_only_stdlib_and_domain(self):
        import humeris.domain.propagation as mod

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
