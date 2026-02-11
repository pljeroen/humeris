"""Tests for access window computation."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator.domain.constellation import ShellConfig, generate_walker_shell
from constellation_generator.domain.orbital_mechanics import OrbitalConstants


# ── Helpers ──────────────────────────────────────────────────────────

_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_satellite(inclination_deg=53.0, altitude_km=500.0):
    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=inclination_deg,
        num_planes=1, sats_per_plane=1,
        phase_factor=0, raan_offset_deg=0,
        shell_name='Test',
    )
    return generate_walker_shell(shell)[0]


# ── Dataclass ────────────────────────────────────────────────────────

class TestAccessWindow:

    def test_frozen(self):
        from constellation_generator.domain.access_windows import AccessWindow

        window = AccessWindow(
            rise_time=_EPOCH,
            set_time=_EPOCH + timedelta(minutes=10),
            max_elevation_deg=45.0,
            duration_seconds=600.0,
        )
        with pytest.raises(AttributeError):
            window.max_elevation_deg = 50.0

    def test_fields(self):
        from constellation_generator.domain.access_windows import AccessWindow

        rise = _EPOCH
        set_ = _EPOCH + timedelta(minutes=10)
        window = AccessWindow(
            rise_time=rise, set_time=set_,
            max_elevation_deg=45.0, duration_seconds=600.0,
        )
        assert window.rise_time == rise
        assert window.set_time == set_
        assert window.max_elevation_deg == 45.0
        assert window.duration_seconds == 600.0


# ── Access window computation ────────────────────────────────────────

class TestComputeAccessWindows:

    def test_iss_like_orbit_has_passes(self):
        """ISS-like orbit (51.6° inc) over mid-latitude station: at least 1 pass in 24h."""
        from constellation_generator.domain.access_windows import compute_access_windows
        from constellation_generator.domain.observation import GroundStation
        from constellation_generator.domain.propagation import derive_orbital_state

        sat = _make_satellite(inclination_deg=51.6, altitude_km=420)
        station = GroundStation(name='Mid', lat_deg=52.0, lon_deg=4.4, alt_m=0.0)
        state = derive_orbital_state(sat, _EPOCH)
        windows = compute_access_windows(
            station, state, _EPOCH,
            timedelta(hours=24), timedelta(seconds=30),
            min_elevation_deg=5.0,
        )
        assert len(windows) >= 1

    def test_equatorial_orbit_polar_station_no_passes(self):
        """Equatorial orbit over polar station: no passes above 10°."""
        from constellation_generator.domain.access_windows import compute_access_windows
        from constellation_generator.domain.observation import GroundStation
        from constellation_generator.domain.propagation import derive_orbital_state

        sat = _make_satellite(inclination_deg=0.0, altitude_km=500)
        station = GroundStation(name='Pole', lat_deg=89.0, lon_deg=0.0, alt_m=0.0)
        state = derive_orbital_state(sat, _EPOCH)
        windows = compute_access_windows(
            station, state, _EPOCH,
            timedelta(hours=24), timedelta(seconds=30),
            min_elevation_deg=10.0,
        )
        assert len(windows) == 0

    def test_windows_chronological(self):
        """Windows must be in chronological order."""
        from constellation_generator.domain.access_windows import compute_access_windows
        from constellation_generator.domain.observation import GroundStation
        from constellation_generator.domain.propagation import derive_orbital_state

        sat = _make_satellite(inclination_deg=51.6, altitude_km=420)
        station = GroundStation(name='Mid', lat_deg=52.0, lon_deg=4.4, alt_m=0.0)
        state = derive_orbital_state(sat, _EPOCH)
        windows = compute_access_windows(
            station, state, _EPOCH,
            timedelta(hours=24), timedelta(seconds=30),
            min_elevation_deg=5.0,
        )
        for i in range(len(windows) - 1):
            assert windows[i].set_time <= windows[i + 1].rise_time

    def test_max_elevation_above_min(self):
        """Within each window, max_elevation ≥ min_elevation."""
        from constellation_generator.domain.access_windows import compute_access_windows
        from constellation_generator.domain.observation import GroundStation
        from constellation_generator.domain.propagation import derive_orbital_state

        sat = _make_satellite(inclination_deg=51.6, altitude_km=420)
        station = GroundStation(name='Mid', lat_deg=52.0, lon_deg=4.4, alt_m=0.0)
        state = derive_orbital_state(sat, _EPOCH)
        min_el = 10.0
        windows = compute_access_windows(
            station, state, _EPOCH,
            timedelta(hours=24), timedelta(seconds=30),
            min_elevation_deg=min_el,
        )
        for w in windows:
            assert w.max_elevation_deg >= min_el

    def test_higher_min_elevation_fewer_windows(self):
        """Higher min_elevation threshold → fewer or equal windows."""
        from constellation_generator.domain.access_windows import compute_access_windows
        from constellation_generator.domain.observation import GroundStation
        from constellation_generator.domain.propagation import derive_orbital_state

        sat = _make_satellite(inclination_deg=51.6, altitude_km=420)
        station = GroundStation(name='Mid', lat_deg=52.0, lon_deg=4.4, alt_m=0.0)
        state = derive_orbital_state(sat, _EPOCH)

        windows_low = compute_access_windows(
            station, state, _EPOCH,
            timedelta(hours=24), timedelta(seconds=30),
            min_elevation_deg=5.0,
        )
        windows_high = compute_access_windows(
            station, state, _EPOCH,
            timedelta(hours=24), timedelta(seconds=30),
            min_elevation_deg=30.0,
        )
        assert len(windows_high) <= len(windows_low)

    def test_duration_positive(self):
        """Each window has positive duration."""
        from constellation_generator.domain.access_windows import compute_access_windows
        from constellation_generator.domain.observation import GroundStation
        from constellation_generator.domain.propagation import derive_orbital_state

        sat = _make_satellite(inclination_deg=51.6, altitude_km=420)
        station = GroundStation(name='Mid', lat_deg=52.0, lon_deg=4.4, alt_m=0.0)
        state = derive_orbital_state(sat, _EPOCH)
        windows = compute_access_windows(
            station, state, _EPOCH,
            timedelta(hours=24), timedelta(seconds=30),
            min_elevation_deg=5.0,
        )
        for w in windows:
            assert w.duration_seconds > 0

    def test_negative_step_raises(self):
        """Negative step should raise ValueError."""
        from constellation_generator.domain.access_windows import compute_access_windows
        from constellation_generator.domain.observation import GroundStation
        from constellation_generator.domain.propagation import derive_orbital_state

        sat = _make_satellite()
        station = GroundStation(name='Test', lat_deg=0.0, lon_deg=0.0)
        state = derive_orbital_state(sat, _EPOCH)
        with pytest.raises(ValueError):
            compute_access_windows(
                station, state, _EPOCH,
                timedelta(hours=1), timedelta(seconds=-10),
            )

    def test_window_open_at_start(self):
        """If satellite is visible at start, rise_time should equal start."""
        from constellation_generator.domain.access_windows import compute_access_windows
        from constellation_generator.domain.observation import GroundStation, compute_observation
        from constellation_generator.domain.propagation import derive_orbital_state, propagate_ecef_to

        # Use equatorial sat over equatorial station — likely visible at t=0
        sat = _make_satellite(inclination_deg=0.0, altitude_km=500)
        station = GroundStation(name='Eq', lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
        state = derive_orbital_state(sat, _EPOCH)

        # Check if satellite is visible at start
        sat_ecef = propagate_ecef_to(state, _EPOCH)
        obs = compute_observation(station, sat_ecef)

        if obs.elevation_deg >= 10.0:
            windows = compute_access_windows(
                station, state, _EPOCH,
                timedelta(hours=2), timedelta(seconds=10),
                min_elevation_deg=10.0,
            )
            assert len(windows) >= 1
            assert windows[0].rise_time == _EPOCH


# ── Domain purity ────────────────────────────────────────────────────

class TestAccessWindowsPurity:

    def test_access_windows_imports_only_stdlib_and_domain(self):
        import constellation_generator.domain.access_windows as mod

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
