"""Tests for grid-based coverage analysis."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator.domain.constellation import ShellConfig, generate_walker_shell
from constellation_generator.domain.orbital_mechanics import OrbitalConstants


# ── Helpers ──────────────────────────────────────────────────────────

_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_states(inclination_deg=53.0, altitude_km=500.0, num_planes=3, sats_per_plane=6):
    from constellation_generator.domain.propagation import derive_orbital_state

    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=inclination_deg,
        num_planes=num_planes, sats_per_plane=sats_per_plane,
        phase_factor=1, raan_offset_deg=0,
        shell_name='Test',
    )
    sats = generate_walker_shell(shell)
    return [derive_orbital_state(s, _EPOCH) for s in sats]


# ── Dataclass ────────────────────────────────────────────────────────

class TestCoveragePoint:

    def test_frozen(self):
        from constellation_generator.domain.coverage import CoveragePoint

        pt = CoveragePoint(lat_deg=0.0, lon_deg=0.0, visible_count=3)
        with pytest.raises(AttributeError):
            pt.visible_count = 5

    def test_fields(self):
        from constellation_generator.domain.coverage import CoveragePoint

        pt = CoveragePoint(lat_deg=45.0, lon_deg=-90.0, visible_count=2)
        assert pt.lat_deg == 45.0
        assert pt.lon_deg == -90.0
        assert pt.visible_count == 2


# ── Coverage computation ─────────────────────────────────────────────

class TestComputeCoverageSnapshot:

    def test_grid_point_count(self):
        """Grid size matches expected lat/lon dimensions."""
        from constellation_generator.domain.coverage import compute_coverage_snapshot

        states = _make_states()
        grid = compute_coverage_snapshot(states, _EPOCH, lat_step_deg=30.0, lon_step_deg=60.0)
        # lat: -90, -60, -30, 0, 30, 60, 90 → 7
        # lon: -180, -120, -60, 0, 60, 120 → 6
        assert len(grid) == 7 * 6

    def test_single_satellite_partial_coverage(self):
        """Single satellite: some points visible, some not."""
        from constellation_generator.domain.coverage import compute_coverage_snapshot
        from constellation_generator.domain.propagation import derive_orbital_state

        shell = ShellConfig(
            altitude_km=500, inclination_deg=53,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0,
            shell_name='Single',
        )
        sat = generate_walker_shell(shell)[0]
        state = derive_orbital_state(sat, _EPOCH)

        grid = compute_coverage_snapshot([state], _EPOCH, lat_step_deg=30.0, lon_step_deg=60.0)
        visible = [p for p in grid if p.visible_count > 0]
        not_visible = [p for p in grid if p.visible_count == 0]
        assert len(visible) > 0
        assert len(not_visible) > 0

    def test_empty_satellite_list_all_zeros(self):
        """No satellites → all visible_count = 0."""
        from constellation_generator.domain.coverage import compute_coverage_snapshot

        grid = compute_coverage_snapshot([], _EPOCH, lat_step_deg=30.0, lon_step_deg=60.0)
        for pt in grid:
            assert pt.visible_count == 0

    def test_higher_altitude_more_coverage(self):
        """Higher altitude satellites have wider footprint → more visible points."""
        from constellation_generator.domain.coverage import compute_coverage_snapshot
        from constellation_generator.domain.propagation import derive_orbital_state

        shell_low = ShellConfig(
            altitude_km=300, inclination_deg=53,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0,
            shell_name='Low',
        )
        shell_high = ShellConfig(
            altitude_km=2000, inclination_deg=53,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0,
            shell_name='High',
        )
        sat_low = generate_walker_shell(shell_low)[0]
        sat_high = generate_walker_shell(shell_high)[0]
        state_low = derive_orbital_state(sat_low, _EPOCH)
        state_high = derive_orbital_state(sat_high, _EPOCH)

        grid_low = compute_coverage_snapshot([state_low], _EPOCH, lat_step_deg=10.0, lon_step_deg=10.0)
        grid_high = compute_coverage_snapshot([state_high], _EPOCH, lat_step_deg=10.0, lon_step_deg=10.0)

        vis_low = sum(1 for p in grid_low if p.visible_count > 0)
        vis_high = sum(1 for p in grid_high if p.visible_count > 0)
        assert vis_high > vis_low

    def test_custom_lat_lon_range(self):
        """Custom range reduces grid size."""
        from constellation_generator.domain.coverage import compute_coverage_snapshot

        states = _make_states()
        grid_full = compute_coverage_snapshot(states, _EPOCH, lat_step_deg=30.0, lon_step_deg=60.0)
        grid_partial = compute_coverage_snapshot(
            states, _EPOCH,
            lat_step_deg=30.0, lon_step_deg=60.0,
            lat_range=(0, 90), lon_range=(0, 180),
        )
        assert len(grid_partial) < len(grid_full)

    def test_higher_min_elevation_fewer_visible(self):
        """Higher min_elevation → fewer visible points."""
        from constellation_generator.domain.coverage import compute_coverage_snapshot

        states = _make_states()
        grid_low = compute_coverage_snapshot(
            states, _EPOCH, lat_step_deg=10.0, lon_step_deg=10.0, min_elevation_deg=5.0,
        )
        grid_high = compute_coverage_snapshot(
            states, _EPOCH, lat_step_deg=10.0, lon_step_deg=10.0, min_elevation_deg=45.0,
        )
        vis_low = sum(p.visible_count for p in grid_low)
        vis_high = sum(p.visible_count for p in grid_high)
        assert vis_high <= vis_low

    def test_visible_count_nonnegative(self):
        """All visible_count values must be >= 0."""
        from constellation_generator.domain.coverage import compute_coverage_snapshot

        states = _make_states()
        grid = compute_coverage_snapshot(states, _EPOCH, lat_step_deg=30.0, lon_step_deg=60.0)
        for pt in grid:
            assert pt.visible_count >= 0


# ── Domain purity ────────────────────────────────────────────────────

class TestCoveragePurity:

    def test_coverage_imports_only_stdlib_and_domain(self):
        import constellation_generator.domain.coverage as mod

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
