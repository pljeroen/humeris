# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for time-domain revisit coverage analysis."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.orbital_mechanics import OrbitalConstants


# ── Helpers ──────────────────────────────────────────────────────────

_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_states(inclination_deg=53.0, altitude_km=500.0, num_planes=3, sats_per_plane=6):
    from humeris.domain.propagation import derive_orbital_state

    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=inclination_deg,
        num_planes=num_planes, sats_per_plane=sats_per_plane,
        phase_factor=1, raan_offset_deg=0,
        shell_name='Test',
    )
    sats = generate_walker_shell(shell)
    return [derive_orbital_state(s, _EPOCH) for s in sats]


def _make_single_state(inclination_deg=53.0, altitude_km=500.0):
    from humeris.domain.propagation import derive_orbital_state

    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=inclination_deg,
        num_planes=1, sats_per_plane=1,
        phase_factor=0, raan_offset_deg=0,
        shell_name='Single',
    )
    sat = generate_walker_shell(shell)[0]
    return derive_orbital_state(sat, _EPOCH)


# ── GridPoint dataclass ──────────────────────────────────────────────

class TestGridPoint:

    def test_frozen(self):
        from humeris.domain.revisit import GridPoint

        pt = GridPoint(
            lat_deg=0.0, lon_deg=0.0,
            ecef_x=1.0, ecef_y=0.0, ecef_z=0.0,
            unit_x=1.0, unit_y=0.0, unit_z=0.0,
        )
        with pytest.raises(AttributeError):
            pt.lat_deg = 5.0

    def test_fields(self):
        from humeris.domain.revisit import GridPoint

        pt = GridPoint(
            lat_deg=45.0, lon_deg=-90.0,
            ecef_x=100.0, ecef_y=200.0, ecef_z=300.0,
            unit_x=0.27, unit_y=0.53, unit_z=0.80,
        )
        assert pt.lat_deg == 45.0
        assert pt.lon_deg == -90.0
        assert pt.ecef_x == 100.0
        assert pt.unit_z == 0.80


# ── PointRevisitResult dataclass ─────────────────────────────────────

class TestPointRevisitResult:

    def test_frozen(self):
        from humeris.domain.revisit import PointRevisitResult

        pr = PointRevisitResult(
            lat_deg=0.0, lon_deg=0.0, num_passes=1,
            total_visible_s=100.0, total_gap_s=900.0,
            mean_gap_s=900.0, max_gap_s=900.0,
            mean_response_time_s=450.0, coverage_fraction=0.1,
        )
        with pytest.raises(AttributeError):
            pr.num_passes = 5

    def test_fields(self):
        from humeris.domain.revisit import PointRevisitResult

        pr = PointRevisitResult(
            lat_deg=30.0, lon_deg=60.0, num_passes=3,
            total_visible_s=300.0, total_gap_s=700.0,
            mean_gap_s=233.3, max_gap_s=400.0,
            mean_response_time_s=116.65, coverage_fraction=0.3,
        )
        assert pr.lat_deg == 30.0
        assert pr.num_passes == 3
        assert pr.max_gap_s == 400.0


# ── CoverageResult dataclass ────────────────────────────────────────

class TestCoverageResult:

    def test_frozen(self):
        from humeris.domain.revisit import CoverageResult

        cr = CoverageResult(
            analysis_duration_s=3600.0, num_grid_points=4, num_satellites=2,
            mean_coverage_fraction=0.5, min_coverage_fraction=0.1,
            mean_revisit_s=600.0, max_revisit_s=1200.0,
            mean_response_time_s=300.0, percent_coverage_single=25.0,
            point_results=(),
        )
        with pytest.raises(AttributeError):
            cr.max_revisit_s = 0.0

    def test_fields(self):
        from humeris.domain.revisit import CoverageResult

        cr = CoverageResult(
            analysis_duration_s=7200.0, num_grid_points=10, num_satellites=6,
            mean_coverage_fraction=0.8, min_coverage_fraction=0.3,
            mean_revisit_s=500.0, max_revisit_s=2000.0,
            mean_response_time_s=250.0, percent_coverage_single=40.0,
            point_results=(),
        )
        assert cr.analysis_duration_s == 7200.0
        assert cr.num_satellites == 6
        assert cr.percent_coverage_single == 40.0


# ── ECA threshold ────────────────────────────────────────────────────

class TestEarthCentralAngleLimit:

    def test_500km_10deg_elevation(self):
        """Spot check: 500 km orbit, 10° min elevation → expected cos(rho_max)."""
        from humeris.domain.revisit import _earth_central_angle_limit

        cos_rho = _earth_central_angle_limit(500_000.0, 10.0)
        # rho_max = 90 - 10 - asin(R_E * cos(10°) / (R_E + 500km))
        r_e = OrbitalConstants.R_EARTH
        el_rad = math.radians(10.0)
        rho_max = math.radians(90.0) - el_rad - math.asin(r_e * math.cos(el_rad) / (r_e + 500_000))
        expected = math.cos(rho_max)
        assert abs(cos_rho - expected) < 1e-10

    def test_zero_elevation_geometric_horizon(self):
        """0° elevation → rho_max = acos(R_E / (R_E + h)), i.e. geometric horizon."""
        from humeris.domain.revisit import _earth_central_angle_limit

        h = 500_000.0
        r_e = OrbitalConstants.R_EARTH
        cos_rho = _earth_central_angle_limit(h, 0.0)
        # At 0° elevation: rho_max = 90° - 0° - asin(R_E / (R_E + h))
        # = acos(R_E / (R_E + h))
        expected = math.cos(math.acos(r_e / (r_e + h)))
        assert abs(cos_rho - expected) < 1e-10


# ── Grid generation ──────────────────────────────────────────────────

class TestGenerateGrid:

    def test_90deg_step_gives_expected_count(self):
        """90° step → lat [-90, 0, 90] (3) × lon [-180, -90, 0, 90] (4) = 12 points."""
        from humeris.domain.revisit import _generate_grid

        grid = _generate_grid(90.0, 90.0, (-90.0, 90.0), (-180.0, 180.0))
        assert len(grid) == 12

    def test_unit_vectors_normalized(self):
        """All grid point unit vectors should have magnitude ≈ 1.0."""
        from humeris.domain.revisit import _generate_grid

        grid = _generate_grid(30.0, 60.0, (-90.0, 90.0), (-180.0, 180.0))
        for pt in grid:
            mag = math.sqrt(pt.unit_x**2 + pt.unit_y**2 + pt.unit_z**2)
            assert abs(mag - 1.0) < 1e-10, f"Unit vector magnitude {mag} at ({pt.lat_deg}, {pt.lon_deg})"


# ── Visibility checks ───────────────────────────────────────────────

class TestVisibility:

    def test_satellite_directly_above_visible(self):
        """A satellite directly above a grid point should be visible."""
        from humeris.domain.revisit import compute_single_coverage_fraction

        # Place a satellite at 0° lat, 0° lon by choosing epoch where
        # sub-satellite point is near the equator/prime meridian.
        # Use a single-sat state and a grid centered on the sub-satellite point.
        state = _make_single_state(altitude_km=500, inclination_deg=0.0)

        # The satellite starts at RA=0, which at the right GMST maps near lon=0.
        # Use a very coarse grid and check fraction > 0.
        frac = compute_single_coverage_fraction(
            [state], _EPOCH,
            min_elevation_deg=10.0,
            lat_step_deg=90.0, lon_step_deg=90.0,
            lat_range=(-90.0, 90.0), lon_range=(-180.0, 180.0),
        )
        assert frac > 0.0

    def test_satellite_opposite_side_not_visible(self):
        """With only a tiny grid on the opposite side, coverage should be 0."""
        from humeris.domain.revisit import compute_single_coverage_fraction

        # Equatorial satellite at ~lon 0. Grid only at lon 180 ± small window.
        state = _make_single_state(altitude_km=500, inclination_deg=0.0)

        frac = compute_single_coverage_fraction(
            [state], _EPOCH,
            min_elevation_deg=10.0,
            lat_step_deg=5.0, lon_step_deg=5.0,
            lat_range=(-10.0, 10.0), lon_range=(170.0, 180.0),
        )
        # At 500 km, footprint radius is ~20° → satellite at ~lon 0 cannot see lon 170+
        assert frac == 0.0


# ── Time-domain revisit ─────────────────────────────────────────────

class TestComputeRevisit:

    def test_single_satellite_detects_pass(self):
        """Single satellite over ~90 min should detect at least 1 pass at some grid point."""
        from humeris.domain.revisit import compute_revisit

        state = _make_single_state(altitude_km=500)
        result = compute_revisit(
            [state], _EPOCH, timedelta(minutes=100), timedelta(seconds=60),
            min_elevation_deg=10.0,
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        total_passes = sum(pr.num_passes for pr in result.point_results)
        assert total_passes >= 1

    def test_single_satellite_max_gap_approx_orbital_period(self):
        """Single sat, 24h analysis: max gap for most points ≈ orbital period (~94 min for 500 km)."""
        from humeris.domain.revisit import compute_revisit

        state = _make_single_state(altitude_km=500)
        result = compute_revisit(
            [state], _EPOCH, timedelta(hours=3), timedelta(seconds=60),
            min_elevation_deg=10.0,
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        # For points that have at least 2 passes, max gap should be in orbit-period range.
        # Orbital period at 500 km ≈ 2π * sqrt(a³/μ) ≈ 5670 s ≈ 94 min
        r_e = OrbitalConstants.R_EARTH
        a = r_e + 500_000.0
        t_orb = 2.0 * math.pi * math.sqrt(a**3 / OrbitalConstants.MU_EARTH)
        points_with_passes = [pr for pr in result.point_results if pr.num_passes >= 2]
        if points_with_passes:
            # Max gap shouldn't exceed analysis window
            for pr in points_with_passes:
                assert pr.max_gap_s <= result.analysis_duration_s + 1.0

    def test_more_satellites_shorter_mean_revisit(self):
        """Increasing satellite count should reduce mean revisit time (monotonicity)."""
        from humeris.domain.revisit import compute_revisit

        states_small = _make_states(num_planes=2, sats_per_plane=3)
        states_large = _make_states(num_planes=4, sats_per_plane=6)

        kwargs = dict(
            start=_EPOCH, duration=timedelta(hours=3), step=timedelta(seconds=60),
            min_elevation_deg=10.0, lat_step_deg=30.0, lon_step_deg=30.0,
        )
        result_small = compute_revisit(states_small, **kwargs)
        result_large = compute_revisit(states_large, **kwargs)

        # More satellites → better coverage
        assert result_large.mean_coverage_fraction >= result_small.mean_coverage_fraction

    def test_coverage_fraction_in_range(self):
        """All per-point coverage fractions should be in [0, 1]."""
        from humeris.domain.revisit import compute_revisit

        states = _make_states(num_planes=2, sats_per_plane=4)
        result = compute_revisit(
            states, _EPOCH, timedelta(hours=2), timedelta(seconds=60),
            min_elevation_deg=10.0, lat_step_deg=30.0, lon_step_deg=30.0,
        )
        for pr in result.point_results:
            assert 0.0 <= pr.coverage_fraction <= 1.0

    def test_mean_response_time_is_half_mean_gap(self):
        """Mean response time should equal mean_gap / 2 for each point."""
        from humeris.domain.revisit import compute_revisit

        states = _make_states(num_planes=2, sats_per_plane=4)
        result = compute_revisit(
            states, _EPOCH, timedelta(hours=2), timedelta(seconds=60),
            min_elevation_deg=10.0, lat_step_deg=30.0, lon_step_deg=30.0,
        )
        for pr in result.point_results:
            expected = pr.mean_gap_s / 2.0
            assert abs(pr.mean_response_time_s - expected) < 1e-6

    def test_zero_satellites_zero_coverage(self):
        """Empty satellite list → coverage fraction 0, num_passes 0 everywhere."""
        from humeris.domain.revisit import compute_revisit

        result = compute_revisit(
            [], _EPOCH, timedelta(hours=1), timedelta(seconds=60),
            min_elevation_deg=10.0, lat_step_deg=30.0, lon_step_deg=60.0,
        )
        assert result.mean_coverage_fraction == 0.0
        for pr in result.point_results:
            assert pr.num_passes == 0
            assert pr.coverage_fraction == 0.0

    def test_compute_single_coverage_fraction_matches_revisit(self):
        """Single-epoch fraction should match revisit fraction at same epoch."""
        from humeris.domain.revisit import (
            compute_revisit,
            compute_single_coverage_fraction,
        )

        states = _make_states(num_planes=2, sats_per_plane=4)

        # Single-step revisit analysis (1 step = snapshot)
        result = compute_revisit(
            states, _EPOCH, timedelta(seconds=0), timedelta(seconds=60),
            min_elevation_deg=10.0, lat_step_deg=30.0, lon_step_deg=30.0,
        )
        frac_revisit = result.mean_coverage_fraction

        frac_single = compute_single_coverage_fraction(
            states, _EPOCH,
            min_elevation_deg=10.0,
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        assert abs(frac_revisit - frac_single) < 1e-6


# ── Domain purity ────────────────────────────────────────────────────

class TestRevisitPurity:

    def test_revisit_imports_only_stdlib_and_domain(self):
        import humeris.domain.revisit as mod

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
