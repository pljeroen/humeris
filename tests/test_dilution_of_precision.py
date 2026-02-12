# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for Geometric Dilution of Precision (GDOP/PDOP/HDOP/VDOP/TDOP).

Covers 4x4 matrix inversion, single-point DOP computation, and
grid-based DOP analysis.
"""

import math
from datetime import datetime, timezone

import pytest

from constellation_generator import (
    OrbitalConstants,
    OrbitalState,
    ShellConfig,
    generate_walker_shell,
    derive_orbital_state,
    propagate_ecef_to,
)

EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
R_E = OrbitalConstants.R_EARTH


def _leo_state(altitude_km=550, inclination_deg=53) -> OrbitalState:
    """Helper: LEO circular orbit state."""
    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=inclination_deg,
        num_planes=1, sats_per_plane=1, phase_factor=0,
        raan_offset_deg=0, shell_name="Test",
    )
    sats = generate_walker_shell(shell)
    return derive_orbital_state(sats[0], EPOCH)


def _constellation_states(num_planes=6, sats_per_plane=6, altitude_km=550):
    """Helper: Walker constellation states."""
    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=53,
        num_planes=num_planes, sats_per_plane=sats_per_plane,
        phase_factor=1, raan_offset_deg=0, shell_name="Walker",
    )
    sats = generate_walker_shell(shell)
    return [derive_orbital_state(sat, EPOCH) for sat in sats]


# --- _invert_4x4 ---

class TestInvert4x4:

    def test_invert_identity(self):
        from constellation_generator.domain.dilution_of_precision import _invert_4x4
        I = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        result = _invert_4x4(I)
        assert result is not None
        for i in range(4):
            for j in range(4):
                expected = 1.0 if i == j else 0.0
                assert abs(result[i][j] - expected) < 1e-10

    def test_invert_known_matrix(self):
        from constellation_generator.domain.dilution_of_precision import _invert_4x4
        # Diagonal matrix: inverse is reciprocals
        m = [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]]
        result = _invert_4x4(m)
        assert result is not None
        assert abs(result[0][0] - 0.5) < 1e-10
        assert abs(result[1][1] - 1.0 / 3.0) < 1e-10
        assert abs(result[2][2] - 0.25) < 1e-10
        assert abs(result[3][3] - 0.2) < 1e-10

    def test_invert_singular_returns_none(self):
        from constellation_generator.domain.dilution_of_precision import _invert_4x4
        # Row of zeros → singular
        m = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        result = _invert_4x4(m)
        assert result is None

    def test_invert_product_is_identity(self):
        from constellation_generator.domain.dilution_of_precision import _invert_4x4
        m = [
            [2, 1, 0, 0],
            [1, 3, 1, 0],
            [0, 1, 4, 1],
            [0, 0, 1, 5],
        ]
        inv = _invert_4x4(m)
        assert inv is not None
        # M * M^-1 should be identity
        for i in range(4):
            for j in range(4):
                val = sum(m[i][k] * inv[k][j] for k in range(4))
                expected = 1.0 if i == j else 0.0
                assert abs(val - expected) < 1e-9, \
                    f"M*M^-1 [{i}][{j}] = {val}, expected {expected}"


# --- compute_dop ---

class TestComputeDop:

    def test_dop_returns_type(self):
        from constellation_generator.domain.dilution_of_precision import (
            compute_dop, DOPResult,
        )
        states = _constellation_states(num_planes=6, sats_per_plane=6)
        sat_ecefs = [propagate_ecef_to(s, EPOCH) for s in states]
        result = compute_dop(0.0, 0.0, sat_ecefs, min_elevation_deg=5.0)
        assert isinstance(result, DOPResult)

    def test_dop_fewer_than_4_sats(self):
        """<4 visible satellites → infinite DOP."""
        from constellation_generator.domain.dilution_of_precision import compute_dop
        # Single satellite ECEF far away
        sat_ecefs = [(R_E + 550_000, 0.0, 0.0)]
        result = compute_dop(0.0, 0.0, sat_ecefs)
        assert result.gdop == float('inf')
        assert result.num_visible < 4

    def test_dop_more_sats_better(self):
        """More satellites → lower (better) GDOP."""
        from constellation_generator.domain.dilution_of_precision import compute_dop
        small = _constellation_states(num_planes=3, sats_per_plane=3)
        large = _constellation_states(num_planes=6, sats_per_plane=6)
        small_ecefs = [propagate_ecef_to(s, EPOCH) for s in small]
        large_ecefs = [propagate_ecef_to(s, EPOCH) for s in large]
        r_small = compute_dop(0.0, 0.0, small_ecefs, min_elevation_deg=5.0)
        r_large = compute_dop(0.0, 0.0, large_ecefs, min_elevation_deg=5.0)
        # More sats should give equal or better DOP (when both have ≥4 visible)
        if r_small.num_visible >= 4 and r_large.num_visible >= 4:
            assert r_large.gdop <= r_small.gdop + 0.1  # tolerance for geometry

    def test_dop_pdop_le_gdop(self):
        """PDOP ≤ GDOP invariant (GDOP includes time component)."""
        from constellation_generator.domain.dilution_of_precision import compute_dop
        states = _constellation_states(num_planes=6, sats_per_plane=6)
        sat_ecefs = [propagate_ecef_to(s, EPOCH) for s in states]
        result = compute_dop(0.0, 0.0, sat_ecefs, min_elevation_deg=5.0)
        if result.num_visible >= 4:
            assert result.pdop <= result.gdop + 1e-9

    def test_dop_component_relation(self):
        """HDOP² + VDOP² ≈ PDOP²."""
        from constellation_generator.domain.dilution_of_precision import compute_dop
        states = _constellation_states(num_planes=6, sats_per_plane=6)
        sat_ecefs = [propagate_ecef_to(s, EPOCH) for s in states]
        result = compute_dop(0.0, 0.0, sat_ecefs, min_elevation_deg=5.0)
        if result.num_visible >= 4:
            pdop_sq = result.pdop ** 2
            sum_sq = result.hdop ** 2 + result.vdop ** 2
            assert abs(pdop_sq - sum_sq) < 1e-6

    def test_dop_all_positive(self):
        from constellation_generator.domain.dilution_of_precision import compute_dop
        states = _constellation_states(num_planes=6, sats_per_plane=6)
        sat_ecefs = [propagate_ecef_to(s, EPOCH) for s in states]
        result = compute_dop(0.0, 0.0, sat_ecefs, min_elevation_deg=5.0)
        assert result.gdop >= 0
        assert result.pdop >= 0
        assert result.hdop >= 0
        assert result.vdop >= 0
        assert result.tdop >= 0

    def test_dop_spread_vs_clustered(self):
        """Well-spread geometry → lower DOP than clustered."""
        from constellation_generator.domain.dilution_of_precision import compute_dop
        # Spread: large constellation
        spread = _constellation_states(num_planes=6, sats_per_plane=6)
        spread_ecefs = [propagate_ecef_to(s, EPOCH) for s in spread]
        # Clustered: single plane, single sat repeated
        clustered_state = _leo_state(altitude_km=550)
        clustered_ecefs = [propagate_ecef_to(clustered_state, EPOCH)] * 10
        r_spread = compute_dop(0.0, 0.0, spread_ecefs, min_elevation_deg=5.0)
        r_clust = compute_dop(0.0, 0.0, clustered_ecefs, min_elevation_deg=5.0)
        # Clustered should have worse (higher) DOP or insufficient satellites
        if r_spread.num_visible >= 4:
            assert r_clust.gdop >= r_spread.gdop or r_clust.num_visible < 4

    def test_dop_num_visible_correct(self):
        from constellation_generator.domain.dilution_of_precision import compute_dop
        states = _constellation_states(num_planes=6, sats_per_plane=6)
        sat_ecefs = [propagate_ecef_to(s, EPOCH) for s in states]
        result = compute_dop(0.0, 0.0, sat_ecefs, min_elevation_deg=5.0)
        assert result.num_visible >= 0
        assert result.num_visible <= len(sat_ecefs)


# --- compute_dop_grid ---

class TestComputeDopGrid:

    def test_dop_grid_returns_list(self):
        from constellation_generator.domain.dilution_of_precision import (
            compute_dop_grid, DOPGridPoint,
        )
        states = _constellation_states(num_planes=4, sats_per_plane=4)
        result = compute_dop_grid(states, EPOCH, lat_step_deg=60, lon_step_deg=60)
        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], DOPGridPoint)

    def test_dop_grid_size_matches(self):
        from constellation_generator.domain.dilution_of_precision import compute_dop_grid
        states = _constellation_states(num_planes=4, sats_per_plane=4)
        result = compute_dop_grid(
            states, EPOCH,
            lat_step_deg=30, lon_step_deg=60,
            min_elevation_deg=10.0,
        )
        # lat: -90, -60, -30, 0, 30, 60, 90 = 7
        # lon: -180, -120, -60, 0, 60, 120 = 6
        # total = 7 * 6 = 42
        assert len(result) == 42

    def test_dop_grid_empty_states(self):
        from constellation_generator.domain.dilution_of_precision import compute_dop_grid
        result = compute_dop_grid([], EPOCH, lat_step_deg=90, lon_step_deg=90)
        for point in result:
            assert point.dop.gdop == float('inf')

    def test_dop_grid_point_has_coordinates(self):
        from constellation_generator.domain.dilution_of_precision import compute_dop_grid
        states = _constellation_states(num_planes=4, sats_per_plane=4)
        result = compute_dop_grid(states, EPOCH, lat_step_deg=90, lon_step_deg=90)
        for point in result:
            assert -90 <= point.lat_deg <= 90
            assert -180 <= point.lon_deg <= 180

    def test_dop_grid_constellation_coverage(self):
        """Dense constellation → at least some grid points with finite DOP."""
        from constellation_generator.domain.dilution_of_precision import compute_dop_grid
        # Need 4+ visible sats per point → large constellation at higher altitude
        states = _constellation_states(
            num_planes=12, sats_per_plane=12, altitude_km=1200,
        )
        result = compute_dop_grid(
            states, EPOCH, lat_step_deg=30, lon_step_deg=60,
            min_elevation_deg=5.0,
        )
        finite_count = sum(1 for p in result if p.dop.gdop < float('inf'))
        assert finite_count > 0


# --- Purity ---

class TestDOPModulePurity:

    def test_dop_module_pure(self):
        """Only stdlib + domain imports."""
        import ast
        import constellation_generator.domain.dilution_of_precision as mod
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {"math", "dataclasses", "datetime"}
        allowed_internal = {"constellation_generator"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import from: {node.module}"
