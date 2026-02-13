# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for inter-satellite link geometry and topology.

Covers Earth blockage detection, ISL link computation, and
constellation-wide topology analysis.
"""

import math
from datetime import datetime, timezone

import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.propagation import OrbitalState, derive_orbital_state, propagate_to
from humeris.domain.constellation import ShellConfig, generate_walker_shell

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


def _multi_sat_states(num_planes=3, sats_per_plane=4, altitude_km=550):
    """Helper: multiple satellite states for topology tests."""
    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=53,
        num_planes=num_planes, sats_per_plane=sats_per_plane,
        phase_factor=1, raan_offset_deg=0, shell_name="Multi",
    )
    sats = generate_walker_shell(shell)
    return [derive_orbital_state(sat, EPOCH) for sat in sats]


# --- is_earth_blocked ---

class TestIsEarthBlocked:

    def test_clear_link_same_side(self):
        """Two sats on the same side of Earth → not blocked."""
        from humeris.domain.inter_satellite_links import is_earth_blocked
        # Both at ~R_E + 550km, near each other in ECI
        r = R_E + 550_000
        pos_a = (r, 0.0, 0.0)
        pos_b = (r * math.cos(0.1), r * math.sin(0.1), 0.0)  # ~6° apart
        assert not is_earth_blocked(pos_a, pos_b)

    def test_blocked_link_opposite_sides(self):
        """Sats on opposite sides of Earth → blocked."""
        from humeris.domain.inter_satellite_links import is_earth_blocked
        r = R_E + 550_000
        pos_a = (r, 0.0, 0.0)
        pos_b = (-r, 0.0, 0.0)
        assert is_earth_blocked(pos_a, pos_b)

    def test_grazing_link(self):
        """Link just grazing Earth limb → correct classification."""
        from humeris.domain.inter_satellite_links import is_earth_blocked
        r = R_E + 550_000
        # Place sats 90° apart — line crosses near Earth surface
        pos_a = (r, 0.0, 0.0)
        pos_b = (0.0, r, 0.0)
        # Midpoint at (r/2, r/2, 0), distance from origin = r/sqrt(2) ≈ 4.9M
        # R_E ≈ 6.37M → midpoint is inside Earth → blocked
        assert is_earth_blocked(pos_a, pos_b)

    def test_same_position_not_blocked(self):
        """Degenerate: same point → not blocked."""
        from humeris.domain.inter_satellite_links import is_earth_blocked
        r = R_E + 550_000
        pos = (r, 0.0, 0.0)
        assert not is_earth_blocked(pos, pos)


# --- compute_isl_link ---

class TestComputeIslLink:

    def test_link_returns_type(self):
        from humeris.domain.inter_satellite_links import (
            compute_isl_link, ISLLink,
        )
        r = R_E + 550_000
        pos_a = (r, 0.0, 0.0)
        pos_b = (r * math.cos(0.1), r * math.sin(0.1), 0.0)
        link = compute_isl_link(pos_a, pos_b, 0, 1)
        assert isinstance(link, ISLLink)

    def test_link_distance_correct(self):
        from humeris.domain.inter_satellite_links import compute_isl_link
        r = R_E + 550_000
        pos_a = (r, 0.0, 0.0)
        pos_b = (r + 1000.0, 0.0, 0.0)
        link = compute_isl_link(pos_a, pos_b, 0, 1)
        assert abs(link.distance_m - 1000.0) < 0.01

    def test_link_blocked_flag(self):
        from humeris.domain.inter_satellite_links import compute_isl_link
        r = R_E + 550_000
        pos_a = (r, 0.0, 0.0)
        pos_b = (-r, 0.0, 0.0)
        link = compute_isl_link(pos_a, pos_b, 0, 1)
        assert link.is_blocked is True


# --- compute_isl_topology ---

class TestComputeIslTopology:

    def test_topology_returns_type(self):
        from humeris.domain.inter_satellite_links import (
            compute_isl_topology, ISLTopology,
        )
        states = _multi_sat_states(num_planes=2, sats_per_plane=2)
        result = compute_isl_topology(states, EPOCH, max_range_km=5000)
        assert isinstance(result, ISLTopology)

    def test_topology_link_count(self):
        """N satellites → N*(N-1)/2 total pairs evaluated."""
        from humeris.domain.inter_satellite_links import compute_isl_topology
        states = _multi_sat_states(num_planes=2, sats_per_plane=2)
        n = len(states)
        result = compute_isl_topology(states, EPOCH, max_range_km=50000)
        # Total links should be at most n*(n-1)/2
        assert len(result.links) <= n * (n - 1) // 2

    def test_topology_max_range_filter(self):
        """Links beyond max_range excluded from active count."""
        from humeris.domain.inter_satellite_links import compute_isl_topology
        states = _multi_sat_states(num_planes=3, sats_per_plane=4)
        # Very short range → most links filtered
        short = compute_isl_topology(states, EPOCH, max_range_km=100)
        # Generous range → more active links
        long = compute_isl_topology(states, EPOCH, max_range_km=10000)
        assert short.num_active_links <= long.num_active_links

    def test_topology_empty_states(self):
        from humeris.domain.inter_satellite_links import compute_isl_topology
        result = compute_isl_topology([], EPOCH, max_range_km=5000)
        assert result.num_active_links == 0
        assert result.num_satellites == 0

    def test_topology_single_sat(self):
        from humeris.domain.inter_satellite_links import compute_isl_topology
        states = [_leo_state()]
        result = compute_isl_topology(states, EPOCH, max_range_km=5000)
        assert result.num_active_links == 0
        assert len(result.links) == 0

    def test_topology_statistics(self):
        """Mean/max distance values correct for active links."""
        from humeris.domain.inter_satellite_links import compute_isl_topology
        states = _multi_sat_states(num_planes=3, sats_per_plane=4)
        result = compute_isl_topology(states, EPOCH, max_range_km=10000)
        if result.num_active_links > 0:
            assert result.mean_distance_m > 0
            assert result.max_distance_m >= result.mean_distance_m
        else:
            assert result.mean_distance_m == 0.0
            assert result.max_distance_m == 0.0

    def test_topology_active_excludes_blocked(self):
        """Blocked links not counted in active count."""
        from humeris.domain.inter_satellite_links import compute_isl_topology
        states = _multi_sat_states(num_planes=3, sats_per_plane=4)
        result = compute_isl_topology(states, EPOCH, max_range_km=50000)
        blocked_count = sum(1 for link in result.links if link.is_blocked)
        in_range_count = sum(
            1 for link in result.links
            if link.distance_m <= 50000 * 1000
        )
        expected_active = in_range_count - sum(
            1 for link in result.links
            if link.distance_m <= 50000 * 1000 and link.is_blocked
        )
        assert result.num_active_links == expected_active


# --- Purity ---

class TestISLModulePurity:

    def test_isl_module_pure(self):
        """Only stdlib + domain imports."""
        import ast
        import humeris.domain.inter_satellite_links as mod
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {"math", "numpy", "dataclasses", "datetime"}
        allowed_internal = {"humeris"}

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
