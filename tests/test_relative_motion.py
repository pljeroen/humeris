# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for CW/Hill relative motion equations."""
import ast
import math
from datetime import datetime, timezone

import pytest

from constellation_generator.domain.relative_motion import (
    CWTrajectory,
    RelativeState,
    compute_relative_state,
    cw_propagate,
    cw_propagate_state,
    is_passively_safe,
)


# Mean motion for 500 km LEO circular orbit
_MU = 3.986004418e14
_R_EARTH = 6_371_000.0
_A = _R_EARTH + 500_000.0
_N = math.sqrt(_MU / _A**3)


class TestCWPropagateState:
    """Tests for single-point analytical CW propagation."""

    def test_zero_initial_stays_origin(self):
        """Zero relative state remains at origin for all time."""
        state = RelativeState(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
        result = cw_propagate_state(state, _N, 3600.0)
        assert abs(result.x) < 1e-10
        assert abs(result.y) < 1e-10
        assert abs(result.z) < 1e-10

    def test_radial_offset_drifts(self):
        """x₀≠0 with ẏ₀=0 → along-track drift."""
        state = RelativeState(x=100.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
        T = 2.0 * math.pi / _N
        result = cw_propagate_state(state, _N, T)
        # One period: y should have drifted (secular term -6πnx₀/n * period)
        assert abs(result.y) > 100.0

    def test_bounded_orbit(self):
        """ẏ₀ = -2nx₀ → no secular along-track drift."""
        x0 = 100.0
        vy0 = -2.0 * _N * x0
        state = RelativeState(x=x0, y=0.0, z=0.0, vx=0.0, vy=vy0, vz=0.0)
        T = 2.0 * math.pi / _N
        result = cw_propagate_state(state, _N, T)
        # After one full period, should return near initial y (no secular drift)
        assert abs(result.y) < 1.0  # meters

    def test_cross_track_harmonic(self):
        """z oscillates with period 2π/n."""
        z0 = 50.0
        state = RelativeState(x=0.0, y=0.0, z=z0, vx=0.0, vy=0.0, vz=0.0)
        T = 2.0 * math.pi / _N
        result = cw_propagate_state(state, _N, T)
        # After one period, z should return to z0
        assert abs(result.z - z0) < 1e-6

    def test_returns_relative_state(self):
        """Return type is RelativeState."""
        state = RelativeState(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
        result = cw_propagate_state(state, _N, 100.0)
        assert isinstance(result, RelativeState)

    def test_symmetry_half_period(self):
        """x(T/2) for bounded orbit: x reverses sign from x(0) for radial offset."""
        x0 = 100.0
        vy0 = -2.0 * _N * x0
        state = RelativeState(x=x0, y=0.0, z=0.0, vx=0.0, vy=vy0, vz=0.0)
        T = 2.0 * math.pi / _N
        half = cw_propagate_state(state, _N, T / 2.0)
        # For bounded CW with vx₀=0: x(T/2) = (4-3cos(π))x₀ + 2(1-cos(π))/n * vy₀
        # cos(π) = -1: x(T/2) = 7x₀ + 4vy₀/n = 7x₀ + 4(-2nx₀)/n = 7x₀ - 8x₀ = -x₀
        assert abs(half.x - (-x0)) < 1e-6


class TestCWTrajectory:
    """Tests for CW trajectory time series."""

    def test_trajectory_returns_type(self):
        """Return type is CWTrajectory."""
        state = RelativeState(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
        result = cw_propagate(state, _N, 3600.0, 60.0)
        assert isinstance(result, CWTrajectory)

    def test_trajectory_snapshot_count(self):
        """Number of snapshots = duration/step + 1."""
        state = RelativeState(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
        result = cw_propagate(state, _N, 3600.0, 60.0)
        assert len(result.states) == 61  # 0, 60, 120, ..., 3600

    def test_trajectory_bounded_flag_true(self):
        """Bounded case correctly flagged."""
        x0 = 100.0
        vy0 = -2.0 * _N * x0
        state = RelativeState(x=x0, y=0.0, z=0.0, vx=0.0, vy=vy0, vz=0.0)
        result = cw_propagate(state, _N, 3600.0, 60.0)
        assert result.is_bounded is True

    def test_trajectory_bounded_flag_false(self):
        """Unbounded case correctly flagged."""
        state = RelativeState(x=100.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
        result = cw_propagate(state, _N, 3600.0, 60.0)
        assert result.is_bounded is False


class TestComputeRelativeState:
    """Tests for converting two OrbitalStates to LVLH relative state."""

    def test_relative_state_from_close_orbits(self):
        """Two close orbits → small relative distance."""
        from constellation_generator.domain.propagation import OrbitalState

        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        chief = OrbitalState(
            semi_major_axis_m=_A,
            eccentricity=0.0,
            inclination_rad=math.radians(53.0),
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=_N,
            reference_epoch=epoch,
        )
        deputy = OrbitalState(
            semi_major_axis_m=_A + 100.0,
            eccentricity=0.0,
            inclination_rad=math.radians(53.0),
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=math.sqrt(_MU / (_A + 100.0) ** 3),
            reference_epoch=epoch,
        )
        rel = compute_relative_state(chief, deputy, epoch)
        dist = math.sqrt(rel.x**2 + rel.y**2 + rel.z**2)
        assert dist < 500.0  # meters — close orbits

    def test_relative_state_type(self):
        """Return type is RelativeState."""
        from constellation_generator.domain.propagation import OrbitalState

        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        state = OrbitalState(
            semi_major_axis_m=_A,
            eccentricity=0.0,
            inclination_rad=math.radians(53.0),
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=_N,
            reference_epoch=epoch,
        )
        rel = compute_relative_state(state, state, epoch)
        assert isinstance(rel, RelativeState)


class TestPassivelySafe:
    """Tests for passive safety check."""

    def test_passively_safe_bounded(self):
        """Bounded orbit with sufficient separation → safe."""
        x0 = 200.0
        vy0 = -2.0 * _N * x0
        state = RelativeState(x=x0, y=0.0, z=0.0, vx=0.0, vy=vy0, vz=0.0)
        T = 2.0 * math.pi / _N
        assert is_passively_safe(state, _N, min_distance_m=10.0, check_duration_s=2 * T) is True

    def test_passively_safe_collision(self):
        """Orbit through origin → not safe."""
        state = RelativeState(x=0.0, y=0.0, z=0.0, vx=1.0, vy=0.0, vz=0.0)
        T = 2.0 * math.pi / _N
        assert is_passively_safe(state, _N, min_distance_m=10.0, check_duration_s=2 * T) is False


class TestRelativeMotionPurity:
    """Domain purity: relative_motion.py must only import stdlib + domain."""

    def test_module_pure(self):
        import constellation_generator.domain.relative_motion as mod

        allowed = {'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
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
