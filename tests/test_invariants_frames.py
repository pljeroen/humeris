# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Invariant tests for coordinate frame conversions (ECI/ECEF).

These verify mathematical properties of rotation transforms that must
always hold regardless of input.

Invariants B1-B4 from the formal invariant specification.
"""

import math
from datetime import datetime, timezone

import pytest

from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
)
from humeris.domain.orbital_mechanics import OrbitalConstants


# Deterministic test vectors (position, velocity pairs)
_TEST_VECTORS = [
    ((7_000_000.0, 0.0, 0.0), (0.0, 7500.0, 0.0), "x-axis LEO"),
    ((0.0, 7_000_000.0, 0.0), (-7500.0, 0.0, 0.0), "y-axis LEO"),
    ((0.0, 0.0, 7_000_000.0), (7500.0, 0.0, 0.0), "z-axis (polar)"),
    ((5_000_000.0, 3_000_000.0, 4_000_000.0), (-2000.0, 6000.0, 3000.0), "arbitrary"),
    ((42_164_000.0, 0.0, 0.0), (0.0, 3075.0, 0.0), "GEO"),
]

_TEST_EPOCHS = [
    datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 6, 21, 6, 30, 0, tzinfo=timezone.utc),
    datetime(2026, 12, 21, 18, 45, 0, tzinfo=timezone.utc),
]


def _mag(v):
    return math.sqrt(sum(x ** 2 for x in v))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class TestB1NormPreservation:
    """B1: Rotation preserves position norm. Velocity includes Coriolis."""

    @pytest.mark.parametrize("pos,vel,label", _TEST_VECTORS)
    def test_position_norm(self, pos, vel, label):
        for epoch in _TEST_EPOCHS:
            gmst = gmst_rad(epoch)
            pos_ecef, _ = eci_to_ecef(pos, vel, gmst)
            assert abs(_mag(pos_ecef) - _mag(pos)) / _mag(pos) < 1e-12, \
                f"{label} at {epoch}"

    @pytest.mark.parametrize("pos,vel,label", _TEST_VECTORS)
    def test_velocity_norm(self, pos, vel, label):
        """v_ECEF = R*v_ECI - omega x r_ECEF matches returned velocity."""
        omega_e = OrbitalConstants.EARTH_ROTATION_RATE
        for epoch in _TEST_EPOCHS:
            gmst = gmst_rad(epoch)
            cos_t = math.cos(gmst)
            sin_t = math.sin(gmst)
            pos_ecef, vel_ecef = eci_to_ecef(pos, vel, gmst)
            # R * v_ECI
            rv_x = cos_t * vel[0] + sin_t * vel[1]
            rv_y = -sin_t * vel[0] + cos_t * vel[1]
            rv_z = vel[2]
            # omega x r_ECEF
            cross_x = -omega_e * pos_ecef[1]
            cross_y = omega_e * pos_ecef[0]
            # Expected: R*v - omega x r
            expected = (rv_x - cross_x, rv_y - cross_y, rv_z)
            for i in range(3):
                assert abs(vel_ecef[i] - expected[i]) < 1e-6, \
                    f"{label} at {epoch} component {i}"


class TestB2DotProductPreservation:
    """B2: Rotation preserves dot products. dot(Ru, Rv) == dot(u, v)."""

    def test_dot_product_preserved(self):
        u_pos = (7_000_000.0, 0.0, 0.0)
        u_vel = (0.0, 7500.0, 0.0)
        v_pos = (0.0, 7_000_000.0, 0.0)
        v_vel = (-7500.0, 0.0, 0.0)

        for epoch in _TEST_EPOCHS:
            gmst = gmst_rad(epoch)
            u_ecef, _ = eci_to_ecef(u_pos, u_vel, gmst)
            v_ecef, _ = eci_to_ecef(v_pos, v_vel, gmst)

            dot_eci = _dot(u_pos, v_pos)
            dot_ecef = _dot(u_ecef, v_ecef)
            assert abs(dot_ecef - dot_eci) < 1e-2, \
                f"Dot product diverged at {epoch}: {dot_eci} vs {dot_ecef}"


class TestB3RoundTrip:
    """B3: ECI->ECEF->ECI round-trip recovers original state.

    Position: inverse rotation by -gmst recovers pos_ECI.
    Velocity: must undo Coriolis before inverse rotation:
        v_ECI = R^T * (v_ECEF + omega x r_ECEF)
    """

    @pytest.mark.parametrize("pos,vel,label", _TEST_VECTORS)
    def test_inverse_rotation(self, pos, vel, label):
        omega_e = OrbitalConstants.EARTH_ROTATION_RATE
        for epoch in _TEST_EPOCHS:
            gmst = gmst_rad(epoch)
            cos_t = math.cos(gmst)
            sin_t = math.sin(gmst)
            pos_ecef, vel_ecef = eci_to_ecef(pos, vel, gmst)

            # Position inverse: R^T * pos_ECEF (rotation by -gmst)
            pos_back = (
                cos_t * pos_ecef[0] - sin_t * pos_ecef[1],
                sin_t * pos_ecef[0] + cos_t * pos_ecef[1],
                pos_ecef[2],
            )
            for i in range(3):
                assert abs(pos_back[i] - pos[i]) < 1e-4, \
                    f"{label} pos[{i}] roundtrip: {pos_back[i]} vs {pos[i]}"

            # Velocity inverse: v_ECI = R^T * (v_ECEF + omega x r_ECEF)
            # omega x r_ECEF = (-omega_E * ry, omega_E * rx, 0)
            corrected_vx = vel_ecef[0] + (-omega_e * pos_ecef[1])
            corrected_vy = vel_ecef[1] + (omega_e * pos_ecef[0])
            corrected_vz = vel_ecef[2]
            vel_back = (
                cos_t * corrected_vx - sin_t * corrected_vy,
                sin_t * corrected_vx + cos_t * corrected_vy,
                corrected_vz,
            )
            for i in range(3):
                assert abs(vel_back[i] - vel[i]) < 1e-4, \
                    f"{label} vel[{i}] roundtrip: {vel_back[i]} vs {vel[i]}"


class TestB4GmstPeriodicity:
    """B4: eci_to_ecef(r, gmst) == eci_to_ecef(r, gmst + 2pi)."""

    @pytest.mark.parametrize("pos,vel,label", _TEST_VECTORS)
    def test_two_pi_periodicity(self, pos, vel, label):
        for epoch in _TEST_EPOCHS:
            gmst = gmst_rad(epoch)
            pos_1, vel_1 = eci_to_ecef(pos, vel, gmst)
            pos_2, vel_2 = eci_to_ecef(pos, vel, gmst + 2 * math.pi)
            for i in range(3):
                assert abs(pos_1[i] - pos_2[i]) < 1e-4, \
                    f"{label} pos[{i}]: {pos_1[i]} vs {pos_2[i]}"
                assert abs(vel_1[i] - vel_2[i]) < 1e-4, \
                    f"{label} vel[{i}]: {vel_1[i]} vs {vel_2[i]}"
