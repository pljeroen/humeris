# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for gravity gradient and aerodynamic torques."""
import ast
import math
from datetime import datetime, timezone

import pytest

from humeris.domain.torques import (
    AerodynamicTorqueResult,
    InertiaTensor,
    TorqueResult,
    compute_aerodynamic_torque,
    compute_gravity_gradient_torque,
)


_R_EARTH = 6_371_000.0
_LEO_R = _R_EARTH + 500_000.0


class TestGravityGradientTorque:
    """Tests for gravity gradient torque computation."""

    def test_gg_torque_returns_type(self):
        """Return type is TorqueResult."""
        inertia = InertiaTensor(ixx=10.0, iyy=20.0, izz=30.0)
        pos = (_LEO_R, 0.0, 0.0)
        result = compute_gravity_gradient_torque(pos, inertia)
        assert isinstance(result, TorqueResult)

    def test_gg_symmetric_inertia_zero(self):
        """Ix=Iy=Iz → zero torque."""
        inertia = InertiaTensor(ixx=10.0, iyy=10.0, izz=10.0)
        pos = (_LEO_R, 1000.0, 0.0)  # Slight off-nadir for non-degenerate case
        result = compute_gravity_gradient_torque(pos, inertia)
        assert result.magnitude_nm < 1e-10

    def test_gg_asymmetric_nonzero(self):
        """Different moments → nonzero torque."""
        inertia = InertiaTensor(ixx=10.0, iyy=20.0, izz=30.0)
        # Position not aligned with single axis
        pos = (_LEO_R * 0.7071, _LEO_R * 0.7071, 0.0)
        result = compute_gravity_gradient_torque(pos, inertia)
        assert result.magnitude_nm > 0

    def test_gg_magnitude_order(self):
        """~1e-5 Nm for typical LEO small sat."""
        inertia = InertiaTensor(ixx=5.0, iyy=10.0, izz=15.0)
        pos = (_LEO_R * 0.7071, _LEO_R * 0.7071, 0.0)
        result = compute_gravity_gradient_torque(pos, inertia)
        assert 1e-8 < result.magnitude_nm < 1e-2

    def test_gg_proportional_to_inertia_diff(self):
        """Larger asymmetry → larger torque."""
        pos = (_LEO_R * 0.7071, _LEO_R * 0.7071, 0.0)
        small = compute_gravity_gradient_torque(
            pos, InertiaTensor(ixx=10.0, iyy=11.0, izz=12.0))
        large = compute_gravity_gradient_torque(
            pos, InertiaTensor(ixx=10.0, iyy=20.0, izz=30.0))
        assert large.magnitude_nm > small.magnitude_nm

    def test_gg_inverse_cube_distance(self):
        """Higher altitude → smaller torque."""
        inertia = InertiaTensor(ixx=10.0, iyy=20.0, izz=30.0)
        pos_low = ((_R_EARTH + 400_000.0) * 0.7071, (_R_EARTH + 400_000.0) * 0.7071, 0.0)
        pos_high = ((_R_EARTH + 800_000.0) * 0.7071, (_R_EARTH + 800_000.0) * 0.7071, 0.0)
        t_low = compute_gravity_gradient_torque(pos_low, inertia)
        t_high = compute_gravity_gradient_torque(pos_high, inertia)
        assert t_low.magnitude_nm > t_high.magnitude_nm


class TestAerodynamicTorque:
    """Tests for aerodynamic torque computation."""

    def test_aero_torque_returns_type(self):
        """Return type is AerodynamicTorqueResult."""
        from humeris.domain.atmosphere import DragConfig
        pos = (_LEO_R, 0.0, 0.0)
        vel = (0.0, 7600.0, 0.0)
        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        cp_offset = (0.01, 0.0, 0.0)
        result = compute_aerodynamic_torque(pos, vel, drag, cp_offset)
        assert isinstance(result, AerodynamicTorqueResult)

    def test_aero_zero_offset_zero_torque(self):
        """cp_offset=(0,0,0) → zero torque."""
        from humeris.domain.atmosphere import DragConfig
        pos = (_LEO_R, 0.0, 0.0)
        vel = (0.0, 7600.0, 0.0)
        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        result = compute_aerodynamic_torque(pos, vel, drag, (0.0, 0.0, 0.0))
        assert result.magnitude_nm < 1e-15

    def test_aero_nonzero_offset(self):
        """Nonzero CP offset → nonzero torque."""
        from humeris.domain.atmosphere import DragConfig
        pos = (_LEO_R, 0.0, 0.0)
        vel = (0.0, 7600.0, 0.0)
        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        result = compute_aerodynamic_torque(pos, vel, drag, (0.01, 0.02, 0.0))
        assert result.magnitude_nm > 0

    def test_aero_proportional_to_offset(self):
        """Bigger offset → bigger torque."""
        from humeris.domain.atmosphere import DragConfig
        pos = (_LEO_R, 0.0, 0.0)
        vel = (0.0, 7600.0, 0.0)
        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        small = compute_aerodynamic_torque(pos, vel, drag, (0.01, 0.0, 0.0))
        large = compute_aerodynamic_torque(pos, vel, drag, (0.1, 0.0, 0.0))
        assert large.magnitude_nm > small.magnitude_nm

    def test_aero_decreases_with_altitude(self):
        """Higher altitude → less drag → less torque."""
        from humeris.domain.atmosphere import DragConfig
        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        cp_off = (0.01, 0.0, 0.0)
        pos_low = (_R_EARTH + 300_000.0, 0.0, 0.0)
        vel_low = (0.0, 7700.0, 0.0)
        pos_high = (_R_EARTH + 800_000.0, 0.0, 0.0)
        vel_high = (0.0, 7400.0, 0.0)
        t_low = compute_aerodynamic_torque(pos_low, vel_low, drag, cp_off)
        t_high = compute_aerodynamic_torque(pos_high, vel_high, drag, cp_off)
        assert t_low.magnitude_nm > t_high.magnitude_nm


class TestTorquesPurity:
    """Domain purity: torques.py must only import stdlib + domain."""

    def test_module_pure(self):
        import humeris.domain.torques as mod

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
