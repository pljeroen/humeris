# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/functorial_composition.py — functorial force model composition."""
import math
from datetime import datetime

import numpy as np
import pytest

from humeris.domain.functorial_composition import (
    CompositionResult,
    ForceCategory,
    NaturalTransformation,
    PullbackForce,
    compose_forces,
    verify_associativity,
    natural_transform_force,
    check_commutativity_diagram,
    pullback_force,
    identity_force,
    compute_force_decomposition,
    ForceModel,
)

# ---------------------------------------------------------------------------
# Test constants — typical LEO state
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, 12, 0, 0)
_POS = (6778137.0, 0.0, 0.0)       # ~400 km altitude on x-axis
_VEL = (0.0, 7668.0, 0.0)          # circular LEO velocity


# ---------------------------------------------------------------------------
# Mock force models
# ---------------------------------------------------------------------------

class ConstantForce:
    """Force model returning a fixed acceleration vector."""

    def __init__(self, ax: float, ay: float, az: float):
        self._a = (ax, ay, az)

    def acceleration(self, epoch, position, velocity):
        return self._a


class RadialForce:
    """Force model with acceleration purely along the position vector."""

    def __init__(self, magnitude: float):
        self._mag = magnitude

    def acceleration(self, epoch, position, velocity):
        r = (position[0] ** 2 + position[1] ** 2 + position[2] ** 2) ** 0.5
        return tuple(self._mag * p / r for p in position)


# ---------------------------------------------------------------------------
# Identity morphism
# ---------------------------------------------------------------------------

class TestIdentityForce:
    def test_identity_force_returns_zero(self):
        f = identity_force()
        acc = f.acceleration(_EPOCH, _POS, _VEL)
        assert acc == (0.0, 0.0, 0.0)

    def test_identity_satisfies_protocol(self):
        f = identity_force()
        assert isinstance(f, ForceModel)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class TestComposeForces:
    def test_compose_single_force(self):
        f = ConstantForce(1.0, 2.0, 3.0)
        result = compose_forces([("f", f)], _EPOCH, _POS, _VEL)
        assert isinstance(result, CompositionResult)
        assert result.total_acceleration == pytest.approx((1.0, 2.0, 3.0))

    def test_compose_two_body_j2(self):
        """Two constant forces compose by vector addition."""
        two_body = ConstantForce(-8.0, 0.0, 0.0)
        j2 = ConstantForce(-0.001, 0.0, 0.0002)
        result = compose_forces(
            [("two_body", two_body), ("j2", j2)],
            _EPOCH, _POS, _VEL,
        )
        assert result.total_acceleration[0] == pytest.approx(-8.001)
        assert result.total_acceleration[1] == pytest.approx(0.0)
        assert result.total_acceleration[2] == pytest.approx(0.0002)

    def test_composition_result_per_model(self):
        """Per-model accelerations sum to total."""
        f1 = ConstantForce(1.0, 2.0, 3.0)
        f2 = ConstantForce(4.0, 5.0, 6.0)
        result = compose_forces(
            [("f1", f1), ("f2", f2)], _EPOCH, _POS, _VEL,
        )
        sum_per_model = [0.0, 0.0, 0.0]
        for _, acc in result.per_model_accelerations:
            for k in range(3):
                sum_per_model[k] += acc[k]
        for k in range(3):
            assert sum_per_model[k] == pytest.approx(result.total_acceleration[k])

    def test_composition_residual_zero_for_linear(self):
        """For additive (linear) forces, the commutativity residual is zero."""
        f1 = ConstantForce(1.0, 0.0, 0.0)
        f2 = ConstantForce(0.0, 1.0, 0.0)
        result = compose_forces(
            [("f1", f1), ("f2", f2)], _EPOCH, _POS, _VEL,
        )
        assert result.composition_residual == pytest.approx(0.0, abs=1e-14)
        assert result.is_order_independent is True

    def test_composition_is_commutative_for_conservative(self):
        """Conservative (state-independent constant) forces commute."""
        f1 = ConstantForce(3.0, -1.0, 2.0)
        f2 = ConstantForce(-0.5, 4.0, 0.0)
        result = compose_forces(
            [("f1", f1), ("f2", f2)], _EPOCH, _POS, _VEL,
        )
        assert result.is_order_independent is True

    def test_compose_with_identity(self):
        """Composing a force with the identity leaves it unchanged."""
        f = ConstantForce(7.0, -3.0, 1.5)
        result_with_id = compose_forces(
            [("f", f), ("id", identity_force())], _EPOCH, _POS, _VEL,
        )
        result_alone = compose_forces(
            [("f", f)], _EPOCH, _POS, _VEL,
        )
        for k in range(3):
            assert result_with_id.total_acceleration[k] == pytest.approx(
                result_alone.total_acceleration[k],
            )


# ---------------------------------------------------------------------------
# Associativity
# ---------------------------------------------------------------------------

class TestAssociativity:
    def test_associativity_three_forces(self):
        """(a + b) + c == a + (b + c) for force superposition."""
        a = [("a", ConstantForce(1.0, 0.0, 0.0))]
        b = [("b", ConstantForce(0.0, 2.0, 0.0))]
        c = [("c", ConstantForce(0.0, 0.0, 3.0))]
        assert verify_associativity(a, b, c, _EPOCH, _POS, _VEL) is True

    def test_associativity_radial_forces(self):
        """Associativity holds for position-dependent (radial) forces too."""
        a = [("a", RadialForce(1e-3))]
        b = [("b", RadialForce(-2e-3))]
        c = [("c", RadialForce(5e-4))]
        assert verify_associativity(a, b, c, _EPOCH, _POS, _VEL) is True


# ---------------------------------------------------------------------------
# Natural transformations
# ---------------------------------------------------------------------------

def _identity_matrix_flat() -> tuple:
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _rotation_z_90_flat() -> tuple:
    """90-degree rotation about z-axis (row-major flattened)."""
    c, s = 0.0, 1.0  # cos(90)=0, sin(90)=1
    return (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


class TestNaturalTransformation:
    def test_natural_transform_identity_rotation(self):
        """Identity rotation preserves the acceleration vector."""
        f = ConstantForce(5.0, -3.0, 2.0)
        t = NaturalTransformation(
            source_frame="ECI",
            target_frame="ECI",
            transform_matrix=_identity_matrix_flat(),
            is_invertible=True,
        )
        result = natural_transform_force(f, t, _EPOCH, _POS, _VEL)
        assert isinstance(result, PullbackForce)
        assert result.acceleration_in_target == pytest.approx((5.0, -3.0, 2.0))
        assert result.jacobian_correction == pytest.approx(0.0, abs=1e-14)

    def test_natural_transform_90deg_rotation(self):
        """90-degree z-rotation maps (1,0,0) -> (0,1,0)."""
        f = ConstantForce(1.0, 0.0, 0.0)
        t = NaturalTransformation(
            source_frame="ECI",
            target_frame="ECEF",
            transform_matrix=_rotation_z_90_flat(),
            is_invertible=True,
        )
        result = natural_transform_force(f, t, _EPOCH, _POS, _VEL)
        assert result.acceleration_in_target[0] == pytest.approx(0.0, abs=1e-14)
        assert result.acceleration_in_target[1] == pytest.approx(1.0)
        assert result.acceleration_in_target[2] == pytest.approx(0.0, abs=1e-14)

    def test_natural_transform_invertible(self):
        """Transform and inverse compose to identity."""
        f = ConstantForce(3.0, -2.0, 7.0)
        r_flat = _rotation_z_90_flat()
        r_mat = np.array(r_flat).reshape(3, 3)
        r_inv_flat = tuple(float(x) for x in r_mat.T.ravel())

        t_fwd = NaturalTransformation("A", "B", r_flat, True)
        t_inv = NaturalTransformation("B", "A", r_inv_flat, True)

        # Forward
        fwd = natural_transform_force(f, t_fwd, _EPOCH, _POS, _VEL)
        # Apply inverse to the forward result via a literal force model
        fwd_force = ConstantForce(*fwd.acceleration_in_target)
        back = natural_transform_force(fwd_force, t_inv, _EPOCH, _POS, _VEL)

        original_acc = f.acceleration(_EPOCH, _POS, _VEL)
        for k in range(3):
            assert back.acceleration_in_target[k] == pytest.approx(
                original_acc[k], abs=1e-12,
            )


# ---------------------------------------------------------------------------
# Pullback
# ---------------------------------------------------------------------------

class TestPullback:
    def test_pullback_identity_frame(self):
        """Pullback with identity rotation == original force evaluation."""
        f = ConstantForce(2.0, -1.0, 0.5)
        result = pullback_force(f, _identity_matrix_flat(), _EPOCH, _POS, _VEL)
        assert isinstance(result, PullbackForce)
        assert result.acceleration_in_target == pytest.approx((2.0, -1.0, 0.5))
        assert result.jacobian_correction == pytest.approx(0.0, abs=1e-14)

    def test_pullback_rotated_frame(self):
        """Pullback with 90-degree z-rotation gives rotated acceleration.

        For a constant (state-independent) force, the pullback is simply R @ a,
        because force(R^T @ pos, R^T @ vel) == force(any state) == a.
        So a' = R @ a.
        """
        f = ConstantForce(1.0, 0.0, 0.0)
        result = pullback_force(f, _rotation_z_90_flat(), _EPOCH, _POS, _VEL)
        # R_z(90) @ (1,0,0) = (0,1,0)
        assert result.acceleration_in_target[0] == pytest.approx(0.0, abs=1e-14)
        assert result.acceleration_in_target[1] == pytest.approx(1.0)
        assert result.acceleration_in_target[2] == pytest.approx(0.0, abs=1e-14)

    def test_pullback_radial_force_state_dependent(self):
        """For a state-dependent force, pullback evaluates at rotated state."""
        f = RadialForce(1.0)
        r_flat = _rotation_z_90_flat()
        r_mat = np.array(r_flat).reshape(3, 3)

        # Manually: R^T @ pos, R^T @ vel gives the source-frame state
        pos_source = tuple(float(x) for x in r_mat.T @ np.array(_POS))
        vel_source = tuple(float(x) for x in r_mat.T @ np.array(_VEL))
        acc_source = np.array(f.acceleration(_EPOCH, pos_source, vel_source))
        expected = tuple(float(x) for x in r_mat @ acc_source)

        result = pullback_force(f, r_flat, _EPOCH, _POS, _VEL)
        for k in range(3):
            assert result.acceleration_in_target[k] == pytest.approx(
                expected[k], abs=1e-10,
            )


# ---------------------------------------------------------------------------
# Commutativity diagram
# ---------------------------------------------------------------------------

class TestCommutativityDiagram:
    def test_commutativity_diagram_conservative(self):
        """Conservative (constant) forces commute through any rotation."""
        forces = [
            ("f1", ConstantForce(1.0, 2.0, 3.0)),
            ("f2", ConstantForce(-1.0, 0.5, -0.5)),
        ]
        transforms = [
            NaturalTransformation("A", "B", _rotation_z_90_flat(), True),
        ]
        commutes, max_res, pairs = check_commutativity_diagram(
            forces, transforms, _EPOCH, _POS, _VEL,
        )
        assert commutes is True
        assert max_res < 1e-10
        assert len(pairs) == 1  # one pair: (f1, f2)

    def test_commutativity_diagram_identity_transform(self):
        """Identity transform trivially commutes."""
        forces = [
            ("f1", ConstantForce(5.0, 0.0, 0.0)),
            ("f2", ConstantForce(0.0, 5.0, 0.0)),
        ]
        transforms = [
            NaturalTransformation("A", "A", _identity_matrix_flat(), True),
        ]
        commutes, max_res, pairs = check_commutativity_diagram(
            forces, transforms, _EPOCH, _POS, _VEL,
        )
        assert commutes is True
        assert max_res == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# RTN decomposition
# ---------------------------------------------------------------------------

class TestForceDecomposition:
    def test_force_decomposition_radial_only(self):
        """A purely radial force decomposes to (r, 0, 0) in RTN."""
        # Position on x-axis → radial = +x direction
        f = ConstantForce(1.0, 0.0, 0.0)
        result = compute_force_decomposition(
            [("f", f)], _EPOCH, _POS, _VEL,
        )
        assert len(result) == 1
        name, radial, along_track, cross_track = result[0]
        assert name == "f"
        assert radial == pytest.approx(1.0)
        assert along_track == pytest.approx(0.0, abs=1e-14)
        assert cross_track == pytest.approx(0.0, abs=1e-14)

    def test_force_decomposition_along_track(self):
        """A force along velocity decomposes to (0, T, 0) in RTN.

        Position = (r, 0, 0), velocity = (0, v, 0).
        h = r x v = (0, 0, r*v)   → N_hat = (0, 0, 1)
        R_hat = (1, 0, 0)
        T_hat = N x R = (0, 0, 1) x (1, 0, 0) = (0, 1, 0)

        So a force along y is purely along-track.
        """
        f = ConstantForce(0.0, 5.0, 0.0)
        result = compute_force_decomposition(
            [("f", f)], _EPOCH, _POS, _VEL,
        )
        _, radial, along_track, cross_track = result[0]
        assert radial == pytest.approx(0.0, abs=1e-14)
        assert along_track == pytest.approx(5.0)
        assert cross_track == pytest.approx(0.0, abs=1e-14)

    def test_force_decomposition_cross_track(self):
        """A force along the angular-momentum direction is cross-track."""
        f = ConstantForce(0.0, 0.0, 1.0)
        result = compute_force_decomposition(
            [("f", f)], _EPOCH, _POS, _VEL,
        )
        _, radial, along_track, cross_track = result[0]
        assert radial == pytest.approx(0.0, abs=1e-14)
        assert along_track == pytest.approx(0.0, abs=1e-14)
        assert cross_track == pytest.approx(1.0)

    def test_force_decomposition_multiple_models(self):
        """Decomposition works for multiple force models."""
        f1 = ConstantForce(1.0, 0.0, 0.0)
        f2 = ConstantForce(0.0, 2.0, 0.0)
        result = compute_force_decomposition(
            [("radial", f1), ("along", f2)], _EPOCH, _POS, _VEL,
        )
        assert len(result) == 2
        assert result[0][0] == "radial"
        assert result[0][1] == pytest.approx(1.0)
        assert result[1][0] == "along"
        assert result[1][2] == pytest.approx(2.0)
