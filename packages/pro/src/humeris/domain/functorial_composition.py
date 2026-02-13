# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Functorial force model composition.

Category-theoretic composition of force models with associativity
verification, natural transformations between reference frames,
and commutative diagram validation.

Force models form a category where objects are phase-space states and
morphisms are force model compositions.  This enables:

1. Functorial composition with guaranteed mathematical properties
   (associativity, identity).
2. Natural transformations between force model representations
   (e.g., body-fixed <-> inertial).
3. Commutative diagram validation for order-independent compositions.
4. Pullback force models derived from coordinate frame changes.

"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from humeris.domain.numerical_propagation import ForceModel


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForceCategory:
    """A category of force models with composition.

    Attributes:
        force_models: Named force models as tuple of (name, ForceModel).
        composition_order: Order of application (tuple of names).
        is_commutative: Whether composition order matters for these models.
    """
    force_models: tuple          # tuple of (str, ForceModel)
    composition_order: tuple     # tuple of str
    is_commutative: bool


@dataclass(frozen=True)
class CompositionResult:
    """Result of composing force models.

    Attributes:
        total_acceleration: Summed (ax, ay, az) in m/s^2.
        per_model_accelerations: tuple of (name, (ax, ay, az)).
        composition_residual: ||f(g(x)) - g(f(x))|| commutativity check.
        is_order_independent: True if composition commutes within tolerance.
    """
    total_acceleration: tuple          # (float, float, float)
    per_model_accelerations: tuple     # tuple of (str, (float, float, float))
    composition_residual: float
    is_order_independent: bool


@dataclass(frozen=True)
class NaturalTransformation:
    """A natural transformation between force model representations.

    Attributes:
        source_frame: Source reference frame name (e.g. "ECI").
        target_frame: Target reference frame name (e.g. "ECEF").
        transform_matrix: Flattened 3x3 rotation matrix (9 floats, row-major).
        is_invertible: Whether the transformation is invertible.
    """
    source_frame: str
    target_frame: str
    transform_matrix: tuple      # 9 floats, row-major 3x3
    is_invertible: bool


@dataclass(frozen=True)
class PullbackForce:
    """Force model in a new frame via pullback along coordinate change.

    Attributes:
        original_frame: Frame in which the force was originally defined.
        target_frame: Frame to which the force has been pulled back.
        acceleration_in_target: (ax, ay, az) in target frame, m/s^2.
        jacobian_correction: Magnitude of Jacobian correction term.
    """
    original_frame: str
    target_frame: str
    acceleration_in_target: tuple   # (float, float, float)
    jacobian_correction: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_matrix(flat: tuple) -> np.ndarray:
    """Unflatten a 9-element tuple into a 3x3 NumPy array."""
    return np.array(flat, dtype=np.float64).reshape(3, 3)


def _flatten_matrix(m: np.ndarray) -> tuple:
    """Flatten a 3x3 NumPy array into a tuple of 9 floats."""
    return tuple(float(x) for x in m.ravel())


def _vec(t: tuple) -> np.ndarray:
    """Tuple of 3 floats -> NumPy 1-D array."""
    return np.array(t, dtype=np.float64)


def _tup(a: np.ndarray) -> tuple:
    """NumPy 1-D array -> tuple of Python floats."""
    return (float(a[0]), float(a[1]), float(a[2]))


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def identity_force() -> object:
    """Return the identity morphism (zero-acceleration force model).

    The returned object satisfies the ForceModel protocol: its
    ``acceleration`` method returns (0.0, 0.0, 0.0) for any input.
    """

    class _IdentityForce:
        def acceleration(
            self,
            epoch: datetime,
            position: tuple[float, float, float],
            velocity: tuple[float, float, float],
        ) -> tuple[float, float, float]:
            return (0.0, 0.0, 0.0)

    return _IdentityForce()


def compose_forces(
    force_models: list,
    epoch: datetime,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    tolerance: float = 1e-10,
) -> CompositionResult:
    """Compose (superpose) named force models and check commutativity.

    Parameters:
        force_models: list of (name: str, model: ForceModel).
        epoch: Evaluation epoch.
        position: (x, y, z) in metres.
        velocity: (vx, vy, vz) in m/s.
        tolerance: Threshold below which composition is order-independent.

    Returns:
        CompositionResult with summed acceleration, per-model breakdown,
        commutativity residual, and order-independence flag.
    """
    per_model: list[tuple] = []
    total = np.zeros(3, dtype=np.float64)

    for name, model in force_models:
        acc = model.acceleration(epoch, position, velocity)
        a_vec = _vec(acc)
        total += a_vec
        per_model.append((name, _tup(a_vec)))

    # Commutativity check: compare forward and reverse summation order.
    # For linear (additive) force superposition the residual is always zero
    # up to floating-point noise.  Non-linear composed transforms may differ.
    total_reverse = np.zeros(3, dtype=np.float64)
    for name, model in reversed(force_models):
        acc = model.acceleration(epoch, position, velocity)
        total_reverse += _vec(acc)

    residual = float(np.linalg.norm(total - total_reverse))
    is_order_independent = residual < tolerance

    return CompositionResult(
        total_acceleration=_tup(total),
        per_model_accelerations=tuple(per_model),
        composition_residual=residual,
        is_order_independent=is_order_independent,
    )


def verify_associativity(
    forces_a: list,
    forces_b: list,
    forces_c: list,
    epoch: datetime,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    tolerance: float = 1e-12,
) -> bool:
    """Verify (a + b) + c == a + (b + c) for force superposition.

    Each argument is a list of (name, ForceModel).  Returns True when the
    two groupings produce the same total acceleration within *tolerance*.
    """
    # (a + b) + c
    ab = compose_forces(forces_a + forces_b, epoch, position, velocity)
    ab_c = compose_forces(
        [("ab", _AccelerationLiteral(ab.total_acceleration))] + forces_c,
        epoch, position, velocity,
    )

    # a + (b + c)
    bc = compose_forces(forces_b + forces_c, epoch, position, velocity)
    a_bc = compose_forces(
        forces_a + [("bc", _AccelerationLiteral(bc.total_acceleration))],
        epoch, position, velocity,
    )

    diff = np.linalg.norm(_vec(ab_c.total_acceleration) - _vec(a_bc.total_acceleration))
    return float(diff) < tolerance


class _AccelerationLiteral:
    """Trivial force model that returns a fixed acceleration vector."""

    def __init__(self, acc: tuple):
        self._acc = acc

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return self._acc


def natural_transform_force(
    force_model: ForceModel,
    transform: NaturalTransformation,
    epoch: datetime,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
) -> PullbackForce:
    """Apply a force model in the source frame, then rotate to the target frame.

    The acceleration is computed in the source frame and mapped to the target
    via *transform.transform_matrix* (a 3x3 rotation, row-major flattened).

    Parameters:
        force_model: Force model producing acceleration in the source frame.
        transform: Natural transformation specifying rotation and frame names.
        epoch: Evaluation epoch.
        position: Position in the source frame (m).
        velocity: Velocity in the source frame (m/s).

    Returns:
        PullbackForce with acceleration expressed in the target frame.
    """
    acc_source = _vec(force_model.acceleration(epoch, position, velocity))
    r_mat = _to_matrix(transform.transform_matrix)
    acc_target = r_mat @ acc_source

    # Jacobian correction magnitude: ||R @ a - a|| captures the rotational
    # mapping effect.  For an identity rotation this is zero.
    jacobian_correction = float(np.linalg.norm(acc_target - acc_source))

    return PullbackForce(
        original_frame=transform.source_frame,
        target_frame=transform.target_frame,
        acceleration_in_target=_tup(acc_target),
        jacobian_correction=jacobian_correction,
    )


def pullback_force(
    force_model: ForceModel,
    rotation_matrix: tuple,
    epoch: datetime,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    source_frame: str = "original",
    target_frame: str = "rotated",
) -> PullbackForce:
    """Compute a force in a rotated frame via pullback.

    Evaluates: a' = R @ force(R^T @ pos, R^T @ vel)

    The Jacobian correction accounts for the fact that the force model is
    evaluated at a rotated state, introducing a coordinate-dependent term.

    Parameters:
        force_model: Force model to pull back.
        rotation_matrix: Flattened 3x3 rotation matrix (9 floats, row-major).
        epoch: Evaluation epoch.
        position: Position in the target frame (m).
        velocity: Velocity in the target frame (m/s).
        source_frame: Name of the original (source) frame.
        target_frame: Name of the rotated (target) frame.

    Returns:
        PullbackForce with acceleration in the target frame and Jacobian
        correction magnitude.
    """
    r_mat = _to_matrix(rotation_matrix)
    r_t = r_mat.T  # R^T: target -> source

    pos_source = _tup(r_t @ _vec(position))
    vel_source = _tup(r_t @ _vec(velocity))

    acc_source = _vec(force_model.acceleration(epoch, pos_source, vel_source))
    acc_target = r_mat @ acc_source

    # Jacobian correction: difference between naive rotation of the original-
    # frame evaluation and the full pullback (which rotates the inputs first).
    acc_naive = _vec(force_model.acceleration(epoch, position, velocity))
    acc_naive_rotated = r_mat @ acc_naive
    jacobian_correction = float(np.linalg.norm(acc_target - acc_naive_rotated))

    return PullbackForce(
        original_frame=source_frame,
        target_frame=target_frame,
        acceleration_in_target=_tup(acc_target),
        jacobian_correction=jacobian_correction,
    )


def check_commutativity_diagram(
    forces: list,
    transforms: list,
    epoch: datetime,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    tolerance: float = 1e-10,
) -> tuple:
    """Verify that a diagram of forces and frame transforms commutes.

    For each pair of forces, check whether applying both forces and then
    transforming gives the same result as transforming and then applying.

    Parameters:
        forces: list of (name: str, model: ForceModel).
        transforms: list of NaturalTransformation.
        epoch: Evaluation epoch.
        position: Position in the source frame (m).
        velocity: Velocity in the source frame (m/s).
        tolerance: Commutativity tolerance.

    Returns:
        (commutes: bool, max_residual: float, per_pair_residuals: tuple)
        where per_pair_residuals is a tuple of (name_i, name_j, residual).
    """
    per_pair: list[tuple] = []
    max_residual = 0.0

    for i, (name_i, force_i) in enumerate(forces):
        for j, (name_j, force_j) in enumerate(forces):
            if j <= i:
                continue

            # Path 1: sum forces, then transform
            acc_i = _vec(force_i.acceleration(epoch, position, velocity))
            acc_j = _vec(force_j.acceleration(epoch, position, velocity))
            sum_then_transform = acc_i + acc_j

            for t in transforms:
                r_mat = _to_matrix(t.transform_matrix)
                sum_then_transform = r_mat @ sum_then_transform

            # Path 2: transform each force, then sum
            transform_then_sum = np.zeros(3, dtype=np.float64)
            for acc_raw in (acc_i, acc_j):
                rotated = acc_raw.copy()
                for t in transforms:
                    r_mat = _to_matrix(t.transform_matrix)
                    rotated = r_mat @ rotated
                transform_then_sum += rotated

            residual = float(np.linalg.norm(sum_then_transform - transform_then_sum))
            max_residual = max(max_residual, residual)
            per_pair.append((name_i, name_j, residual))

    commutes = max_residual < tolerance
    return (commutes, max_residual, tuple(per_pair))


def compute_force_decomposition(
    force_models: list,
    epoch: datetime,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
) -> tuple:
    """Decompose each force model's acceleration into RTN components.

    RTN (Radial, Along-track, Cross-track) is constructed from the
    position and velocity vectors:

    - R (radial): unit position vector
    - N (cross-track): unit(position x velocity)
    - T (along-track): N x R (completes the right-handed triad)

    Parameters:
        force_models: list of (name: str, model: ForceModel).
        epoch: Evaluation epoch.
        position: (x, y, z) in metres.
        velocity: (vx, vy, vz) in m/s.

    Returns:
        Tuple of (name, radial, along_track, cross_track) for each force.
    """
    r_vec = _vec(position)
    v_vec = _vec(velocity)

    r_mag = float(np.linalg.norm(r_vec))
    if r_mag < 1e-30:
        # Degenerate: position at origin, cannot define RTN frame.
        return tuple(
            (name, 0.0, 0.0, 0.0) for name, _ in force_models
        )

    r_hat = r_vec / r_mag

    h_vec = np.cross(r_vec, v_vec)
    h_mag = float(np.linalg.norm(h_vec))
    if h_mag < 1e-30:
        # Degenerate: rectilinear orbit, cross-track undefined.
        # Fall back to radial-only decomposition.
        results = []
        for name, model in force_models:
            acc = _vec(model.acceleration(epoch, position, velocity))
            radial = float(np.dot(acc, r_hat))
            results.append((name, radial, 0.0, 0.0))
        return tuple(results)

    n_hat = h_vec / h_mag        # cross-track
    t_hat = np.cross(n_hat, r_hat)  # along-track

    results = []
    for name, model in force_models:
        acc = _vec(model.acceleration(epoch, position, velocity))
        radial = float(np.dot(acc, r_hat))
        along_track = float(np.dot(acc, t_hat))
        cross_track = float(np.dot(acc, n_hat))
        results.append((name, radial, along_track, cross_track))

    return tuple(results)
