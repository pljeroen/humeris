# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/koopman_propagation.py — Koopman operator fast propagation."""
import ast
import math
from datetime import datetime, timezone, timedelta

import numpy as np

from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.koopman_propagation import (
    KoopmanModel,
    KoopmanPrediction,
    fit_koopman_model,
    predict_koopman,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _circular_orbit_state(alt_km=550.0, inc_deg=53.0, raan_deg=0.0):
    """Create a circular orbit state for testing."""
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(inc_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=0.0, true_anomaly_rad=0.0,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


def _generate_training_data(state, num_steps=200, step_s=30.0):
    """Generate position/velocity snapshots from two-body propagation."""
    positions = []
    velocities = []
    for i in range(num_steps):
        t = _EPOCH + timedelta(seconds=i * step_s)
        pos, vel = propagate_to(state, t)
        positions.append((pos[0], pos[1], pos[2]))
        velocities.append((vel[0], vel[1], vel[2]))
    return positions, velocities


class TestFitKoopmanModel:
    def test_returns_koopman_model_type(self):
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0)
        assert isinstance(model, KoopmanModel)

    def test_low_training_error_circular_orbit(self):
        """Circular orbit should be well-captured by Koopman: training error < 1%."""
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state, num_steps=200, step_s=30.0)
        model = fit_koopman_model(positions, velocities, step_s=30.0)
        assert model.training_error < 0.01

    def test_singular_values_positive_and_decreasing(self):
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0)
        svs = model.singular_values
        assert len(svs) > 0
        assert all(s > 0 for s in svs), f"Non-positive singular values: {svs}"
        for i in range(len(svs) - 1):
            assert svs[i] >= svs[i + 1] - 1e-10, (
                f"Singular values not decreasing at index {i}: {svs[i]} < {svs[i+1]}"
            )

    def test_koopman_matrix_shape(self):
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0, n_observables=12)
        assert model.n_observables == 12
        assert len(model.koopman_matrix) == 12 * 12

    def test_mean_state_length(self):
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0, n_observables=12)
        assert len(model.mean_state) == 12

    def test_training_step_stored(self):
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=42.0)
        assert model.training_step_s == 42.0

    def test_minimum_6_observables(self):
        """With n_observables=6, model should still work (state-only, no lifted observables)."""
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0, n_observables=6)
        assert model.n_observables == 6
        assert len(model.koopman_matrix) == 6 * 6

    def test_12_observables_lower_error_than_6(self):
        """More observables should generally give equal or better fit."""
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state, num_steps=200, step_s=30.0)
        model_6 = fit_koopman_model(positions, velocities, step_s=30.0, n_observables=6)
        model_12 = fit_koopman_model(positions, velocities, step_s=30.0, n_observables=12)
        # 12 observables has at least as many degrees of freedom
        assert model_12.training_error <= model_6.training_error + 1e-6


class TestPredictKoopman:
    def test_returns_prediction_type(self):
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0)
        pred = predict_koopman(
            model, positions[0], velocities[0],
            duration_s=3000.0, step_s=30.0,
        )
        assert isinstance(pred, KoopmanPrediction)

    def test_correct_number_of_steps(self):
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0)
        pred = predict_koopman(
            model, positions[0], velocities[0],
            duration_s=300.0, step_s=30.0,
        )
        expected_steps = int(300.0 / 30.0) + 1  # includes t=0
        assert len(pred.times_s) == expected_steps
        assert len(pred.positions_eci) == expected_steps
        assert len(pred.velocities_eci) == expected_steps

    def test_prediction_matches_twobody_within_1km_one_period(self):
        """Koopman prediction should match two-body propagation within 1 km
        over one orbital period for a circular orbit."""
        state = _circular_orbit_state(alt_km=550.0)
        a = _R_E + 550_000.0
        period_s = 2 * math.pi * math.sqrt(a ** 3 / _MU)
        step_s = 30.0
        # Generate training data for 2 periods
        num_train = int(2 * period_s / step_s) + 1
        positions, velocities = _generate_training_data(
            state, num_steps=num_train, step_s=step_s,
        )
        model = fit_koopman_model(positions, velocities, step_s=step_s)

        # Predict for 1 period
        pred = predict_koopman(
            model, positions[0], velocities[0],
            duration_s=period_s, step_s=step_s,
        )

        # Compare against direct propagation
        max_error_m = 0.0
        for i, t_s in enumerate(pred.times_s):
            t = _EPOCH + timedelta(seconds=t_s)
            ref_pos, _ = propagate_to(state, t)
            pred_pos = pred.positions_eci[i]
            dx = pred_pos[0] - ref_pos[0]
            dy = pred_pos[1] - ref_pos[1]
            dz = pred_pos[2] - ref_pos[2]
            err = math.sqrt(dx**2 + dy**2 + dz**2)
            if err > max_error_m:
                max_error_m = err

        assert max_error_m < 1000.0, (
            f"Max position error {max_error_m:.1f} m exceeds 1 km threshold"
        )

    def test_initial_position_matches(self):
        """First prediction step should match the initial conditions."""
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0)
        pred = predict_koopman(
            model, positions[0], velocities[0],
            duration_s=300.0, step_s=30.0,
        )
        # First time should be 0
        assert pred.times_s[0] == 0.0
        # First position should match initial
        p0 = pred.positions_eci[0]
        for j in range(3):
            assert abs(p0[j] - positions[0][j]) < 1.0, (
                f"Initial position mismatch: component {j}"
            )

    def test_model_stored_in_prediction(self):
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0)
        pred = predict_koopman(
            model, positions[0], velocities[0],
            duration_s=300.0, step_s=30.0,
        )
        assert pred.model is model

    def test_prediction_with_6_observables(self):
        """Prediction should work with minimum observables."""
        state = _circular_orbit_state()
        positions, velocities = _generate_training_data(state)
        model = fit_koopman_model(positions, velocities, step_s=30.0, n_observables=6)
        pred = predict_koopman(
            model, positions[0], velocities[0],
            duration_s=300.0, step_s=30.0,
        )
        assert len(pred.positions_eci) == int(300.0 / 30.0) + 1


class TestKoopmanPropagationPurity:
    def test_module_pure(self):
        import humeris.domain.koopman_propagation as mod
        source = ast.parse(open(mod.__file__).read())
        for node in ast.walk(source):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                else:
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                assert top in {"math", "dataclasses", "datetime", "logging", "typing", "enum", "numpy", "humeris", "__future__"}, (
                    f"Forbidden import: {top}"
                )
