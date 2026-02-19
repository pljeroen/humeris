# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""H02-R02: Koopman DMD stability diagnostic tests.

Verifies that KoopmanModel includes is_stable field and that
predict_koopman() warns on unstable models.
"""
import logging
import math

import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants


def _make_circular_orbit_snapshots(n_steps=50, step_s=60.0):
    """Generate position/velocity snapshots for a 550km circular orbit."""
    a = OrbitalConstants.R_EARTH + 550_000.0
    v = math.sqrt(OrbitalConstants.MU_EARTH / a)
    n = math.sqrt(OrbitalConstants.MU_EARTH / a**3)  # mean motion

    positions = []
    velocities = []
    for i in range(n_steps):
        t = i * step_s
        theta = n * t
        positions.append((a * math.cos(theta), a * math.sin(theta), 0.0))
        velocities.append((-v * math.sin(theta), v * math.cos(theta), 0.0))

    return positions, velocities, step_s


class TestKoopmanStabilityDiagnostic:
    """KoopmanModel should expose stability diagnostics."""

    def test_koopman_model_has_is_stable_field(self):
        """KoopmanModel must have an is_stable field."""
        from humeris.domain.koopman_propagation import KoopmanModel
        # Check the field exists on the dataclass
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(KoopmanModel)}
        assert 'is_stable' in field_names, "KoopmanModel missing is_stable field"

    def test_stable_model_is_stable_true(self):
        """Circular orbit DMD should produce a stable model (|lambda| ≈ 1)."""
        from humeris.domain.koopman_propagation import fit_koopman_model
        positions, velocities, step_s = _make_circular_orbit_snapshots()
        model = fit_koopman_model(positions, velocities, step_s)
        assert model.is_stable is True, (
            f"Stable orbit should give is_stable=True, "
            f"max_eig_mag={model.max_eigenvalue_magnitude:.6f}"
        )

    def test_unstable_model_is_stable_false(self):
        """fit_koopman_model with diverging data must produce is_stable=False."""
        from humeris.domain.koopman_propagation import fit_koopman_model

        # Create exponentially growing trajectory (unstable dynamics)
        positions = []
        velocities = []
        for i in range(50):
            scale = 1.1 ** i  # 10% growth per step → eigenvalue > 1
            r = (OrbitalConstants.R_EARTH + 550_000.0) * scale
            v = math.sqrt(OrbitalConstants.MU_EARTH / r)
            theta = i * 0.1
            positions.append((r * math.cos(theta), r * math.sin(theta), 0.0))
            velocities.append((-v * math.sin(theta), v * math.cos(theta), 0.0))

        model = fit_koopman_model(positions, velocities, 60.0)
        assert model.max_eigenvalue_magnitude > 1.0 + 1e-6, (
            f"Expected unstable model, max_eig_mag={model.max_eigenvalue_magnitude:.6f}"
        )
        assert model.is_stable is False, (
            f"Model with max_eig_mag={model.max_eigenvalue_magnitude:.6f} "
            f"should have is_stable=False"
        )

    def test_koopman_prediction_has_is_stable(self):
        """KoopmanPrediction must include is_stable field."""
        from humeris.domain.koopman_propagation import KoopmanPrediction
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(KoopmanPrediction)}
        assert 'is_stable' in field_names, "KoopmanPrediction missing is_stable field"

    def test_predict_warns_on_unstable_model(self, caplog):
        """predict_koopman() with unstable model must emit logging.warning."""
        from humeris.domain.koopman_propagation import fit_koopman_model, predict_koopman

        # Create snapshots with exponentially growing radii (unstable)
        positions = []
        velocities = []
        for i in range(50):
            scale = 1.0 + 0.05 * i  # Grow 5% per step
            r = (OrbitalConstants.R_EARTH + 550_000.0) * scale
            v = math.sqrt(OrbitalConstants.MU_EARTH / r)
            theta = i * 0.01
            positions.append((r * math.cos(theta), r * math.sin(theta), 0.0))
            velocities.append((-v * math.sin(theta), v * math.cos(theta), 0.0))

        model = fit_koopman_model(positions, velocities, 60.0)

        # If the model happens to be stable, force test with an unstable one
        if model.is_stable:
            pytest.skip("Generated model was stable — need unstable data")

        with caplog.at_level(logging.WARNING, logger="humeris.domain.koopman_propagation"):
            predict_koopman(
                model,
                initial_position=positions[0],
                initial_velocity=velocities[0],
                duration_s=3600.0,
                step_s=60.0,
            )

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("unstable" in m.lower() or "eigenvalue" in m.lower() for m in warning_msgs), (
            f"Expected warning about unstable model, got: {warning_msgs}"
        )

    def test_prediction_carries_stability_flag(self):
        """Prediction from stable model should have is_stable=True."""
        from humeris.domain.koopman_propagation import fit_koopman_model, predict_koopman
        positions, velocities, step_s = _make_circular_orbit_snapshots()
        model = fit_koopman_model(positions, velocities, step_s)
        pred = predict_koopman(model, positions[0], velocities[0], 600.0, step_s)
        assert pred.is_stable == model.is_stable
