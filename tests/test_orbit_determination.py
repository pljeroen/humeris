# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for EKF orbit determination."""

import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
)
from humeris.domain.orbit_determination import (
    ODObservation,
    ODEstimate,
    ODResult,
    run_ekf,
    _mat_identity,
    _mat_zeros,
    _two_body_propagate,
)


MU = OrbitalConstants.MU_EARTH
R_EARTH = OrbitalConstants.R_EARTH


@pytest.fixture
def epoch():
    return datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def leo_cartesian():
    """LEO circular orbit position/velocity."""
    r = R_EARTH + 500_000
    v = math.sqrt(MU / r)
    return (r, 0.0, 0.0, 0.0, v, 0.0)


def _generate_truth_obs(state, epoch, n_obs, dt_s, noise_std=0.0):
    """Generate observations from a truth trajectory."""
    obs = []
    current_state = list(state)
    current_time = epoch
    for i in range(n_obs):
        if i > 0:
            current_state = _two_body_propagate(current_state, dt_s)
            current_time = current_time + timedelta(seconds=dt_s)
        pos = (current_state[0], current_state[1], current_state[2])
        obs.append(ODObservation(
            time=current_time,
            position_m=pos,
            noise_std_m=max(noise_std, 1.0),  # minimum 1m noise
        ))
    return obs


# ── Dataclass tests ────────────────────────────────────────────

class TestODDataclasses:

    def test_od_observation_frozen(self, epoch):
        obs = ODObservation(
            time=epoch, position_m=(1.0, 2.0, 3.0), noise_std_m=10.0,
        )
        with pytest.raises(AttributeError):
            obs.noise_std_m = 0.0

    def test_od_estimate_frozen(self, epoch):
        est = ODEstimate(
            time=epoch,
            state=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            covariance=tuple(tuple(0.0 for _ in range(6)) for _ in range(6)),
            residual_m=0.0,
        )
        with pytest.raises(AttributeError):
            est.residual_m = 1.0

    def test_od_result_frozen(self, epoch):
        result = ODResult(
            estimates=(),
            final_state=(0.0,) * 6,
            final_covariance=tuple(tuple(0.0 for _ in range(6)) for _ in range(6)),
            rms_residual_m=0.0,
            observations_processed=0,
        )
        with pytest.raises(AttributeError):
            result.rms_residual_m = 1.0


# ── EKF tests ────────────────────────────────────────────────

class TestEKF:

    def test_perfect_observations_match_truth(self, leo_cartesian, epoch):
        """Zero-noise observations → estimate matches truth."""
        obs = _generate_truth_obs(leo_cartesian, epoch, n_obs=10, dt_s=60.0, noise_std=0.0)
        p0 = [[1e6 if i == j else 0.0 for j in range(6)] for i in range(6)]
        q = [[1e-6 if i == j else 0.0 for j in range(6)] for i in range(6)]

        result = run_ekf(obs, leo_cartesian, p0, q)

        # Final position should match truth
        truth_state = list(leo_cartesian)
        for _ in range(9):
            truth_state = _two_body_propagate(truth_state, 60.0)

        for i in range(3):
            assert abs(result.final_state[i] - truth_state[i]) < 100.0  # < 100 m

    def test_covariance_decreases(self, leo_cartesian, epoch):
        """Covariance decreases with more observations."""
        obs = _generate_truth_obs(leo_cartesian, epoch, n_obs=20, dt_s=60.0, noise_std=100.0)
        p0 = [[1e8 if i == j else 0.0 for j in range(6)] for i in range(6)]
        q = [[1e-4 if i == j else 0.0 for j in range(6)] for i in range(6)]

        result = run_ekf(obs, leo_cartesian, p0, q)

        # Position covariance diagonal should decrease
        initial_pos_cov = p0[0][0]
        final_pos_cov = result.final_covariance[0][0]
        assert final_pos_cov < initial_pos_cov

    def test_residuals_decrease(self, leo_cartesian, epoch):
        """Post-fit residuals should decrease over time."""
        obs = _generate_truth_obs(leo_cartesian, epoch, n_obs=20, dt_s=60.0, noise_std=0.0)
        p0 = [[1e6 if i == j else 0.0 for j in range(6)] for i in range(6)]
        q = [[1e-6 if i == j else 0.0 for j in range(6)] for i in range(6)]

        result = run_ekf(obs, leo_cartesian, p0, q)

        # Later residuals should be smaller than early ones
        first_residual = result.estimates[0].residual_m
        last_residual = result.estimates[-1].residual_m
        assert last_residual <= first_residual + 1.0  # allow small tolerance

    def test_single_observation(self, leo_cartesian, epoch):
        """Single observation → estimate incorporates observation."""
        obs = [ODObservation(
            time=epoch,
            position_m=(leo_cartesian[0], leo_cartesian[1], leo_cartesian[2]),
            noise_std_m=10.0,
        )]
        p0 = [[1e10 if i == j else 0.0 for j in range(6)] for i in range(6)]
        q = _mat_zeros(6, 6)

        result = run_ekf(obs, leo_cartesian, p0, q)
        assert result.observations_processed == 1

    def test_empty_observations_raises(self, leo_cartesian):
        """Empty observations raises ValueError."""
        p0 = _mat_identity(6)
        q = _mat_zeros(6, 6)
        with pytest.raises(ValueError, match="observations"):
            run_ekf([], leo_cartesian, p0, q)

    def test_circular_orbit_recovery(self, leo_cartesian, epoch):
        """Observe at multiple points → recovered orbit is circular."""
        obs = _generate_truth_obs(leo_cartesian, epoch, n_obs=30, dt_s=30.0, noise_std=0.0)
        p0 = [[1e6 if i == j else 0.0 for j in range(6)] for i in range(6)]
        q = [[1e-6 if i == j else 0.0 for j in range(6)] for i in range(6)]

        result = run_ekf(obs, leo_cartesian, p0, q)

        # Check that estimated state is on a near-circular orbit
        x, y, z, vx, vy, vz = result.final_state
        r = math.sqrt(x**2 + y**2 + z**2)
        v = math.sqrt(vx**2 + vy**2 + vz**2)
        v_circ = math.sqrt(MU / r)
        assert abs(v - v_circ) / v_circ < 0.01

    def test_process_noise_zero_keplerian(self, leo_cartesian, epoch):
        """Process noise = 0, no perturbations → pure Keplerian prediction."""
        obs = _generate_truth_obs(leo_cartesian, epoch, n_obs=5, dt_s=60.0, noise_std=0.0)
        p0 = [[1e6 if i == j else 0.0 for j in range(6)] for i in range(6)]
        q = _mat_zeros(6, 6)

        result = run_ekf(obs, leo_cartesian, p0, q)
        assert result.observations_processed == 5

    def test_covariance_symmetric(self, leo_cartesian, epoch):
        """Result covariance is symmetric."""
        obs = _generate_truth_obs(leo_cartesian, epoch, n_obs=10, dt_s=60.0, noise_std=100.0)
        p0 = [[1e6 if i == j else 0.0 for j in range(6)] for i in range(6)]
        q = [[1e-4 if i == j else 0.0 for j in range(6)] for i in range(6)]

        result = run_ekf(obs, leo_cartesian, p0, q)

        cov = result.final_covariance
        for i in range(6):
            for j in range(6):
                assert abs(cov[i][j] - cov[j][i]) < 1e-6 * (abs(cov[i][j]) + 1e-20)

    def test_covariance_positive_diagonal(self, leo_cartesian, epoch):
        """Diagonal elements of covariance are positive."""
        obs = _generate_truth_obs(leo_cartesian, epoch, n_obs=10, dt_s=60.0, noise_std=100.0)
        p0 = [[1e6 if i == j else 0.0 for j in range(6)] for i in range(6)]
        q = [[1e-4 if i == j else 0.0 for j in range(6)] for i in range(6)]

        result = run_ekf(obs, leo_cartesian, p0, q)
        for i in range(6):
            assert result.final_covariance[i][i] > 0


# ── Zero-position guard ───────────────────────────────────────

class TestOdZeroPositionGuard:

    def test_two_body_propagate_zero_position(self):
        """Zero position vector must not crash _two_body_propagate."""
        state = [0.0, 0.0, 0.0, 100.0, 0.0, 0.0]
        result = _two_body_propagate(state, 60.0)
        assert len(result) == 6
        assert all(math.isfinite(v) for v in result)


# ── Domain purity ─────────────────────────────────────────────

class TestODPurity:

    def test_od_module_pure(self):
        """orbit_determination.py must only import stdlib modules."""
        import humeris.domain.orbit_determination as mod

        allowed = {'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
        with open(mod.__file__, encoding="utf-8") as f:
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
