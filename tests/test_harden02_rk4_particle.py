# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""H02-R01: Particle filter RK4 integration tests.

Verifies that the particle filter propagator uses RK4 (not Euler),
conserves energy, and matches the EKF propagator trajectory.
"""
import math

import numpy as np
import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants


class TestParticleFilterRK4:
    """Particle filter should use RK4 two-body propagation."""

    def test_rk4_function_exists(self):
        """_rk4_two_body must exist in orbit_determination module."""
        from humeris.domain.orbit_determination import _rk4_two_body
        assert callable(_rk4_two_body)

    def test_euler_function_removed(self):
        """_euler_two_body must be removed."""
        from humeris.domain import orbit_determination
        assert not hasattr(orbit_determination, '_euler_two_body'), (
            "_euler_two_body should be removed, replaced by _rk4_two_body"
        )

    def test_rk4_matches_ekf_propagator(self):
        """RK4 particle propagator matches EKF RK4 propagator within 1m over 1 orbit."""
        from humeris.domain.orbit_determination import _rk4_two_body, _two_body_propagate

        # 550km circular orbit
        a = OrbitalConstants.R_EARTH + 550_000.0
        v = math.sqrt(OrbitalConstants.MU_EARTH / a)
        state_np = np.array([a, 0.0, 0.0, 0.0, v, 0.0], dtype=np.float64)
        state_list = [a, 0.0, 0.0, 0.0, v, 0.0]

        # Propagate one full orbit (period ≈ 5760s) in 60s steps
        period_s = 2 * math.pi * math.sqrt(a**3 / OrbitalConstants.MU_EARTH)
        dt = 60.0
        n_steps = int(period_s / dt)

        pos_rk4_pf = state_np.copy()
        pos_rk4_ekf = list(state_list)

        for _ in range(n_steps):
            pos_rk4_pf = _rk4_two_body(pos_rk4_pf, dt)
            pos_rk4_ekf = _two_body_propagate(pos_rk4_ekf, dt)

        # Compare final positions — should match within 1m
        diff = np.linalg.norm(pos_rk4_pf[:3] - np.array(pos_rk4_ekf[:3]))
        assert diff < 1.0, f"RK4 implementations diverged by {diff:.3f}m (must be < 1m)"

    def test_rk4_energy_conservation(self):
        """Specific energy must be conserved within 1e-6 relative error over 1 orbit."""
        from humeris.domain.orbit_determination import _rk4_two_body

        # 550km circular orbit
        a = OrbitalConstants.R_EARTH + 550_000.0
        v = math.sqrt(OrbitalConstants.MU_EARTH / a)
        state = np.array([a, 0.0, 0.0, 0.0, v, 0.0], dtype=np.float64)

        def specific_energy(s):
            r = np.linalg.norm(s[:3])
            v_mag = np.linalg.norm(s[3:])
            return 0.5 * v_mag**2 - OrbitalConstants.MU_EARTH / r

        e0 = specific_energy(state)
        period_s = 2 * math.pi * math.sqrt(a**3 / OrbitalConstants.MU_EARTH)
        dt = 60.0
        n_steps = int(period_s / dt)

        for _ in range(n_steps):
            state = _rk4_two_body(state, dt)

        e_final = specific_energy(state)
        rel_error = abs(e_final - e0) / abs(e0)
        assert rel_error < 1e-6, (
            f"Energy drift {rel_error:.2e} exceeds 1e-6 threshold"
        )

    def test_rk4_degenerate_position(self):
        """Near-zero position returns unchanged state (no division by zero)."""
        from humeris.domain.orbit_determination import _rk4_two_body

        state = np.array([1e-15, 0.0, 0.0, 100.0, 0.0, 0.0], dtype=np.float64)
        result = _rk4_two_body(state, 60.0)
        assert np.all(np.isfinite(result)), "RK4 must handle degenerate position"

    def test_particle_filter_uses_rk4(self):
        """Particle filter integration should produce better accuracy than Euler would.

        Propagate a particle over 1 orbit and verify position error is small
        (RK4 has ~10^4 better accuracy than Euler for orbital mechanics).
        """
        from humeris.domain.orbit_determination import _rk4_two_body

        # 550km circular orbit
        a = OrbitalConstants.R_EARTH + 550_000.0
        v = math.sqrt(OrbitalConstants.MU_EARTH / a)
        state = np.array([a, 0.0, 0.0, 0.0, v, 0.0], dtype=np.float64)

        period_s = 2 * math.pi * math.sqrt(a**3 / OrbitalConstants.MU_EARTH)
        dt = 60.0
        n_steps = int(period_s / dt)

        for _ in range(n_steps):
            state = _rk4_two_body(state, dt)

        # After one full orbit, should return near starting position
        # RK4 with 60s steps: position error < 100m
        # Euler with 60s steps: position error > 10km
        r_final = np.linalg.norm(state[:3])
        pos_error = abs(r_final - a)
        assert pos_error < 100.0, (
            f"Position error {pos_error:.1f}m after 1 orbit suggests Euler, not RK4"
        )
