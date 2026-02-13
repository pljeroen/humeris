# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Invariant tests for numerical propagation.

Verifies conservation laws (energy, angular momentum) and symmetry
properties (reversibility, Kepler period) for all integrators.

Invariants P1-P5 from the formal invariant specification.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator import (
    OrbitalConstants,
    ShellConfig,
    generate_walker_shell,
    derive_orbital_state,
)
from constellation_generator.domain.numerical_propagation import (
    TwoBodyGravity,
    propagate_numerical,
)


MU = OrbitalConstants.MU_EARTH


@pytest.fixture
def epoch():
    return datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def leo_state(epoch):
    shell = ShellConfig(
        altitude_km=500, inclination_deg=53, num_planes=1,
        sats_per_plane=1, phase_factor=0, raan_offset_deg=0,
        shell_name="InvTest",
    )
    sat = generate_walker_shell(shell)[0]
    return derive_orbital_state(sat, epoch)


def _energy(pos, vel):
    """Specific orbital energy: E = v^2/2 - mu/r."""
    r = math.sqrt(sum(p**2 for p in pos))
    v = math.sqrt(sum(vi**2 for vi in vel))
    return 0.5 * v**2 - MU / r


def _angular_momentum_mag(pos, vel):
    """Magnitude of angular momentum: |h| = |r x v|."""
    hx = pos[1] * vel[2] - pos[2] * vel[1]
    hy = pos[2] * vel[0] - pos[0] * vel[2]
    hz = pos[0] * vel[1] - pos[1] * vel[0]
    return math.sqrt(hx**2 + hy**2 + hz**2)


_INTEGRATORS = ["rk4", "verlet", "yoshida"]


# ── P1: Energy Conservation (Two-Body) ─────────────────────────

class TestP1EnergyConservation:
    """Energy must be conserved under two-body gravity for all integrators."""

    @pytest.mark.parametrize("integrator", _INTEGRATORS)
    def test_energy_conservation_10_orbits(self, leo_state, epoch, integrator):
        period_s = 2 * math.pi / leo_state.mean_motion_rad_s
        result = propagate_numerical(
            leo_state, timedelta(seconds=10 * period_s), timedelta(seconds=10),
            [TwoBodyGravity()], epoch=epoch, integrator=integrator,
        )

        e0 = _energy(result.steps[0].position_eci, result.steps[0].velocity_eci)
        max_drift = max(
            abs(_energy(s.position_eci, s.velocity_eci) - e0) / abs(e0)
            for s in result.steps
        )

        # All integrators should conserve energy well with 10s step
        if integrator == "yoshida":
            assert max_drift < 1e-9, f"Yoshida energy drift {max_drift}"
        else:
            assert max_drift < 1e-7, f"{integrator} energy drift {max_drift}"


# ── P2: Angular Momentum Conservation (Two-Body) ───────────────

class TestP2AngularMomentumConservation:
    """Angular momentum must be conserved under central-force dynamics."""

    @pytest.mark.parametrize("integrator", _INTEGRATORS)
    def test_angular_momentum_10_orbits(self, leo_state, epoch, integrator):
        period_s = 2 * math.pi / leo_state.mean_motion_rad_s
        result = propagate_numerical(
            leo_state, timedelta(seconds=10 * period_s), timedelta(seconds=10),
            [TwoBodyGravity()], epoch=epoch, integrator=integrator,
        )

        h0 = _angular_momentum_mag(
            result.steps[0].position_eci, result.steps[0].velocity_eci,
        )
        max_drift = max(
            abs(_angular_momentum_mag(s.position_eci, s.velocity_eci) - h0) / h0
            for s in result.steps
        )

        assert max_drift < 1e-9, f"{integrator} h drift {max_drift}"


# ── P3: Reversibility ──────────────────────────────────────────

class TestP3SymplecticBoundedness:
    """Symplectic integrators bound energy oscillation (no secular drift)."""

    @pytest.mark.parametrize("integrator", ["verlet", "yoshida"])
    def test_energy_bounded_no_drift(self, leo_state, epoch, integrator):
        """Energy oscillates but doesn't drift for symplectic integrators."""
        period_s = 2 * math.pi / leo_state.mean_motion_rad_s
        result = propagate_numerical(
            leo_state, timedelta(seconds=10 * period_s), timedelta(seconds=10),
            [TwoBodyGravity()], epoch=epoch, integrator=integrator,
        )

        energies = [
            _energy(s.position_eci, s.velocity_eci) for s in result.steps
        ]
        e0 = energies[0]

        # Split into first and second half — drift means second half differs
        mid = len(energies) // 2
        mean_first = sum(energies[:mid]) / mid
        mean_second = sum(energies[mid:]) / (len(energies) - mid)

        # No secular drift: mean energy in both halves should be nearly equal
        assert abs(mean_second - mean_first) / abs(e0) < 1e-9


# ── P4: Kepler Period ──────────────────────────────────────────

class TestP4KeplerPeriod:
    """After one orbital period, position returns to start."""

    @pytest.mark.parametrize("integrator", _INTEGRATORS)
    def test_kepler_period_return(self, leo_state, epoch, integrator):
        period_s = 2 * math.pi / leo_state.mean_motion_rad_s
        result = propagate_numerical(
            leo_state, timedelta(seconds=period_s), timedelta(seconds=10),
            [TwoBodyGravity()], epoch=epoch, integrator=integrator,
        )

        r0 = result.steps[0].position_eci
        rf = result.steps[-1].position_eci
        mag_r0 = math.sqrt(sum(p**2 for p in r0))
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(r0, rf)))

        # Step-size dependent: 10s step gives ~1% position error over one orbit
        assert dist / mag_r0 < 1e-2, f"{integrator} period return error {dist / mag_r0}"


# ── P5: Integrator Agreement ──────────────────────────────────

class TestP5IntegratorAgreement:
    """All integrators agree on final position within tolerance."""

    def test_agreement_one_orbit(self, leo_state, epoch):
        period_s = 2 * math.pi / leo_state.mean_motion_rad_s
        results = {}
        for integ in _INTEGRATORS:
            r = propagate_numerical(
                leo_state, timedelta(seconds=period_s), timedelta(seconds=10),
                [TwoBodyGravity()], epoch=epoch, integrator=integ,
            )
            results[integ] = r.steps[-1].position_eci

        # Compare all pairs
        mag = math.sqrt(sum(p**2 for p in results["rk4"]))
        for a_name, b_name in [("rk4", "verlet"), ("rk4", "yoshida"), ("verlet", "yoshida")]:
            dist = math.sqrt(sum(
                (results[a_name][i] - results[b_name][i])**2
                for i in range(3)
            ))
            assert dist / mag < 1e-2, f"{a_name} vs {b_name}: {dist / mag}"
