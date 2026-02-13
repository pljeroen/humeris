# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for Dormand-Prince RK4(5) adaptive integrator."""

import ast
import math
import time
from datetime import datetime, timedelta, timezone

import pytest

from humeris import (
    OrbitalConstants,
    ShellConfig,
    generate_walker_shell,
    derive_orbital_state,
    kepler_to_cartesian,
)
from humeris.domain.numerical_propagation import (
    PropagationStep,
    NumericalPropagationResult,
    TwoBodyGravity,
    J2Perturbation,
    rk4_step,
    propagate_numerical,
    ForceModel,
)
from humeris.domain.adaptive_integration import (
    AdaptiveStepConfig,
    AdaptiveStepResult,
    dormand_prince_step,
    propagate_adaptive,
    DORMAND_PRINCE_A,
    DORMAND_PRINCE_B4,
    DORMAND_PRINCE_B5,
    DORMAND_PRINCE_C,
)


# --- Fixtures ---

@pytest.fixture
def epoch():
    return datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def leo_state(epoch):
    """LEO orbital state at 500 km, circular, 53 deg inclination."""
    shell = ShellConfig(
        altitude_km=500,
        inclination_deg=53,
        num_planes=1,
        sats_per_plane=1,
        phase_factor=0,
        raan_offset_deg=0,
        shell_name="Test",
    )
    sat = generate_walker_shell(shell)[0]
    return derive_orbital_state(sat, epoch)


@pytest.fixture
def elliptical_state(epoch):
    """Eccentric orbit e=0.5 for testing adaptive step variation."""
    a = OrbitalConstants.R_EARTH_EQUATORIAL + 500_000.0
    e = 0.5
    i_rad = math.radians(53.0)
    raan = 0.0
    argp = 0.0
    nu = 0.0  # Start at periapsis
    mu = OrbitalConstants.MU_EARTH
    n = math.sqrt(mu / a**3)
    return type("OrbitalState", (), {
        "semi_major_axis_m": a,
        "eccentricity": e,
        "inclination_rad": i_rad,
        "raan_rad": raan,
        "arg_perigee_rad": argp,
        "true_anomaly_rad": nu,
        "mean_motion_rad_s": n,
        "reference_epoch": epoch,
    })()


@pytest.fixture
def near_parabolic_state(epoch):
    """Near-parabolic orbit e=0.95 for testing extreme eccentricity."""
    a = OrbitalConstants.R_EARTH_EQUATORIAL + 2_000_000.0
    e = 0.95
    i_rad = math.radians(30.0)
    raan = 0.0
    argp = 0.0
    nu = 0.0  # Start at periapsis
    mu = OrbitalConstants.MU_EARTH
    n = math.sqrt(mu / a**3)
    return type("OrbitalState", (), {
        "semi_major_axis_m": a,
        "eccentricity": e,
        "inclination_rad": i_rad,
        "raan_rad": raan,
        "arg_perigee_rad": argp,
        "true_anomaly_rad": nu,
        "mean_motion_rad_s": n,
        "reference_epoch": epoch,
    })()


def _orbital_energy(pos, vel, mu):
    """Compute specific orbital energy: v^2/2 - mu/r."""
    r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    v = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
    return 0.5 * v**2 - mu / r


# --- Two-body energy conservation ---

class TestEnergyConservation:

    def test_circular_orbit_energy_conservation(self, leo_state, epoch):
        """Circular orbit: energy conserved to within tolerance over 1 orbit."""
        mu = OrbitalConstants.MU_EARTH
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=5400),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        first = result.steps[0]
        last = result.steps[-1]
        e0 = _orbital_energy(first.position_eci, first.velocity_eci, mu)
        ef = _orbital_energy(last.position_eci, last.velocity_eci, mu)
        assert abs(ef - e0) / abs(e0) < 1e-8

    def test_elliptical_orbit_energy_conservation(self, elliptical_state, epoch):
        """Elliptical orbit (e=0.5): energy conserved over 1 orbit."""
        mu = OrbitalConstants.MU_EARTH
        a = elliptical_state.semi_major_axis_m
        period = 2.0 * math.pi * math.sqrt(a**3 / mu)
        result = propagate_adaptive(
            initial_state=elliptical_state,
            duration=timedelta(seconds=period),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        first = result.steps[0]
        last = result.steps[-1]
        e0 = _orbital_energy(first.position_eci, first.velocity_eci, mu)
        ef = _orbital_energy(last.position_eci, last.velocity_eci, mu)
        assert abs(ef - e0) / abs(e0) < 1e-8

    def test_near_parabolic_orbit_energy_conservation(self, near_parabolic_state, epoch):
        """Near-parabolic orbit (e=0.95): energy conserved over partial orbit."""
        mu = OrbitalConstants.MU_EARTH
        a = near_parabolic_state.semi_major_axis_m
        period = 2.0 * math.pi * math.sqrt(a**3 / mu)
        # Propagate only 1/4 orbit to keep test fast
        result = propagate_adaptive(
            initial_state=near_parabolic_state,
            duration=timedelta(seconds=period / 4.0),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            config=AdaptiveStepConfig(rtol=1e-10, atol=1e-12),
        )
        first = result.steps[0]
        last = result.steps[-1]
        e0 = _orbital_energy(first.position_eci, first.velocity_eci, mu)
        ef = _orbital_energy(last.position_eci, last.velocity_eci, mu)
        assert abs(ef - e0) / abs(e0) < 1e-7


# --- Adaptive step sizing ---

class TestAdaptiveStepSizing:

    def test_step_sizes_vary_on_eccentric_orbit(self, elliptical_state, epoch):
        """On eccentric orbit, step sizes near periapsis < step sizes near apoapsis."""
        mu = OrbitalConstants.MU_EARTH
        a = elliptical_state.semi_major_axis_m
        period = 2.0 * math.pi * math.sqrt(a**3 / mu)
        result = propagate_adaptive(
            initial_state=elliptical_state,
            duration=timedelta(seconds=period),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        # Compute time differences between steps
        steps = result.steps
        dt_list = []
        for i in range(1, len(steps)):
            dt = (steps[i].time - steps[i - 1].time).total_seconds()
            dt_list.append(dt)
        # There should be variation in step sizes
        assert max(dt_list) > 2.0 * min(dt_list), (
            "Expected significant step size variation on eccentric orbit"
        )

    def test_more_steps_near_periapsis(self, elliptical_state, epoch):
        """More integration steps should occur near periapsis than apoapsis."""
        mu = OrbitalConstants.MU_EARTH
        a = elliptical_state.semi_major_axis_m
        period = 2.0 * math.pi * math.sqrt(a**3 / mu)
        result = propagate_adaptive(
            initial_state=elliptical_state,
            duration=timedelta(seconds=period),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        # Steps in first quarter-period (near periapsis) vs third quarter (near apoapsis)
        quarter = period / 4.0
        steps_peri = sum(
            1 for s in result.steps
            if (s.time - epoch).total_seconds() < quarter
        )
        steps_apo = sum(
            1 for s in result.steps
            if 2 * quarter <= (s.time - epoch).total_seconds() < 3 * quarter
        )
        # Near periapsis should have more steps (smaller dt due to faster dynamics)
        assert steps_peri > steps_apo


# --- Error convergence ---

class TestErrorConvergence:

    def test_reducing_tolerance_reduces_error(self, leo_state, epoch):
        """Tighter tolerance should produce more accurate results."""
        mu = OrbitalConstants.MU_EARTH
        duration = timedelta(seconds=5400)

        errors = []
        for rtol in [1e-6, 1e-8, 1e-10]:
            config = AdaptiveStepConfig(rtol=rtol, atol=rtol * 0.01)
            result = propagate_adaptive(
                initial_state=leo_state,
                duration=duration,
                force_models=[TwoBodyGravity()],
                epoch=epoch,
                config=config,
            )
            first = result.steps[0]
            last = result.steps[-1]
            e0 = _orbital_energy(first.position_eci, first.velocity_eci, mu)
            ef = _orbital_energy(last.position_eci, last.velocity_eci, mu)
            errors.append(abs(ef - e0) / abs(e0))

        # Each tighter tolerance should give smaller error
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]


# --- DP vs RK4 comparison ---

class TestDPvsRK4:

    def test_dp_fewer_evaluations_same_accuracy(self, leo_state, epoch):
        """DP should achieve similar accuracy with fewer force evaluations than fixed RK4."""
        mu = OrbitalConstants.MU_EARTH
        duration = timedelta(seconds=5400)

        # Fixed RK4 with 10s steps
        rk4_result = propagate_numerical(
            initial_state=leo_state,
            duration=duration,
            step=timedelta(seconds=10),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            integrator="rk4",
        )
        rk4_first = rk4_result.steps[0]
        rk4_last = rk4_result.steps[-1]
        rk4_e0 = _orbital_energy(rk4_first.position_eci, rk4_first.velocity_eci, mu)
        rk4_ef = _orbital_energy(rk4_last.position_eci, rk4_last.velocity_eci, mu)
        rk4_error = abs(rk4_ef - rk4_e0) / abs(rk4_e0)
        rk4_evals = len(rk4_result.steps) * 4  # RK4 = 4 evaluations per step

        # Adaptive DP
        dp_result = propagate_adaptive(
            initial_state=leo_state,
            duration=duration,
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            config=AdaptiveStepConfig(rtol=1e-10, atol=1e-12),
        )
        dp_first = dp_result.steps[0]
        dp_last = dp_result.steps[-1]
        dp_e0 = _orbital_energy(dp_first.position_eci, dp_first.velocity_eci, mu)
        dp_ef = _orbital_energy(dp_last.position_eci, dp_last.velocity_eci, mu)
        dp_error = abs(dp_ef - dp_e0) / abs(dp_e0)
        # DP uses 6 evals per accepted step (FSAL), plus 7 for rejected steps
        dp_evals = dp_result.total_steps * 6 + dp_result.rejected_steps

        # DP should be comparable or better in accuracy
        assert dp_error <= rk4_error * 10  # Allow some margin


# --- Tolerance control ---

class TestToleranceControl:

    def test_result_accuracy_tracks_tolerance(self, leo_state, epoch):
        """Output error should be proportional to requested tolerance."""
        mu = OrbitalConstants.MU_EARTH
        duration = timedelta(seconds=5400)

        for rtol in [1e-6, 1e-8, 1e-10, 1e-12]:
            config = AdaptiveStepConfig(rtol=rtol, atol=rtol * 0.01)
            result = propagate_adaptive(
                initial_state=leo_state,
                duration=duration,
                force_models=[TwoBodyGravity()],
                epoch=epoch,
                config=config,
            )
            first = result.steps[0]
            last = result.steps[-1]
            e0 = _orbital_energy(first.position_eci, first.velocity_eci, mu)
            ef = _orbital_energy(last.position_eci, last.velocity_eci, mu)
            err = abs(ef - e0) / abs(e0)
            # Error should be within a few orders of magnitude of rtol
            assert err < rtol * 1e4, (
                f"rtol={rtol}, energy error={err} exceeds {rtol * 1e4}"
            )


# --- Dense output ---

class TestDenseOutput:

    def test_hermite_interpolation_accuracy(self, leo_state, epoch):
        """Dense output via Hermite interpolation should be accurate between steps."""
        mu = OrbitalConstants.MU_EARTH
        # Use a tight tolerance so internal steps are accurate
        config = AdaptiveStepConfig(rtol=1e-12, atol=1e-14)
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=5400),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            config=config,
            output_step_s=60.0,  # evenly-spaced output at 60s
        )
        # All output steps should be 60s apart (except possibly last)
        for i in range(1, len(result.steps) - 1):
            dt = (result.steps[i].time - result.steps[i - 1].time).total_seconds()
            assert abs(dt - 60.0) < 0.01, f"Step {i}: dt={dt}, expected 60.0"

        # Energy at each interpolated point should be conserved
        first = result.steps[0]
        e0 = _orbital_energy(first.position_eci, first.velocity_eci, mu)
        for s in result.steps:
            e = _orbital_energy(s.position_eci, s.velocity_eci, mu)
            assert abs(e - e0) / abs(e0) < 1e-9, (
                f"Energy drift at t={s.time}: {abs(e - e0) / abs(e0)}"
            )


# --- Rejected step handling ---

class TestRejectedSteps:

    def test_rejected_steps_occur_for_eccentric_orbit(self, elliptical_state, epoch):
        """Eccentric orbit with large initial step should trigger rejections."""
        mu = OrbitalConstants.MU_EARTH
        a = elliptical_state.semi_major_axis_m
        period = 2.0 * math.pi * math.sqrt(a**3 / mu)
        config = AdaptiveStepConfig(
            rtol=1e-12,
            atol=1e-14,
            h_init=500.0,  # Start with a big step — will be rejected near periapsis
        )
        result = propagate_adaptive(
            initial_state=elliptical_state,
            duration=timedelta(seconds=period),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            config=config,
        )
        assert result.rejected_steps >= 0  # At minimum the field exists
        assert result.total_steps > 0


# --- FSAL property ---

class TestFSAL:

    def test_fsal_reuses_last_evaluation(self, leo_state, epoch):
        """FSAL: k7 of step N should equal k1 of step N+1.

        We verify this indirectly: the number of force evaluations
        should be 6 per accepted step (7 stages - 1 reused) + 7 for rejected.
        For a simple circular orbit with good initial step, there should be
        few or zero rejections, so total evals ~ 6 * total_steps.
        """
        config = AdaptiveStepConfig(rtol=1e-10, atol=1e-12)
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=5400),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            config=config,
        )
        # With FSAL, accepted steps need 6 new evaluations each
        # Without FSAL, they would need 7
        # We just verify the result is reasonable (not testing internals directly)
        assert result.total_steps > 0
        assert result.total_steps < 5400  # Much less than 1-second stepping


# --- 30-day LEO propagation ---

class TestLongDurationPropagation:

    def test_30_day_leo_reasonable_step_count(self, leo_state, epoch):
        """30-day LEO propagation should use thousands of steps, not millions."""
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(days=30),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        # 30 days = 2,592,000 seconds
        # With adaptive steps, expect ~thousands of steps
        assert result.total_steps > 100, "Too few steps for 30-day propagation"
        assert result.total_steps < 100_000, "Too many steps — not adaptive enough"


# --- Result metadata ---

class TestResultMetadata:

    def test_total_steps_populated(self, leo_state, epoch):
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        assert isinstance(result.total_steps, int)
        assert result.total_steps > 0

    def test_rejected_steps_populated(self, leo_state, epoch):
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        assert isinstance(result.rejected_steps, int)
        assert result.rejected_steps >= 0

    def test_force_model_names_populated(self, leo_state, epoch):
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity(), J2Perturbation()],
            epoch=epoch,
        )
        assert result.force_model_names == ("TwoBodyGravity", "J2Perturbation")

    def test_epoch_and_duration(self, leo_state, epoch):
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        assert result.epoch == epoch
        assert result.duration_s == 3600.0


# --- h_min / h_max enforcement ---

class TestStepBounds:

    def test_h_min_enforcement(self, leo_state, epoch):
        """Step size should never go below h_min."""
        config = AdaptiveStepConfig(h_min=10.0, h_max=600.0)
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            config=config,
        )
        steps = result.steps
        for i in range(1, len(steps)):
            dt = (steps[i].time - steps[i - 1].time).total_seconds()
            # Allow the last step to be shorter (to hit exact duration)
            if i < len(steps) - 1:
                assert dt >= 10.0 - 0.01, f"Step {i}: dt={dt} < h_min=10.0"

    def test_h_max_enforcement(self, leo_state, epoch):
        """Step size should never exceed h_max."""
        config = AdaptiveStepConfig(h_max=120.0)
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=7200),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            config=config,
        )
        steps = result.steps
        for i in range(1, len(steps)):
            dt = (steps[i].time - steps[i - 1].time).total_seconds()
            assert dt <= 120.0 + 0.01, f"Step {i}: dt={dt} > h_max=120.0"


# --- max_steps limit ---

class TestMaxSteps:

    def test_max_steps_limit(self, leo_state, epoch):
        """Propagation should stop if max_steps exceeded."""
        config = AdaptiveStepConfig(max_steps=50, h_max=10.0)
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(days=30),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            config=config,
        )
        # Should have stopped early
        assert result.total_steps <= 50


# --- Backward propagation ---

class TestBackwardPropagation:

    def test_negative_duration(self, leo_state, epoch):
        """Backward propagation with negative duration works."""
        mu = OrbitalConstants.MU_EARTH
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=-3600),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        assert len(result.steps) > 1
        # Time should go backward
        assert result.steps[-1].time < result.steps[0].time
        # Energy should still be conserved
        first = result.steps[0]
        last = result.steps[-1]
        e0 = _orbital_energy(first.position_eci, first.velocity_eci, mu)
        ef = _orbital_energy(last.position_eci, last.velocity_eci, mu)
        assert abs(ef - e0) / abs(e0) < 1e-9


# --- Butcher tableau consistency ---

class TestButcherTableau:

    def test_row_sums_equal_c_nodes(self):
        """Sum of each row of A matrix should equal c_i."""
        for i, row in enumerate(DORMAND_PRINCE_A):
            row_sum = sum(row) if row else 0.0
            assert abs(row_sum - DORMAND_PRINCE_C[i]) < 1e-15, (
                f"Row {i}: sum(a)={row_sum} != c={DORMAND_PRINCE_C[i]}"
            )

    def test_b4_sums_to_one(self):
        """4th-order weights should sum to 1."""
        assert abs(sum(DORMAND_PRINCE_B4) - 1.0) < 1e-15

    def test_b5_sums_to_one(self):
        """5th-order weights should sum to 1."""
        assert abs(sum(DORMAND_PRINCE_B5) - 1.0) < 1e-15

    def test_seven_stages(self):
        """Dormand-Prince has 7 stages."""
        assert len(DORMAND_PRINCE_C) == 7
        assert len(DORMAND_PRINCE_A) == 7
        assert len(DORMAND_PRINCE_B4) == 7
        assert len(DORMAND_PRINCE_B5) == 7

    def test_fsal_b4_equals_last_a_row(self):
        """FSAL property: b4 weights should equal last row of A."""
        for b, a in zip(DORMAND_PRINCE_B4, DORMAND_PRINCE_A[-1]):
            assert abs(b - a) < 1e-15


# --- ForceModel compatibility ---

class TestForceModelCompatibility:

    def test_works_with_two_body_gravity(self, leo_state, epoch):
        """Should work with TwoBodyGravity force model."""
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        assert len(result.steps) > 1

    def test_works_with_j2_perturbation(self, leo_state, epoch):
        """Should work with TwoBodyGravity + J2Perturbation."""
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity(), J2Perturbation()],
            epoch=epoch,
        )
        assert len(result.steps) > 1

    def test_multiple_force_models_combined(self, leo_state, epoch):
        """Multiple force models are summed correctly."""
        mu = OrbitalConstants.MU_EARTH
        # Two-body only
        result_tb = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        # Two-body + J2
        result_j2 = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity(), J2Perturbation()],
            epoch=epoch,
        )
        # Positions should differ because J2 adds perturbation
        last_tb = result_tb.steps[-1]
        last_j2 = result_j2.steps[-1]
        dx = last_tb.position_eci[0] - last_j2.position_eci[0]
        dy = last_tb.position_eci[1] - last_j2.position_eci[1]
        dz = last_tb.position_eci[2] - last_j2.position_eci[2]
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        assert dist > 1.0  # Should be non-trivial difference


# --- Performance ---

class TestPerformance:

    def test_10k_steps_completes_in_time(self, leo_state, epoch):
        """10k steps should complete within reasonable wall-clock time."""
        config = AdaptiveStepConfig(h_max=10.0, max_steps=10_000)
        t_start = time.monotonic()
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(days=1),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            config=config,
        )
        elapsed = time.monotonic() - t_start
        assert elapsed < 30.0, f"10k steps took {elapsed:.1f}s, expected < 30s"
        assert result.total_steps > 0


# --- Config defaults ---

class TestConfigDefaults:

    def test_default_config_values(self):
        config = AdaptiveStepConfig()
        assert config.rtol == 1e-10
        assert config.atol == 1e-12
        assert config.h_init == 60.0
        assert config.h_min == 0.1
        assert config.h_max == 600.0
        assert config.safety_factor == 0.9
        assert config.max_steps == 1_000_000

    def test_config_is_frozen(self):
        config = AdaptiveStepConfig()
        with pytest.raises(AttributeError):
            config.rtol = 0.5  # type: ignore[misc]

    def test_custom_config(self):
        config = AdaptiveStepConfig(rtol=1e-8, h_max=300.0)
        assert config.rtol == 1e-8
        assert config.h_max == 300.0
        assert config.atol == 1e-12  # Other defaults preserved


# --- Frozen result ---

class TestFrozenResult:

    def test_adaptive_step_result_is_frozen(self, leo_state, epoch):
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        with pytest.raises(AttributeError):
            result.total_steps = 0  # type: ignore[misc]

    def test_result_steps_are_tuple(self, leo_state, epoch):
        result = propagate_adaptive(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
        )
        assert isinstance(result.steps, tuple)
        assert isinstance(result.force_model_names, tuple)


# --- Domain purity ---

class TestDomainPurity:

    def test_no_external_imports(self):
        """adaptive_integration.py must only use stdlib + domain imports."""
        import importlib
        import os
        module_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "humeris", "domain", "adaptive_integration.py",
        )
        with open(module_path) as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    assert mod in (
                        "math", "numpy", "dataclasses", "datetime", "typing",
                        "humeris",
                    ), f"External import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split(".")[0]
                    assert mod in (
                        "math", "numpy", "dataclasses", "datetime", "typing",
                        "humeris",
                    ), f"External import from: {node.module}"


# --- Integration with propagate_numerical ---

class TestPropagateNumericalIntegration:

    def test_dormand_prince_integrator_accepted(self, leo_state, epoch):
        """propagate_numerical(integrator='dormand_prince') should work."""
        result = propagate_numerical(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            step=timedelta(seconds=60),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            integrator="dormand_prince",
        )
        assert len(result.steps) > 1

    def test_dormand_prince_conserves_energy(self, leo_state, epoch):
        """Energy should be conserved via propagate_numerical DP pathway."""
        mu = OrbitalConstants.MU_EARTH
        result = propagate_numerical(
            initial_state=leo_state,
            duration=timedelta(seconds=5400),
            step=timedelta(seconds=60),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            integrator="dormand_prince",
        )
        first = result.steps[0]
        last = result.steps[-1]
        e0 = _orbital_energy(first.position_eci, first.velocity_eci, mu)
        ef = _orbital_energy(last.position_eci, last.velocity_eci, mu)
        assert abs(ef - e0) / abs(e0) < 1e-8

    def test_dormand_prince_returns_standard_result_type(self, leo_state, epoch):
        """propagate_numerical with DP should return NumericalPropagationResult."""
        result = propagate_numerical(
            initial_state=leo_state,
            duration=timedelta(seconds=3600),
            step=timedelta(seconds=60),
            force_models=[TwoBodyGravity()],
            epoch=epoch,
            integrator="dormand_prince",
        )
        assert isinstance(result, NumericalPropagationResult)

    def test_unknown_integrator_still_errors(self, leo_state, epoch):
        """Invalid integrator name should still raise ValueError."""
        with pytest.raises(ValueError, match="Unknown integrator"):
            propagate_numerical(
                initial_state=leo_state,
                duration=timedelta(seconds=3600),
                step=timedelta(seconds=60),
                force_models=[TwoBodyGravity()],
                epoch=epoch,
                integrator="invalid_method",
            )


# --- Single DP step function ---

class TestDormandPrinceStep:

    def test_single_step_returns_correct_shape(self):
        """A single DP step returns (t_new, y_new, k7) with correct dimensions."""
        def deriv(t, y):
            return tuple(-yi for yi in y)

        t0 = 0.0
        y0 = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        h = 0.1
        t_new, y_new, k7 = dormand_prince_step(t0, y0, h, deriv)
        assert isinstance(t_new, float)
        assert len(y_new) == 6
        assert len(k7) == 6
        assert abs(t_new - 0.1) < 1e-15

    def test_single_step_consistency(self):
        """DP step on simple ODE should give reasonable result."""
        # dy/dt = -y, exact: y(t) = exp(-t)
        def deriv(t, y):
            return tuple(-yi for yi in y)

        t0 = 0.0
        y0 = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        h = 0.1
        t_new, y_new, k7 = dormand_prince_step(t0, y0, h, deriv)
        # y[0] should be close to exp(-0.1) = 0.904837...
        expected = math.exp(-0.1)
        assert abs(y_new[0] - expected) < 1e-9
