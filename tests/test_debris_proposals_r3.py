# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for R3 math fixes and creative proposals P46/P48/P51/P58/P59."""
import math
from dataclasses import fields
from datetime import datetime, timedelta

import numpy as np
import pytest

# ── Math fix R3-16: EKF covariance symmetry ─────────────────────────

from humeris.domain.orbit_determination import (
    ODObservation,
    ODResult,
    run_ekf,
)


def _leo_state() -> tuple[float, float, float, float, float, float]:
    """LEO reference state at ~400 km altitude."""
    r = 6_771_000.0  # m
    v = math.sqrt(3.986004418e14 / r)
    return (r, 0.0, 0.0, 0.0, v, 0.0)


def _identity_6x6(scale: float = 1.0) -> list[list[float]]:
    return [[scale if i == j else 0.0 for j in range(6)] for i in range(6)]


class TestR316EKFCovarianceSymmetry:
    """R3-16: EKF covariance symmetry enforcement after prediction step."""

    def _run_ekf_basic(self) -> ODResult:
        state = _leo_state()
        t0 = datetime(2026, 1, 1)
        obs = [
            ODObservation(
                time=t0 + timedelta(seconds=i * 60),
                position_m=(state[0] + i * 10.0, state[1], state[2]),
                noise_std_m=100.0,
            )
            for i in range(5)
        ]
        p0 = _identity_6x6(1e6)
        q = _identity_6x6(1.0)
        return run_ekf(obs, state, p0, q)

    def test_final_covariance_is_symmetric(self):
        result = self._run_ekf_basic()
        cov = result.final_covariance
        for i in range(6):
            for j in range(6):
                # Tolerance accounts for Joseph-form update floating-point noise
                assert abs(cov[i][j] - cov[j][i]) < 1e-10, (
                    f"Covariance not symmetric at ({i},{j}): "
                    f"{cov[i][j]} vs {cov[j][i]}"
                )

    def test_all_intermediate_covariances_symmetric(self):
        result = self._run_ekf_basic()
        for k, est in enumerate(result.estimates):
            cov = est.covariance
            for i in range(6):
                for j in range(6):
                    assert abs(cov[i][j] - cov[j][i]) < 1e-10, (
                        f"Step {k}: Covariance not symmetric at ({i},{j})"
                    )

    def test_covariance_diagonal_positive(self):
        result = self._run_ekf_basic()
        cov = result.final_covariance
        for i in range(6):
            assert cov[i][i] > 0, f"Diagonal element {i} not positive: {cov[i][i]}"

    def test_symmetry_survives_many_steps(self):
        """With many steps, numerical drift could break symmetry without fix."""
        state = _leo_state()
        t0 = datetime(2026, 1, 1)
        obs = [
            ODObservation(
                time=t0 + timedelta(seconds=i * 120),
                position_m=(state[0] + i * 5.0, state[1], state[2]),
                noise_std_m=200.0,
            )
            for i in range(20)
        ]
        p0 = _identity_6x6(1e8)
        q = _identity_6x6(10.0)
        result = run_ekf(obs, state, p0, q)
        cov = result.final_covariance
        max_asym = 0.0
        for i in range(6):
            for j in range(i + 1, 6):
                max_asym = max(max_asym, abs(cov[i][j] - cov[j][i]))
        assert max_asym < 1e-10


# ── Math fix R3-17: Zero-inclination collision velocity guard ───────

from humeris.domain.kessler_heatmap import _mean_collision_velocity


class TestR317ZeroInclinationGuard:
    """R3-17: sin(i) clipped to sin(1 deg) for equatorial orbits."""

    def test_equatorial_velocity_nonzero(self):
        """At 0 deg inclination, velocity must not be zero."""
        v = _mean_collision_velocity(400.0, 450.0, 0.0, 0.01)
        assert v > 0, f"Collision velocity at equatorial orbit is {v}"

    def test_equatorial_velocity_uses_sin_1_deg(self):
        """Velocity should be V_circ * sqrt(2) * sin(1 deg) at i~0."""
        v = _mean_collision_velocity(400.0, 450.0, 0.0, 0.01)
        # Compute expected value
        from humeris.domain.orbital_mechanics import OrbitalConstants
        mid_alt_m = 425.0 * 1000.0
        r = OrbitalConstants.R_EARTH + mid_alt_m
        v_circ = math.sqrt(OrbitalConstants.MU_EARTH / r)
        expected = v_circ * math.sqrt(2.0) * math.sin(math.radians(1.0))
        assert abs(v - expected) / expected < 1e-6

    def test_polar_velocity_unchanged(self):
        """sin(90 deg) = 1 > sin(1 deg), so no clipping needed."""
        v = _mean_collision_velocity(400.0, 450.0, 85.0, 95.0)
        from humeris.domain.orbital_mechanics import OrbitalConstants
        mid_alt_m = 425.0 * 1000.0
        r = OrbitalConstants.R_EARTH + mid_alt_m
        v_circ = math.sqrt(OrbitalConstants.MU_EARTH / r)
        expected = v_circ * math.sqrt(2.0) * math.sin(math.radians(90.0))
        assert abs(v - expected) / expected < 1e-6

    def test_low_inclination_clipped(self):
        """At 0.5 deg inclination, sin(0.5) < sin(1), so should be clipped."""
        v = _mean_collision_velocity(400.0, 450.0, 0.0, 1.0)
        # mid inc = 0.5 deg, sin(0.5 deg) < sin(1 deg)
        from humeris.domain.orbital_mechanics import OrbitalConstants
        mid_alt_m = 425.0 * 1000.0
        r = OrbitalConstants.R_EARTH + mid_alt_m
        v_circ = math.sqrt(OrbitalConstants.MU_EARTH / r)
        # Should use sin(1 deg) since sin(0.5 deg) < sin(1 deg)
        expected = v_circ * math.sqrt(2.0) * math.sin(math.radians(1.0))
        assert abs(v - expected) / expected < 1e-6


# ── P46: Reynolds Decomposition of Debris Density ───────────────────

from humeris.domain.kessler_heatmap import (
    DebrisDensityEvolution,
    ReynoldsDebrisDecomposition,
    compute_reynolds_decomposition,
    compute_debris_density_evolution,
)


class TestP46ReynoldsDecomposition:
    """P46: Reynolds decomposition of debris density evolution."""

    def _make_evolution(self, n_alt: int = 10, n_time: int = 20) -> DebrisDensityEvolution:
        """Create synthetic density evolution with known properties."""
        alt_bins = tuple(300.0 + i * 50.0 for i in range(n_alt))
        time_steps = tuple(float(t) for t in range(n_time))
        # Base density + sinusoidal fluctuation
        density = []
        for t in range(n_time):
            row = tuple(
                1e-8 * (1.0 + 0.5 * math.sin(2.0 * math.pi * t / n_time + i * 0.1))
                for i in range(n_alt)
            )
            density.append(row)
        peak = tuple(max(row) for row in density)
        total = tuple(sum(row) * 50.0 for row in density)
        return DebrisDensityEvolution(
            altitude_bins=alt_bins,
            time_steps=time_steps,
            density_evolution=tuple(density),
            peak_density_trajectory=peak,
            total_mass_trajectory=total,
        )

    def test_returns_frozen_dataclass(self):
        evo = self._make_evolution()
        result = compute_reynolds_decomposition(evo)
        assert isinstance(result, ReynoldsDebrisDecomposition)
        with pytest.raises(AttributeError):
            result.mean_density = (0.0,)  # type: ignore[misc]

    def test_mean_density_shape(self):
        evo = self._make_evolution(n_alt=10)
        result = compute_reynolds_decomposition(evo)
        assert len(result.mean_density) == 10

    def test_fluctuation_rms_positive(self):
        evo = self._make_evolution()
        result = compute_reynolds_decomposition(evo)
        for rms in result.fluctuation_rms:
            assert rms >= 0.0

    def test_mean_density_positive(self):
        evo = self._make_evolution()
        result = compute_reynolds_decomposition(evo)
        for m in result.mean_density:
            assert m > 0.0

    def test_tke_nonnegative(self):
        evo = self._make_evolution()
        result = compute_reynolds_decomposition(evo)
        for tke in result.turbulent_kinetic_energy:
            assert tke >= 0.0

    def test_turbulence_intensity_nonnegative(self):
        evo = self._make_evolution()
        result = compute_reynolds_decomposition(evo)
        for ti in result.turbulence_intensity:
            assert ti >= 0.0

    def test_constant_density_zero_fluctuation(self):
        """Constant density should give zero fluctuation."""
        n_alt, n_time = 5, 10
        alt_bins = tuple(300.0 + i * 50.0 for i in range(n_alt))
        time_steps = tuple(float(t) for t in range(n_time))
        const_val = 1e-8
        density = tuple(
            tuple(const_val for _ in range(n_alt))
            for _ in range(n_time)
        )
        evo = DebrisDensityEvolution(
            altitude_bins=alt_bins,
            time_steps=time_steps,
            density_evolution=density,
            peak_density_trajectory=tuple(const_val for _ in range(n_time)),
            total_mass_trajectory=tuple(const_val * n_alt * 50.0 for _ in range(n_time)),
        )
        result = compute_reynolds_decomposition(evo)
        for rms in result.fluctuation_rms:
            assert abs(rms) < 1e-20

    def test_eddy_diffusivity_shape(self):
        evo = self._make_evolution(n_alt=10)
        result = compute_reynolds_decomposition(evo)
        assert len(result.eddy_diffusivity) == 10
        # Boundary values should be 0
        assert result.eddy_diffusivity[0] == 0.0
        assert result.eddy_diffusivity[-1] == 0.0

    def test_mean_eddy_diffusivity_positive(self):
        evo = self._make_evolution()
        result = compute_reynolds_decomposition(evo)
        assert result.mean_eddy_diffusivity >= 0.0

    def test_minimum_time_steps_required(self):
        """Should raise ValueError with < 2 time steps."""
        evo = DebrisDensityEvolution(
            altitude_bins=(400.0,),
            time_steps=(0.0,),
            density_evolution=((1e-8,),),
            peak_density_trajectory=(1e-8,),
            total_mass_trajectory=(1e-8,),
        )
        with pytest.raises(ValueError, match="Need >= 2"):
            compute_reynolds_decomposition(evo)

    def test_from_real_evolution(self):
        """Run on actual Fokker-Planck output."""
        evo = compute_debris_density_evolution(
            n_altitude_bins=20,
            duration_years=5.0,
            step_years=0.5,
        )
        result = compute_reynolds_decomposition(evo)
        assert len(result.mean_density) == 20
        assert result.mean_eddy_diffusivity >= 0.0

    def test_tke_equals_half_variance(self):
        """TKE = 0.5 * <rho'^2>."""
        evo = self._make_evolution()
        result = compute_reynolds_decomposition(evo)
        for i, tke in enumerate(result.turbulent_kinetic_energy):
            expected = 0.5 * result.fluctuation_rms[i] ** 2
            assert abs(tke - expected) < 1e-25


# ── P48: Boundary Layer Flow Regime Transition ──────────────────────

from humeris.domain.atmosphere import (
    FlowRegimeTransition,
    compute_flow_regime_transition,
    _knudsen_number,
    _mean_free_path,
    _cd_interpolation,
)


class TestP48FlowRegimeTransition:
    """P48: Boundary layer flow regime transition analysis."""

    def test_returns_frozen_dataclass(self):
        result = compute_flow_regime_transition(400.0)
        assert isinstance(result, FlowRegimeTransition)
        with pytest.raises(AttributeError):
            result.cd_corrected = 0.0  # type: ignore[misc]

    def test_knudsen_increases_with_altitude(self):
        kn_low = _knudsen_number(200.0, 1.0)
        kn_high = _knudsen_number(800.0, 1.0)
        assert kn_high > kn_low

    def test_mean_free_path_increases_with_altitude(self):
        mfp_low = _mean_free_path(200.0)
        mfp_high = _mean_free_path(800.0)
        assert mfp_high > mfp_low

    def test_cd_interpolation_limits(self):
        """At extreme Kn values, C_D should approach limits."""
        cd_fm = 2.2
        cd_cont = 1.0
        # Very high Kn -> free molecular
        assert abs(_cd_interpolation(1000.0, cd_fm, cd_cont) - cd_fm) < 0.01
        # Very low Kn -> continuum
        assert abs(_cd_interpolation(0.001, cd_fm, cd_cont) - cd_cont) < 0.01

    def test_cd_interpolation_at_kn_1(self):
        """At Kn = 1 (= Kn_ref), f = 0.5, so C_D = midpoint."""
        cd_fm = 2.2
        cd_cont = 1.0
        cd_mid = _cd_interpolation(1.0, cd_fm, cd_cont, kn_ref=1.0)
        expected = 0.5 * cd_fm + 0.5 * cd_cont
        assert abs(cd_mid - expected) < 1e-10

    def test_boundary_layer_altitude_reasonable(self):
        """Boundary layer altitude should be in a reasonable LEO range."""
        result = compute_flow_regime_transition(400.0, char_length_m=1.0)
        assert 100.0 < result.boundary_layer_altitude_km < 600.0

    def test_boundary_layer_thickness_positive(self):
        result = compute_flow_regime_transition(400.0)
        assert result.boundary_layer_thickness_km > 0

    def test_cd_corrected_between_limits(self):
        result = compute_flow_regime_transition(400.0)
        assert result.cd_continuum <= result.cd_corrected <= result.cd_free_molecular

    def test_lifetime_correction_factor_positive(self):
        result = compute_flow_regime_transition(400.0)
        assert result.lifetime_correction_factor > 0

    def test_smaller_spacecraft_higher_knudsen(self):
        """Smaller L_char -> higher Kn (more free-molecular)."""
        r_small = compute_flow_regime_transition(400.0, char_length_m=0.1)
        r_large = compute_flow_regime_transition(400.0, char_length_m=10.0)
        assert r_small.knudsen_at_altitude > r_large.knudsen_at_altitude

    def test_invalid_char_length(self):
        with pytest.raises(ValueError, match="char_length_m"):
            compute_flow_regime_transition(400.0, char_length_m=0.0)

    def test_high_altitude_free_molecular(self):
        """At very high altitude, should be in free-molecular regime."""
        result = compute_flow_regime_transition(1500.0, char_length_m=1.0)
        assert result.knudsen_at_altitude > 1.0
        assert abs(result.cd_corrected - result.cd_free_molecular) < 0.3


# ── P51: Generating Function for Conjunction Counting ───────────────

from humeris.domain.conjunction import (
    ConjunctionCountDistribution,
    compute_conjunction_count_distribution,
)


class TestP51ConjunctionCountDistribution:
    """P51: Generating function for conjunction counting."""

    def test_returns_frozen_dataclass(self):
        result = compute_conjunction_count_distribution([0.1, 0.2])
        assert isinstance(result, ConjunctionCountDistribution)
        with pytest.raises(AttributeError):
            result.mean_conjunctions = 0.0  # type: ignore[misc]

    def test_mean_equals_sum_of_probabilities(self):
        probs = [0.1, 0.2, 0.3]
        result = compute_conjunction_count_distribution(probs)
        assert abs(result.mean_conjunctions - sum(probs)) < 1e-12

    def test_variance_formula(self):
        probs = [0.1, 0.2, 0.3]
        result = compute_conjunction_count_distribution(probs)
        expected_var = sum(p * (1 - p) for p in probs)
        assert abs(result.variance - expected_var) < 1e-12

    def test_prob_zero_is_product(self):
        probs = [0.1, 0.2, 0.3]
        result = compute_conjunction_count_distribution(probs)
        expected = math.prod(1.0 - p for p in probs)
        assert abs(result.prob_zero_conjunctions - expected) < 1e-12

    def test_cdf_sums_to_one(self):
        probs = [0.1, 0.2, 0.3, 0.4]
        result = compute_conjunction_count_distribution(probs)
        # CDF at max_k should be 1.0
        assert abs(result.prob_at_most_k[-1] - 1.0) < 1e-10

    def test_cdf_monotonic(self):
        probs = [0.1, 0.2, 0.3]
        result = compute_conjunction_count_distribution(probs)
        for i in range(len(result.prob_at_most_k) - 1):
            assert result.prob_at_most_k[i] <= result.prob_at_most_k[i + 1] + 1e-15

    def test_single_probability(self):
        result = compute_conjunction_count_distribution([0.7])
        assert abs(result.prob_zero_conjunctions - 0.3) < 1e-12
        assert abs(result.mean_conjunctions - 0.7) < 1e-12
        assert len(result.prob_at_most_k) == 2  # k=0, k=1

    def test_all_certain_conjunctions(self):
        """All p=1 means exactly n conjunctions."""
        probs = [1.0, 1.0, 1.0]
        result = compute_conjunction_count_distribution(probs)
        assert abs(result.prob_zero_conjunctions) < 1e-12
        assert abs(result.mean_conjunctions - 3.0) < 1e-12

    def test_all_zero_probabilities(self):
        """All p=0 means P(K=0)=1."""
        probs = [0.0, 0.0, 0.0]
        result = compute_conjunction_count_distribution(probs)
        assert abs(result.prob_zero_conjunctions - 1.0) < 1e-12
        assert abs(result.mean_conjunctions) < 1e-12

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            compute_conjunction_count_distribution([0.1, 1.5])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_conjunction_count_distribution([])

    def test_skewness_sign(self):
        """For mostly small p values, skewness should be positive."""
        probs = [0.01] * 20
        result = compute_conjunction_count_distribution(probs)
        assert result.skewness > 0


# ── P58: Coupled Debris-Operations Feedback ─────────────────────────

from humeris.domain.cascade_analysis import (
    CoupledDebrisOperations,
    compute_coupled_debris_operations,
)


class TestP58CoupledDebrisOperations:
    """P58: Coupled debris-operations feedback model."""

    def test_returns_frozen_dataclass(self):
        result = compute_coupled_debris_operations()
        assert isinstance(result, CoupledDebrisOperations)
        with pytest.raises(AttributeError):
            result.is_stable = False  # type: ignore[misc]

    def test_populations_tuple_structure(self):
        result = compute_coupled_debris_operations(duration_years=10.0)
        assert len(result.populations) == 4  # time, sat, debris, prop
        times, sats, debris, prop = result.populations
        assert len(times) == len(sats) == len(debris) == len(prop)

    def test_populations_nonnegative(self):
        result = compute_coupled_debris_operations()
        _, sats, debris, prop = result.populations
        for v in sats:
            assert v >= 0
        for v in debris:
            assert v >= 0
        for v in prop:
            assert v >= 0

    def test_equilibrium_values_nonnegative(self):
        result = compute_coupled_debris_operations()
        assert result.equilibrium_satellites >= 0
        assert result.equilibrium_debris >= 0
        assert result.equilibrium_propellant_rate >= 0

    def test_eigenvalues_count(self):
        result = compute_coupled_debris_operations()
        assert len(result.jacobian_eigenvalues) == 3

    def test_oscillation_period_positive(self):
        result = compute_coupled_debris_operations()
        assert result.oscillation_period_years > 0

    def test_hill_function_saturation(self):
        """With very high debris, maneuver rate should approach mu_0."""
        # High debris -> Hill function near saturation -> more satellite losses
        result_high = compute_coupled_debris_operations(
            initial_debris=100000.0, k_m=100.0, mu_0=0.5,
            duration_years=50.0,
        )
        # Low debris -> Hill function near zero
        result_low = compute_coupled_debris_operations(
            initial_debris=1.0, k_m=100000.0, mu_0=0.5,
            duration_years=50.0,
        )
        # High debris should result in fewer equilibrium satellites
        assert result_high.equilibrium_satellites < result_low.equilibrium_satellites

    def test_negative_initial_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_coupled_debris_operations(initial_satellites=-1.0)

    def test_zero_launch_rate_depletion(self):
        """With no launches, satellites should decline."""
        result = compute_coupled_debris_operations(
            launch_rate=0.0,
            duration_years=200.0,
        )
        assert result.equilibrium_satellites < 100.0

    def test_stability_with_low_coupling(self):
        """Low collision rate should give stable equilibrium."""
        result = compute_coupled_debris_operations(
            collision_rate=1e-10,
            duration_years=200.0,
        )
        assert result.is_stable


# ── P59: Optimal OD Scheduling via Fisher Information ───────────────

from humeris.domain.orbit_determination import (
    OptimalObservationSchedule,
    compute_optimal_observation_schedule,
)


class TestP59OptimalObservationSchedule:
    """P59: Optimal observation schedule via Fisher information."""

    def _reference_state(self) -> tuple[float, float, float, float, float, float]:
        r = 6_771_000.0
        v = math.sqrt(3.986004418e14 / r)
        return (r, 0.0, 0.0, 0.0, v, 0.0)

    def test_returns_frozen_dataclass(self):
        state = self._reference_state()
        result = compute_optimal_observation_schedule(
            state,
            [0.0, 60.0, 120.0, 180.0, 240.0],
            [100.0] * 5,
            n_select=3,
        )
        assert isinstance(result, OptimalObservationSchedule)
        with pytest.raises(AttributeError):
            result.condition_number = 0.0  # type: ignore[misc]

    def test_selects_correct_count(self):
        state = self._reference_state()
        result = compute_optimal_observation_schedule(
            state,
            [float(i * 60) for i in range(10)],
            [100.0] * 10,
            n_select=5,
        )
        assert len(result.selected_indices) == 5

    def test_selected_indices_unique(self):
        state = self._reference_state()
        result = compute_optimal_observation_schedule(
            state,
            [float(i * 60) for i in range(10)],
            [100.0] * 10,
            n_select=5,
        )
        assert len(set(result.selected_indices)) == len(result.selected_indices)

    def test_fisher_det_positive(self):
        state = self._reference_state()
        result = compute_optimal_observation_schedule(
            state,
            [0.0, 300.0, 600.0, 900.0, 1200.0],
            [100.0] * 5,
            n_select=3,
        )
        assert result.fisher_information_det > 0

    def test_optimal_beats_or_ties_baseline(self):
        """D-optimal should give >= baseline FIM determinant."""
        state = self._reference_state()
        result = compute_optimal_observation_schedule(
            state,
            [float(i * 120) for i in range(20)],
            [100.0] * 20,
            n_select=5,
        )
        assert result.information_gain_ratio >= 1.0 - 1e-10

    def test_condition_number_positive(self):
        state = self._reference_state()
        result = compute_optimal_observation_schedule(
            state,
            [0.0, 300.0, 600.0, 900.0],
            [100.0] * 4,
            n_select=3,
        )
        assert result.condition_number > 0

    def test_empty_candidates_raises(self):
        state = self._reference_state()
        with pytest.raises(ValueError, match="must not be empty"):
            compute_optimal_observation_schedule(state, [], [], n_select=1)

    def test_n_select_exceeds_candidates_raises(self):
        state = self._reference_state()
        with pytest.raises(ValueError, match="n_select"):
            compute_optimal_observation_schedule(
                state, [0.0, 60.0], [100.0, 100.0], n_select=5,
            )

    def test_lower_noise_higher_information(self):
        """Lower measurement noise should increase FIM determinant."""
        state = self._reference_state()
        times = [float(i * 300) for i in range(10)]
        result_low = compute_optimal_observation_schedule(
            state, times, [10.0] * 10, n_select=5,
        )
        result_high = compute_optimal_observation_schedule(
            state, times, [1000.0] * 10, n_select=5,
        )
        assert result_low.fisher_information_det > result_high.fisher_information_det

    def test_more_observations_more_information(self):
        """Selecting more observations should give higher FIM det."""
        state = self._reference_state()
        times = [float(i * 300) for i in range(10)]
        noise = [100.0] * 10
        result_3 = compute_optimal_observation_schedule(
            state, times, noise, n_select=3,
        )
        result_7 = compute_optimal_observation_schedule(
            state, times, noise, n_select=7,
        )
        assert result_7.fisher_information_det >= result_3.fisher_information_det
