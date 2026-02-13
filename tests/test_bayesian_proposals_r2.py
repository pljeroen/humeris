# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for Round 2 creative proposals: P21, P27, P22, P23, P29.

P21: Bayesian Conjunction Probability via Beta-Binomial (conjunction.py)
P27: Black-Scholes Maneuver Option Pricing (conjunction_management.py)
P22: Bayesian Model Selection for Force Models (orbit_determination.py)
P23: Particle Filter for Non-Gaussian OD (orbit_determination.py)
P29: Compressed Sensing for Sparse OD (orbit_determination.py)
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
MU = OrbitalConstants.MU_EARTH
R_EARTH = OrbitalConstants.R_EARTH


# ═══════════════════════════════════════════════════════════════════
# P21: Bayesian Conjunction Probability via Beta-Binomial
# ═══════════════════════════════════════════════════════════════════


class TestBayesianCollisionProbabilityDataclass:
    """Verify BayesianCollisionProbability is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.conjunction import BayesianCollisionProbability

        result = BayesianCollisionProbability(
            posterior_mean=1e-5,
            posterior_std=1e-6,
            credible_interval_95=(1e-6, 2e-5),
            exceedance_probability=0.3,
            point_estimate=1e-5,
        )
        with pytest.raises(AttributeError):
            result.posterior_mean = 0.5

    def test_all_fields_present(self):
        from humeris.domain.conjunction import BayesianCollisionProbability

        result = BayesianCollisionProbability(
            posterior_mean=1e-5,
            posterior_std=1e-6,
            credible_interval_95=(1e-6, 2e-5),
            exceedance_probability=0.3,
            point_estimate=1e-5,
        )
        assert hasattr(result, "posterior_mean")
        assert hasattr(result, "posterior_std")
        assert hasattr(result, "credible_interval_95")
        assert hasattr(result, "exceedance_probability")
        assert hasattr(result, "point_estimate")


class TestRegularizedIncompleteBeta:
    """Test the regularized incomplete beta function implementation."""

    def test_boundary_zero(self):
        from humeris.domain.conjunction import _regularized_incomplete_beta

        assert _regularized_incomplete_beta(0.0, 2.0, 3.0) == 0.0

    def test_boundary_one(self):
        from humeris.domain.conjunction import _regularized_incomplete_beta

        assert _regularized_incomplete_beta(1.0, 2.0, 3.0) == 1.0

    def test_symmetric_case(self):
        """I_0.5(a, a) = 0.5 for any a > 0 (symmetry)."""
        from humeris.domain.conjunction import _regularized_incomplete_beta

        result = _regularized_incomplete_beta(0.5, 5.0, 5.0)
        assert abs(result - 0.5) < 1e-8

    def test_known_value_beta_1_1(self):
        """Beta(1, 1) = Uniform[0,1], so I_x(1, 1) = x."""
        from humeris.domain.conjunction import _regularized_incomplete_beta

        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = _regularized_incomplete_beta(x, 1.0, 1.0)
            assert abs(result - x) < 1e-8, f"I_{x}(1,1) = {result}, expected {x}"

    def test_known_value_beta_2_1(self):
        """Beta(2, 1): I_x(2, 1) = x^2."""
        from humeris.domain.conjunction import _regularized_incomplete_beta

        for x in [0.2, 0.5, 0.8]:
            result = _regularized_incomplete_beta(x, 2.0, 1.0)
            expected = x ** 2
            assert abs(result - expected) < 1e-8, (
                f"I_{x}(2,1) = {result}, expected {expected}"
            )

    def test_known_value_beta_1_2(self):
        """Beta(1, 2): I_x(1, 2) = 1 - (1-x)^2."""
        from humeris.domain.conjunction import _regularized_incomplete_beta

        for x in [0.2, 0.5, 0.8]:
            result = _regularized_incomplete_beta(x, 1.0, 2.0)
            expected = 1.0 - (1.0 - x) ** 2
            assert abs(result - expected) < 1e-8, (
                f"I_{x}(1,2) = {result}, expected {expected}"
            )

    def test_monotonic(self):
        """I_x(a, b) is monotonically increasing in x."""
        from humeris.domain.conjunction import _regularized_incomplete_beta

        a, b = 3.0, 5.0
        prev = 0.0
        for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            val = _regularized_incomplete_beta(x, a, b)
            assert val >= prev, f"Not monotonic at x={x}: {val} < {prev}"
            prev = val

    def test_symmetry_relation(self):
        """I_x(a, b) = 1 - I_{1-x}(b, a)."""
        from humeris.domain.conjunction import _regularized_incomplete_beta

        a, b, x = 3.0, 7.0, 0.3
        lhs = _regularized_incomplete_beta(x, a, b)
        rhs = 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)
        assert abs(lhs - rhs) < 1e-10


class TestLogGamma:
    """Test the log-gamma function implementation."""

    def test_known_values(self):
        """Check against known log-gamma values."""
        from humeris.domain.conjunction import _log_gamma

        # log(Gamma(1)) = log(1) = 0
        assert abs(_log_gamma(1.0)) < 1e-12

        # log(Gamma(2)) = log(1!) = 0
        assert abs(_log_gamma(2.0)) < 1e-12

        # log(Gamma(3)) = log(2!) = log(2) ≈ 0.6931
        assert abs(_log_gamma(3.0) - math.log(2.0)) < 1e-10

        # log(Gamma(4)) = log(3!) = log(6) ≈ 1.7918
        assert abs(_log_gamma(4.0) - math.log(6.0)) < 1e-10

    def test_half_integer(self):
        """Gamma(0.5) = sqrt(pi), so log(Gamma(0.5)) = 0.5*log(pi)."""
        from humeris.domain.conjunction import _log_gamma

        expected = 0.5 * math.log(math.pi)
        result = _log_gamma(0.5)
        assert abs(result - expected) < 1e-10


class TestBayesianCollisionProbability:
    """Test the bayesian_collision_probability function."""

    def test_basic_call(self):
        from humeris.domain.conjunction import bayesian_collision_probability

        result = bayesian_collision_probability(
            pc_point=1e-4,
            n_obs=100,
            cov_condition_number=10.0,
        )
        assert result.point_estimate == 1e-4
        assert 0 < result.posterior_mean < 1
        assert result.posterior_std > 0

    def test_posterior_mean_near_point_estimate(self):
        """With high kappa, posterior mean should be close to point estimate."""
        from humeris.domain.conjunction import bayesian_collision_probability

        pc = 0.001
        result = bayesian_collision_probability(
            pc_point=pc,
            n_obs=1000,
            cov_condition_number=1.0,
        )
        # kappa = 1000^2 / 1 = 1e6 — very tight posterior
        assert abs(result.posterior_mean - pc) < pc * 0.01  # within 1%

    def test_low_quality_widens_posterior(self):
        """Poor observation quality (high condition number) should widen posterior."""
        from humeris.domain.conjunction import bayesian_collision_probability

        pc = 0.001
        good = bayesian_collision_probability(
            pc_point=pc, n_obs=100, cov_condition_number=1.0,
        )
        poor = bayesian_collision_probability(
            pc_point=pc, n_obs=100, cov_condition_number=1000.0,
        )
        assert poor.posterior_std > good.posterior_std

    def test_credible_interval_contains_mean(self):
        from humeris.domain.conjunction import bayesian_collision_probability

        result = bayesian_collision_probability(
            pc_point=0.01, n_obs=50, cov_condition_number=5.0,
        )
        lo, hi = result.credible_interval_95
        assert lo <= result.posterior_mean <= hi

    def test_credible_interval_ordered(self):
        from humeris.domain.conjunction import bayesian_collision_probability

        result = bayesian_collision_probability(
            pc_point=0.01, n_obs=50, cov_condition_number=5.0,
        )
        lo, hi = result.credible_interval_95
        assert lo < hi

    def test_exceedance_probability_range(self):
        from humeris.domain.conjunction import bayesian_collision_probability

        result = bayesian_collision_probability(
            pc_point=1e-4, n_obs=100, cov_condition_number=10.0,
            threshold=1e-4,
        )
        assert 0 <= result.exceedance_probability <= 1

    def test_exceedance_high_threshold_low_probability(self):
        """Exceedance probability should be low when threshold >> Pc."""
        from humeris.domain.conjunction import bayesian_collision_probability

        result = bayesian_collision_probability(
            pc_point=1e-6, n_obs=100, cov_condition_number=2.0,
            threshold=0.1,
        )
        assert result.exceedance_probability < 0.01

    def test_exceedance_low_threshold_high_probability(self):
        """Exceedance probability should be high when threshold << Pc."""
        from humeris.domain.conjunction import bayesian_collision_probability

        result = bayesian_collision_probability(
            pc_point=0.1, n_obs=100, cov_condition_number=2.0,
            threshold=1e-6,
        )
        assert result.exceedance_probability > 0.99

    def test_validation_pc_out_of_range(self):
        from humeris.domain.conjunction import bayesian_collision_probability

        with pytest.raises(ValueError, match="pc_point"):
            bayesian_collision_probability(
                pc_point=-0.1, n_obs=10, cov_condition_number=1.0,
            )
        with pytest.raises(ValueError, match="pc_point"):
            bayesian_collision_probability(
                pc_point=1.5, n_obs=10, cov_condition_number=1.0,
            )

    def test_validation_n_obs(self):
        from humeris.domain.conjunction import bayesian_collision_probability

        with pytest.raises(ValueError, match="n_obs"):
            bayesian_collision_probability(
                pc_point=0.01, n_obs=0, cov_condition_number=1.0,
            )

    def test_validation_condition_number(self):
        from humeris.domain.conjunction import bayesian_collision_probability

        with pytest.raises(ValueError, match="cov_condition_number"):
            bayesian_collision_probability(
                pc_point=0.01, n_obs=10, cov_condition_number=0.5,
            )

    def test_zero_pc_point(self):
        """Zero point estimate should still produce valid result."""
        from humeris.domain.conjunction import bayesian_collision_probability

        result = bayesian_collision_probability(
            pc_point=0.0, n_obs=100, cov_condition_number=1.0,
        )
        assert result.posterior_mean >= 0
        assert result.posterior_std > 0

    def test_near_one_pc_point(self):
        """Near-1 point estimate should still produce valid result."""
        from humeris.domain.conjunction import bayesian_collision_probability

        result = bayesian_collision_probability(
            pc_point=1.0, n_obs=100, cov_condition_number=1.0,
        )
        assert result.posterior_mean <= 1.0
        assert result.posterior_std > 0


# ═══════════════════════════════════════════════════════════════════
# P27: Black-Scholes Maneuver Option Pricing
# ═══════════════════════════════════════════════════════════════════


class TestManeuverOptionDataclass:
    """Verify ManeuverOption is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.conjunction_management import ManeuverOption

        result = ManeuverOption(
            option_value_ms=0.1,
            optimal_threshold_m=500.0,
            should_maneuver_now=False,
            time_value_ms=0.05,
            intrinsic_value_ms=0.05,
        )
        with pytest.raises(AttributeError):
            result.option_value_ms = 1.0

    def test_all_fields_present(self):
        from humeris.domain.conjunction_management import ManeuverOption

        result = ManeuverOption(
            option_value_ms=0.1,
            optimal_threshold_m=500.0,
            should_maneuver_now=False,
            time_value_ms=0.05,
            intrinsic_value_ms=0.05,
        )
        assert hasattr(result, "option_value_ms")
        assert hasattr(result, "optimal_threshold_m")
        assert hasattr(result, "should_maneuver_now")
        assert hasattr(result, "time_value_ms")
        assert hasattr(result, "intrinsic_value_ms")


class TestNormalCDF:
    """Test the normal CDF helper."""

    def test_at_zero(self):
        from humeris.domain.conjunction_management import _normal_cdf

        assert abs(_normal_cdf(0.0) - 0.5) < 1e-10

    def test_symmetry(self):
        from humeris.domain.conjunction_management import _normal_cdf

        for x in [0.5, 1.0, 2.0, 3.0]:
            assert abs(_normal_cdf(x) + _normal_cdf(-x) - 1.0) < 1e-10

    def test_known_values(self):
        from humeris.domain.conjunction_management import _normal_cdf

        # N(1) ≈ 0.8413
        assert abs(_normal_cdf(1.0) - 0.8413) < 0.001
        # N(2) ≈ 0.9772
        assert abs(_normal_cdf(2.0) - 0.9772) < 0.001


class TestManeuverOptionPricing:
    """Test compute_maneuver_option."""

    def test_basic_call(self):
        from humeris.domain.conjunction_management import compute_maneuver_option

        result = compute_maneuver_option(
            miss_distance_m=1000.0,
            miss_distance_sigma_m=200.0,
            safe_distance_m=500.0,
            time_to_tca_s=3600.0,
            maneuver_cost_ms_per_m=0.001,
        )
        assert result.option_value_ms >= 0
        assert result.optimal_threshold_m > 0

    def test_deep_in_the_money(self):
        """Miss distance << safe distance: high intrinsic value, should maneuver."""
        from humeris.domain.conjunction_management import compute_maneuver_option

        result = compute_maneuver_option(
            miss_distance_m=100.0,
            miss_distance_sigma_m=50.0,
            safe_distance_m=1000.0,
            time_to_tca_s=3600.0,
            maneuver_cost_ms_per_m=0.001,
        )
        assert result.intrinsic_value_ms > 0
        assert result.should_maneuver_now is True

    def test_deep_out_of_money(self):
        """Miss distance >> safe distance: low option value."""
        from humeris.domain.conjunction_management import compute_maneuver_option

        result = compute_maneuver_option(
            miss_distance_m=10000.0,
            miss_distance_sigma_m=100.0,
            safe_distance_m=500.0,
            time_to_tca_s=3600.0,
            maneuver_cost_ms_per_m=0.001,
        )
        assert result.intrinsic_value_ms == 0.0
        # Option value should be small (far OTM)
        assert result.option_value_ms < 0.5  # relative to cost scale

    def test_time_value_positive(self):
        """With time remaining and uncertainty, time value should be >= 0."""
        from humeris.domain.conjunction_management import compute_maneuver_option

        result = compute_maneuver_option(
            miss_distance_m=500.0,
            miss_distance_sigma_m=200.0,
            safe_distance_m=500.0,
            time_to_tca_s=7200.0,
            maneuver_cost_ms_per_m=0.001,
        )
        assert result.time_value_ms >= 0

    def test_option_value_geq_intrinsic(self):
        """Option value must always be >= intrinsic value (no free lunch)."""
        from humeris.domain.conjunction_management import compute_maneuver_option

        for miss in [200.0, 500.0, 1000.0, 2000.0]:
            result = compute_maneuver_option(
                miss_distance_m=miss,
                miss_distance_sigma_m=100.0,
                safe_distance_m=500.0,
                time_to_tca_s=3600.0,
                maneuver_cost_ms_per_m=0.001,
            )
            assert result.option_value_ms >= result.intrinsic_value_ms - 1e-15

    def test_higher_sigma_higher_option_value(self):
        """More uncertainty should increase option value (volatility smile)."""
        from humeris.domain.conjunction_management import compute_maneuver_option

        low_vol = compute_maneuver_option(
            miss_distance_m=600.0,
            miss_distance_sigma_m=50.0,
            safe_distance_m=500.0,
            time_to_tca_s=3600.0,
            maneuver_cost_ms_per_m=0.001,
        )
        high_vol = compute_maneuver_option(
            miss_distance_m=600.0,
            miss_distance_sigma_m=300.0,
            safe_distance_m=500.0,
            time_to_tca_s=3600.0,
            maneuver_cost_ms_per_m=0.001,
        )
        assert high_vol.option_value_ms >= low_vol.option_value_ms

    def test_optimal_threshold_above_strike(self):
        """American put critical boundary S* should be >= K."""
        from humeris.domain.conjunction_management import compute_maneuver_option

        result = compute_maneuver_option(
            miss_distance_m=1000.0,
            miss_distance_sigma_m=200.0,
            safe_distance_m=500.0,
            time_to_tca_s=3600.0,
            maneuver_cost_ms_per_m=0.001,
        )
        # For American put with r=0, mu=0: beta1 = 1, threshold -> inf
        # In practice, threshold >= K
        assert result.optimal_threshold_m >= 500.0

    def test_validation_negative_miss(self):
        from humeris.domain.conjunction_management import compute_maneuver_option

        with pytest.raises(ValueError, match="miss_distance_m"):
            compute_maneuver_option(
                miss_distance_m=-100.0,
                miss_distance_sigma_m=50.0,
                safe_distance_m=500.0,
                time_to_tca_s=3600.0,
                maneuver_cost_ms_per_m=0.001,
            )

    def test_validation_zero_sigma(self):
        from humeris.domain.conjunction_management import compute_maneuver_option

        with pytest.raises(ValueError, match="miss_distance_sigma_m"):
            compute_maneuver_option(
                miss_distance_m=1000.0,
                miss_distance_sigma_m=0.0,
                safe_distance_m=500.0,
                time_to_tca_s=3600.0,
                maneuver_cost_ms_per_m=0.001,
            )

    def test_validation_negative_time(self):
        from humeris.domain.conjunction_management import compute_maneuver_option

        with pytest.raises(ValueError, match="time_to_tca_s"):
            compute_maneuver_option(
                miss_distance_m=1000.0,
                miss_distance_sigma_m=200.0,
                safe_distance_m=500.0,
                time_to_tca_s=-100.0,
                maneuver_cost_ms_per_m=0.001,
            )


# ═══════════════════════════════════════════════════════════════════
# P22: Bayesian Model Selection for Force Models
# ═══════════════════════════════════════════════════════════════════


class TestForceModelRankingDataclass:
    """Verify ForceModelRanking is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.orbit_determination import ForceModelRanking

        result = ForceModelRanking(
            rankings=(("two-body", -100.0),),
            best_model_name="two-body",
            bayes_factor_best_vs_full=1.0,
            computation_savings_percent=50.0,
        )
        with pytest.raises(AttributeError):
            result.best_model_name = "other"


class TestForceModelSelection:
    """Test select_force_model."""

    def test_basic_ranking(self):
        from humeris.domain.orbit_determination import (
            ForceModelCandidate,
            select_force_model,
        )

        candidates = [
            ForceModelCandidate(name="two-body", n_params=0, rss=100.0),
            ForceModelCandidate(name="J2", n_params=1, rss=50.0),
            ForceModelCandidate(name="J2+drag", n_params=3, rss=45.0),
        ]
        result = select_force_model(candidates, n_observations=100)

        assert len(result.rankings) == 3
        # Best model is first in rankings
        assert result.rankings[0][0] == result.best_model_name

    def test_simpler_model_preferred_when_fit_similar(self):
        """BIC penalizes complexity: same RSS but fewer params should win."""
        from humeris.domain.orbit_determination import (
            ForceModelCandidate,
            select_force_model,
        )

        candidates = [
            ForceModelCandidate(name="simple", n_params=1, rss=50.0),
            ForceModelCandidate(name="complex", n_params=10, rss=49.0),
        ]
        result = select_force_model(candidates, n_observations=100)
        # Nearly identical RSS but simple has far fewer params
        assert result.best_model_name == "simple"

    def test_complex_model_wins_with_much_better_fit(self):
        """When complex model dramatically improves fit, it should win."""
        from humeris.domain.orbit_determination import (
            ForceModelCandidate,
            select_force_model,
        )

        candidates = [
            ForceModelCandidate(name="simple", n_params=1, rss=1000.0),
            ForceModelCandidate(name="complex", n_params=5, rss=1.0),
        ]
        result = select_force_model(candidates, n_observations=100)
        assert result.best_model_name == "complex"

    def test_bayes_factor_positive(self):
        from humeris.domain.orbit_determination import (
            ForceModelCandidate,
            select_force_model,
        )

        candidates = [
            ForceModelCandidate(name="A", n_params=1, rss=50.0),
            ForceModelCandidate(name="B", n_params=3, rss=40.0),
        ]
        result = select_force_model(candidates, n_observations=50)
        assert result.bayes_factor_best_vs_full > 0

    def test_computation_savings(self):
        from humeris.domain.orbit_determination import (
            ForceModelCandidate,
            select_force_model,
        )

        candidates = [
            ForceModelCandidate(name="minimal", n_params=2, rss=50.0),
            ForceModelCandidate(name="full", n_params=10, rss=48.0),
        ]
        result = select_force_model(candidates, n_observations=100)
        if result.best_model_name == "minimal":
            assert result.computation_savings_percent == pytest.approx(80.0)

    def test_single_candidate(self):
        from humeris.domain.orbit_determination import (
            ForceModelCandidate,
            select_force_model,
        )

        candidates = [
            ForceModelCandidate(name="only", n_params=2, rss=50.0),
        ]
        result = select_force_model(candidates, n_observations=100)
        assert result.best_model_name == "only"
        assert result.bayes_factor_best_vs_full == pytest.approx(1.0)

    def test_rankings_sorted_by_bic(self):
        from humeris.domain.orbit_determination import (
            ForceModelCandidate,
            select_force_model,
        )

        candidates = [
            ForceModelCandidate(name="A", n_params=1, rss=100.0),
            ForceModelCandidate(name="B", n_params=2, rss=50.0),
            ForceModelCandidate(name="C", n_params=5, rss=30.0),
        ]
        result = select_force_model(candidates, n_observations=200)
        bics = [bic for _, bic in result.rankings]
        assert bics == sorted(bics)

    def test_validation_empty_candidates(self):
        from humeris.domain.orbit_determination import select_force_model

        with pytest.raises(ValueError, match="candidates"):
            select_force_model([], n_observations=100)

    def test_validation_too_few_observations(self):
        from humeris.domain.orbit_determination import (
            ForceModelCandidate,
            select_force_model,
        )

        with pytest.raises(ValueError, match="n_observations"):
            select_force_model(
                [ForceModelCandidate(name="A", n_params=1, rss=10.0)],
                n_observations=1,
            )


# ═══════════════════════════════════════════════════════════════════
# P23: Particle Filter for Non-Gaussian OD
# ═══════════════════════════════════════════════════════════════════


class TestParticleFilterResultDataclass:
    """Verify ParticleFilterResult is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.orbit_determination import ParticleFilterResult

        result = ParticleFilterResult(
            estimates=(),
            final_state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            final_covariance=(),
            rms_residual=0.0,
            mean_effective_particles=100.0,
            resampling_count=0,
        )
        with pytest.raises(AttributeError):
            result.rms_residual = 1.0


def _generate_leo_observations(n_obs=10, dt_s=60.0, noise_std=100.0):
    """Generate synthetic LEO observations for particle filter testing."""
    from humeris.domain.orbit_determination import ODObservation

    r = R_EARTH + 500_000.0
    v = math.sqrt(MU / r)
    state = [r, 0.0, 0.0, 0.0, v, 0.0]

    obs_list = []
    t = EPOCH
    for i in range(n_obs):
        pos = (state[0], state[1], state[2])
        obs_list.append(ODObservation(
            time=t,
            position_m=pos,
            noise_std_m=noise_std,
        ))
        # Simple two-body step for ground truth
        r_mag = math.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
        acc = -MU / r_mag**3
        state[0] += state[3] * dt_s
        state[1] += state[4] * dt_s
        state[2] += state[5] * dt_s
        state[3] += acc * state[0] * dt_s
        state[4] += acc * state[1] * dt_s
        state[5] += acc * state[2] * dt_s
        t += timedelta(seconds=dt_s)

    initial_state = (r, 0.0, 0.0, 0.0, v, 0.0)
    return obs_list, initial_state


class TestParticleFilter:
    """Test run_particle_filter."""

    def test_basic_call(self):
        from humeris.domain.orbit_determination import run_particle_filter

        obs, init_state = _generate_leo_observations(n_obs=5, dt_s=60.0)
        init_cov = [[0.0] * 6 for _ in range(6)]
        for i in range(3):
            init_cov[i][i] = 1000.0 ** 2  # 1 km position
        for i in range(3, 6):
            init_cov[i][i] = 10.0 ** 2    # 10 m/s velocity

        result = run_particle_filter(
            observations=obs,
            initial_state=init_state,
            initial_covariance=init_cov,
            n_particles=100,
        )
        assert len(result.estimates) == 5
        assert len(result.final_state) == 6
        assert result.rms_residual >= 0

    def test_estimates_have_correct_times(self):
        from humeris.domain.orbit_determination import run_particle_filter

        obs, init_state = _generate_leo_observations(n_obs=5, dt_s=60.0)
        init_cov = [[0.0] * 6 for _ in range(6)]
        for i in range(3):
            init_cov[i][i] = 1000.0 ** 2
        for i in range(3, 6):
            init_cov[i][i] = 10.0 ** 2

        result = run_particle_filter(
            observations=obs,
            initial_state=init_state,
            initial_covariance=init_cov,
            n_particles=50,
        )
        for est, ob in zip(result.estimates, obs):
            assert est.time == ob.time

    def test_rms_residual_finite(self):
        from humeris.domain.orbit_determination import run_particle_filter

        obs, init_state = _generate_leo_observations(n_obs=5)
        init_cov = [[0.0] * 6 for _ in range(6)]
        for i in range(3):
            init_cov[i][i] = 1000.0 ** 2
        for i in range(3, 6):
            init_cov[i][i] = 10.0 ** 2

        result = run_particle_filter(
            observations=obs,
            initial_state=init_state,
            initial_covariance=init_cov,
            n_particles=100,
        )
        assert math.isfinite(result.rms_residual)
        assert result.rms_residual >= 0

    def test_resampling_count_nonnegative(self):
        from humeris.domain.orbit_determination import run_particle_filter

        obs, init_state = _generate_leo_observations(n_obs=10)
        init_cov = [[0.0] * 6 for _ in range(6)]
        for i in range(3):
            init_cov[i][i] = 1000.0 ** 2
        for i in range(3, 6):
            init_cov[i][i] = 10.0 ** 2

        result = run_particle_filter(
            observations=obs,
            initial_state=init_state,
            initial_covariance=init_cov,
            n_particles=100,
        )
        assert result.resampling_count >= 0

    def test_mean_effective_particles_positive(self):
        from humeris.domain.orbit_determination import run_particle_filter

        obs, init_state = _generate_leo_observations(n_obs=5)
        init_cov = [[0.0] * 6 for _ in range(6)]
        for i in range(3):
            init_cov[i][i] = 1000.0 ** 2
        for i in range(3, 6):
            init_cov[i][i] = 10.0 ** 2

        result = run_particle_filter(
            observations=obs,
            initial_state=init_state,
            initial_covariance=init_cov,
            n_particles=100,
        )
        assert result.mean_effective_particles > 0

    def test_covariance_shape(self):
        from humeris.domain.orbit_determination import run_particle_filter

        obs, init_state = _generate_leo_observations(n_obs=3)
        init_cov = [[0.0] * 6 for _ in range(6)]
        for i in range(3):
            init_cov[i][i] = 1000.0 ** 2
        for i in range(3, 6):
            init_cov[i][i] = 10.0 ** 2

        result = run_particle_filter(
            observations=obs,
            initial_state=init_state,
            initial_covariance=init_cov,
            n_particles=50,
        )
        assert len(result.final_covariance) == 6
        for row in result.final_covariance:
            assert len(row) == 6

    def test_validation_empty_observations(self):
        from humeris.domain.orbit_determination import run_particle_filter

        with pytest.raises(ValueError, match="observations"):
            run_particle_filter(
                observations=[],
                initial_state=(0, 0, 0, 0, 0, 0),
                initial_covariance=[[0]*6]*6,
            )

    def test_validation_too_few_particles(self):
        from humeris.domain.orbit_determination import run_particle_filter

        obs, init_state = _generate_leo_observations(n_obs=2)
        with pytest.raises(ValueError, match="n_particles"):
            run_particle_filter(
                observations=obs,
                initial_state=init_state,
                initial_covariance=[[0]*6]*6,
                n_particles=1,
            )


# ═══════════════════════════════════════════════════════════════════
# P29: Compressed Sensing for Sparse OD
# ═══════════════════════════════════════════════════════════════════


class TestCompressedSensingODDataclass:
    """Verify CompressedSensingOD is a frozen dataclass."""

    def test_frozen(self):
        from humeris.domain.orbit_determination import CompressedSensingOD

        result = CompressedSensingOD(
            state_correction=(0, 0, 0, 0, 0, 0),
            sparsity_level=0,
            rms_residual=0.0,
            compression_ratio=0.0,
            ista_iterations=0,
        )
        with pytest.raises(AttributeError):
            result.sparsity_level = 5


class TestCompressedSensingOD:
    """Test compressed_sensing_od."""

    def test_basic_call(self):
        from humeris.domain.orbit_determination import (
            compressed_sensing_od,
            ODObservation,
        )

        r = R_EARTH + 500_000.0
        predicted = (r, 0.0, 0.0, 0.0, 0.0, 0.0)
        obs = [
            ODObservation(
                time=EPOCH,
                position_m=(r + 100.0, 50.0, -30.0),
                noise_std_m=100.0,
            ),
        ]
        result = compressed_sensing_od(predicted, obs)
        assert len(result.state_correction) == 6
        assert result.ista_iterations > 0

    def test_correction_reduces_residual(self):
        """Applying the correction should reduce residual vs. predicted."""
        from humeris.domain.orbit_determination import (
            compressed_sensing_od,
            ODObservation,
        )

        r = R_EARTH + 500_000.0
        predicted = (r, 0.0, 0.0, 0.0, 0.0, 0.0)
        obs_pos = (r + 500.0, 200.0, -100.0)
        obs = [
            ODObservation(time=EPOCH, position_m=obs_pos, noise_std_m=100.0),
        ]

        result = compressed_sensing_od(predicted, obs, lambda_reg=0.01)

        # Original residual
        orig_residual = math.sqrt(
            (obs_pos[0] - predicted[0])**2
            + (obs_pos[1] - predicted[1])**2
            + (obs_pos[2] - predicted[2])**2
        ) / math.sqrt(3)

        # ISTA residual should be smaller
        assert result.rms_residual < orig_residual

    def test_high_lambda_produces_sparse_solution(self):
        """High L1 penalty should produce sparser corrections."""
        from humeris.domain.orbit_determination import (
            compressed_sensing_od,
            ODObservation,
        )

        r = R_EARTH + 500_000.0
        predicted = (r, 0.0, 0.0, 0.0, 0.0, 0.0)
        obs = [
            ODObservation(
                time=EPOCH,
                position_m=(r + 100.0, 50.0, -30.0),
                noise_std_m=100.0,
            ),
            ODObservation(
                time=EPOCH + timedelta(seconds=60),
                position_m=(r + 110.0, 55.0, -25.0),
                noise_std_m=100.0,
            ),
        ]

        low_lambda = compressed_sensing_od(predicted, obs, lambda_reg=0.01)
        high_lambda = compressed_sensing_od(predicted, obs, lambda_reg=100.0)

        assert high_lambda.sparsity_level <= low_lambda.sparsity_level

    def test_zero_lambda_full_correction(self):
        """With lambda=0, ISTA becomes gradient descent (no sparsity)."""
        from humeris.domain.orbit_determination import (
            compressed_sensing_od,
            ODObservation,
        )

        r = R_EARTH + 500_000.0
        predicted = (r, 0.0, 0.0, 0.0, 0.0, 0.0)
        obs = [
            ODObservation(
                time=EPOCH,
                position_m=(r + 1000.0, 500.0, -300.0),
                noise_std_m=100.0,
            ),
        ]

        result = compressed_sensing_od(predicted, obs, lambda_reg=0.0)
        # Position corrections should be close to the residual
        assert abs(result.state_correction[0] - 1000.0) < 1.0
        assert abs(result.state_correction[1] - 500.0) < 1.0
        assert abs(result.state_correction[2] - (-300.0)) < 1.0

    def test_sparsity_level_range(self):
        from humeris.domain.orbit_determination import (
            compressed_sensing_od,
            ODObservation,
        )

        r = R_EARTH + 500_000.0
        predicted = (r, 0.0, 0.0, 0.0, 0.0, 0.0)
        obs = [
            ODObservation(
                time=EPOCH,
                position_m=(r + 100.0, 0.0, 0.0),
                noise_std_m=100.0,
            ),
        ]
        result = compressed_sensing_od(predicted, obs)
        assert 0 <= result.sparsity_level <= 6

    def test_compression_ratio_range(self):
        from humeris.domain.orbit_determination import (
            compressed_sensing_od,
            ODObservation,
        )

        r = R_EARTH + 500_000.0
        predicted = (r, 0.0, 0.0, 0.0, 0.0, 0.0)
        obs = [
            ODObservation(
                time=EPOCH,
                position_m=(r + 100.0, 50.0, 0.0),
                noise_std_m=100.0,
            ),
        ]
        result = compressed_sensing_od(predicted, obs)
        assert 0.0 <= result.compression_ratio <= 1.0

    def test_multiple_observations(self):
        from humeris.domain.orbit_determination import (
            compressed_sensing_od,
            ODObservation,
        )

        r = R_EARTH + 500_000.0
        predicted = (r, 0.0, 0.0, 0.0, 0.0, 0.0)
        obs = []
        for i in range(5):
            obs.append(ODObservation(
                time=EPOCH + timedelta(seconds=i * 60),
                position_m=(r + 100.0, 50.0 + i * 10, -30.0),
                noise_std_m=100.0,
            ))

        result = compressed_sensing_od(predicted, obs, lambda_reg=0.1)
        assert result.ista_iterations > 0
        assert math.isfinite(result.rms_residual)

    def test_validation_empty_observations(self):
        from humeris.domain.orbit_determination import compressed_sensing_od

        with pytest.raises(ValueError, match="observations"):
            compressed_sensing_od(
                predicted_state=(0, 0, 0, 0, 0, 0),
                observations=[],
            )

    def test_validation_negative_lambda(self):
        from humeris.domain.orbit_determination import (
            compressed_sensing_od,
            ODObservation,
        )

        obs = [ODObservation(time=EPOCH, position_m=(1e7, 0, 0), noise_std_m=100.0)]
        with pytest.raises(ValueError, match="lambda_reg"):
            compressed_sensing_od(
                predicted_state=(1e7, 0, 0, 0, 0, 0),
                observations=obs,
                lambda_reg=-1.0,
            )

    def test_velocity_components_sparse_with_position_only_obs(self):
        """Position-only observations: velocity corrections should be zero (sparse)."""
        from humeris.domain.orbit_determination import (
            compressed_sensing_od,
            ODObservation,
        )

        r = R_EARTH + 500_000.0
        predicted = (r, 0.0, 0.0, 0.0, 0.0, 0.0)
        obs = [
            ODObservation(
                time=EPOCH,
                position_m=(r + 500.0, 200.0, -100.0),
                noise_std_m=100.0,
            ),
        ]
        result = compressed_sensing_od(predicted, obs, lambda_reg=1.0)
        # H matrix has zeros in velocity columns, so velocity corrections
        # should be zero regardless of lambda
        assert abs(result.state_correction[3]) < 1e-10
        assert abs(result.state_correction[4]) < 1e-10
        assert abs(result.state_correction[5]) < 1e-10
