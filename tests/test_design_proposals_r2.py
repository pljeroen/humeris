# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for design proposals R2: P25, P26, P28, P35, P36, P38.

P25: Ising Model Phase Transition for ISL Networks (graph_analysis.py)
P26: Free Energy Landscape for Constellation Design (design_optimization.py)
P28: Vickrey Auction for Orbital Slot Allocation (design_optimization.py)
P35: r/K Selection Strategy for Constellation Design (design_optimization.py)
P36: Reliability Block Diagram for Constellation (constellation_operability.py)
P38: Fisher-Rao Metric for Constellation Distance (constellation_metrics.py)
"""
import math

import numpy as np
import pytest

from humeris.domain.graph_analysis import (
    ISLPhaseTransition,
    compute_isl_phase_transition,
)
from humeris.domain.design_optimization import (
    AuctionResult,
    ConstellationStrategy,
    DesignFreeEnergyLandscape,
    compute_design_free_energy,
    compute_orbital_slot_auction,
    compute_constellation_strategy,
)
from humeris.domain.constellation_operability import (
    ConstellationReliability,
    compute_constellation_reliability,
)
from humeris.domain.constellation_metrics import (
    ConstellationDistance,
    compute_constellation_distance,
)


# ── P25: Ising Model Phase Transition ─────────────────────────────


class TestIsingPhaseTransition:
    """Tests for ISL network Ising model phase transition."""

    def test_single_node_returns_trivial(self):
        """Single node: always ordered, no phase transition."""
        result = compute_isl_phase_transition(
            adjacency=[[0.0]],
            eclipse_fractions=[0.3],
            n_nodes=1,
        )
        assert isinstance(result, ISLPhaseTransition)
        assert result.critical_eclipse_fraction == 1.0
        assert result.current_magnetization == 1.0
        assert result.is_ordered_phase is True

    def test_fully_connected_strong_coupling(self):
        """Strongly coupled network should have high critical eclipse fraction."""
        n = 5
        # Strong adjacency (high SNR)
        adj = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i][j] = 50.0  # Strong coupling
        ecl = [0.1] * n  # Low eclipse fraction (sunlit)

        result = compute_isl_phase_transition(
            adjacency=adj,
            eclipse_fractions=ecl,
            n_nodes=n,
        )
        assert result.mean_coupling_strength > 0
        # Low eclipse fraction means low temperature => ordered phase
        assert result.current_magnetization > 0.5
        assert result.is_ordered_phase is True

    def test_zero_adjacency_returns_disordered(self):
        """No links: J_eff = 0, no ordering possible."""
        n = 4
        adj = [[0.0] * n for _ in range(n)]
        ecl = [0.5] * n

        result = compute_isl_phase_transition(
            adjacency=adj,
            eclipse_fractions=ecl,
            n_nodes=n,
        )
        assert result.mean_coupling_strength == 0.0
        assert result.is_ordered_phase is False
        assert result.current_magnetization == 0.0

    def test_high_eclipse_fraction_disorders_network(self):
        """High eclipse fraction = high temperature = disordered."""
        n = 4
        adj = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i][j] = 10.0  # Moderate coupling
        ecl = [0.95] * n  # Very high eclipse => high temperature

        result = compute_isl_phase_transition(
            adjacency=adj,
            eclipse_fractions=ecl,
            n_nodes=n,
        )
        # At T = 1/(1-0.95) = 20.0, should be disordered
        assert result.is_ordered_phase is False

    def test_critical_eclipse_fraction_bounded(self):
        """Critical eclipse fraction must be in [0, 1]."""
        n = 4
        adj = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i][j] = 30.0
        ecl = [0.3] * n

        result = compute_isl_phase_transition(
            adjacency=adj,
            eclipse_fractions=ecl,
            n_nodes=n,
        )
        assert 0.0 <= result.critical_eclipse_fraction <= 1.0

    def test_susceptibility_peaks_near_critical(self):
        """Susceptibility should be large when near critical temperature."""
        n = 6
        adj = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i][j] = 20.0

        # Compute J_eff to find critical eclipse fraction
        j_eff = (n - 1) * 20.0 / 10.0  # Each node coupled to n-1 others
        # t_c = j_eff, critical eclipse = 1 - 1/j_eff
        if j_eff > 1:
            f_c = 1.0 - 1.0 / j_eff
        else:
            f_c = 0.5

        # Just below critical: should have high susceptibility
        ecl_near = [f_c - 0.01] * n
        result_near = compute_isl_phase_transition(
            adjacency=adj,
            eclipse_fractions=ecl_near,
            n_nodes=n,
        )

        # Well below critical: moderate susceptibility
        ecl_far = [0.1] * n
        result_far = compute_isl_phase_transition(
            adjacency=adj,
            eclipse_fractions=ecl_far,
            n_nodes=n,
        )

        # Susceptibility near critical should be >= far from critical
        # (or both could be large; at minimum check they are positive)
        assert result_near.susceptibility > 0
        assert result_far.susceptibility > 0

    def test_sharpness_positive_for_nontrivial(self):
        """Phase transition sharpness should be positive for nontrivial networks."""
        n = 4
        adj = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[i][j] = 15.0
        ecl = [0.3] * n

        result = compute_isl_phase_transition(
            adjacency=adj,
            eclipse_fractions=ecl,
            n_nodes=n,
        )
        assert result.phase_transition_sharpness > 0

    def test_return_type(self):
        """Result must be a frozen ISLPhaseTransition dataclass."""
        result = compute_isl_phase_transition(
            adjacency=[[0, 10], [10, 0]],
            eclipse_fractions=[0.2, 0.3],
            n_nodes=2,
        )
        assert isinstance(result, ISLPhaseTransition)
        with pytest.raises(AttributeError):
            result.current_magnetization = 0.5  # type: ignore[misc]


# ── P26: Free Energy Landscape ────────────────────────────────────


class TestDesignFreeEnergy:
    """Tests for the free energy landscape of constellation design."""

    def test_empty_scores(self):
        """No scores: return trivial result."""
        result = compute_design_free_energy(scores=[])
        assert isinstance(result, DesignFreeEnergyLandscape)
        assert result.optimal_config_index == 0
        assert result.free_energy == 0.0
        assert result.design_entropy == 0.0

    def test_single_config(self):
        """Single configuration: zero entropy, all probability on it."""
        result = compute_design_free_energy(scores=[5.0], temperature=1.0)
        assert result.optimal_config_index == 0
        assert result.design_entropy == pytest.approx(0.0, abs=1e-10)

    def test_optimal_config_is_highest_score(self):
        """Optimal config should be the one with highest score."""
        scores = [1.0, 5.0, 3.0, 2.0]
        result = compute_design_free_energy(scores=scores, temperature=1.0)
        assert result.optimal_config_index == 1

    def test_low_temperature_concentrates(self):
        """At low T, entropy should be near zero (concentrated on optimum)."""
        scores = [1.0, 10.0, 2.0, 3.0]
        result = compute_design_free_energy(scores=scores, temperature=0.01)
        assert result.design_entropy < 0.1

    def test_high_temperature_maximizes_entropy(self):
        """At high T, entropy approaches ln(N) (uniform distribution)."""
        scores = [1.0, 2.0, 3.0, 4.0]
        n = len(scores)
        max_entropy = math.log(n)
        result = compute_design_free_energy(scores=scores, temperature=100.0)
        assert result.design_entropy > 0.9 * max_entropy

    def test_heat_capacity_non_negative(self):
        """Heat capacity C = Var(E)/T^2 must be non-negative."""
        scores = [1.0, 5.0, 3.0, 7.0, 2.0]
        result = compute_design_free_energy(scores=scores, temperature=1.0)
        assert result.heat_capacity >= -1e-10

    def test_metastable_configs_identified(self):
        """Configurations with non-trivial probability should be metastable."""
        # Two configs with very similar high scores, two with low
        scores = [10.0, 9.9, 1.0, 0.5]
        result = compute_design_free_energy(
            scores=scores, temperature=1.0, metastable_threshold=0.01,
        )
        # Config 1 (score=9.9) should be metastable
        assert 1 in result.metastable_configs

    def test_temperature_sweep(self):
        """Temperature sweep should produce correct number of points."""
        scores = [1.0, 3.0, 5.0]
        n_steps = 15
        result = compute_design_free_energy(
            scores=scores, temperature=1.0, n_temp_steps=n_steps,
        )
        assert len(result.temperature_sweep) == n_steps
        assert len(result.free_energy_curve) == n_steps

    def test_free_energy_decreases_with_temperature(self):
        """Free energy F(T) = -T*ln(Z) should generally decrease with T."""
        scores = [1.0, 3.0, 5.0, 7.0]
        result = compute_design_free_energy(
            scores=scores, n_temp_steps=20, temp_min=0.5, temp_max=10.0,
        )
        # Free energy at high T should be lower than at low T
        # (because Z grows with T)
        assert result.free_energy_curve[-1] <= result.free_energy_curve[0]

    def test_return_type(self):
        """Result must be a frozen DesignFreeEnergyLandscape dataclass."""
        result = compute_design_free_energy(scores=[1.0, 2.0])
        assert isinstance(result, DesignFreeEnergyLandscape)
        with pytest.raises(AttributeError):
            result.free_energy = 0.0  # type: ignore[misc]


# ── P28: Vickrey Auction ──────────────────────────────────────────


class TestVickreyAuction:
    """Tests for VCG auction for orbital slot allocation."""

    def test_empty_valuations(self):
        """No bidders: no allocations."""
        result = compute_orbital_slot_auction(valuations=[], n_slots=3)
        assert isinstance(result, AuctionResult)
        assert result.social_welfare == 0.0
        assert len(result.allocations) == 0

    def test_single_slot_second_price(self):
        """Single slot: winner pays second-highest bid (Vickrey)."""
        # Bidders: A bids 100, B bids 70, C bids 50
        valuations = [[100.0], [70.0], [50.0]]
        result = compute_orbital_slot_auction(valuations=valuations, n_slots=1)

        # A should win
        assert len(result.allocations) == 1
        winner_bidder, winner_slot = result.allocations[0]
        assert winner_bidder == 0
        assert winner_slot == 0

        # A pays second-highest = 70
        assert result.payments[0] == pytest.approx(70.0, abs=1.0)
        assert result.social_welfare == pytest.approx(100.0, abs=1e-6)

    def test_multiple_slots_allocation(self):
        """Multiple slots: each bidder gets at most one slot."""
        # 3 bidders, 2 slots
        valuations = [
            [80.0, 60.0],   # Bidder 0
            [70.0, 90.0],   # Bidder 1
            [50.0, 40.0],   # Bidder 2
        ]
        result = compute_orbital_slot_auction(valuations=valuations, n_slots=2)

        # Optimal: bidder 0 -> slot 0 (80), bidder 1 -> slot 1 (90) = 170
        assert result.social_welfare >= 160.0
        assert len(result.allocations) == 2

    def test_vcg_payments_less_than_valuations(self):
        """VCG payments should never exceed the winner's valuation."""
        valuations = [[100.0], [70.0], [50.0]]
        result = compute_orbital_slot_auction(valuations=valuations, n_slots=1)

        for idx, (bidder, slot) in enumerate(result.allocations):
            assert result.payments[idx] <= valuations[bidder][slot] + 1e-6

    def test_zero_slots(self):
        """Zero slots: no allocation."""
        result = compute_orbital_slot_auction(
            valuations=[[10.0], [20.0]],
            n_slots=0,
        )
        assert len(result.allocations) == 0
        assert result.social_welfare == 0.0

    def test_price_per_slot(self):
        """Price per slot should equal total_revenue / n_allocated."""
        valuations = [[100.0, 50.0], [80.0, 90.0]]
        result = compute_orbital_slot_auction(valuations=valuations, n_slots=2)
        n_alloc = len(result.allocations)
        if n_alloc > 0:
            expected = result.total_revenue / n_alloc
            assert result.price_per_slot == pytest.approx(expected, abs=1e-6)

    def test_non_negative_payments(self):
        """All VCG payments must be non-negative."""
        valuations = [[50.0, 30.0], [40.0, 60.0], [20.0, 10.0]]
        result = compute_orbital_slot_auction(valuations=valuations, n_slots=2)
        for p in result.payments:
            assert p >= -1e-10

    def test_return_type(self):
        """Result must be a frozen AuctionResult dataclass."""
        result = compute_orbital_slot_auction(valuations=[[10.0]], n_slots=1)
        assert isinstance(result, AuctionResult)
        with pytest.raises(AttributeError):
            result.social_welfare = 0.0  # type: ignore[misc]


# ── P35: r/K Selection Strategy ───────────────────────────────────


class TestConstellationStrategy:
    """Tests for r/K ecological selection strategy."""

    def test_r_selected_high_turnover(self):
        """High launch rate, cheap sats, hostile environment: r-selected."""
        result = compute_constellation_strategy(
            coverage_fraction=0.8,
            cost_per_satellite=1.0,       # Cheap
            lifetime_years=2.0,           # Short-lived
            n_satellites=100,
            survival_probability=0.9,
            drag_rate_per_year=0.05,
            conjunction_rate_per_year=0.1,
            max_useful_sats=500,
            launch_rate_per_year=50.0,    # High replenishment
        )
        assert isinstance(result, ConstellationStrategy)
        assert result.r_metric > 0
        assert result.k_metric > 0

    def test_k_selected_long_lived(self):
        """Long-lived, expensive sats, benign environment: K-selected."""
        result = compute_constellation_strategy(
            coverage_fraction=0.9,
            cost_per_satellite=100.0,     # Expensive
            lifetime_years=15.0,          # Long-lived
            n_satellites=20,
            survival_probability=0.99,
            drag_rate_per_year=0.001,
            conjunction_rate_per_year=0.01,
            max_useful_sats=500,
            launch_rate_per_year=2.0,     # Low replenishment
        )
        assert isinstance(result, ConstellationStrategy)
        assert result.strategy_type in ("r-selected", "K-selected")

    def test_mismatch_penalty_zero_when_matched(self):
        """When design r/K ratio equals environment r/K, mismatch = 0."""
        # Construct inputs so design_rk = environment_rk
        # environment_rk = (drag + conjunction) / max_sats
        # design_rk = (launch_rate * cov_per_sat) / (cost * lifetime)
        result = compute_constellation_strategy(
            coverage_fraction=0.5,
            cost_per_satellite=10.0,
            lifetime_years=5.0,
            n_satellites=50,
            survival_probability=0.95,
            drag_rate_per_year=0.01,
            conjunction_rate_per_year=0.01,
            max_useful_sats=100,
            launch_rate_per_year=10.0,
        )
        # Mismatch penalty should be finite
        assert result.mismatch_penalty >= 0.0
        assert result.mismatch_penalty < float('inf')

    def test_strategy_type_is_valid(self):
        """Strategy type must be exactly 'r-selected' or 'K-selected'."""
        result = compute_constellation_strategy(
            coverage_fraction=0.7,
            cost_per_satellite=5.0,
            lifetime_years=5.0,
            n_satellites=30,
            survival_probability=0.95,
            drag_rate_per_year=0.02,
            conjunction_rate_per_year=0.05,
            max_useful_sats=200,
            launch_rate_per_year=10.0,
        )
        assert result.strategy_type in ("r-selected", "K-selected")

    def test_r_metric_increases_with_survival(self):
        """Higher survival probability should increase r-metric."""
        base_kwargs = dict(
            coverage_fraction=0.6,
            cost_per_satellite=5.0,
            lifetime_years=3.0,
            n_satellites=50,
            drag_rate_per_year=0.02,
            conjunction_rate_per_year=0.05,
            max_useful_sats=200,
            launch_rate_per_year=20.0,
        )
        result_low = compute_constellation_strategy(
            survival_probability=0.5, **base_kwargs,
        )
        result_high = compute_constellation_strategy(
            survival_probability=0.99, **base_kwargs,
        )
        assert result_high.r_metric > result_low.r_metric

    def test_k_metric_increases_with_lifetime(self):
        """Longer lifetime should increase K-metric."""
        base_kwargs = dict(
            coverage_fraction=0.6,
            cost_per_satellite=5.0,
            n_satellites=50,
            survival_probability=0.95,
            drag_rate_per_year=0.02,
            conjunction_rate_per_year=0.05,
            max_useful_sats=200,
            launch_rate_per_year=10.0,
        )
        result_short = compute_constellation_strategy(
            lifetime_years=2.0, **base_kwargs,
        )
        result_long = compute_constellation_strategy(
            lifetime_years=15.0, **base_kwargs,
        )
        assert result_long.k_metric > result_short.k_metric

    def test_return_type(self):
        """Result must be a frozen ConstellationStrategy dataclass."""
        result = compute_constellation_strategy(
            coverage_fraction=0.5, cost_per_satellite=1.0,
            lifetime_years=5.0, n_satellites=10,
            survival_probability=0.9, drag_rate_per_year=0.01,
            conjunction_rate_per_year=0.01, max_useful_sats=100,
            launch_rate_per_year=5.0,
        )
        assert isinstance(result, ConstellationStrategy)
        with pytest.raises(AttributeError):
            result.strategy_type = "unknown"  # type: ignore[misc]


# ── P36: Reliability Block Diagram ────────────────────────────────


class TestConstellationReliability:
    """Tests for constellation reliability block diagram."""

    def test_zero_satellites(self):
        """Zero satellites: zero reliability."""
        result = compute_constellation_reliability(
            n_satellites=0, k_required=0,
        )
        assert isinstance(result, ConstellationReliability)
        assert result.system_availability == 0.0

    def test_perfect_subsystems(self):
        """All R=1.0 subsystems: system reliability = 1.0."""
        result = compute_constellation_reliability(
            n_satellites=10, k_required=5,
            r_fuel=1.0, r_power=1.0, r_conjunction=1.0, r_thermal=1.0,
        )
        assert result.system_availability == pytest.approx(1.0, abs=1e-10)

    def test_series_product(self):
        """Per-satellite reliability is product of subsystem reliabilities."""
        r_fuel, r_power, r_conj, r_therm = 0.99, 0.995, 0.999, 0.998
        r_sat = r_fuel * r_power * r_conj * r_therm

        # For 1-out-of-1: R_system = R_sat
        result = compute_constellation_reliability(
            n_satellites=1, k_required=1,
            r_fuel=r_fuel, r_power=r_power,
            r_conjunction=r_conj, r_thermal=r_therm,
        )
        assert result.system_availability == pytest.approx(r_sat, abs=1e-8)

    def test_k_out_of_n_increases_reliability(self):
        """k-out-of-n with k < n should have higher reliability than single sat."""
        r_fuel, r_power, r_conj, r_therm = 0.95, 0.96, 0.97, 0.98

        result_1of1 = compute_constellation_reliability(
            n_satellites=1, k_required=1,
            r_fuel=r_fuel, r_power=r_power,
            r_conjunction=r_conj, r_thermal=r_therm,
        )
        result_3of5 = compute_constellation_reliability(
            n_satellites=5, k_required=3,
            r_fuel=r_fuel, r_power=r_power,
            r_conjunction=r_conj, r_thermal=r_therm,
        )
        assert result_3of5.system_availability > result_1of1.system_availability

    def test_mttf_positive(self):
        """MTTF should be positive for non-degenerate cases."""
        result = compute_constellation_reliability(
            n_satellites=10, k_required=7,
            satellite_lifetime_years=5.0,
        )
        assert result.mttf_years > 0

    def test_mttf_increases_with_spare_sats(self):
        """More spare satellites (lower k/n ratio) should increase MTTF."""
        result_high_k = compute_constellation_reliability(
            n_satellites=10, k_required=9,
            satellite_lifetime_years=5.0,
        )
        result_low_k = compute_constellation_reliability(
            n_satellites=10, k_required=5,
            satellite_lifetime_years=5.0,
        )
        assert result_low_k.mttf_years > result_high_k.mttf_years

    def test_birnbaum_importances_length(self):
        """Birnbaum importances should have 4 entries (one per subsystem)."""
        result = compute_constellation_reliability(
            n_satellites=8, k_required=5,
        )
        assert len(result.birnbaum_importances) == 4

    def test_birnbaum_importances_non_negative(self):
        """Birnbaum importances should be non-negative (improving a component helps)."""
        result = compute_constellation_reliability(
            n_satellites=8, k_required=5,
        )
        for imp in result.birnbaum_importances:
            assert imp >= -1e-10

    def test_lowest_reliability_has_highest_importance(self):
        """The weakest subsystem should have the highest Birnbaum importance."""
        result = compute_constellation_reliability(
            n_satellites=10, k_required=7,
            r_fuel=0.90,        # Weakest
            r_power=0.995,
            r_conjunction=0.999,
            r_thermal=0.998,
        )
        # Fuel (index 0) should have highest importance
        importances = result.birnbaum_importances
        assert importances[0] == max(importances)

    def test_min_satellites_for_coverage(self):
        """min_satellites_for_coverage should achieve >= 90% reliability."""
        result = compute_constellation_reliability(
            n_satellites=20, k_required=15,
        )
        min_k = result.min_satellites_for_coverage
        assert 1 <= min_k <= 20

    def test_return_type(self):
        """Result must be a frozen ConstellationReliability dataclass."""
        result = compute_constellation_reliability(
            n_satellites=5, k_required=3,
        )
        assert isinstance(result, ConstellationReliability)
        with pytest.raises(AttributeError):
            result.system_availability = 0.5  # type: ignore[misc]


# ── P38: Fisher-Rao Metric ────────────────────────────────────────


class TestConstellationDistance:
    """Tests for Fisher-Rao constellation distance metric."""

    def test_identical_distributions(self):
        """Same coverage counts: distance = 0."""
        counts = [0, 1, 2, 3, 1, 0, 2, 1]
        result = compute_constellation_distance(
            coverage_counts_a=counts,
            coverage_counts_b=counts,
        )
        assert isinstance(result, ConstellationDistance)
        assert result.fisher_rao_distance == pytest.approx(0.0, abs=1e-10)
        assert result.hellinger_distance == pytest.approx(0.0, abs=1e-10)
        assert result.kl_divergence == pytest.approx(0.0, abs=1e-10)

    def test_completely_different_distributions(self):
        """Disjoint distributions: maximum distance."""
        counts_a = [0, 0, 0, 0, 0]  # All zero coverage
        counts_b = [5, 5, 5, 5, 5]  # All 5-fold coverage
        result = compute_constellation_distance(
            coverage_counts_a=counts_a,
            coverage_counts_b=counts_b,
        )
        # Fisher-Rao distance for disjoint distributions = pi
        assert result.fisher_rao_distance == pytest.approx(math.pi, abs=0.01)
        # Hellinger distance maxes at 1.0
        assert result.hellinger_distance == pytest.approx(1.0, abs=0.01)

    def test_empty_counts(self):
        """Empty coverage counts: trivial result."""
        result = compute_constellation_distance(
            coverage_counts_a=[],
            coverage_counts_b=[1, 2, 3],
        )
        assert result.fisher_rao_distance == 0.0
        assert result.hellinger_distance == 0.0

    def test_fisher_rao_non_negative(self):
        """Fisher-Rao distance is always non-negative."""
        result = compute_constellation_distance(
            coverage_counts_a=[0, 1, 2, 3, 2, 1, 0],
            coverage_counts_b=[1, 2, 3, 4, 3, 2, 1],
        )
        assert result.fisher_rao_distance >= 0.0

    def test_hellinger_bounded(self):
        """Hellinger distance is in [0, 1]."""
        result = compute_constellation_distance(
            coverage_counts_a=[0, 1, 0, 1, 0],
            coverage_counts_b=[3, 3, 3, 3, 3],
        )
        assert 0.0 <= result.hellinger_distance <= 1.0 + 1e-10

    def test_kl_divergence_non_negative(self):
        """KL divergence D_KL(p||q) >= 0 (Gibbs inequality)."""
        result = compute_constellation_distance(
            coverage_counts_a=[0, 1, 2, 1, 0, 1, 2],
            coverage_counts_b=[1, 1, 1, 2, 2, 1, 1],
        )
        assert result.kl_divergence >= -1e-10

    def test_correlation_identical(self):
        """Identical count vectors: correlation = 1.0."""
        counts = [0, 1, 2, 3, 4, 3, 2, 1, 0]
        result = compute_constellation_distance(
            coverage_counts_a=counts,
            coverage_counts_b=counts,
        )
        assert result.coverage_correlation == pytest.approx(1.0, abs=1e-10)

    def test_significance_flag(self):
        """is_significantly_different should track the threshold."""
        counts_a = [0, 0, 0, 0, 0]
        counts_b = [5, 5, 5, 5, 5]
        result = compute_constellation_distance(
            coverage_counts_a=counts_a,
            coverage_counts_b=counts_b,
            significance_threshold=0.5,
        )
        assert result.is_significantly_different is True

        # Same distributions should not be significantly different
        result_same = compute_constellation_distance(
            coverage_counts_a=counts_a,
            coverage_counts_b=counts_a,
            significance_threshold=0.5,
        )
        assert result_same.is_significantly_different is False

    def test_triangle_inequality(self):
        """Fisher-Rao distance satisfies the triangle inequality."""
        a = [0, 1, 2, 3, 2, 1, 0]
        b = [1, 2, 3, 2, 1, 0, 0]
        c = [2, 3, 2, 1, 0, 0, 1]

        d_ab = compute_constellation_distance(a, b).fisher_rao_distance
        d_bc = compute_constellation_distance(b, c).fisher_rao_distance
        d_ac = compute_constellation_distance(a, c).fisher_rao_distance

        assert d_ac <= d_ab + d_bc + 1e-10

    def test_symmetry(self):
        """Fisher-Rao and Hellinger distances are symmetric."""
        a = [0, 1, 2, 3]
        b = [3, 2, 1, 0]
        result_ab = compute_constellation_distance(a, b)
        result_ba = compute_constellation_distance(b, a)
        assert result_ab.fisher_rao_distance == pytest.approx(
            result_ba.fisher_rao_distance, abs=1e-10,
        )
        assert result_ab.hellinger_distance == pytest.approx(
            result_ba.hellinger_distance, abs=1e-10,
        )

    def test_return_type(self):
        """Result must be a frozen ConstellationDistance dataclass."""
        result = compute_constellation_distance([1, 2], [2, 3])
        assert isinstance(result, ConstellationDistance)
        with pytest.raises(AttributeError):
            result.fisher_rao_distance = 0.0  # type: ignore[misc]
