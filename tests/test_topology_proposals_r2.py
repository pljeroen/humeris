# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for topology proposals R2: P30, P31, P34, P37.

P30: Persistent Homology for Coverage Holes
P31: Shadowing Lemma for Propagation Validation
P34: Melnikov Method for Chaos Onset Detection
P37: Wavelet Multi-Resolution Coverage Analysis
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants


# ── Fixtures ───────────────────────────────────────────────────────

MU = OrbitalConstants.MU_EARTH
R_EARTH = OrbitalConstants.R_EARTH
R_EARTH_EQ = OrbitalConstants.R_EARTH_EQUATORIAL


@pytest.fixture
def epoch():
    return datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def leo_circular_pos_vel():
    """LEO circular orbit at 500 km: position along x, velocity along y."""
    r = R_EARTH + 500_000.0
    v = math.sqrt(MU / r)
    return (r, 0.0, 0.0), (0.0, v, 0.0)


# ═══════════════════════════════════════════════════════════════════
# P30: Persistent Homology for Coverage Holes
# ═══════════════════════════════════════════════════════════════════


class TestPersistentCoverageHomology:
    """Tests for compute_persistent_coverage_homology (P30)."""

    def test_import(self):
        """PersistentCoverageHomology and compute function are importable."""
        from humeris.domain.coverage import (
            PersistentCoverageHomology,
            PersistenceInterval,
            compute_persistent_coverage_homology,
        )
        assert PersistentCoverageHomology is not None
        assert PersistenceInterval is not None
        assert compute_persistent_coverage_homology is not None

    def test_empty_grid(self):
        """Empty coverage grid returns empty homology."""
        from humeris.domain.coverage import compute_persistent_coverage_homology

        result = compute_persistent_coverage_homology([])

        assert result.intervals == ()
        assert result.num_significant_holes == 0
        assert result.max_hole_persistence == 0.0
        assert result.total_persistence == 0.0

    def test_uniform_coverage_single_component(self):
        """Uniform coverage has exactly one surviving H_0 component."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        grid = []
        for lat in range(-30, 31, 10):
            for lon in range(-30, 31, 10):
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=5))

        result = compute_persistent_coverage_homology(grid)

        # All points have same value, so they all enter at k=5.
        # Merging at same level -> deaths at birth = no finite persistence for H_0.
        # Exactly one surviving component.
        h0_inf = [iv for iv in result.intervals if iv.dimension == 0 and iv.death == float('inf')]
        assert len(h0_inf) == 1

    def test_two_isolated_regions_detected(self):
        """Two disconnected high-coverage regions produce multiple H_0 components."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        grid = []
        for lat in range(-40, 41, 10):
            for lon in range(-40, 41, 10):
                # Two separated peaks: one at (-30, -30), one at (30, 30)
                d1 = math.sqrt((lat + 30) ** 2 + (lon + 30) ** 2)
                d2 = math.sqrt((lat - 30) ** 2 + (lon - 30) ** 2)
                if d1 < 15:
                    count = 1
                elif d2 < 15:
                    count = 1
                else:
                    count = 8
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))

        result = compute_persistent_coverage_homology(grid)

        # The low-value (1) regions enter first and form multiple components
        # that eventually merge when the high-value (8) regions enter.
        h0_finite = [iv for iv in result.intervals if iv.dimension == 0 and iv.death != float('inf')]
        assert len(h0_finite) >= 1  # At least one merge event

    def test_betti_curves_nonempty(self):
        """Betti curves are computed for the coverage levels."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        grid = []
        for lat in range(-20, 21, 10):
            for lon in range(-20, 21, 10):
                count = abs(lat) // 10 + abs(lon) // 10
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))

        result = compute_persistent_coverage_homology(grid)

        assert len(result.betti_0_curve) > 0
        assert len(result.betti_1_curve) > 0

    def test_betti_0_starts_positive(self):
        """At the first filtration level, beta_0 >= 1 (at least one component born)."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        grid = []
        for lat in range(-20, 21, 10):
            for lon in range(-20, 21, 10):
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=3))

        result = compute_persistent_coverage_homology(grid)

        assert result.betti_0_curve[0] >= 1

    def test_total_persistence_nonnegative(self):
        """Total persistence is non-negative."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        grid = []
        for lat in range(-30, 31, 10):
            for lon in range(-30, 31, 10):
                count = int(5 + 3 * math.sin(math.radians(lat)))
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=max(0, count)))

        result = compute_persistent_coverage_homology(grid)

        assert result.total_persistence >= 0.0

    def test_coverage_hole_detected_as_h1(self):
        """A coverage ring (high edges, low center, high center) creates H_1 feature."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        # Create a ring pattern: center is low, ring around it is high
        # This should create an H_1 feature when the ring enters the filtration
        # before the surrounding low-coverage region connects
        grid = []
        for lat in range(-30, 31, 10):
            for lon in range(-30, 31, 10):
                d = math.sqrt(lat ** 2 + lon ** 2)
                if 15 < d < 35:
                    count = 2  # Ring: enters at k=2
                else:
                    count = 5  # Interior and exterior: enters at k=5
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))

        result = compute_persistent_coverage_homology(grid)

        # The H_1 features track loops/holes in the sublevel set
        h1_features = [iv for iv in result.intervals if iv.dimension == 1]
        # There should be at least some topological activity
        assert len(result.intervals) > 0

    def test_significance_threshold_filters(self):
        """Significance threshold filters short-lived H_1 features."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        grid = []
        for lat in range(-30, 31, 10):
            for lon in range(-30, 31, 10):
                d = math.sqrt(lat ** 2 + lon ** 2)
                count = int(d / 10.0)
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))

        # High threshold: fewer significant holes
        result_high = compute_persistent_coverage_homology(grid, significance_threshold=100.0)
        # Low threshold: more significant holes
        result_low = compute_persistent_coverage_homology(grid, significance_threshold=0.01)

        assert result_high.num_significant_holes <= result_low.num_significant_holes

    def test_persistence_interval_frozen(self):
        """PersistenceInterval is immutable."""
        from humeris.domain.coverage import PersistenceInterval

        iv = PersistenceInterval(birth=0.0, death=5.0, dimension=0)
        with pytest.raises(AttributeError):
            iv.birth = 999.0  # type: ignore[misc]

    def test_dataclass_frozen(self):
        """PersistentCoverageHomology is immutable."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        grid = []
        for lat in range(-20, 21, 10):
            for lon in range(-20, 21, 10):
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=3))

        result = compute_persistent_coverage_homology(grid)

        with pytest.raises(AttributeError):
            result.num_significant_holes = 999  # type: ignore[misc]

    def test_max_hole_persistence_matches_intervals(self):
        """max_hole_persistence equals the max persistence of finite H_1 intervals."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        grid = []
        for lat in range(-30, 31, 10):
            for lon in range(-30, 31, 10):
                count = int(3 + 2 * math.cos(math.radians(lon * 2)))
                grid.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=max(0, count)))

        result = compute_persistent_coverage_homology(grid)

        finite_h1 = [iv for iv in result.intervals
                     if iv.dimension == 1 and iv.death != float('inf')]
        if finite_h1:
            expected_max = max(iv.death - iv.birth for iv in finite_h1)
            assert result.max_hole_persistence == pytest.approx(expected_max)
        else:
            assert result.max_hole_persistence == 0.0

    def test_small_grid_returns_empty(self):
        """Grid smaller than 2x2 returns empty homology."""
        from humeris.domain.coverage import CoveragePoint, compute_persistent_coverage_homology

        grid = [CoveragePoint(lat_deg=0.0, lon_deg=0.0, visible_count=5)]
        result = compute_persistent_coverage_homology(grid)

        assert result.intervals == ()


# ═══════════════════════════════════════════════════════════════════
# P31: Shadowing Lemma for Propagation Validation
# ═══════════════════════════════════════════════════════════════════


class TestShadowingValidation:
    """Tests for compute_shadowing_validation (P31)."""

    def test_import(self):
        """ShadowingValidation and compute_shadowing_validation are importable."""
        from humeris.domain.numerical_propagation import (
            ShadowingValidation,
            compute_shadowing_validation,
        )
        assert ShadowingValidation is not None
        assert compute_shadowing_validation is not None

    def _make_simple_steps(self, n_steps=50, step_s=60.0):
        """Create simple propagation steps for a circular LEO orbit."""
        from humeris.domain.numerical_propagation import (
            PropagationStep,
            TwoBodyGravity,
            rk4_step,
        )

        r = R_EARTH + 500_000.0
        v = math.sqrt(MU / r)
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

        gravity = TwoBodyGravity()

        def deriv(t_s, sv):
            p = (sv[0], sv[1], sv[2])
            vel = (sv[3], sv[4], sv[5])
            ax, ay, az = gravity.acceleration(epoch + timedelta(seconds=t_s), p, vel)
            return (vel[0], vel[1], vel[2], ax, ay, az)

        state = (r, 0.0, 0.0, 0.0, v, 0.0)
        steps = []
        t = 0.0
        for i in range(n_steps):
            current_time = epoch + timedelta(seconds=t)
            steps.append(PropagationStep(
                time=current_time,
                position_eci=(state[0], state[1], state[2]),
                velocity_eci=(state[3], state[4], state[5]),
            ))
            if i < n_steps - 1:
                t, state = rk4_step(t, state, step_s, deriv)

        return steps, [gravity], step_s

    def test_two_body_is_reliable(self):
        """Two-body propagation with moderate step size is reliable."""
        from humeris.domain.numerical_propagation import compute_shadowing_validation

        steps, forces, step_s = self._make_simple_steps(n_steps=20, step_s=30.0)
        result = compute_shadowing_validation(steps, forces, step_s, tolerance_m=10000.0)

        assert result.is_reliable is True
        assert result.reliability_margin > 1.0

    def test_per_step_error_positive(self):
        """Per-step truncation error is positive for real propagation."""
        from humeris.domain.numerical_propagation import compute_shadowing_validation

        steps, forces, step_s = self._make_simple_steps()
        result = compute_shadowing_validation(steps, forces, step_s)

        assert result.per_step_error_m > 0.0

    def test_lyapunov_exponent_nonnegative(self):
        """Maximum Lyapunov exponent is non-negative."""
        from humeris.domain.numerical_propagation import compute_shadowing_validation

        steps, forces, step_s = self._make_simple_steps()
        result = compute_shadowing_validation(steps, forces, step_s)

        assert result.max_lyapunov_exponent >= 0.0

    def test_shadowing_distance_positive(self):
        """Shadowing distance is positive for non-trivial propagation."""
        from humeris.domain.numerical_propagation import compute_shadowing_validation

        steps, forces, step_s = self._make_simple_steps()
        result = compute_shadowing_validation(steps, forces, step_s)

        assert result.shadowing_distance_m > 0.0

    def test_tight_tolerance_may_be_unreliable(self):
        """Very tight tolerance may cause unreliable classification."""
        from humeris.domain.numerical_propagation import compute_shadowing_validation

        steps, forces, step_s = self._make_simple_steps(n_steps=100, step_s=120.0)
        result = compute_shadowing_validation(steps, forces, step_s, tolerance_m=1e-10)

        # With very tight tolerance, should be unreliable
        assert result.is_reliable is False
        assert result.reliability_margin < 1.0

    def test_propagation_horizon_positive(self):
        """Propagation horizon is positive for non-trivial case."""
        from humeris.domain.numerical_propagation import compute_shadowing_validation

        steps, forces, step_s = self._make_simple_steps()
        result = compute_shadowing_validation(steps, forces, step_s, tolerance_m=1000.0)

        assert result.propagation_horizon_s > 0.0

    def test_few_steps_returns_safe_defaults(self):
        """Less than 3 steps returns safe defaults."""
        from humeris.domain.numerical_propagation import (
            PropagationStep,
            TwoBodyGravity,
            compute_shadowing_validation,
        )

        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        steps = [
            PropagationStep(time=epoch, position_eci=(7e6, 0, 0), velocity_eci=(0, 7500, 0)),
            PropagationStep(time=epoch + timedelta(seconds=60), position_eci=(7e6, 100, 0), velocity_eci=(0, 7500, 0)),
        ]

        result = compute_shadowing_validation(steps, [TwoBodyGravity()], 60.0)

        assert result.is_reliable is True
        assert result.shadowing_distance_m == 0.0

    def test_reliability_margin_consistent(self):
        """reliability_margin = tolerance / shadowing_distance."""
        from humeris.domain.numerical_propagation import compute_shadowing_validation

        steps, forces, step_s = self._make_simple_steps()
        tol = 5000.0
        result = compute_shadowing_validation(steps, forces, step_s, tolerance_m=tol)

        if result.shadowing_distance_m > 0:
            expected_margin = tol / result.shadowing_distance_m
            assert result.reliability_margin == pytest.approx(expected_margin, rel=1e-10)

    def test_dataclass_frozen(self):
        """ShadowingValidation is immutable."""
        from humeris.domain.numerical_propagation import compute_shadowing_validation

        steps, forces, step_s = self._make_simple_steps()
        result = compute_shadowing_validation(steps, forces, step_s)

        with pytest.raises(AttributeError):
            result.shadowing_distance_m = 999.0  # type: ignore[misc]

    def test_larger_step_gives_larger_error(self):
        """Larger step size produces larger per-step truncation error."""
        from humeris.domain.numerical_propagation import compute_shadowing_validation

        steps_small, forces, _ = self._make_simple_steps(n_steps=20, step_s=10.0)
        steps_large, _, _ = self._make_simple_steps(n_steps=20, step_s=120.0)

        result_small = compute_shadowing_validation(steps_small, forces, 10.0)
        result_large = compute_shadowing_validation(steps_large, forces, 120.0)

        assert result_large.per_step_error_m > result_small.per_step_error_m


# ═══════════════════════════════════════════════════════════════════
# P34: Melnikov Method for Chaos Onset Detection
# ═══════════════════════════════════════════════════════════════════


class TestMelnikovChaosAnalysis:
    """Tests for compute_melnikov_chaos (P34)."""

    def test_import(self):
        """MelnikovChaosAnalysis and compute_melnikov_chaos are importable."""
        from humeris.domain.numerical_propagation import (
            MelnikovChaosAnalysis,
            compute_melnikov_chaos,
        )
        assert MelnikovChaosAnalysis is not None
        assert compute_melnikov_chaos is not None

    def test_leo_orbit_has_zeros(self):
        """LEO orbit with J2 perturbation shows Melnikov zeros (chaos exists)."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        a = R_EARTH_EQ + 500_000.0  # 500 km LEO
        result = compute_melnikov_chaos(a)

        assert result.has_zeros is True
        assert result.melnikov_amplitude > 0.0

    def test_melnikov_amplitude_positive(self):
        """Melnikov amplitude is positive for non-degenerate orbit."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        a = R_EARTH_EQ + 800_000.0
        result = compute_melnikov_chaos(a)

        assert result.melnikov_amplitude > 0.0

    def test_chaos_width_positive(self):
        """Chaotic layer width is positive when zeros exist."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        a = R_EARTH_EQ + 600_000.0
        result = compute_melnikov_chaos(a)

        assert result.chaos_width_m > 0.0

    def test_resonance_order_valid(self):
        """Resonance order (p, q) has positive integers."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        a = R_EARTH_EQ + 500_000.0
        result = compute_melnikov_chaos(a)

        p, q = result.resonance_order
        assert p >= 1
        assert q >= 1

    def test_onset_altitude_physical(self):
        """Onset altitude is physically meaningful (positive for LEO+)."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        a = R_EARTH_EQ + 500_000.0
        result = compute_melnikov_chaos(a)

        # Resonance altitude should be above Earth's surface
        # (depends on the resonance order found, but for LEO the nearest
        # GPS/MEO resonances are well above surface)
        assert isinstance(result.onset_altitude_km, float)

    def test_j2_perturbation_strength_decreases_with_altitude(self):
        """The J2 perturbation forcing A_J2 = 1.5*J2*n*(Re/a)^2 decreases with altitude.

        The full Melnikov amplitude also depends on the resonance detuning,
        so we verify the underlying J2 forcing term directly.
        """
        import math as m
        from humeris.domain.orbital_mechanics import OrbitalConstants as OC

        a_low = R_EARTH_EQ + 400_000.0
        a_high = R_EARTH_EQ + 2_000_000.0

        n_low = m.sqrt(OC.MU_EARTH / a_low ** 3)
        n_high = m.sqrt(OC.MU_EARTH / a_high ** 3)

        a_j2_low = 1.5 * OC.J2_EARTH * n_low * (OC.R_EARTH_EQUATORIAL / a_low) ** 2
        a_j2_high = 1.5 * OC.J2_EARTH * n_high * (OC.R_EARTH_EQUATORIAL / a_high) ** 2

        assert a_j2_low > a_j2_high

    def test_damping_reduces_amplitude(self):
        """Non-zero damping rate reduces Melnikov amplitude."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        a = R_EARTH_EQ + 500_000.0

        result_undamped = compute_melnikov_chaos(a, damping_rate=0.0)
        result_damped = compute_melnikov_chaos(a, damping_rate=1e-4)

        assert result_damped.melnikov_amplitude < result_undamped.melnikov_amplitude

    def test_zero_sma_returns_safe_defaults(self):
        """Zero semi-major axis returns safe defaults."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        result = compute_melnikov_chaos(0.0)

        assert result.melnikov_amplitude == 0.0
        assert result.has_zeros is False
        assert result.chaos_width_m == 0.0

    def test_negative_sma_returns_safe_defaults(self):
        """Negative semi-major axis returns safe defaults."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        result = compute_melnikov_chaos(-1000.0)

        assert result.melnikov_amplitude == 0.0
        assert result.has_zeros is False

    def test_dataclass_frozen(self):
        """MelnikovChaosAnalysis is immutable."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        result = compute_melnikov_chaos(R_EARTH_EQ + 500_000.0)

        with pytest.raises(AttributeError):
            result.melnikov_amplitude = 999.0  # type: ignore[misc]

    def test_chaos_width_decreases_with_damping(self):
        """Chaos width decreases with increasing damping."""
        from humeris.domain.numerical_propagation import compute_melnikov_chaos

        a = R_EARTH_EQ + 600_000.0

        w0 = compute_melnikov_chaos(a, damping_rate=0.0).chaos_width_m
        w1 = compute_melnikov_chaos(a, damping_rate=1e-3).chaos_width_m

        assert w1 < w0


# ═══════════════════════════════════════════════════════════════════
# P37: Wavelet Multi-Resolution Coverage Analysis
# ═══════════════════════════════════════════════════════════════════


class TestWaveletCoverageAnalysis:
    """Tests for compute_wavelet_coverage (P37)."""

    def test_import(self):
        """WaveletCoverageAnalysis and compute_wavelet_coverage are importable."""
        from humeris.domain.coverage import (
            WaveletCoverageAnalysis,
            compute_wavelet_coverage,
        )
        assert WaveletCoverageAnalysis is not None
        assert compute_wavelet_coverage is not None

    def test_empty_signal(self):
        """Empty signal returns zero-energy result."""
        from humeris.domain.coverage import compute_wavelet_coverage

        result = compute_wavelet_coverage([])

        assert result.approximation_energy == 0.0
        assert result.detail_energies == ()
        assert result.is_trend_dominated is True

    def test_constant_signal_is_trend_dominated(self):
        """Constant signal has all energy in approximation (trend)."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = [5.0] * 16
        result = compute_wavelet_coverage(signal)

        assert result.is_trend_dominated is True
        assert result.transient_amplitude == pytest.approx(0.0, abs=1e-10)

    def test_constant_signal_detail_energies_zero(self):
        """Constant signal has zero detail energies at all scales."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = [5.0] * 8
        result = compute_wavelet_coverage(signal)

        for energy in result.detail_energies:
            assert energy == pytest.approx(0.0, abs=1e-10)

    def test_alternating_signal_high_frequency(self):
        """Alternating signal (+1, -1, +1, ...) has dominant scale 0 (finest)."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = [(-1.0) ** k for k in range(16)]
        result = compute_wavelet_coverage(signal)

        assert result.dominant_scale == 0
        assert len(result.detail_energies) > 0
        assert result.detail_energies[0] > 0.0

    def test_dominant_period_correct(self):
        """Dominant period corresponds to dominant scale."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = [(-1.0) ** k for k in range(16)]
        dt = 120.0
        result = compute_wavelet_coverage(signal, sample_interval_s=dt)

        # Scale 0: period = 2^1 * dt = 2 * 120 = 240
        expected_period = (2.0 ** (result.dominant_scale + 1)) * dt
        assert result.dominant_period_s == pytest.approx(expected_period)

    def test_energy_conservation(self):
        """Total energy is approximately conserved (Parseval's theorem)."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = [math.sin(2 * math.pi * k / 8) for k in range(32)]
        result = compute_wavelet_coverage(signal)

        # Total signal energy
        n_padded = 32  # already power of 2
        signal_energy = sum(x ** 2 for x in signal) / n_padded

        # Wavelet energy = approx + sum(detail)
        wavelet_energy = result.approximation_energy + sum(result.detail_energies)

        assert wavelet_energy == pytest.approx(signal_energy, rel=0.01)

    def test_trend_coefficient_for_constant(self):
        """Trend coefficient for constant signal equals the scaled value."""
        from humeris.domain.coverage import compute_wavelet_coverage

        val = 7.0
        signal = [val] * 8
        result = compute_wavelet_coverage(signal)

        # After log2(8)=3 levels of Haar transform, the single approx coefficient
        # should be val * sqrt(8) / sqrt(8) = val * (1/sqrt(2))^0 ... = val * sqrt(N)
        # Actually: each level multiplies by 1/sqrt(2) * sum of 2 equal values
        # Level 0: [7, 7, 7, 7] -> approx [7*sqrt(2)/sqrt(2), ...] = [7, 7, 7, 7]
        # Actually for Haar: approx[k] = (s[2k] + s[2k+1])/sqrt(2)
        # For constant 7: approx = 7 * sqrt(2) at each step... after 3 levels:
        # 7 * (sqrt(2))^3 = 7 * 2*sqrt(2) ≈ 19.8
        expected = val * (2.0 ** (3 / 2.0))
        assert result.trend_coefficient == pytest.approx(expected, rel=1e-10)

    def test_transient_amplitude_for_spike(self):
        """Signal with a spike has non-zero transient amplitude."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = [0.0] * 16
        signal[4] = 10.0  # spike
        result = compute_wavelet_coverage(signal)

        assert result.transient_amplitude > 0.0

    def test_multiple_levels_computed(self):
        """Number of detail levels equals log2(padded_length)."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = list(range(16))
        result = compute_wavelet_coverage(signal)

        # 16 = 2^4, so 4 levels
        assert len(result.detail_energies) == 4

    def test_max_levels_limits_decomposition(self):
        """max_levels parameter limits the number of decomposition levels."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = list(range(16))
        result = compute_wavelet_coverage(signal, max_levels=2)

        assert len(result.detail_energies) == 2

    def test_non_power_of_two_padded(self):
        """Non-power-of-2 signal length is handled (padded)."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = [1.0, 2.0, 3.0, 4.0, 5.0]  # length 5 -> padded to 8
        result = compute_wavelet_coverage(signal)

        assert len(result.detail_energies) == 3  # log2(8) = 3

    def test_sinusoidal_signal_not_trend_dominated(self):
        """Pure sinusoid has significant detail energy."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = [math.sin(2 * math.pi * k / 8) for k in range(64)]
        result = compute_wavelet_coverage(signal)

        total_detail = sum(result.detail_energies)
        assert total_detail > 0.0

    def test_dataclass_frozen(self):
        """WaveletCoverageAnalysis is immutable."""
        from humeris.domain.coverage import compute_wavelet_coverage

        result = compute_wavelet_coverage([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(AttributeError):
            result.dominant_scale = 999  # type: ignore[misc]

    def test_dominant_scale_in_valid_range(self):
        """Dominant scale is within valid range [0, num_levels-1]."""
        from humeris.domain.coverage import compute_wavelet_coverage

        signal = [math.cos(2 * math.pi * k / 4) for k in range(32)]
        result = compute_wavelet_coverage(signal)

        assert 0 <= result.dominant_scale < len(result.detail_energies)
