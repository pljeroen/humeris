# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for CascadeSIR epidemic model in domain/cascade_analysis.py."""
import math

from humeris.domain.cascade_analysis import CascadeSIR, compute_cascade_sir
from humeris.domain.orbital_mechanics import OrbitalConstants


class TestCascadeSIRSubcritical:
    """Subcritical case: R_0 < 1, debris decays toward zero."""

    def test_subcritical_r0_less_than_one(self):
        """Low density, high drag lifetime => R_0 < 1."""
        result = compute_cascade_sir(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-9,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=1000,
            launch_rate_per_year=0.0,
            fragments_per_collision=100.0,
            drag_lifetime_years=5.0,
            collision_cross_section_km2=1e-5,
            duration_years=50.0,
        )
        assert result.r_0 < 1.0
        assert not result.is_supercritical

    def test_subcritical_debris_decays(self):
        """In subcritical regime, infected (debris) at end < at start."""
        result = compute_cascade_sir(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-9,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=1000,
            launch_rate_per_year=0.0,
            fragments_per_collision=100.0,
            drag_lifetime_years=5.0,
            collision_cross_section_km2=1e-5,
            duration_years=50.0,
        )
        assert result.infected[-1] < result.infected[0]

    def test_subcritical_equilibrium_zero(self):
        """Equilibrium debris is 0 when subcritical."""
        result = compute_cascade_sir(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-9,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=1000,
            launch_rate_per_year=0.0,
            fragments_per_collision=100.0,
            drag_lifetime_years=5.0,
            collision_cross_section_km2=1e-5,
            duration_years=50.0,
        )
        assert result.equilibrium_debris == 0.0


class TestCascadeSIRSupercritical:
    """Supercritical case: R_0 > 1, debris grows."""

    def _supercritical_result(self):
        return compute_cascade_sir(
            shell_volume_km3=1e9,
            spatial_density_per_km3=1e-6,
            mean_collision_velocity_ms=14_000.0,
            satellite_count=50_000,
            launch_rate_per_year=1000.0,
            fragments_per_collision=200.0,
            drag_lifetime_years=100.0,
            collision_cross_section_km2=1e-4,
            duration_years=100.0,
        )

    def test_supercritical_r0_greater_than_one(self):
        result = self._supercritical_result()
        assert result.r_0 > 1.0
        assert result.is_supercritical

    def test_supercritical_debris_grows(self):
        """In supercritical regime, debris at some point exceeds initial."""
        result = self._supercritical_result()
        max_infected = max(result.infected)
        assert max_infected > result.infected[0]

    def test_supercritical_time_to_peak_positive(self):
        result = self._supercritical_result()
        assert result.time_to_peak_years > 0.0


class TestCascadeSIRR0Computation:
    """Verify R_0 against hand-calculated value."""

    def test_r0_hand_calculation(self):
        """
        Hand calculation:
        beta = collision_cross_section * (velocity_ms * 0.001) * seconds_per_year / shell_volume
             = 1e-5 * (10_000 * 0.001) * 31_557_600 / 1e10
             = 1e-5 * 10 * 31_557_600 / 1e10
             = 315_576_000 * 1e-5 / 1e10
             = 3155.76 / 1e10
             = 3.15576e-7

        gamma = 1 / 25 = 0.04
        R_0 = fragments * beta * satellite_count / gamma
            = 100 * 3.15576e-7 * 5000 / 0.04
            = 100 * 3.15576e-7 * 125000
            = 3.15576e-5 * 125000
            = 3.9447
        """
        result = compute_cascade_sir(
            shell_volume_km3=1e10,
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=5000,
            launch_rate_per_year=0.0,
            fragments_per_collision=100.0,
            drag_lifetime_years=25.0,
            collision_cross_section_km2=1e-5,
            duration_years=10.0,
        )
        assert abs(result.r_0 - 3.9447) < 0.01


class TestCascadeSIRConservation:
    """Total population conservation: S + I + R ~ S_0 + I_0 + launch_rate * t."""

    def test_conservation_no_launch(self):
        """Without launches: S + I + R = S_0 + I_0 at all times."""
        result = compute_cascade_sir(
            shell_volume_km3=1e10,
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=5000,
            launch_rate_per_year=0.0,
            fragments_per_collision=100.0,
            drag_lifetime_years=25.0,
            collision_cross_section_km2=1e-5,
            duration_years=50.0,
            step_years=0.1,
        )
        s0 = result.susceptible[0]
        i0 = result.infected[0]
        initial_total = s0 + i0

        # Note: SIR with fragment amplification does NOT conserve total count
        # when fragments_per_collision > 1. The conservation law is:
        # dS/dt + dI/dt + dR/dt = launch_rate + (fragments - 1) * beta * S * I
        # So we check a weaker form: S + I + R >= initial at each step
        # (population can only grow from fragmentation)
        for idx in range(len(result.time_series_years)):
            total = (result.susceptible[idx] + result.infected[idx]
                     + result.recovered[idx])
            # With fragmentation, total >= initial (fragments add population)
            # Allow small numerical tolerance for Euler integration
            assert total >= initial_total - 1.0, (
                f"Population decreased at t={result.time_series_years[idx]}: "
                f"{total} < {initial_total}"
            )

    def test_conservation_with_launch(self):
        """With launches, total population grows by at least launch_rate * t."""
        launch_rate = 100.0
        result = compute_cascade_sir(
            shell_volume_km3=1e12,
            spatial_density_per_km3=1e-9,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=1000,
            launch_rate_per_year=launch_rate,
            fragments_per_collision=100.0,
            drag_lifetime_years=25.0,
            collision_cross_section_km2=1e-5,
            duration_years=10.0,
            step_years=0.1,
        )
        s0 = result.susceptible[0]
        i0 = result.infected[0]
        initial_total = s0 + i0

        # At the final time step, total >= initial + launch_rate * t
        t_final = result.time_series_years[-1]
        total_final = (result.susceptible[-1] + result.infected[-1]
                       + result.recovered[-1])
        # Allow tolerance for Euler integration
        expected_min = initial_total + launch_rate * t_final
        assert total_final >= expected_min - 10.0


class TestCascadeSIRZeroDebris:
    """With zero initial debris, nothing happens (no infection seed)."""

    def test_zero_initial_debris_stays_zero(self):
        result = compute_cascade_sir(
            shell_volume_km3=1e10,
            spatial_density_per_km3=0.0,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=5000,
            launch_rate_per_year=0.0,
            fragments_per_collision=100.0,
            drag_lifetime_years=25.0,
            collision_cross_section_km2=1e-5,
            duration_years=50.0,
        )
        # All infected values should be zero
        for val in result.infected:
            assert val == 0.0

    def test_zero_initial_debris_susceptible_unchanged(self):
        result = compute_cascade_sir(
            shell_volume_km3=1e10,
            spatial_density_per_km3=0.0,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=5000,
            launch_rate_per_year=0.0,
            fragments_per_collision=100.0,
            drag_lifetime_years=25.0,
            collision_cross_section_km2=1e-5,
            duration_years=50.0,
        )
        for val in result.susceptible:
            assert val == 5000.0


class TestCascadeSIRDataclass:
    """Verify CascadeSIR is a proper frozen dataclass."""

    def test_frozen(self):
        result = compute_cascade_sir(
            shell_volume_km3=1e10,
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=5000,
            duration_years=10.0,
        )
        import pytest
        with pytest.raises(AttributeError):
            result.r_0 = 99.0

    def test_tuples_not_lists(self):
        """Time series fields are tuples (immutable)."""
        result = compute_cascade_sir(
            shell_volume_km3=1e10,
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=5000,
            duration_years=10.0,
        )
        assert isinstance(result.time_series_years, tuple)
        assert isinstance(result.susceptible, tuple)
        assert isinstance(result.infected, tuple)
        assert isinstance(result.recovered, tuple)

    def test_time_series_length(self):
        """Time series length matches expected number of steps."""
        result = compute_cascade_sir(
            shell_volume_km3=1e10,
            spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=5000,
            duration_years=10.0,
            step_years=0.5,
        )
        expected_steps = int(10.0 / 0.5) + 1
        assert len(result.time_series_years) == expected_steps
        assert len(result.susceptible) == expected_steps
        assert len(result.infected) == expected_steps
        assert len(result.recovered) == expected_steps


# ── estimate_drag_lifetime_years ──────────────────────────────────

class TestEstimateDragLifetime:
    """Tests for altitude-dependent drag lifetime estimation."""

    def test_200km_short_lifetime(self):
        """At 200 km, drag lifetime is very short (< 2 years)."""
        from humeris.domain.atmosphere import estimate_drag_lifetime_years

        lifetime = estimate_drag_lifetime_years(200.0)
        assert lifetime < 2.0

    def test_550km_moderate_lifetime(self):
        """At 550 km (Starlink altitude), drag lifetime is 5-20 years."""
        from humeris.domain.atmosphere import estimate_drag_lifetime_years

        lifetime = estimate_drag_lifetime_years(550.0)
        assert 5.0 < lifetime < 20.0

    def test_800km_long_lifetime(self):
        """At 800 km, drag lifetime is > 30 years."""
        from humeris.domain.atmosphere import estimate_drag_lifetime_years

        lifetime = estimate_drag_lifetime_years(800.0)
        assert lifetime > 30.0

    def test_monotonic_increase(self):
        """Drag lifetime increases monotonically with altitude."""
        from humeris.domain.atmosphere import estimate_drag_lifetime_years

        altitudes = [200, 300, 400, 500, 600, 700, 800, 900]
        lifetimes = [estimate_drag_lifetime_years(a) for a in altitudes]
        for i in range(len(lifetimes) - 1):
            assert lifetimes[i] < lifetimes[i + 1], (
                f"Lifetime decreased: {altitudes[i]} km={lifetimes[i]:.1f} yr "
                f"> {altitudes[i+1]} km={lifetimes[i+1]:.1f} yr"
            )

    def test_meo_very_long(self):
        """MEO altitudes (>1000 km) have very long drag lifetimes (>100 yr)."""
        from humeris.domain.atmosphere import estimate_drag_lifetime_years

        lifetime = estimate_drag_lifetime_years(1000.0)
        assert lifetime > 100.0

    def test_all_positive(self):
        """Drag lifetime is always positive."""
        from humeris.domain.atmosphere import estimate_drag_lifetime_years

        for alt in [150, 200, 300, 500, 700, 900]:
            assert estimate_drag_lifetime_years(alt) > 0.0


# ── Cascade SIR R₀ with realistic parameters ─────────────────────

class TestCascadeSIRRealisticR0:
    """R₀ should be physically reasonable with altitude-appropriate drag lifetime."""

    def test_starlink_r0_below_2(self):
        """Starlink-like constellation at 550 km: R₀ should be < 2.0.

        With ~6500 sats at 550 km, drag lifetime ~8-15 yr gives R₀ ~ 0.4-1.2.
        The old hardcoded 25 yr gave R₀ ~ 1.7 — unrealistically high.
        """
        from humeris.domain.atmosphere import estimate_drag_lifetime_years

        alt_km = 550.0
        n_sats = 6500
        shell_r = (OrbitalConstants.R_EARTH_EQUATORIAL / 1000.0) + alt_km
        shell_vol = 4.0 * math.pi * shell_r ** 2 * 50.0
        drag_life = estimate_drag_lifetime_years(alt_km)

        result = compute_cascade_sir(
            shell_volume_km3=shell_vol,
            spatial_density_per_km3=0.005 * n_sats / shell_vol,
            mean_collision_velocity_ms=10_000.0,
            satellite_count=n_sats,
            drag_lifetime_years=drag_life,
            duration_years=1.0,
        )
        assert result.r_0 < 2.0, (
            f"Starlink R₀={result.r_0:.2f} with drag_life={drag_life:.1f} yr "
            f"— should be < 2.0"
        )

    def test_glonass_subcritical(self):
        """GLONASS at ~19,100 km with 24 sats: R₀ should be << 1 (subcritical).

        At MEO, drag is negligible, but with only 24 satellites the collision
        rate is far too low for cascade.
        """
        from humeris.domain.atmosphere import estimate_drag_lifetime_years

        alt_km = 19100.0
        n_sats = 24
        shell_r = (OrbitalConstants.R_EARTH_EQUATORIAL / 1000.0) + alt_km
        shell_vol = 4.0 * math.pi * shell_r ** 2 * 200.0  # wider shell for MEO
        # At MEO, clamp lifetime to table max
        drag_life = min(estimate_drag_lifetime_years(min(alt_km, 900.0)), 500.0)

        result = compute_cascade_sir(
            shell_volume_km3=shell_vol,
            spatial_density_per_km3=0.005 * n_sats / shell_vol,
            mean_collision_velocity_ms=7_000.0,
            satellite_count=n_sats,
            drag_lifetime_years=drag_life,
            duration_years=1.0,
        )
        assert result.r_0 < 0.1, (
            f"GLONASS R₀={result.r_0:.4f} — should be << 1"
        )
