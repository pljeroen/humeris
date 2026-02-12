# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for domain/statistical_analysis.py — collision probability, survival, availability, correlation."""
import ast
import math
from datetime import datetime, timezone

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.lifetime import OrbitLifetimeResult, DecayPoint, compute_orbit_lifetime
from constellation_generator.domain.statistical_analysis import (
    CollisionProbabilityAnalytical,
    LifetimeSurvivalCurve,
    MissionAvailabilityProfile,
    RadiationEclipseCorrelation,
    compute_analytical_collision_probability,
    compute_lifetime_survival_curve,
    compute_mission_availability,
    compute_radiation_eclipse_correlation,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_DRAG = DragConfig(cd=2.2, area_m2=0.05, mass_kg=100.0)


def _state(alt_km=550.0, inc_deg=53.0, raan_deg=0.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(inc_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=0.0, true_anomaly_rad=0.0,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


def _make_lifetime_result():
    """Create a minimal OrbitLifetimeResult for testing survival curves."""
    return compute_orbit_lifetime(
        semi_major_axis_m=_R_E + 400_000.0,
        eccentricity=0.0,
        drag_config=_DRAG,
        epoch=_EPOCH,
        step_days=30.0,
        max_years=25.0,
    )


class TestAnalyticalCollisionProbability:
    def test_returns_type(self):
        result = compute_analytical_collision_probability(
            miss_distance_m=500.0,
            b_radial_m=300.0, b_cross_m=400.0,
            sigma_radial_m=200.0, sigma_cross_m=200.0,
            combined_radius_m=10.0,
        )
        assert isinstance(result, CollisionProbabilityAnalytical)

    def test_centered_known(self):
        """d=0, equal σ → P_c = 1 - exp(-r²/(2σ²))."""
        r = 10.0
        sigma = 200.0
        result = compute_analytical_collision_probability(
            miss_distance_m=0.0,
            b_radial_m=0.0, b_cross_m=0.0,
            sigma_radial_m=sigma, sigma_cross_m=sigma,
            combined_radius_m=r,
        )
        expected = 1.0 - math.exp(-(r ** 2) / (2.0 * sigma ** 2))
        assert abs(result.analytical_pc - expected) < expected * 0.1 + 1e-10

    def test_far_miss_low(self):
        result = compute_analytical_collision_probability(
            miss_distance_m=10_000.0,
            b_radial_m=7000.0, b_cross_m=7000.0,
            sigma_radial_m=100.0, sigma_cross_m=100.0,
            combined_radius_m=10.0,
        )
        assert result.analytical_pc < 1e-6

    def test_normalized_miss_positive(self):
        result = compute_analytical_collision_probability(
            miss_distance_m=500.0,
            b_radial_m=300.0, b_cross_m=400.0,
            sigma_radial_m=200.0, sigma_cross_m=200.0,
            combined_radius_m=10.0,
        )
        assert result.normalized_miss_distance >= 0.0


class TestLifetimeSurvivalCurve:
    def test_returns_type(self):
        lt = _make_lifetime_result()
        result = compute_lifetime_survival_curve(lt)
        assert isinstance(result, LifetimeSurvivalCurve)

    def test_monotone_decreasing(self):
        lt = _make_lifetime_result()
        result = compute_lifetime_survival_curve(lt)
        for i in range(len(result.survival_fraction) - 1):
            assert result.survival_fraction[i] >= result.survival_fraction[i + 1] - 1e-10

    def test_starts_at_one(self):
        lt = _make_lifetime_result()
        result = compute_lifetime_survival_curve(lt)
        assert abs(result.survival_fraction[0] - 1.0) < 0.05

    def test_hazard_rate_positive(self):
        lt = _make_lifetime_result()
        result = compute_lifetime_survival_curve(lt)
        assert all(h >= -1e-10 for h in result.hazard_rate_per_day)

    def test_hazard_rate_increases(self):
        lt = _make_lifetime_result()
        result = compute_lifetime_survival_curve(lt)
        if len(result.hazard_rate_per_day) >= 3:
            # General trend: hazard increases (check first vs last quartile)
            n = len(result.hazard_rate_per_day)
            first_q = sum(result.hazard_rate_per_day[:n // 4]) / max(1, n // 4)
            last_q = sum(result.hazard_rate_per_day[3 * n // 4:]) / max(1, n - 3 * n // 4)
            assert last_q >= first_q - 1e-10

    def test_half_life_in_range(self):
        lt = _make_lifetime_result()
        result = compute_lifetime_survival_curve(lt)
        assert result.half_life_altitude_km >= lt.re_entry_altitude_km
        assert result.half_life_altitude_km <= lt.initial_altitude_km


class TestMissionAvailability:
    def test_returns_type(self):
        result = compute_mission_availability(
            state=_state(alt_km=550.0),
            drag_config=_DRAG,
            epoch=_EPOCH,
            isp_s=300.0, dry_mass_kg=100.0,
            propellant_budget_kg=20.0,
            mission_years=5.0,
        )
        assert isinstance(result, MissionAvailabilityProfile)

    def test_starts_high(self):
        result = compute_mission_availability(
            state=_state(alt_km=550.0),
            drag_config=_DRAG,
            epoch=_EPOCH,
            isp_s=300.0, dry_mass_kg=100.0,
            propellant_budget_kg=20.0,
            mission_years=5.0,
        )
        assert result.total_availability[0] > 0.5

    def test_availability_decreases(self):
        result = compute_mission_availability(
            state=_state(alt_km=550.0),
            drag_config=_DRAG,
            epoch=_EPOCH,
            isp_s=300.0, dry_mass_kg=100.0,
            propellant_budget_kg=20.0,
            mission_years=5.0,
        )
        # Overall: last value ≤ first value
        assert result.total_availability[-1] <= result.total_availability[0] + 1e-6

    def test_reliability_range(self):
        result = compute_mission_availability(
            state=_state(alt_km=550.0),
            drag_config=_DRAG,
            epoch=_EPOCH,
            isp_s=300.0, dry_mass_kg=100.0,
            propellant_budget_kg=20.0,
            mission_years=5.0,
        )
        assert 0.0 <= result.mission_reliability <= 1.0 + 1e-10

    def test_critical_factor_valid(self):
        result = compute_mission_availability(
            state=_state(alt_km=550.0),
            drag_config=_DRAG,
            epoch=_EPOCH,
            isp_s=300.0, dry_mass_kg=100.0,
            propellant_budget_kg=20.0,
            mission_years=5.0,
        )
        assert result.critical_factor in {"fuel", "power", "conjunction"}


class TestRadiationEclipseCorrelation:
    def test_returns_type(self):
        result = compute_radiation_eclipse_correlation(
            state=_state(alt_km=550.0), epoch=_EPOCH, num_months=6,
        )
        assert isinstance(result, RadiationEclipseCorrelation)

    def test_correlation_range(self):
        result = compute_radiation_eclipse_correlation(
            state=_state(alt_km=550.0), epoch=_EPOCH, num_months=6,
        )
        assert -1.0 - 1e-10 <= result.dose_eclipse_correlation <= 1.0 + 1e-10


class TestStatisticalAnalysisPurity:
    def test_module_pure(self):
        import constellation_generator.domain.statistical_analysis as mod
        source = ast.parse(open(mod.__file__).read())
        for node in ast.walk(source):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                else:
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                assert top in {"math", "dataclasses", "datetime", "typing", "enum", "constellation_generator", "__future__"}, (
                    f"Forbidden import: {top}"
                )
