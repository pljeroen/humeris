# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for solar-aware end-of-life prediction in operational_prediction.py."""
import ast
import math
from datetime import datetime, timezone, timedelta

from humeris.domain.propagation import OrbitalState
from humeris.domain.atmosphere import DragConfig, atmospheric_density
from humeris.domain.nrlmsise00 import (
    NRLMSISE00Model,
    SpaceWeather,
    SpaceWeatherHistory,
)
from humeris.domain.maneuver_detection import (
    ManeuverEvent,
    ManeuverDetectionResult,
)
from humeris.domain.kessler_heatmap import KesslerHeatMap
from humeris.domain.operational_prediction import (
    SolarDecayPoint,
    SolarAwareEOL,
    compute_solar_aware_eol,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _state(alt_km=400.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(53.0),
        raan_rad=0.0, arg_perigee_rad=0.0, true_anomaly_rad=0.0,
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


def _drag():
    return DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)


def _solar_max_weather():
    """Space weather for solar maximum (high F10.7)."""
    return SpaceWeather(f107_daily=250.0, f107_average=220.0, ap_daily=25.0)


def _solar_min_weather():
    """Space weather for solar minimum (low F10.7)."""
    return SpaceWeather(f107_daily=70.0, f107_average=70.0, ap_daily=4.0)


def _maneuver_result(n_events=5, span_years=2.0):
    """Create a ManeuverDetectionResult with n_events over span_years."""
    events = []
    for k in range(n_events):
        t = _EPOCH + timedelta(days=k * span_years * 365.25 / max(n_events, 1))
        events.append(ManeuverEvent(
            detection_time=t,
            cusum_value=6.0,
            residual_magnitude_m=50.0,
            detection_type="cusum",
        ))
    return ManeuverDetectionResult(
        events=tuple(events),
        cusum_history=(1.0,) * max(n_events, 1),
        threshold=5.0,
        mean_residual_m=10.0,
        max_cusum=6.0,
    )


def _kessler_supercritical():
    """Create a KesslerHeatMap with k_eff > 1.0."""
    return KesslerHeatMap(
        cells=(),
        altitude_bins_km=(200.0, 2000.0),
        inclination_bins_deg=(0.0, 180.0),
        total_objects=10000,
        peak_density_altitude_km=800.0,
        peak_density_inclination_deg=98.0,
        peak_density_per_km3=5e-7,
        cascade_k_eff=1.5,
        is_supercritical=True,
    )


def _kessler_subcritical():
    """Create a KesslerHeatMap with k_eff < 1.0."""
    return KesslerHeatMap(
        cells=(),
        altitude_bins_km=(200.0, 2000.0),
        inclination_bins_deg=(0.0, 180.0),
        total_objects=100,
        peak_density_altitude_km=800.0,
        peak_density_inclination_deg=98.0,
        peak_density_per_km3=1e-10,
        cascade_k_eff=0.1,
        is_supercritical=False,
    )


# ---------------------------------------------------------------------------
# Core tests (1-8)
# ---------------------------------------------------------------------------


class TestCoreOutput:
    """Core output structure and monotonicity tests."""

    def test_returns_solar_aware_eol_dataclass(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert isinstance(result, SolarAwareEOL)

    def test_end_of_life_mode_in_valid_set(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        valid = {"fuel_depletion", "reentry", "cascade_critical", "indeterminate"}
        assert result.end_of_life_mode in valid

    def test_decay_profile_non_empty_tuple_of_solar_decay_point(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert isinstance(result.decay_profile, tuple)
        assert len(result.decay_profile) > 0
        assert all(isinstance(p, SolarDecayPoint) for p in result.decay_profile)

    def test_altitude_decreases_over_time(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        profile = result.decay_profile
        if len(profile) >= 2:
            assert profile[-1].altitude_km <= profile[0].altitude_km

    def test_propellant_remaining_decreases_over_time(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        profile = result.decay_profile
        if len(profile) >= 2:
            assert profile[-1].propellant_remaining_kg <= profile[0].propellant_remaining_kg

    def test_small_propellant_budget_early_depletion(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=0.1,
        )
        assert result.fuel_depletion_time is not None
        assert result.end_of_life_mode == "fuel_depletion"

    def test_reentry_time_set_when_altitude_drops(self):
        # Low altitude, high drag — should reenter
        result = compute_solar_aware_eol(
            _state(alt_km=250.0), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=200.0,
            mission_years=50.0,
        )
        assert result.reentry_time is not None

    def test_controlled_deorbit_feasible_when_fuel_remains(self):
        # Large propellant budget — fuel lasts beyond reentry
        result = compute_solar_aware_eol(
            _state(alt_km=250.0), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=500.0,
            mission_years=50.0,
        )
        if result.reentry_time is not None:
            if result.fuel_depletion_time is None:
                assert result.controlled_deorbit_feasible is True
            elif result.fuel_depletion_time > result.reentry_time:
                assert result.controlled_deorbit_feasible is True


# ---------------------------------------------------------------------------
# Solar cycle tests (9-12)
# ---------------------------------------------------------------------------


class TestSolarCycle:
    """Solar activity influence on predictions."""

    def test_solar_max_shorter_lifetime_than_solar_min(self):
        result_max = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            space_weather_history=None,
            static_space_weather=_solar_max_weather(),
        )
        result_min = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            space_weather_history=None,
            static_space_weather=_solar_min_weather(),
        )
        # Higher density at solar max → faster fuel depletion or reentry
        eol_max = result_max.end_of_life_time or _EPOCH + timedelta(days=365.25 * 100)
        eol_min = result_min.end_of_life_time or _EPOCH + timedelta(days=365.25 * 100)
        assert eol_max <= eol_min

    def test_density_ratio_gt_1_at_solar_max(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            static_space_weather=_solar_max_weather(),
        )
        assert result.density_ratio_vs_static > 1.0

    def test_density_ratio_lt_1_at_solar_min(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            static_space_weather=_solar_min_weather(),
        )
        assert result.density_ratio_vs_static < 1.0

    def test_mean_f107_within_space_weather_range(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            static_space_weather=_solar_max_weather(),
        )
        assert 50.0 <= result.mean_f107 <= 400.0

    def test_solar_cycle_phase_classification(self):
        result_max = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            static_space_weather=_solar_max_weather(),
        )
        result_min = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            static_space_weather=_solar_min_weather(),
        )
        assert result_max.solar_cycle_phase == "maximum"
        assert result_min.solar_cycle_phase == "minimum"


# ---------------------------------------------------------------------------
# Propellant and maneuver tests (13-17)
# ---------------------------------------------------------------------------


class TestPropellantManeuver:
    """Propellant consumption and maneuver adjustment tests."""

    def test_zero_budget_immediate_depletion(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=0.0,
        )
        assert result.fuel_depletion_time is not None
        # Should deplete at or very near epoch
        delta = (result.fuel_depletion_time - _EPOCH).total_seconds()
        assert delta < 86400.0 * 60  # within 60 days

    def test_large_budget_no_depletion(self):
        # Use lower BC (heavier spacecraft) to prevent natural decay
        low_bc_drag = DragConfig(cd=2.2, area_m2=2.0, mass_kg=500.0)
        result = compute_solar_aware_eol(
            _state(alt_km=500.0), low_bc_drag, _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=10000.0,
            mission_years=10.0,
        )
        assert result.fuel_depletion_time is None

    def test_maneuver_events_increase_rate(self):
        result_no = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        result_with = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            maneuver_result=_maneuver_result(n_events=10, span_years=2.0),
        )
        assert result_with.observed_maneuver_rate_per_year > result_no.observed_maneuver_rate_per_year

    def test_maneuver_events_increase_adjusted_dv(self):
        result_no = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        result_with = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            maneuver_result=_maneuver_result(n_events=10, span_years=2.0),
        )
        assert result_with.maneuver_adjusted_dv_per_year_ms > result_no.maneuver_adjusted_dv_per_year_ms

    def test_no_maneuvers_rate_zero(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert result.observed_maneuver_rate_per_year == 0.0


# ---------------------------------------------------------------------------
# Environment tests (18-21)
# ---------------------------------------------------------------------------


class TestEnvironment:
    """Environment risk classification tests."""

    def test_no_kessler_k_eff_zero(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert result.cascade_k_eff == 0.0

    def test_supercritical_kessler_critical_risk(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            kessler_heatmap=_kessler_supercritical(),
        )
        assert result.environment_risk_level == "critical"

    def test_high_cwi_elevated_risk(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            conjunction_weather_index=0.8,
        )
        assert result.environment_risk_level in ("high", "critical")

    def test_no_environment_data_low_risk(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert result.environment_risk_level == "low"


# ---------------------------------------------------------------------------
# Integration tests (22-26)
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration / end-to-end tests."""

    def test_full_pipeline_produces_reasonable_numbers(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert result.mean_f107 > 0.0
        assert result.density_ratio_vs_static > 0.0
        assert result.maneuver_adjusted_dv_per_year_ms >= 0.0

    def test_400km_leo_lifetime_reasonable(self):
        # Bc=0.044 m²/kg is high drag; 400 km lifetime is ~1-3 years
        result = compute_solar_aware_eol(
            _state(alt_km=400.0), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            mission_years=30.0,
        )
        if result.end_of_life_time is not None:
            lifetime_years = (result.end_of_life_time - _EPOCH).total_seconds() / (365.25 * 86400)
            assert 0.1 <= lifetime_years <= 30.0

    def test_600km_longer_than_400km(self):
        result_400 = compute_solar_aware_eol(
            _state(alt_km=400.0), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            mission_years=30.0,
        )
        result_600 = compute_solar_aware_eol(
            _state(alt_km=600.0), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            mission_years=30.0,
        )
        eol_400 = result_400.end_of_life_time or _EPOCH + timedelta(days=365.25 * 100)
        eol_600 = result_600.end_of_life_time or _EPOCH + timedelta(days=365.25 * 100)
        assert eol_600 >= eol_400

    def test_all_optional_params_none_works(self):
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        assert isinstance(result, SolarAwareEOL)

    def test_defaults_roughly_match_static_eol(self):
        """Solar-aware EOL with moderate defaults should be comparable to static."""
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
        )
        # density_ratio should be within order-of-magnitude of 1.0
        assert 0.01 < result.density_ratio_vs_static < 100.0


# ---------------------------------------------------------------------------
# Edge cases and purity (27-30)
# ---------------------------------------------------------------------------


class TestEdgeCasesAndPurity:
    """Edge cases and domain purity."""

    def test_very_low_altitude_rapid_reentry(self):
        result = compute_solar_aware_eol(
            _state(alt_km=150.0), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            mission_years=5.0,
        )
        assert result.reentry_time is not None
        days_to_reentry = (result.reentry_time - _EPOCH).total_seconds() / 86400.0
        assert days_to_reentry < 365.0  # should reenter within a year

    def test_very_high_altitude_negligible_drag(self):
        result = compute_solar_aware_eol(
            _state(alt_km=900.0), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            mission_years=25.0,
        )
        # At 900 km drag is very low — should not reenter within 25 years
        assert result.reentry_time is None

    def test_step_size_consistency(self):
        """Profile points should be spaced roughly step_days apart."""
        result = compute_solar_aware_eol(
            _state(), _drag(), _EPOCH, isp_s=300.0,
            dry_mass_kg=400.0, propellant_budget_kg=50.0,
            step_days=30.0,
        )
        profile = result.decay_profile
        if len(profile) >= 3:
            dt_01 = (profile[1].time - profile[0].time).total_seconds()
            dt_12 = (profile[2].time - profile[1].time).total_seconds()
            expected = 30.0 * 86400.0
            assert abs(dt_01 - expected) < 1.0
            assert abs(dt_12 - expected) < 1.0

    def test_domain_purity_no_non_domain_imports(self):
        """operational_prediction.py must only import from stdlib and domain."""
        import pathlib
        import humeris.domain.operational_prediction as _mod
        src = pathlib.Path(_mod.__file__)
        tree = ast.parse(src.read_text())
        allowed_prefixes = (
            "humeris.domain.",
            "humeris.domain",
        )
        stdlib = {
            "math", "dataclasses", "datetime", "typing", "enum", "collections",
            "functools", "itertools", "abc", "pathlib", "json",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    assert name in stdlib or name == "numpy", (
                        f"Non-domain import: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                top = node.module.split(".")[0]
                if top in stdlib or top == "numpy":
                    continue
                assert any(
                    node.module.startswith(p) for p in allowed_prefixes
                ), f"Non-domain import: {node.module}"
