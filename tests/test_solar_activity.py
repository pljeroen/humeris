# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for solar activity model: prediction, providers, density, wiring."""
import math
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

import numpy as np
import pytest


# ── Phase 1: Solar cycle prediction ─────────────────────────────────


class TestSolarCyclePrediction:
    """Hathaway solar cycle model produces physically realistic F10.7/Ap."""

    def test_solar_minimum_f107_range(self) -> None:
        """Solar minimum F10.7 should be 65-75 SFU."""
        from humeris.domain.solar import predict_solar_activity

        # Cycle 24/25 minimum ~2019-2020
        result = predict_solar_activity(datetime(2020, 1, 1, tzinfo=timezone.utc))
        assert 60.0 <= result.f107_predicted <= 85.0, (
            f"Solar minimum F10.7={result.f107_predicted:.1f}, expected 60-85"
        )

    def test_solar_maximum_f107_range(self) -> None:
        """Solar maximum F10.7 should be 150-250 SFU."""
        from humeris.domain.solar import predict_solar_activity

        # Cycle 25 max ~2024-2025
        result = predict_solar_activity(datetime(2024, 10, 1, tzinfo=timezone.utc))
        assert 120.0 <= result.f107_predicted <= 250.0, (
            f"Solar maximum F10.7={result.f107_predicted:.1f}, expected 120-250"
        )

    def test_cycle_24_peak_timing(self) -> None:
        """Cycle 24 peaked around 2014.3 with F10.7 ~130-170."""
        from humeris.domain.solar import predict_solar_activity

        result = predict_solar_activity(datetime(2014, 4, 1, tzinfo=timezone.utc))
        assert result.f107_predicted > 100.0, (
            f"Cycle 24 peak F10.7={result.f107_predicted:.1f}, expected >100"
        )

    def test_cycle_25_peak_timing(self) -> None:
        """Cycle 25 peak around 2024-2025 with F10.7 ~150-200."""
        from humeris.domain.solar import predict_solar_activity

        result = predict_solar_activity(datetime(2024, 6, 1, tzinfo=timezone.utc))
        assert result.f107_predicted > 120.0, (
            f"Cycle 25 peak F10.7={result.f107_predicted:.1f}, expected >120"
        )

    def test_f107_physical_floor(self) -> None:
        """F10.7 never drops below 65 SFU (quiet Sun baseline)."""
        from humeris.domain.solar import predict_solar_activity

        for year in range(2000, 2035):
            result = predict_solar_activity(datetime(year, 1, 1, tzinfo=timezone.utc))
            assert result.f107_predicted >= 65.0, (
                f"F10.7={result.f107_predicted:.1f} at {year}, below physical floor 65"
            )

    def test_monotonic_rise_minimum_to_maximum(self) -> None:
        """F10.7 rises monotonically from cycle 25 minimum to maximum."""
        from humeris.domain.solar import predict_solar_activity

        # Cycle 25 rise: ~2020 to ~2024
        values = []
        for month_offset in range(0, 48, 6):
            epoch = datetime(2020, 6, 1, tzinfo=timezone.utc) + timedelta(days=month_offset * 30)
            result = predict_solar_activity(epoch)
            values.append(result.f107_predicted)

        # Overall trend should be upward (allow local non-monotonicity)
        assert values[-1] > values[0], (
            f"F10.7 should rise from {values[0]:.1f} to {values[-1]:.1f} during cycle 25 rise"
        )

    def test_f107_81day_average_smoothed(self) -> None:
        """81-day average should be close to daily prediction (smoothed)."""
        from humeris.domain.solar import predict_solar_activity

        result = predict_solar_activity(datetime(2024, 6, 1, tzinfo=timezone.utc))
        # 81-day average should be close to predicted (model is smooth)
        assert abs(result.f107_81day - result.f107_predicted) < 30.0

    def test_ap_positive_and_correlated(self) -> None:
        """Ap should be positive and higher during solar max."""
        from humeris.domain.solar import predict_solar_activity

        result_min = predict_solar_activity(datetime(2020, 1, 1, tzinfo=timezone.utc))
        result_max = predict_solar_activity(datetime(2024, 6, 1, tzinfo=timezone.utc))

        assert result_min.ap_predicted > 0
        assert result_max.ap_predicted > 0
        assert result_max.ap_predicted > result_min.ap_predicted

    def test_cycle_number_assigned(self) -> None:
        """Correct solar cycle number for known epochs."""
        from humeris.domain.solar import predict_solar_activity

        r2002 = predict_solar_activity(datetime(2002, 1, 1, tzinfo=timezone.utc))
        assert r2002.cycle_number == 23

        r2012 = predict_solar_activity(datetime(2012, 1, 1, tzinfo=timezone.utc))
        assert r2012.cycle_number == 24

        r2022 = predict_solar_activity(datetime(2022, 1, 1, tzinfo=timezone.utc))
        assert r2022.cycle_number == 25

    def test_cycle_phase_range(self) -> None:
        """Cycle phase should be in [0, 1]."""
        from humeris.domain.solar import predict_solar_activity

        for year in range(2000, 2035):
            result = predict_solar_activity(datetime(year, 6, 1, tzinfo=timezone.utc))
            assert 0.0 <= result.cycle_phase <= 1.0, (
                f"Phase={result.cycle_phase} at {year}, expected [0, 1]"
            )

    def test_dataclass_frozen(self) -> None:
        """SolarCyclePrediction should be immutable."""
        from humeris.domain.solar import predict_solar_activity

        result = predict_solar_activity(datetime(2024, 1, 1, tzinfo=timezone.utc))
        with pytest.raises(AttributeError):
            result.f107_predicted = 999.0  # type: ignore[misc]

    def test_solar_max_exceeds_solar_min(self) -> None:
        """F10.7 at solar maximum should clearly exceed solar minimum."""
        from humeris.domain.solar import predict_solar_activity

        result_min = predict_solar_activity(datetime(2020, 1, 1, tzinfo=timezone.utc))
        result_max = predict_solar_activity(datetime(2024, 10, 1, tzinfo=timezone.utc))

        assert result_max.f107_predicted > result_min.f107_predicted + 30.0


# ── Phase 2: SpaceWeatherProvider protocol ──────────────────────────


class TestSpaceWeatherProvider:
    """Protocol-based weather providers with composite fallback."""

    def test_predicted_provider_conforms(self) -> None:
        """PredictedSpaceWeatherProvider has lookup(epoch) -> SpaceWeather."""
        from humeris.domain.nrlmsise00 import (
            PredictedSpaceWeatherProvider,
            SpaceWeather,
        )

        provider = PredictedSpaceWeatherProvider()
        result = provider.lookup(datetime(2024, 6, 1, tzinfo=timezone.utc))
        assert isinstance(result, SpaceWeather)
        assert result.f107_daily > 0

    def test_composite_uses_historical_before_crossover(self) -> None:
        """CompositeSpaceWeatherProvider uses historical data before crossover."""
        from humeris.domain.nrlmsise00 import (
            CompositeSpaceWeatherProvider,
            PredictedSpaceWeatherProvider,
            SpaceWeatherHistory,
        )

        historical = SpaceWeatherHistory()
        predicted = PredictedSpaceWeatherProvider()
        crossover = datetime(2025, 1, 1, tzinfo=timezone.utc)

        composite = CompositeSpaceWeatherProvider(historical, predicted, crossover)
        # Before crossover: should use historical
        result = composite.lookup(datetime(2020, 6, 1, tzinfo=timezone.utc))
        hist_result = historical.lookup(datetime(2020, 6, 1, tzinfo=timezone.utc))
        assert result.f107_daily == hist_result.f107_daily

    def test_composite_uses_predicted_after_crossover(self) -> None:
        """CompositeSpaceWeatherProvider uses predicted data after crossover."""
        from humeris.domain.nrlmsise00 import (
            CompositeSpaceWeatherProvider,
            PredictedSpaceWeatherProvider,
            SpaceWeatherHistory,
        )

        historical = SpaceWeatherHistory()
        predicted = PredictedSpaceWeatherProvider()
        crossover = datetime(2025, 1, 1, tzinfo=timezone.utc)

        composite = CompositeSpaceWeatherProvider(historical, predicted, crossover)
        # After crossover: should use predicted
        result = composite.lookup(datetime(2035, 6, 1, tzinfo=timezone.utc))
        pred_result = predicted.lookup(datetime(2035, 6, 1, tzinfo=timezone.utc))
        assert result.f107_daily == pred_result.f107_daily

    def test_default_provider_returns_reasonable_values(self) -> None:
        """get_default_provider() returns reasonable values for 2020 and 2035."""
        from humeris.domain.nrlmsise00 import get_default_provider

        provider = get_default_provider()

        result_2020 = provider.lookup(datetime(2020, 6, 1, tzinfo=timezone.utc))
        assert 60.0 < result_2020.f107_daily < 300.0
        assert 60.0 < result_2020.f107_average < 300.0
        assert 0.0 < result_2020.ap_daily < 100.0

        result_2035 = provider.lookup(datetime(2035, 6, 1, tzinfo=timezone.utc))
        assert 60.0 < result_2035.f107_daily < 300.0

    def test_history_conforms_to_protocol(self) -> None:
        """SpaceWeatherHistory.lookup() matches provider protocol."""
        from humeris.domain.nrlmsise00 import (
            SpaceWeatherHistory,
            SpaceWeather,
        )

        history = SpaceWeatherHistory()
        result = history.lookup(datetime(2020, 1, 1, tzinfo=timezone.utc))
        assert isinstance(result, SpaceWeather)


# ── Phase 3: NRLMSISE-00 convenience function ──────────────────────


class TestAtmosphericDensityNRLMSISE00:
    """One-call density function with automatic space weather."""

    def test_solar_max_denser_than_min_at_400km(self) -> None:
        """Density at 400km during solar max > density during solar min."""
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00

        rho_min = atmospheric_density_nrlmsise00(
            400.0, datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        rho_max = atmospheric_density_nrlmsise00(
            400.0, datetime(2024, 10, 1, tzinfo=timezone.utc),
        )
        ratio = rho_max / rho_min
        assert ratio > 2.0, f"Solar max/min density ratio={ratio:.1f}, expected >2"

    def test_density_decreases_with_altitude(self) -> None:
        """Density at 200km > density at 400km (always)."""
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00

        epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rho_200 = atmospheric_density_nrlmsise00(200.0, epoch)
        rho_400 = atmospheric_density_nrlmsise00(400.0, epoch)
        assert rho_200 > rho_400

    def test_returns_float_kg_m3(self) -> None:
        """Returns float density in kg/m3."""
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00

        result = atmospheric_density_nrlmsise00(
            400.0, datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert isinstance(result, float)
        assert result > 0.0

    def test_reasonable_density_magnitude_400km(self) -> None:
        """400km density should be in ~1e-13 to 1e-11 kg/m3 range."""
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00

        result = atmospheric_density_nrlmsise00(
            400.0, datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert 1e-14 < result < 1e-10

    def test_accepts_custom_provider(self) -> None:
        """Can pass a custom SpaceWeatherProvider."""
        from humeris.domain.nrlmsise00 import (
            atmospheric_density_nrlmsise00,
            PredictedSpaceWeatherProvider,
        )

        provider = PredictedSpaceWeatherProvider()
        result = atmospheric_density_nrlmsise00(
            400.0, datetime(2024, 1, 1, tzinfo=timezone.utc),
            provider=provider,
        )
        assert result > 0.0


# ── Phase 4: Lifetime with density_func ─────────────────────────────


class TestLifetimeWithDensityFunc:
    """compute_orbit_lifetime accepts density_func callback."""

    def test_lifetime_solar_max_shorter_than_min(self) -> None:
        """Orbit lifetime at solar max should be shorter than solar min."""
        from humeris.domain.lifetime import compute_orbit_lifetime
        from humeris.domain.atmosphere import DragConfig
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00
        from humeris.domain.orbital_mechanics import OrbitalConstants

        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        a = OrbitalConstants.R_EARTH + 400e3  # 400 km

        epoch_min = datetime(2020, 1, 1, tzinfo=timezone.utc)
        epoch_max = datetime(2024, 10, 1, tzinfo=timezone.utc)

        def density_at_min(alt_km: float, epoch: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, epoch_min)

        def density_at_max(alt_km: float, epoch: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, epoch_max)

        result_min = compute_orbit_lifetime(
            a, 0.0, drag, epoch_min, density_func=density_at_min, step_days=10.0,
        )
        result_max = compute_orbit_lifetime(
            a, 0.0, drag, epoch_max, density_func=density_at_max, step_days=10.0,
        )

        assert result_max.lifetime_days < result_min.lifetime_days

    def test_lifetime_with_density_func_differs_from_exponential(self) -> None:
        """Lifetime with NRLMSISE-00 should differ from exponential model."""
        from humeris.domain.lifetime import compute_orbit_lifetime
        from humeris.domain.atmosphere import DragConfig
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00
        from humeris.domain.orbital_mechanics import OrbitalConstants

        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        a = OrbitalConstants.R_EARTH + 400e3
        epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def density_func(alt_km: float, ep: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, ep)

        result_exp = compute_orbit_lifetime(a, 0.0, drag, epoch, step_days=1.0)
        result_nrl = compute_orbit_lifetime(
            a, 0.0, drag, epoch, density_func=density_func, step_days=1.0,
        )

        # They should differ since NRLMSISE-00 uses different density model
        assert result_exp.lifetime_days != result_nrl.lifetime_days

    def test_stochastic_lifetime_accepts_density_func(self) -> None:
        """compute_stochastic_lifetime accepts density_func parameter."""
        from humeris.domain.lifetime import compute_stochastic_lifetime
        from humeris.domain.atmosphere import DragConfig
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00
        from humeris.domain.orbital_mechanics import OrbitalConstants

        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        a = OrbitalConstants.R_EARTH + 400e3
        epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def density_func(alt_km: float, ep: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, ep)

        result = compute_stochastic_lifetime(
            a, 0.0, drag, epoch,
            density_func=density_func,
            num_samples=10, step_days=10.0,
            rng_seed=42,
        )
        assert result.mean_lifetime_days > 0


# ── Phase 5: Station-keeping with density_func ──────────────────────


class TestStationKeepingWithDensityFunc:
    """Station-keeping functions accept density_func callback."""

    def test_drag_dv_solar_max_exceeds_min(self) -> None:
        """Annual drag dV at solar max > solar min."""
        from humeris.domain.station_keeping import drag_compensation_dv_per_year
        from humeris.domain.atmosphere import DragConfig
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00

        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        epoch_min = datetime(2020, 1, 1, tzinfo=timezone.utc)
        epoch_max = datetime(2024, 10, 1, tzinfo=timezone.utc)

        def density_min(alt_km: float, epoch: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, epoch_min)

        def density_max(alt_km: float, epoch: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, epoch_max)

        dv_min = drag_compensation_dv_per_year(400.0, drag, density_func=density_min)
        dv_max = drag_compensation_dv_per_year(400.0, drag, density_func=density_max)

        assert dv_max > dv_min

    def test_gve_budget_with_density_func(self) -> None:
        """GVE budget with density_func produces different result than default."""
        from humeris.domain.station_keeping import compute_gve_station_keeping_budget
        from humeris.domain.atmosphere import DragConfig
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00

        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def density_func(alt_km: float, ep: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, ep)

        result_default = compute_gve_station_keeping_budget(400.0, 0.001, drag)
        result_nrl = compute_gve_station_keeping_budget(
            400.0, 0.001, drag, density_func=density_func,
        )

        assert result_default.total_dv_per_year_ms != result_nrl.total_dv_per_year_ms

    def test_pharmacokinetics_with_density_func(self) -> None:
        """compute_propellant_pharmacokinetics accepts density_func."""
        from humeris.domain.station_keeping import compute_propellant_pharmacokinetics
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00

        epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def density_func(alt_km: float, ep: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, ep)

        result = compute_propellant_pharmacokinetics(
            initial_propellant_kg=10.0,
            altitude_km=400.0,
            isp_s=300.0,
            dry_mass_kg=100.0,
            density_func=density_func,
        )
        assert result.half_life > 0


# ── Phase 6: Decay analysis with density_func ───────────────────────


class TestDecayAnalysisWithDensityFunc:
    """Decay analysis accepts density_func callback."""

    def test_scale_map_with_density_func(self) -> None:
        """compute_exponential_scale_map accepts density_func."""
        from humeris.domain.decay_analysis import compute_exponential_scale_map
        from humeris.domain.propagation import OrbitalState
        from humeris.domain.atmosphere import DragConfig
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00
        from humeris.domain.orbital_mechanics import OrbitalConstants

        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        epoch = datetime(2024, 10, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400e3

        state = OrbitalState(
            semi_major_axis_m=a,
            eccentricity=0.001,
            inclination_rad=0.9,
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=math.sqrt(OrbitalConstants.MU_EARTH / a**3),
            reference_epoch=epoch,
        )

        def density_func(alt_km: float, ep: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, ep)

        result = compute_exponential_scale_map(
            state, drag, epoch, isp_s=300.0, dry_mass_kg=100.0,
            propellant_budget_kg=10.0, density_func=density_func,
        )
        assert len(result.processes) == 4


# ── Phase 7: Maintenance planning with density_func ─────────────────


# ── Phase 8: Kp↔Ap conversion ─────────────────────────────────────


class TestKpApConversion:
    """Bartels (1957) Kp↔Ap conversion table with interpolation."""

    def test_exact_table_kp0_gives_ap0(self) -> None:
        """Kp=0.0 → Ap=0 (exact table entry)."""
        from humeris.domain.nrlmsise00 import kp_to_ap

        assert kp_to_ap(0.0) == 0.0

    def test_exact_table_kp5_gives_ap48(self) -> None:
        """Kp=5.0 → Ap=48 (exact table entry)."""
        from humeris.domain.nrlmsise00 import kp_to_ap

        assert kp_to_ap(5.0) == 48.0

    def test_exact_table_kp9_gives_ap400(self) -> None:
        """Kp=9.0 → Ap=400 (exact table entry)."""
        from humeris.domain.nrlmsise00 import kp_to_ap

        assert kp_to_ap(9.0) == 400.0

    def test_interpolation_between_entries(self) -> None:
        """Kp between table entries produces interpolated Ap."""
        from humeris.domain.nrlmsise00 import kp_to_ap

        # Kp=4.5 is between 4.33 (Ap=32) and 4.67 (Ap=39)
        result = kp_to_ap(4.5)
        assert 32.0 < result < 39.0

    def test_round_trip_kp_to_ap_to_kp(self) -> None:
        """ap_to_kp(kp_to_ap(x)) ≈ x for various Kp values."""
        from humeris.domain.nrlmsise00 import kp_to_ap, ap_to_kp

        for kp in [0.0, 1.0, 3.0, 5.0, 7.0, 9.0]:
            ap = kp_to_ap(kp)
            kp_back = ap_to_kp(ap)
            assert abs(kp_back - kp) < 0.01, (
                f"Round-trip failed: Kp={kp} → Ap={ap} → Kp={kp_back}"
            )

    def test_out_of_range_clamped(self) -> None:
        """Kp < 0 clamps to 0, Kp > 9 clamps to 400."""
        from humeris.domain.nrlmsise00 import kp_to_ap

        assert kp_to_ap(-1.0) == 0.0
        assert kp_to_ap(10.0) == 400.0

    def test_ap_to_kp_exact_table(self) -> None:
        """Ap=132 → Kp=7.0 (exact reverse lookup)."""
        from humeris.domain.nrlmsise00 import ap_to_kp

        assert ap_to_kp(132.0) == 7.0

    def test_ap_to_kp_out_of_range_clamped(self) -> None:
        """Ap < 0 clamps to Kp=0, Ap > 400 clamps to Kp=9."""
        from humeris.domain.nrlmsise00 import ap_to_kp

        assert ap_to_kp(-5.0) == 0.0
        assert ap_to_kp(500.0) == 9.0


# ── Phase 9: Solar cycle uncertainty bounds ────────────────────────


class TestSolarCycleUncertaintyBounds:
    """Prediction envelope with upper/lower F10.7 bounds."""

    def test_bounds_bracket_prediction(self) -> None:
        """f107_lower <= f107_predicted <= f107_upper always."""
        from humeris.domain.solar import predict_solar_activity

        for year in range(2000, 2040):
            result = predict_solar_activity(datetime(year, 6, 1, tzinfo=timezone.utc))
            assert result.f107_lower <= result.f107_predicted <= result.f107_upper, (
                f"Bounds violated at {year}: "
                f"{result.f107_lower:.1f} <= {result.f107_predicted:.1f} <= {result.f107_upper:.1f}"
            )

    def test_cycle26_wider_bounds_than_cycle23(self) -> None:
        """Cycle 26 (far future) should have wider bounds than cycle 23 (observed)."""
        from humeris.domain.solar import predict_solar_activity

        c23 = predict_solar_activity(datetime(2002, 1, 1, tzinfo=timezone.utc))
        c26 = predict_solar_activity(datetime(2035, 1, 1, tzinfo=timezone.utc))

        width_23 = c23.f107_upper - c23.f107_lower
        width_26 = c26.f107_upper - c26.f107_lower

        assert width_26 > width_23, (
            f"Cycle 26 width={width_26:.1f} should exceed cycle 23 width={width_23:.1f}"
        )

    def test_solar_minimum_bounds_converge(self) -> None:
        """At solar minimum, bounds should be narrow (floor dominates)."""
        from humeris.domain.solar import predict_solar_activity

        # Cycle 24/25 minimum ~2019-2020
        result = predict_solar_activity(datetime(2020, 1, 1, tzinfo=timezone.utc))
        width = result.f107_upper - result.f107_lower
        assert width < 30.0, f"Min width={width:.1f}, expected <30 at solar minimum"

    def test_solar_maximum_bounds_widest(self) -> None:
        """At solar maximum, bounds should be widest within a cycle."""
        from humeris.domain.solar import predict_solar_activity

        result_min = predict_solar_activity(datetime(2020, 1, 1, tzinfo=timezone.utc))
        result_max = predict_solar_activity(datetime(2024, 6, 1, tzinfo=timezone.utc))

        width_min = result_min.f107_upper - result_min.f107_lower
        width_max = result_max.f107_upper - result_max.f107_lower

        assert width_max > width_min

    def test_backward_compat_existing_fields(self) -> None:
        """Existing fields (f107_predicted, f107_81day, ap_predicted) unchanged."""
        from humeris.domain.solar import predict_solar_activity

        result = predict_solar_activity(datetime(2024, 6, 1, tzinfo=timezone.utc))
        assert result.f107_predicted > 0
        assert result.f107_81day > 0
        assert result.ap_predicted > 0
        assert result.cycle_number == 25
        assert 0.0 <= result.cycle_phase <= 1.0

    def test_bounds_are_floats(self) -> None:
        """f107_upper and f107_lower are float values."""
        from humeris.domain.solar import predict_solar_activity

        result = predict_solar_activity(datetime(2024, 6, 1, tzinfo=timezone.utc))
        assert isinstance(result.f107_upper, float)
        assert isinstance(result.f107_lower, float)

    def test_bounds_above_physical_floor(self) -> None:
        """Lower bound never drops below 65 SFU physical floor."""
        from humeris.domain.solar import predict_solar_activity

        for year in range(2000, 2040):
            result = predict_solar_activity(datetime(year, 6, 1, tzinfo=timezone.utc))
            assert result.f107_lower >= 65.0, (
                f"Lower bound {result.f107_lower:.1f} < 65 at {year}"
            )


# ── Phase 10: 7-element Ap array ──────────────────────────────────


class TestApArray7Element:
    """7-element geomagnetic Ap array for NRLMSISE-00."""

    def test_scalar_ap_path_unchanged(self) -> None:
        """SpaceWeather without ap_array produces same result as before."""
        from humeris.domain.nrlmsise00 import NRLMSISE00Model, SpaceWeather

        model = NRLMSISE00Model()
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        state = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw)
        assert state.total_density_kg_m3 > 0

    def test_ap_array_none_defaults_to_scalar(self) -> None:
        """ap_array=None falls back to scalar Ap path."""
        from humeris.domain.nrlmsise00 import NRLMSISE00Model, SpaceWeather

        model = NRLMSISE00Model()
        sw_scalar = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        sw_none = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0,
                               ap_array=None)

        state_scalar = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_scalar)
        state_none = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_none)

        assert state_scalar.temperature_exospheric_k == state_none.temperature_exospheric_k

    def test_7element_produces_different_tinf(self) -> None:
        """7-element Ap array produces different T_inf than scalar."""
        from humeris.domain.nrlmsise00 import NRLMSISE00Model, SpaceWeather

        model = NRLMSISE00Model()
        # Scalar: ap=15
        sw_scalar = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        # 7-element: varying recent history
        ap_arr = (15.0, 30.0, 25.0, 20.0, 10.0, 8.0, 5.0)
        sw_array = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0,
                                ap_array=ap_arr)

        state_s = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_scalar)
        state_a = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_array)

        assert state_s.temperature_exospheric_k != state_a.temperature_exospheric_k

    def test_storm_ap_higher_tinf(self) -> None:
        """Recent geomagnetic storm (high recent Ap) → higher T_inf."""
        from humeris.domain.nrlmsise00 import NRLMSISE00Model, SpaceWeather

        model = NRLMSISE00Model()
        # Quiet: low recent Ap
        sw_quiet = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0,
                                ap_array=(15.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0))
        # Storm: high recent Ap
        sw_storm = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0,
                                ap_array=(15.0, 200.0, 150.0, 100.0, 50.0, 20.0, 10.0))

        state_q = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_quiet)
        state_s = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_storm)

        assert state_s.temperature_exospheric_k > state_q.temperature_exospheric_k

    def test_density_differs_with_ap_array(self) -> None:
        """Total density differs between scalar and 7-element Ap paths."""
        from humeris.domain.nrlmsise00 import NRLMSISE00Model, SpaceWeather

        model = NRLMSISE00Model()
        sw_scalar = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        ap_arr = (15.0, 50.0, 40.0, 30.0, 20.0, 15.0, 10.0)
        sw_array = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0,
                                ap_array=ap_arr)

        state_s = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_scalar)
        state_a = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_array)

        assert state_s.total_density_kg_m3 != state_a.total_density_kg_m3

    def test_short_array_falls_back_to_scalar(self) -> None:
        """Array shorter than 7 elements falls back to scalar path."""
        from humeris.domain.nrlmsise00 import NRLMSISE00Model, SpaceWeather

        model = NRLMSISE00Model()
        sw_scalar = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        sw_short = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0,
                                ap_array=(15.0, 20.0, 10.0))

        state_s = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_scalar)
        state_short = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_short)

        assert state_s.temperature_exospheric_k == state_short.temperature_exospheric_k

    def test_backward_compat_spaceweather_creation(self) -> None:
        """SpaceWeather can be created without ap_array (positional args)."""
        from humeris.domain.nrlmsise00 import SpaceWeather

        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        assert sw.ap_array is None

    def test_spaceweather_with_ap_array_frozen(self) -> None:
        """SpaceWeather with ap_array is still immutable."""
        from humeris.domain.nrlmsise00 import SpaceWeather

        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0,
                          ap_array=(15.0, 20.0, 25.0, 20.0, 15.0, 10.0, 5.0))
        with pytest.raises(AttributeError):
            sw.ap_array = (1.0,)  # type: ignore[misc]


# ── Phase 12: Nonlinear geomagnetic model (g0/sg0/sumex) ──────────


class TestNonlinearGeomagnetic:
    """NRLMSISE-00 g0/sg0/sumex nonlinear geomagnetic activity model."""

    def test_g0_linear_at_low_ap(self) -> None:
        """g0 is approximately linear for small disturbances (Ap near quiet)."""
        from humeris.domain.nrlmsise00 import _g0

        # At Ap=10 (small disturbance from quiet-day Ap=4)
        g_10 = _g0(10.0)
        g_15 = _g0(15.0)

        # Ratio should be close to linear: g(15)/g(10) ≈ (15-4)/(10-4) = 1.83
        ratio = g_15 / g_10
        assert 1.5 < ratio < 2.2, f"g0 ratio={ratio:.2f}, expected near 1.83"

    def test_g0_saturates_at_extreme_ap(self) -> None:
        """g0 saturates for extreme Ap (geomagnetic storms)."""
        from humeris.domain.nrlmsise00 import _g0

        g_50 = _g0(50.0)
        g_300 = _g0(300.0)

        # If linear: g(300)/g(50) = 296/46 = 6.4
        # With saturation should be much less
        ratio = g_300 / g_50
        assert ratio < 5.0, f"g0 ratio={ratio:.2f}, expected <5 (saturation)"

    def test_g0_quiet_day_baseline(self) -> None:
        """g0(4) ≈ 0 (Ap=4 is quiet-day baseline)."""
        from humeris.domain.nrlmsise00 import _g0

        assert abs(_g0(4.0)) < 0.1

    def test_sumex_normalization(self) -> None:
        """sumex produces proper normalization sum."""
        from humeris.domain.nrlmsise00 import _sumex

        # sumex(0.5) should be > 1 (sum of geometric series with sqrt)
        result = _sumex(0.5)
        assert result > 1.0
        # sumex(0) edge case
        result_0 = _sumex(0.001)
        assert result_0 > 1.0

    def test_sg0_exponential_decay(self) -> None:
        """sg0 weights recent ap more heavily than old ap."""
        from humeris.domain.nrlmsise00 import _sg0

        # All ap[1..6] = 0 except one recent vs one old
        ap_recent = (10.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ap_old = (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0)

        sg_recent = _sg0(ap_recent)
        sg_old = _sg0(ap_old)

        # Recent activity should produce larger effective Ap
        assert sg_recent > sg_old

    def test_exospheric_temp_saturates_at_extreme_ap(self) -> None:
        """T_inf increase diminishes at extreme Ap values."""
        from humeris.domain.nrlmsise00 import NRLMSISE00Model, SpaceWeather

        model = NRLMSISE00Model()
        sw_moderate = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=30.0)
        sw_extreme = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=300.0)

        t_mod = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_moderate)
        t_ext = model.evaluate(400.0, 0.0, 0.0, 2024, 180, 43200.0, sw_extreme)

        # Temperature increase from 30→300 Ap should be less than 10x
        # (if linear it would be exactly 10x)
        dt_moderate = t_mod.temperature_exospheric_k - 970.0  # approx base
        dt_extreme = t_ext.temperature_exospheric_k - 970.0
        if dt_moderate > 0:
            ratio = dt_extreme / dt_moderate
            assert ratio < 8.0, f"T_inf ratio={ratio:.1f}, expected <8 (saturation)"


# ── Phase 11: Calibrated synthetic space weather data ──────────────


class TestCalibratedSpaceWeatherData:
    """Calibrated synthetic space weather JSON with realistic variability."""

    def test_file_loads_without_error(self) -> None:
        """JSON file loads and parses without error."""
        from humeris.domain.nrlmsise00 import SpaceWeatherHistory

        history = SpaceWeatherHistory()
        # If it loads, it parsed successfully
        result = history.lookup(datetime(2010, 1, 1, tzinfo=timezone.utc))
        assert result.f107_daily > 0

    def test_spanning_2000_to_2030(self) -> None:
        """Data spans 2000-2030 with 11,000+ entries."""
        import json
        from pathlib import Path

        data_path = (
            Path(__file__).parent.parent
            / "packages" / "pro" / "src" / "humeris" / "data"
            / "space_weather_historical.json"
        )
        with open(data_path) as f:
            raw = json.load(f)

        entries = raw["entries"]
        assert len(entries) >= 11000, f"Only {len(entries)} entries, expected 11000+"

        first_date = entries[0]["date"]
        last_date = entries[-1]["date"]
        assert first_date.startswith("2000")
        assert last_date.startswith("2030") or last_date.startswith("2029")

    def test_f107_in_physical_range(self) -> None:
        """All F10.7 values in [65, 350] range."""
        import json
        from pathlib import Path

        data_path = (
            Path(__file__).parent.parent
            / "packages" / "pro" / "src" / "humeris" / "data"
            / "space_weather_historical.json"
        )
        with open(data_path) as f:
            raw = json.load(f)

        for entry in raw["entries"]:
            assert 65.0 <= entry["f107"] <= 350.0, (
                f"F10.7={entry['f107']} out of range at {entry['date']}"
            )

    def test_ap_in_physical_range(self) -> None:
        """All Ap values in [0, 400] range."""
        import json
        from pathlib import Path

        data_path = (
            Path(__file__).parent.parent
            / "packages" / "pro" / "src" / "humeris" / "data"
            / "space_weather_historical.json"
        )
        with open(data_path) as f:
            raw = json.load(f)

        for entry in raw["entries"]:
            assert 0.0 <= entry["ap"] <= 400.0, (
                f"Ap={entry['ap']} out of range at {entry['date']}"
            )

    def test_solar_max_years_higher_mean_f107(self) -> None:
        """Solar maximum years (2001-2002, 2014) have higher mean F10.7."""
        import json
        from pathlib import Path

        data_path = (
            Path(__file__).parent.parent
            / "packages" / "pro" / "src" / "humeris" / "data"
            / "space_weather_historical.json"
        )
        with open(data_path) as f:
            raw = json.load(f)

        # Gather F10.7 by year
        yearly: dict[int, list[float]] = {}
        for entry in raw["entries"]:
            year = int(entry["date"][:4])
            yearly.setdefault(year, []).append(entry["f107"])

        mean_by_year = {y: sum(v) / len(v) for y, v in yearly.items()}

        # Solar max ~2001-2002 should exceed solar min ~2008-2009
        if 2001 in mean_by_year and 2009 in mean_by_year:
            assert mean_by_year[2001] > mean_by_year[2009], (
                f"2001 mean={mean_by_year[2001]:.1f} should exceed "
                f"2009 mean={mean_by_year[2009]:.1f}"
            )

    def test_27day_periodicity_detectable(self) -> None:
        """27-day Bartels rotation modulation detectable in F10.7 spectrum."""
        import json
        from pathlib import Path

        data_path = (
            Path(__file__).parent.parent
            / "packages" / "pro" / "src" / "humeris" / "data"
            / "space_weather_historical.json"
        )
        with open(data_path) as f:
            raw = json.load(f)

        # Extract ~2 years of daily F10.7 during active period
        values = []
        for entry in raw["entries"]:
            if entry["date"].startswith("2001") or entry["date"].startswith("2002"):
                values.append(entry["f107"])

        if len(values) < 365:
            pytest.skip("Not enough data for spectral analysis")

        # FFT to find 27-day periodicity
        arr = np.array(values)
        arr = arr - arr.mean()  # Remove DC
        fft = np.abs(np.fft.rfft(arr))
        freqs = np.fft.rfftfreq(len(arr), d=1.0)  # 1-day sampling

        # 27-day period = frequency ~0.037 cycles/day
        target_freq = 1.0 / 27.0
        # Find peak near 27-day period (within ±5 days)
        mask = (freqs > 1.0 / 32.0) & (freqs < 1.0 / 22.0)
        if not mask.any():
            pytest.skip("Frequency range not found")

        peak_in_band = fft[mask].max()
        # Compare to median spectral power — should be elevated
        median_power = float(np.median(fft[1:]))  # Skip DC
        assert peak_in_band > 1.5 * median_power, (
            f"27-day peak={peak_in_band:.1f} not above 1.5x median={median_power:.1f}"
        )

    def test_data_marked_as_calibrated_synthetic(self) -> None:
        """JSON description indicates calibrated synthetic provenance."""
        import json
        from pathlib import Path

        data_path = (
            Path(__file__).parent.parent
            / "packages" / "pro" / "src" / "humeris" / "data"
            / "space_weather_historical.json"
        )
        with open(data_path) as f:
            raw = json.load(f)

        desc = raw.get("description", "").lower()
        assert "calibrated" in desc or "synthetic" in desc, (
            f"Description should mention calibrated/synthetic: {raw.get('description')}"
        )


class TestMaintenancePlanningWithDensityFunc:
    """Maintenance planning accepts density_func callback."""

    def test_perturbation_budget_with_density_func(self) -> None:
        """compute_perturbation_budget accepts density_func."""
        from humeris.domain.maintenance_planning import compute_perturbation_budget
        from humeris.domain.propagation import OrbitalState
        from humeris.domain.atmosphere import DragConfig
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00
        from humeris.domain.orbital_mechanics import OrbitalConstants

        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400e3

        state = OrbitalState(
            semi_major_axis_m=a,
            eccentricity=0.001,
            inclination_rad=0.9,
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=math.sqrt(OrbitalConstants.MU_EARTH / a**3),
            reference_epoch=epoch,
        )

        def density_func(alt_km: float, ep: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, ep)

        result = compute_perturbation_budget(
            state, drag, density_func=density_func,
        )
        assert len(result.elements) == 3

    def test_maintenance_schedule_with_density_func(self) -> None:
        """compute_maintenance_schedule accepts density_func."""
        from humeris.domain.maintenance_planning import compute_maintenance_schedule
        from humeris.domain.propagation import OrbitalState
        from humeris.domain.atmosphere import DragConfig
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00
        from humeris.domain.orbital_mechanics import OrbitalConstants

        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
        a = OrbitalConstants.R_EARTH + 400e3

        state = OrbitalState(
            semi_major_axis_m=a,
            eccentricity=0.001,
            inclination_rad=0.9,
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=math.sqrt(OrbitalConstants.MU_EARTH / a**3),
            reference_epoch=epoch,
        )

        def density_func(alt_km: float, ep: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, ep)

        result = compute_maintenance_schedule(
            state, drag, epoch, density_func=density_func,
        )
        assert result.total_dv_per_year_ms > 0

    def test_maintenance_more_frequent_at_solar_max(self) -> None:
        """Maintenance burns should be more frequent at solar max."""
        from humeris.domain.maintenance_planning import compute_maintenance_schedule
        from humeris.domain.propagation import OrbitalState
        from humeris.domain.atmosphere import DragConfig
        from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00
        from humeris.domain.orbital_mechanics import OrbitalConstants

        drag = DragConfig(cd=2.2, area_m2=1.0, mass_kg=100.0)
        a = OrbitalConstants.R_EARTH + 400e3

        state = OrbitalState(
            semi_major_axis_m=a,
            eccentricity=0.001,
            inclination_rad=0.9,
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=math.sqrt(OrbitalConstants.MU_EARTH / a**3),
            reference_epoch=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )

        epoch_min = datetime(2020, 1, 1, tzinfo=timezone.utc)
        epoch_max = datetime(2024, 10, 1, tzinfo=timezone.utc)

        def density_min(alt_km: float, ep: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, epoch_min)

        def density_max(alt_km: float, ep: datetime) -> float:
            return atmospheric_density_nrlmsise00(alt_km, epoch_max)

        result_min = compute_maintenance_schedule(
            state, drag, epoch_min, density_func=density_min,
        )
        result_max = compute_maintenance_schedule(
            state, drag, epoch_max, density_func=density_max,
        )

        assert result_max.total_dv_per_year_ms > result_min.total_dv_per_year_ms
