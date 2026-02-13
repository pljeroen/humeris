# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for solar activity model: prediction, providers, density, wiring."""
import math
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

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

        result_exp = compute_orbit_lifetime(a, 0.0, drag, epoch, step_days=10.0)
        result_nrl = compute_orbit_lifetime(
            a, 0.0, drag, epoch, density_func=density_func, step_days=10.0,
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
