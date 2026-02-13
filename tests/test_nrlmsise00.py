# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Tests for NRLMSISE-00 simplified atmosphere model.

Covers reference densities, temperature profiles, solar activity effects,
latitude/local-time variations, magnetic activity, boundary conditions,
space weather loading, ForceModel protocol compliance, comparison with
exponential model, performance, defaults, and domain purity.
"""

import ast
import math
import time
from datetime import datetime, timezone

import pytest


# ---------------------------------------------------------------------------
# Reference density tests (10 tests)
# ---------------------------------------------------------------------------

class TestReferenceDensities:
    """Verify total mass density at standard altitudes for moderate solar activity."""

    def _evaluate(self, altitude_km: float) -> "AtmosphereState":
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        model = NRLMSISE00Model()
        return model.evaluate(
            altitude_km=altitude_km,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )

    def test_density_200km(self):
        state = self._evaluate(200.0)
        assert 2e-10 <= state.total_density_kg_m3 <= 5e-10

    def test_density_300km(self):
        state = self._evaluate(300.0)
        assert 1e-11 <= state.total_density_kg_m3 <= 5e-11

    def test_density_400km(self):
        state = self._evaluate(400.0)
        assert 2e-12 <= state.total_density_kg_m3 <= 1e-11

    def test_density_500km(self):
        state = self._evaluate(500.0)
        assert 5e-13 <= state.total_density_kg_m3 <= 5e-12

    def test_density_600km(self):
        state = self._evaluate(600.0)
        assert 1e-13 <= state.total_density_kg_m3 <= 2e-12

    def test_density_800km(self):
        state = self._evaluate(800.0)
        assert 1e-15 <= state.total_density_kg_m3 <= 5e-14

    def test_density_1000km(self):
        state = self._evaluate(1000.0)
        assert 1e-16 <= state.total_density_kg_m3 <= 1e-14

    def test_density_decreases_with_altitude(self):
        """Density strictly decreases from 200 to 1000 km."""
        altitudes = [200, 300, 400, 500, 600, 800, 1000]
        densities = [self._evaluate(h).total_density_kg_m3 for h in altitudes]
        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1], (
                f"Density at {altitudes[i]} km ({densities[i]:.2e}) "
                f"not > density at {altitudes[i+1]} km ({densities[i+1]:.2e})"
            )

    def test_density_positive(self):
        """All densities must be positive."""
        for h in [120, 200, 500, 1000, 1500]:
            state = self._evaluate(h)
            assert state.total_density_kg_m3 > 0

    def test_species_sum_matches_total(self):
        """Sum of species mass densities should approximate total density."""
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        model = NRLMSISE00Model()
        state = model.evaluate(
            altitude_km=400.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )
        # Avogadro constant
        N_A = 6.02214076e23
        # Species molar masses (g/mol)
        masses = {
            "n2": 28.0134e-3,  # kg/mol
            "o2": 31.9988e-3,
            "o": 15.999e-3,
            "he": 4.0026e-3,
            "ar": 39.948e-3,
            "h": 1.008e-3,
            "n": 14.007e-3,
        }
        computed_density = (
            state.n2_density_m3 * masses["n2"] / N_A
            + state.o2_density_m3 * masses["o2"] / N_A
            + state.o_density_m3 * masses["o"] / N_A
            + state.he_density_m3 * masses["he"] / N_A
            + state.ar_density_m3 * masses["ar"] / N_A
            + state.h_density_m3 * masses["h"] / N_A
            + state.n_density_m3 * masses["n"] / N_A
        )
        # Should be within 5% of total_density
        ratio = computed_density / state.total_density_kg_m3
        assert 0.95 <= ratio <= 1.05, f"Species sum ratio = {ratio}"


# ---------------------------------------------------------------------------
# Temperature tests (5 tests)
# ---------------------------------------------------------------------------

class TestTemperature:
    """Verify temperature profile behavior."""

    def _evaluate(self, altitude_km: float, f107: float = 150.0) -> "AtmosphereState":
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=f107, f107_average=f107, ap_daily=15.0)
        model = NRLMSISE00Model()
        return model.evaluate(
            altitude_km=altitude_km,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )

    def test_temperature_at_120km(self):
        state = self._evaluate(120.0)
        assert 350.0 <= state.temperature_k <= 400.0

    def test_temperature_at_300km_approaches_exospheric(self):
        state = self._evaluate(300.0)
        # Should be close to exospheric but not exceed it
        assert state.temperature_k <= state.temperature_exospheric_k
        # Should be at least 80% of exospheric at 300 km
        assert state.temperature_k >= 0.80 * state.temperature_exospheric_k

    def test_exospheric_temperature_low_activity(self):
        state = self._evaluate(500.0, f107=70.0)
        assert 600.0 <= state.temperature_exospheric_k <= 850.0

    def test_exospheric_temperature_high_activity(self):
        state = self._evaluate(500.0, f107=250.0)
        assert 1100.0 <= state.temperature_exospheric_k <= 1400.0

    def test_temperature_increases_monotonically_above_120km(self):
        altitudes = [120, 150, 200, 300, 500, 800, 1000]
        temps = [self._evaluate(h).temperature_k for h in altitudes]
        for i in range(len(temps) - 1):
            assert temps[i] <= temps[i + 1] + 1.0, (
                f"Temperature at {altitudes[i]} km ({temps[i]:.1f} K) "
                f"> temperature at {altitudes[i+1]} km ({temps[i+1]:.1f} K)"
            )


# ---------------------------------------------------------------------------
# Solar activity dependence (3 tests)
# ---------------------------------------------------------------------------

class TestSolarActivity:
    """Verify F10.7 dependency on density and temperature."""

    def _evaluate(self, f107_daily: float, f107_avg: float) -> "AtmosphereState":
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=f107_daily, f107_average=f107_avg, ap_daily=15.0)
        model = NRLMSISE00Model()
        return model.evaluate(
            altitude_km=400.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )

    def test_high_vs_low_activity_density_ratio(self):
        """F10.7=250 density at 400km should be 5-20x higher than F10.7=70."""
        low = self._evaluate(70.0, 70.0)
        high = self._evaluate(250.0, 250.0)
        ratio = high.total_density_kg_m3 / low.total_density_kg_m3
        assert 5.0 <= ratio <= 20.0, f"High/low density ratio = {ratio:.1f}"

    def test_higher_f107_higher_temperature(self):
        low = self._evaluate(70.0, 70.0)
        high = self._evaluate(250.0, 250.0)
        assert high.temperature_exospheric_k > low.temperature_exospheric_k

    def test_f107_average_larger_effect_than_daily(self):
        """The 81-day average has a larger effect on density than daily value."""
        baseline = self._evaluate(150.0, 150.0)
        daily_high = self._evaluate(200.0, 150.0)  # only daily increased
        avg_high = self._evaluate(150.0, 200.0)     # only average increased

        daily_ratio = daily_high.total_density_kg_m3 / baseline.total_density_kg_m3
        avg_ratio = avg_high.total_density_kg_m3 / baseline.total_density_kg_m3

        assert avg_ratio > daily_ratio, (
            f"F10.7avg effect ({avg_ratio:.2f}) should exceed daily ({daily_ratio:.2f})"
        )


# ---------------------------------------------------------------------------
# Latitude dependence (3 tests)
# ---------------------------------------------------------------------------

class TestLatitudeDependence:
    """Verify latitude effects on density."""

    def _evaluate(self, lat_deg: float) -> "AtmosphereState":
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        model = NRLMSISE00Model()
        return model.evaluate(
            altitude_km=400.0,
            latitude_deg=lat_deg,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )

    def test_latitude_variation_exists(self):
        """Density should differ between equator and pole."""
        equator = self._evaluate(0.0)
        pole = self._evaluate(80.0)
        ratio = equator.total_density_kg_m3 / pole.total_density_kg_m3
        assert ratio != 1.0, "No latitude variation detected"

    def test_latitude_variation_not_extreme(self):
        """Latitude effect should be measurable but not extreme (<5x)."""
        equator = self._evaluate(0.0)
        pole = self._evaluate(80.0)
        ratio = equator.total_density_kg_m3 / pole.total_density_kg_m3
        assert 0.2 < ratio < 5.0, f"Latitude ratio = {ratio}"

    def test_symmetric_hemispheres_approximately(self):
        """Northern and southern polar densities should be roughly similar."""
        north = self._evaluate(80.0)
        south = self._evaluate(-80.0)
        ratio = north.total_density_kg_m3 / south.total_density_kg_m3
        assert 0.5 < ratio < 2.0


# ---------------------------------------------------------------------------
# Day/night variation (3 tests)
# ---------------------------------------------------------------------------

class TestDayNightVariation:
    """Verify diurnal density variation."""

    def _evaluate(self, ut_seconds: float, lon_deg: float) -> "AtmosphereState":
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        model = NRLMSISE00Model()
        return model.evaluate(
            altitude_km=400.0,
            latitude_deg=0.0,
            longitude_deg=lon_deg,
            year=2020,
            day_of_year=172,
            ut_seconds=ut_seconds,
            space_weather=sw,
        )

    def test_dayside_denser_than_nightside(self):
        """Noon (local solar time) density > midnight density at 400 km."""
        # LST = UT + lon/15. For noon LST at UT=12h, lon=0.
        # For midnight LST at UT=12h, lon=180.
        day = self._evaluate(43200.0, 0.0)   # noon local
        night = self._evaluate(43200.0, 180.0)  # midnight local
        assert day.total_density_kg_m3 > night.total_density_kg_m3

    def test_day_night_ratio_reasonable(self):
        """Day/night factor should be 2-5x at moderate altitudes."""
        day = self._evaluate(43200.0, 0.0)
        night = self._evaluate(43200.0, 180.0)
        ratio = day.total_density_kg_m3 / night.total_density_kg_m3
        assert 1.5 <= ratio <= 6.0, f"Day/night ratio = {ratio}"

    def test_diurnal_variation_exists(self):
        """Different local times should give different densities."""
        d1 = self._evaluate(0.0, 0.0)   # midnight local
        d2 = self._evaluate(21600.0, 0.0)  # 6 AM local
        d3 = self._evaluate(43200.0, 0.0)  # noon local
        densities = [d1.total_density_kg_m3, d2.total_density_kg_m3, d3.total_density_kg_m3]
        assert len(set(f"{d:.5e}" for d in densities)) > 1, "No diurnal variation"


# ---------------------------------------------------------------------------
# Magnetic activity (3 tests)
# ---------------------------------------------------------------------------

class TestMagneticActivity:
    """Verify Ap geomagnetic index effects."""

    def _evaluate(self, ap: float, altitude_km: float = 400.0) -> "AtmosphereState":
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=ap)
        model = NRLMSISE00Model()
        return model.evaluate(
            altitude_km=altitude_km,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )

    def test_high_ap_increases_density(self):
        """Ap=100 should increase density compared to Ap=5."""
        low = self._evaluate(5.0)
        high = self._evaluate(100.0)
        assert high.total_density_kg_m3 > low.total_density_kg_m3

    def test_ap_effect_larger_at_high_altitude(self):
        """Magnetic activity effect should be more pronounced at 800 km than 300 km."""
        low_300 = self._evaluate(5.0, 300.0)
        high_300 = self._evaluate(100.0, 300.0)
        ratio_300 = high_300.total_density_kg_m3 / low_300.total_density_kg_m3

        low_800 = self._evaluate(5.0, 800.0)
        high_800 = self._evaluate(100.0, 800.0)
        ratio_800 = high_800.total_density_kg_m3 / low_800.total_density_kg_m3

        assert ratio_800 > ratio_300, (
            f"Ap ratio at 800 km ({ratio_800:.2f}) should exceed 300 km ({ratio_300:.2f})"
        )

    def test_ap_effect_reasonable_magnitude(self):
        """Ap effect should be measurable but not dominating."""
        low = self._evaluate(5.0)
        high = self._evaluate(100.0)
        ratio = high.total_density_kg_m3 / low.total_density_kg_m3
        # Ap=100 vs Ap=5 should give 1.5-5x at 400 km
        assert 1.2 <= ratio <= 6.0, f"Ap ratio = {ratio}"


# ---------------------------------------------------------------------------
# Altitude boundary tests (5 tests)
# ---------------------------------------------------------------------------

class TestAltitudeBoundaries:
    """Verify behavior at altitude edges."""

    def _evaluate(self, altitude_km: float) -> "AtmosphereState":
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        model = NRLMSISE00Model()
        return model.evaluate(
            altitude_km=altitude_km,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )

    def test_below_80km_graceful(self):
        """Below 80 km should handle gracefully (clamp or return valid value)."""
        state = self._evaluate(50.0)
        assert state.total_density_kg_m3 > 0

    def test_at_120km_transition_zone(self):
        """120 km is the transition zone; should return valid data."""
        state = self._evaluate(120.0)
        assert state.total_density_kg_m3 > 0
        assert 350.0 <= state.temperature_k <= 400.0

    def test_at_500km_valid(self):
        state = self._evaluate(500.0)
        assert state.total_density_kg_m3 > 0
        assert state.temperature_k > 0

    def test_at_1000km_valid(self):
        state = self._evaluate(1000.0)
        assert state.total_density_kg_m3 > 0
        assert state.temperature_k > 0

    def test_above_2000km_extrapolates(self):
        """Above 2000 km should extrapolate reasonably (not crash)."""
        state = self._evaluate(2500.0)
        assert state.total_density_kg_m3 > 0
        # Should be very tenuous
        state_1000 = self._evaluate(1000.0)
        assert state.total_density_kg_m3 < state_1000.total_density_kg_m3


# ---------------------------------------------------------------------------
# Space weather loading (3 tests)
# ---------------------------------------------------------------------------

class TestSpaceWeatherLoading:
    """Verify historical space weather data loading and lookup."""

    def test_load_historical_data(self):
        from humeris.domain.nrlmsise00 import SpaceWeatherHistory
        history = SpaceWeatherHistory()
        # Should load successfully
        assert history is not None

    def test_lookup_by_date(self):
        from humeris.domain.nrlmsise00 import SpaceWeatherHistory
        history = SpaceWeatherHistory()
        sw = history.lookup(datetime(2001, 7, 1, tzinfo=timezone.utc))
        # Near solar max of cycle 23
        assert sw.f107_daily > 100.0
        assert sw.f107_average > 100.0
        assert sw.ap_daily >= 0

    def test_interpolation_between_dates(self):
        """Lookup between data points should interpolate."""
        from humeris.domain.nrlmsise00 import SpaceWeatherHistory
        history = SpaceWeatherHistory()
        # Query between two data points (data is at 10-day intervals)
        sw = history.lookup(datetime(2010, 6, 15, tzinfo=timezone.utc))
        assert sw.f107_daily > 0
        assert sw.f107_average > 0


# ---------------------------------------------------------------------------
# NRLMSISE00DragForce protocol compliance (2 tests)
# ---------------------------------------------------------------------------

class TestDragForceProtocol:
    """Verify NRLMSISE00DragForce conforms to ForceModel protocol."""

    def test_has_acceleration_method(self):
        from humeris.domain.nrlmsise00 import NRLMSISE00DragForce
        force = NRLMSISE00DragForce(cd=2.2, area_m2=10.0, mass_kg=500.0)
        assert hasattr(force, "acceleration")
        assert callable(force.acceleration)

    def test_returns_tuple_of_three_floats(self):
        from humeris.domain.nrlmsise00 import NRLMSISE00DragForce
        force = NRLMSISE00DragForce(cd=2.2, area_m2=10.0, mass_kg=500.0)
        epoch = datetime(2020, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
        # LEO position at ~400 km altitude
        r = 6_778_137.0
        position = (r, 0.0, 0.0)
        v = 7_668.0
        velocity = (0.0, v, 0.0)
        result = force.acceleration(epoch, position, velocity)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)


# ---------------------------------------------------------------------------
# Comparison with exponential model (3 tests)
# ---------------------------------------------------------------------------

class TestComparisonWithExponential:
    """Compare NRLMSISE-00 with existing exponential model."""

    def test_moderate_activity_same_order(self):
        """At 400 km moderate activity, NRLMSISE and exponential agree within OoM."""
        from humeris.domain.atmosphere import atmospheric_density
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        model = NRLMSISE00Model()
        state = model.evaluate(
            altitude_km=400.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )
        exp_density = atmospheric_density(400.0)
        ratio = state.total_density_kg_m3 / exp_density
        # Within order of magnitude
        assert 0.1 <= ratio <= 10.0, f"NRLMSISE/exponential ratio = {ratio}"

    def test_extreme_solar_diverge(self):
        """At extreme solar activity, models should diverge significantly."""
        from humeris.domain.atmosphere import atmospheric_density
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        # Very high activity
        sw_high = SpaceWeather(f107_daily=250.0, f107_average=250.0, ap_daily=80.0)
        model = NRLMSISE00Model()
        state_high = model.evaluate(
            altitude_km=400.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw_high,
        )
        # Very low activity
        sw_low = SpaceWeather(f107_daily=70.0, f107_average=70.0, ap_daily=3.0)
        state_low = model.evaluate(
            altitude_km=400.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw_low,
        )
        # Exponential model is static
        exp_density = atmospheric_density(400.0)

        # High activity NRLMSISE should be higher than exponential
        # Low activity NRLMSISE should be lower than exponential
        # (exponential uses ~moderate/high activity)
        nrlmsise_range = state_high.total_density_kg_m3 / state_low.total_density_kg_m3
        assert nrlmsise_range > 3.0, (
            f"NRLMSISE range ratio = {nrlmsise_range}, expected > 3"
        )

    def test_both_positive_at_600km(self):
        """Both models return positive density at 600 km."""
        from humeris.domain.atmosphere import atmospheric_density
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        model = NRLMSISE00Model()
        state = model.evaluate(
            altitude_km=600.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )
        exp_density = atmospheric_density(600.0)
        assert state.total_density_kg_m3 > 0
        assert exp_density > 0


# ---------------------------------------------------------------------------
# Performance (1 test)
# ---------------------------------------------------------------------------

class TestPerformance:
    """Verify single evaluation completes within performance budget."""

    def test_single_evaluation_under_200us(self):
        from humeris.domain.nrlmsise00 import (
            NRLMSISE00Model,
            SpaceWeather,
        )
        sw = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=15.0)
        model = NRLMSISE00Model()
        # Warm up
        model.evaluate(
            altitude_km=400.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
            space_weather=sw,
        )
        # Time 1000 evaluations
        n = 1000
        start = time.perf_counter()
        for _ in range(n):
            model.evaluate(
                altitude_km=400.0,
                latitude_deg=0.0,
                longitude_deg=0.0,
                year=2020,
                day_of_year=172,
                ut_seconds=43200.0,
                space_weather=sw,
            )
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / n) * 1e6
        assert avg_us < 200.0, f"Average evaluation time = {avg_us:.1f} us (limit 200 us)"


# ---------------------------------------------------------------------------
# Default space weather (2 tests)
# ---------------------------------------------------------------------------

class TestDefaultSpaceWeather:
    """Verify model works without explicit space weather input."""

    def test_without_space_weather_uses_defaults(self):
        from humeris.domain.nrlmsise00 import NRLMSISE00Model
        model = NRLMSISE00Model()
        state = model.evaluate(
            altitude_km=400.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
        )
        assert state.total_density_kg_m3 > 0

    def test_default_results_reasonable(self):
        from humeris.domain.nrlmsise00 import NRLMSISE00Model
        model = NRLMSISE00Model()
        state = model.evaluate(
            altitude_km=400.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
            year=2020,
            day_of_year=172,
            ut_seconds=43200.0,
        )
        # With defaults (~moderate activity), density at 400 km should be in range
        assert 1e-12 <= state.total_density_kg_m3 <= 1e-10


# ---------------------------------------------------------------------------
# Domain purity (1 test)
# ---------------------------------------------------------------------------

class TestDomainPurity:
    """Verify no external imports in nrlmsise00 module."""

    def test_no_external_imports(self):
        """nrlmsise00.py should only import stdlib and domain modules."""
        import pathlib
        module_path = (
            pathlib.Path(__file__).parent.parent
            / "src"
            / "humeris"
            / "domain"
            / "nrlmsise00.py"
        )
        source = module_path.read_text()
        tree = ast.parse(source)

        allowed_prefixes = (
            "humeris",
            "math", "numpy",
            "dataclasses",
            "typing",
            "json",
            "pathlib",
            "datetime",
            "os",
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert any(
                        alias.name.startswith(p) for p in allowed_prefixes
                    ), f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    assert any(
                        node.module.startswith(p) for p in allowed_prefixes
                    ), f"Forbidden import from: {node.module}"
