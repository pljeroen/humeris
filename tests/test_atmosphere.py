# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for atmospheric density model and drag acceleration."""
import ast
import math

import pytest

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.atmosphere import AtmosphereModel


# ── DragConfig ──────────────────────────────────────────────────────

class TestDragConfig:

    def test_frozen(self):
        """DragConfig is immutable."""
        from constellation_generator.domain.atmosphere import DragConfig

        cfg = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        with pytest.raises(AttributeError):
            cfg.cd = 3.0

    def test_fields(self):
        """DragConfig exposes cd, area_m2, mass_kg."""
        from constellation_generator.domain.atmosphere import DragConfig

        cfg = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        assert cfg.cd == 2.2
        assert cfg.area_m2 == 10.0
        assert cfg.mass_kg == 260.0

    def test_ballistic_coefficient(self):
        """ballistic_coefficient = cd * area_m2 / mass_kg."""
        from constellation_generator.domain.atmosphere import DragConfig

        cfg = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        expected = 2.2 * 10.0 / 260.0
        assert abs(cfg.ballistic_coefficient - expected) < 1e-12


# ── Atmospheric density ─────────────────────────────────────────────

class TestAtmosphericDensity:

    def test_density_200km(self):
        """Density at 200 km ≈ 2.5e-10 kg/m³ (Vallado Table 8-4)."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        rho = atmospheric_density(200.0)
        assert abs(rho - 2.541e-10) / 2.541e-10 < 0.01

    def test_density_500km(self):
        """Density at 500 km ≈ 2.15e-12 kg/m³ (Vallado Table 8-4)."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        rho = atmospheric_density(500.0)
        assert abs(rho - 2.150e-12) / 2.150e-12 < 0.01

    def test_monotonically_decreasing(self):
        """Density decreases with altitude from 200 to 1000 km."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        altitudes = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        densities = [atmospheric_density(h) for h in altitudes]
        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1], (
                f"Density at {altitudes[i]} km not > density at {altitudes[i+1]} km"
            )

    def test_exact_table_boundary(self):
        """At an exact table boundary, density = table base value."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        rho = atmospheric_density(300.0)
        assert abs(rho - 2.508e-11) / 2.508e-11 < 1e-6

    def test_positive_everywhere(self):
        """Density is positive across entire valid range."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        for h in range(100, 2001, 50):
            rho = atmospheric_density(float(h))
            assert rho > 0, f"Non-positive density at {h} km"

    def test_below_100km_raises(self):
        """Below 100 km raises ValueError."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        with pytest.raises(ValueError):
            atmospheric_density(99.0)

    def test_above_2000km_raises(self):
        """Above 2000 km raises ValueError."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        with pytest.raises(ValueError):
            atmospheric_density(2001.0)


# ── Atmosphere model configurability ───────────────────────────────

class TestAtmosphereModel:

    def test_enum_values(self):
        """AtmosphereModel has VALLADO_4TH and HIGH_ACTIVITY values."""
        assert AtmosphereModel.VALLADO_4TH.value == "vallado_4th"
        assert AtmosphereModel.HIGH_ACTIVITY.value == "high_activity"

    def test_vallado_200km_spot_check(self):
        """Vallado 4th ed table at 200km: rho ≈ 2.789e-10."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        rho = atmospheric_density(200.0, model=AtmosphereModel.VALLADO_4TH)
        assert abs(rho - 2.789e-10) / 2.789e-10 < 0.01

    def test_vallado_500km_spot_check(self):
        """Vallado 4th ed table at 500km: rho ≈ 6.967e-13."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        rho = atmospheric_density(500.0, model=AtmosphereModel.VALLADO_4TH)
        assert abs(rho - 6.967e-13) / 6.967e-13 < 0.01

    def test_vallado_has_entries_110_to_140(self):
        """Vallado table covers 110-140km without gap (no ValueError)."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        for alt in [110, 120, 130, 140]:
            rho = atmospheric_density(float(alt), model=AtmosphereModel.VALLADO_4TH)
            assert rho > 0

    def test_high_activity_matches_original(self):
        """HIGH_ACTIVITY model gives same results as original default."""
        from constellation_generator.domain.atmosphere import atmospheric_density

        # These values match the original _ATMOSPHERE_TABLE
        rho = atmospheric_density(200.0, model=AtmosphereModel.HIGH_ACTIVITY)
        assert abs(rho - 2.541e-10) / 2.541e-10 < 0.01

    def test_model_in_semi_major_axis_decay_rate(self):
        """semi_major_axis_decay_rate accepts model parameter."""
        from constellation_generator.domain.atmosphere import (
            semi_major_axis_decay_rate, DragConfig,
        )

        cfg = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        a = OrbitalConstants.R_EARTH + 500_000
        rate_high = semi_major_axis_decay_rate(a, 0.0, cfg, model=AtmosphereModel.HIGH_ACTIVITY)
        rate_vallado = semi_major_axis_decay_rate(a, 0.0, cfg, model=AtmosphereModel.VALLADO_4TH)
        # Both should be negative
        assert rate_high < 0
        assert rate_vallado < 0
        # High activity should have more drag (more negative)
        assert rate_high < rate_vallado


# ── Drag acceleration ───────────────────────────────────────────────

class TestDragAcceleration:

    def test_known_value(self):
        """Drag acceleration = 0.5 * rho * v² * B_c for known inputs."""
        from constellation_generator.domain.atmosphere import drag_acceleration, DragConfig

        cfg = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        rho = 2.5e-10
        v = 7600.0  # m/s, typical LEO
        expected = 0.5 * rho * v**2 * cfg.ballistic_coefficient
        result = drag_acceleration(rho, v, cfg)
        assert abs(result - expected) < 1e-15

    def test_zero_density_zero_accel(self):
        """Zero density → zero drag acceleration."""
        from constellation_generator.domain.atmosphere import drag_acceleration, DragConfig

        cfg = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        assert drag_acceleration(0.0, 7600.0, cfg) == 0.0

    def test_higher_bc_more_drag(self):
        """Higher ballistic coefficient → more drag acceleration."""
        from constellation_generator.domain.atmosphere import drag_acceleration, DragConfig

        low_bc = DragConfig(cd=2.2, area_m2=5.0, mass_kg=260.0)
        high_bc = DragConfig(cd=2.2, area_m2=20.0, mass_kg=260.0)
        rho = 2.5e-10
        v = 7600.0
        assert drag_acceleration(rho, v, high_bc) > drag_acceleration(rho, v, low_bc)


# ── Semi-major axis decay rate ──────────────────────────────────────

class TestDecayRate:

    def test_negative_at_300km(self):
        """Decay rate is negative (orbit shrinks) at 300 km."""
        from constellation_generator.domain.atmosphere import semi_major_axis_decay_rate, DragConfig

        cfg = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        a = OrbitalConstants.R_EARTH + 300_000
        rate = semi_major_axis_decay_rate(a, 0.0, cfg)
        assert rate < 0

    def test_faster_at_300km_than_600km(self):
        """Decay is faster (more negative) at 300 km than 600 km."""
        from constellation_generator.domain.atmosphere import semi_major_axis_decay_rate, DragConfig

        cfg = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        a_300 = OrbitalConstants.R_EARTH + 300_000
        a_600 = OrbitalConstants.R_EARTH + 600_000
        rate_300 = semi_major_axis_decay_rate(a_300, 0.0, cfg)
        rate_600 = semi_major_axis_decay_rate(a_600, 0.0, cfg)
        assert rate_300 < rate_600  # more negative = faster decay


# ── Domain purity ───────────────────────────────────────────────────

class TestAtmospherePurity:

    def test_atmosphere_module_pure(self):
        """atmosphere.py must only import stdlib modules."""
        import constellation_generator.domain.atmosphere as mod

        allowed = {'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('constellation_generator'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'constellation_generator':
                        assert False, f"Disallowed import from '{node.module}'"
