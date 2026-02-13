# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for beta-angle thermal analysis."""

import ast
import math

import pytest

from constellation_generator.domain.thermal import (
    ThermalConfig,
    ThermalDangerZone,
    ThermalEquilibrium,
    ThermalProfile,
    compute_thermal_equilibrium,
    compute_thermal_extremes,
    flag_thermal_danger_zones,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def typical_config():
    """Typical LEO spacecraft thermal config."""
    return ThermalConfig(
        absorptivity=0.3,
        emissivity=0.8,
        solar_area_m2=2.0,
        radiator_area_m2=4.0,
        internal_power_w=100.0,
    )


@pytest.fixture
def black_body_config():
    """Ideal black body (alpha=1, epsilon=1)."""
    return ThermalConfig(
        absorptivity=1.0,
        emissivity=1.0,
        solar_area_m2=1.0,
        radiator_area_m2=1.0,
        internal_power_w=0.0,
    )


# ── Dataclass tests ────────────────────────────────────────────

class TestThermalDataclasses:

    def test_thermal_config_frozen(self):
        cfg = ThermalConfig(absorptivity=0.3, emissivity=0.8,
                            solar_area_m2=2.0, radiator_area_m2=4.0)
        with pytest.raises(AttributeError):
            cfg.absorptivity = 0.5

    def test_thermal_equilibrium_frozen(self):
        eq = ThermalEquilibrium(
            temperature_k=300.0, absorbed_power_w=100.0,
            emitted_power_w=100.0, is_sunlit=True, beta_deg=0.0,
        )
        with pytest.raises(AttributeError):
            eq.temperature_k = 0.0

    def test_danger_zone_frozen(self):
        dz = ThermalDangerZone(
            start_beta_deg=70.0, end_beta_deg=90.0,
            peak_temperature_k=350.0, reason="continuous_sunlight",
        )
        with pytest.raises(AttributeError):
            dz.reason = "other"


# ── Thermal equilibrium computation ────────────────────────────

class TestThermalEquilibrium:

    def test_black_body_at_1au(self, black_body_config):
        """Ideal black body at 1 AU ≈ 394 K (single face absorbing)."""
        eq = compute_thermal_equilibrium(
            black_body_config, eclipse_fraction=0.0,
            albedo_flux_w_m2=0.0, earth_ir_w_m2=0.0,
        )
        # T = (S / sigma)^0.25 ≈ 394 K for S=1361 W/m²
        assert 380 < eq.temperature_k < 410

    def test_eclipse_fraction_0_full_solar(self, typical_config):
        """Eclipse fraction 0 → full solar absorbed."""
        eq = compute_thermal_equilibrium(typical_config, eclipse_fraction=0.0)
        assert eq.absorbed_power_w > 0
        assert eq.is_sunlit is True

    def test_eclipse_fraction_1_no_direct_solar(self, typical_config):
        """Eclipse fraction 1 → only earth IR + internal power."""
        eq = compute_thermal_equilibrium(typical_config, eclipse_fraction=1.0)
        # Should still have temperature from Earth IR + internal
        assert eq.temperature_k > 0
        assert eq.is_sunlit is False

    def test_hot_hotter_than_cold(self, typical_config):
        """Full sun hotter than full eclipse."""
        hot = compute_thermal_equilibrium(typical_config, eclipse_fraction=0.0)
        cold = compute_thermal_equilibrium(typical_config, eclipse_fraction=1.0)
        assert hot.temperature_k > cold.temperature_k

    def test_temperature_decreases_with_eclipse_fraction(self, typical_config):
        """More eclipse → lower temperature."""
        t0 = compute_thermal_equilibrium(typical_config, eclipse_fraction=0.0).temperature_k
        t5 = compute_thermal_equilibrium(typical_config, eclipse_fraction=0.5).temperature_k
        t1 = compute_thermal_equilibrium(typical_config, eclipse_fraction=1.0).temperature_k
        assert t0 > t5 > t1

    def test_energy_balance(self, typical_config):
        """Absorbed ≈ emitted at equilibrium."""
        eq = compute_thermal_equilibrium(typical_config, eclipse_fraction=0.0)
        assert abs(eq.absorbed_power_w - eq.emitted_power_w) / eq.absorbed_power_w < 1e-10

    def test_internal_power_raises_temperature(self):
        """Internal dissipation increases equilibrium temperature."""
        cfg_no_power = ThermalConfig(
            absorptivity=0.3, emissivity=0.8,
            solar_area_m2=2.0, radiator_area_m2=4.0,
            internal_power_w=0.0,
        )
        cfg_with_power = ThermalConfig(
            absorptivity=0.3, emissivity=0.8,
            solar_area_m2=2.0, radiator_area_m2=4.0,
            internal_power_w=500.0,
        )
        t_no = compute_thermal_equilibrium(cfg_no_power).temperature_k
        t_with = compute_thermal_equilibrium(cfg_with_power).temperature_k
        assert t_with > t_no


# ── Config validation ──────────────────────────────────────────

class TestConfigValidation:

    def test_absorptivity_out_of_range(self):
        with pytest.raises(ValueError, match="absorptivity"):
            compute_thermal_equilibrium(ThermalConfig(
                absorptivity=1.5, emissivity=0.8,
                solar_area_m2=1.0, radiator_area_m2=1.0,
            ))

    def test_emissivity_zero(self):
        with pytest.raises(ValueError, match="emissivity"):
            compute_thermal_equilibrium(ThermalConfig(
                absorptivity=0.3, emissivity=0.0,
                solar_area_m2=1.0, radiator_area_m2=1.0,
            ))

    def test_negative_area(self):
        with pytest.raises(ValueError, match="solar_area_m2"):
            compute_thermal_equilibrium(ThermalConfig(
                absorptivity=0.3, emissivity=0.8,
                solar_area_m2=-1.0, radiator_area_m2=1.0,
            ))


# ── Thermal extremes ───────────────────────────────────────────

class TestThermalExtremes:

    def test_hot_case_hotter(self, typical_config):
        """Hot case temperature > cold case temperature."""
        profile = compute_thermal_extremes(typical_config)
        assert profile.hot_case.temperature_k > profile.cold_case.temperature_k

    def test_returns_thermal_profile(self, typical_config):
        """Returns ThermalProfile type."""
        profile = compute_thermal_extremes(typical_config)
        assert isinstance(profile, ThermalProfile)

    def test_danger_zones_present(self, typical_config):
        """Danger zones flagged for high beta."""
        profile = compute_thermal_extremes(typical_config)
        assert len(profile.danger_zones) > 0

    def test_hot_case_beta_90(self, typical_config):
        """Hot case is at beta = 90°."""
        profile = compute_thermal_extremes(typical_config)
        assert profile.hot_case.beta_deg == 90.0


# ── Danger zone flagging ──────────────────────────────────────

class TestDangerZones:

    def test_low_inclination_no_danger(self):
        """Low-inclination orbit (always has eclipses) → no danger zones."""
        # Beta stays below 70° for equatorial
        history = [(float(i), 10.0 * math.sin(2 * math.pi * i / 365))
                    for i in range(365)]
        zones = flag_thermal_danger_zones(history)
        assert len(zones) == 0

    def test_high_beta_flagged(self):
        """Beta > 70° creates a danger zone."""
        history = [(float(i), 80.0) for i in range(100)]
        zones = flag_thermal_danger_zones(history)
        assert len(zones) > 0

    def test_intermittent_danger_zones(self):
        """Beta crossing threshold creates multiple zones."""
        history = []
        for i in range(360):
            beta = 40.0 + 40.0 * math.sin(2 * math.pi * i / 180)
            history.append((float(i), beta))
        zones = flag_thermal_danger_zones(history)
        assert len(zones) >= 2


# ── Domain purity ─────────────────────────────────────────────

class TestThermalPurity:

    def test_thermal_module_pure(self):
        """thermal.py must only import stdlib modules."""
        import constellation_generator.domain.thermal as mod

        allowed = {'math', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
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
