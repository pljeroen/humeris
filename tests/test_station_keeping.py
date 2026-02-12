# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for station-keeping delta-V budgets and propellant computation."""
import ast
import math

import pytest

from constellation_generator.domain.atmosphere import DragConfig


# ── Helpers ──────────────────────────────────────────────────────────

STARLINK_DRAG = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
G0 = 9.80665  # standard gravity m/s²


# ── StationKeepingConfig / StationKeepingBudget ──────────────────────

class TestStationKeepingConfig:

    def test_frozen(self):
        """StationKeepingConfig is immutable."""
        from constellation_generator.domain.station_keeping import StationKeepingConfig

        cfg = StationKeepingConfig(
            target_altitude_km=550, inclination_deg=53,
            drag_config=STARLINK_DRAG, isp_s=300,
            dry_mass_kg=250, propellant_mass_kg=10,
        )
        with pytest.raises(AttributeError):
            cfg.isp_s = 400

    def test_fields(self):
        """StationKeepingConfig exposes all fields."""
        from constellation_generator.domain.station_keeping import StationKeepingConfig

        cfg = StationKeepingConfig(
            target_altitude_km=550, inclination_deg=53,
            drag_config=STARLINK_DRAG, isp_s=300,
            dry_mass_kg=250, propellant_mass_kg=10,
        )
        assert cfg.target_altitude_km == 550
        assert cfg.inclination_deg == 53
        assert cfg.isp_s == 300
        assert cfg.dry_mass_kg == 250
        assert cfg.propellant_mass_kg == 10


class TestStationKeepingBudget:

    def test_frozen(self):
        """StationKeepingBudget is immutable."""
        from constellation_generator.domain.station_keeping import StationKeepingBudget

        b = StationKeepingBudget(
            drag_dv_per_year_ms=10.0, plane_dv_per_year_ms=2.0,
            total_dv_per_year_ms=12.0, propellant_per_year_kg=1.0,
            operational_lifetime_years=5.0, total_dv_capacity_ms=60.0,
        )
        with pytest.raises(AttributeError):
            b.total_dv_per_year_ms = 20.0


# ── drag_compensation_dv_per_year ────────────────────────────────────

class TestDragCompensationDv:

    def test_positive_at_300km(self):
        """Drag ΔV compensation is positive at 300 km."""
        from constellation_generator.domain.station_keeping import drag_compensation_dv_per_year

        dv = drag_compensation_dv_per_year(300.0, STARLINK_DRAG)
        assert dv > 0

    def test_higher_altitude_less_dv(self):
        """Higher altitude → less drag ΔV needed."""
        from constellation_generator.domain.station_keeping import drag_compensation_dv_per_year

        dv_300 = drag_compensation_dv_per_year(300.0, STARLINK_DRAG)
        dv_600 = drag_compensation_dv_per_year(600.0, STARLINK_DRAG)
        assert dv_300 > dv_600

    def test_higher_bc_more_dv(self):
        """Higher ballistic coefficient → more drag ΔV."""
        from constellation_generator.domain.station_keeping import drag_compensation_dv_per_year

        low_bc = DragConfig(cd=2.2, area_m2=5.0, mass_kg=260.0)
        high_bc = DragConfig(cd=2.2, area_m2=20.0, mass_kg=260.0)
        assert drag_compensation_dv_per_year(400.0, high_bc) > drag_compensation_dv_per_year(400.0, low_bc)

    def test_order_of_magnitude_550km(self):
        """Drag ΔV at 550 km with B_c=0.085 is order 10-200 m/s/yr."""
        from constellation_generator.domain.station_keeping import drag_compensation_dv_per_year

        dv = drag_compensation_dv_per_year(550.0, STARLINK_DRAG)
        assert 10.0 < dv < 500.0


# ── plane_maintenance_dv_per_year ────────────────────────────────────

class TestPlaneMaintenanceDv:

    def test_positive(self):
        """Plane maintenance ΔV is positive."""
        from constellation_generator.domain.station_keeping import plane_maintenance_dv_per_year

        dv = plane_maintenance_dv_per_year(53.0, 550.0)
        assert dv > 0

    def test_larger_correction_more_dv(self):
        """Larger inclination correction → more ΔV."""
        from constellation_generator.domain.station_keeping import plane_maintenance_dv_per_year

        dv_small = plane_maintenance_dv_per_year(53.0, 550.0, delta_inclination_deg=0.01)
        dv_large = plane_maintenance_dv_per_year(53.0, 550.0, delta_inclination_deg=0.1)
        assert dv_large > dv_small


# ── Tsiolkovsky ──────────────────────────────────────────────────────

class TestTsiolkovsky:

    def test_known_value(self):
        """Isp=300s, m0=500 kg, mf=400 kg → ΔV ≈ 656 m/s."""
        from constellation_generator.domain.station_keeping import tsiolkovsky_dv

        dv = tsiolkovsky_dv(300.0, 400.0, 100.0)
        expected = 300.0 * G0 * math.log(500.0 / 400.0)
        assert abs(dv - expected) < 0.01
        assert abs(dv - 656.0) < 10.0  # ~656 m/s

    def test_zero_propellant_zero_dv(self):
        """Zero propellant → zero ΔV."""
        from constellation_generator.domain.station_keeping import tsiolkovsky_dv

        assert tsiolkovsky_dv(300.0, 400.0, 0.0) == 0.0

    def test_negative_mass_raises(self):
        """Negative dry mass raises ValueError."""
        from constellation_generator.domain.station_keeping import tsiolkovsky_dv

        with pytest.raises(ValueError):
            tsiolkovsky_dv(300.0, -10.0, 100.0)

    def test_negative_propellant_raises(self):
        """Negative propellant raises ValueError."""
        from constellation_generator.domain.station_keeping import tsiolkovsky_dv

        with pytest.raises(ValueError):
            tsiolkovsky_dv(300.0, 400.0, -10.0)


# ── compute_station_keeping_budget ───────────────────────────────────

class TestComputeBudget:

    def test_total_equals_sum(self):
        """Total ΔV/year = drag + plane."""
        from constellation_generator.domain.station_keeping import (
            StationKeepingConfig, compute_station_keeping_budget,
        )

        cfg = StationKeepingConfig(
            target_altitude_km=550, inclination_deg=53,
            drag_config=STARLINK_DRAG, isp_s=300,
            dry_mass_kg=250, propellant_mass_kg=10,
        )
        budget = compute_station_keeping_budget(cfg)
        assert abs(budget.total_dv_per_year_ms - (budget.drag_dv_per_year_ms + budget.plane_dv_per_year_ms)) < 1e-10

    def test_lifetime_positive(self):
        """Operational lifetime is positive."""
        from constellation_generator.domain.station_keeping import (
            StationKeepingConfig, compute_station_keeping_budget,
        )

        cfg = StationKeepingConfig(
            target_altitude_km=550, inclination_deg=53,
            drag_config=STARLINK_DRAG, isp_s=300,
            dry_mass_kg=250, propellant_mass_kg=10,
        )
        budget = compute_station_keeping_budget(cfg)
        assert budget.operational_lifetime_years > 0

    def test_more_propellant_longer_life(self):
        """More propellant ≈ longer operational life."""
        from constellation_generator.domain.station_keeping import (
            StationKeepingConfig, compute_station_keeping_budget,
        )

        cfg_small = StationKeepingConfig(
            target_altitude_km=550, inclination_deg=53,
            drag_config=STARLINK_DRAG, isp_s=300,
            dry_mass_kg=250, propellant_mass_kg=5,
        )
        cfg_large = StationKeepingConfig(
            target_altitude_km=550, inclination_deg=53,
            drag_config=STARLINK_DRAG, isp_s=300,
            dry_mass_kg=250, propellant_mass_kg=50,
        )
        budget_small = compute_station_keeping_budget(cfg_small)
        budget_large = compute_station_keeping_budget(cfg_large)
        assert budget_large.operational_lifetime_years > budget_small.operational_lifetime_years


# ── Domain purity ───────────────────────────────────────────────────

class TestStationKeepingPurity:

    def test_station_keeping_module_pure(self):
        """station_keeping.py must only import stdlib modules."""
        import constellation_generator.domain.station_keeping as mod

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
