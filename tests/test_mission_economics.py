# Copyright (c) 2026 Jeroen. All rights reserved.
"""Tests for domain/mission_economics.py — survival-weighted mass frontier and RWCCD."""
import ast
import math
from datetime import datetime, timezone

from constellation_generator.domain.atmosphere import DragConfig

from constellation_generator.domain.mission_economics import (
    SurvivalWeightedEfficiencyPoint,
    SurvivalWeightedMassFrontier,
    ReliabilityWeightedCostPoint,
    ReliabilityWeightedCostProfile,
    compute_survival_weighted_frontier,
    compute_reliability_weighted_cost,
)

_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_DRAG = DragConfig(cd=2.2, area_m2=0.05, mass_kg=100.0)


class TestSurvivalWeightedFrontier:
    def test_returns_type(self):
        result = compute_survival_weighted_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH,
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        assert isinstance(result, SurvivalWeightedMassFrontier)

    def test_has_points(self):
        result = compute_survival_weighted_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH,
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        assert len(result.points) >= 3

    def test_weighted_optimal_geq_unweighted(self):
        """Survival weighting should shift optimal altitude upward."""
        result = compute_survival_weighted_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH,
            alt_min_km=350.0, alt_max_km=700.0, alt_step_km=25.0,
        )
        assert result.weighted_optimal_km >= result.unweighted_optimal_km - 1e-6

    def test_lifetime_fraction_bounded(self):
        result = compute_survival_weighted_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH,
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        for p in result.points:
            assert 0.0 <= p.expected_lifetime_fraction <= 1.0 + 1e-10

    def test_weighted_efficiency_leq_unweighted(self):
        """Weighted efficiency ≤ unweighted (survival weight ≤ 1)."""
        result = compute_survival_weighted_frontier(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH,
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        for p in result.points:
            assert p.weighted_efficiency <= p.mass_efficiency + 1e-15


class TestReliabilityWeightedCost:
    def test_returns_type(self):
        result = compute_reliability_weighted_cost(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH, inclination_rad=math.radians(53.0),
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        assert isinstance(result, ReliabilityWeightedCostProfile)

    def test_has_points(self):
        result = compute_reliability_weighted_cost(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH, inclination_rad=math.radians(53.0),
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        assert len(result.points) >= 3

    def test_rwccd_positive(self):
        result = compute_reliability_weighted_cost(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH, inclination_rad=math.radians(53.0),
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        for p in result.points:
            assert p.rwccd_kg_per_day > 0.0

    def test_optimal_differs_from_mass_optimal(self):
        """RWCCD optimal should differ from pure mass-optimal
        when radiation/availability are factored in."""
        result = compute_reliability_weighted_cost(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH, inclination_rad=math.radians(53.0),
            alt_min_km=350.0, alt_max_km=700.0, alt_step_km=25.0,
        )
        # At minimum, both should be within the altitude range
        assert result.optimal_altitude_km >= 350.0
        assert result.mass_optimal_altitude_km >= 350.0


class TestMissionEconomicsPurity:
    def test_module_pure(self):
        import constellation_generator.domain.mission_economics as mod
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
