# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for parametric Walker constellation trade studies."""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator.domain.orbital_mechanics import OrbitalConstants


# ── Helpers ──────────────────────────────────────────────────────────

_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


# ── WalkerConfig dataclass ───────────────────────────────────────────

class TestWalkerConfig:

    def test_frozen(self):
        from constellation_generator.domain.trade_study import WalkerConfig

        wc = WalkerConfig(
            altitude_km=500.0, inclination_deg=53.0,
            num_planes=6, sats_per_plane=10, phase_factor=1,
        )
        with pytest.raises(AttributeError):
            wc.altitude_km = 600.0

    def test_fields(self):
        from constellation_generator.domain.trade_study import WalkerConfig

        wc = WalkerConfig(
            altitude_km=550.0, inclination_deg=97.5,
            num_planes=4, sats_per_plane=8, phase_factor=2,
        )
        assert wc.altitude_km == 550.0
        assert wc.inclination_deg == 97.5
        assert wc.num_planes == 4
        assert wc.sats_per_plane == 8
        assert wc.phase_factor == 2


# ── TradePoint dataclass ─────────────────────────────────────────────

class TestTradePoint:

    def test_frozen(self):
        from constellation_generator.domain.trade_study import WalkerConfig, TradePoint
        from constellation_generator.domain.revisit import CoverageResult

        wc = WalkerConfig(
            altitude_km=500.0, inclination_deg=53.0,
            num_planes=2, sats_per_plane=4, phase_factor=0,
        )
        cr = CoverageResult(
            analysis_duration_s=3600.0, num_grid_points=4, num_satellites=8,
            mean_coverage_fraction=0.5, min_coverage_fraction=0.1,
            mean_revisit_s=600.0, max_revisit_s=1200.0,
            mean_response_time_s=300.0, percent_coverage_single=25.0,
            point_results=(),
        )
        tp = TradePoint(config=wc, total_satellites=8, coverage=cr)
        with pytest.raises(AttributeError):
            tp.total_satellites = 10

    def test_fields(self):
        from constellation_generator.domain.trade_study import WalkerConfig, TradePoint
        from constellation_generator.domain.revisit import CoverageResult

        wc = WalkerConfig(
            altitude_km=500.0, inclination_deg=53.0,
            num_planes=2, sats_per_plane=4, phase_factor=0,
        )
        cr = CoverageResult(
            analysis_duration_s=3600.0, num_grid_points=4, num_satellites=8,
            mean_coverage_fraction=0.5, min_coverage_fraction=0.1,
            mean_revisit_s=600.0, max_revisit_s=1200.0,
            mean_response_time_s=300.0, percent_coverage_single=25.0,
            point_results=(),
        )
        tp = TradePoint(config=wc, total_satellites=8, coverage=cr)
        assert tp.total_satellites == 8
        assert tp.config.altitude_km == 500.0


# ── TradeStudyResult dataclass ───────────────────────────────────────

class TestTradeStudyResult:

    def test_frozen(self):
        from constellation_generator.domain.trade_study import TradeStudyResult

        tsr = TradeStudyResult(
            points=(), analysis_duration_s=3600.0, min_elevation_deg=10.0,
        )
        with pytest.raises(AttributeError):
            tsr.analysis_duration_s = 0.0

    def test_fields(self):
        from constellation_generator.domain.trade_study import TradeStudyResult

        tsr = TradeStudyResult(
            points=(), analysis_duration_s=7200.0, min_elevation_deg=5.0,
        )
        assert tsr.analysis_duration_s == 7200.0
        assert tsr.min_elevation_deg == 5.0
        assert tsr.points == ()


# ── generate_walker_configs ──────────────────────────────────────────

class TestGenerateWalkerConfigs:

    def test_single_values_one_config(self):
        from constellation_generator.domain.trade_study import generate_walker_configs

        configs = generate_walker_configs(
            altitude_range=(500.0,),
            inclination_range=(53.0,),
            planes_range=(4,),
            sats_per_plane_range=(8,),
        )
        assert len(configs) == 1
        assert configs[0].altitude_km == 500.0
        assert configs[0].phase_factor == 0

    def test_cartesian_product(self):
        """2 altitudes × 2 inclinations = 4 configs."""
        from constellation_generator.domain.trade_study import generate_walker_configs

        configs = generate_walker_configs(
            altitude_range=(500.0, 600.0),
            inclination_range=(53.0, 97.5),
            planes_range=(4,),
            sats_per_plane_range=(8,),
        )
        assert len(configs) == 4

    def test_validates_altitude_positive(self):
        from constellation_generator.domain.trade_study import generate_walker_configs

        with pytest.raises(ValueError):
            generate_walker_configs(
                altitude_range=(0.0,),
                inclination_range=(53.0,),
                planes_range=(4,),
                sats_per_plane_range=(8,),
            )

    def test_validates_planes_positive(self):
        from constellation_generator.domain.trade_study import generate_walker_configs

        with pytest.raises(ValueError):
            generate_walker_configs(
                altitude_range=(500.0,),
                inclination_range=(53.0,),
                planes_range=(0,),
                sats_per_plane_range=(8,),
            )

    def test_phase_factor_none_defaults_zero(self):
        from constellation_generator.domain.trade_study import generate_walker_configs

        configs = generate_walker_configs(
            altitude_range=(500.0,),
            inclination_range=(53.0,),
            planes_range=(4,),
            sats_per_plane_range=(8,),
            phase_factor_range=None,
        )
        assert all(c.phase_factor == 0 for c in configs)

    def test_phase_factor_range_included(self):
        from constellation_generator.domain.trade_study import generate_walker_configs

        configs = generate_walker_configs(
            altitude_range=(500.0,),
            inclination_range=(53.0,),
            planes_range=(4,),
            sats_per_plane_range=(8,),
            phase_factor_range=(0, 1),
        )
        assert len(configs) == 2
        assert configs[0].phase_factor == 0
        assert configs[1].phase_factor == 1


# ── run_walker_trade_study ───────────────────────────────────────────

class TestRunWalkerTradeStudy:

    def test_single_config_one_trade_point(self):
        from constellation_generator.domain.trade_study import (
            WalkerConfig, run_walker_trade_study,
        )

        config = WalkerConfig(
            altitude_km=500.0, inclination_deg=53.0,
            num_planes=2, sats_per_plane=3, phase_factor=0,
        )
        result = run_walker_trade_study(
            [config], _EPOCH, timedelta(hours=1), timedelta(seconds=120),
            min_elevation_deg=10.0,
            lat_step_deg=30.0, lon_step_deg=60.0,
        )
        assert len(result.points) == 1
        assert result.points[0].total_satellites == 6
        assert result.points[0].config == config

    def test_two_configs_two_trade_points(self):
        from constellation_generator.domain.trade_study import (
            WalkerConfig, run_walker_trade_study,
        )

        configs = [
            WalkerConfig(
                altitude_km=500.0, inclination_deg=53.0,
                num_planes=2, sats_per_plane=3, phase_factor=0,
            ),
            WalkerConfig(
                altitude_km=600.0, inclination_deg=53.0,
                num_planes=3, sats_per_plane=4, phase_factor=1,
            ),
        ]
        result = run_walker_trade_study(
            configs, _EPOCH, timedelta(hours=1), timedelta(seconds=120),
            min_elevation_deg=10.0,
            lat_step_deg=30.0, lon_step_deg=60.0,
        )
        assert len(result.points) == 2
        assert result.points[0].total_satellites == 6
        assert result.points[1].total_satellites == 12


# ── pareto_front_indices ─────────────────────────────────────────────

class TestParetoFrontIndices:

    def test_one_dominates_other(self):
        """Point (2, 2) dominates (3, 3) → only index 0 returned."""
        from constellation_generator.domain.trade_study import pareto_front_indices

        front = pareto_front_indices([2.0, 3.0], [2.0, 3.0])
        assert front == [0]

    def test_two_non_dominated(self):
        """(2, 4) and (4, 2) are non-dominated → both returned."""
        from constellation_generator.domain.trade_study import pareto_front_indices

        front = pareto_front_indices([2.0, 4.0], [4.0, 2.0])
        assert sorted(front) == [0, 1]

    def test_identical_points_one_returned(self):
        """Identical points: neither dominates the other → both are non-dominated."""
        from constellation_generator.domain.trade_study import pareto_front_indices

        front = pareto_front_indices([2.0, 2.0], [3.0, 3.0])
        # Both are non-dominated (neither strictly dominates the other)
        assert len(front) == 2

    def test_empty_input(self):
        from constellation_generator.domain.trade_study import pareto_front_indices

        front = pareto_front_indices([], [])
        assert front == []


# ── Domain purity ────────────────────────────────────────────────────

class TestTradeStudyPurity:

    def test_trade_study_imports_only_stdlib_and_domain(self):
        import constellation_generator.domain.trade_study as mod

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
