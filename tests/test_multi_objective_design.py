# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for domain/multi_objective_design.py — Pareto surface and entropy-collision efficiency."""
import ast
import math
from datetime import datetime, timezone

from constellation_generator.domain.propagation import OrbitalState
from constellation_generator.domain.atmosphere import DragConfig

from constellation_generator.domain.multi_objective_design import (
    ParetoPoint,
    ParetoSurface,
    EntropyCollisionEfficiency,
    compute_pareto_surface,
    compute_entropy_collision_efficiency,
)

_MU = 3.986004418e14
_R_E = 6_371_000.0
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
_DRAG = DragConfig(cd=2.2, area_m2=0.05, mass_kg=100.0)


def _state(alt_km=550.0, inc_deg=53.0, raan_deg=0.0, ta_deg=0.0):
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=0.0,
        inclination_rad=math.radians(inc_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=0.0, true_anomaly_rad=math.radians(ta_deg),
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


def _small_constellation(alt_km=20200.0):
    """GPS-like constellation for DOP visibility."""
    return [
        _state(alt_km=alt_km, inc_deg=55.0, raan_deg=r, ta_deg=t)
        for r in [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        for t in [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    ]


class TestParetoSurface:
    def test_returns_type(self):
        result = compute_pareto_surface(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH, inclination_rad=math.radians(53.0),
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        assert isinstance(result, ParetoSurface)

    def test_has_points(self):
        result = compute_pareto_surface(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH, inclination_rad=math.radians(53.0),
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        assert len(result.points) >= 3

    def test_pareto_front_nonempty(self):
        result = compute_pareto_surface(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH, inclination_rad=math.radians(53.0),
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        assert result.num_pareto_optimal >= 1

    def test_pareto_points_nondominated(self):
        """No Pareto-optimal point should be dominated by another."""
        result = compute_pareto_surface(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH, inclination_rad=math.radians(53.0),
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        front = [p for p in result.points if p.is_pareto_optimal]
        for i, a in enumerate(front):
            for j, b in enumerate(front):
                if i == j:
                    continue
                # b should not dominate a
                dominates = (
                    b.information_efficiency >= a.information_efficiency
                    and b.controllability >= a.controllability
                    and b.mass_efficiency >= a.mass_efficiency
                    and (
                        b.information_efficiency > a.information_efficiency
                        or b.controllability > a.controllability
                        or b.mass_efficiency > a.mass_efficiency
                    )
                )
                assert not dominates

    def test_objectives_positive(self):
        result = compute_pareto_surface(
            drag_config=_DRAG, isp_s=300.0, dry_mass_kg=100.0,
            injection_altitude_km=200.0, mission_years=5.0, num_sats=24,
            epoch=_EPOCH, inclination_rad=math.radians(53.0),
            alt_min_km=400.0, alt_max_km=600.0, alt_step_km=50.0,
        )
        for p in result.points:
            assert p.information_efficiency >= 0.0
            assert p.controllability >= 0.0
            assert p.mass_efficiency >= 0.0


class TestEntropyCollisionEfficiency:
    def test_returns_type(self):
        states = [_state(alt_km=550.0, raan_deg=r) for r in [0.0, 90.0, 180.0, 270.0]]
        candidate = _state(alt_km=550.0, raan_deg=45.0)
        result = compute_entropy_collision_efficiency(
            states, candidate, _EPOCH,
            miss_distance_m=500.0, sigma_radial_m=100.0,
            sigma_cross_m=200.0, combined_radius_m=5.0,
        )
        assert isinstance(result, EntropyCollisionEfficiency)

    def test_ratio_positive(self):
        states = [_state(alt_km=550.0, raan_deg=r) for r in [0.0, 90.0, 180.0, 270.0]]
        candidate = _state(alt_km=550.0, raan_deg=45.0)
        result = compute_entropy_collision_efficiency(
            states, candidate, _EPOCH,
            miss_distance_m=500.0, sigma_radial_m=100.0,
            sigma_cross_m=200.0, combined_radius_m=5.0,
        )
        assert result.information_risk_ratio >= 0.0

    def test_entropy_gain_nonnegative(self):
        states = [_state(alt_km=550.0, raan_deg=r) for r in [0.0, 90.0, 180.0, 270.0]]
        candidate = _state(alt_km=550.0, raan_deg=45.0)
        result = compute_entropy_collision_efficiency(
            states, candidate, _EPOCH,
            miss_distance_m=500.0, sigma_radial_m=100.0,
            sigma_cross_m=200.0, combined_radius_m=5.0,
        )
        assert result.entropy_gain >= 0.0


class TestMultiObjectiveDesignPurity:
    def test_module_pure(self):
        import constellation_generator.domain.multi_objective_design as mod
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
