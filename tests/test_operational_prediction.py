# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/operational_prediction.py — EOL mode and maneuver contact feasibility."""
import ast
import math
from datetime import datetime, timezone, timedelta

from humeris.domain.propagation import OrbitalState
from humeris.domain.atmosphere import DragConfig
from humeris.domain.observation import GroundStation

from humeris.domain.operational_prediction import (
    EndOfLifePrediction,
    ManeuverContactWindow,
    ManeuverContactFeasibility,
    compute_end_of_life_mode,
    compute_maneuver_contact_feasibility,
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


def _make_survival_curve():
    from humeris.domain.lifetime import compute_orbit_lifetime
    from humeris.domain.statistical_analysis import compute_lifetime_survival_curve
    state = _state()
    lt = compute_orbit_lifetime(state.semi_major_axis_m, state.eccentricity, _drag(), _EPOCH)
    return compute_lifetime_survival_curve(lt)


def _make_availability():
    from humeris.domain.statistical_analysis import compute_mission_availability
    return compute_mission_availability(
        _state(), _drag(), _EPOCH,
        isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
        mission_years=5.0,
    )


def _make_propellant_profile():
    from humeris.domain.lifetime import compute_orbit_lifetime
    from humeris.domain.mission_analysis import compute_propellant_profile
    state = _state()
    lt = compute_orbit_lifetime(state.semi_major_axis_m, state.eccentricity, _drag(), _EPOCH)
    return compute_propellant_profile(
        lt.decay_profile, _drag(),
        isp_s=220.0, dry_mass_kg=400.0, propellant_budget_kg=50.0,
    )


class TestComputeEndOfLifeMode:
    def test_returns_type(self):
        result = compute_end_of_life_mode(
            _make_survival_curve(), _make_propellant_profile(), _make_availability(),
        )
        assert isinstance(result, EndOfLifePrediction)

    def test_mode_is_valid(self):
        result = compute_end_of_life_mode(
            _make_survival_curve(), _make_propellant_profile(), _make_availability(),
        )
        valid_modes = {"fuel_depletion", "reentry", "conjunction", "indeterminate"}
        assert result.end_of_life_mode in valid_modes

    def test_hazard_ratio_non_negative(self):
        result = compute_end_of_life_mode(
            _make_survival_curve(), _make_propellant_profile(), _make_availability(),
        )
        assert result.hazard_ratio_at_eol >= 0.0

    def test_deorbit_feasibility_boolean(self):
        result = compute_end_of_life_mode(
            _make_survival_curve(), _make_propellant_profile(), _make_availability(),
        )
        assert isinstance(result.controlled_deorbit_feasible, bool)


class TestComputeManeuverContactFeasibility:
    def test_returns_type(self):
        from humeris.domain.maintenance_planning import compute_maintenance_schedule
        state = _state()
        schedule = compute_maintenance_schedule(state, _drag(), _EPOCH)
        stations = [GroundStation(name="Test", lat_deg=52.0, lon_deg=4.0)]
        result = compute_maneuver_contact_feasibility(
            schedule, [state], stations, _EPOCH,
        )
        assert isinstance(result, ManeuverContactFeasibility)

    def test_feasibility_fraction_bounded(self):
        from humeris.domain.maintenance_planning import compute_maintenance_schedule
        state = _state()
        schedule = compute_maintenance_schedule(state, _drag(), _EPOCH)
        stations = [GroundStation(name="Test", lat_deg=52.0, lon_deg=4.0)]
        result = compute_maneuver_contact_feasibility(
            schedule, [state], stations, _EPOCH,
        )
        assert 0.0 <= result.feasibility_fraction <= 1.0

    def test_counts_consistent(self):
        from humeris.domain.maintenance_planning import compute_maintenance_schedule
        state = _state()
        schedule = compute_maintenance_schedule(state, _drag(), _EPOCH)
        stations = [GroundStation(name="Test", lat_deg=52.0, lon_deg=4.0)]
        result = compute_maneuver_contact_feasibility(
            schedule, [state], stations, _EPOCH,
        )
        assert result.feasible_count + result.infeasible_count == len(result.windows)

    def test_window_has_burn_info(self):
        from humeris.domain.maintenance_planning import compute_maintenance_schedule
        state = _state()
        schedule = compute_maintenance_schedule(state, _drag(), _EPOCH)
        stations = [GroundStation(name="Test", lat_deg=52.0, lon_deg=4.0)]
        result = compute_maneuver_contact_feasibility(
            schedule, [state], stations, _EPOCH,
        )
        if result.windows:
            w = result.windows[0]
            assert isinstance(w, ManeuverContactWindow)
            assert isinstance(w.burn_description, str)


class TestOperationalPredictionPurity:
    def test_no_external_deps(self):
        import humeris.domain.operational_prediction as mod
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())
        allowed = {"math", "numpy", "dataclasses", "datetime", "humeris"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed, f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed, f"Forbidden import from: {node.module}"
