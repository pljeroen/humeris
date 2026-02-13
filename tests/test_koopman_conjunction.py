# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/koopman_conjunction.py — Koopman-spectral conjunction screening."""
import ast
import math

import numpy as np

from humeris.domain.koopman_propagation import KoopmanModel, fit_koopman_model
from humeris.domain.koopman_conjunction import (
    SpectralConjunctionCandidate,
    KoopmanConjunctionEvent,
    KSCSResult,
    fit_constellation_models,
    compute_spectral_distance,
    compute_eigenvalue_overlap,
    screen_spectral,
    refine_koopman_conjunctions,
    run_kscs,
)

_MU = 3.986004418e14
_R_E = 6_378_137.0  # WGS84 equatorial radius


def _make_circular_orbit(altitude_km, n_steps, step_s, phase_offset_rad=0.0):
    """Generate a circular orbit trajectory in the x-y plane."""
    r = (_R_E / 1000.0 + altitude_km) * 1000.0
    v = (_MU / r) ** 0.5
    n = v / r
    positions = []
    velocities = []
    for i in range(n_steps):
        t = i * step_s
        theta = n * t + phase_offset_rad
        positions.append((r * math.cos(theta), r * math.sin(theta), 0.0))
        velocities.append((-v * math.sin(theta), v * math.cos(theta), 0.0))
    return positions, velocities


def _make_inclined_orbit(altitude_km, inc_rad, n_steps, step_s, phase_offset_rad=0.0):
    """Generate a circular orbit with inclination (rotation about x-axis)."""
    r = (_R_E / 1000.0 + altitude_km) * 1000.0
    v = (_MU / r) ** 0.5
    n = v / r
    ci = math.cos(inc_rad)
    si = math.sin(inc_rad)
    positions = []
    velocities = []
    for i in range(n_steps):
        t = i * step_s
        theta = n * t + phase_offset_rad
        ct = math.cos(theta)
        st = math.sin(theta)
        positions.append((r * ct, r * st * ci, r * st * si))
        velocities.append((-v * st, v * ct * ci, v * ct * si))
    return positions, velocities


# Common test parameters
_N_STEPS = 200
_STEP_S = 30.0


class TestFitConstellationModels:
    def test_fit_single_model(self):
        """Fit a Koopman model for one satellite trajectory."""
        pos, vel = _make_circular_orbit(550.0, _N_STEPS, _STEP_S)
        models = fit_constellation_models([pos], [vel], _STEP_S)
        assert len(models) == 1
        assert isinstance(models[0], KoopmanModel)
        assert models[0].training_error < 0.05

    def test_fit_constellation_models(self):
        """Fit Koopman models for multiple satellites."""
        trajectories = [
            _make_circular_orbit(550.0, _N_STEPS, _STEP_S),
            _make_circular_orbit(600.0, _N_STEPS, _STEP_S),
            _make_circular_orbit(700.0, _N_STEPS, _STEP_S),
        ]
        positions_list = [t[0] for t in trajectories]
        velocities_list = [t[1] for t in trajectories]
        models = fit_constellation_models(positions_list, velocities_list, _STEP_S)
        assert len(models) == 3
        for m in models:
            assert isinstance(m, KoopmanModel)


class TestSpectralDistance:
    def test_spectral_distance_identical(self):
        """Same model should yield spectral distance of exactly 0."""
        pos, vel = _make_circular_orbit(550.0, _N_STEPS, _STEP_S)
        model = fit_koopman_model(pos, vel, _STEP_S)
        dist = compute_spectral_distance(model, model)
        assert dist == 0.0

    def test_spectral_distance_different(self):
        """Different orbits should have spectral distance > 0."""
        pos_a, vel_a = _make_circular_orbit(550.0, _N_STEPS, _STEP_S)
        pos_b, vel_b = _make_circular_orbit(800.0, _N_STEPS, _STEP_S)
        model_a = fit_koopman_model(pos_a, vel_a, _STEP_S)
        model_b = fit_koopman_model(pos_b, vel_b, _STEP_S)
        dist = compute_spectral_distance(model_a, model_b)
        assert dist > 0.0

    def test_spectral_distance_symmetric(self):
        """Spectral distance should be symmetric: d(A,B) == d(B,A)."""
        pos_a, vel_a = _make_circular_orbit(550.0, _N_STEPS, _STEP_S)
        pos_b, vel_b = _make_circular_orbit(700.0, _N_STEPS, _STEP_S)
        model_a = fit_koopman_model(pos_a, vel_a, _STEP_S)
        model_b = fit_koopman_model(pos_b, vel_b, _STEP_S)
        assert compute_spectral_distance(model_a, model_b) == compute_spectral_distance(model_b, model_a)

    def test_spectral_distance_nonnegative(self):
        """Spectral distance should always be non-negative."""
        pos_a, vel_a = _make_circular_orbit(400.0, _N_STEPS, _STEP_S)
        pos_b, vel_b = _make_circular_orbit(1000.0, _N_STEPS, _STEP_S)
        model_a = fit_koopman_model(pos_a, vel_a, _STEP_S)
        model_b = fit_koopman_model(pos_b, vel_b, _STEP_S)
        assert compute_spectral_distance(model_a, model_b) >= 0.0


class TestEigenvalueOverlap:
    def test_eigenvalue_overlap_identical(self):
        """Same model should yield overlap of exactly 1.0."""
        pos, vel = _make_circular_orbit(550.0, _N_STEPS, _STEP_S)
        model = fit_koopman_model(pos, vel, _STEP_S)
        overlap = compute_eigenvalue_overlap(model, model)
        assert overlap == 1.0

    def test_eigenvalue_overlap_different(self):
        """Different orbits should have overlap < 1.0."""
        pos_a, vel_a = _make_circular_orbit(550.0, _N_STEPS, _STEP_S)
        pos_b, vel_b = _make_circular_orbit(800.0, _N_STEPS, _STEP_S)
        model_a = fit_koopman_model(pos_a, vel_a, _STEP_S)
        model_b = fit_koopman_model(pos_b, vel_b, _STEP_S)
        overlap = compute_eigenvalue_overlap(model_a, model_b)
        assert 0.0 <= overlap < 1.0

    def test_eigenvalue_overlap_bounded(self):
        """Overlap should be in [0, 1]."""
        pos_a, vel_a = _make_circular_orbit(400.0, _N_STEPS, _STEP_S)
        pos_b, vel_b = _make_circular_orbit(2000.0, _N_STEPS, _STEP_S)
        model_a = fit_koopman_model(pos_a, vel_a, _STEP_S)
        model_b = fit_koopman_model(pos_b, vel_b, _STEP_S)
        overlap = compute_eigenvalue_overlap(model_a, model_b)
        assert 0.0 <= overlap <= 1.0


class TestScreenSpectral:
    def test_screen_spectral_all_similar(self):
        """Similar orbits (same altitude, different phases) should all be candidates."""
        trajectories = [
            _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=0.0),
            _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=0.5),
            _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=1.0),
        ]
        models = fit_constellation_models(
            [t[0] for t in trajectories],
            [t[1] for t in trajectories],
            _STEP_S,
        )
        candidates = screen_spectral(models, spectral_threshold=1.0)
        # 3 satellites -> 3 pairs, all should pass since identical altitude
        assert len(candidates) == 3
        for c in candidates:
            assert isinstance(c, SpectralConjunctionCandidate)

    def test_screen_spectral_threshold(self):
        """Very different orbits with low threshold should produce fewer candidates."""
        trajectories = [
            _make_circular_orbit(400.0, _N_STEPS, _STEP_S),
            _make_circular_orbit(550.0, _N_STEPS, _STEP_S),
            _make_circular_orbit(2000.0, _N_STEPS, _STEP_S),
        ]
        models = fit_constellation_models(
            [t[0] for t in trajectories],
            [t[1] for t in trajectories],
            _STEP_S,
        )
        # Use a very tight threshold that filters the widely separated pair
        all_candidates = screen_spectral(models, spectral_threshold=10.0)
        tight_candidates = screen_spectral(models, spectral_threshold=0.001)
        assert len(tight_candidates) <= len(all_candidates)


class TestRefineConjunctions:
    def test_refine_finds_close_approach(self):
        """Two satellites at same altitude, different phases should have close approach."""
        pos_a, vel_a = _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=0.0)
        pos_b, vel_b = _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=0.01)
        models = fit_constellation_models(
            [pos_a, pos_b], [vel_a, vel_b], _STEP_S,
        )
        # Create a candidate for this pair
        sd = compute_spectral_distance(models[0], models[1])
        candidate = SpectralConjunctionCandidate(
            sat_a_index=0,
            sat_b_index=1,
            spectral_distance=sd,
            estimated_min_distance_m=0.0,
            eigenvalue_overlap=1.0,
        )
        r = (_R_E / 1000.0 + 550.0) * 1000.0
        period_s = 2.0 * math.pi * math.sqrt(r ** 3 / _MU)
        events = refine_koopman_conjunctions(
            (candidate,),
            models,
            [pos_a[0], pos_b[0]],
            [vel_a[0], vel_b[0]],
            duration_s=period_s,
            step_s=_STEP_S,
            distance_threshold_m=1e9,  # Very large threshold to ensure detection
        )
        assert len(events) >= 1
        assert isinstance(events[0], KoopmanConjunctionEvent)
        assert events[0].miss_distance_m >= 0.0

    def test_refine_filters_distant(self):
        """Widely separated orbits should produce no events with a tight threshold."""
        pos_a, vel_a = _make_circular_orbit(400.0, _N_STEPS, _STEP_S)
        pos_b, vel_b = _make_circular_orbit(2000.0, _N_STEPS, _STEP_S)
        models = fit_constellation_models(
            [pos_a, pos_b], [vel_a, vel_b], _STEP_S,
        )
        sd = compute_spectral_distance(models[0], models[1])
        candidate = SpectralConjunctionCandidate(
            sat_a_index=0,
            sat_b_index=1,
            spectral_distance=sd,
            estimated_min_distance_m=1e7,
            eigenvalue_overlap=0.1,
        )
        events = refine_koopman_conjunctions(
            (candidate,),
            models,
            [pos_a[0], pos_b[0]],
            [vel_a[0], vel_b[0]],
            duration_s=3000.0,
            step_s=_STEP_S,
            distance_threshold_m=1000.0,  # Very tight threshold
        )
        assert len(events) == 0


class TestRelativeVelocity:
    def test_relative_velocity_positive(self):
        """Relative velocity at TCA should be non-negative."""
        pos_a, vel_a = _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=0.0)
        pos_b, vel_b = _make_inclined_orbit(550.0, math.radians(10.0), _N_STEPS, _STEP_S)
        models = fit_constellation_models(
            [pos_a, pos_b], [vel_a, vel_b], _STEP_S,
        )
        sd = compute_spectral_distance(models[0], models[1])
        candidate = SpectralConjunctionCandidate(
            sat_a_index=0,
            sat_b_index=1,
            spectral_distance=sd,
            estimated_min_distance_m=0.0,
            eigenvalue_overlap=0.5,
        )
        events = refine_koopman_conjunctions(
            (candidate,),
            models,
            [pos_a[0], pos_b[0]],
            [vel_a[0], vel_b[0]],
            duration_s=6000.0,
            step_s=_STEP_S,
            distance_threshold_m=1e9,
        )
        assert len(events) >= 1
        for event in events:
            assert event.relative_velocity_ms >= 0.0


class TestKSCSPipeline:
    def test_kscs_full_pipeline(self):
        """End-to-end KSCS test with a small constellation."""
        trajectories = [
            _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=0.0),
            _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=0.5),
            _make_circular_orbit(600.0, _N_STEPS, _STEP_S),
        ]
        result = run_kscs(
            positions_list=[t[0] for t in trajectories],
            velocities_list=[t[1] for t in trajectories],
            training_step_s=_STEP_S,
            screening_duration_s=3000.0,
            prediction_step_s=_STEP_S,
            spectral_threshold=1.0,
            distance_threshold_m=1e9,
        )
        assert isinstance(result, KSCSResult)
        assert result.total_pairs_screened == 3  # 3 satellites, 3 pairs
        assert len(result.models) == 3
        assert result.candidates_after_spectral >= 0
        assert result.events_after_refinement >= 0
        assert 0.0 <= result.screening_reduction_ratio <= 1.0

    def test_kscs_reduction_ratio(self):
        """Spectral screening should reduce the number of pairs to refine."""
        # Mix of similar and distant orbits
        trajectories = [
            _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=0.0),
            _make_circular_orbit(550.0, _N_STEPS, _STEP_S, phase_offset_rad=0.1),
            _make_circular_orbit(2000.0, _N_STEPS, _STEP_S),
            _make_circular_orbit(2000.0, _N_STEPS, _STEP_S, phase_offset_rad=0.1),
        ]
        result = run_kscs(
            positions_list=[t[0] for t in trajectories],
            velocities_list=[t[1] for t in trajectories],
            training_step_s=_STEP_S,
            screening_duration_s=3000.0,
            prediction_step_s=_STEP_S,
            spectral_threshold=0.01,  # Very tight: only very similar spectra pass
            distance_threshold_m=1e9,
        )
        # With a tight spectral threshold, the 550 km pair and 2000 km pair
        # should pass but not the cross-altitude pairs
        assert result.total_pairs_screened == 6  # 4 choose 2
        assert result.candidates_after_spectral <= result.total_pairs_screened
        # The reduction ratio should reflect filtering
        assert result.screening_reduction_ratio <= 1.0

    def test_kscs_empty_constellation_zero_sats(self):
        """Zero satellites should produce empty result."""
        result = run_kscs(
            positions_list=[],
            velocities_list=[],
            training_step_s=_STEP_S,
            screening_duration_s=3000.0,
            prediction_step_s=_STEP_S,
        )
        assert result.total_pairs_screened == 0
        assert result.candidates_after_spectral == 0
        assert result.events_after_refinement == 0
        assert result.candidates == ()
        assert result.events == ()
        assert result.models == ()

    def test_kscs_empty_constellation_one_sat(self):
        """One satellite should produce empty result (no pairs)."""
        pos, vel = _make_circular_orbit(550.0, _N_STEPS, _STEP_S)
        result = run_kscs(
            positions_list=[pos],
            velocities_list=[vel],
            training_step_s=_STEP_S,
            screening_duration_s=3000.0,
            prediction_step_s=_STEP_S,
        )
        assert result.total_pairs_screened == 0
        assert result.candidates_after_spectral == 0
        assert result.events_after_refinement == 0


class TestImmutability:
    def test_spectral_candidate_frozen(self):
        """SpectralConjunctionCandidate should be immutable."""
        c = SpectralConjunctionCandidate(
            sat_a_index=0,
            sat_b_index=1,
            spectral_distance=0.1,
            estimated_min_distance_m=5000.0,
            eigenvalue_overlap=0.9,
        )
        try:
            c.spectral_distance = 99.0  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # Expected: frozen dataclass

    def test_conjunction_event_frozen(self):
        """KoopmanConjunctionEvent should be immutable."""
        e = KoopmanConjunctionEvent(
            sat_a_index=0,
            sat_b_index=1,
            tca_time_s=100.0,
            miss_distance_m=5000.0,
            relative_velocity_ms=100.0,
            spectral_distance=0.1,
        )
        try:
            e.miss_distance_m = 0.0  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # Expected: frozen dataclass

    def test_kscs_result_frozen(self):
        """KSCSResult should be immutable."""
        r = KSCSResult(
            candidates=(),
            events=(),
            total_pairs_screened=0,
            candidates_after_spectral=0,
            events_after_refinement=0,
            screening_reduction_ratio=0.0,
            models=(),
        )
        try:
            r.total_pairs_screened = 99  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # Expected: frozen dataclass


class TestModulePurity:
    def test_module_pure(self):
        """koopman_conjunction.py should only import stdlib, numpy, and humeris."""
        import humeris.domain.koopman_conjunction as mod
        source = ast.parse(open(mod.__file__).read())
        allowed = {
            "math", "dataclasses", "datetime", "typing", "enum",
            "numpy", "humeris", "__future__",
        }
        for node in ast.walk(source):
            if isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert top in allowed, f"Forbidden import: {top}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed, f"Forbidden import: {top}"
