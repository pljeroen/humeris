# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Koopman-spectral conjunction screening (KSCS).

Rapid conjunction screening using Koopman operator spectral decomposition.
Lifts orbital trajectories to observable space where propagation is linear,
enabling O(N²) spectral distance screening instead of O(N² × T) pairwise
propagation.

The key insight: in Koopman observable space, conjunction geometry is
determined by eigenvalue alignment. Two orbits that share dominant Koopman
eigenvalues will periodically approach each other.

Uses numpy for eigenvalue decomposition and distance computations.
"""
import math
from dataclasses import dataclass

import numpy as np

from humeris.domain.koopman_propagation import (
    KoopmanModel,
    fit_koopman_model,
    predict_koopman,
)


@dataclass(frozen=True)
class SpectralConjunctionCandidate:
    """Candidate conjunction pair from spectral screening."""
    sat_a_index: int
    sat_b_index: int
    spectral_distance: float  # Distance in Koopman eigenvalue space
    estimated_min_distance_m: float  # Rough estimate from spectral analysis
    eigenvalue_overlap: float  # 0-1, how similar the Koopman spectra are


@dataclass(frozen=True)
class KoopmanConjunctionEvent:
    """Refined conjunction event from Koopman prediction."""
    sat_a_index: int
    sat_b_index: int
    tca_time_s: float  # Time of closest approach from start
    miss_distance_m: float  # Distance at TCA
    relative_velocity_ms: float  # Relative velocity at TCA
    spectral_distance: float  # Original spectral screening metric


@dataclass(frozen=True)
class KSCSResult:
    """Complete KSCS screening result."""
    candidates: tuple  # Tuple of SpectralConjunctionCandidate (pre-filter)
    events: tuple  # Tuple of KoopmanConjunctionEvent (refined)
    total_pairs_screened: int  # N*(N-1)/2
    candidates_after_spectral: int  # Pairs passing spectral filter
    events_after_refinement: int  # Pairs with actual close approaches
    screening_reduction_ratio: float  # candidates/total_pairs (lower = more efficient)
    models: tuple  # Tuple of KoopmanModel (one per satellite)


def fit_constellation_models(
    positions_list: list,
    velocities_list: list,
    step_s: float,
    n_observables: int = 12,
) -> tuple:
    """Fit Koopman models for multiple satellites.

    Args:
        positions_list: List of trajectories, each a list of (x, y, z) tuples.
        velocities_list: List of trajectories, each a list of (vx, vy, vz) tuples.
        step_s: Time step between consecutive snapshots in seconds.
        n_observables: Number of observables for each Koopman model (6-12).

    Returns:
        Tuple of KoopmanModel, one per satellite.
    """
    models = []
    for positions, velocities in zip(positions_list, velocities_list):
        model = fit_koopman_model(positions, velocities, step_s, n_observables)
        models.append(model)
    return tuple(models)


def _extract_eigenvalue_magnitudes(model: KoopmanModel) -> np.ndarray:
    """Extract sorted eigenvalue magnitudes (descending) from a Koopman model."""
    n_obs = model.n_observables
    k_matrix = np.array(model.koopman_matrix, dtype=np.float64).reshape(n_obs, n_obs)
    eigenvalues = np.linalg.eigvals(k_matrix)
    magnitudes = np.sort(np.abs(eigenvalues))[::-1]
    return magnitudes


def compute_spectral_distance(model_a: KoopmanModel, model_b: KoopmanModel) -> float:
    """Compute distance between two Koopman models in eigenvalue space.

    Uses Wasserstein-1 distance between sorted eigenvalue magnitude
    distributions. Lower distance means more similar dynamics and
    higher conjunction risk.

    Args:
        model_a: First satellite's Koopman model.
        model_b: Second satellite's Koopman model.

    Returns:
        Spectral distance (non-negative). 0.0 for identical spectra.
    """
    mag_a = _extract_eigenvalue_magnitudes(model_a)
    mag_b = _extract_eigenvalue_magnitudes(model_b)

    # Pad shorter vector with zeros if sizes differ
    max_len = max(len(mag_a), len(mag_b))
    if len(mag_a) < max_len:
        mag_a = np.pad(mag_a, (0, max_len - len(mag_a)))
    if len(mag_b) < max_len:
        mag_b = np.pad(mag_b, (0, max_len - len(mag_b)))

    return float(np.mean(np.abs(mag_a - mag_b)))


def compute_eigenvalue_overlap(model_a: KoopmanModel, model_b: KoopmanModel) -> float:
    """Compute overlap between dominant eigenvalues of two Koopman models.

    Sorts eigenvalues by magnitude, compares the top-k values.
    Returns a value in [0, 1] where 1.0 means identical spectra.

    Args:
        model_a: First satellite's Koopman model.
        model_b: Second satellite's Koopman model.

    Returns:
        Eigenvalue overlap in [0.0, 1.0].
    """
    mag_a = _extract_eigenvalue_magnitudes(model_a)
    mag_b = _extract_eigenvalue_magnitudes(model_b)

    k = min(len(mag_a), len(mag_b))
    if k == 0:
        return 0.0

    top_a = mag_a[:k]
    top_b = mag_b[:k]

    max_mag = max(float(np.max(top_a)), float(np.max(top_b)), 1e-15)
    mean_diff = float(np.mean(np.abs(top_a - top_b)))
    overlap = 1.0 - mean_diff / max_mag

    return max(0.0, min(1.0, overlap))


def screen_spectral(
    models: tuple,
    spectral_threshold: float = 0.5,
) -> tuple:
    """Screen all satellite pairs using spectral distance.

    Computes pairwise spectral distance between Koopman models and
    returns candidates where distance is below the threshold.

    Args:
        models: Tuple of KoopmanModel, one per satellite.
        spectral_threshold: Maximum spectral distance for a candidate pair.

    Returns:
        Tuple of SpectralConjunctionCandidate for pairs below threshold.
    """
    n = len(models)
    candidates = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_spectral_distance(models[i], models[j])
            if dist <= spectral_threshold:
                overlap = compute_eigenvalue_overlap(models[i], models[j])
                # Rough estimate: mean state distance as proxy for physical separation
                mean_a = np.array(models[i].mean_state[:3], dtype=np.float64)
                mean_b = np.array(models[j].mean_state[:3], dtype=np.float64)
                estimated_min = float(np.linalg.norm(mean_a - mean_b))

                candidates.append(SpectralConjunctionCandidate(
                    sat_a_index=i,
                    sat_b_index=j,
                    spectral_distance=dist,
                    estimated_min_distance_m=estimated_min,
                    eigenvalue_overlap=overlap,
                ))

    return tuple(candidates)


def refine_koopman_conjunctions(
    candidates: tuple,
    models: tuple,
    initial_positions: list,
    initial_velocities: list,
    duration_s: float,
    step_s: float,
    distance_threshold_m: float = 50000.0,
) -> tuple:
    """Refine spectral candidates using Koopman propagation.

    For each candidate pair, propagates both satellites using their
    Koopman models, finds the time of minimum distance, and filters
    by the distance threshold.

    Args:
        candidates: Tuple of SpectralConjunctionCandidate from screen_spectral.
        models: Tuple of KoopmanModel, one per satellite.
        initial_positions: List of initial (x, y, z) tuples per satellite.
        initial_velocities: List of initial (vx, vy, vz) tuples per satellite.
        duration_s: Screening duration in seconds.
        step_s: Prediction step size in seconds.
        distance_threshold_m: Maximum miss distance for a conjunction event.

    Returns:
        Tuple of KoopmanConjunctionEvent for pairs with close approaches.
    """
    events = []

    for candidate in candidates:
        idx_a = candidate.sat_a_index
        idx_b = candidate.sat_b_index

        pred_a = predict_koopman(
            models[idx_a],
            initial_positions[idx_a],
            initial_velocities[idx_a],
            duration_s,
            step_s,
        )
        pred_b = predict_koopman(
            models[idx_b],
            initial_positions[idx_b],
            initial_velocities[idx_b],
            duration_s,
            step_s,
        )

        # Find time of minimum distance
        n_steps = min(len(pred_a.positions_eci), len(pred_b.positions_eci))
        if n_steps == 0:
            continue

        pos_a = np.array(pred_a.positions_eci[:n_steps], dtype=np.float64)
        pos_b = np.array(pred_b.positions_eci[:n_steps], dtype=np.float64)
        diff = pos_a - pos_b
        distances = np.sqrt(np.sum(diff * diff, axis=1))

        min_idx = int(np.argmin(distances))
        min_dist = float(distances[min_idx])

        if min_dist <= distance_threshold_m:
            tca_time = float(pred_a.times_s[min_idx])

            # Compute relative velocity at TCA
            vel_a = np.array(pred_a.velocities_eci[min_idx], dtype=np.float64)
            vel_b = np.array(pred_b.velocities_eci[min_idx], dtype=np.float64)
            rel_vel = float(np.linalg.norm(vel_a - vel_b))

            events.append(KoopmanConjunctionEvent(
                sat_a_index=idx_a,
                sat_b_index=idx_b,
                tca_time_s=tca_time,
                miss_distance_m=min_dist,
                relative_velocity_ms=rel_vel,
                spectral_distance=candidate.spectral_distance,
            ))

    return tuple(events)


def run_kscs(
    positions_list: list,
    velocities_list: list,
    training_step_s: float,
    screening_duration_s: float,
    prediction_step_s: float,
    spectral_threshold: float = 0.5,
    distance_threshold_m: float = 50000.0,
    n_observables: int = 12,
) -> KSCSResult:
    """Run the complete Koopman-Spectral Conjunction Screening pipeline.

    Pipeline:
        1. Fit Koopman models for all satellites.
        2. Spectral screening: O(N^2) eigenvalue comparison (cheap).
        3. Koopman propagation for candidates only: O(K x T).
        4. Refine TCAs and filter by distance threshold.

    Args:
        positions_list: List of trajectories, each a list of (x, y, z) tuples.
        velocities_list: List of trajectories, each a list of (vx, vy, vz) tuples.
        training_step_s: Time step of the training snapshots in seconds.
        screening_duration_s: Duration of the screening window in seconds.
        prediction_step_s: Step size for Koopman prediction in seconds.
        spectral_threshold: Maximum spectral distance for candidate pairs.
        distance_threshold_m: Maximum miss distance for conjunction events.
        n_observables: Number of observables for Koopman models (6-12).

    Returns:
        KSCSResult with candidates, events, and screening statistics.
    """
    n_sats = len(positions_list)
    total_pairs = n_sats * (n_sats - 1) // 2

    if n_sats < 2:
        return KSCSResult(
            candidates=(),
            events=(),
            total_pairs_screened=total_pairs,
            candidates_after_spectral=0,
            events_after_refinement=0,
            screening_reduction_ratio=0.0,
            models=(),
        )

    # Step 1: Fit Koopman models
    models = fit_constellation_models(
        positions_list, velocities_list, training_step_s, n_observables,
    )

    # Step 2: Spectral screening
    candidates = screen_spectral(models, spectral_threshold)

    # Step 3: Koopman propagation and TCA refinement
    # Use the first position/velocity from each trajectory as initial conditions
    initial_positions = [pos_list[0] for pos_list in positions_list]
    initial_velocities = [vel_list[0] for vel_list in velocities_list]

    events = refine_koopman_conjunctions(
        candidates,
        models,
        initial_positions,
        initial_velocities,
        screening_duration_s,
        prediction_step_s,
        distance_threshold_m,
    )

    # Compute reduction ratio
    n_candidates = len(candidates)
    reduction = n_candidates / total_pairs if total_pairs > 0 else 0.0

    return KSCSResult(
        candidates=candidates,
        events=events,
        total_pairs_screened=total_pairs,
        candidates_after_spectral=n_candidates,
        events_after_refinement=len(events),
        screening_reduction_ratio=reduction,
        models=models,
    )
