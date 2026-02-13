# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for Kessler syndrome heat map."""
import ast
import math
from datetime import datetime, timezone

import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.propagation import OrbitalState
from humeris.domain.kessler_heatmap import (
    KesslerCell,
    KesslerHeatMap,
    KesslerPersistence,
    _compute_lyapunov_estimate,
    _PERCOLATION_THRESHOLD_2D,
    classify_kessler_risk,
    compute_kessler_heatmap,
    update_persistence,
)


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _circular_state(altitude_km, inclination_deg):
    """Create a circular OrbitalState."""
    a = OrbitalConstants.R_EARTH + altitude_km * 1000.0
    n = math.sqrt(OrbitalConstants.MU_EARTH / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a,
        eccentricity=0.0,
        inclination_rad=math.radians(inclination_deg),
        raan_rad=0.0,
        arg_perigee_rad=0.0,
        true_anomaly_rad=0.0,
        mean_motion_rad_s=n,
        reference_epoch=EPOCH,
    )


# ── classify_kessler_risk ───────────────────────────────────────────


class TestClassifyKesslerRisk:

    def test_low(self):
        assert classify_kessler_risk(1e-10) == "low"
        assert classify_kessler_risk(0.0) == "low"

    def test_moderate(self):
        assert classify_kessler_risk(5e-9) == "moderate"

    def test_high(self):
        assert classify_kessler_risk(5e-8) == "high"

    def test_critical(self):
        assert classify_kessler_risk(5e-7) == "critical"

    def test_boundary_moderate(self):
        """Exactly 1e-9 -> low (not moderate)."""
        assert classify_kessler_risk(1e-9) == "low"

    def test_boundary_high(self):
        """Exactly 1e-8 -> moderate (not high)."""
        assert classify_kessler_risk(1e-8) == "moderate"

    def test_boundary_critical(self):
        """Exactly 1e-7 -> high (not critical)."""
        assert classify_kessler_risk(1e-7) == "high"


# ── KesslerCell ─────────────────────────────────────────────────────


class TestKesslerCell:

    def test_frozen(self):
        cell = KesslerCell(
            altitude_min_km=200.0, altitude_max_km=250.0,
            inclination_min_deg=0.0, inclination_max_deg=10.0,
            object_count=5, spatial_density_per_km3=1e-10,
            mean_collision_velocity_ms=7500.0,
            risk_level="low",
        )
        with pytest.raises(AttributeError):
            cell.object_count = 10


# ── KesslerHeatMap ──────────────────────────────────────────────────


class TestKesslerHeatMap:

    def test_frozen(self):
        """KesslerHeatMap is immutable."""
        hm = compute_kessler_heatmap([])
        with pytest.raises(AttributeError):
            hm.total_objects = 99

    def test_new_fields_present(self):
        """KesslerHeatMap exposes cross-disciplinary metrics."""
        hm = compute_kessler_heatmap([])
        assert hasattr(hm, "population_entropy")
        assert hasattr(hm, "percolation_fraction")
        assert hasattr(hm, "is_percolation_risk")
        assert hasattr(hm, "cascade_k_eff")
        assert hasattr(hm, "is_supercritical")

    def test_empty_states_defaults(self):
        """Empty states produce zero-value cross-disciplinary metrics."""
        hm = compute_kessler_heatmap([])
        assert hm.population_entropy == 0.0
        assert hm.percolation_fraction == 0.0
        assert hm.is_percolation_risk is False
        assert hm.cascade_k_eff == 0.0
        assert hm.is_supercritical is False


# ── KesslerPersistence ──────────────────────────────────────────────


class TestKesslerPersistence:

    def test_frozen(self):
        p = KesslerPersistence(
            altitude_min_km=200.0, altitude_max_km=250.0,
            inclination_min_deg=0.0, inclination_max_deg=10.0,
            consecutive_high_count=2, is_chronic=False,
        )
        with pytest.raises(AttributeError):
            p.consecutive_high_count = 5

    def test_fields(self):
        p = KesslerPersistence(
            altitude_min_km=500.0, altitude_max_km=550.0,
            inclination_min_deg=50.0, inclination_max_deg=60.0,
            consecutive_high_count=3, is_chronic=True,
        )
        assert p.altitude_min_km == 500.0
        assert p.altitude_max_km == 550.0
        assert p.inclination_min_deg == 50.0
        assert p.inclination_max_deg == 60.0
        assert p.consecutive_high_count == 3
        assert p.is_chronic is True


# ── Volume fraction correction ──────────────────────────────────────


class TestVolumeFractionCorrection:

    def test_90_deg_band_larger_than_0_deg_band(self):
        """Band near 90 deg inclination has larger spherical zone volume
        than band near 0 deg, due to cos-based geometry.

        cos(80)-cos(90) = cos(80) ~ 0.174
        cos(0)-cos(10) = 1 - cos(10) ~ 0.015

        So the 80-90 deg band should have a MUCH larger volume fraction
        than the 0-10 deg band, approximately 10x or more.
        """
        # Put same number of objects in each band at same altitude
        states_equator = [_circular_state(550, 5) for _ in range(10)]
        states_polar = [_circular_state(550, 85) for _ in range(10)]

        hm_eq = compute_kessler_heatmap(
            states_equator, altitude_step_km=100.0, inclination_step_deg=10.0,
        )
        hm_po = compute_kessler_heatmap(
            states_polar, altitude_step_km=100.0, inclination_step_deg=10.0,
        )

        # Find the occupied cell in each heatmap
        cell_eq = [c for c in hm_eq.cells if c.object_count > 0][0]
        cell_po = [c for c in hm_po.cells if c.object_count > 0][0]

        # Same object count, but equatorial band has smaller volume, so
        # higher density. cos(0)-cos(10) << cos(80)-cos(90).
        assert cell_eq.spatial_density_per_km3 > cell_po.spatial_density_per_km3

        # The ratio should be approximately cos(80)-cos(90) / (1-cos(10))
        # = 0.1736 / 0.01519 ~ 11.4
        ratio = cell_eq.spatial_density_per_km3 / cell_po.spatial_density_per_km3
        assert ratio > 5.0, f"Expected large ratio from cos geometry, got {ratio}"


# ── Collision velocity formula ──────────────────────────────────────


class TestCollisionVelocity:

    def test_90_deg_inclination_near_v_circ_sqrt2(self):
        """At 90 deg inclination, V_rel = V_circ * sqrt(2) * sin(90) = V_circ * sqrt(2)."""
        states = [_circular_state(550, 90)]
        hm = compute_kessler_heatmap(
            states, altitude_step_km=1800.0, inclination_step_deg=180.0,
            altitude_min_km=200.0, altitude_max_km=2000.0,
        )

        # Single cell covering the full grid
        cell = [c for c in hm.cells if c.object_count > 0][0]

        # V_circ at midpoint altitude of the cell
        mid_alt_m = ((cell.altitude_min_km + cell.altitude_max_km) / 2.0) * 1000.0
        r = OrbitalConstants.R_EARTH + mid_alt_m
        v_circ = math.sqrt(OrbitalConstants.MU_EARTH / r)

        # Mid-inclination of cell [0,180] is 90 deg, sin(90)=1
        expected = v_circ * math.sqrt(2.0)
        assert cell.mean_collision_velocity_ms == pytest.approx(expected, rel=0.05)

    def test_high_inclination_much_faster_than_low(self):
        """90 deg inclination should give much higher collision velocity
        than near-equatorial, because V_rel ~ sin(i_mid)."""
        states_low = [_circular_state(550, 5)]
        states_high = [_circular_state(550, 85)]

        hm_low = compute_kessler_heatmap(
            states_low, altitude_step_km=100.0, inclination_step_deg=10.0,
        )
        hm_high = compute_kessler_heatmap(
            states_high, altitude_step_km=100.0, inclination_step_deg=10.0,
        )

        cell_low = [c for c in hm_low.cells if c.object_count > 0][0]
        cell_high = [c for c in hm_high.cells if c.object_count > 0][0]

        # sin(85 deg) / sin(5 deg) ~ 11.4, but minimum is 0.05, so
        # for a 0-10 deg band mid at 5 deg, sin(5)=0.087 > 0.05, still used.
        assert cell_high.mean_collision_velocity_ms > cell_low.mean_collision_velocity_ms * 5.0

    def test_near_equatorial_uses_minimum(self):
        """For very small inclination (< ~1 deg), sin(1 deg) minimum applies."""
        # Band 0-2 deg, mid = 1 deg, sin(1) = 0.01745 is the new minimum floor
        # (R3-17 fix: clip to sin(1 deg) to avoid zero velocity at equatorial orbits)
        states = [_circular_state(550, 1)]
        hm = compute_kessler_heatmap(
            states, altitude_step_km=1800.0, inclination_step_deg=2.0,
            altitude_min_km=200.0, altitude_max_km=2000.0,
        )

        cell = [c for c in hm.cells if c.object_count > 0][0]
        mid_alt_m = ((cell.altitude_min_km + cell.altitude_max_km) / 2.0) * 1000.0
        r = OrbitalConstants.R_EARTH + mid_alt_m
        v_circ = math.sqrt(OrbitalConstants.MU_EARTH / r)

        # Should use the sin(1 deg) minimum floor
        expected_clamped = v_circ * math.sqrt(2.0) * math.sin(math.radians(1.0))
        assert cell.mean_collision_velocity_ms == pytest.approx(expected_clamped, rel=0.01)

    def test_all_cells_positive_velocity(self):
        """Every cell has positive collision velocity."""
        states = [_circular_state(550, 53)]
        hm = compute_kessler_heatmap(states)
        for cell in hm.cells:
            assert cell.mean_collision_velocity_ms > 0

    def test_collision_velocity_below_escape(self):
        """Collision velocity stays below escape velocity (~20 km/s LEO)."""
        states = [_circular_state(550, 53)]
        hm = compute_kessler_heatmap(states)
        for cell in hm.cells:
            assert cell.mean_collision_velocity_ms < 20000.0


# ── compute_kessler_heatmap ─────────────────────────────────────────


class TestComputeKesslerHeatmap:

    def test_empty_states(self):
        """Empty states -> zero total objects."""
        hm = compute_kessler_heatmap([])
        assert hm.total_objects == 0
        assert len(hm.cells) > 0  # Grid still exists

    def test_single_object(self):
        """Single object at 550 km, 53 deg."""
        states = [_circular_state(550, 53)]
        hm = compute_kessler_heatmap(states)
        assert hm.total_objects == 1
        # Peak should be near 550 km altitude
        assert 500.0 <= hm.peak_density_altitude_km <= 600.0

    def test_concentrated_cluster(self):
        """Many objects at same altitude -> high peak density."""
        states = [_circular_state(550, 53) for _ in range(100)]
        hm = compute_kessler_heatmap(states)
        assert hm.total_objects == 100
        assert hm.peak_density_per_km3 > 0

    def test_grid_dimensions(self):
        """Grid has correct number of bins."""
        hm = compute_kessler_heatmap(
            [],
            altitude_step_km=100.0,
            inclination_step_deg=30.0,
            altitude_min_km=200.0,
            altitude_max_km=1000.0,
        )
        # (1000-200)/100 = 8 altitude bins
        # 180/30 = 6 inclination bins
        assert len(hm.altitude_bins_km) == 9  # 8 bins + 1 edge
        assert len(hm.inclination_bins_deg) == 7  # 6 bins + 1 edge
        assert len(hm.cells) == 8 * 6

    def test_objects_outside_range_excluded(self):
        """Objects outside altitude range are not counted."""
        states = [_circular_state(100, 53)]  # Below 200 km minimum
        hm = compute_kessler_heatmap(states, altitude_min_km=200.0)
        assert hm.total_objects == 0

    def test_two_altitude_bands(self):
        """Objects at different altitudes go to different bins."""
        states = [_circular_state(300, 53), _circular_state(700, 53)]
        hm = compute_kessler_heatmap(states, altitude_step_km=200.0)
        assert hm.total_objects == 2

        # Check that they're in different cells
        occupied = [c for c in hm.cells if c.object_count > 0]
        assert len(occupied) == 2

    def test_spatial_density_consistent(self):
        """Spatial density = count / volume, nonzero for occupied cells."""
        states = [_circular_state(550, 53) for _ in range(10)]
        hm = compute_kessler_heatmap(states)
        for cell in hm.cells:
            if cell.object_count > 0:
                assert cell.spatial_density_per_km3 > 0

    def test_risk_levels_consistent(self):
        """Risk levels match density thresholds."""
        states = [_circular_state(550, 53)]
        hm = compute_kessler_heatmap(states)
        for cell in hm.cells:
            expected = classify_kessler_risk(cell.spatial_density_per_km3)
            assert cell.risk_level == expected

    def test_multiple_inclinations(self):
        """Objects at different inclinations go to different bins."""
        states = [
            _circular_state(550, 10),
            _circular_state(550, 90),
            _circular_state(550, 170),
        ]
        hm = compute_kessler_heatmap(states, inclination_step_deg=30.0)
        assert hm.total_objects == 3

    def test_peak_density_identification(self):
        """Peak density correctly identifies the densest cell."""
        # Concentrate 50 objects at 550 km, 53 deg
        states = [_circular_state(550, 53) for _ in range(50)]
        # Add 1 object elsewhere
        states.append(_circular_state(800, 10))
        hm = compute_kessler_heatmap(states)
        assert hm.peak_density_altitude_km != 0.0
        assert hm.peak_density_per_km3 > 0


# ── Population entropy ──────────────────────────────────────────────


class TestPopulationEntropy:

    def test_concentrated_low_entropy(self):
        """All objects in a single cell -> low entropy (log2(1) = 0)."""
        states = [_circular_state(550, 53) for _ in range(50)]
        hm = compute_kessler_heatmap(states)
        assert hm.population_entropy == pytest.approx(0.0, abs=1e-12)

    def test_spread_higher_entropy(self):
        """Objects spread across multiple cells -> higher entropy."""
        # Spread objects across distinct altitude/inclination bins
        states = (
            [_circular_state(300, 10) for _ in range(25)]
            + [_circular_state(700, 50) for _ in range(25)]
            + [_circular_state(1100, 90) for _ in range(25)]
            + [_circular_state(1500, 130) for _ in range(25)]
        )
        hm = compute_kessler_heatmap(states, altitude_step_km=200.0, inclination_step_deg=30.0)
        # 4 equal groups -> entropy = log2(4) = 2.0
        assert hm.population_entropy == pytest.approx(2.0, abs=0.01)

    def test_empty_zero_entropy(self):
        """No objects -> entropy = 0."""
        hm = compute_kessler_heatmap([])
        assert hm.population_entropy == 0.0

    def test_two_equal_groups(self):
        """Two equal groups -> entropy = log2(2) = 1.0."""
        states = (
            [_circular_state(300, 10) for _ in range(30)]
            + [_circular_state(700, 50) for _ in range(30)]
        )
        hm = compute_kessler_heatmap(states, altitude_step_km=200.0, inclination_step_deg=30.0)
        assert hm.population_entropy == pytest.approx(1.0, abs=0.01)


# ── Percolation fraction ────────────────────────────────────────────


class TestPercolationFraction:

    def test_no_objects_zero_fraction(self):
        """Empty constellation -> no high/critical cells -> 0 fraction."""
        hm = compute_kessler_heatmap([])
        assert hm.percolation_fraction == 0.0
        assert hm.is_percolation_risk is False

    def test_fraction_is_ratio(self):
        """Percolation fraction is the ratio of high+critical cells to total cells."""
        states = [_circular_state(550, 53) for _ in range(10)]
        hm = compute_kessler_heatmap(states)
        n_elevated = sum(
            1 for c in hm.cells if c.risk_level in ("high", "critical")
        )
        expected = n_elevated / len(hm.cells)
        assert hm.percolation_fraction == pytest.approx(expected, abs=1e-12)

    def test_threshold_value(self):
        """Percolation threshold constant matches 2D square lattice."""
        assert _PERCOLATION_THRESHOLD_2D == pytest.approx(0.5927, abs=1e-4)

    def test_is_percolation_risk_tracks_threshold(self):
        """is_percolation_risk is True iff fraction >= threshold."""
        hm = compute_kessler_heatmap([])
        if hm.percolation_fraction >= _PERCOLATION_THRESHOLD_2D:
            assert hm.is_percolation_risk is True
        else:
            assert hm.is_percolation_risk is False


# ── k_eff computation ───────────────────────────────────────────────


class TestCascadeKEff:

    def test_empty_zero(self):
        """No objects -> k_eff = 0."""
        hm = compute_kessler_heatmap([])
        assert hm.cascade_k_eff == 0.0
        assert hm.is_supercritical is False

    def test_nonzero_with_objects(self):
        """With objects present, k_eff should be non-zero."""
        states = [_circular_state(550, 53) for _ in range(50)]
        hm = compute_kessler_heatmap(states)
        assert hm.cascade_k_eff > 0.0

    def test_supercritical_flag(self):
        """is_supercritical is True iff k_eff > 1."""
        states = [_circular_state(550, 53) for _ in range(10)]
        hm = compute_kessler_heatmap(states)
        assert hm.is_supercritical == (hm.cascade_k_eff > 1.0)

    def test_k_eff_scales_with_fragments(self):
        """Doubling mean_fragments_per_collision doubles k_eff."""
        states = [_circular_state(550, 53) for _ in range(20)]
        hm1 = compute_kessler_heatmap(
            states, mean_fragments_per_collision=100.0,
        )
        hm2 = compute_kessler_heatmap(
            states, mean_fragments_per_collision=200.0,
        )
        assert hm2.cascade_k_eff == pytest.approx(2.0 * hm1.cascade_k_eff, rel=1e-6)

    def test_k_eff_scales_with_cross_section(self):
        """Doubling mean_cross_section_km2 doubles k_eff."""
        states = [_circular_state(550, 53) for _ in range(20)]
        hm1 = compute_kessler_heatmap(
            states, mean_cross_section_km2=1e-6,
        )
        hm2 = compute_kessler_heatmap(
            states, mean_cross_section_km2=2e-6,
        )
        assert hm2.cascade_k_eff == pytest.approx(2.0 * hm1.cascade_k_eff, rel=1e-6)

    def test_k_eff_scales_with_lifetime(self):
        """Doubling orbital_lifetime_years doubles k_eff."""
        states = [_circular_state(550, 53) for _ in range(20)]
        hm1 = compute_kessler_heatmap(
            states, orbital_lifetime_years=25.0,
        )
        hm2 = compute_kessler_heatmap(
            states, orbital_lifetime_years=50.0,
        )
        assert hm2.cascade_k_eff == pytest.approx(2.0 * hm1.cascade_k_eff, rel=1e-6)


# ── update_persistence ──────────────────────────────────────────────


class TestUpdatePersistence:

    def _make_cell(self, risk_level):
        """Create a minimal KesslerCell with given risk level."""
        return KesslerCell(
            altitude_min_km=500.0, altitude_max_km=550.0,
            inclination_min_deg=50.0, inclination_max_deg=60.0,
            object_count=10, spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=7500.0,
            risk_level=risk_level,
        )

    def test_first_evaluation_high(self):
        """First evaluation: high cell -> count=1, not chronic."""
        cells = [self._make_cell("high")]
        result = update_persistence(cells, None)
        assert len(result) == 1
        assert result[0].consecutive_high_count == 1
        assert result[0].is_chronic is False

    def test_first_evaluation_low(self):
        """First evaluation: low cell -> count=0, not chronic."""
        cells = [self._make_cell("low")]
        result = update_persistence(cells, None)
        assert len(result) == 1
        assert result[0].consecutive_high_count == 0
        assert result[0].is_chronic is False

    def test_second_evaluation_continued_high(self):
        """Second evaluation: still high -> count=2, not yet chronic."""
        cells = [self._make_cell("critical")]

        # First evaluation
        p1 = update_persistence(cells, None)
        assert p1[0].consecutive_high_count == 1

        # Second evaluation (same cell still critical)
        p2 = update_persistence(cells, p1)
        assert p2[0].consecutive_high_count == 2
        assert p2[0].is_chronic is False

    def test_third_evaluation_becomes_chronic(self):
        """Third consecutive high/critical evaluation -> chronic."""
        cells = [self._make_cell("high")]

        p1 = update_persistence(cells, None)
        p2 = update_persistence(cells, p1)
        p3 = update_persistence(cells, p2)

        assert p3[0].consecutive_high_count == 3
        assert p3[0].is_chronic is True

    def test_de_escalation_resets_count(self):
        """If a cell drops to low/moderate, the counter resets to 0."""
        high_cell = [self._make_cell("high")]
        low_cell = [self._make_cell("low")]

        # Build up two consecutive high evaluations
        p1 = update_persistence(high_cell, None)
        p2 = update_persistence(high_cell, p1)
        assert p2[0].consecutive_high_count == 2

        # De-escalate: cell is now low risk
        p3 = update_persistence(low_cell, p2)
        assert p3[0].consecutive_high_count == 0
        assert p3[0].is_chronic is False

    def test_de_escalation_after_chronic_resets(self):
        """Even a chronic cell resets if risk drops."""
        cells_high = [self._make_cell("critical")]
        cells_low = [self._make_cell("moderate")]

        p1 = update_persistence(cells_high, None)
        p2 = update_persistence(cells_high, p1)
        p3 = update_persistence(cells_high, p2)
        assert p3[0].is_chronic is True

        # De-escalate
        p4 = update_persistence(cells_low, p3)
        assert p4[0].consecutive_high_count == 0
        assert p4[0].is_chronic is False

    def test_multiple_cells_independent(self):
        """Each cell tracks persistence independently."""
        cell_a = KesslerCell(
            altitude_min_km=500.0, altitude_max_km=550.0,
            inclination_min_deg=50.0, inclination_max_deg=60.0,
            object_count=10, spatial_density_per_km3=1e-8,
            mean_collision_velocity_ms=7500.0,
            risk_level="high",
        )
        cell_b = KesslerCell(
            altitude_min_km=600.0, altitude_max_km=650.0,
            inclination_min_deg=50.0, inclination_max_deg=60.0,
            object_count=1, spatial_density_per_km3=1e-11,
            mean_collision_velocity_ms=7400.0,
            risk_level="low",
        )

        p1 = update_persistence([cell_a, cell_b], None)
        assert p1[0].consecutive_high_count == 1  # cell_a: high
        assert p1[1].consecutive_high_count == 0  # cell_b: low


# ── _compute_lyapunov_estimate ──────────────────────────────────────


class TestComputeLyapunovEstimate:

    def test_growing_density_positive(self):
        """Increasing density -> positive Lyapunov exponent."""
        result = _compute_lyapunov_estimate(
            current_peak_density=2e-8,
            previous_peak_density=1e-8,
            time_interval_years=1.0,
        )
        assert result > 0.0
        # Should be ln(2)/1.0 = 0.693...
        assert result == pytest.approx(math.log(2.0), rel=1e-6)

    def test_declining_density_negative(self):
        """Decreasing density -> negative Lyapunov exponent."""
        result = _compute_lyapunov_estimate(
            current_peak_density=1e-8,
            previous_peak_density=2e-8,
            time_interval_years=1.0,
        )
        assert result < 0.0
        assert result == pytest.approx(-math.log(2.0), rel=1e-6)

    def test_stable_density_zero(self):
        """No change in density -> zero Lyapunov exponent."""
        result = _compute_lyapunov_estimate(
            current_peak_density=1e-8,
            previous_peak_density=1e-8,
            time_interval_years=1.0,
        )
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_zero_time_interval_returns_zero(self):
        """Zero time interval -> 0 (guard clause)."""
        result = _compute_lyapunov_estimate(
            current_peak_density=2e-8,
            previous_peak_density=1e-8,
            time_interval_years=0.0,
        )
        assert result == 0.0

    def test_negative_time_interval_returns_zero(self):
        """Negative time interval -> 0 (guard clause)."""
        result = _compute_lyapunov_estimate(
            current_peak_density=2e-8,
            previous_peak_density=1e-8,
            time_interval_years=-1.0,
        )
        assert result == 0.0

    def test_zero_current_density_returns_zero(self):
        """Zero current density -> 0 (guard clause)."""
        result = _compute_lyapunov_estimate(
            current_peak_density=0.0,
            previous_peak_density=1e-8,
            time_interval_years=1.0,
        )
        assert result == 0.0

    def test_zero_previous_density_returns_zero(self):
        """Zero previous density -> 0 (guard clause)."""
        result = _compute_lyapunov_estimate(
            current_peak_density=1e-8,
            previous_peak_density=0.0,
            time_interval_years=1.0,
        )
        assert result == 0.0

    def test_scales_inversely_with_time(self):
        """lambda = ln(ratio)/dt, so halving dt doubles lambda."""
        lam1 = _compute_lyapunov_estimate(2e-8, 1e-8, 2.0)
        lam2 = _compute_lyapunov_estimate(2e-8, 1e-8, 1.0)
        assert lam2 == pytest.approx(2.0 * lam1, rel=1e-6)


# ── Domain purity ───────────────────────────────────────────────────


class TestKesslerHeatmapPurity:

    def test_module_pure(self):
        """kessler_heatmap.py only imports stdlib + numpy + domain."""
        import humeris.domain.kessler_heatmap as mod

        allowed = {
            'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum',
            '__future__', 'datetime',
        }
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('humeris'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'humeris':
                        assert False, f"Disallowed import from '{node.module}'"
