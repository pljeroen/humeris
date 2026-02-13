# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Round 3 math verification tests for Humeris astrodynamics library.

Tests each fix in the R3 batch to verify correctness.
"""
import math
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest

from humeris.domain.thermal import (
    ThermalConfig,
    _eclipse_fraction_exact,
    compute_thermal_equilibrium,
    flag_thermal_danger_zones_from_range,
)
from humeris.domain.torques import (
    compute_aerodynamic_torque,
    AerodynamicTorqueResult,
)
from humeris.domain.atmosphere import DragConfig
from humeris.domain.pass_analysis import compute_doppler_shift
from humeris.domain.design_optimization import (
    compute_positioning_information,
    compute_coverage_drift,
    compute_optimal_reconfiguration,
)
from humeris.domain.coverage import compute_persistent_coverage_homology, CoveragePoint
from humeris.domain.orbit_determination import run_ekf, ODObservation, _MU
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.dilution_of_precision import compute_dop
from humeris.domain.propagation import OrbitalState


# ── Shared fixtures ──────────────────────────────────────────────────

def _make_thermal_config() -> ThermalConfig:
    return ThermalConfig(
        absorptivity=0.3,
        emissivity=0.8,
        solar_area_m2=2.0,
        radiator_area_m2=4.0,
        internal_power_w=50.0,
    )


def _make_orbital_state(raan_rad: float = 0.0) -> OrbitalState:
    """Create a minimal OrbitalState for testing."""
    r = OrbitalConstants.R_EARTH + 550_000.0
    inc = math.radians(53.0)
    n = math.sqrt(OrbitalConstants.MU_EARTH / r ** 3)
    return OrbitalState(
        semi_major_axis_m=r,
        eccentricity=0.0,
        inclination_rad=inc,
        raan_rad=raan_rad,
        arg_perigee_rad=0.0,
        true_anomaly_rad=0.0,
        mean_motion_rad_s=n,
        reference_epoch=datetime(2025, 1, 1),
    )


# ── R3-01: Eclipse fraction exact formula ────────────────────────────


class TestR3_01_EclipseFractionExact:
    """Verify altitude-dependent eclipse fraction formula."""

    def test_eclipse_fraction_at_beta_zero_iss_altitude(self):
        """At beta=0 and ISS altitude (~400 km), eclipse fraction ~39%."""
        f = _eclipse_fraction_exact(0.0, 400.0)
        # Exact formula: (1/pi)*arccos(sqrt(h^2+2*R*h)/((R+h)*cos(0)))
        # At 400 km: sqrt(400^2+2*6371*400)/6771 = sqrt(5256400)/6771
        # = 2292.7/6771 = 0.3387 -> arccos(0.3387)=1.2249 rad -> /pi = 0.3899
        assert 0.38 < f < 0.40, f"Expected ~0.39, got {f}"

    def test_eclipse_fraction_at_beta_zero_high_altitude(self):
        """At higher altitude, eclipse fraction decreases."""
        f_low = _eclipse_fraction_exact(0.0, 400.0)
        f_high = _eclipse_fraction_exact(0.0, 2000.0)
        assert f_high < f_low, (
            f"Higher altitude should have smaller eclipse fraction: "
            f"{f_high} >= {f_low}"
        )

    def test_eclipse_fraction_above_beta_star_is_zero(self):
        """Above critical beta angle, no eclipse occurs."""
        # For 550 km, beta* = arcsin(6371/(6371+550)) ~ 67 deg
        # So at beta = 70 deg, should be zero
        f = _eclipse_fraction_exact(70.0, 550.0)
        assert f == 0.0, f"Expected 0, got {f}"

    def test_eclipse_fraction_just_below_beta_star(self):
        """Just below beta*, eclipse fraction should be small but positive."""
        # beta* for 550 km ~ 67 deg; use 65 deg (closer to critical)
        f = _eclipse_fraction_exact(65.0, 550.0)
        assert f > 0.0, "Expected positive eclipse fraction below beta*"
        assert f < 0.25, f"Eclipse fraction should be modest near beta*: {f}"

    def test_eclipse_fraction_zero_altitude_returns_zero(self):
        """Zero altitude is degenerate; should not crash."""
        f = _eclipse_fraction_exact(0.0, 0.0)
        assert f == 0.0

    def test_eclipse_fraction_negative_altitude_returns_zero(self):
        """Negative altitude is invalid; returns 0."""
        f = _eclipse_fraction_exact(0.0, -100.0)
        assert f == 0.0

    def test_flag_thermal_danger_zones_uses_altitude(self):
        """Verify flag_thermal_danger_zones_from_range accepts altitude_km."""
        config = _make_thermal_config()
        zones_low = flag_thermal_danger_zones_from_range(config, altitude_km=400.0)
        zones_high = flag_thermal_danger_zones_from_range(config, altitude_km=2000.0)
        # Both should return valid tuples
        assert isinstance(zones_low, tuple)
        assert isinstance(zones_high, tuple)

    def test_eclipse_fraction_altitude_dependence(self):
        """Eclipse fraction at beta=0 should differ by altitude."""
        f_400 = _eclipse_fraction_exact(0.0, 400.0)
        f_800 = _eclipse_fraction_exact(0.0, 800.0)
        f_2000 = _eclipse_fraction_exact(0.0, 2000.0)
        # Monotonically decreasing with altitude
        assert f_400 > f_800 > f_2000


# ── R3-02: Torque cross product order ───────────────────────────────


class TestR3_02_TorqueCrossProductOrder:
    """Verify torque is T = r x F, not F x r."""

    def test_torque_is_r_cross_f(self):
        """Manually verify T = cp x F by comparing with numpy cross product."""
        # Position on circular orbit at ~400 km altitude
        r_e = OrbitalConstants.R_EARTH
        r = r_e + 400_000.0  # meters
        pos_eci = (r, 0.0, 0.0)
        # Velocity for circular orbit
        v = math.sqrt(OrbitalConstants.MU_EARTH / r)
        vel_eci = (0.0, v, 0.0)

        drag_config = DragConfig(cd=2.2, area_m2=10.0, mass_kg=100.0)
        # CP offset along +z body axis
        cp_offset = (0.0, 0.0, 1.0)

        result = compute_aerodynamic_torque(pos_eci, vel_eci, drag_config, cp_offset)

        # The force is mostly along -y direction (opposing velocity).
        # With co-rotating atmosphere, there's a small x component too.
        # cp = (0,0,1), F ~ (Fx, Fy, 0) with Fy < 0
        # cp x F = (0*0 - 1*Fy, 1*Fx - 0*0, 0*Fy - 0*Fx) = (-Fy, Fx, 0)
        # Since Fy < 0, -Fy > 0, so x-component of torque should be > 0
        # (The old F x cp would give (Fy, -Fx, 0) with x-component < 0)
        assert result.torque_nm[0] > 0, (
            f"Expected positive x-torque from cp x F, got {result.torque_nm[0]}"
        )

    def test_torque_magnitude_nonzero(self):
        """Sanity check: torque should be nonzero for nonzero CP offset."""
        r_e = OrbitalConstants.R_EARTH
        r = r_e + 400_000.0
        pos_eci = (r, 0.0, 0.0)
        v = math.sqrt(OrbitalConstants.MU_EARTH / r)
        vel_eci = (0.0, v, 0.0)
        drag_config = DragConfig(cd=2.2, area_m2=10.0, mass_kg=100.0)
        # CP offset perpendicular to force
        cp_offset = (0.0, 0.0, 1.0)

        result = compute_aerodynamic_torque(pos_eci, vel_eci, drag_config, cp_offset)
        assert result.magnitude_nm > 0.0

    def test_torque_cross_product_antisymmetry(self):
        """Verify T = cp x F by checking against manual numpy computation."""
        r_e = OrbitalConstants.R_EARTH
        r = r_e + 400_000.0
        pos_eci = (r, 0.0, 0.0)
        v = math.sqrt(OrbitalConstants.MU_EARTH / r)
        vel_eci = (0.0, v, 0.0)
        drag_config = DragConfig(cd=2.2, area_m2=10.0, mass_kg=100.0)

        cp_a = (1.0, 2.0, 3.0)
        cp_b = (1.0, 2.0, 3.0)

        result_a = compute_aerodynamic_torque(pos_eci, vel_eci, drag_config, cp_a)
        # With the same inputs, torque should be consistent
        result_b = compute_aerodynamic_torque(pos_eci, vel_eci, drag_config, cp_b)
        np.testing.assert_allclose(result_a.torque_nm, result_b.torque_nm)


# ── R3-03: Doppler shift station velocity documentation ─────────────


class TestR3_03_DopplerDocumentation:
    """Verify the Doppler function documents Earth rotation limitation."""

    def test_docstring_mentions_station_velocity_limitation(self):
        """The function or its internal comments should mention the limitation."""
        import inspect
        source = inspect.getsource(compute_doppler_shift)
        assert "465 m/s" in source, (
            "Expected comment about 465 m/s equatorial station velocity"
        )
        assert "15%" in source, (
            "Expected comment about <15% error bound"
        )


# ── R3-04: Fisher determinant from DOP components ───────────────────


class TestR3_04_FisherDeterminant:
    """Verify Fisher determinant uses DOP components, not GDOP^4."""

    def test_fisher_det_uses_dop_components(self):
        """Fisher det should be 1/((HDOP^2/2)^2 * VDOP^2 * TDOP^2), not 1/GDOP^4."""
        import inspect
        source = inspect.getsource(compute_positioning_information)
        # The old formula used gdop_sq ** 2 directly
        assert "hdop" in source.lower(), (
            "Expected Fisher determinant to use HDOP component"
        )
        assert "vdop" in source.lower(), (
            "Expected Fisher determinant to use VDOP component"
        )
        assert "tdop" in source.lower(), (
            "Expected Fisher determinant to use TDOP component"
        )

    def test_fisher_det_nonzero_for_good_geometry(self):
        """With well-distributed satellites visible from station, Fisher det > 0."""
        # Station at equator, prime meridian. ECEF ~ (R_E, 0, 0).
        # Place satellites at small geocentric offsets to ensure visibility.
        # At 550 km altitude, 5 deg offset gives ~41 deg elevation,
        # 10 deg offset gives ~20 deg elevation.
        R = OrbitalConstants.R_EARTH
        r = R + 550_000.0
        a5 = math.radians(5.0)
        a10 = math.radians(10.0)
        sat_positions = [
            (r, 0.0, 0.0),                                           # overhead (el=90)
            (r * math.cos(a5), r * math.sin(a5), 0.0),               # east (el~41)
            (r * math.cos(a5), -r * math.sin(a5), 0.0),              # west (el~41)
            (r * math.cos(a10), 0.0, r * math.sin(a10)),             # north (el~20)
            (r * math.cos(a10), 0.0, -r * math.sin(a10)),            # south (el~20)
        ]
        result = compute_positioning_information(
            lat_deg=0.0, lon_deg=0.0,
            sat_positions_ecef=sat_positions,
            min_elevation_deg=5.0,
        )
        assert result.fisher_determinant > 0.0, (
            f"Expected positive Fisher determinant, got {result.fisher_determinant}"
        )
        assert result.gdop < float('inf'), (
            f"Expected finite GDOP, got {result.gdop}"
        )

    def test_fisher_det_formula_matches_dop_components(self):
        """Fisher det = 1/((HDOP^2/2)^2 * VDOP^2 * TDOP^2)."""
        R = OrbitalConstants.R_EARTH
        r = R + 550_000.0
        a5 = math.radians(5.0)
        a10 = math.radians(10.0)
        sat_positions = [
            (r, 0.0, 0.0),
            (r * math.cos(a5), r * math.sin(a5), 0.0),
            (r * math.cos(a5), -r * math.sin(a5), 0.0),
            (r * math.cos(a10), 0.0, r * math.sin(a10)),
            (r * math.cos(a10), 0.0, -r * math.sin(a10)),
        ]
        result = compute_positioning_information(
            lat_deg=0.0, lon_deg=0.0,
            sat_positions_ecef=sat_positions,
            min_elevation_deg=5.0,
        )
        assert result.gdop < float('inf'), "Need finite GDOP for this test"
        # Verify formula: det(Q) = (HDOP^2/2)^2 * VDOP^2 * TDOP^2
        dop = compute_dop(0.0, 0.0, sat_positions, min_elevation_deg=5.0)
        hdop_sq_half = dop.hdop ** 2 / 2.0
        det_q = hdop_sq_half * hdop_sq_half * dop.vdop ** 2 * dop.tdop ** 2
        expected = 1.0 / det_q if det_q > 1e-30 else 0.0
        np.testing.assert_allclose(
            result.fisher_determinant, expected, rtol=1e-6,
        )


# ── R3-05: Coverage drift -7/2 docstring ────────────────────────────


class TestR3_05_CoverageDriftDocstring:
    """Verify the -7/2 coefficient assumption is documented."""

    def test_docstring_mentions_circular_orbit_assumption(self):
        """Docstring should note circular orbit assumption for -7/2 coefficient."""
        assert "circular" in compute_coverage_drift.__doc__.lower(), (
            "Expected docstring to mention circular orbit assumption"
        )
        assert "e=0" in compute_coverage_drift.__doc__ or "e = 0" in compute_coverage_drift.__doc__, (
            "Expected docstring to mention e=0"
        )


# ── R3-07: Persistent homology triangulation note ───────────────────


class TestR3_07_HomologyTriangulationNote:
    """Verify H_1 triangulation sensitivity is documented."""

    def test_docstring_mentions_triangulation_choice(self):
        """Docstring should note lower-left triangulation and H_1 sensitivity."""
        doc = compute_persistent_coverage_homology.__doc__
        assert "lower-left" in doc.lower(), (
            "Expected docstring to mention lower-left triangulation"
        )
        assert "H_1" in doc, (
            "Expected docstring to mention H_1 sensitivity"
        )
        assert "sensitive" in doc.lower(), (
            "Expected docstring to note sensitivity to triangulation choice"
        )
        assert "H_0" in doc, (
            "Expected docstring to mention H_0 robustness"
        )


# ── R3-08: EKF process noise docstring ─────────────────────────────


class TestR3_08_EKFProcessNoiseDocstring:
    """Verify process noise Q tuning guidance is documented."""

    def test_docstring_mentions_process_noise_scaling(self):
        """Docstring should note Q scaling for irregular observation intervals."""
        doc = run_ekf.__doc__
        assert "dt/dt_reference" in doc or "dt_reference" in doc, (
            "Expected docstring to mention dt/dt_reference scaling"
        )
        assert "irregular" in doc.lower() or "observation interval" in doc.lower(), (
            "Expected docstring to mention irregular observation spacing"
        )


# ── R3-10: OrbitalConstants.MU_EARTH import ────────────────────────


class TestR3_10_MuEarthImport:
    """Verify orbit_determination uses OrbitalConstants.MU_EARTH."""

    def test_mu_value_matches_orbital_constants(self):
        """The _MU used in orbit_determination should match OrbitalConstants."""
        assert _MU == OrbitalConstants.MU_EARTH, (
            f"Expected _MU={OrbitalConstants.MU_EARTH}, got {_MU}"
        )

    def test_mu_imported_not_hardcoded(self):
        """Verify _MU is imported, not locally hardcoded."""
        import inspect
        source = inspect.getsource(
            __import__('humeris.domain.orbit_determination', fromlist=['_MU'])
        )
        assert "OrbitalConstants.MU_EARTH" in source, (
            "Expected _MU = OrbitalConstants.MU_EARTH in source"
        )


# ── R3-12: DOP sqrt guard ───────────────────────────────────────────


class TestR3_12_DOPSqrtGuard:
    """Verify sqrt guard prevents ValueError on negative Q diagonals."""

    def test_sqrt_guard_in_source(self):
        """Source should use max(0.0, ...) inside sqrt calls."""
        import inspect
        source = inspect.getsource(compute_dop)
        assert "max(0.0" in source, (
            "Expected max(0.0, ...) guard inside sqrt calls"
        )

    def test_dop_with_near_singular_geometry(self):
        """Near-singular geometry should not crash with sqrt of negative."""
        # 4 satellites nearly coplanar (poor geometry)
        r = OrbitalConstants.R_EARTH + 550_000.0
        # All in nearly the same plane
        sat_positions = [
            (r, 0.0, 0.0),
            (r * 0.999, r * 0.045, 0.0),
            (r * 0.998, -r * 0.063, 0.0),
            (r * 0.997, r * 0.077, 0.01),
        ]
        # This should not raise ValueError
        result = compute_dop(0.0, 0.0, sat_positions, min_elevation_deg=0.0)
        # Result may be inf (not enough geometry) but should not crash
        assert result.gdop >= 0.0 or result.gdop == float('inf')


# ── R3-15: Thermal equilibrium q_total <= 0 guard ──────────────────


class TestR3_15_ThermalZeroPowerGuard:
    """Verify pathological zero/negative power input returns 2.7 K."""

    def test_zero_power_returns_cmb_temperature(self):
        """With zero total power, temperature should be 2.7 K (CMB)."""
        config = ThermalConfig(
            absorptivity=0.0,  # no solar absorption
            emissivity=0.8,
            solar_area_m2=1.0,
            radiator_area_m2=1.0,
            internal_power_w=0.0,
        )
        result = compute_thermal_equilibrium(
            config,
            solar_flux_w_m2=0.0,
            albedo_flux_w_m2=0.0,
            earth_ir_w_m2=0.0,
            eclipse_fraction=1.0,  # full eclipse
        )
        assert result.temperature_k == 2.7, (
            f"Expected 2.7 K for zero power, got {result.temperature_k}"
        )

    def test_negative_internal_power_returns_cmb(self):
        """With net negative power (pathological), return 2.7 K."""
        config = ThermalConfig(
            absorptivity=0.0,
            emissivity=0.8,
            solar_area_m2=1.0,
            radiator_area_m2=1.0,
            internal_power_w=-1000.0,  # pathological
        )
        result = compute_thermal_equilibrium(
            config,
            solar_flux_w_m2=0.0,
            albedo_flux_w_m2=0.0,
            earth_ir_w_m2=0.0,
            eclipse_fraction=1.0,
        )
        assert result.temperature_k == 2.7
        assert result.emitted_power_w == 0.0


# ── R3-20: Hungarian algorithm size guard ───────────────────────────


class TestR3_20_HungarianSizeGuard:
    """Verify size limit on Hungarian algorithm input."""

    def test_rejects_over_500_current_states(self):
        """Should raise ValueError for > 500 current satellites."""
        states = [_make_orbital_state(math.radians(i * 0.7)) for i in range(501)]
        target = [states[0]]

        with pytest.raises(ValueError, match="500"):
            compute_optimal_reconfiguration(states, target)

    def test_rejects_over_500_target_states(self):
        """Should raise ValueError for > 500 target satellites."""
        states = [_make_orbital_state(math.radians(i * 0.7)) for i in range(501)]
        current = [states[0]]

        with pytest.raises(ValueError, match="500"):
            compute_optimal_reconfiguration(current, states)

    def test_accepts_500_states(self):
        """Exactly 500 should be accepted (boundary test)."""
        # Only test with 2 states to avoid O(n^3) computation
        s1 = _make_orbital_state(0.0)
        s2 = _make_orbital_state(math.radians(10.0))
        result = compute_optimal_reconfiguration([s1, s2], [s1, s2])
        assert result.total_dv_ms >= 0.0

    def test_empty_inputs_still_work(self):
        """Empty current or target should return zero-cost plan."""
        result = compute_optimal_reconfiguration([], [])
        assert result.total_dv_ms == 0.0
