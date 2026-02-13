# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for orbital transfer maneuvers."""
import ast
import math

import pytest

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.maneuvers import (
    FiniteBurnConfig,
    FiniteBurnResult,
    ManeuverBurn,
    TransferPlan,
    add_propellant_estimate,
    bielliptic_transfer,
    combined_plane_and_altitude,
    compute_finite_burn,
    finite_burn_loss,
    hohmann_transfer,
    low_thrust_spiral,
    phasing_maneuver,
    plane_change_dv,
)


# ── Helpers ────────────────────────────────────────────────────────

R_EARTH = OrbitalConstants.R_EARTH
MU = OrbitalConstants.MU_EARTH
R_LEO = R_EARTH + 400_000        # 400 km LEO
R_GEO = R_EARTH + 35_786_000     # GEO


# ── Dataclass tests ───────────────────────────────────────────────

class TestManeuverBurn:

    def test_frozen(self):
        """ManeuverBurn is immutable."""
        b = ManeuverBurn(delta_v_ms=100.0, description="test")
        with pytest.raises(AttributeError):
            b.delta_v_ms = 0.0

    def test_fields(self):
        """ManeuverBurn exposes delta_v_ms and description."""
        b = ManeuverBurn(delta_v_ms=100.0, description="burn 1")
        assert b.delta_v_ms == 100.0
        assert b.description == "burn 1"


class TestTransferPlan:

    def test_frozen(self):
        """TransferPlan is immutable."""
        tp = TransferPlan(
            burns=(), total_delta_v_ms=0.0, transfer_time_s=0.0,
        )
        with pytest.raises(AttributeError):
            tp.total_delta_v_ms = 1.0

    def test_fields(self):
        """TransferPlan exposes expected fields."""
        tp = TransferPlan(
            burns=(ManeuverBurn(delta_v_ms=100.0, description="b1"),),
            total_delta_v_ms=100.0,
            transfer_time_s=3600.0,
            propellant_mass_kg=5.0,
        )
        assert len(tp.burns) == 1
        assert tp.propellant_mass_kg == 5.0


# ── Hohmann transfer ──────────────────────────────────────────────

class TestHohmannTransfer:

    def test_leo_to_geo_total_dv(self):
        """Hohmann LEO (400km) → GEO: total dV ≈ 3.86 km/s."""
        plan = hohmann_transfer(R_LEO, R_GEO)
        assert plan.total_delta_v_ms == pytest.approx(3857, rel=0.02)

    def test_same_radius_zero_dv(self):
        """Hohmann with r1 == r2 → zero dV."""
        plan = hohmann_transfer(R_LEO, R_LEO)
        assert plan.total_delta_v_ms == pytest.approx(0.0, abs=1e-10)

    def test_transfer_time_leo_geo(self):
        """Hohmann LEO → GEO transfer time ≈ 5.3 hours."""
        plan = hohmann_transfer(R_LEO, R_GEO)
        hours = plan.transfer_time_s / 3600.0
        assert hours == pytest.approx(5.3, rel=0.05)

    def test_two_burns(self):
        """Hohmann transfer produces exactly 2 burns."""
        plan = hohmann_transfer(R_LEO, R_GEO)
        assert len(plan.burns) == 2

    def test_deorbit_works(self):
        """Hohmann with r1 > r2 (deorbit) gives positive dV."""
        plan = hohmann_transfer(R_GEO, R_LEO)
        assert plan.total_delta_v_ms > 0

    def test_raises_on_non_positive_radius(self):
        """Raises ValueError for r <= 0."""
        with pytest.raises(ValueError):
            hohmann_transfer(0, R_GEO)
        with pytest.raises(ValueError):
            hohmann_transfer(R_LEO, -1)


# ── Bi-elliptic transfer ─────────────────────────────────────────

class TestBiellipticTransfer:

    def test_three_burns(self):
        """Bi-elliptic produces 3 burns."""
        r_int = R_GEO * 2
        plan = bielliptic_transfer(R_LEO, R_GEO, r_int)
        assert len(plan.burns) == 3

    def test_more_efficient_for_high_ratio(self):
        """Bi-elliptic more efficient than Hohmann for r2/r1 > 11.94."""
        r1 = R_EARTH + 200_000
        r2 = r1 * 15  # ratio > 11.94
        r_int = r2 * 2
        hohmann = hohmann_transfer(r1, r2)
        bielliptic = bielliptic_transfer(r1, r2, r_int)
        assert bielliptic.total_delta_v_ms < hohmann.total_delta_v_ms

    def test_raises_on_invalid_intermediate(self):
        """Raises ValueError for intermediate radius < max(r1, r2)."""
        with pytest.raises(ValueError):
            bielliptic_transfer(R_LEO, R_GEO, R_LEO)


# ── Plane change ──────────────────────────────────────────────────

class TestPlaneChange:

    def test_90_degree_change(self):
        """90° plane change at 7.5 km/s → dV ≈ 10.6 km/s."""
        v = 7500.0
        dv = plane_change_dv(v, math.radians(90.0))
        assert dv == pytest.approx(10607, rel=0.01)

    def test_zero_angle_zero_dv(self):
        """0° plane change → 0 dV."""
        dv = plane_change_dv(7500.0, 0.0)
        assert dv == pytest.approx(0.0, abs=1e-10)


# ── Combined plane + altitude ────────────────────────────────────

class TestCombinedTransfer:

    def test_combined_between_pure_values(self):
        """Combined dV between pure Hohmann and Hohmann + plane change sum."""
        di_rad = math.radians(28.5)  # typical GTO inclination change
        combined = combined_plane_and_altitude(R_LEO, R_GEO, di_rad)
        hohmann = hohmann_transfer(R_LEO, R_GEO)
        v_geo = math.sqrt(MU / R_GEO)
        pure_plane = plane_change_dv(v_geo, di_rad)
        # Combined should be less than doing them separately
        assert combined.total_delta_v_ms < hohmann.total_delta_v_ms + pure_plane


# ── Phasing maneuver ─────────────────────────────────────────────

class TestPhasingManeuver:

    def test_correct_transfer_time(self):
        """Phasing maneuver transfer time = n_orbits * adjusted period."""
        plan = phasing_maneuver(R_LEO, math.radians(30.0), n_orbits=1)
        # Transfer time should be close to one orbital period
        T_nominal = 2.0 * math.pi * math.sqrt(R_LEO**3 / MU)
        assert plan.transfer_time_s == pytest.approx(T_nominal, rel=0.2)

    def test_two_burns(self):
        """Phasing maneuver produces 2 burns."""
        plan = phasing_maneuver(R_LEO, math.radians(30.0))
        assert len(plan.burns) == 2


# ── Propellant estimate ──────────────────────────────────────────

class TestPropellantEstimate:

    def test_tsiolkovsky_check(self):
        """Propellant estimate matches Tsiolkovsky equation."""
        plan = hohmann_transfer(R_LEO, R_GEO)
        isp = 300.0
        dry_mass = 500.0
        plan_with_prop = add_propellant_estimate(plan, isp, dry_mass)
        assert plan_with_prop.propellant_mass_kg is not None
        # Check: m_prop = m_dry * (exp(dV / (Isp * g0)) - 1)
        g0 = 9.80665
        expected = dry_mass * (math.exp(plan.total_delta_v_ms / (isp * g0)) - 1.0)
        assert plan_with_prop.propellant_mass_kg == pytest.approx(expected, rel=1e-10)

    def test_burns_preserved(self):
        """add_propellant_estimate preserves original burns and delta-V."""
        plan = hohmann_transfer(R_LEO, R_GEO)
        plan_with_prop = add_propellant_estimate(plan, 300.0, 500.0)
        assert plan_with_prop.total_delta_v_ms == plan.total_delta_v_ms
        assert plan_with_prop.burns == plan.burns


# ── Finite burn ─────────────────────────────────────────────────

class TestFiniteBurnDataclasses:

    def test_finite_burn_config_frozen(self):
        cfg = FiniteBurnConfig(thrust_n=1000.0, isp_s=300.0, initial_mass_kg=500.0)
        with pytest.raises(AttributeError):
            cfg.thrust_n = 0.0

    def test_finite_burn_result_frozen(self):
        r = FiniteBurnResult(delta_v_ms=100.0, burn_duration_s=10.0,
                             propellant_mass_kg=5.0, final_mass_kg=495.0,
                             thrust_arc_deg=1.0)
        with pytest.raises(AttributeError):
            r.delta_v_ms = 0.0


class TestFiniteBurn:

    def test_impulsive_limit(self):
        """Very high thrust → burn_duration ≈ 0."""
        cfg = FiniteBurnConfig(thrust_n=1e8, isp_s=300.0, initial_mass_kg=500.0)
        result = compute_finite_burn(100.0, cfg)
        assert result.burn_duration_s < 0.1

    def test_mass_conservation(self):
        """initial = final + propellant."""
        cfg = FiniteBurnConfig(thrust_n=5000.0, isp_s=300.0, initial_mass_kg=500.0)
        result = compute_finite_burn(200.0, cfg)
        assert abs(cfg.initial_mass_kg - result.final_mass_kg - result.propellant_mass_kg) < 1e-10

    def test_tsiolkovsky_consistency(self):
        """dv = Isp * g0 * ln(m0/mf)."""
        cfg = FiniteBurnConfig(thrust_n=5000.0, isp_s=300.0, initial_mass_kg=500.0)
        dv_target = 200.0
        result = compute_finite_burn(dv_target, cfg)
        g0 = 9.80665
        dv_check = cfg.isp_s * g0 * math.log(cfg.initial_mass_kg / result.final_mass_kg)
        assert abs(dv_check - dv_target) < 1e-8

    def test_burn_arc_bounded(self):
        """Burn arc < 360° for reasonable thrust levels."""
        cfg = FiniteBurnConfig(thrust_n=5000.0, isp_s=300.0, initial_mass_kg=500.0)
        result = compute_finite_burn(200.0, cfg)
        assert 0 < result.thrust_arc_deg < 360

    def test_zero_thrust_raises(self):
        with pytest.raises(ValueError, match="thrust_n"):
            compute_finite_burn(100.0, FiniteBurnConfig(
                thrust_n=0.0, isp_s=300.0, initial_mass_kg=500.0))

    def test_negative_isp_raises(self):
        with pytest.raises(ValueError, match="isp_s"):
            compute_finite_burn(100.0, FiniteBurnConfig(
                thrust_n=1000.0, isp_s=-1.0, initial_mass_kg=500.0))


class TestFiniteBurnLoss:

    def test_loss_positive(self):
        """Finite burn effective dv < impulsive dv."""
        effective = finite_burn_loss(100.0, 60.0, 7500.0)
        assert effective < 100.0
        assert effective > 0

    def test_zero_duration_no_loss(self):
        """Zero burn duration → no loss (impulsive limit)."""
        effective = finite_burn_loss(100.0, 0.0, 7500.0)
        assert abs(effective - 100.0) < 1e-10


class TestLowThrustSpiral:

    def test_spiral_dv_exceeds_hohmann(self):
        """Low-thrust spiral dv > Hohmann dv (less efficient)."""
        cfg = FiniteBurnConfig(thrust_n=0.5, isp_s=3000.0, initial_mass_kg=500.0)
        spiral = low_thrust_spiral(R_LEO, R_GEO, cfg)
        hohmann = hohmann_transfer(R_LEO, R_GEO)
        assert spiral.delta_v_ms > hohmann.total_delta_v_ms

    def test_spiral_mass_conservation(self):
        cfg = FiniteBurnConfig(thrust_n=0.5, isp_s=3000.0, initial_mass_kg=500.0)
        result = low_thrust_spiral(R_LEO, R_GEO, cfg)
        assert abs(cfg.initial_mass_kg - result.final_mass_kg - result.propellant_mass_kg) < 1e-10

    def test_invalid_radius_raises(self):
        cfg = FiniteBurnConfig(thrust_n=0.5, isp_s=3000.0, initial_mass_kg=500.0)
        with pytest.raises(ValueError):
            low_thrust_spiral(0.0, R_GEO, cfg)


# ── Domain purity ─────────────────────────────────────────────────

class TestManeuversPurity:

    def test_maneuvers_module_pure(self):
        """maneuvers.py must only import stdlib modules."""
        import humeris.domain.maneuvers as mod

        allowed = {'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
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
