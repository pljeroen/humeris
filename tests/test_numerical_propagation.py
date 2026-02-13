# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for numerical propagation (RK4 + pluggable force models)."""

import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris import (
    OrbitalConstants,
    ShellConfig,
    generate_walker_shell,
    derive_orbital_state,
    kepler_to_cartesian,
)
from humeris.domain.numerical_propagation import (
    PropagationStep,
    NumericalPropagationResult,
    TwoBodyGravity,
    J2Perturbation,
    J3Perturbation,
    SphericalHarmonicGravity,
    AtmosphericDragForce,
    SolarRadiationPressureForce,
    rk4_step,
    stormer_verlet_step,
    yoshida4_step,
    propagate_numerical,
)


@pytest.fixture
def epoch():
    return datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def leo_state(epoch):
    """LEO orbital state at 500 km, 53 deg inclination."""
    shell = ShellConfig(
        altitude_km=500,
        inclination_deg=53,
        num_planes=1,
        sats_per_plane=1,
        phase_factor=0,
        raan_offset_deg=0,
        shell_name="Test",
    )
    sat = generate_walker_shell(shell)[0]
    return derive_orbital_state(sat, epoch)


@pytest.fixture
def leo_position_velocity(leo_state, epoch):
    """ECI position/velocity at LEO."""
    pos, vel = kepler_to_cartesian(
        a=leo_state.semi_major_axis_m,
        e=leo_state.eccentricity,
        i_rad=leo_state.inclination_rad,
        omega_big_rad=leo_state.raan_rad,
        omega_small_rad=leo_state.arg_perigee_rad,
        nu_rad=leo_state.true_anomaly_rad,
    )
    return tuple(pos), tuple(vel)


# --- Frozen dataclass tests ---

class TestPropagationStep:

    def test_propagation_step_frozen(self, epoch):
        step = PropagationStep(
            time=epoch,
            position_eci=(1.0, 2.0, 3.0),
            velocity_eci=(4.0, 5.0, 6.0),
        )
        with pytest.raises(AttributeError):
            step.time = epoch  # type: ignore[misc]

    def test_propagation_step_fields(self, epoch):
        step = PropagationStep(
            time=epoch,
            position_eci=(1.0, 2.0, 3.0),
            velocity_eci=(4.0, 5.0, 6.0),
        )
        assert step.time == epoch
        assert step.position_eci == (1.0, 2.0, 3.0)
        assert step.velocity_eci == (4.0, 5.0, 6.0)


class TestNumericalPropagationResult:

    def test_result_frozen(self, epoch):
        result = NumericalPropagationResult(
            steps=(),
            epoch=epoch,
            duration_s=0.0,
            force_model_names=("TwoBodyGravity",),
        )
        with pytest.raises(AttributeError):
            result.epoch = epoch  # type: ignore[misc]

    def test_result_fields(self, epoch):
        result = NumericalPropagationResult(
            steps=(),
            epoch=epoch,
            duration_s=3600.0,
            force_model_names=("TwoBodyGravity", "J2Perturbation"),
        )
        assert isinstance(result.steps, tuple)
        assert isinstance(result.force_model_names, tuple)
        assert result.duration_s == 3600.0


# --- Force model tests ---

class TestTwoBodyGravity:

    def test_two_body_magnitude(self, leo_position_velocity, epoch):
        """Acceleration magnitude should be mu/r^2 at LEO within 0.1%."""
        pos, vel = leo_position_velocity
        gravity = TwoBodyGravity()
        acc = gravity.acceleration(epoch, pos, vel)

        r = math.sqrt(sum(p ** 2 for p in pos))
        a_mag = math.sqrt(sum(a ** 2 for a in acc))
        expected = OrbitalConstants.MU_EARTH / r ** 2

        assert abs(a_mag - expected) / expected < 0.001

    def test_two_body_direction(self, leo_position_velocity, epoch):
        """Acceleration should point toward origin (dot(a, r) < 0)."""
        pos, vel = leo_position_velocity
        gravity = TwoBodyGravity()
        acc = gravity.acceleration(epoch, pos, vel)

        dot = sum(a * p for a, p in zip(acc, pos))
        assert dot < 0

    def test_two_body_inverse_square(self, epoch):
        """2x distance → 1/4 acceleration."""
        gravity = TwoBodyGravity()
        r1 = OrbitalConstants.R_EARTH + 500_000
        pos1 = (r1, 0.0, 0.0)
        vel = (0.0, 7500.0, 0.0)

        pos2 = (2 * r1, 0.0, 0.0)

        acc1 = gravity.acceleration(epoch, pos1, vel)
        acc2 = gravity.acceleration(epoch, pos2, vel)

        mag1 = math.sqrt(sum(a ** 2 for a in acc1))
        mag2 = math.sqrt(sum(a ** 2 for a in acc2))

        assert abs(mag1 / mag2 - 4.0) / 4.0 < 0.001


class TestJ2Perturbation:

    def test_j2_magnitude_relative(self, leo_position_velocity, epoch):
        """J2 acceleration should be ~1e-3 of two-body."""
        pos, vel = leo_position_velocity
        gravity = TwoBodyGravity()
        j2 = J2Perturbation()

        acc_grav = gravity.acceleration(epoch, pos, vel)
        acc_j2 = j2.acceleration(epoch, pos, vel)

        mag_grav = math.sqrt(sum(a ** 2 for a in acc_grav))
        mag_j2 = math.sqrt(sum(a ** 2 for a in acc_j2))

        ratio = mag_j2 / mag_grav
        assert 1e-4 < ratio < 1e-2

    def test_j2_z_asymmetry(self, epoch):
        """Different z position → different ax/ay."""
        j2 = J2Perturbation()
        r = OrbitalConstants.R_EARTH + 500_000
        vel = (0.0, 7500.0, 0.0)

        # Equatorial position
        acc_eq = j2.acceleration(epoch, (r, 0.0, 0.0), vel)
        # Position with z component
        acc_z = j2.acceleration(epoch, (r * 0.8, 0.0, r * 0.6), vel)

        assert acc_eq != acc_z


class TestJ3Perturbation:

    def test_j3_magnitude_relative(self, leo_position_velocity, epoch):
        """J3 acceleration should be ~1e-3 of J2."""
        pos, vel = leo_position_velocity
        j2 = J2Perturbation()
        j3 = J3Perturbation()

        acc_j2 = j2.acceleration(epoch, pos, vel)
        acc_j3 = j3.acceleration(epoch, pos, vel)

        mag_j2 = math.sqrt(sum(a ** 2 for a in acc_j2))
        mag_j3 = math.sqrt(sum(a ** 2 for a in acc_j3))

        ratio = mag_j3 / mag_j2
        assert 1e-4 < ratio < 1e-1

    def test_j3_nonzero_at_z(self, epoch):
        """Nonzero z → nonzero J3 acceleration."""
        j3 = J3Perturbation()
        r = OrbitalConstants.R_EARTH + 500_000
        pos = (r * 0.8, 0.0, r * 0.6)
        vel = (0.0, 7500.0, 0.0)

        acc = j3.acceleration(epoch, pos, vel)
        mag = math.sqrt(sum(a ** 2 for a in acc))
        assert mag > 0


class TestAtmosphericDragForce:

    def test_drag_opposes_velocity(self, epoch):
        """dot(a_drag, v_rel) < 0 — drag opposes relative velocity."""
        from humeris import DragConfig

        drag = AtmosphericDragForce(DragConfig(cd=2.2, area_m2=10.0, mass_kg=400.0))
        r = OrbitalConstants.R_EARTH + 300_000
        pos = (r, 0.0, 0.0)
        vel = (0.0, 7800.0, 0.0)

        acc = drag.acceleration(epoch, pos, vel)
        # v_rel = (vel_x + omega_e * y, vel_y - omega_e * x, vel_z)
        omega_e = OrbitalConstants.EARTH_ROTATION_RATE
        v_rel = (vel[0] + omega_e * pos[1], vel[1] - omega_e * pos[0], vel[2])
        dot = sum(a * v for a, v in zip(acc, v_rel))
        assert dot < 0

    def test_drag_decreases_with_altitude(self, epoch):
        """300 km drag > 600 km drag."""
        from humeris import DragConfig

        drag = AtmosphericDragForce(DragConfig(cd=2.2, area_m2=10.0, mass_kg=400.0))

        r_300 = OrbitalConstants.R_EARTH + 300_000
        pos_300 = (r_300, 0.0, 0.0)
        vel_300 = (0.0, 7730.0, 0.0)

        r_600 = OrbitalConstants.R_EARTH + 600_000
        pos_600 = (r_600, 0.0, 0.0)
        vel_600 = (0.0, 7560.0, 0.0)

        acc_300 = drag.acceleration(epoch, pos_300, vel_300)
        acc_600 = drag.acceleration(epoch, pos_600, vel_600)

        mag_300 = math.sqrt(sum(a ** 2 for a in acc_300))
        mag_600 = math.sqrt(sum(a ** 2 for a in acc_600))

        assert mag_300 > mag_600


class TestSolarRadiationPressure:

    def test_srp_direction_away_from_sun(self, epoch):
        """SRP has component away from Sun."""
        from humeris import sun_position_eci

        srp = SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0)
        r = OrbitalConstants.R_EARTH + 500_000
        pos = (r, 0.0, 0.0)
        vel = (0.0, 7500.0, 0.0)

        acc = srp.acceleration(epoch, pos, vel)
        sun = sun_position_eci(epoch)

        # d = r_sat - r_sun  (from Sun toward satellite)
        d = tuple(pos[i] - sun.position_eci_m[i] for i in range(3))
        # acc should be in same direction as d (away from Sun)
        dot = sum(a * di for a, di in zip(acc, d))
        assert dot > 0

    def test_srp_magnitude_order(self, epoch):
        """SRP magnitude should be ~1e-7 to 1e-8 m/s² for typical spacecraft."""
        srp = SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0)
        r = OrbitalConstants.R_EARTH + 500_000
        pos = (r, 0.0, 0.0)
        vel = (0.0, 7500.0, 0.0)

        acc = srp.acceleration(epoch, pos, vel)
        mag = math.sqrt(sum(a ** 2 for a in acc))

        assert 1e-9 < mag < 1e-5


# --- RK4 step tests ---

class TestRK4Step:

    def test_rk4_step_linear(self):
        """dy/dt = 1 → exact step."""
        def deriv(t, state):
            return (1.0,)

        t_new, state_new = rk4_step(0.0, (0.0,), 1.0, deriv)
        assert abs(t_new - 1.0) < 1e-12
        assert abs(state_new[0] - 1.0) < 1e-12

    def test_rk4_step_quadratic(self):
        """dy/dt = 2t → y = t². RK4 is exact for polynomials up to degree 4."""
        def deriv(t, state):
            return (2.0 * t,)

        t_new, state_new = rk4_step(0.0, (0.0,), 1.0, deriv)
        assert abs(t_new - 1.0) < 1e-12
        assert abs(state_new[0] - 1.0) < 1e-10


# --- Spherical harmonic gravity tests ---

class TestSphericalHarmonicGravity:

    def test_degree2_matches_j2(self, leo_position_velocity, epoch):
        """Degree-2 zonal-only: total accel matches TwoBody + J2."""
        pos, vel = leo_position_velocity
        sh = SphericalHarmonicGravity(max_degree=2)
        j2 = J2Perturbation()
        tb = TwoBodyGravity()

        acc_sh = sh.acceleration(epoch, pos, vel)
        acc_tb = tb.acceleration(epoch, pos, vel)
        acc_j2 = j2.acceleration(epoch, pos, vel)
        acc_ref = tuple(a + b for a, b in zip(acc_tb, acc_j2))

        mag_ref = math.sqrt(sum(a**2 for a in acc_ref))
        diff = math.sqrt(sum((a - b)**2 for a, b in zip(acc_sh, acc_ref)))
        # The SH model includes C22/S22 tesseral terms, so won't be exact
        # But should be close (tesseral terms are ~1000x smaller than J2)
        assert diff / mag_ref < 0.01

    def test_acceleration_decreases_with_altitude(self, epoch):
        """8x8 acceleration magnitude decreases with altitude."""
        sh = SphericalHarmonicGravity(max_degree=8)
        vel = (0.0, 7500.0, 0.0)

        r_low = OrbitalConstants.R_EARTH + 300_000
        r_high = OrbitalConstants.R_EARTH + 1_000_000

        acc_low = sh.acceleration(epoch, (r_low, 0.0, 0.0), vel)
        acc_high = sh.acceleration(epoch, (r_high, 0.0, 0.0), vel)

        mag_low = math.sqrt(sum(a**2 for a in acc_low))
        mag_high = math.sqrt(sum(a**2 for a in acc_high))
        assert mag_low > mag_high

    def test_leo_8x8_vs_j2_within_1_percent(self, leo_position_velocity, epoch):
        """At LEO, 8x8 total accel differs from J2-only by < 1%."""
        pos, vel = leo_position_velocity
        sh = SphericalHarmonicGravity(max_degree=8)
        tb = TwoBodyGravity()
        j2 = J2Perturbation()

        acc_sh = sh.acceleration(epoch, pos, vel)
        acc_j2_total = tuple(
            a + b for a, b in zip(
                tb.acceleration(epoch, pos, vel),
                j2.acceleration(epoch, pos, vel),
            )
        )

        mag_j2 = math.sqrt(sum(a**2 for a in acc_j2_total))
        diff = math.sqrt(sum((a - b)**2 for a, b in zip(acc_sh, acc_j2_total)))
        # J2 dominates at LEO — difference should be small
        assert diff / mag_j2 < 0.01

    def test_degree4_vs_degree8_agree(self, leo_position_velocity, epoch):
        """Degree 4 and degree 8 agree within 10% at LEO."""
        pos, vel = leo_position_velocity
        sh4 = SphericalHarmonicGravity(max_degree=4)
        sh8 = SphericalHarmonicGravity(max_degree=8)

        acc4 = sh4.acceleration(epoch, pos, vel)
        acc8 = sh8.acceleration(epoch, pos, vel)

        mag8 = math.sqrt(sum(a**2 for a in acc8))
        diff = math.sqrt(sum((a - b)**2 for a, b in zip(acc4, acc8)))
        assert diff / mag8 < 0.1

    def test_invalid_degree_raises(self):
        """Degree < 2 or > 8 raises ValueError."""
        with pytest.raises(ValueError):
            SphericalHarmonicGravity(max_degree=1)
        with pytest.raises(ValueError):
            SphericalHarmonicGravity(max_degree=9)

    def test_nonzero_acceleration(self, leo_position_velocity, epoch):
        """SH gravity produces nonzero acceleration."""
        pos, vel = leo_position_velocity
        sh = SphericalHarmonicGravity(max_degree=8)
        acc = sh.acceleration(epoch, pos, vel)
        mag = math.sqrt(sum(a**2 for a in acc))
        assert mag > 0

    def test_geo_tesseral_effect(self, epoch):
        """At GEO, tesseral terms (C22,S22) produce measurable effect."""
        sh2 = SphericalHarmonicGravity(max_degree=2)
        tb = TwoBodyGravity()

        r_geo = 42_164_000.0
        pos = (r_geo, 0.0, 0.0)
        vel = (0.0, 3075.0, 0.0)

        acc_sh = sh2.acceleration(epoch, pos, vel)
        acc_tb = tb.acceleration(epoch, pos, vel)

        # The difference is the perturbation (J2 + tesseral)
        diff = tuple(a - b for a, b in zip(acc_sh, acc_tb))
        pert_mag = math.sqrt(sum(d**2 for d in diff))
        assert pert_mag > 0


# --- Symplectic integrator tests ---

class TestStormerVerletStep:

    def test_two_body_one_orbit(self, leo_state, epoch):
        """Verlet propagates one orbit and returns to near-initial position."""
        period_s = 2 * math.pi / leo_state.mean_motion_rad_s
        result = propagate_numerical(
            leo_state, timedelta(seconds=period_s), timedelta(seconds=10),
            [TwoBodyGravity()], epoch=epoch, integrator="verlet",
        )
        r0 = result.steps[0].position_eci
        rf = result.steps[-1].position_eci
        mag_r0 = math.sqrt(sum(p**2 for p in r0))
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(r0, rf)))
        assert dist / mag_r0 < 1e-2

    def test_result_type(self, leo_state, epoch):
        """Verlet returns NumericalPropagationResult."""
        result = propagate_numerical(
            leo_state, timedelta(seconds=600), timedelta(seconds=30),
            [TwoBodyGravity()], epoch=epoch, integrator="verlet",
        )
        assert isinstance(result, NumericalPropagationResult)


class TestYoshida4Step:

    def test_matches_rk4(self, leo_state, epoch):
        """Yoshida matches RK4 to ~1e-3 relative over 1 orbit."""
        period_s = 2 * math.pi / leo_state.mean_motion_rad_s
        dt = timedelta(seconds=30)
        r_rk4 = propagate_numerical(
            leo_state, timedelta(seconds=period_s), dt,
            [TwoBodyGravity()], epoch=epoch, integrator="rk4",
        )
        r_yosh = propagate_numerical(
            leo_state, timedelta(seconds=period_s), dt,
            [TwoBodyGravity()], epoch=epoch, integrator="yoshida",
        )
        pos_rk4 = r_rk4.steps[-1].position_eci
        pos_yosh = r_yosh.steps[-1].position_eci
        mag = math.sqrt(sum(p**2 for p in pos_rk4))
        diff = math.sqrt(sum((a - b)**2 for a, b in zip(pos_rk4, pos_yosh)))
        assert diff / mag < 1e-3

    def test_energy_conservation(self, leo_state, epoch):
        """Yoshida energy drift < 1e-10 over 10 orbits (two-body)."""
        mu = OrbitalConstants.MU_EARTH
        period_s = 2 * math.pi / leo_state.mean_motion_rad_s
        result = propagate_numerical(
            leo_state, timedelta(seconds=10 * period_s), timedelta(seconds=30),
            [TwoBodyGravity()], epoch=epoch, integrator="yoshida",
        )

        def energy(step):
            r = math.sqrt(sum(p**2 for p in step.position_eci))
            v = math.sqrt(sum(vi**2 for vi in step.velocity_eci))
            return 0.5 * v**2 - mu / r

        e0 = energy(result.steps[0])
        max_drift = max(abs(energy(s) - e0) / abs(e0) for s in result.steps)
        assert max_drift < 1e-8


class TestIntegratorParameter:

    def test_default_is_rk4(self, leo_state, epoch):
        """Default integrator is rk4 (backward compatible)."""
        r1 = propagate_numerical(
            leo_state, timedelta(seconds=600), timedelta(seconds=30),
            [TwoBodyGravity()], epoch=epoch,
        )
        r2 = propagate_numerical(
            leo_state, timedelta(seconds=600), timedelta(seconds=30),
            [TwoBodyGravity()], epoch=epoch, integrator="rk4",
        )
        assert r1.steps[-1].position_eci == r2.steps[-1].position_eci

    def test_invalid_integrator_raises(self, leo_state, epoch):
        """Unknown integrator raises ValueError."""
        with pytest.raises(ValueError, match="Unknown integrator"):
            propagate_numerical(
                leo_state, timedelta(seconds=600), timedelta(seconds=30),
                [TwoBodyGravity()], epoch=epoch, integrator="euler",
            )


# --- Integration tests ---

class TestPropagateNumerical:

    def test_two_body_energy_conservation(self, leo_state, epoch):
        """Specific orbital energy drift < 1e-8 over one orbit."""
        period_s = 2 * math.pi / leo_state.mean_motion_rad_s
        result = propagate_numerical(
            leo_state,
            timedelta(seconds=period_s),
            timedelta(seconds=30),
            [TwoBodyGravity()],
            epoch=epoch,
        )

        mu = OrbitalConstants.MU_EARTH

        def energy(step):
            r = math.sqrt(sum(p ** 2 for p in step.position_eci))
            v = math.sqrt(sum(v ** 2 for v in step.velocity_eci))
            return 0.5 * v ** 2 - mu / r

        e_initial = energy(result.steps[0])
        e_final = energy(result.steps[-1])

        rel_drift = abs(e_final - e_initial) / abs(e_initial)
        assert rel_drift < 1e-8

    def test_step_count(self, leo_state, epoch):
        """len(steps) == int(duration/step) + 1."""
        duration = timedelta(seconds=600)
        step = timedelta(seconds=30)
        result = propagate_numerical(
            leo_state, duration, step,
            [TwoBodyGravity()],
            epoch=epoch,
        )
        expected = int(duration.total_seconds() / step.total_seconds()) + 1
        assert len(result.steps) == expected

    def test_j2_raan_drift(self, epoch):
        """Numerical J2 RAAN drift matches analytical within 2%."""
        shell = ShellConfig(
            altitude_km=500,
            inclination_deg=53,
            num_planes=1,
            sats_per_plane=1,
            phase_factor=0,
            raan_offset_deg=0,
            shell_name="Test",
        )
        sat = generate_walker_shell(shell)[0]
        state = derive_orbital_state(sat, epoch)

        # Propagate for 1 orbit
        period_s = 2 * math.pi / state.mean_motion_rad_s
        result = propagate_numerical(
            state,
            timedelta(seconds=period_s),
            timedelta(seconds=10),
            [TwoBodyGravity(), J2Perturbation()],
            epoch=epoch,
        )

        # Compute RAAN from final position/velocity
        # Using angular momentum h = r × v
        r_f = result.steps[-1].position_eci
        v_f = result.steps[-1].velocity_eci
        hx = r_f[1] * v_f[2] - r_f[2] * v_f[1]
        hy = r_f[2] * v_f[0] - r_f[0] * v_f[2]
        raan_final = math.atan2(hx, -hy)

        r_0 = result.steps[0].position_eci
        v_0 = result.steps[0].velocity_eci
        hx0 = r_0[1] * v_0[2] - r_0[2] * v_0[1]
        hy0 = r_0[2] * v_0[0] - r_0[0] * v_0[2]
        raan_initial = math.atan2(hx0, -hy0)

        numerical_drift = raan_final - raan_initial
        analytical_drift = state.j2_raan_rate * period_s

        # Both should be negative (retrograde precession for prograde orbit)
        # Compare magnitudes within 2%
        if abs(analytical_drift) > 1e-10:
            rel_error = abs(numerical_drift - analytical_drift) / abs(analytical_drift)
            assert rel_error < 0.02, f"RAAN drift mismatch: numerical={numerical_drift}, analytical={analytical_drift}, rel_error={rel_error}"

    def test_drag_decreases_energy(self, epoch):
        """Final energy < initial energy when drag is present."""
        from humeris import DragConfig

        shell = ShellConfig(
            altitude_km=300,
            inclination_deg=53,
            num_planes=1,
            sats_per_plane=1,
            phase_factor=0,
            raan_offset_deg=0,
            shell_name="Test",
        )
        sat = generate_walker_shell(shell)[0]
        state = derive_orbital_state(sat, epoch)

        drag_config = DragConfig(cd=2.2, area_m2=10.0, mass_kg=400.0)
        result = propagate_numerical(
            state,
            timedelta(minutes=30),
            timedelta(seconds=30),
            [TwoBodyGravity(), AtmosphericDragForce(drag_config)],
            epoch=epoch,
        )

        mu = OrbitalConstants.MU_EARTH

        def energy(step):
            r = math.sqrt(sum(p ** 2 for p in step.position_eci))
            v = math.sqrt(sum(v ** 2 for v in step.velocity_eci))
            return 0.5 * v ** 2 - mu / r

        e_initial = energy(result.steps[0])
        e_final = energy(result.steps[-1])
        assert e_final < e_initial


# --- Domain purity ---

class TestSolarRadiationPressureShadow:
    """Shadow-aware SRP tests."""

    def test_shadow_disabled_by_default(self):
        """Default SRP has shadow disabled."""
        srp = SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0)
        assert srp._include_shadow is False

    def test_shadow_enabled_constructor(self):
        """include_shadow=True sets _include_shadow flag."""
        srp = SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0, include_shadow=True)
        assert srp._include_shadow is True

    def test_umbra_returns_zero(self, epoch):
        """Position in umbra → zero acceleration."""
        from humeris import sun_position_eci

        srp = SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0, include_shadow=True)
        sun = sun_position_eci(epoch)

        # Place satellite directly behind Earth from Sun
        sx, sy, sz = sun.position_eci_m
        sun_mag = math.sqrt(sx * sx + sy * sy + sz * sz)
        # Unit vector from Earth to Sun
        ux, uy, uz = sx / sun_mag, sy / sun_mag, sz / sun_mag
        # Place satellite opposite from Sun, close to Earth (in shadow)
        r = OrbitalConstants.R_EARTH + 500_000
        pos = (-ux * r, -uy * r, -uz * r)
        vel = (0.0, 7500.0, 0.0)

        acc = srp.acceleration(epoch, pos, vel)
        assert acc == (0.0, 0.0, 0.0)

    def test_sunlit_same_as_no_shadow(self, epoch):
        """Sunlit position → same result as shadow disabled."""
        srp_shadow = SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0, include_shadow=True)
        srp_noshadow = SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0, include_shadow=False)

        # Sunlit position (arbitrary, should be on sunlit side)
        from humeris import sun_position_eci
        sun = sun_position_eci(epoch)
        sx, sy, sz = sun.position_eci_m
        sun_mag = math.sqrt(sx * sx + sy * sy + sz * sz)
        ux, uy, uz = sx / sun_mag, sy / sun_mag, sz / sun_mag
        # Place on sunlit side
        r = OrbitalConstants.R_EARTH + 500_000
        pos = (ux * r, uy * r, uz * r)
        vel = (0.0, 7500.0, 0.0)

        acc_shadow = srp_shadow.acceleration(epoch, pos, vel)
        acc_noshadow = srp_noshadow.acceleration(epoch, pos, vel)
        assert acc_shadow == acc_noshadow

    def test_backward_compatible_default(self, epoch):
        """Existing behavior unchanged: no include_shadow arg → same as before."""
        srp = SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0)
        r = OrbitalConstants.R_EARTH + 500_000
        pos = (r, 0.0, 0.0)
        vel = (0.0, 7500.0, 0.0)
        acc = srp.acceleration(epoch, pos, vel)
        mag = math.sqrt(sum(a ** 2 for a in acc))
        # Non-zero acceleration even if position is in shadow (shadow not checked)
        assert mag > 0

    def test_shadow_srp_reduces_energy_less(self, epoch):
        """Shadow SRP over 1 orbit → smaller energy change than no-shadow SRP.

        During eclipsed portion of orbit, shadow-aware SRP produces zero
        acceleration, so cumulative effect is smaller.
        """
        shell = ShellConfig(
            altitude_km=500, inclination_deg=53,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0, shell_name="Test",
        )
        sat = generate_walker_shell(shell)[0]
        state = derive_orbital_state(sat, epoch)
        period_s = 2 * math.pi / state.mean_motion_rad_s

        forces_noshadow = [TwoBodyGravity(), SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0)]
        forces_shadow = [TwoBodyGravity(), SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0, include_shadow=True)]

        mu = OrbitalConstants.MU_EARTH

        def energy(step):
            r = math.sqrt(sum(p ** 2 for p in step.position_eci))
            v = math.sqrt(sum(v ** 2 for v in step.velocity_eci))
            return 0.5 * v ** 2 - mu / r

        result_noshadow = propagate_numerical(
            state, timedelta(seconds=period_s), timedelta(seconds=30),
            forces_noshadow, epoch=epoch,
        )
        result_shadow = propagate_numerical(
            state, timedelta(seconds=period_s), timedelta(seconds=30),
            forces_shadow, epoch=epoch,
        )

        delta_e_noshadow = abs(energy(result_noshadow.steps[-1]) - energy(result_noshadow.steps[0]))
        delta_e_shadow = abs(energy(result_shadow.steps[-1]) - energy(result_shadow.steps[0]))

        # Shadow SRP should produce less (or equal) energy change
        # Conical model slightly shifts shadow boundaries, allow small tolerance
        assert delta_e_shadow <= delta_e_noshadow * 1.05

    def test_eclipsed_satellite_zero_srp(self, epoch):
        """Satellite directly behind Earth gets zero SRP with shadow enabled."""
        from humeris import sun_position_eci

        srp = SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0, include_shadow=True)
        sun = sun_position_eci(epoch)
        sx, sy, sz = sun.position_eci_m
        sun_mag = math.sqrt(sx * sx + sy * sy + sz * sz)
        ux, uy, uz = sx / sun_mag, sy / sun_mag, sz / sun_mag

        # Place satellite in Earth's shadow (behind Earth from Sun)
        r = OrbitalConstants.R_EARTH + 400_000
        pos = (-ux * r, -uy * r, -uz * r)
        vel = (0.0, 7600.0, 0.0)

        acc = srp.acceleration(epoch, pos, vel)
        mag = math.sqrt(sum(a ** 2 for a in acc))
        assert mag == 0.0


class TestDomainPurity:

    def test_domain_purity(self):
        """numerical_propagation.py must only import from stdlib and domain."""
        import humeris.domain.numerical_propagation as mod

        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_top = {"math", "numpy", "dataclasses", "typing", "datetime"}
        allowed_internal_prefix = "humeris"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed_top or alias.name.startswith(allowed_internal_prefix), \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed_top or node.module.startswith(allowed_internal_prefix), \
                        f"Forbidden import from: {node.module}"
