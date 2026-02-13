# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for Cunningham spherical harmonic gravity (gravity_field.py)."""

import ast
import math
import os
from datetime import datetime, timedelta, timezone

import pytest

from humeris import OrbitalConstants, kepler_to_cartesian
from humeris.domain.gravity_field import (
    GravityFieldModel,
    CunninghamGravity,
    load_gravity_field,
)
from humeris.domain.numerical_propagation import (
    ForceModel,
    TwoBodyGravity,
    SphericalHarmonicGravity,
    propagate_numerical,
)


@pytest.fixture
def epoch():
    return datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def leo_position():
    """LEO position at 500 km altitude along x-axis with inclined velocity."""
    r = OrbitalConstants.R_EARTH_EQUATORIAL + 500_000.0
    pos = (r, 0.0, 0.0)
    vel = (0.0, 7500.0, 0.0)
    return pos, vel


@pytest.fixture
def model_70():
    return load_gravity_field(max_degree=70)


@pytest.fixture
def model_8():
    return load_gravity_field(max_degree=8)


# ── GravityFieldModel tests ─────────────────────────────────────────


class TestGravityFieldModel:

    def test_load_default_coefficient_count(self, model_70):
        """Default load produces 2553 coefficient pairs (degree 2-70)."""
        # Triangular count: sum_{n=2}^{70} (n+1) = 2553
        expected = sum(n + 1 for n in range(2, 71))
        assert expected == 2553
        # c_bar/s_bar are extended to degree N+1 for V/W recursion:
        # size = (max_degree+2)*(max_degree+3)//2 = 72*73//2 = 2628
        assert len(model_70.c_bar) == 72 * 73 // 2
        assert len(model_70.s_bar) == 72 * 73 // 2

    def test_truncation_degree_20(self):
        """max_degree=20 loads a subset."""
        model = load_gravity_field(max_degree=20)
        assert model.max_degree == 20
        # Extended to degree 21 for V/W recursion: (22*23)//2 = 253
        assert len(model.c_bar) == 22 * 23 // 2

    def test_c20_matches_reference(self, model_70):
        """C̄(2,0) matches the well-known EGM96 value."""
        # Triangular index for (2,0): 2*(2+1)//2 + 0 = 3
        idx = 2 * 3 // 2  # = 3
        c20 = model_70.c_bar[idx]
        assert abs(c20 - (-0.484165371736e-3)) < 1e-15

    def test_model_is_frozen(self, model_70):
        """GravityFieldModel is immutable."""
        with pytest.raises(AttributeError):
            model_70.max_degree = 10  # type: ignore[misc]

    def test_invalid_degree_raises(self):
        """max_degree < 2 or exceeding data file raises ValueError."""
        with pytest.raises(ValueError):
            load_gravity_field(max_degree=1)
        with pytest.raises(ValueError, match="exceeds data file"):
            load_gravity_field(max_degree=71)

    def test_egm96_reference_constants(self, model_70):
        """Model uses EGM96 reference constants, not WGS84."""
        assert model_70.gm == 3.986004415e14
        assert model_70.radius == 6378136.3
        # These differ from OrbitalConstants
        assert model_70.gm != OrbitalConstants.MU_EARTH
        assert model_70.radius != OrbitalConstants.R_EARTH_EQUATORIAL


# ── CunninghamGravity tests ──────────────────────────────────────────


class TestCunninghamGravity:

    def test_force_model_protocol(self, model_8):
        """CunninghamGravity satisfies the ForceModel protocol."""
        grav = CunninghamGravity(model_8)
        assert isinstance(grav, ForceModel)

    def test_no_nan_at_north_pole(self, model_8, epoch):
        """No NaN at exact North Pole (the polar singularity test)."""
        r = OrbitalConstants.R_EARTH_EQUATORIAL + 400_000.0
        pos = (0.0, 0.0, r)
        vel = (7500.0, 0.0, 0.0)
        grav = CunninghamGravity(model_8)
        acc = grav.acceleration(epoch, pos, vel)
        assert all(math.isfinite(a) for a in acc), f"NaN/Inf at North Pole: {acc}"

    def test_no_nan_at_south_pole(self, model_8, epoch):
        """No NaN at exact South Pole."""
        r = OrbitalConstants.R_EARTH_EQUATORIAL + 400_000.0
        pos = (0.0, 0.0, -r)
        vel = (7500.0, 0.0, 0.0)
        grav = CunninghamGravity(model_8)
        acc = grav.acceleration(epoch, pos, vel)
        assert all(math.isfinite(a) for a in acc), f"NaN/Inf at South Pole: {acc}"

    def test_degree8_matches_existing_sh(self, model_8, epoch):
        """Cunningham degree-8 matches existing SphericalHarmonicGravity within 0.1%.

        SphericalHarmonicGravity includes the central body term, so compare
        TwoBody + CunninghamGravity(8) vs SphericalHarmonicGravity(8).
        """
        r = OrbitalConstants.R_EARTH_EQUATORIAL + 500_000.0
        # Use an inclined position for more interesting gravity field
        pos = (r * 0.6, r * 0.3, r * 0.5)
        vel = (0.0, 7500.0, 0.0)

        # Existing: full gravity
        sh = SphericalHarmonicGravity(max_degree=8)
        acc_sh = sh.acceleration(epoch, pos, vel)

        # New: two-body + Cunningham perturbation
        tb = TwoBodyGravity()
        cg = CunninghamGravity(model_8)
        acc_tb = tb.acceleration(epoch, pos, vel)
        acc_cg = cg.acceleration(epoch, pos, vel)
        acc_new = tuple(a + b for a, b in zip(acc_tb, acc_cg))

        mag_sh = math.sqrt(sum(a ** 2 for a in acc_sh))
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(acc_sh, acc_new)))

        # Allow 0.1% — small differences expected due to different algorithms
        # and different reference constants (EGM96 GM vs WGS84 GM)
        assert diff / mag_sh < 0.001, (
            f"Mismatch: diff={diff:.6e}, mag={mag_sh:.6e}, "
            f"ratio={diff/mag_sh:.6e}"
        )

    def test_acceleration_decreases_with_altitude(self, model_70, epoch):
        """Perturbation magnitude decreases with altitude."""
        grav = CunninghamGravity(model_70)
        vel = (0.0, 7500.0, 0.0)

        r_low = OrbitalConstants.R_EARTH_EQUATORIAL + 300_000.0
        r_high = OrbitalConstants.R_EARTH_EQUATORIAL + 1_000_000.0

        acc_low = grav.acceleration(epoch, (r_low, 0.0, 0.0), vel)
        acc_high = grav.acceleration(epoch, (r_high, 0.0, 0.0), vel)

        mag_low = math.sqrt(sum(a ** 2 for a in acc_low))
        mag_high = math.sqrt(sum(a ** 2 for a in acc_high))
        assert mag_low > mag_high

    def test_perturbation_magnitude_at_leo(self, model_70, epoch, leo_position):
        """Perturbation is ~1e-3 of central body acceleration at LEO."""
        pos, vel = leo_position
        grav = CunninghamGravity(model_70)
        tb = TwoBodyGravity()

        acc_pert = grav.acceleration(epoch, pos, vel)
        acc_tb = tb.acceleration(epoch, pos, vel)

        mag_pert = math.sqrt(sum(a ** 2 for a in acc_pert))
        mag_tb = math.sqrt(sum(a ** 2 for a in acc_tb))

        ratio = mag_pert / mag_tb
        assert 1e-4 < ratio < 1e-2, f"Perturbation ratio: {ratio:.6e}"

    def test_convergence_degree20_vs_70_at_leo(self, epoch, leo_position):
        """Degree 20 and 70 agree within 1% at LEO."""
        pos, vel = leo_position
        m20 = load_gravity_field(max_degree=20)
        m70 = load_gravity_field(max_degree=70)
        g20 = CunninghamGravity(m20)
        g70 = CunninghamGravity(m70)

        acc20 = g20.acceleration(epoch, pos, vel)
        acc70 = g70.acceleration(epoch, pos, vel)

        mag70 = math.sqrt(sum(a ** 2 for a in acc70))
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(acc20, acc70)))
        assert diff / mag70 < 0.01

    def test_convergence_degree8_vs_20_at_geo(self, epoch):
        """Degree 8 and 20 agree within 0.1% at GEO."""
        r_geo = 42_164_000.0
        pos = (r_geo, 0.0, 0.0)
        vel = (0.0, 3075.0, 0.0)

        m8 = load_gravity_field(max_degree=8)
        m20 = load_gravity_field(max_degree=20)
        g8 = CunninghamGravity(m8)
        g20 = CunninghamGravity(m20)

        acc8 = g8.acceleration(epoch, pos, vel)
        acc20 = g20.acceleration(epoch, pos, vel)

        mag20 = math.sqrt(sum(a ** 2 for a in acc20))
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(acc8, acc20)))
        assert diff / mag20 < 0.001

    def test_geo_tesseral_effect(self, epoch):
        """C₂₂/S₂₂ produce nonzero along-track acceleration at GEO."""
        r_geo = 42_164_000.0
        pos = (r_geo, 0.0, 0.0)
        vel = (0.0, 3075.0, 0.0)

        m = load_gravity_field(max_degree=2)
        grav = CunninghamGravity(m)
        acc = grav.acceleration(epoch, pos, vel)

        # Along-track (y) should be nonzero due to tesseral terms
        assert abs(acc[1]) > 0

    def test_returns_tuple_of_three_floats(self, model_8, epoch, leo_position):
        """Acceleration returns a 3-tuple of floats."""
        pos, vel = leo_position
        grav = CunninghamGravity(model_8)
        acc = grav.acceleration(epoch, pos, vel)
        assert isinstance(acc, tuple)
        assert len(acc) == 3
        assert all(isinstance(a, float) for a in acc)


# ── Integration tests ────────────────────────────────────────────────


class TestCunninghamIntegration:

    def test_propagate_numerical_with_cunningham(self, epoch):
        """propagate_numerical() completes with CunninghamGravity."""
        from humeris import (
            ShellConfig,
            generate_walker_shell,
            derive_orbital_state,
        )

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

        m = load_gravity_field(max_degree=20)
        forces = [TwoBodyGravity(), CunninghamGravity(m)]
        result = propagate_numerical(
            state,
            duration=timedelta(minutes=10),
            step=timedelta(seconds=30),
            force_models=forces,
            epoch=epoch,
        )
        assert len(result.steps) > 0
        # Verify no NaN in positions
        for step in result.steps:
            assert all(math.isfinite(p) for p in step.position_eci)

    def test_energy_bounded_one_orbit(self, epoch):
        """Specific energy stays bounded over one orbit (TwoBody + Cunningham)."""
        from humeris import (
            ShellConfig,
            generate_walker_shell,
            derive_orbital_state,
        )

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
        period_s = 2 * math.pi / state.mean_motion_rad_s

        m = load_gravity_field(max_degree=8)
        forces = [TwoBodyGravity(), CunninghamGravity(m)]
        result = propagate_numerical(
            state,
            duration=timedelta(seconds=period_s),
            step=timedelta(seconds=30),
            force_models=forces,
            epoch=epoch,
        )

        mu = OrbitalConstants.MU_EARTH
        energies = []
        for step in result.steps:
            r = math.sqrt(sum(p ** 2 for p in step.position_eci))
            v = math.sqrt(sum(v ** 2 for v in step.velocity_eci))
            energy = 0.5 * v * v - mu / r
            energies.append(energy)

        e_initial = energies[0]
        for e in energies:
            # Energy should stay within 0.2% of initial
            # (perturbation-heavy model at 30s steps has more drift than pure two-body)
            assert abs(e - e_initial) / abs(e_initial) < 0.002

    def test_existing_sh_unchanged(self, epoch):
        """SphericalHarmonicGravity still works (backward compatibility)."""
        r = OrbitalConstants.R_EARTH_EQUATORIAL + 500_000.0
        pos = (r, 0.0, 0.0)
        vel = (0.0, 7500.0, 0.0)

        sh = SphericalHarmonicGravity(max_degree=8)
        acc = sh.acceleration(epoch, pos, vel)
        mag = math.sqrt(sum(a ** 2 for a in acc))
        assert mag > 0


# ── Domain purity ────────────────────────────────────────────────────


class TestDomainPurity:

    def test_gravity_field_domain_purity(self):
        """gravity_field.py must only import from stdlib and domain."""
        import humeris.domain.gravity_field as mod

        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_top = {"math", "numpy", "dataclasses", "typing", "datetime", "json", "pathlib"}
        allowed_internal_prefix = "humeris"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed_top or alias.name.startswith(
                        allowed_internal_prefix
                    ), f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed_top or node.module.startswith(
                        allowed_internal_prefix
                    ), f"Forbidden import from: {node.module}"
