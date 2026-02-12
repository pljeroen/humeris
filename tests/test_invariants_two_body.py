# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Invariant tests for two-body Keplerian mechanics.

These verify mathematical properties that must always hold for the
Keplerian-to-Cartesian transformation, independent of specific inputs.

Invariants A1-A6 from the formal invariant specification.
"""

import math

import pytest

from constellation_generator.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
)


MU = OrbitalConstants.MU_EARTH

# Deterministic parameter grid covering circular, elliptical, and varied geometry.
_TEST_CASES = [
    # (a_m, e, i_rad, raan_rad, argp_rad, nu_rad, label)
    (OrbitalConstants.R_EARTH + 500_000, 0.0, math.radians(53), 0.0, 0.0, 0.0, "LEO circular 53deg"),
    (OrbitalConstants.R_EARTH + 500_000, 0.0, math.radians(53), 0.0, 0.0, math.pi / 2, "LEO circular nu=90"),
    (OrbitalConstants.R_EARTH + 500_000, 0.0, math.radians(97.4), math.radians(45), 0.0, math.radians(180), "SSO circular nu=180"),
    (OrbitalConstants.R_EARTH + 500_000, 0.0, 0.0, 0.0, 0.0, 0.0, "Equatorial circular"),
    (OrbitalConstants.R_EARTH + 500_000, 0.0, math.pi / 2, 0.0, 0.0, math.pi / 4, "Polar circular nu=45"),
    (OrbitalConstants.R_EARTH + 2_000_000, 0.1, math.radians(53), math.radians(90), math.radians(30), math.radians(60), "MEO e=0.1"),
    (OrbitalConstants.R_EARTH + 500_000, 0.01, math.radians(28.5), math.radians(120), math.radians(270), math.radians(45), "Near-circular e=0.01"),
    (26_600_000, 0.0, math.radians(55), math.radians(0), 0.0, math.radians(120), "GPS altitude circular"),
    (42_164_000, 0.0, 0.0, 0.0, 0.0, 0.0, "GEO circular"),
]


def _cross(a, b):
    """Cross product of two 3-vectors."""
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _mag(v):
    return math.sqrt(sum(x ** 2 for x in v))


class TestA1ConicRadialDistance:
    """A1: |r_eci| must match conic section formula r = a(1-e^2)/(1+e*cos(nu))."""

    @pytest.mark.parametrize("a,e,i,raan,argp,nu,label", _TEST_CASES)
    def test_radial_distance(self, a, e, i, raan, argp, nu, label):
        pos, _ = kepler_to_cartesian(a, e, i, raan, argp, nu)
        r_computed = _mag(pos)
        r_expected = a * (1 - e ** 2) / (1 + e * math.cos(nu))
        assert abs(r_computed - r_expected) / r_expected < 1e-10, \
            f"{label}: r_computed={r_computed}, r_expected={r_expected}"


class TestA2AngularMomentumMagnitude:
    """A2: |h| = |r x v| must equal sqrt(mu * a * (1 - e^2))."""

    @pytest.mark.parametrize("a,e,i,raan,argp,nu,label", _TEST_CASES)
    def test_angular_momentum(self, a, e, i, raan, argp, nu, label):
        pos, vel = kepler_to_cartesian(a, e, i, raan, argp, nu)
        h = _cross(pos, vel)
        h_mag = _mag(h)
        h_expected = math.sqrt(MU * a * (1 - e ** 2))
        assert abs(h_mag - h_expected) / h_expected < 1e-10, \
            f"{label}: |h|={h_mag}, expected={h_expected}"


class TestA3OrbitalEnergy:
    """A3: epsilon = v^2/2 - mu/|r| must equal -mu/(2a) for ellipses."""

    @pytest.mark.parametrize("a,e,i,raan,argp,nu,label", _TEST_CASES)
    def test_specific_energy(self, a, e, i, raan, argp, nu, label):
        pos, vel = kepler_to_cartesian(a, e, i, raan, argp, nu)
        r = _mag(pos)
        v = _mag(vel)
        energy_computed = 0.5 * v ** 2 - MU / r
        energy_expected = -MU / (2 * a)
        assert abs(energy_computed - energy_expected) / abs(energy_expected) < 1e-10, \
            f"{label}: epsilon={energy_computed}, expected={energy_expected}"


class TestA4Orthogonality:
    """A4: Angular momentum h is perpendicular to both r and v.

    Uses relative tolerance: |dot(h, r)| / (|h| * |r|) < epsilon,
    since absolute values scale with vector magnitudes.
    """

    @pytest.mark.parametrize("a,e,i,raan,argp,nu,label", _TEST_CASES)
    def test_h_perpendicular_to_r(self, a, e, i, raan, argp, nu, label):
        pos, vel = kepler_to_cartesian(a, e, i, raan, argp, nu)
        h = _cross(pos, vel)
        dot_hr = _dot(h, pos)
        relative = abs(dot_hr) / (_mag(h) * _mag(pos))
        assert relative < 1e-10, f"{label}: |dot(h,r)|/(|h|*|r|)={relative}"

    @pytest.mark.parametrize("a,e,i,raan,argp,nu,label", _TEST_CASES)
    def test_h_perpendicular_to_v(self, a, e, i, raan, argp, nu, label):
        pos, vel = kepler_to_cartesian(a, e, i, raan, argp, nu)
        h = _cross(pos, vel)
        dot_hv = _dot(h, vel)
        relative = abs(dot_hv) / (_mag(h) * _mag(vel))
        assert relative < 1e-10, f"{label}: |dot(h,v)|/(|h|*|v|)={relative}"


class TestA5CircularOrbitSimplifications:
    """A5: For circular orbits (e=0): |r|=a, |v|=sqrt(mu/a), v^2*r=mu."""

    _CIRCULAR = [(a, e, i, raan, argp, nu, label) for a, e, i, raan, argp, nu, label in _TEST_CASES if e == 0.0]

    @pytest.mark.parametrize("a,e,i,raan,argp,nu,label", _CIRCULAR)
    def test_radius_equals_sma(self, a, e, i, raan, argp, nu, label):
        pos, _ = kepler_to_cartesian(a, e, i, raan, argp, nu)
        assert abs(_mag(pos) - a) / a < 1e-10, f"{label}"

    @pytest.mark.parametrize("a,e,i,raan,argp,nu,label", _CIRCULAR)
    def test_speed_equals_circular(self, a, e, i, raan, argp, nu, label):
        _, vel = kepler_to_cartesian(a, e, i, raan, argp, nu)
        v_circ = math.sqrt(MU / a)
        assert abs(_mag(vel) - v_circ) / v_circ < 1e-10, f"{label}"

    @pytest.mark.parametrize("a,e,i,raan,argp,nu,label", _CIRCULAR)
    def test_vis_viva_circular(self, a, e, i, raan, argp, nu, label):
        pos, vel = kepler_to_cartesian(a, e, i, raan, argp, nu)
        assert abs(_mag(vel) ** 2 * _mag(pos) - MU) / MU < 1e-10, f"{label}"


class TestA6PlaneNormalVsInclination:
    """A6: Inclination computed from h_z/|h| must match input inclination."""

    @pytest.mark.parametrize("a,e,i,raan,argp,nu,label", _TEST_CASES)
    def test_inclination_from_angular_momentum(self, a, e, i, raan, argp, nu, label):
        pos, vel = kepler_to_cartesian(a, e, i, raan, argp, nu)
        h = _cross(pos, vel)
        h_mag = _mag(h)
        if h_mag < 1e-10:
            pytest.skip("Degenerate angular momentum")
        i_computed = math.acos(max(-1.0, min(1.0, h[2] / h_mag)))
        assert abs(i_computed - i) < 1e-10, \
            f"{label}: i_computed={math.degrees(i_computed)}, i_input={math.degrees(i)}"
