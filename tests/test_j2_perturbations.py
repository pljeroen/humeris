# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for J2 secular perturbation functions."""
import ast
import math

import pytest

from constellation_generator.domain.orbital_mechanics import (
    OrbitalConstants,
    sso_inclination_deg,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _mean_motion(a_m: float) -> float:
    """Mean motion n = sqrt(mu/a^3) in rad/s."""
    return math.sqrt(OrbitalConstants.MU_EARTH / a_m**3)


# ── RAAN rate ────────────────────────────────────────────────────────

class TestJ2RAANRate:

    def test_prograde_orbit_negative_raan_rate(self):
        """Prograde orbit (i < 90°) → RAAN drifts westward (negative)."""
        from constellation_generator.domain.orbital_mechanics import j2_raan_rate

        a = OrbitalConstants.R_EARTH + 500_000
        n = _mean_motion(a)
        rate = j2_raan_rate(n, a, 0.0, math.radians(53.0))
        assert rate < 0

    def test_retrograde_orbit_positive_raan_rate(self):
        """Retrograde orbit (i > 90°) → RAAN drifts eastward (positive)."""
        from constellation_generator.domain.orbital_mechanics import j2_raan_rate

        a = OrbitalConstants.R_EARTH + 500_000
        n = _mean_motion(a)
        rate = j2_raan_rate(n, a, 0.0, math.radians(120.0))
        assert rate > 0

    def test_polar_orbit_zero_raan_rate(self):
        """Polar orbit (i = 90°) → zero RAAN drift."""
        from constellation_generator.domain.orbital_mechanics import j2_raan_rate

        a = OrbitalConstants.R_EARTH + 500_000
        n = _mean_motion(a)
        rate = j2_raan_rate(n, a, 0.0, math.radians(90.0))
        assert abs(rate) < 1e-20

    def test_sso_raan_rate_matches_earth_orbital_rate(self):
        """At SSO inclination, RAAN rate ≈ Earth's orbital angular velocity (~0.9856°/day)."""
        from constellation_generator.domain.orbital_mechanics import j2_raan_rate

        alt_km = 500.0
        inc_deg = sso_inclination_deg(alt_km)
        a = OrbitalConstants.R_EARTH + alt_km * 1000
        n = _mean_motion(a)
        rate = j2_raan_rate(n, a, 0.0, math.radians(inc_deg))
        rate_deg_per_day = math.degrees(rate) * 86400
        # Earth's orbital angular velocity ≈ 0.9856°/day
        assert abs(rate_deg_per_day - 0.9856) < 0.05


# ── Argument of perigee rate ─────────────────────────────────────────

class TestJ2ArgPerigeeRate:

    def test_critical_inclination_zero_rate(self):
        """At critical inclination (63.4°), argument of perigee rate = 0."""
        from constellation_generator.domain.orbital_mechanics import j2_arg_perigee_rate

        a = OrbitalConstants.R_EARTH + 500_000
        n = _mean_motion(a)
        rate = j2_arg_perigee_rate(n, a, 0.0, math.radians(63.4349))
        assert abs(rate) < 1e-10

    def test_low_inclination_positive_rate(self):
        """Below critical inclination, argument of perigee advances (positive)."""
        from constellation_generator.domain.orbital_mechanics import j2_arg_perigee_rate

        a = OrbitalConstants.R_EARTH + 500_000
        n = _mean_motion(a)
        rate = j2_arg_perigee_rate(n, a, 0.0, math.radians(30.0))
        assert rate > 0

    def test_high_inclination_negative_rate(self):
        """Above critical inclination, argument of perigee regresses (negative)."""
        from constellation_generator.domain.orbital_mechanics import j2_arg_perigee_rate

        a = OrbitalConstants.R_EARTH + 500_000
        n = _mean_motion(a)
        rate = j2_arg_perigee_rate(n, a, 0.0, math.radians(80.0))
        assert rate < 0


# ── Mean motion correction ───────────────────────────────────────────

class TestJ2MeanMotionCorrection:

    def test_corrected_exceeds_unperturbed(self):
        """J2-corrected mean motion > unperturbed for typical LEO."""
        from constellation_generator.domain.orbital_mechanics import j2_mean_motion_correction

        a = OrbitalConstants.R_EARTH + 500_000
        n = _mean_motion(a)
        n_corrected = j2_mean_motion_correction(n, a, 0.0, math.radians(53.0))
        assert n_corrected > n

    def test_correction_returns_float(self):
        """Mean motion correction returns a float."""
        from constellation_generator.domain.orbital_mechanics import j2_mean_motion_correction

        a = OrbitalConstants.R_EARTH + 500_000
        n = _mean_motion(a)
        result = j2_mean_motion_correction(n, a, 0.0, math.radians(53.0))
        assert isinstance(result, float)

    def test_correction_small_relative_to_n(self):
        """J2 correction is small (< 1%) relative to unperturbed mean motion."""
        from constellation_generator.domain.orbital_mechanics import j2_mean_motion_correction

        a = OrbitalConstants.R_EARTH + 500_000
        n = _mean_motion(a)
        n_corrected = j2_mean_motion_correction(n, a, 0.0, math.radians(53.0))
        relative_diff = (n_corrected - n) / n
        assert 0 < relative_diff < 0.01


# ── Domain purity ────────────────────────────────────────────────────

# ── J3 constant ─────────────────────────────────────────────────────

class TestJ3Constant:

    def test_j3_earth_value(self):
        """J3_EARTH zonal harmonic coefficient has expected value."""
        assert OrbitalConstants.J3_EARTH == pytest.approx(-2.53215e-6)


# ── Domain purity ────────────────────────────────────────────────────

class TestJ2Purity:

    def test_orbital_mechanics_still_pure(self):
        """orbital_mechanics.py must only import stdlib modules."""
        import constellation_generator.domain.orbital_mechanics as mod

        allowed = {'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('constellation_generator'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'constellation_generator':
                        assert False, f"Disallowed import from '{node.module}'"
