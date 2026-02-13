# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Invariant tests for J2 secular rates and SSO inclination derivation.

These verify sign relationships, monotonicity, and domain limits
that must hold for J2-based secular perturbation formulas.

Invariants E1-E3 (J2 rates) and F1-F4 (SSO inclination).
"""

import math

import pytest

from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    j2_raan_rate,
    j2_arg_perigee_rate,
    sso_inclination_deg,
)


MU = OrbitalConstants.MU_EARTH
R_E = OrbitalConstants.R_EARTH


def _mean_motion(a):
    return math.sqrt(MU / a ** 3)


# --- E: J2 secular rate invariants ---

class TestE1RaanRateSign:
    """E1: sign(dOmega/dt) == -sign(cos(i)).

    Prograde (i<90) -> cos(i)>0 -> dOmega/dt < 0 (regression).
    Retrograde (i>90) -> cos(i)<0 -> dOmega/dt > 0 (progression).
    """

    _INCLINATIONS = [
        (math.radians(28.5), "Cape Canaveral"),
        (math.radians(53), "Starlink"),
        (math.radians(63.4), "Critical inclination"),
        (math.radians(90), "Polar"),
        (math.radians(97.4), "SSO retrograde"),
        (math.radians(120), "Retrograde"),
    ]

    @pytest.mark.parametrize("i_rad,label", _INCLINATIONS)
    def test_sign_relationship(self, i_rad, label):
        a = R_E + 500_000
        n = _mean_motion(a)
        rate = j2_raan_rate(n, a, 0.0, i_rad)

        cos_i = math.cos(i_rad)
        if abs(cos_i) < 1e-10:
            # Polar: rate should be ~zero
            assert abs(rate) < 1e-15, f"{label}: rate={rate} should be ~0"
        else:
            assert rate * cos_i < 0, \
                f"{label}: sign(rate)={math.copysign(1, rate)}, sign(cos_i)={math.copysign(1, cos_i)}"


class TestE2RaanRateMagnitudeVsAltitude:
    """E2: |dOmega/dt| decreases with increasing semi-major axis."""

    _ALTITUDES_KM = [300, 400, 500, 600, 700, 800, 1000, 1500, 2000]

    def test_magnitude_decreasing(self):
        i_rad = math.radians(53)
        prev_mag = float("inf")
        for alt in self._ALTITUDES_KM:
            a = R_E + alt * 1000
            n = _mean_motion(a)
            rate = j2_raan_rate(n, a, 0.0, i_rad)
            mag = abs(rate)
            assert mag < prev_mag, \
                f"alt={alt}km: |rate|={mag} >= prev {prev_mag}"
            prev_mag = mag


class TestE3RaanRateEccentricityDependence:
    """E3: For fixed a, i, increasing e increases |dOmega/dt| via (1-e^2)^-2."""

    def test_magnitude_increases_with_eccentricity(self):
        a = R_E + 500_000
        i_rad = math.radians(53)
        n = _mean_motion(a)

        eccentricities = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
        prev_mag = 0.0
        for e in eccentricities:
            rate = j2_raan_rate(n, a, e, i_rad)
            mag = abs(rate)
            assert mag >= prev_mag - 1e-20, \
                f"e={e}: |rate|={mag} < prev {prev_mag}"
            prev_mag = mag


# --- F: SSO inclination derivation invariants ---

class TestF1InclinationOutputRange:
    """F1: SSO inclination output in [0, 180] degrees."""

    _SSO_ALTITUDES = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

    @pytest.mark.parametrize("alt_km", _SSO_ALTITUDES)
    def test_output_range(self, alt_km):
        i = sso_inclination_deg(alt_km)
        assert 0.0 <= i <= 180.0, f"alt={alt_km}km: inclination={i} out of [0, 180]"


class TestF2SsoIsRetrograde:
    """F2: SSO inclination should be > 90 degrees (retrograde orbit)."""

    _SSO_ALTITUDES = [200, 400, 600, 800, 1000]

    @pytest.mark.parametrize("alt_km", _SSO_ALTITUDES)
    def test_retrograde(self, alt_km):
        i = sso_inclination_deg(alt_km)
        assert i > 90.0, f"alt={alt_km}km: inclination={i} should be > 90"


class TestF3SsoTrendVsAltitude:
    """F3: SSO inclination increases with altitude in the typical LEO band.

    As altitude increases, r^3.5 increases, making cos(i) more negative,
    pushing inclination further from 90° (more retrograde).
    """

    def test_trend(self):
        altitudes = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        incls = [sso_inclination_deg(alt) for alt in altitudes]
        for i in range(len(incls) - 1):
            assert incls[i] <= incls[i + 1] + 0.01, \
                f"SSO trend violated: i({altitudes[i]})={incls[i]:.3f} > i({altitudes[i+1]})={incls[i+1]:.3f}"


class TestF4SsoFailFast:
    """F4: Outside supported altitude range, function must fail explicitly."""

    def test_very_high_altitude_raises(self):
        """At extreme altitude, cos(i) goes out of [-1, 1] -> acos raises."""
        with pytest.raises((ValueError, Exception)):
            # Very high altitude -> cos_i will be < -1
            sso_inclination_deg(50_000)
