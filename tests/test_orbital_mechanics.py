# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for orbital mechanics edge cases.

Covers kepler_to_cartesian, sso_inclination_deg, and j2_raan_rate
with boundary and edge-case inputs.
"""
import math

import pytest

from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
    sso_inclination_deg,
    j2_raan_rate,
)


class TestOrbitalMechanicsEdgeCases:
    """Edge-case tests for orbital mechanics functions."""

    def test_kepler_to_cartesian_circular(self):
        """Known 550km altitude circular orbit at i=53 deg.

        Verify altitude matches within 1 km.
        """
        alt_km = 550.0
        a = OrbitalConstants.R_EARTH + alt_km * 1000.0
        e = 0.0
        i_rad = math.radians(53.0)
        raan_rad = 0.0
        argp_rad = 0.0
        nu_rad = 0.0

        pos_eci, vel_eci = kepler_to_cartesian(a, e, i_rad, raan_rad, argp_rad, nu_rad)

        # Position magnitude should equal semi-major axis for circular orbit at nu=0
        r = math.sqrt(pos_eci[0] ** 2 + pos_eci[1] ** 2 + pos_eci[2] ** 2)
        computed_alt_km = (r - OrbitalConstants.R_EARTH) / 1000.0
        assert abs(computed_alt_km - alt_km) < 1.0, (
            f"Expected altitude ~{alt_km} km, got {computed_alt_km:.2f} km"
        )

    def test_sso_inclination_550km(self):
        """SSO inclination at 550 km should be approximately 97.4 deg."""
        inc_deg = sso_inclination_deg(550.0)
        assert abs(inc_deg - 97.4) < 1.0, (
            f"Expected SSO inclination ~97.4 deg, got {inc_deg:.2f} deg"
        )

    def test_sso_inclination_negative_altitude_raises(self):
        """Negative altitude should raise ValueError."""
        with pytest.raises(ValueError):
            sso_inclination_deg(-100.0)

    def test_j2_raan_rate_equatorial(self):
        """Equatorial orbit (i=0) should have finite RAAN rate."""
        alt_km = 550.0
        a = OrbitalConstants.R_EARTH + alt_km * 1000.0
        n = math.sqrt(OrbitalConstants.MU_EARTH / a ** 3)
        e = 0.0
        i_rad = 0.0

        rate = j2_raan_rate(n, a, e, i_rad)
        assert math.isfinite(rate), f"RAAN rate should be finite, got {rate}"
        # Equatorial orbit: cos(0) = 1, so rate should be at its maximum magnitude
        assert rate != 0.0, "Equatorial RAAN rate should be non-zero"

    def test_j2_raan_rate_polar(self):
        """Polar orbit (i=90 deg) should have RAAN rate approximately zero."""
        alt_km = 550.0
        a = OrbitalConstants.R_EARTH + alt_km * 1000.0
        n = math.sqrt(OrbitalConstants.MU_EARTH / a ** 3)
        e = 0.0
        i_rad = math.radians(90.0)

        rate = j2_raan_rate(n, a, e, i_rad)
        assert abs(rate) < 1e-10, (
            f"Polar RAAN rate should be ~0, got {rate}"
        )
