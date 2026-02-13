# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Tests for satellite enrichment adapter module.

Verifies that compute_satellite_enrichment produces correct orbital
analysis data from a Satellite + epoch.
"""
import math
import os
import tempfile
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.adapters.enrichment import (
    SatelliteEnrichment,
    compute_satellite_enrichment,
)


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

SHELL = ShellConfig(
    altitude_km=550,
    inclination_deg=53,
    num_planes=2,
    sats_per_plane=3,
    phase_factor=1,
    raan_offset_deg=0,
    shell_name="Test",
)


@pytest.fixture
def satellites():
    return generate_walker_shell(SHELL)


@pytest.fixture
def enrichment(satellites):
    return compute_satellite_enrichment(satellites[0], EPOCH)


class TestEnrichmentDataclass:
    """SatelliteEnrichment is a frozen dataclass with the right fields."""

    def test_is_frozen(self, enrichment):
        with pytest.raises(FrozenInstanceError):
            enrichment.altitude_km = 999.0

    def test_has_altitude_km(self, enrichment):
        assert hasattr(enrichment, 'altitude_km')
        assert isinstance(enrichment.altitude_km, float)

    def test_has_inclination_deg(self, enrichment):
        assert hasattr(enrichment, 'inclination_deg')
        assert isinstance(enrichment.inclination_deg, float)

    def test_has_orbital_period_min(self, enrichment):
        assert hasattr(enrichment, 'orbital_period_min')
        assert isinstance(enrichment.orbital_period_min, float)

    def test_has_beta_angle_deg(self, enrichment):
        assert hasattr(enrichment, 'beta_angle_deg')
        assert isinstance(enrichment.beta_angle_deg, float)

    def test_has_atmospheric_density_kg_m3(self, enrichment):
        assert hasattr(enrichment, 'atmospheric_density_kg_m3')
        assert isinstance(enrichment.atmospheric_density_kg_m3, float)

    def test_has_l_shell(self, enrichment):
        assert hasattr(enrichment, 'l_shell')
        assert isinstance(enrichment.l_shell, float)


class TestEnrichmentValues:
    """Computed enrichment values are physically plausible."""

    def test_altitude_near_550km(self, enrichment):
        """Shell altitude is 550 km, so enrichment should be close."""
        assert 540.0 < enrichment.altitude_km < 560.0

    def test_inclination_near_53deg(self, enrichment):
        """Shell inclination is 53 degrees."""
        assert 52.0 < enrichment.inclination_deg < 54.0

    def test_orbital_period_around_96min(self, enrichment):
        """LEO at 550 km has period ~96 minutes."""
        assert 94.0 < enrichment.orbital_period_min < 98.0

    def test_beta_angle_within_range(self, enrichment):
        """Beta angle must be in [-90, 90] degrees."""
        assert -90.0 <= enrichment.beta_angle_deg <= 90.0

    def test_atmospheric_density_positive(self, enrichment):
        """Atmospheric density at 550 km is small but positive."""
        assert enrichment.atmospheric_density_kg_m3 > 0.0
        # At 550 km, density is on the order of 1e-13 kg/m³
        assert enrichment.atmospheric_density_kg_m3 < 1e-10

    def test_l_shell_greater_than_one(self, enrichment):
        """L-shell at LEO altitudes should be > 1."""
        assert enrichment.l_shell > 1.0

    def test_l_shell_reasonable(self, enrichment):
        """L-shell at 550 km, 53° inc should be roughly 1-3."""
        assert enrichment.l_shell < 5.0


class TestEnrichmentWithDifferentOrbits:
    """Enrichment varies correctly with orbit parameters."""

    def test_higher_altitude_longer_period(self):
        """Higher orbit has longer period."""
        shell_low = ShellConfig(
            altitude_km=400, inclination_deg=53,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0,
            shell_name="Low",
        )
        shell_high = ShellConfig(
            altitude_km=800, inclination_deg=53,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0,
            shell_name="High",
        )
        sats_low = generate_walker_shell(shell_low)
        sats_high = generate_walker_shell(shell_high)
        e_low = compute_satellite_enrichment(sats_low[0], EPOCH)
        e_high = compute_satellite_enrichment(sats_high[0], EPOCH)
        assert e_high.orbital_period_min > e_low.orbital_period_min

    def test_higher_altitude_lower_density(self):
        """Higher orbit has lower atmospheric density."""
        shell_low = ShellConfig(
            altitude_km=300, inclination_deg=53,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0,
            shell_name="Low",
        )
        shell_high = ShellConfig(
            altitude_km=800, inclination_deg=53,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0,
            shell_name="High",
        )
        sats_low = generate_walker_shell(shell_low)
        sats_high = generate_walker_shell(shell_high)
        e_low = compute_satellite_enrichment(sats_low[0], EPOCH)
        e_high = compute_satellite_enrichment(sats_high[0], EPOCH)
        assert e_low.atmospheric_density_kg_m3 > e_high.atmospheric_density_kg_m3

    def test_different_inclination_changes_inclination_field(self):
        """Enrichment inclination reflects actual orbital plane."""
        shell_30 = ShellConfig(
            altitude_km=550, inclination_deg=30,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0,
            shell_name="Inc30",
        )
        shell_80 = ShellConfig(
            altitude_km=550, inclination_deg=80,
            num_planes=1, sats_per_plane=1,
            phase_factor=0, raan_offset_deg=0,
            shell_name="Inc80",
        )
        sats_30 = generate_walker_shell(shell_30)
        sats_80 = generate_walker_shell(shell_80)
        e_30 = compute_satellite_enrichment(sats_30[0], EPOCH)
        e_80 = compute_satellite_enrichment(sats_80[0], EPOCH)
        assert 28.0 < e_30.inclination_deg < 32.0
        assert 78.0 < e_80.inclination_deg < 82.0


class TestEnrichmentEdgeCases:
    """Edge cases for enrichment computation."""

    def test_epoch_none_uses_j2000_fallback(self, satellites):
        """When epoch is None, enrichment should still compute."""
        result = compute_satellite_enrichment(satellites[0], None)
        assert result.altitude_km > 0.0
        assert result.orbital_period_min > 0.0

    def test_enrichment_for_all_satellites_in_shell(self, satellites):
        """All satellites in the shell should produce valid enrichment."""
        for sat in satellites:
            e = compute_satellite_enrichment(sat, EPOCH)
            assert 540.0 < e.altitude_km < 560.0
            assert e.orbital_period_min > 0.0
            assert e.atmospheric_density_kg_m3 > 0.0
