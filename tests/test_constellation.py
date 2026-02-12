# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for synthetic constellation generation."""
import math
import os
import re

import pytest

from constellation_generator.domain.orbital_mechanics import (
    kepler_to_cartesian,
    OrbitalConstants,
    sso_inclination_deg,
)
from constellation_generator.domain.constellation import (
    ShellConfig,
    Satellite,
    generate_walker_shell,
    generate_sso_band_configs,
)
from constellation_generator.domain.serialization import (
    format_position,
    format_velocity,
    build_satellite_entity,
)


# ── Kepler → Cartesian ──────────────────────────────────────────────

class TestKeplerToCartesian:

    def test_circular_orbit_radius(self):
        a = OrbitalConstants.R_EARTH + 500_000
        pos, vel = kepler_to_cartesian(a=a, e=0.0, i_rad=0.0,
                                        omega_big_rad=0.0, omega_small_rad=0.0,
                                        nu_rad=0.0)
        r = math.sqrt(sum(p**2 for p in pos))
        assert abs(r - a) < 1.0

    def test_circular_orbit_velocity_magnitude(self):
        a = OrbitalConstants.R_EARTH + 500_000
        pos, vel = kepler_to_cartesian(a=a, e=0.0, i_rad=0.0,
                                        omega_big_rad=0.0, omega_small_rad=0.0,
                                        nu_rad=0.0)
        v_mag = math.sqrt(sum(v**2 for v in vel))
        v_expected = math.sqrt(OrbitalConstants.MU_EARTH / a)
        assert abs(v_mag - v_expected) < 0.1

    def test_circular_orbit_perpendicularity(self):
        a = OrbitalConstants.R_EARTH + 500_000
        pos, vel = kepler_to_cartesian(a=a, e=0.0, i_rad=math.radians(45),
                                        omega_big_rad=math.radians(30),
                                        omega_small_rad=0.0,
                                        nu_rad=math.radians(60))
        dot = sum(p * v for p, v in zip(pos, vel))
        assert abs(dot) < 1.0

    def test_equatorial_orbit_z_zero(self):
        a = 7_000_000
        pos, vel = kepler_to_cartesian(a=a, e=0.0, i_rad=0.0,
                                        omega_big_rad=0.0, omega_small_rad=0.0,
                                        nu_rad=math.radians(45))
        assert abs(pos[2]) < 1e-6
        assert abs(vel[2]) < 1e-6

    def test_multiple_true_anomalies_same_radius(self):
        a = 7_000_000
        for nu_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            pos, _ = kepler_to_cartesian(a=a, e=0.0, i_rad=math.radians(53),
                                          omega_big_rad=0.0, omega_small_rad=0.0,
                                          nu_rad=math.radians(nu_deg))
            r = math.sqrt(sum(p**2 for p in pos))
            assert abs(r - a) < 1.0


# ── SSO inclination ─────────────────────────────────────────────────

class TestSSOInclination:

    def test_sso_inclination_retrograde(self):
        for alt_km in [525, 800, 1200, 1800, 2200]:
            inc = sso_inclination_deg(alt_km)
            assert 90 < inc < 180

    def test_sso_inclination_increases_with_altitude(self):
        prev_inc = 0
        for alt_km in range(525, 2201, 100):
            inc = sso_inclination_deg(alt_km)
            assert inc > prev_inc
            prev_inc = inc

    def test_sso_500km_reference(self):
        inc = sso_inclination_deg(500)
        assert 96.5 < inc < 98.5


# ── Walker constellation ────────────────────────────────────────────

class TestWalkerConstellation:

    def test_total_satellite_count(self):
        config = ShellConfig(altitude_km=500, inclination_deg=30,
                           num_planes=3, sats_per_plane=4, phase_factor=1,
                           raan_offset_deg=0, shell_name="Test")
        sats = generate_walker_shell(config)
        assert len(sats) == 12

    def test_raan_spacing(self):
        config = ShellConfig(altitude_km=500, inclination_deg=53,
                           num_planes=4, sats_per_plane=2, phase_factor=1,
                           raan_offset_deg=10, shell_name="Test")
        sats = generate_walker_shell(config)

        planes = {}
        for sat in sats:
            if sat.plane_index not in planes:
                planes[sat.plane_index] = sat.raan_deg

        raans = [planes[i] for i in sorted(planes.keys())]
        for i in range(len(raans) - 1):
            spacing = raans[i + 1] - raans[i]
            assert abs(spacing - 90.0) < 0.01

        assert abs(raans[0] - 10.0) < 0.01

    def test_unique_names(self):
        config = ShellConfig(altitude_km=500, inclination_deg=30,
                           num_planes=5, sats_per_plane=10, phase_factor=1,
                           raan_offset_deg=0, shell_name="UniqueTest")
        sats = generate_walker_shell(config)
        names = [s.name for s in sats]
        assert len(names) == len(set(names))

    def test_phase_factor_applied(self):
        cfg_no = ShellConfig(altitude_km=500, inclination_deg=30,
                            num_planes=3, sats_per_plane=4, phase_factor=0,
                            raan_offset_deg=0, shell_name="NoPhase")
        cfg_with = ShellConfig(altitude_km=500, inclination_deg=30,
                              num_planes=3, sats_per_plane=4, phase_factor=1,
                              raan_offset_deg=0, shell_name="WithPhase")

        sats_no = generate_walker_shell(cfg_no)
        sats_with = generate_walker_shell(cfg_with)

        diff_no = sum((a - b)**2 for a, b in zip(sats_no[0].position_eci, sats_no[4].position_eci))
        diff_with = sum((a - b)**2 for a, b in zip(sats_with[0].position_eci, sats_with[4].position_eci))
        assert diff_no != pytest.approx(diff_with, rel=0.01)


# ── Serialization format ────────────────────────────────────────────

class TestSerialization:

    def test_position_format(self):
        result = format_position([1234.567, 8901.234, 5678.901])
        assert result == "1234.567;5678.901;8901.234"

    def test_velocity_format(self):
        result = format_velocity([1.234567, 8.901234, 5.678901])
        assert result == "1.234567;5.678901;8.901234"

    def test_position_format_regex(self):
        result = format_position([-1234.5, 0.0, 99999.123456789])
        assert re.match(r"^-?\d+\.\d{3};-?\d+\.\d{3};-?\d+\.\d{3}$", result)

    def test_velocity_format_regex(self):
        result = format_velocity([-0.001, 7654.321, -999.0])
        assert re.match(r"^-?\d+\.\d{6};-?\d+\.\d{6};-?\d+\.\d{6}$", result)


# ── Shell configuration ─────────────────────────────────────────────

class TestShellConfiguration:

    def test_sso_band_generation(self):
        configs = generate_sso_band_configs(start_alt_km=525, end_alt_km=700,
                                            step_km=50, sats_per_plane=72)
        altitudes = [c.altitude_km for c in configs]
        assert altitudes == [525, 575, 625, 675]

    def test_sso_band_inclinations_are_sso(self):
        configs = generate_sso_band_configs(start_alt_km=525, end_alt_km=600,
                                            step_km=50, sats_per_plane=72)
        for c in configs:
            assert 90 < c.inclination_deg < 180

    def test_shell_config_immutable(self):
        config = ShellConfig(altitude_km=500, inclination_deg=30,
                           num_planes=3, sats_per_plane=4, phase_factor=1,
                           raan_offset_deg=0, shell_name="Test")
        with pytest.raises((AttributeError, TypeError)):
            config.altitude_km = 600


# ── Domain purity ───────────────────────────────────────────────────

class TestDomainPurity:

    def test_domain_imports_only_allowed_stdlib(self):
        """Domain modules must not import json, os, sys, pathlib, etc."""
        import ast
        allowed = {'math', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
        domain_dir = os.path.join(os.path.dirname(__file__), '..', 'src',
                                  'constellation_generator', 'domain')
        for fname in os.listdir(domain_dir):
            if not fname.endswith('.py') or fname == '__init__.py':
                continue
            with open(os.path.join(domain_dir, fname)) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split('.')[0]
                        if root not in allowed and not root.startswith('constellation_generator'):
                            assert False, f"Disallowed import '{alias.name}' in domain/{fname}"
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:  # absolute imports
                        root = node.module.split('.')[0]
                        if root not in allowed and root != 'constellation_generator':
                            assert False, f"Disallowed import from '{node.module}' in domain/{fname}"


# ── Integration ─────────────────────────────────────────────────────

class TestIntegration:

    def test_build_satellite_entities(self):
        config = ShellConfig(altitude_km=500, inclination_deg=30,
                           num_planes=2, sats_per_plane=3, phase_factor=1,
                           raan_offset_deg=0, shell_name="IntTest")
        sats = generate_walker_shell(config)

        template = {"Name": "Satellite", "Id": 0, "Mass": 100}
        entities = []
        for idx, sat in enumerate(sats):
            entity = build_satellite_entity(sat, template, base_id=100 + idx)
            entities.append(entity)

        assert len(entities) == 6
        assert all(e['Id'] != 0 for e in entities)
        assert all(';' in e['Position'] for e in entities)
        assert all(';' in e['Velocity'] for e in entities)
        assert template['Id'] == 0  # not mutated
