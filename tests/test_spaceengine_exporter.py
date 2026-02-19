# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for SpaceEngine .sc catalog exporter."""
import math
import re
from datetime import datetime, timezone

import pytest

from humeris.domain.constellation import Satellite, ShellConfig, generate_walker_shell
from humeris.domain.atmosphere import DragConfig
from humeris.domain.orbital_mechanics import OrbitalConstants


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

SHELL = ShellConfig(
    altitude_km=550, inclination_deg=53, num_planes=2, sats_per_plane=3,
    phase_factor=1, raan_offset_deg=0, shell_name="Test",
)


def _make_satellites() -> list[Satellite]:
    return generate_walker_shell(SHELL)


def _read_sc(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# File structure
# ---------------------------------------------------------------------------

class TestScFileStructure:
    """The .sc file must be valid SpaceEngine catalog syntax."""

    def test_creates_file(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "constellation.sc")
        count = SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        assert count == len(sats)
        assert (tmp_path / "constellation.sc").exists()

    def test_returns_satellite_count(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        count = SpaceEngineExporter().export(sats, path, epoch=EPOCH)
        assert count == 6

    def test_empty_list_returns_zero(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        path = str(tmp_path / "empty.sc")
        count = SpaceEngineExporter().export([], path, epoch=EPOCH)
        assert count == 0

    def test_file_is_utf8_text(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        assert len(content) > 0
        assert isinstance(content, str)


# ---------------------------------------------------------------------------
# Object type and parent
# ---------------------------------------------------------------------------

class TestObjectDefinition:
    """Each satellite should be a Moon object parented to Earth."""

    def test_each_satellite_has_moon_tag(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        # SpaceEngine uses Moon type for artificial satellites orbiting planets
        moon_count = len(re.findall(r'^Moon\s+"', content, re.MULTILINE))
        assert moon_count == len(sats)

    def test_parent_body_is_earth(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        parent_count = content.count('ParentBody "Earth"')
        assert parent_count == len(sats)

    def test_satellite_names_preserved(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        for sat in sats:
            assert f'"{sat.name}"' in content


# ---------------------------------------------------------------------------
# Orbital elements
# ---------------------------------------------------------------------------

class TestOrbitalElements:
    """Keplerian elements must be correct in SpaceEngine units."""

    def test_semimajor_axis_in_au(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        # 550 km altitude = R_earth + 550km ≈ 6928 km ≈ 4.63e-5 AU
        expected_a_m = OrbitalConstants.R_EARTH_EQUATORIAL + 550_000
        expected_a_au = expected_a_m / 1.496e11
        matches = re.findall(r'SemiMajorAxis\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            actual = float(m)
            assert abs(actual - expected_a_au) / expected_a_au < 0.001

    def test_inclination_correct(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        matches = re.findall(r'Inclination\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert abs(float(m) - 53.0) < 0.1

    def test_eccentricity_zero_for_circular(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        matches = re.findall(r'Eccentricity\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert float(m) == 0.0

    def test_ascending_node_matches_raan(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        # Extract name-node pairs by parsing blocks
        blocks = re.findall(
            r'Moon\s+"([^"]+)".*?AscendingNode\s+([\d.eE+-]+)',
            content, re.DOTALL,
        )
        assert len(blocks) == len(sats)

        raan_by_name = {sat.name: sat.raan_deg % 360.0 for sat in sats}
        for name, node_str in blocks:
            expected = raan_by_name[name]
            actual = float(node_str) % 360.0
            assert abs(actual - expected) < 0.01, f"{name}: {actual} != {expected}"

    def test_mean_anomaly_matches_true_anomaly(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        blocks = re.findall(
            r'Moon\s+"([^"]+)".*?MeanAnomaly\s+([\d.eE+-]+)',
            content, re.DOTALL,
        )
        assert len(blocks) == len(sats)

        nu_by_name = {sat.name: sat.true_anomaly_deg % 360.0 for sat in sats}
        for name, ma_str in blocks:
            expected = nu_by_name[name]
            actual = float(ma_str) % 360.0
            assert abs(actual - expected) < 0.01

    def test_ref_plane_is_equator(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        ref_count = content.count('RefPlane "Equator"')
        assert ref_count == len(sats)

    def test_period_in_years(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        # Orbital period for 550km LEO ≈ 96 min ≈ 1.83e-4 years
        a_m = OrbitalConstants.R_EARTH_EQUATORIAL + 550_000
        period_s = 2 * math.pi * math.sqrt(a_m**3 / OrbitalConstants.MU_EARTH)
        expected_yr = period_s / (365.25 * 86400)

        matches = re.findall(r'Period\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            actual = float(m)
            assert abs(actual - expected_yr) / expected_yr < 0.001


# ---------------------------------------------------------------------------
# Physical properties from DragConfig
# ---------------------------------------------------------------------------

class TestPhysicalProperties:
    """When DragConfig is provided, satellites get mass and radius."""

    def test_radius_set_from_drag_config(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter(drag_config=drag).export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        matches = re.findall(r'Radius\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert float(m) > 0

    def test_mass_set_from_drag_config(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter(drag_config=drag).export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        matches = re.findall(r'Mass\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert float(m) > 0

    def test_no_mass_without_drag_config(self, tmp_path):
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        content = _read_sc(path)
        assert "Mass" not in content


class TestAcosClamping:
    """math.acos argument must be clamped to [-1, 1]."""

    def test_near_polar_orbit_no_domain_error(self, tmp_path):
        """Satellite with hz/h_mag ≈ 1.0 should not raise ValueError."""
        from humeris.domain.constellation import Satellite
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter
        from humeris.domain.orbital_mechanics import OrbitalConstants

        r = OrbitalConstants.R_EARTH + 550_000.0
        sat = Satellite(
            name="NearPolar",
            plane_index=0,
            sat_index=0,
            position_eci=(r, 0.0, 0.0),
            velocity_eci=(0.0, 7600.0, 1e-10),
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "polar.sc")
        SpaceEngineExporter().export([sat], path, epoch=EPOCH)


class TestSpaceEngineNameSanitization:
    """Satellite names with special characters must not break .sc output."""

    def test_special_chars_stripped(self, tmp_path):
        """Quotes and newlines in name must be sanitized in the .sc catalog."""
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sat = Satellite(
            name='Test"sat\nfoo',
            plane_index=0,
            sat_index=0,
            position_eci=(7e6, 0.0, 0.0),
            velocity_eci=(0.0, 7500.0, 0.0),
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export([sat], path, epoch=EPOCH)

        content = _read_sc(path)
        # Each Moon line must have balanced quotes and no raw newlines within name
        for line in content.split("\n"):
            if line.strip().startswith("Moon"):
                assert line.count('"') % 2 == 0, f"Unbalanced quotes: {line}"

    def test_zero_position_no_crash(self, tmp_path):
        """Satellite at position (0,0,0) must not crash the exporter."""
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sat = Satellite(
            name="ZeroPos",
            plane_index=0,
            sat_index=0,
            position_eci=(0.0, 0.0, 0.0),
            velocity_eci=(0.0, 0.0, 0.0),
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export([sat], path, epoch=EPOCH)

        content = _read_sc(path)
        assert len(content) > 0

    def test_trailing_newline(self, tmp_path):
        """Exported file must end with a single newline."""
        from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

        sats = _make_satellites()[:1]
        path = str(tmp_path / "test.sc")
        SpaceEngineExporter().export(sats, path, epoch=EPOCH)

        with open(path, "rb") as f:
            raw = f.read()
        assert raw.endswith(b"\n"), "File must end with a newline"
