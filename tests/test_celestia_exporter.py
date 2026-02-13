# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for Celestia .ssc exporter."""
import math
import re
from datetime import datetime, timezone

import pytest

from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.constellation import (
    Satellite,
    ShellConfig,
    generate_walker_shell,
)
from constellation_generator.domain.orbital_mechanics import OrbitalConstants


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


def _read_ssc(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _extract_blocks(text: str) -> list[str]:
    """Extract individual spacecraft blocks from .ssc text."""
    blocks = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if '"Sol/Earth"' in lines[i]:
            depth = 0
            start = i
            while i < len(lines):
                depth += lines[i].count("{")
                depth -= lines[i].count("}")
                i += 1
                if depth <= 0:
                    break
            blocks.append("\n".join(lines[start:i]))
            continue
        i += 1
    return blocks


# ---------------------------------------------------------------------------
# File structure
# ---------------------------------------------------------------------------

class TestFileStructure:
    """The .ssc file must be valid Celestia catalog syntax."""

    def test_creates_file(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "constellation.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        assert (tmp_path / "constellation.ssc").exists()

    def test_correct_number_of_objects(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        # Each satellite block starts with "SatName" "Sol/Earth"
        object_count = content.count('"Sol/Earth"')
        assert object_count == len(sats)

    def test_names_preserved(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        for sat in sats:
            assert f'"{sat.name}"' in content

    def test_class_is_spacecraft(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        class_count = content.count('Class "spacecraft"')
        assert class_count == len(sats)

    def test_empty_list_creates_empty_file(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        path = str(tmp_path / "empty.ssc")
        CelestiaExporter().export([], path, epoch=EPOCH)

        content = _read_ssc(path)
        assert '"Sol/Earth"' not in content


# ---------------------------------------------------------------------------
# Orbital elements
# ---------------------------------------------------------------------------

class TestOrbitalElements:
    """Keplerian elements must be correct in Celestia units."""

    def test_semimajor_axis_in_km(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        expected_a_km = (OrbitalConstants.R_EARTH + 550_000) / 1000.0
        matches = re.findall(r'SemiMajorAxis\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            actual = float(m)
            assert abs(actual - expected_a_km) / expected_a_km < 0.001

    def test_inclination_correct(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        matches = re.findall(r'Inclination\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert abs(float(m) - 53.0) < 0.1

    def test_raan_matches_satellite(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        blocks = _extract_blocks(content)
        assert len(blocks) == len(sats)

        for block, sat in zip(blocks, sats):
            match = re.search(r'AscendingNode\s+([\d.eE+-]+)', block)
            assert match is not None
            actual = float(match.group(1)) % 360.0
            expected = sat.raan_deg % 360.0
            assert abs(actual - expected) < 0.01

    def test_mean_anomaly_matches_true_anomaly(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        blocks = _extract_blocks(content)
        assert len(blocks) == len(sats)

        for block, sat in zip(blocks, sats):
            match = re.search(r'MeanAnomaly\s+([\d.eE+-]+)', block)
            assert match is not None
            actual = float(match.group(1)) % 360.0
            expected = sat.true_anomaly_deg % 360.0
            assert abs(actual - expected) < 0.01

    def test_period_in_days(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        a_m = OrbitalConstants.R_EARTH + 550_000
        period_s = 2 * math.pi * math.sqrt(a_m**3 / OrbitalConstants.MU_EARTH)
        expected_days = period_s / 86400.0

        matches = re.findall(r'Period\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            actual = float(m)
            assert abs(actual - expected_days) / expected_days < 0.001

    def test_epoch_julian_date(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        # J2000 = 2451545.0, EPOCH = 2026-03-20 12:00:00 UTC
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        expected_jd = 2451545.0 + (EPOCH - j2000).total_seconds() / 86400.0

        matches = re.findall(r'Epoch\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            actual = float(m)
            assert abs(actual - expected_jd) < 0.001

    def test_eccentricity_zero(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        matches = re.findall(r'Eccentricity\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert float(m) == 0.0


# ---------------------------------------------------------------------------
# Physical properties
# ---------------------------------------------------------------------------

class TestPhysicalProperties:
    """Physical properties from DragConfig and defaults."""

    def test_mass_from_drag_config(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        path = str(tmp_path / "test.ssc")
        CelestiaExporter(drag_config=drag).export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        matches = re.findall(r'Mass\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert abs(float(m) - 260.0) < 0.01

    def test_radius_from_drag_config(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        path = str(tmp_path / "test.ssc")
        CelestiaExporter(drag_config=drag).export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        expected_radius_km = math.sqrt(10.0 / math.pi) / 1000.0
        matches = re.findall(r'Radius\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            actual = float(m)
            assert abs(actual - expected_radius_km) / expected_radius_km < 0.01

    def test_default_radius_without_drag_config(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        matches = re.findall(r'Radius\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert float(m) == 0.001

    def test_no_mass_without_drag_config(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        assert "Mass" not in content


# ---------------------------------------------------------------------------
# Return count
# ---------------------------------------------------------------------------

class TestReturnCount:
    """Export returns the number of satellites exported."""

    def test_returns_satellite_count(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        count = CelestiaExporter().export(sats, path, epoch=EPOCH)
        assert count == 6

    def test_empty_list_returns_zero(self, tmp_path):
        from constellation_generator.adapters.celestia_exporter import CelestiaExporter

        path = str(tmp_path / "empty.ssc")
        count = CelestiaExporter().export([], path, epoch=EPOCH)
        assert count == 0
