# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for Celestia .ssc exporter."""
import math
import re
from datetime import datetime, timezone

import pytest

from humeris.domain.atmosphere import DragConfig
from humeris.domain.constellation import (
    Satellite,
    ShellConfig,
    generate_walker_shell,
)
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
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "constellation.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        assert (tmp_path / "constellation.ssc").exists()

    def test_correct_number_of_objects(self, tmp_path):
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        # Each satellite block starts with "SatName" "Sol/Earth"
        object_count = content.count('"Sol/Earth"')
        assert object_count == len(sats)

    def test_names_preserved(self, tmp_path):
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        for sat in sats:
            assert f'"{sat.name}"' in content

    def test_class_is_spacecraft(self, tmp_path):
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        class_count = content.count('Class "spacecraft"')
        assert class_count == len(sats)

    def test_empty_list_creates_empty_file(self, tmp_path):
        from humeris.adapters.celestia_exporter import CelestiaExporter

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
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        expected_a_km = (OrbitalConstants.R_EARTH_EQUATORIAL + 550_000) / 1000.0
        matches = re.findall(r'SemiMajorAxis\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            actual = float(m)
            assert abs(actual - expected_a_km) / expected_a_km < 0.001

    def test_inclination_correct(self, tmp_path):
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        matches = re.findall(r'Inclination\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert abs(float(m) - 53.0) < 0.1

    def test_raan_matches_satellite(self, tmp_path):
        from humeris.adapters.celestia_exporter import CelestiaExporter

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
        from humeris.adapters.celestia_exporter import CelestiaExporter

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
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        a_m = OrbitalConstants.R_EARTH_EQUATORIAL + 550_000
        period_s = 2 * math.pi * math.sqrt(a_m**3 / OrbitalConstants.MU_EARTH)
        expected_days = period_s / 86400.0

        matches = re.findall(r'Period\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            actual = float(m)
            assert abs(actual - expected_days) / expected_days < 0.001

    def test_epoch_julian_date(self, tmp_path):
        from humeris.adapters.celestia_exporter import CelestiaExporter

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
        from humeris.adapters.celestia_exporter import CelestiaExporter

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
        from humeris.adapters.celestia_exporter import CelestiaExporter

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
        from humeris.adapters.celestia_exporter import CelestiaExporter

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
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        content = _read_ssc(path)
        matches = re.findall(r'Radius\s+([\d.eE+-]+)', content)
        assert len(matches) == len(sats)
        for m in matches:
            assert float(m) == 0.001

    def test_no_mass_without_drag_config(self, tmp_path):
        from humeris.adapters.celestia_exporter import CelestiaExporter

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
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ssc")
        count = CelestiaExporter().export(sats, path, epoch=EPOCH)
        assert count == 6

    def test_empty_list_returns_zero(self, tmp_path):
        from humeris.adapters.celestia_exporter import CelestiaExporter

        path = str(tmp_path / "empty.ssc")
        count = CelestiaExporter().export([], path, epoch=EPOCH)
        assert count == 0


class TestAcosClamping:
    """math.acos argument must be clamped to [-1, 1]."""

    def test_near_polar_orbit_no_domain_error(self, tmp_path):
        """Satellite with hz/h_mag ≈ 1.0 should not raise ValueError."""
        from humeris.domain.constellation import Satellite
        from humeris.adapters.celestia_exporter import CelestiaExporter
        from humeris.domain.orbital_mechanics import OrbitalConstants

        r = OrbitalConstants.R_EARTH + 550_000.0
        # Pure polar orbit: velocity purely in Z direction
        # h = r × v = (r,0,0) × (0,0,vz) = (0, -r*vz, 0)
        # → hz=0, so hz/h_mag = 0. That's fine.
        # Instead: make h nearly parallel to z-axis
        # h = (0, 0, r*v) → hz/h_mag = 1.0 exactly
        sat = Satellite(
            name="NearPolar",
            plane_index=0,
            sat_index=0,
            position_eci=(r, 0.0, 0.0),
            velocity_eci=(0.0, 7600.0, 1e-10),  # tiny z component
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "polar.ssc")
        # This must not raise ValueError from math.acos domain error
        CelestiaExporter().export([sat], path, epoch=EPOCH)


class TestCelestiaNameSanitization:
    """Satellite names with special characters must not break .ssc output."""

    def test_special_chars_stripped(self, tmp_path):
        """Quotes and newlines in name must be sanitized in the .ssc block."""
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sat = Satellite(
            name='Test"sat\nfoo',
            plane_index=0,
            sat_index=0,
            position_eci=(7e6, 0.0, 0.0),
            velocity_eci=(0.0, 7500.0, 0.0),
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export([sat], path, epoch=EPOCH)

        content = _read_ssc(path)
        blocks = _extract_blocks(content)
        assert len(blocks) == 1
        # The block should not contain raw newlines within a quoted name
        # and should not contain unescaped quotes that break the format
        first_line = blocks[0].split("\n")[0]
        # Name is the first quoted string before "Sol/Earth"
        assert "\n" not in first_line
        assert first_line.count('"') % 2 == 0, f"Unbalanced quotes: {first_line}"

    def test_zero_position_no_crash(self, tmp_path):
        """Satellite at position (0,0,0) must not crash the exporter."""
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sat = Satellite(
            name="ZeroPos",
            plane_index=0,
            sat_index=0,
            position_eci=(0.0, 0.0, 0.0),
            velocity_eci=(0.0, 0.0, 0.0),
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export([sat], path, epoch=EPOCH)

        content = _read_ssc(path)
        assert len(content) > 0

    def test_trailing_newline(self, tmp_path):
        """Exported file must end with a single newline."""
        from humeris.adapters.celestia_exporter import CelestiaExporter

        sats = _make_satellites()[:1]
        path = str(tmp_path / "test.ssc")
        CelestiaExporter().export(sats, path, epoch=EPOCH)

        with open(path, "rb") as f:
            raw = f.read()
        assert raw.endswith(b"\n"), "File must end with a newline"
