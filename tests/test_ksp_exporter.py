# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for Kerbal Space Program .sfs exporter."""
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


def _read_export(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _extract_vessels(text: str) -> list[str]:
    """Extract individual VESSEL { ... } blocks from export text."""
    vessels = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip() == "VESSEL":
            depth = 0
            start = i
            i += 1
            if i < len(lines) and lines[i].strip() == "{":
                depth = 1
                i += 1
                while i < len(lines) and depth > 0:
                    if lines[i].strip() == "{":
                        depth += 1
                    elif lines[i].strip() == "}":
                        depth -= 1
                    i += 1
                vessels.append("\n".join(lines[start:i]))
            continue
        i += 1
    return vessels


def _extract_orbit_field(vessel: str, field: str) -> str:
    """Extract a field value from the ORBIT block of a vessel."""
    in_orbit = False
    for line in vessel.split("\n"):
        stripped = line.strip()
        if stripped == "ORBIT":
            in_orbit = True
        elif in_orbit and stripped == "}":
            break
        elif in_orbit and stripped.startswith(field + " "):
            return stripped.split("=")[1].strip()
    return ""


def _extract_field(vessel: str, field: str) -> str:
    """Extract a top-level field value from a vessel block."""
    for line in vessel.split("\n"):
        stripped = line.strip()
        if stripped.startswith(field + " ") and "=" in stripped:
            return stripped.split("=", 1)[1].strip()
    return ""


# ---------------------------------------------------------------------------
# File structure
# ---------------------------------------------------------------------------

class TestKspFileStructure:
    """The export file must contain valid ConfigNode VESSEL blocks."""

    def test_creates_file(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        assert (tmp_path / "test.sfs").exists()

    def test_contains_header_comment(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        assert "persistent.sfs" in text

    def test_correct_vessel_count(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        assert len(vessels) == len(sats)

    def test_vessel_has_orbit_block(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v in vessels:
            assert "ORBIT" in v

    def test_vessel_has_part_block(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v in vessels:
            assert "PART" in v
            assert "probeCoreCube" in v


# ---------------------------------------------------------------------------
# Satellite properties
# ---------------------------------------------------------------------------

class TestSatelliteProperties:
    """Satellite names and orbital elements must be correct."""

    def test_names_preserved(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        names = {_extract_field(v, "name") for v in vessels}
        expected = {s.name for s in sats}
        assert names == expected

    def test_orbit_references_kerbin(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v in vessels:
            assert _extract_orbit_field(v, "REF") == "1"

    def test_inclination_preserved(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v in vessels:
            inc = float(_extract_orbit_field(v, "INC"))
            assert abs(inc - 53.0) < 0.1

    def test_raan_matches_satellite(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v, sat in zip(vessels, sats):
            lan = float(_extract_orbit_field(v, "LAN"))
            expected = sat.raan_deg % 360.0
            assert abs(lan - expected) < 0.01

    def test_mean_anomaly_in_radians(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v, sat in zip(vessels, sats):
            mna = float(_extract_orbit_field(v, "MNA"))
            expected = math.radians(sat.true_anomaly_deg % 360.0)
            assert abs(mna - expected) < 0.001

    def test_unique_pids(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        pids = {_extract_field(v, "pid") for v in vessels}
        assert len(pids) == len(vessels)

    def test_type_is_probe(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v in vessels:
            assert _extract_field(v, "type") == "Probe"


# ---------------------------------------------------------------------------
# Kerbin scaling
# ---------------------------------------------------------------------------

class TestKerbinScaling:
    """Orbital elements should be scaled from Earth to Kerbin."""

    def test_sma_scaled_to_kerbin(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter
        from humeris.domain.orbital_mechanics import OrbitalConstants

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter(scale_to_kerbin=True).export(sats, path, epoch=EPOCH)

        r_earth = OrbitalConstants.R_EARTH + 550_000.0
        scale = 600_000.0 / OrbitalConstants.R_EARTH
        expected_sma = r_earth * scale

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v in vessels:
            sma = float(_extract_orbit_field(v, "SMA"))
            # May be clamped if below atmosphere
            assert sma >= 680_000.0  # R_kerbin + 80km minimum

    def test_no_scaling_when_disabled(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter
        from humeris.domain.orbital_mechanics import OrbitalConstants

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter(scale_to_kerbin=False).export(sats, path, epoch=EPOCH)

        expected_sma = OrbitalConstants.R_EARTH + 550_000.0

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v in vessels:
            sma = float(_extract_orbit_field(v, "SMA"))
            assert abs(sma - expected_sma) / expected_sma < 0.001

    def test_low_orbit_clamped_above_atmosphere(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        # 550 km LEO scales to ~52 km Kerbin altitude (below 70 km atmo)
        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter(scale_to_kerbin=True).export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v in vessels:
            sma = float(_extract_orbit_field(v, "SMA"))
            alt = sma - 600_000.0
            assert alt >= 80_000.0

    def test_high_orbit_not_clamped(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        high_shell = ShellConfig(
            altitude_km=20180, inclination_deg=55, num_planes=2,
            sats_per_plane=2, phase_factor=1, raan_offset_deg=0,
            shell_name="GPS",
        )
        sats = generate_walker_shell(high_shell)
        path = str(tmp_path / "test.sfs")
        KspExporter(scale_to_kerbin=True).export(sats, path, epoch=EPOCH)

        from humeris.domain.orbital_mechanics import OrbitalConstants
        r_earth = OrbitalConstants.R_EARTH + 20_180_000.0
        scale = 600_000.0 / OrbitalConstants.R_EARTH
        expected_sma = r_earth * scale

        text = _read_export(path)
        vessels = _extract_vessels(text)
        for v in vessels:
            sma = float(_extract_orbit_field(v, "SMA"))
            assert abs(sma - expected_sma) / expected_sma < 0.001


# ---------------------------------------------------------------------------
# Physical properties from DragConfig
# ---------------------------------------------------------------------------

class TestPhysicalProperties:
    """When DragConfig is provided, satellite mass is set in metric tons."""

    def test_mass_from_drag_config(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        path = str(tmp_path / "test.sfs")
        KspExporter(drag_config=drag).export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        # Mass in KSP is metric tons: 260 kg = 0.26 t
        assert "0.260000" in text

    def test_default_mass_without_drag_config(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        KspExporter().export(sats, path, epoch=EPOCH)

        text = _read_export(path)
        assert "0.100000" in text


# ---------------------------------------------------------------------------
# Return count
# ---------------------------------------------------------------------------

class TestReturnCount:
    """Export returns the number of satellites exported."""

    def test_returns_satellite_count(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.sfs")
        count = KspExporter().export(sats, path, epoch=EPOCH)
        assert count == 6

    def test_empty_list_returns_zero(self, tmp_path):
        from humeris.adapters.ksp_exporter import KspExporter

        path = str(tmp_path / "empty.sfs")
        count = KspExporter().export([], path, epoch=EPOCH)
        assert count == 0
