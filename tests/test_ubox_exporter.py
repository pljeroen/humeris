# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for Universe Sandbox .ubox exporter."""
import math
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone

import pytest

from constellation_generator.domain.constellation import Satellite, ShellConfig, generate_walker_shell
from constellation_generator.domain.atmosphere import DragConfig


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


# ---------------------------------------------------------------------------
# Basic export — ZIP structure and XML validity
# ---------------------------------------------------------------------------

class TestUboxFileStructure:
    """The .ubox file must be a ZIP containing valid XML."""

    def test_creates_valid_zip(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        count = UboxExporter().export(sats, path, epoch=EPOCH)

        assert count == len(sats)
        assert zipfile.is_zipfile(path)

    def test_zip_contains_simulation_xml(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "simulation.xml" in names

    def test_xml_parses_without_error(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        with zipfile.ZipFile(path) as zf:
            xml_bytes = zf.read("simulation.xml")
        root = ET.fromstring(xml_bytes)
        assert root.tag == "Simulation"


# ---------------------------------------------------------------------------
# Earth body
# ---------------------------------------------------------------------------

class TestEarthBody:
    """The simulation must include Earth as the central body."""

    def test_earth_object_present(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        earth_bodies = [b for b in root.findall("Body")
                        if b.find("Object") is not None and b.find("Object").text == "Earth"]
        assert len(earth_bodies) == 1


# ---------------------------------------------------------------------------
# Satellite bodies — Keplerian elements
# ---------------------------------------------------------------------------

class TestSatelliteKeplerianElements:
    """Each satellite should be exported with Keplerian orbital elements."""

    def test_correct_number_of_satellite_bodies(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        sat_bodies = [b for b in root.findall("Body") if b.find("Orbit") is not None]
        assert len(sat_bodies) == len(sats)

    def test_satellite_names_preserved(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        names = {b.find("Name").text for b in root.findall("Body")
                 if b.find("Name") is not None and b.find("Orbit") is not None}
        expected = {s.name for s in sats}
        assert names == expected

    def test_orbit_references_earth(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        for body in root.findall("Body"):
            orbit = body.find("Orbit")
            if orbit is not None:
                assert orbit.get("body") == "Earth"

    def test_semimajor_axis_correct(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter
        from constellation_generator.domain.orbital_mechanics import OrbitalConstants

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        expected_a_km = (OrbitalConstants.R_EARTH + 550_000) / 1000.0

        root = _parse_ubox(path)
        for body in root.findall("Body"):
            orbit = body.find("Orbit")
            if orbit is not None:
                a_km = float(orbit.get("a"))
                assert abs(a_km - expected_a_km) < 0.1

    def test_inclination_correct(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        for body in root.findall("Body"):
            orbit = body.find("Orbit")
            if orbit is not None:
                inc = float(orbit.get("i"))
                assert abs(inc - 53.0) < 0.1

    def test_raan_matches_satellite(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        raan_by_name = {}
        for body in root.findall("Body"):
            orbit = body.find("Orbit")
            name_el = body.find("Name")
            if orbit is not None and name_el is not None:
                raan_by_name[name_el.text] = float(orbit.get("node"))

        for sat in sats:
            expected = sat.raan_deg % 360.0
            actual = raan_by_name[sat.name] % 360.0
            assert abs(actual - expected) < 0.01, f"{sat.name}: {actual} != {expected}"


# ---------------------------------------------------------------------------
# Settings — epoch date
# ---------------------------------------------------------------------------

class TestSettings:
    """Simulation settings should include the epoch."""

    def test_date_setting_present(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        settings = root.find("Settings")
        assert settings is not None
        assert settings.get("date") is not None

    def test_date_matches_epoch(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        settings = root.find("Settings")
        assert settings.get("date") == "2026-03-20"

    def test_focus_on_earth(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        settings = root.find("Settings")
        assert settings.get("focus") == "Earth"


# ---------------------------------------------------------------------------
# Enhanced export — physical properties from DragConfig
# ---------------------------------------------------------------------------

class TestPhysicalProperties:
    """When DragConfig is provided, satellites get mass and diameter."""

    def test_mass_set_from_drag_config(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        path = str(tmp_path / "test.ubox")
        UboxExporter(drag_config=drag).export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        for body in root.findall("Body"):
            mass_el = body.find("Mass")
            if body.find("Orbit") is not None:
                assert mass_el is not None
                # Mass in ubox is 10^20 kg, so 260 kg = 260 / 1e20
                mass_val = float(mass_el.text)
                assert mass_val > 0

    def test_diameter_set_from_drag_config(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        path = str(tmp_path / "test.ubox")
        UboxExporter(drag_config=drag).export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        for body in root.findall("Body"):
            diam_el = body.find("Diameter")
            if body.find("Orbit") is not None:
                assert diam_el is not None
                diam_val = float(diam_el.text)
                # sqrt(10/pi)*2 ≈ 3.57 m = 0.00357 km
                assert diam_val > 0

    def test_no_mass_without_drag_config(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        for body in root.findall("Body"):
            if body.find("Orbit") is not None:
                assert body.find("Mass") is None


# ---------------------------------------------------------------------------
# Return count
# ---------------------------------------------------------------------------

class TestReturnCount:
    """Export returns the number of satellites exported."""

    def test_returns_satellite_count(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        count = UboxExporter().export(sats, path, epoch=EPOCH)
        assert count == 6

    def test_empty_list_returns_zero(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        path = str(tmp_path / "empty.ubox")
        count = UboxExporter().export([], path, epoch=EPOCH)
        assert count == 0

        # Should still produce valid ubox with just Earth
        root = _parse_ubox(path)
        assert root.find("Body") is not None


# ---------------------------------------------------------------------------
# Trail visualization
# ---------------------------------------------------------------------------

class TestTrailSettings:
    """Satellites should have orbit trails enabled."""

    def test_trail_segments_set(self, tmp_path):
        from constellation_generator.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        root = _parse_ubox(path)
        settings = root.find("Settings")
        trail = settings.get("trailsegments")
        assert trail is not None
        assert int(trail) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ubox(path: str) -> ET.Element:
    """Extract and parse simulation.xml from a .ubox file."""
    with zipfile.ZipFile(path) as zf:
        xml_bytes = zf.read("simulation.xml")
    return ET.fromstring(xml_bytes)
