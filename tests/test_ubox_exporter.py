# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for Universe Sandbox .ubox exporter."""
import json
import math
import zipfile
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


def _parse_simulation(path: str) -> dict:
    """Extract and parse simulation.json from a .ubox file."""
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("simulation.json"))


# ---------------------------------------------------------------------------
# ZIP structure and metadata files
# ---------------------------------------------------------------------------

class TestUboxFileStructure:
    """The .ubox file must be a ZIP containing simulation.json + metadata."""

    def test_creates_valid_zip(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        count = UboxExporter().export(sats, path, epoch=EPOCH)

        assert count == len(sats)
        assert zipfile.is_zipfile(path)

    def test_zip_contains_simulation_json(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "simulation.json" in names

    def test_zip_contains_version_ini(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        with zipfile.ZipFile(path) as zf:
            assert "version.ini" in zf.namelist()

    def test_zip_contains_info_json(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        with zipfile.ZipFile(path) as zf:
            assert "info.json" in zf.namelist()
            info = json.loads(zf.read("info.json"))
            assert "Name" in info

    def test_zip_contains_ui_state_json(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        with zipfile.ZipFile(path) as zf:
            assert "ui-state.json" in zf.namelist()

    def test_simulation_json_parses_valid(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        assert "Settings" in sim
        assert "Entities" in sim


# ---------------------------------------------------------------------------
# Earth entity
# ---------------------------------------------------------------------------

class TestEarthEntity:
    """The simulation must include Earth as the central body."""

    def test_earth_entity_present(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        earth = [e for e in sim["Entities"] if e.get("Name") == "Earth"]
        assert len(earth) == 1

    def test_earth_is_body_type(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        earth = [e for e in sim["Entities"] if e.get("Name") == "Earth"][0]
        assert earth["$type"] == "Body"

    def test_earth_has_id_3(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        earth = [e for e in sim["Entities"] if e.get("Name") == "Earth"][0]
        assert earth["Id"] == 3

    def test_earth_has_no_parent(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        earth = [e for e in sim["Entities"] if e.get("Name") == "Earth"][0]
        assert earth["Parent"] == -1

    def test_earth_has_celestial_component(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        earth = [e for e in sim["Entities"] if e.get("Name") == "Earth"][0]
        types = {c["$type"] for c in earth["Components"]}
        assert "Celestial" in types

    def test_earth_has_appearance_component(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        earth = [e for e in sim["Entities"] if e.get("Name") == "Earth"][0]
        types = {c["$type"] for c in earth["Components"]}
        assert "AppearanceComponent" in types


# ---------------------------------------------------------------------------
# Satellite entities — ECI state vectors
# ---------------------------------------------------------------------------

class TestSatelliteEntities:
    """Each satellite should be exported with ECI state vectors."""

    def test_correct_number_of_satellite_entities(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        sat_entities = [e for e in sim["Entities"] if e.get("Name") != "Earth"]
        assert len(sat_entities) == len(sats)

    def test_satellite_names_preserved(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        names = {e["Name"] for e in sim["Entities"] if e.get("Name") != "Earth"}
        expected = {s.name for s in sats}
        assert names == expected

    def test_satellites_reference_earth_as_parent(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                assert entity["Parent"] == 3

    def test_satellites_have_relative_to_1(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                assert entity["RelativeTo"] == 1

    def test_satellite_flags_146(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                assert entity["Flags"] == 146

    def test_position_is_semicolon_separated(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                pos = entity["Position"]
                parts = pos.split(";")
                assert len(parts) == 3
                for p in parts:
                    float(p)  # must parse as float

    def test_velocity_is_semicolon_separated(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                vel = entity["Velocity"]
                parts = vel.split(";")
                assert len(parts) == 3
                for p in parts:
                    float(p)

    def test_position_magnitude_is_orbital_radius(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter
        from humeris.domain.orbital_mechanics import OrbitalConstants

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        expected_r = OrbitalConstants.R_EARTH + 550_000.0
        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                parts = [float(p) for p in entity["Position"].split(";")]
                r = math.sqrt(sum(p**2 for p in parts))
                assert abs(r - expected_r) / expected_r < 0.001

    def test_unique_entity_ids(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        ids = [e["Id"] for e in sim["Entities"]]
        assert len(ids) == len(set(ids))

    def test_satellite_has_particle_component(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                types = {c["$type"] for c in entity["Components"]}
                assert "ParticleComponent" in types


# ---------------------------------------------------------------------------
# Settings — epoch date
# ---------------------------------------------------------------------------

class TestSettings:
    """Simulation settings should include the epoch and camera targeting Earth."""

    def test_date_setting_present(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        assert "Date" in sim

    def test_date_contains_epoch(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        assert "2026-03-20" in sim["Date"]

    def test_camera_targets_earth(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        assert sim["Settings"]["CameraTargetId"] == 3

    def test_custom_name(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter(name="MyConstellation").export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        assert sim["Name"] == "MyConstellation"


# ---------------------------------------------------------------------------
# Enhanced export — physical properties from DragConfig
# ---------------------------------------------------------------------------

class TestPhysicalProperties:
    """When DragConfig is provided, satellites get mass and radius."""

    def test_mass_set_from_drag_config(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        path = str(tmp_path / "test.ubox")
        UboxExporter(drag_config=drag).export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                assert entity["Mass"] == 260.0
                assert entity["PhysicsMass"] == 260.0

    def test_radius_derived_from_area(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
        path = str(tmp_path / "test.ubox")
        UboxExporter(drag_config=drag).export(sats, path, epoch=EPOCH)

        expected_radius = math.sqrt(10.0 / math.pi)
        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                assert abs(entity["Radius"] - expected_radius) < 0.001

    def test_default_mass_without_drag_config(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        UboxExporter().export(sats, path, epoch=EPOCH)

        sim = _parse_simulation(path)
        for entity in sim["Entities"]:
            if entity.get("Name") != "Earth":
                assert entity["Mass"] == 500.0


# ---------------------------------------------------------------------------
# Return count
# ---------------------------------------------------------------------------

class TestReturnCount:
    """Export returns the number of satellites exported."""

    def test_returns_satellite_count(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        sats = _make_satellites()
        path = str(tmp_path / "test.ubox")
        count = UboxExporter().export(sats, path, epoch=EPOCH)
        assert count == 6

    def test_empty_list_returns_zero(self, tmp_path):
        from humeris.adapters.ubox_exporter import UboxExporter

        path = str(tmp_path / "empty.ubox")
        count = UboxExporter().export([], path, epoch=EPOCH)
        assert count == 0

        sim = _parse_simulation(path)
        earth = [e for e in sim["Entities"] if e.get("Name") == "Earth"]
        assert len(earth) == 1
