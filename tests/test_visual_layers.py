# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Tests for optional visual layers in KML and Blender exporters.

Verifies:
- KML: include_orbits, include_planes, include_isl constructor params
- Blender: include_orbits, color_by_plane constructor params
"""
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pytest

from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.adapters.kml_exporter import KmlExporter
from humeris.adapters.blender_exporter import BlenderExporter


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

SHELL = ShellConfig(
    altitude_km=550,
    inclination_deg=53,
    num_planes=2,
    sats_per_plane=3,
    phase_factor=1,
    raan_offset_deg=0,
    shell_name="Layers",
)

KML_NS = "http://www.opengis.net/kml/2.2"


def _ns(tag):
    return f"{{{KML_NS}}}{tag}"


@pytest.fixture
def satellites():
    return generate_walker_shell(SHELL)


# === KML Visual Layers ===

class TestKmlIncludeOrbits:
    """KML include_orbits parameter controls orbit LineStrings."""

    def test_orbits_present_by_default(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter().export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            linestrings = root.iter(_ns("LineString"))
            count = sum(1 for _ in linestrings)
            assert count == len(satellites)
        finally:
            os.unlink(path)

    def test_orbits_omitted_when_false(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter(include_orbits=False).export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            linestrings = root.iter(_ns("LineString"))
            count = sum(1 for _ in linestrings)
            assert count == 0
        finally:
            os.unlink(path)

    def test_orbits_explicitly_true(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter(include_orbits=True).export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            linestrings = root.iter(_ns("LineString"))
            count = sum(1 for _ in linestrings)
            assert count == len(satellites)
        finally:
            os.unlink(path)

    def test_positions_still_present_without_orbits(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter(include_orbits=False).export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            points = list(root.iter(_ns("Point")))
            assert len(points) == len(satellites)
        finally:
            os.unlink(path)


class TestKmlIncludePlanes:
    """KML include_planes parameter organizes by orbital plane."""

    def test_planes_off_by_default(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter().export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            doc = root.find(_ns("Document"))
            # Default: one Folder per satellite, no plane Folders
            folders = doc.findall(_ns("Folder"))
            assert len(folders) == len(satellites)
        finally:
            os.unlink(path)

    def test_planes_on_creates_plane_folders(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter(include_planes=True).export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            doc = root.find(_ns("Document"))
            # Top-level folders should be plane folders (2 planes)
            plane_folders = doc.findall(_ns("Folder"))
            assert len(plane_folders) == 2
            # Each plane folder contains satellite folders
            for pf in plane_folders:
                sat_folders = pf.findall(_ns("Folder"))
                assert len(sat_folders) == 3  # 3 sats per plane
        finally:
            os.unlink(path)


class TestKmlIncludeIsl:
    """KML include_isl parameter adds ISL topology LineStrings."""

    def test_isl_off_by_default(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "ISL" not in content
        finally:
            os.unlink(path)

    def test_isl_on_adds_isl_folder(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter(include_isl=True).export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            doc = root.find(_ns("Document"))
            folders = doc.findall(_ns("Folder"))
            folder_names = [f.find(_ns("name")).text for f in folders]
            assert "ISL Topology" in folder_names
        finally:
            os.unlink(path)


# === Blender Visual Layers ===

class TestBlenderIncludeOrbits:
    """Blender include_orbits parameter controls NURBS orbit curves."""

    def test_orbits_present_by_default(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.py')
        os.close(fd)
        try:
            BlenderExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "curves.new" in content
        finally:
            os.unlink(path)

    def test_orbits_omitted_when_false(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.py')
        os.close(fd)
        try:
            BlenderExporter(include_orbits=False).export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "curves.new" not in content
        finally:
            os.unlink(path)

    def test_satellites_still_present_without_orbits(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.py')
        os.close(fd)
        try:
            BlenderExporter(include_orbits=False).export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            count = content.count("primitive_ico_sphere_add")
            assert count == len(satellites)
        finally:
            os.unlink(path)


class TestBlenderColorByPlane:
    """Blender color_by_plane parameter assigns materials per plane."""

    def test_no_materials_by_default(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.py')
        os.close(fd)
        try:
            BlenderExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "materials.new" not in content
        finally:
            os.unlink(path)

    def test_materials_created_when_true(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.py')
        os.close(fd)
        try:
            BlenderExporter(color_by_plane=True).export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "materials.new" in content
        finally:
            os.unlink(path)

    def test_material_assigned_to_satellites(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.py')
        os.close(fd)
        try:
            BlenderExporter(color_by_plane=True).export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "data.materials.append" in content
        finally:
            os.unlink(path)

    def test_script_is_valid_python(self, satellites):
        import ast
        fd, path = tempfile.mkstemp(suffix='.py')
        os.close(fd)
        try:
            BlenderExporter(color_by_plane=True, include_orbits=False).export(
                satellites, path, epoch=EPOCH
            )
            with open(path) as f:
                content = f.read()
            ast.parse(content)
        finally:
            os.unlink(path)
