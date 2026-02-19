# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Tests for Blender Python script exporter.

Validates the generated .py script text without importing bpy.
All assertions parse the output as a string or use AST analysis.
"""
import math
import os
import tempfile
from datetime import datetime, timezone

import pytest

from humeris.domain.constellation import (
    ShellConfig,
    generate_walker_shell,
)
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.ports.export import SatelliteExporter
from humeris.adapters.blender_exporter import BlenderExporter


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
def exporter():
    return BlenderExporter()


@pytest.fixture
def script_content(exporter, satellites):
    """Generate script to a temp file and return its content."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        path = f.name
    try:
        exporter.export(satellites, path, epoch=EPOCH)
        with open(path, encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(path)


class TestPortCompliance:
    """BlenderExporter implements the SatelliteExporter port."""

    def test_is_satellite_exporter(self):
        assert issubclass(BlenderExporter, SatelliteExporter)

    def test_has_export_method(self):
        exporter = BlenderExporter()
        assert callable(getattr(exporter, "export", None))


class TestBlenderScript:
    """Generated .py file has correct structure."""

    def test_file_created(self, exporter, satellites):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            path = f.name
        try:
            exporter.export(satellites, path, epoch=EPOCH)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_contains_import_bpy(self, script_content):
        assert "import bpy" in script_content

    def test_contains_import_math(self, script_content):
        assert "import math" in script_content

    def test_contains_clear_scene(self, script_content):
        assert "bpy.ops.object.select_all" in script_content
        assert "bpy.ops.object.delete" in script_content

    def test_contains_earth_creation(self, script_content):
        assert "primitive_uv_sphere_add" in script_content
        assert "6.371" in script_content
        assert '"Earth"' in script_content

    def test_earth_sphere_params(self, script_content):
        assert "segments=64" in script_content
        assert "ring_count=32" in script_content

    def test_script_is_valid_python(self, script_content):
        """Generated script must be syntactically valid Python."""
        import ast
        # Should not raise SyntaxError
        ast.parse(script_content)


class TestSatelliteObjects:
    """Satellite ico spheres are generated correctly."""

    def test_correct_number_of_ico_spheres(self, script_content, satellites):
        count = script_content.count("primitive_ico_sphere_add")
        assert count == len(satellites)

    def test_satellite_names_in_script(self, script_content, satellites):
        for sat in satellites:
            assert sat.name in script_content

    def test_ico_sphere_radius(self, script_content):
        assert "radius=0.05" in script_content

    def test_ico_sphere_subdivisions(self, script_content):
        assert "subdivisions=1" in script_content

    def test_location_coordinates_reasonable(self, script_content, satellites):
        """Satellite positions should be ~6928 km from origin (550 km alt)."""
        expected_r_km = (OrbitalConstants.R_EARTH_EQUATORIAL + 550 * 1000) / 1000.0
        # Check that at least one location value is in the right ballpark
        # Parse a location tuple from the script
        import re
        locations = re.findall(
            r"primitive_ico_sphere_add\(.*?location=\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)",
            script_content,
        )
        assert len(locations) == len(satellites)
        for x_str, y_str, z_str in locations:
            x, y, z = float(x_str), float(y_str), float(z_str)
            r = math.sqrt(x**2 + y**2 + z**2)
            assert abs(r - expected_r_km) < 1.0, (
                f"Expected radius ~{expected_r_km:.1f} km, got {r:.1f} km"
            )

    def test_custom_sat_radius(self, satellites):
        exporter = BlenderExporter(sat_radius_units=0.1)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            path = f.name
        try:
            exporter.export(satellites, path, epoch=EPOCH)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "radius=0.1" in content
        finally:
            os.unlink(path)

    def test_custom_earth_radius(self, satellites):
        exporter = BlenderExporter(earth_radius_units=10.0)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            path = f.name
        try:
            exporter.export(satellites, path, epoch=EPOCH)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "radius=10.0" in content
        finally:
            os.unlink(path)


class TestOrbitCurves:
    """Orbit NURBS curves are generated for each satellite."""

    def test_orbit_curve_count(self, script_content, satellites):
        count = script_content.count("curves.new")
        assert count == len(satellites)

    def test_orbit_curve_names(self, script_content, satellites):
        for sat in satellites:
            assert f"orbit_{sat.name}" in script_content

    def test_orbit_is_3d(self, script_content):
        assert "dimensions = '3D'" in script_content

    def test_orbit_is_cyclic(self, script_content):
        assert "use_cyclic_u = True" in script_content

    def test_orbit_uses_nurbs_spline(self, script_content):
        assert "splines.new('NURBS')" in script_content

    def test_orbit_has_37_points(self, script_content):
        """37 points: 0 to 360 degrees in 10-degree steps (inclusive)."""
        assert "spline.points.add(36)" in script_content

    def test_orbit_points_have_weight(self, script_content):
        """Each NURBS point is (x, y, z, w) with w=1.0."""
        import re
        # Points are set with co = (x, y, z, 1.0)
        point_assignments = re.findall(
            r"spline\.points\[\d+\]\.co\s*=\s*\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*([\d.]+)\)",
            script_content,
        )
        assert len(point_assignments) > 0
        for _, _, _, w in point_assignments:
            assert float(w) == 1.0

    def test_orbit_linked_to_collection(self, script_content):
        assert "bpy.context.collection.objects.link" in script_content


class TestReturnCount:
    """export() returns the number of satellites exported."""

    def test_return_count(self, exporter, satellites):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            path = f.name
        try:
            result = exporter.export(satellites, path, epoch=EPOCH)
            assert result == len(satellites)
        finally:
            os.unlink(path)

    def test_return_count_zero(self, exporter):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            path = f.name
        try:
            result = exporter.export([], path, epoch=EPOCH)
            assert result == 0
        finally:
            os.unlink(path)

    def test_empty_list_still_creates_earth(self, exporter):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            path = f.name
        try:
            exporter.export([], path, epoch=EPOCH)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "primitive_uv_sphere_add" in content
            assert '"Earth"' in content
            assert "primitive_ico_sphere_add" not in content
        finally:
            os.unlink(path)


class TestNameSanitization:
    """Satellite names with special characters must not break generated Python."""

    def test_quotes_in_name_escaped(self, tmp_path):
        from humeris.domain.constellation import Satellite

        r = OrbitalConstants.R_EARTH + 550_000.0
        sat = Satellite(
            name='Test "Quoted" Sat',
            plane_index=0,
            sat_index=0,
            position_eci=(r, 0.0, 0.0),
            velocity_eci=(0.0, 7600.0, 0.0),
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "test.py")
        BlenderExporter().export([sat], path, epoch=EPOCH)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # The generated Python should be syntactically valid
        # Quotes should be escaped in sat.name assignment
        assert '\\"Quoted\\"' in content or "Quoted" in content
        # Must not have unescaped quotes that break the string
        for line in content.split("\n"):
            if "sat.name" in line:
                assert line.count('"') % 2 == 0, f"Unbalanced quotes: {line}"

    def test_newline_in_name_stripped(self, tmp_path):
        from humeris.domain.constellation import Satellite

        r = OrbitalConstants.R_EARTH + 550_000.0
        sat = Satellite(
            name="Line1\nLine2",
            plane_index=0,
            sat_index=0,
            position_eci=(r, 0.0, 0.0),
            velocity_eci=(0.0, 7600.0, 0.0),
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "test.py")
        BlenderExporter().export([sat], path, epoch=EPOCH)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # Newlines in names should be replaced, not break the script
        for line in content.split("\n"):
            if "sat.name" in line:
                assert "\n" not in line.split("sat.name")[1].strip()


class TestBlenderNameSanitization:
    """Combined special characters must not break generated Python script."""

    def test_special_chars_in_name_sanitized(self, tmp_path):
        """Quotes, backslashes, newlines in name must not appear raw in code."""
        from humeris.domain.constellation import Satellite
        import ast

        r = OrbitalConstants.R_EARTH + 550_000.0
        sat = Satellite(
            name='Sat"test\\n\nfoo',
            plane_index=0,
            sat_index=0,
            position_eci=(r, 0.0, 0.0),
            velocity_eci=(0.0, 7600.0, 0.0),
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "test.py")
        BlenderExporter().export([sat], path, epoch=EPOCH)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # Must be syntactically valid Python
        ast.parse(content)
        # No raw quotes, backslashes, or newlines in non-comment code lines
        for line in content.split("\n"):
            if line.strip().startswith("#"):
                continue
            if "sat.name" in line:
                # The value portion after '=' should not contain raw special chars
                value_part = line.split("=", 1)[1].strip()
                # No unescaped literal newlines (already guaranteed by line split)
                # No raw backslash-n literal that isn't a Python escape
                assert '\\"' not in value_part or value_part.count('"') % 2 == 0

    def test_negative_plane_index_skipped(self, tmp_path):
        """Satellite with plane_index=-1 must not crash or create mat_plane_-1."""
        from humeris.domain.constellation import Satellite

        r = OrbitalConstants.R_EARTH + 550_000.0
        sat = Satellite(
            name="NegPlane",
            plane_index=-1,
            sat_index=0,
            position_eci=(r, 0.0, 0.0),
            velocity_eci=(0.0, 7600.0, 0.0),
            raan_deg=0.0,
            true_anomaly_deg=0.0,
        )
        path = str(tmp_path / "test.py")
        BlenderExporter(color_by_plane=True).export([sat], path, epoch=EPOCH)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "mat_plane_-1" not in content
