# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Tests for enrichment data in all exporters.

Verifies that each exporter includes orbital analysis fields
(altitude, inclination, period, beta angle, density, L-shell)
in its output.
"""
import csv
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone

import pytest

from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.adapters.csv_exporter import CsvSatelliteExporter
from humeris.adapters.geojson_exporter import GeoJsonSatelliteExporter
from humeris.adapters.kml_exporter import KmlExporter
from humeris.adapters.blender_exporter import BlenderExporter
from humeris.adapters.ubox_exporter import UboxExporter
from humeris.adapters.celestia_exporter import CelestiaExporter
from humeris.adapters.spaceengine_exporter import SpaceEngineExporter
from humeris.adapters.ksp_exporter import KspExporter
from humeris.adapters.stellarium_exporter import StellariumExporter


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

SHELL = ShellConfig(
    altitude_km=550,
    inclination_deg=53,
    num_planes=1,
    sats_per_plane=2,
    phase_factor=0,
    raan_offset_deg=0,
    shell_name="Enrich",
)

ENRICHMENT_FIELDS = [
    'altitude_km', 'inclination_deg', 'orbital_period_min',
    'beta_angle_deg', 'atmospheric_density_kg_m3', 'l_shell',
]


@pytest.fixture
def satellites():
    return generate_walker_shell(SHELL)


@pytest.fixture
def tmp_path_file(request):
    """Provide a temp file path with given suffix, cleaned up after test."""
    suffix = getattr(request, 'param', '.tmp')
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestCsvEnrichment:
    """CSV export includes enrichment columns."""

    def test_header_has_enrichment_columns(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        try:
            CsvSatelliteExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
            for field in ENRICHMENT_FIELDS:
                assert field in header, f"Missing column: {field}"
        finally:
            os.unlink(path)

    def test_enrichment_values_are_numeric(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        try:
            CsvSatelliteExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            for field in ENRICHMENT_FIELDS:
                float(row[field])  # must not raise
        finally:
            os.unlink(path)

    def test_altitude_value_plausible(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        try:
            CsvSatelliteExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            alt = float(row['altitude_km'])
            assert 540.0 < alt < 560.0
        finally:
            os.unlink(path)


class TestGeoJsonEnrichment:
    """GeoJSON export includes enrichment properties."""

    def test_properties_have_enrichment_fields(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.geojson')
        os.close(fd)
        try:
            GeoJsonSatelliteExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                data = json.load(f)
            props = data['features'][0]['properties']
            for field in ENRICHMENT_FIELDS:
                assert field in props, f"Missing property: {field}"
        finally:
            os.unlink(path)

    def test_enrichment_values_are_numbers(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.geojson')
        os.close(fd)
        try:
            GeoJsonSatelliteExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                data = json.load(f)
            props = data['features'][0]['properties']
            for field in ENRICHMENT_FIELDS:
                assert isinstance(props[field], (int, float))
        finally:
            os.unlink(path)


class TestKmlEnrichment:
    """KML export includes enrichment in ExtendedData and description."""

    KML_NS = "http://www.opengis.net/kml/2.2"

    def _ns(self, tag):
        return f"{{{self.KML_NS}}}{tag}"

    def test_position_placemark_has_extended_data(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter().export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            doc = root.find(self._ns("Document"))
            folder = doc.findall(self._ns("Folder"))[0]
            pos_pm = folder.findall(self._ns("Placemark"))[0]
            ext = pos_pm.find(self._ns("ExtendedData"))
            assert ext is not None
        finally:
            os.unlink(path)

    def test_extended_data_has_enrichment_fields(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter().export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            doc = root.find(self._ns("Document"))
            folder = doc.findall(self._ns("Folder"))[0]
            pos_pm = folder.findall(self._ns("Placemark"))[0]
            ext = pos_pm.find(self._ns("ExtendedData"))
            data_elements = ext.findall(self._ns("Data"))
            data_names = {d.get("name") for d in data_elements}
            for field in ENRICHMENT_FIELDS:
                assert field in data_names, f"Missing ExtendedData: {field}"
        finally:
            os.unlink(path)

    def test_position_placemark_has_description(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.kml')
        os.close(fd)
        try:
            KmlExporter().export(satellites, path, epoch=EPOCH)
            tree = ET.parse(path)
            root = tree.getroot()
            doc = root.find(self._ns("Document"))
            folder = doc.findall(self._ns("Folder"))[0]
            pos_pm = folder.findall(self._ns("Placemark"))[0]
            desc = pos_pm.find(self._ns("description"))
            assert desc is not None
            assert "altitude" in desc.text.lower() or "Altitude" in desc.text
        finally:
            os.unlink(path)


class TestBlenderEnrichment:
    """Blender script includes custom properties for enrichment."""

    def test_script_has_custom_properties(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.py')
        os.close(fd)
        try:
            BlenderExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            for field in ENRICHMENT_FIELDS:
                assert f'["{field}"]' in content, f"Missing custom prop: {field}"
        finally:
            os.unlink(path)


class TestUboxEnrichment:
    """Universe Sandbox export includes enrichment in Description."""

    def test_satellite_entity_has_description(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.ubox')
        os.close(fd)
        try:
            UboxExporter().export(satellites, path, epoch=EPOCH)
            with zipfile.ZipFile(path) as zf:
                sim = json.loads(zf.read("simulation.json"))
            # First entity is Earth, second is first satellite
            sat_entity = sim["Entities"][1]
            desc = sat_entity.get("Description")
            assert desc is not None
            assert "altitude" in desc.lower() or "Altitude" in desc
        finally:
            os.unlink(path)


class TestCelestiaEnrichment:
    """Celestia export includes enrichment comment block."""

    def test_output_has_enrichment_comments(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.ssc')
        os.close(fd)
        try:
            CelestiaExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "# Altitude" in content or "# altitude" in content
            assert "# Inclination" in content or "# inclination" in content
        finally:
            os.unlink(path)


class TestSpaceEngineEnrichment:
    """SpaceEngine export includes enrichment comment block."""

    def test_output_has_enrichment_comments(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.sc')
        os.close(fd)
        try:
            SpaceEngineExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "// Altitude" in content or "// altitude" in content
            assert "// Inclination" in content or "// inclination" in content
        finally:
            os.unlink(path)


class TestKspEnrichment:
    """KSP export includes enrichment comment block per VESSEL."""

    def test_output_has_enrichment_comments(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.sfs')
        os.close(fd)
        try:
            KspExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "// Altitude" in content or "// altitude" in content
            assert "// Inclination" in content or "// inclination" in content
        finally:
            os.unlink(path)


class TestStellariumEnrichment:
    """Stellarium TLE export includes enrichment comment lines."""

    def test_output_has_enrichment_comments(self, satellites):
        fd, path = tempfile.mkstemp(suffix='.tle')
        os.close(fd)
        try:
            StellariumExporter().export(satellites, path, epoch=EPOCH)
            with open(path) as f:
                content = f.read()
            assert "# Altitude" in content or "# altitude" in content
            assert "Period:" in content or "period:" in content
        finally:
            os.unlink(path)
