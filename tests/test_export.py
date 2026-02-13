# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Tests for CSV and GeoJSON satellite export.

Verifies port compliance, output format, and adapter behavior.
"""
import ast
import csv
import json
import os
import tempfile

import pytest

from constellation_generator.domain.constellation import ShellConfig, generate_walker_shell
from constellation_generator.ports.export import SatelliteExporter
from constellation_generator.adapters.csv_exporter import CsvSatelliteExporter
from constellation_generator.adapters.geojson_exporter import GeoJsonSatelliteExporter


def _make_satellites(count=3):
    """Generate a small set of test satellites."""
    shell = ShellConfig(
        altitude_km=500, inclination_deg=53,
        num_planes=1, sats_per_plane=count,
        phase_factor=0, raan_offset_deg=0,
        shell_name='ExportTest',
    )
    return generate_walker_shell(shell)


class TestPortCompliance:
    """Exporters implement the SatelliteExporter port."""

    def test_csv_exporter_is_satellite_exporter(self):
        assert issubclass(CsvSatelliteExporter, SatelliteExporter)

    def test_geojson_exporter_is_satellite_exporter(self):
        assert issubclass(GeoJsonSatelliteExporter, SatelliteExporter)

    def test_csv_exporter_has_export_method(self):
        exporter = CsvSatelliteExporter()
        assert callable(getattr(exporter, 'export', None))

    def test_geojson_exporter_has_export_method(self):
        exporter = GeoJsonSatelliteExporter()
        assert callable(getattr(exporter, 'export', None))


class TestCsvExporter:
    """CSV export produces correct format."""

    def test_header_row(self):
        sats = _make_satellites(2)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            CsvSatelliteExporter().export(sats, path)
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
            expected = ['name', 'lat_deg', 'lon_deg', 'alt_km', 'epoch',
                        'plane_index', 'sat_index', 'raan_deg', 'true_anomaly_deg']
            assert header == expected
        finally:
            os.unlink(path)

    def test_row_count(self):
        sats = _make_satellites(5)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            count = CsvSatelliteExporter().export(sats, path)
            assert count == 5
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 6  # header + 5 rows
        finally:
            os.unlink(path)

    def test_returns_satellite_count(self):
        sats = _make_satellites(3)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            result = CsvSatelliteExporter().export(sats, path)
            assert result == 3
        finally:
            os.unlink(path)

    def test_coordinate_values_plausible(self):
        sats = _make_satellites(1)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            CsvSatelliteExporter().export(sats, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            lat = float(row['lat_deg'])
            lon = float(row['lon_deg'])
            alt = float(row['alt_km'])
            assert -90 <= lat <= 90
            assert -180 <= lon <= 180
            assert 400 < alt < 600
        finally:
            os.unlink(path)

    def test_empty_satellite_list(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            count = CsvSatelliteExporter().export([], path)
            assert count == 0
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1  # header only
        finally:
            os.unlink(path)

    def test_epoch_column_for_synthetic(self):
        """Synthetic satellites (epoch=None) get the fallback epoch serialized."""
        sats = _make_satellites(1)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            CsvSatelliteExporter().export(sats, path)
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            # Fallback is J2000 epoch when no sat.epoch and no caller epoch
            assert row['epoch'] != ''
            assert '2000-01-01' in row['epoch']
        finally:
            os.unlink(path)

    def test_epoch_column_with_caller_epoch(self):
        """When caller provides epoch, that epoch is serialized for synthetic sats."""
        from datetime import datetime, timezone
        sats = _make_satellites(1)
        caller_epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            path = f.name
        try:
            CsvSatelliteExporter().export(sats, path, epoch=caller_epoch)
            with open(path) as f:
                reader = csv.DictReader(f)
                row = next(reader)
            assert '2026-03-20' in row['epoch']
        finally:
            os.unlink(path)


class TestGeoJsonExporter:
    """GeoJSON export produces correct structure."""

    def test_feature_collection_type(self):
        sats = _make_satellites(2)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            path = f.name
        try:
            GeoJsonSatelliteExporter().export(sats, path)
            with open(path) as f:
                data = json.load(f)
            assert data['type'] == 'FeatureCollection'
        finally:
            os.unlink(path)

    def test_feature_count(self):
        sats = _make_satellites(4)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            path = f.name
        try:
            count = GeoJsonSatelliteExporter().export(sats, path)
            assert count == 4
            with open(path) as f:
                data = json.load(f)
            assert len(data['features']) == 4
        finally:
            os.unlink(path)

    def test_coordinate_order_lon_lat_alt(self):
        """GeoJSON spec requires [longitude, latitude, altitude]."""
        sats = _make_satellites(1)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            path = f.name
        try:
            GeoJsonSatelliteExporter().export(sats, path)
            with open(path) as f:
                data = json.load(f)
            coords = data['features'][0]['geometry']['coordinates']
            assert len(coords) == 3
            lon, lat, alt = coords
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90
            assert 400 < alt < 600
        finally:
            os.unlink(path)

    def test_point_geometry(self):
        sats = _make_satellites(1)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            path = f.name
        try:
            GeoJsonSatelliteExporter().export(sats, path)
            with open(path) as f:
                data = json.load(f)
            assert data['features'][0]['geometry']['type'] == 'Point'
        finally:
            os.unlink(path)

    def test_properties_include_name(self):
        sats = _make_satellites(1)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            path = f.name
        try:
            GeoJsonSatelliteExporter().export(sats, path)
            with open(path) as f:
                data = json.load(f)
            props = data['features'][0]['properties']
            assert 'name' in props
            assert props['name'] == sats[0].name
        finally:
            os.unlink(path)

    def test_properties_include_orbital_metadata(self):
        sats = _make_satellites(1)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            path = f.name
        try:
            GeoJsonSatelliteExporter().export(sats, path)
            with open(path) as f:
                data = json.load(f)
            props = data['features'][0]['properties']
            assert 'plane_index' in props
            assert 'sat_index' in props
            assert 'raan_deg' in props
            assert 'true_anomaly_deg' in props
        finally:
            os.unlink(path)

    def test_empty_satellite_list(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            path = f.name
        try:
            count = GeoJsonSatelliteExporter().export([], path)
            assert count == 0
            with open(path) as f:
                data = json.load(f)
            assert data['features'] == []
        finally:
            os.unlink(path)


class TestExportPurity:
    """Adapter purity: exporters may use stdlib csv/json but no other external deps."""

    def _check_imports(self, module_path, allowed_stdlib):
        with open(module_path) as f:
            tree = ast.parse(f.read())

        allowed_internal = {'constellation_generator'}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split('.')[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split('.')[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import from: {node.module}"

    def test_csv_exporter_no_external_deps(self):
        import constellation_generator.adapters.csv_exporter as mod
        self._check_imports(mod.__file__, {'csv', 'datetime', 'math', 'numpy'})

    def test_geojson_exporter_no_external_deps(self):
        import constellation_generator.adapters.geojson_exporter as mod
        self._check_imports(mod.__file__, {'json', 'datetime', 'math', 'numpy'})
