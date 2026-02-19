# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""H02-R04: J2000 epoch fallback warning tests.

Verifies that exporters log a warning when defaulting to J2000 epoch.
"""
import logging
import math
import os
import tempfile
from datetime import datetime, timezone

import pytest

from humeris.domain.constellation import Satellite
from humeris.domain.orbital_mechanics import OrbitalConstants, kepler_to_cartesian


def _make_satellite(epoch=None):
    """Create a test satellite, optionally with an epoch."""
    a = OrbitalConstants.R_EARTH + 550_000.0
    pos, vel = kepler_to_cartesian(a, 0.0, math.radians(53.0), 0.0, 0.0, 0.0)
    return Satellite(
        name="TEST-SAT",
        position_eci=pos,
        velocity_eci=vel,
        plane_index=0,
        sat_index=0,
        raan_deg=0.0,
        true_anomaly_deg=0.0,
        epoch=epoch,
    )


class TestCsvEpochWarning:
    """CSV exporter should warn on J2000 fallback."""

    def test_no_warning_with_explicit_epoch(self, tmp_path, caplog):
        from humeris.adapters.csv_exporter import CsvSatelliteExporter
        sat = _make_satellite()
        path = str(tmp_path / "test.csv")
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)

        with caplog.at_level(logging.WARNING, logger="humeris.adapters.csv_exporter"):
            CsvSatelliteExporter().export([sat], path, epoch=epoch)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0, f"No warning expected with explicit epoch, got: {warnings}"

    def test_warning_on_j2000_fallback(self, tmp_path, caplog):
        from humeris.adapters.csv_exporter import CsvSatelliteExporter
        sat = _make_satellite(epoch=None)
        path = str(tmp_path / "test.csv")

        with caplog.at_level(logging.WARNING, logger="humeris.adapters.csv_exporter"):
            CsvSatelliteExporter().export([sat], path, epoch=None)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) > 0, "Expected J2000 fallback warning"
        assert any("J2000" in r.message for r in warnings), (
            f"Warning should mention J2000, got: {[r.message for r in warnings]}"
        )

    def test_no_warning_when_sat_has_epoch(self, tmp_path, caplog):
        from humeris.adapters.csv_exporter import CsvSatelliteExporter
        sat = _make_satellite(epoch=datetime(2026, 6, 1, tzinfo=timezone.utc))
        path = str(tmp_path / "test.csv")

        with caplog.at_level(logging.WARNING, logger="humeris.adapters.csv_exporter"):
            CsvSatelliteExporter().export([sat], path, epoch=None)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0, f"No warning expected when sat.epoch is set, got: {warnings}"


class TestGeoJsonEpochWarning:
    """GeoJSON exporter should warn on J2000 fallback."""

    def test_no_warning_with_explicit_epoch(self, tmp_path, caplog):
        from humeris.adapters.geojson_exporter import GeoJsonSatelliteExporter
        sat = _make_satellite()
        path = str(tmp_path / "test.geojson")
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)

        with caplog.at_level(logging.WARNING, logger="humeris.adapters.geojson_exporter"):
            GeoJsonSatelliteExporter().export([sat], path, epoch=epoch)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0

    def test_warning_on_j2000_fallback(self, tmp_path, caplog):
        from humeris.adapters.geojson_exporter import GeoJsonSatelliteExporter
        sat = _make_satellite(epoch=None)
        path = str(tmp_path / "test.geojson")

        with caplog.at_level(logging.WARNING, logger="humeris.adapters.geojson_exporter"):
            GeoJsonSatelliteExporter().export([sat], path, epoch=None)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) > 0, "Expected J2000 fallback warning"
        assert any("J2000" in r.message for r in warnings)


class TestCelestiaEpochWarning:
    """Celestia exporter should warn on J2000 fallback."""

    def test_no_warning_with_explicit_epoch(self, tmp_path, caplog):
        from humeris.adapters.celestia_exporter import CelestiaExporter
        sat = _make_satellite()
        path = str(tmp_path / "test.ssc")
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)

        with caplog.at_level(logging.WARNING, logger="humeris.adapters.celestia_exporter"):
            CelestiaExporter().export([sat], path, epoch=epoch)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0

    def test_warning_on_j2000_fallback(self, tmp_path, caplog):
        from humeris.adapters.celestia_exporter import CelestiaExporter
        sat = _make_satellite(epoch=None)
        path = str(tmp_path / "test.ssc")

        with caplog.at_level(logging.WARNING, logger="humeris.adapters.celestia_exporter"):
            CelestiaExporter().export([sat], path, epoch=None)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) > 0, "Expected J2000 fallback warning"
        assert any("J2000" in r.message for r in warnings)


class TestKmlEpochWarning:
    """KML exporter should warn on J2000 fallback."""

    def test_no_warning_with_explicit_epoch(self, tmp_path, caplog):
        from humeris.adapters.kml_exporter import KmlExporter
        sat = _make_satellite()
        path = str(tmp_path / "test.kml")
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)

        with caplog.at_level(logging.WARNING, logger="humeris.adapters.kml_exporter"):
            KmlExporter().export([sat], path, epoch=epoch)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0

    def test_warning_on_j2000_fallback(self, tmp_path, caplog):
        from humeris.adapters.kml_exporter import KmlExporter
        sat = _make_satellite(epoch=None)
        path = str(tmp_path / "test.kml")

        with caplog.at_level(logging.WARNING, logger="humeris.adapters.kml_exporter"):
            KmlExporter().export([sat], path, epoch=None)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) > 0, "Expected J2000 fallback warning"
        assert any("J2000" in r.message for r in warnings)
