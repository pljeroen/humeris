"""Tests for CZML exporter adapter."""

import ast
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator import (
    ShellConfig,
    generate_walker_shell,
    derive_orbital_state,
    CoveragePoint,
    GroundTrackPoint,
)
from constellation_generator.adapters.czml_exporter import (
    constellation_packets,
    ground_track_packets,
    coverage_packets,
    write_czml,
)


@pytest.fixture
def epoch():
    return datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def orbital_states(epoch):
    shell = ShellConfig(
        altitude_km=550,
        inclination_deg=53,
        num_planes=2,
        sats_per_plane=3,
        phase_factor=1,
        raan_offset_deg=0,
        shell_name="Test",
    )
    sats = generate_walker_shell(shell)
    return [derive_orbital_state(s, epoch) for s in sats]


@pytest.fixture
def packets(orbital_states, epoch):
    return constellation_packets(
        orbital_states,
        epoch,
        timedelta(hours=2),
        timedelta(seconds=60),
    )


@pytest.fixture
def sample_ground_track(epoch):
    return [
        GroundTrackPoint(time=epoch + timedelta(seconds=i * 60), lat_deg=10.0 + i, lon_deg=20.0 + i * 2, alt_km=550.0)
        for i in range(10)
    ]


@pytest.fixture
def sample_coverage():
    points = []
    for lat in range(-90, 91, 10):
        for lon in range(-180, 180, 10):
            count = 3 if -60 <= lat <= 60 else 0
            points.append(CoveragePoint(lat_deg=float(lat), lon_deg=float(lon), visible_count=count))
    return points


class TestConstellationPackets:
    """Tests for constellation_packets function."""

    def test_document_packet_first(self, packets):
        """First packet has id: 'document', version: '1.0'."""
        doc = packets[0]
        assert doc["id"] == "document"
        assert doc["version"] == "1.0"

    def test_packet_count_matches_satellites(self, orbital_states, packets):
        """N orbital states → N+1 packets (document + N satellites)."""
        assert len(packets) == len(orbital_states) + 1

    def test_position_uses_cartographic_degrees(self, packets):
        """Satellite packets have position.cartographicDegrees."""
        for pkt in packets[1:]:
            assert "position" in pkt
            assert "cartographicDegrees" in pkt["position"]

    def test_position_values_in_range(self, packets):
        """lon in [-180,180], lat in [-90,90], alt > 0."""
        for pkt in packets[1:]:
            coords = pkt["position"]["cartographicDegrees"]
            # coords = [secs, lon, lat, height, secs, lon, lat, height, ...]
            for i in range(0, len(coords), 4):
                lon = coords[i + 1]
                lat = coords[i + 2]
                alt = coords[i + 3]
                assert -180 <= lon <= 180, f"lon out of range: {lon}"
                assert -90 <= lat <= 90, f"lat out of range: {lat}"
                assert alt > 0, f"alt must be positive: {alt}"

    def test_interpolation_lagrange_5(self, packets):
        """Algorithm=LAGRANGE, degree=5."""
        for pkt in packets[1:]:
            pos = pkt["position"]
            assert pos["interpolationAlgorithm"] == "LAGRANGE"
            assert pos["interpolationDegree"] == 5

    def test_satellite_has_point_and_label(self, packets):
        """Each satellite packet has point and label."""
        for pkt in packets[1:]:
            assert "point" in pkt
            assert "label" in pkt

    def test_satellite_has_path(self, packets):
        """Each satellite packet has path with leadTime/trailTime."""
        for pkt in packets[1:]:
            assert "path" in pkt
            path = pkt["path"]
            assert "leadTime" in path
            assert "trailTime" in path

    def test_empty_orbital_states(self, epoch):
        """Empty orbital states → document packet only."""
        result = constellation_packets([], epoch, timedelta(hours=2), timedelta(seconds=60))
        assert len(result) == 1
        assert result[0]["id"] == "document"

    def test_clock_interval_matches_duration(self, epoch, packets):
        """Clock interval = epoch to epoch+duration."""
        doc = packets[0]
        clock = doc["clock"]
        expected_start = epoch.strftime("%Y-%m-%dT%H:%M:%SZ")
        expected_end = (epoch + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        interval = clock["interval"]
        assert interval == f"{expected_start}/{expected_end}"


class TestGroundTrackPackets:
    """Tests for ground_track_packets function."""

    def test_ground_track_polyline_present(self, sample_ground_track):
        """Polyline with clampToGround."""
        pkts = ground_track_packets(sample_ground_track)
        polyline_pkt = pkts[1]
        assert "polyline" in polyline_pkt
        assert polyline_pkt["polyline"]["clampToGround"] is True

    def test_ground_track_position_count(self, sample_ground_track):
        """Coordinate triples match track length."""
        pkts = ground_track_packets(sample_ground_track)
        polyline_pkt = pkts[1]
        coords = polyline_pkt["polyline"]["positions"]["cartographicDegrees"]
        # 3 values per point: lon, lat, height
        assert len(coords) == len(sample_ground_track) * 3

    def test_ground_track_empty(self):
        """Empty track → document packet only."""
        pkts = ground_track_packets([])
        assert len(pkts) == 1
        assert pkts[0]["id"] == "document"


class TestCoveragePackets:
    """Tests for coverage_packets function."""

    def test_coverage_rectangle_count(self, sample_coverage):
        """One rectangle per nonzero-count grid point."""
        pkts = coverage_packets(sample_coverage, lat_step_deg=10, lon_step_deg=10)
        nonzero = sum(1 for p in sample_coverage if p.visible_count > 0)
        # packets = document + rectangles
        assert len(pkts) == nonzero + 1

    def test_coverage_wsen_bounds(self, sample_coverage):
        """W < E, S < N for every rectangle."""
        pkts = coverage_packets(sample_coverage, lat_step_deg=10, lon_step_deg=10)
        for pkt in pkts[1:]:
            rect = pkt["rectangle"]
            wsen = rect["coordinates"]["wsenDegrees"]
            w, s, e, n = wsen
            assert w < e, f"W ({w}) must be < E ({e})"
            assert s < n, f"S ({s}) must be < N ({n})"

    def test_coverage_all_zero_no_rectangles(self):
        """All visible_count=0 → document only."""
        points = [
            CoveragePoint(lat_deg=0.0, lon_deg=0.0, visible_count=0),
            CoveragePoint(lat_deg=10.0, lon_deg=10.0, visible_count=0),
        ]
        pkts = coverage_packets(points, lat_step_deg=10, lon_step_deg=10)
        assert len(pkts) == 1
        assert pkts[0]["id"] == "document"


class TestWriteCzml:
    """Tests for write_czml function."""

    def test_write_and_read_roundtrip(self, packets):
        """Write, read, verify valid JSON array."""
        with tempfile.NamedTemporaryFile(suffix=".czml", delete=False, mode="w") as f:
            path = f.name
        try:
            write_czml(packets, path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == len(packets)
            assert data[0]["id"] == "document"
        finally:
            os.unlink(path)

    def test_write_returns_count(self, packets):
        """Return value == len(packets)."""
        with tempfile.NamedTemporaryFile(suffix=".czml", delete=False, mode="w") as f:
            path = f.name
        try:
            result = write_czml(packets, path)
            assert result == len(packets)
        finally:
            os.unlink(path)


class TestCzmlExporterPurity:
    """Adapter purity: czml_exporter may use stdlib json/math/datetime but no other external deps."""

    def test_czml_exporter_no_external_deps(self):
        import constellation_generator.adapters.czml_exporter as mod

        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {"json", "math", "datetime"}
        allowed_internal = {"constellation_generator"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import from: {node.module}"
