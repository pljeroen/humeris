"""Tests for CZML exporter adapter."""

import ast
import json
import math
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


class TestConstellationPacketsNumerical:
    """Tests for constellation_packets_numerical function."""

    @pytest.fixture
    def numerical_results(self, epoch, orbital_states):
        from constellation_generator.domain.numerical_propagation import (
            TwoBodyGravity,
            propagate_numerical,
        )
        return [
            propagate_numerical(
                s, timedelta(hours=2), timedelta(seconds=60),
                [TwoBodyGravity()], epoch=epoch,
            )
            for s in orbital_states
        ]

    @pytest.fixture
    def numerical_packets(self, numerical_results):
        from constellation_generator.adapters.czml_exporter import constellation_packets_numerical
        return constellation_packets_numerical(numerical_results)

    def test_document_packet_first(self, numerical_packets):
        doc = numerical_packets[0]
        assert doc["id"] == "document"
        assert doc["version"] == "1.0"

    def test_packet_count_matches_results(self, numerical_results, numerical_packets):
        assert len(numerical_packets) == len(numerical_results) + 1

    def test_position_uses_cartographic_degrees(self, numerical_packets):
        for pkt in numerical_packets[1:]:
            assert "position" in pkt
            assert "cartographicDegrees" in pkt["position"]

    def test_position_values_in_range(self, numerical_packets):
        for pkt in numerical_packets[1:]:
            coords = pkt["position"]["cartographicDegrees"]
            for i in range(0, len(coords), 4):
                lon = coords[i + 1]
                lat = coords[i + 2]
                alt = coords[i + 3]
                assert -180 <= lon <= 180, f"lon out of range: {lon}"
                assert -90 <= lat <= 90, f"lat out of range: {lat}"
                assert alt > 0, f"alt must be positive: {alt}"

    def test_interpolation_lagrange_5(self, numerical_packets):
        for pkt in numerical_packets[1:]:
            pos = pkt["position"]
            assert pos["interpolationAlgorithm"] == "LAGRANGE"
            assert pos["interpolationDegree"] == 5

    def test_satellite_has_point_label_path(self, numerical_packets):
        for pkt in numerical_packets[1:]:
            assert "point" in pkt
            assert "label" in pkt
            assert "path" in pkt

    def test_empty_results(self):
        from constellation_generator.adapters.czml_exporter import constellation_packets_numerical
        result = constellation_packets_numerical([])
        assert len(result) == 1
        assert result[0]["id"] == "document"

    def test_clock_interval_matches_duration(self, numerical_results, numerical_packets):
        doc = numerical_packets[0]
        clock = doc["clock"]
        first_step = numerical_results[0].steps[0]
        last_step = numerical_results[0].steps[-1]
        expected_start = first_step.time.strftime("%Y-%m-%dT%H:%M:%SZ")
        expected_end = last_step.time.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert clock["interval"] == f"{expected_start}/{expected_end}"

    def test_custom_sat_names(self, numerical_results):
        from constellation_generator.adapters.czml_exporter import constellation_packets_numerical
        custom = [f"MySat-{i}" for i in range(len(numerical_results))]
        packets = constellation_packets_numerical(numerical_results, sat_names=custom)
        for idx, pkt in enumerate(packets[1:]):
            assert pkt["name"] == custom[idx]

    def test_default_sat_names(self, numerical_packets):
        for idx, pkt in enumerate(numerical_packets[1:]):
            assert pkt["name"] == f"Sat-{idx}"


class TestInputValidation:
    """Validation of step size, sat_names, and edge case inputs."""

    def test_zero_step_raises(self, orbital_states, epoch):
        """step=timedelta(0) must raise ValueError."""
        with pytest.raises(ValueError, match="step"):
            constellation_packets(orbital_states, epoch, timedelta(hours=2), timedelta(0))

    def test_negative_step_raises(self, orbital_states, epoch):
        """Negative step must raise ValueError."""
        with pytest.raises(ValueError, match="step"):
            constellation_packets(orbital_states, epoch, timedelta(hours=2), timedelta(seconds=-1))

    def test_few_steps_adapts_interpolation(self, epoch, orbital_states):
        """With few data points, interpolation degree must be reduced below 5."""
        # duration=2h, step=1h → 3 data points → max degree 2
        pkts = constellation_packets(
            orbital_states, epoch, timedelta(hours=2), timedelta(hours=1),
        )
        for pkt in pkts[1:]:
            assert pkt["position"]["interpolationDegree"] <= 2

    def test_sat_names_length_mismatch_raises(self, epoch, orbital_states):
        """sat_names shorter than results must raise ValueError."""
        from constellation_generator.adapters.czml_exporter import constellation_packets_numerical
        from constellation_generator.domain.numerical_propagation import (
            TwoBodyGravity,
            propagate_numerical,
        )
        results = [
            propagate_numerical(
                s, timedelta(hours=2), timedelta(seconds=60),
                [TwoBodyGravity()], epoch=epoch,
            )
            for s in orbital_states
        ]
        with pytest.raises(ValueError, match="sat_names"):
            constellation_packets_numerical(results, sat_names=["only-one"])

    def test_empty_steps_raises(self, epoch):
        """NumericalPropagationResult with empty steps must raise ValueError."""
        from constellation_generator.adapters.czml_exporter import constellation_packets_numerical
        from constellation_generator.domain.numerical_propagation import NumericalPropagationResult
        result = NumericalPropagationResult(
            steps=(),
            epoch=epoch,
            duration_s=0.0,
            force_model_names=(),
        )
        with pytest.raises(ValueError, match="steps"):
            constellation_packets_numerical([result])


class TestPlaneColoring:
    """Satellites in different orbital planes get distinct colors."""

    def test_different_planes_have_different_point_colors(self, orbital_states, packets):
        """Two-plane constellation: satellites in plane 0 vs plane 1 have different colors."""
        # Get point colors from first satellite in each plane
        colors_by_plane: dict[int, list[int]] = {}
        for idx, state in enumerate(orbital_states):
            pkt = packets[idx + 1]
            color = pkt["point"]["color"]["rgba"]
            plane = round(math.degrees(state.raan_rad))
            colors_by_plane.setdefault(plane, color)

        unique_colors = set(tuple(c) for c in colors_by_plane.values())
        assert len(unique_colors) > 1, "All planes have same color"

    def test_same_plane_shares_color(self, orbital_states, packets):
        """Satellites within the same plane share the same point color."""
        plane_colors: dict[int, set] = {}
        for idx, state in enumerate(orbital_states):
            pkt = packets[idx + 1]
            color = tuple(pkt["point"]["color"]["rgba"])
            plane = round(math.degrees(state.raan_rad))
            plane_colors.setdefault(plane, set()).add(color)

        for plane, colors in plane_colors.items():
            assert len(colors) == 1, f"Plane {plane} has inconsistent colors: {colors}"

    def test_path_color_matches_point_hue(self, packets):
        """Path color uses same RGB as point but with lower alpha."""
        for pkt in packets[1:]:
            point_rgba = pkt["point"]["color"]["rgba"]
            path_rgba = pkt["path"]["material"]["solidColor"]["color"]["rgba"]
            # Same RGB channels
            assert point_rgba[:3] == path_rgba[:3]
            # Path alpha <= point alpha (more transparent)
            assert path_rgba[3] <= point_rgba[3]


class TestSatelliteDescription:
    """Satellite packets include orbital parameter descriptions."""

    def test_satellite_has_description(self, packets):
        """Each satellite packet has a description field."""
        for pkt in packets[1:]:
            assert "description" in pkt, f"Missing description in {pkt['id']}"

    def test_description_contains_altitude(self, packets):
        """Description includes altitude information."""
        for pkt in packets[1:]:
            desc = pkt["description"]
            assert "km" in desc.lower(), f"No altitude in description: {desc}"

    def test_description_contains_inclination(self, packets):
        """Description includes inclination."""
        for pkt in packets[1:]:
            desc = pkt["description"]
            assert "incl" in desc.lower(), f"No inclination in description: {desc}"


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
