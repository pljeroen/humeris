"""Tests for advanced CZML visualization features.

Eclipse-aware coloring, sensor footprints, ground station access,
conjunction replay, coverage evolution, and J2 precession timelapse.
"""

import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator import (
    ShellConfig,
    generate_walker_shell,
    derive_orbital_state,
    GroundStation,
    SensorType,
    SensorConfig,
)
from constellation_generator.adapters.czml_visualization import (
    eclipse_constellation_packets,
    sensor_footprint_packets,
    ground_station_packets,
    conjunction_replay_packets,
    coverage_evolution_packets,
    precession_constellation_packets,
)


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_states_and_sats(n_sats=4):
    shell = ShellConfig(
        altitude_km=550, inclination_deg=53,
        num_planes=2, sats_per_plane=n_sats // 2,
        phase_factor=1, raan_offset_deg=0, shell_name="Test",
    )
    sats = generate_walker_shell(shell)
    states = [derive_orbital_state(s, EPOCH) for s in sats]
    return states, sats


@pytest.fixture
def states_and_sats():
    return _make_states_and_sats()


@pytest.fixture
def orbital_states(states_and_sats):
    return states_and_sats[0]


@pytest.fixture
def satellites(states_and_sats):
    return states_and_sats[1]


class TestEclipseConstellationPackets:
    """Eclipse-aware satellite coloring in CZML."""

    def test_document_packet_present(self, orbital_states):
        pkts = eclipse_constellation_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        assert pkts[0]["id"] == "document"
        assert pkts[0]["version"] == "1.0"

    def test_packet_count(self, orbital_states):
        pkts = eclipse_constellation_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        assert len(pkts) == len(orbital_states) + 1

    def test_point_color_uses_intervals(self, orbital_states):
        pkts = eclipse_constellation_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        for pkt in pkts[1:]:
            color = pkt["point"]["color"]
            assert isinstance(color, list), "Point color must be interval list"
            assert len(color) > 0
            assert "interval" in color[0]
            assert "rgba" in color[0]

    def test_has_position_and_path(self, orbital_states):
        pkts = eclipse_constellation_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        for pkt in pkts[1:]:
            assert "position" in pkt
            assert "path" in pkt
            assert "label" in pkt

    def test_empty_states(self):
        pkts = eclipse_constellation_packets(
            [], EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        assert len(pkts) == 1
        assert pkts[0]["id"] == "document"

    def test_zero_step_raises(self, orbital_states):
        """step=timedelta(0) must raise ValueError."""
        with pytest.raises(ValueError, match="step"):
            eclipse_constellation_packets(
                orbital_states, EPOCH, timedelta(hours=2), timedelta(0),
            )


class TestSensorFootprintPackets:
    """Sensor FOV footprints sweeping the ground."""

    @pytest.fixture
    def sensor(self):
        return SensorConfig(SensorType.CIRCULAR, half_angle_deg=30.0)

    def test_document_packet_present(self, orbital_states, sensor):
        pkts = sensor_footprint_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60), sensor,
        )
        assert pkts[0]["id"] == "document"

    def test_footprint_entity_per_satellite(self, orbital_states, sensor):
        pkts = sensor_footprint_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60), sensor,
        )
        assert len(pkts) == len(orbital_states) + 1

    def test_ellipse_present(self, orbital_states, sensor):
        pkts = sensor_footprint_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60), sensor,
        )
        for pkt in pkts[1:]:
            assert "ellipse" in pkt
            assert "semiMajorAxis" in pkt["ellipse"]
            assert "semiMinorAxis" in pkt["ellipse"]

    def test_position_ground_level(self, orbital_states, sensor):
        """Footprint positions should be at ground level."""
        pkts = sensor_footprint_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60), sensor,
        )
        for pkt in pkts[1:]:
            coords = pkt["position"]["cartographicDegrees"]
            # First position altitude (index 3)
            alt = coords[3]
            assert alt == 0.0, f"Footprint altitude should be 0, got {alt}"

    def test_empty_states(self, sensor):
        pkts = sensor_footprint_packets(
            [], EPOCH, timedelta(hours=2), timedelta(seconds=60), sensor,
        )
        assert len(pkts) == 1

    def test_zero_step_raises(self, orbital_states, sensor):
        """step=timedelta(0) must raise ValueError."""
        with pytest.raises(ValueError, match="step"):
            sensor_footprint_packets(
                orbital_states, EPOCH, timedelta(hours=2), timedelta(0), sensor,
            )


class TestGroundStationPackets:
    """Ground station with visibility circle and access lines."""

    @pytest.fixture
    def station(self):
        return GroundStation(name="TestStation", lat_deg=52.0, lon_deg=4.0, alt_m=0.0)

    def test_document_packet_present(self, station, orbital_states):
        pkts = ground_station_packets(
            station, orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        assert pkts[0]["id"] == "document"

    def test_station_marker_present(self, station, orbital_states):
        pkts = ground_station_packets(
            station, orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        station_pkt = pkts[1]
        assert "point" in station_pkt
        assert "label" in station_pkt

    def test_station_position_correct(self, station, orbital_states):
        pkts = ground_station_packets(
            station, orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        station_pkt = pkts[1]
        pos = station_pkt["position"]["cartographicDegrees"]
        assert pos[0] == station.lon_deg
        assert pos[1] == station.lat_deg

    def test_visibility_circle_present(self, station, orbital_states):
        pkts = ground_station_packets(
            station, orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        circle_pkts = [p for p in pkts if "ellipse" in p]
        assert len(circle_pkts) >= 1, "Should have visibility circle entity"
        circle = circle_pkts[0]["ellipse"]
        assert "semiMajorAxis" in circle
        assert circle["semiMajorAxis"] > 0

    def test_station_ids_unique_per_name(self, orbital_states):
        """Different stations must produce different CZML entity IDs."""
        station_a = GroundStation(name="Alpha", lat_deg=52.0, lon_deg=4.0, alt_m=0.0)
        station_b = GroundStation(name="Beta", lat_deg=40.0, lon_deg=-74.0, alt_m=0.0)
        pkts_a = ground_station_packets(
            station_a, orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        pkts_b = ground_station_packets(
            station_b, orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        id_a = pkts_a[1]["id"]
        id_b = pkts_b[1]["id"]
        assert id_a != id_b, "Station IDs must differ for different station names"


class TestConjunctionReplayPackets:
    """Conjunction event 3D replay."""

    @pytest.fixture
    def close_states(self):
        """Two satellites in close proximity."""
        shell = ShellConfig(
            altitude_km=550, inclination_deg=53,
            num_planes=1, sats_per_plane=2,
            phase_factor=1, raan_offset_deg=0, shell_name="Close",
        )
        sats = generate_walker_shell(shell)
        return [derive_orbital_state(s, EPOCH) for s in sats]

    def test_document_packet_present(self, close_states):
        pkts = conjunction_replay_packets(
            close_states[0], close_states[1], EPOCH,
            timedelta(minutes=30), timedelta(seconds=10),
        )
        assert pkts[0]["id"] == "document"

    def test_two_satellite_entities(self, close_states):
        pkts = conjunction_replay_packets(
            close_states[0], close_states[1], EPOCH,
            timedelta(minutes=30), timedelta(seconds=10),
        )
        sat_pkts = [p for p in pkts if p.get("id", "").startswith("conjunction-sat")]
        assert len(sat_pkts) == 2

    def test_proximity_line_present(self, close_states):
        pkts = conjunction_replay_packets(
            close_states[0], close_states[1], EPOCH,
            timedelta(minutes=30), timedelta(seconds=10),
        )
        line_pkts = [p for p in pkts if "polyline" in p]
        assert len(line_pkts) >= 1, "Should have proximity line"

    def test_clock_centered_on_event(self, close_states):
        pkts = conjunction_replay_packets(
            close_states[0], close_states[1], EPOCH,
            timedelta(minutes=30), timedelta(seconds=10),
        )
        doc = pkts[0]
        clock = doc["clock"]
        interval = clock["interval"]
        assert "2026-03-20" in interval

    def test_zero_step_raises(self, close_states):
        """step=timedelta(0) must raise ValueError."""
        with pytest.raises(ValueError, match="step"):
            conjunction_replay_packets(
                close_states[0], close_states[1], EPOCH,
                timedelta(minutes=30), timedelta(0),
            )


class TestCoverageEvolutionPackets:
    """Time-varying coverage heatmap."""

    def test_document_packet_present(self, orbital_states):
        pkts = coverage_evolution_packets(
            orbital_states, EPOCH, timedelta(hours=1), timedelta(minutes=30),
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        assert pkts[0]["id"] == "document"

    def test_rectangle_entities_present(self, orbital_states):
        pkts = coverage_evolution_packets(
            orbital_states, EPOCH, timedelta(hours=1), timedelta(minutes=30),
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        rect_pkts = [p for p in pkts if "rectangle" in p]
        assert len(rect_pkts) > 0, "Should have coverage rectangles"

    def test_color_uses_intervals(self, orbital_states):
        pkts = coverage_evolution_packets(
            orbital_states, EPOCH, timedelta(hours=1), timedelta(minutes=30),
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        rect_pkts = [p for p in pkts if "rectangle" in p]
        assert len(rect_pkts) > 0
        color = rect_pkts[0]["rectangle"]["material"]["solidColor"]["color"]
        assert isinstance(color, list), "Coverage color must be interval list"
        assert len(color) > 0
        assert "interval" in color[0]

    def test_grid_coverage(self, orbital_states):
        """Grid rectangles should exist for coverage cells."""
        pkts = coverage_evolution_packets(
            orbital_states, EPOCH, timedelta(hours=1), timedelta(minutes=30),
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        rect_pkts = [p for p in pkts if "rectangle" in p]
        assert len(rect_pkts) >= 1

    def test_zero_step_raises(self, orbital_states):
        """coverage_step=timedelta(0) must raise ValueError."""
        with pytest.raises(ValueError, match="step"):
            coverage_evolution_packets(
                orbital_states, EPOCH, timedelta(hours=1), timedelta(0),
            )


class TestPrecessionConstellationPackets:
    """J2 RAAN precession timelapse."""

    def test_document_packet_present(self, satellites):
        pkts = precession_constellation_packets(
            satellites, EPOCH, timedelta(days=7), timedelta(hours=1),
        )
        assert pkts[0]["id"] == "document"

    def test_packet_count_matches(self, satellites):
        pkts = precession_constellation_packets(
            satellites, EPOCH, timedelta(days=7), timedelta(hours=1),
        )
        assert len(pkts) == len(satellites) + 1

    def test_includes_j2_effect(self, satellites):
        """J2-propagated positions differ from non-J2 over 7 days."""
        from constellation_generator.adapters.czml_exporter import constellation_packets

        j2_pkts = precession_constellation_packets(
            satellites, EPOCH, timedelta(days=7), timedelta(hours=1),
        )
        states_no_j2 = [derive_orbital_state(s, EPOCH, include_j2=False) for s in satellites]
        no_j2_pkts = constellation_packets(
            states_no_j2, EPOCH, timedelta(days=7), timedelta(hours=1),
        )
        j2_coords = j2_pkts[1]["position"]["cartographicDegrees"]
        no_j2_coords = no_j2_pkts[1]["position"]["cartographicDegrees"]
        j2_last_lon = j2_coords[-3]
        no_j2_last_lon = no_j2_coords[-3]
        assert j2_last_lon != pytest.approx(no_j2_last_lon, abs=0.1), \
            "J2 should produce measurably different longitudes over 7 days"


class TestCzmlVisualizationPurity:
    """Adapter purity: only stdlib + internal imports allowed."""

    def test_no_external_deps(self):
        import constellation_generator.adapters.czml_visualization as mod

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
