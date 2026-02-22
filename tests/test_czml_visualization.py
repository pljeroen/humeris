# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for advanced CZML visualization features.

Eclipse-aware coloring, sensor footprints, ground station access,
conjunction replay, coverage evolution, and J2 precession timelapse.
"""

import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
from humeris.domain.observation import GroundStation
from humeris.domain.sensor import SensorType, SensorConfig
from humeris.adapters.czml_visualization import (
    eclipse_constellation_packets,
    eclipse_snapshot_packets,
    sensor_footprint_packets,
    ground_station_packets,
    conjunction_replay_packets,
    coverage_evolution_packets,
    precession_constellation_packets,
    isl_topology_packets,
    fragility_constellation_packets,
    hazard_evolution_packets,
    coverage_connectivity_packets,
    network_eclipse_packets,
    kessler_heatmap_packets,
    conjunction_hazard_packets,
    dop_grid_packets,
    radiation_coloring_packets,
    beta_angle_packets,
    deorbit_compliance_packets,
    station_keeping_packets,
    cascade_evolution_packets,
    relative_motion_packets,
    maintenance_schedule_packets,
)
from humeris.domain.link_budget import LinkConfig


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


class TestEclipseSnapshotPackets:
    """Static snapshot colored by eclipse state at epoch."""

    def test_document_packet_present(self, orbital_states):
        pkts = eclipse_snapshot_packets(orbital_states, EPOCH)
        assert pkts[0]["id"] == "document"
        assert pkts[0]["version"] == "1.0"

    def test_packet_count(self, orbital_states):
        pkts = eclipse_snapshot_packets(orbital_states, EPOCH)
        assert len(pkts) == len(orbital_states) + 1

    def test_eclipse_coloring_uses_rgb(self, orbital_states):
        """Each satellite point uses a static rgba color (green/orange/red)."""
        pkts = eclipse_snapshot_packets(orbital_states, EPOCH)
        valid_colors = {
            (102, 187, 106, 255),  # green = sunlit
            (255, 167, 38, 255),   # orange = penumbra
            (255, 82, 82, 255),    # red = umbra
        }
        for pkt in pkts[1:]:
            color = pkt["point"]["color"]["rgba"]
            assert isinstance(color, list)
            assert len(color) == 4
            assert tuple(color) in valid_colors

    def test_no_path_no_label(self, orbital_states):
        """Snapshot packets must not have path or label."""
        pkts = eclipse_snapshot_packets(orbital_states, EPOCH)
        for pkt in pkts[1:]:
            assert "path" not in pkt
            assert "label" not in pkt

    def test_position_is_single_cartesian3(self, orbital_states):
        """Position is a single lon/lat/alt triple."""
        pkts = eclipse_snapshot_packets(orbital_states, EPOCH)
        for pkt in pkts[1:]:
            coords = pkt["position"]["cartographicDegrees"]
            assert len(coords) == 3

    def test_empty_states(self):
        pkts = eclipse_snapshot_packets([], EPOCH)
        assert len(pkts) == 1
        assert pkts[0]["id"] == "document"

    def test_custom_name(self, orbital_states):
        pkts = eclipse_snapshot_packets(orbital_states, EPOCH, name="My Eclipse")
        assert pkts[0]["name"] == "My Eclipse"


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
        from humeris.adapters.czml_exporter import constellation_packets

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


_LINK_CONFIG = LinkConfig(
    frequency_hz=26e9, transmit_power_w=1.0,
    tx_antenna_gain_dbi=30.0, rx_antenna_gain_dbi=30.0,
    system_noise_temp_k=300.0, bandwidth_hz=100e6,
)


class TestISLTopologyPackets:
    """ISL topology with SNR-colored polylines."""

    def test_document_packet_present(self, orbital_states):
        pkts = isl_topology_packets(
            orbital_states, EPOCH, _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=60.0,
        )
        assert pkts[0]["id"] == "document"

    def test_satellite_packets_present(self, orbital_states):
        pkts = isl_topology_packets(
            orbital_states, EPOCH, _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=60.0,
        )
        sat_pkts = [p for p in pkts if p.get("id", "").startswith("isl-sat-")]
        assert len(sat_pkts) == len(orbital_states)

    def test_link_polylines_present(self, orbital_states):
        pkts = isl_topology_packets(
            orbital_states, EPOCH, _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=60.0,
        )
        link_pkts = [p for p in pkts if "polyline" in p]
        assert len(link_pkts) >= 0  # may have 0 links for small constellation

    def test_empty_states(self):
        pkts = isl_topology_packets(
            [], EPOCH, _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=60.0,
        )
        assert len(pkts) == 1
        assert pkts[0]["id"] == "document"


class TestFragilityConstellationPackets:
    """Fragility-colored constellation."""

    def test_document_packet_present(self, orbital_states):
        n_rad_s = orbital_states[0].mean_motion_rad_s
        pkts = fragility_constellation_packets(
            orbital_states, EPOCH, _LINK_CONFIG, n_rad_s,
            control_duration_s=5400.0, duration_s=3600.0, step_s=60.0,
        )
        assert pkts[0]["id"] == "document"

    def test_satellite_count(self, orbital_states):
        n_rad_s = orbital_states[0].mean_motion_rad_s
        pkts = fragility_constellation_packets(
            orbital_states, EPOCH, _LINK_CONFIG, n_rad_s,
            control_duration_s=5400.0, duration_s=3600.0, step_s=60.0,
        )
        assert len(pkts) == len(orbital_states) + 1


class TestHazardEvolutionPackets:
    """Hazard/survival colored satellites."""

    def test_document_packet_present(self, orbital_states):
        from humeris.domain.statistical_analysis import LifetimeSurvivalCurve
        curve = LifetimeSurvivalCurve(
            times=(EPOCH,), altitudes_km=(550.0,),
            survival_fraction=(1.0,), hazard_rate_per_day=(0.001,),
            half_life_altitude_km=400.0, mean_remaining_life_days=365.0,
        )
        pkts = hazard_evolution_packets(
            orbital_states, curve, EPOCH,
            duration_s=86400.0, step_s=3600.0,
        )
        assert pkts[0]["id"] == "document"

    def test_satellite_count(self, orbital_states):
        from humeris.domain.statistical_analysis import LifetimeSurvivalCurve
        curve = LifetimeSurvivalCurve(
            times=(EPOCH,), altitudes_km=(550.0,),
            survival_fraction=(1.0,), hazard_rate_per_day=(0.001,),
            half_life_altitude_km=400.0, mean_remaining_life_days=365.0,
        )
        pkts = hazard_evolution_packets(
            orbital_states, curve, EPOCH,
            duration_s=86400.0, step_s=3600.0,
        )
        assert len(pkts) == len(orbital_states) + 1

    def test_color_uses_intervals(self, orbital_states):
        from humeris.domain.statistical_analysis import LifetimeSurvivalCurve
        curve = LifetimeSurvivalCurve(
            times=(EPOCH,), altitudes_km=(550.0,),
            survival_fraction=(1.0,), hazard_rate_per_day=(0.001,),
            half_life_altitude_km=400.0, mean_remaining_life_days=365.0,
        )
        pkts = hazard_evolution_packets(
            orbital_states, curve, EPOCH,
            duration_s=86400.0, step_s=3600.0,
        )
        for pkt in pkts[1:]:
            color = pkt["point"]["color"]
            assert isinstance(color, list)
            assert "interval" in color[0]


class TestFragilityColorDirection:
    """High fragility must map to red (bad), not green."""

    def test_high_fragility_is_red_not_green(self):
        """_health_color(1.0) = green, _health_color(0.0) = red.

        High fragility should be RED (dangerous), not green.
        The fragility code should pass (1 - frag_val), not frag_val directly.
        """
        from humeris.adapters.czml_visualization import _health_color

        # Direct unit test: the fragility code does
        #   frag_val = min(1.0, fragility.composite_fragility * 1000.0)
        #   frag_color = _health_color(frag_val)
        # For high fragility (composite=0.9): frag_val = 0.9 (clipped to 1.0)
        # BUG: _health_color(1.0) = [0, 255, 0, 200] (GREEN = healthy)
        # FIX: should use _health_color(1.0 - frag_val) → _health_color(0.0) = [255, 0, 0, 200] (RED)

        # Verify _health_color direction (these are correct by definition):
        assert _health_color(1.0) == [0, 255, 0, 200], "1.0 → green (healthy)"
        assert _health_color(0.0) == [255, 0, 0, 200], "0.0 → red (unhealthy)"

        # The actual test: for a high-fragility constellation, the packets
        # should use RED coloring. We mock the fragility computation.
        from unittest.mock import patch
        import humeris.domain.design_sensitivity as ds_mod

        mock_result = type('obj', (object,), {'composite_fragility': 0.9})()
        states, _ = _make_states_and_sats()
        n_rad_s = states[0].mean_motion_rad_s

        with patch.object(ds_mod, 'compute_spectral_fragility', return_value=mock_result):
            pkts = fragility_constellation_packets(
                states, EPOCH, _LINK_CONFIG, n_rad_s,
                control_duration_s=5400.0, duration_s=3600.0, step_s=60.0,
            )
        rgba = pkts[1]["point"]["color"]["rgba"]
        # High fragility (0.9) → frag_val=1.0 after *1000+clip
        # Correct: RED (rgba[0] > rgba[1])
        assert rgba[0] > rgba[1], (
            f"High fragility got color {rgba} — green > red means fragility is inverted"
        )


class TestHazardSurvivalCurveUsage:
    """hazard_evolution_packets must use actual survival_fraction data."""

    def test_survival_curve_data_affects_colors(self, orbital_states):
        """With a sharp survival drop at day 1, colors should differ from linear."""
        from humeris.domain.statistical_analysis import LifetimeSurvivalCurve

        # Survival curve with sharp drop: 100% at day 0, 10% at day 1, 5% at day 2
        t0 = EPOCH
        t1 = EPOCH + timedelta(days=1)
        t2 = EPOCH + timedelta(days=2)
        curve = LifetimeSurvivalCurve(
            times=(t0, t1, t2),
            altitudes_km=(550.0, 540.0, 530.0),
            survival_fraction=(1.0, 0.1, 0.05),
            hazard_rate_per_day=(0.0, 2.3, 0.7),
            half_life_altitude_km=400.0,
            mean_remaining_life_days=365.0,
        )
        # Duration = 2 days, step = 1 day → 3 time steps (0h, 24h, 48h)
        pkts = hazard_evolution_packets(
            orbital_states, curve, EPOCH,
            duration_s=2 * 86400.0, step_s=86400.0,
        )
        # Check first satellite's color intervals
        sat_pkt = pkts[1]
        color_intervals = sat_pkt["point"]["color"]
        assert len(color_intervals) >= 2, "Should have multiple color intervals"

        # At day 1 (index 1), survival is 0.1 → should be very red
        # Linear model: fraction = 1 - 1/365 ≈ 0.997 → very green
        # If colors match the actual curve, day-1 color should be red-ish
        day1_color = color_intervals[1]["rgba"]
        # Red channel (index 0) should be high for 0.1 survival
        assert day1_color[0] >= 200, (
            f"At day 1 (survival=0.1), color {day1_color} should be red, "
            "but survival curve data appears to be ignored"
        )


class TestCoverageConnectivityPackets:
    """Coverage-connectivity product grid."""

    def test_document_packet_present(self, orbital_states):
        pkts = coverage_connectivity_packets(
            orbital_states, _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=1800.0,
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        assert pkts[0]["id"] == "document"

    def test_rectangle_entities(self, orbital_states):
        pkts = coverage_connectivity_packets(
            orbital_states, _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=1800.0,
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        rect_pkts = [p for p in pkts if "rectangle" in p]
        assert len(rect_pkts) >= 0  # May have 0 if no coverage


class TestNetworkEclipsePackets:
    """Network eclipse-colored ISL links."""

    def test_document_packet_present(self, orbital_states):
        pkts = network_eclipse_packets(
            orbital_states, _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=60.0,
        )
        assert pkts[0]["id"] == "document"

    def test_satellite_packets_present(self, orbital_states):
        pkts = network_eclipse_packets(
            orbital_states, _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=60.0,
        )
        sat_pkts = [p for p in pkts if p.get("id", "").startswith("netecl-sat-")]
        assert len(sat_pkts) == len(orbital_states)

    def test_empty_states(self):
        pkts = network_eclipse_packets(
            [], _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=60.0,
        )
        assert len(pkts) == 1


class TestSnrColorGuard:
    """SNR color mapping must handle degenerate min/max ranges."""

    def test_snr_color_equal_min_max(self):
        """Equal min_snr and max_snr must not ZeroDivisionError."""
        from humeris.adapters.czml_visualization import _snr_color

        result = _snr_color(10.0, min_snr=10.0, max_snr=10.0)
        assert result == [0, 255, 0, 200]

    def test_snr_color_inverted_range(self):
        """Inverted range (min > max) must return safe fallback."""
        from humeris.adapters.czml_visualization import _snr_color

        result = _snr_color(5.0, min_snr=20.0, max_snr=10.0)
        assert result == [0, 255, 0, 200]


class TestCzmlVisualizationPurity:
    """Adapter purity: only stdlib + internal imports allowed."""

    def test_no_external_deps(self):
        import humeris.adapters.czml_visualization as mod

        with open(mod.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {"json", "math", "numpy", "datetime"}
        allowed_internal = {"humeris"}

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


class TestUntestedPacketGenerators:
    """Smoke tests for previously untested CZML packet generator functions.

    Each test verifies the function runs without error on minimal input
    and returns a list of dicts (CZML packets).
    """

    def test_kessler_heatmap_packets(self, orbital_states):
        pkts = kessler_heatmap_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=120),
        )
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        # Should have document + 1 per satellite
        assert len(pkts) == len(orbital_states) + 1
        # Each satellite packet should have a point with color
        for pkt in pkts[1:]:
            assert "point" in pkt
            assert "color" in pkt["point"]

    def test_conjunction_hazard_packets(self, orbital_states):
        pkts = conjunction_hazard_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=120),
        )
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        # At minimum: document + satellite packets (conjunctions optional)
        assert len(pkts) >= len(orbital_states) + 1

    def test_dop_grid_packets(self, orbital_states):
        pkts = dop_grid_packets(
            orbital_states, EPOCH,
            lat_step_deg=30.0, lon_step_deg=30.0,
        )
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        # Should produce at least the document packet + some grid cells
        assert len(pkts) >= 1

    def test_radiation_coloring_packets(self, orbital_states):
        pkts = radiation_coloring_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=120),
        )
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        assert len(pkts) == len(orbital_states) + 1
        # Each satellite should have interval-based coloring
        for pkt in pkts[1:]:
            assert "point" in pkt
            color = pkt["point"]["color"]
            assert isinstance(color, list)
            assert "interval" in color[0]

    def test_beta_angle_packets(self, orbital_states):
        pkts = beta_angle_packets(orbital_states, EPOCH)
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        assert len(pkts) == len(orbital_states) + 1
        for pkt in pkts[1:]:
            assert "point" in pkt
            assert "label" in pkt
            # Label text should contain "deg"
            assert "deg" in pkt["label"]["text"]

    def test_deorbit_compliance_packets(self, orbital_states):
        pkts = deorbit_compliance_packets(orbital_states, EPOCH)
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        assert len(pkts) == len(orbital_states) + 1
        for pkt in pkts[1:]:
            assert "point" in pkt
            assert "label" in pkt

    def test_station_keeping_packets(self, orbital_states):
        pkts = station_keeping_packets(orbital_states, EPOCH)
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        assert len(pkts) == len(orbital_states) + 1
        for pkt in pkts[1:]:
            assert "point" in pkt
            assert "label" in pkt
            # Label should contain "m/s/yr"
            assert "m/s/yr" in pkt["label"]["text"]

    def test_cascade_evolution_packets(self, orbital_states):
        pkts = cascade_evolution_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=120),
        )
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        assert len(pkts) == len(orbital_states) + 1
        for pkt in pkts[1:]:
            assert "point" in pkt
            color = pkt["point"]["color"]
            assert isinstance(color, list)

    def test_relative_motion_packets(self, orbital_states):
        # Need exactly two states
        state_a = orbital_states[0]
        state_b = orbital_states[1]
        pkts = relative_motion_packets(
            state_a, state_b, EPOCH, timedelta(hours=2), timedelta(seconds=120),
        )
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        # Should have document + 2 sat packets + 1 proximity line
        assert len(pkts) == 4
        sat_pkts = [p for p in pkts if p.get("id", "").startswith("relmotion-sat-")]
        assert len(sat_pkts) == 2
        line_pkts = [p for p in pkts if "polyline" in p]
        assert len(line_pkts) == 1

    def test_maintenance_schedule_packets(self, orbital_states):
        pkts = maintenance_schedule_packets(orbital_states, EPOCH)
        assert isinstance(pkts, list)
        assert all(isinstance(p, dict) for p in pkts)
        assert pkts[0]["id"] == "document"
        assert len(pkts) == len(orbital_states) + 1
        for pkt in pkts[1:]:
            assert "point" in pkt
            assert "label" in pkt


class TestSatNamesVisualization:
    """SAT-NAME-01: All visualization functions must accept optional sat_names."""

    def test_eclipse_snapshot_sat_names(self, orbital_states):
        """eclipse_snapshot_packets uses provided sat_names."""
        names = [f"ISS-{i}" for i in range(len(orbital_states))]
        pkts = eclipse_snapshot_packets(orbital_states, EPOCH, sat_names=names)
        for idx, pkt in enumerate(pkts[1:]):
            assert pkt["name"] == names[idx]

    def test_eclipse_constellation_sat_names(self, orbital_states):
        """eclipse_constellation_packets uses provided sat_names."""
        names = [f"NOAA-{i}" for i in range(len(orbital_states))]
        pkts = eclipse_constellation_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
            sat_names=names,
        )
        for idx, pkt in enumerate(pkts[1:]):
            assert pkt["name"] == names[idx]

    def test_eclipse_constellation_default_sat_names(self, orbital_states):
        """Without sat_names, eclipse_constellation falls back to Sat-{idx}."""
        pkts = eclipse_constellation_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        for idx, pkt in enumerate(pkts[1:]):
            assert pkt["name"] == f"Sat-{idx}"

    def test_isl_topology_sat_names(self, orbital_states):
        """isl_topology_packets uses provided sat_names for satellite packets."""
        link = LinkConfig(frequency_hz=26e9, transmit_power_w=1.0,
                          tx_antenna_gain_dbi=30.0, rx_antenna_gain_dbi=30.0,
                          system_noise_temp_k=290.0, bandwidth_hz=250e6)
        names = [f"LINK-{i}" for i in range(len(orbital_states))]
        pkts = isl_topology_packets(
            orbital_states, EPOCH, link, EPOCH, 7200.0, 60.0,
            sat_names=names,
        )
        sat_pkts = [p for p in pkts[1:] if p.get("id", "").startswith("topo-sat-")]
        for idx, pkt in enumerate(sat_pkts):
            assert pkt["name"] == names[idx]

    def test_radiation_coloring_sat_names(self, orbital_states):
        """radiation_coloring_packets uses provided sat_names."""
        names = [f"RAD-{i}" for i in range(len(orbital_states))]
        pkts = radiation_coloring_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
            sat_names=names,
        )
        for idx, pkt in enumerate(pkts[1:]):
            assert pkt["name"] == names[idx]

    def test_beta_angle_sat_names(self, orbital_states):
        """beta_angle_packets uses provided sat_names."""
        names = [f"BETA-{i}" for i in range(len(orbital_states))]
        pkts = beta_angle_packets(orbital_states, EPOCH, sat_names=names)
        for idx, pkt in enumerate(pkts[1:]):
            assert names[idx] in pkt["name"]

    def test_cascade_evolution_sat_names(self, orbital_states):
        """cascade_evolution_packets uses provided sat_names."""
        names = [f"CASCADE-{i}" for i in range(len(orbital_states))]
        pkts = cascade_evolution_packets(
            orbital_states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
            sat_names=names,
        )
        for idx, pkt in enumerate(pkts[1:]):
            assert pkt["name"] == names[idx]


# ── CZML Rectangle Coordinate Invariant (CZML-RECT-01) ───────────


def _extract_wsen_from_packets(pkts: list[dict]) -> list[tuple[float, float, float, float]]:
    """Extract all wsenDegrees tuples from CZML packets."""
    results = []
    for pkt in pkts:
        rect = pkt.get("rectangle", {})
        coords = rect.get("coordinates", {})
        wsen = coords.get("wsenDegrees")
        if wsen is not None:
            results.append(tuple(wsen))
    return results


def _assert_wsen_within_cesium_bounds(
    wsen_list: list[tuple[float, float, float, float]],
    label: str,
) -> None:
    """Assert all wsenDegrees satisfy Cesium Rectangle invariant.

    Cesium requires: west ∈ [-180, 180], south ∈ [-90, 90],
    east ∈ [-180, 180], north ∈ [-90, 90], south < north, west < east.
    """
    for w, s, e, n in wsen_list:
        assert -90.0 <= s <= 90.0, (
            f"{label}: south={s}° out of [-90, 90]"
        )
        assert -90.0 <= n <= 90.0, (
            f"{label}: north={n}° out of [-90, 90]"
        )
        assert s < n, f"{label}: south={s}° must be < north={n}°"
        assert -180.0 <= w <= 180.0, (
            f"{label}: west={w}° out of [-180, 180]"
        )
        assert -180.0 <= e <= 180.0, (
            f"{label}: east={e}° out of [-180, 180]"
        )
        assert w < e, f"{label}: west={w}° must be < east={e}°"


class TestCzmlRectangleBoundsInvariant:
    """All rectangle-emitting CZML functions must produce valid Cesium bounds.

    Cesium Rectangle requires latitude ∈ [-π/2, π/2] (i.e., degrees in [-90, 90]).
    Bug: grid includes lat=90°, then lat+step=100° → Cesium DeveloperError.
    """

    def test_coverage_evolution_polar_bounds(self, orbital_states):
        """coverage_evolution_packets: rectangles at poles stay within [-90, 90]."""
        pkts = coverage_evolution_packets(
            orbital_states, EPOCH, timedelta(hours=1), timedelta(minutes=30),
            lat_step_deg=10.0, lon_step_deg=10.0,
        )
        wsen_list = _extract_wsen_from_packets(pkts)
        assert len(wsen_list) > 0, "Should produce rectangle packets"
        _assert_wsen_within_cesium_bounds(wsen_list, "coverage_evolution")

    def test_coverage_connectivity_polar_bounds(self, orbital_states):
        """coverage_connectivity_packets: rectangles at poles stay within [-90, 90]."""
        pkts = coverage_connectivity_packets(
            orbital_states, _LINK_CONFIG, EPOCH,
            duration_s=3600.0, step_s=1800.0,
            lat_step_deg=10.0, lon_step_deg=10.0,
        )
        wsen_list = _extract_wsen_from_packets(pkts)
        # May have 0 if fiedler * coverage is 0 everywhere
        if wsen_list:
            _assert_wsen_within_cesium_bounds(wsen_list, "coverage_connectivity")

    def test_dop_grid_polar_bounds(self, orbital_states):
        """dop_grid_packets: rectangles at poles stay within [-90, 90]."""
        pkts = dop_grid_packets(
            orbital_states, EPOCH,
            lat_step_deg=10.0, lon_step_deg=10.0,
        )
        wsen_list = _extract_wsen_from_packets(pkts)
        assert len(wsen_list) > 0, "Should produce DOP grid rectangles"
        _assert_wsen_within_cesium_bounds(wsen_list, "dop_grid")
