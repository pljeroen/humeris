# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Invariant tests for exporters.

These verify determinism, epoch fidelity, and coordinate ordering
properties that must hold for all export adapters.

Invariants G1-G3 from the formal invariant specification.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
from humeris.domain.ground_track import GroundTrackPoint
from humeris.domain.coverage import CoveragePoint
from humeris.adapters.czml_exporter import (
    constellation_packets,
    ground_track_packets,
    coverage_packets,
    write_czml,
)


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_states(n_sats=4):
    shell = ShellConfig(
        altitude_km=550, inclination_deg=53,
        num_planes=2, sats_per_plane=n_sats // 2,
        phase_factor=1, raan_offset_deg=0, shell_name="Test",
    )
    sats = generate_walker_shell(shell)
    return [derive_orbital_state(s, EPOCH) for s in sats]


class TestG1EpochFidelity:
    """G1: Exported epoch matches computed epoch."""

    def test_czml_epoch_matches(self):
        states = _make_states()
        packets = constellation_packets(
            states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
        )
        doc = packets[0]
        expected_start = EPOCH.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert expected_start in doc["clock"]["interval"]

        # Each satellite packet epoch must match
        for pkt in packets[1:]:
            assert pkt["position"]["epoch"] == expected_start

    def test_ground_track_epoch_not_blank(self):
        """Ground track packets must have document id, not blank."""
        track = [
            GroundTrackPoint(
                time=EPOCH + timedelta(seconds=i * 60),
                lat_deg=10.0 + i, lon_deg=20.0 + i, alt_km=550.0,
            )
            for i in range(5)
        ]
        packets = ground_track_packets(track)
        assert packets[0]["id"] == "document"
        assert packets[0]["name"] == "Ground Track"


class TestG2Determinism:
    """G2: Same inputs produce identical outputs (byte-stable)."""

    def test_czml_deterministic(self):
        states = _make_states()
        pkts1 = constellation_packets(
            states, EPOCH, timedelta(hours=1), timedelta(seconds=60),
        )
        pkts2 = constellation_packets(
            states, EPOCH, timedelta(hours=1), timedelta(seconds=60),
        )
        assert len(pkts1) == len(pkts2)
        for p1, p2 in zip(pkts1, pkts2):
            assert p1 == p2, "Packets differ between identical runs"

    def test_czml_file_deterministic(self):
        states = _make_states()
        pkts = constellation_packets(
            states, EPOCH, timedelta(hours=1), timedelta(seconds=60),
        )
        paths = []
        for _ in range(2):
            with tempfile.NamedTemporaryFile(suffix=".czml", delete=False, mode="w") as f:
                path = f.name
            paths.append(path)
            write_czml(pkts, path)

        try:
            with open(paths[0], encoding="utf-8") as f1, open(paths[1], encoding="utf-8") as f2:
                assert f1.read() == f2.read(), "File outputs differ"
        finally:
            for p in paths:
                os.unlink(p)

    def test_coverage_packets_deterministic(self):
        points = [
            CoveragePoint(lat_deg=0.0, lon_deg=0.0, visible_count=3),
            CoveragePoint(lat_deg=10.0, lon_deg=10.0, visible_count=5),
            CoveragePoint(lat_deg=20.0, lon_deg=20.0, visible_count=0),
        ]
        pkts1 = coverage_packets(points, lat_step_deg=10, lon_step_deg=10)
        pkts2 = coverage_packets(points, lat_step_deg=10, lon_step_deg=10)
        assert pkts1 == pkts2


class TestG3CoordinateOrdering:
    """G3: Coordinate ordering in CZML matches spec: [seconds, lon, lat, height]."""

    def test_cartographic_degrees_ordering(self):
        states = _make_states(2)
        pkts = constellation_packets(
            states, EPOCH, timedelta(minutes=10), timedelta(minutes=5),
        )
        for pkt in pkts[1:]:
            coords = pkt["position"]["cartographicDegrees"]
            # Verify structure: [sec, lon, lat, alt, sec, lon, lat, alt, ...]
            assert len(coords) % 4 == 0, f"Coords not multiple of 4: {len(coords)}"
            for i in range(0, len(coords), 4):
                t_offset = coords[i]
                lon = coords[i + 1]
                lat = coords[i + 2]
                alt = coords[i + 3]
                # t_offset should be non-negative seconds
                assert t_offset >= 0, f"Negative time offset: {t_offset}"
                # lon/lat/alt in valid ranges
                assert -180 <= lon <= 180
                assert -90 <= lat <= 90
                assert alt > 0

    def test_ground_track_cartographic_ordering(self):
        """Ground track coords: [lon, lat, height, lon, lat, height, ...]."""
        track = [
            GroundTrackPoint(
                time=EPOCH + timedelta(seconds=i * 60),
                lat_deg=10.0 + i, lon_deg=20.0 + i * 2, alt_km=550.0,
            )
            for i in range(3)
        ]
        pkts = ground_track_packets(track)
        coords = pkts[1]["polyline"]["positions"]["cartographicDegrees"]
        assert len(coords) == 3 * 3, f"Expected 9 coords, got {len(coords)}"
        # Verify ordering matches input
        for i, pt in enumerate(track):
            assert coords[i * 3] == pt.lon_deg
            assert coords[i * 3 + 1] == pt.lat_deg
            assert coords[i * 3 + 2] == 0.0  # clamped to ground
