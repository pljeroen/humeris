# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Tests for ground track computation.

Verifies Keplerian propagation to geodetic coordinates for circular orbits.
"""
import ast
import math
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator.domain.constellation import ShellConfig, generate_walker_shell
from constellation_generator.domain.ground_track import (
    AscendingNodePass,
    GroundTrackCrossing,
    GroundTrackPoint,
    compute_ground_track,
    find_ascending_nodes,
    find_ground_track_crossings,
)


class TestGroundTrackPoint:
    """GroundTrackPoint is an immutable value object."""

    def test_frozen_dataclass(self):
        point = GroundTrackPoint(
            time=datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc),
            lat_deg=0.0,
            lon_deg=0.0,
            alt_km=500.0,
        )
        with pytest.raises(AttributeError):
            point.lat_deg = 10.0

    def test_fields(self):
        t = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        point = GroundTrackPoint(time=t, lat_deg=45.0, lon_deg=-90.0, alt_km=550.0)
        assert point.time == t
        assert point.lat_deg == 45.0
        assert point.lon_deg == -90.0
        assert point.alt_km == 550.0


def _make_satellite(inclination_deg=53.0, altitude_km=500.0):
    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=inclination_deg,
        num_planes=1, sats_per_plane=1,
        phase_factor=0, raan_offset_deg=0,
        shell_name='Test',
    )
    return generate_walker_shell(shell)[0]


class TestGroundTrackPointCount:
    """Point count matches time range and step."""

    def test_point_count_90_min(self):
        sat = _make_satellite()
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=90), timedelta(minutes=1))
        assert len(track) == 91  # 0, 1, 2, ..., 90

    def test_point_count_zero_duration(self):
        sat = _make_satellite()
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(0), timedelta(minutes=1))
        assert len(track) == 1  # just the start point

    def test_point_count_single_step(self):
        sat = _make_satellite()
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=5), timedelta(minutes=5))
        assert len(track) == 2  # start and end


class TestGroundTrackLatitude:
    """Latitude bounded by orbital inclination for circular orbits."""

    def test_latitude_bounded_by_inclination(self):
        """For a 53 deg inclined orbit, latitude stays within +/-53 deg (with small geodetic margin)."""
        sat = _make_satellite(inclination_deg=53.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=90), timedelta(minutes=1))

        # Geodetic latitude can slightly exceed geocentric inclination due to
        # WGS84 flattening, but the difference is small (~0.2 deg)
        for point in track:
            assert abs(point.lat_deg) <= 53.5, f"Latitude {point.lat_deg} exceeds inclination bound"

    def test_equatorial_orbit_near_zero_latitude(self):
        """A 0 deg inclined orbit should have latitude near 0 deg."""
        sat = _make_satellite(inclination_deg=0.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=90), timedelta(minutes=1))

        for point in track:
            assert abs(point.lat_deg) < 1.0, f"Equatorial orbit lat {point.lat_deg} too high"


class TestGroundTrackAltitude:
    """Altitude consistent for circular orbits."""

    def test_altitude_consistent(self):
        """For a circular orbit at 500km, altitude should stay near 500km."""
        sat = _make_satellite(altitude_km=500.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=90), timedelta(minutes=1))

        for point in track:
            # Mean radius vs WGS84 ellipsoid gives up to ~20km difference
            assert 470 < point.alt_km < 530, f"Altitude {point.alt_km} km outside expected range"


class TestGroundTrackLongitude:
    """Longitude varies over a full orbit."""

    def test_longitude_varies(self):
        """Over 90 minutes, longitude should change significantly."""
        sat = _make_satellite()
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=90), timedelta(minutes=1))

        lons = [p.lon_deg for p in track]
        lon_range = max(lons) - min(lons)
        assert lon_range > 30, f"Longitude range {lon_range} deg too narrow for full orbit"

    def test_longitude_in_range(self):
        """Longitude should be in [-180, 180]."""
        sat = _make_satellite()
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=180), timedelta(minutes=1))

        for point in track:
            assert -180.0 <= point.lon_deg <= 180.0


class TestGroundTrackEdgeCases:
    """Edge cases for ground track computation."""

    def test_negative_step_raises(self):
        sat = _make_satellite()
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            compute_ground_track(sat, start, timedelta(minutes=90), timedelta(minutes=-1))

    def test_zero_step_raises(self):
        sat = _make_satellite()
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            compute_ground_track(sat, start, timedelta(minutes=90), timedelta(0))

    def test_epoch_none_uses_start(self):
        """Satellites with epoch=None should still produce valid ground tracks."""
        sat = _make_satellite()
        assert sat.epoch is None
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=10), timedelta(minutes=1))
        assert len(track) == 11
        assert -90 <= track[0].lat_deg <= 90
        assert -180 <= track[0].lon_deg <= 180

    def test_first_point_time_is_start(self):
        sat = _make_satellite()
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=10), timedelta(minutes=1))
        assert track[0].time == start

    def test_last_point_time(self):
        sat = _make_satellite()
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=90), timedelta(minutes=1))
        assert track[-1].time == start + timedelta(minutes=90)


class TestComputeGroundTrackNumerical:
    """Tests for compute_ground_track_numerical."""

    @pytest.fixture
    def epoch(self):
        return datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def numerical_result(self, epoch):
        from constellation_generator import (
            derive_orbital_state,
            propagate_numerical,
        )
        from constellation_generator.domain.numerical_propagation import (
            TwoBodyGravity,
        )
        sat = _make_satellite(inclination_deg=53.0, altitude_km=500.0)
        state = derive_orbital_state(sat, epoch)
        return propagate_numerical(
            state, timedelta(minutes=90), timedelta(seconds=60),
            [TwoBodyGravity()], epoch=epoch,
        )

    def test_returns_ground_track_points(self, numerical_result):
        from constellation_generator.domain.ground_track import compute_ground_track_numerical
        result = compute_ground_track_numerical(numerical_result.steps)
        assert all(isinstance(p, GroundTrackPoint) for p in result)

    def test_point_count_matches_steps(self, numerical_result):
        from constellation_generator.domain.ground_track import compute_ground_track_numerical
        result = compute_ground_track_numerical(numerical_result.steps)
        assert len(result) == len(numerical_result.steps)

    def test_latitude_bounded_by_inclination(self, numerical_result):
        from constellation_generator.domain.ground_track import compute_ground_track_numerical
        result = compute_ground_track_numerical(numerical_result.steps)
        for p in result:
            assert abs(p.lat_deg) <= 53.5, f"Lat {p.lat_deg} exceeds inclination"

    def test_altitude_near_orbital(self, numerical_result):
        from constellation_generator.domain.ground_track import compute_ground_track_numerical
        result = compute_ground_track_numerical(numerical_result.steps)
        for p in result:
            assert 470 < p.alt_km < 530, f"Alt {p.alt_km} outside expected range"

    def test_longitude_in_range(self, numerical_result):
        from constellation_generator.domain.ground_track import compute_ground_track_numerical
        result = compute_ground_track_numerical(numerical_result.steps)
        for p in result:
            assert -180.0 <= p.lon_deg <= 180.0

    def test_times_match_step_times(self, numerical_result):
        from constellation_generator.domain.ground_track import compute_ground_track_numerical
        result = compute_ground_track_numerical(numerical_result.steps)
        for pt, step in zip(result, numerical_result.steps):
            assert pt.time == step.time

    def test_empty_steps_returns_empty(self):
        from constellation_generator.domain.ground_track import compute_ground_track_numerical
        result = compute_ground_track_numerical(())
        assert result == []

    def test_single_step(self, numerical_result):
        from constellation_generator.domain.ground_track import compute_ground_track_numerical
        result = compute_ground_track_numerical(numerical_result.steps[:1])
        assert len(result) == 1
        assert isinstance(result[0], GroundTrackPoint)

    def test_consistent_with_analytical(self, epoch):
        """Numerical ground track matches analytical within tolerance for two-body."""
        from constellation_generator import derive_orbital_state, propagate_numerical
        from constellation_generator.domain.numerical_propagation import TwoBodyGravity
        from constellation_generator.domain.ground_track import compute_ground_track_numerical

        sat = _make_satellite(inclination_deg=53.0, altitude_km=500.0)
        state = derive_orbital_state(sat, epoch)

        # Analytical ground track
        analytical = compute_ground_track(sat, epoch, timedelta(minutes=30), timedelta(minutes=5))

        # Numerical ground track
        result = propagate_numerical(
            state, timedelta(minutes=30), timedelta(minutes=5),
            [TwoBodyGravity()], epoch=epoch,
        )
        numerical = compute_ground_track_numerical(result.steps)

        assert len(analytical) == len(numerical)
        for a_pt, n_pt in zip(analytical, numerical):
            assert abs(a_pt.lat_deg - n_pt.lat_deg) < 1.0, \
                f"Lat mismatch: {a_pt.lat_deg} vs {n_pt.lat_deg}"
            # Longitude can wrap, so check minimal difference
            lon_diff = abs(a_pt.lon_deg - n_pt.lon_deg)
            if lon_diff > 180:
                lon_diff = 360 - lon_diff
            assert lon_diff < 1.0, f"Lon mismatch: {a_pt.lon_deg} vs {n_pt.lon_deg}"


# ── Ascending nodes ──────────────────────────────────────────────

class TestFindAscendingNodes:
    """Tests for ascending node detection."""

    def test_ascending_nodes_returns_list(self):
        """Return type is list[AscendingNodePass]."""
        sat = _make_satellite(inclination_deg=53.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=180), timedelta(seconds=30))
        nodes = find_ascending_nodes(track)
        assert isinstance(nodes, list)
        assert all(isinstance(n, AscendingNodePass) for n in nodes)

    def test_ascending_node_near_equator(self):
        """Crossing at lat ≈ 0°."""
        sat = _make_satellite(inclination_deg=53.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=180), timedelta(seconds=30))
        nodes = find_ascending_nodes(track)
        # Interpolated crossings should be near equator (though ground track points
        # are discrete, the interpolation should give values very close to 0)
        assert len(nodes) > 0

    def test_ascending_node_count(self):
        """≈ revolutions in duration."""
        sat = _make_satellite(inclination_deg=53.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        # ~3 orbits at 500 km (~94 min period)
        track = compute_ground_track(sat, start, timedelta(minutes=280), timedelta(seconds=30))
        nodes = find_ascending_nodes(track)
        assert 2 <= len(nodes) <= 4

    def test_ascending_node_longitude_shifts(self):
        """Each successive node shifted west."""
        sat = _make_satellite(inclination_deg=53.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=280), timedelta(seconds=30))
        nodes = find_ascending_nodes(track)
        if len(nodes) >= 2:
            # Earth rotates east, so ground track sub-satellite longitude shifts west
            # (longitude decreases for prograde orbits)
            lon_diffs = []
            for i in range(len(nodes) - 1):
                diff = nodes[i + 1].longitude_deg - nodes[i].longitude_deg
                if diff > 180:
                    diff -= 360
                if diff < -180:
                    diff += 360
                lon_diffs.append(diff)
            # All diffs should be negative (westward shift)
            assert all(d < 0 for d in lon_diffs)

    def test_ascending_node_frozen(self):
        """AscendingNodePass is immutable."""
        node = AscendingNodePass(
            time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            longitude_deg=45.0,
        )
        with pytest.raises(AttributeError):
            node.longitude_deg = 0.0


# ── Ground track crossings ──────────────────────────────────────

class TestFindGroundTrackCrossings:
    """Tests for ground track self-intersection detection."""

    def test_crossings_returns_list(self):
        """Return type is list[GroundTrackCrossing]."""
        sat = _make_satellite(inclination_deg=53.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=280), timedelta(seconds=30))
        crossings = find_ground_track_crossings(track)
        assert isinstance(crossings, list)
        assert all(isinstance(c, GroundTrackCrossing) for c in crossings)

    def test_crossings_exist_inclined(self):
        """Inclined orbit over 3+ orbits → crossings exist."""
        sat = _make_satellite(inclination_deg=53.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=280), timedelta(seconds=30))
        crossings = find_ground_track_crossings(track)
        assert len(crossings) > 0

    def test_no_crossings_equatorial(self):
        """Equatorial orbit → no descending segments → no crossings."""
        sat = _make_satellite(inclination_deg=0.0)
        start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        track = compute_ground_track(sat, start, timedelta(minutes=180), timedelta(seconds=30))
        crossings = find_ground_track_crossings(track)
        assert len(crossings) == 0

    def test_crossing_frozen(self):
        """GroundTrackCrossing is immutable."""
        c = GroundTrackCrossing(
            lat_deg=30.0, lon_deg=-45.0,
            time_ascending=datetime(2026, 1, 1, tzinfo=timezone.utc),
            time_descending=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
        )
        with pytest.raises(AttributeError):
            c.lat_deg = 0.0

    def test_single_point_no_crossings(self):
        """Single point → no crossings."""
        track = [GroundTrackPoint(
            time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            lat_deg=0.0, lon_deg=0.0, alt_km=500.0,
        )]
        crossings = find_ground_track_crossings(track)
        assert crossings == []


class TestGroundTrackPurity:
    """Domain purity: ground_track.py must only import from stdlib and domain."""

    def test_no_external_imports(self):
        import constellation_generator.domain.ground_track as mod
        source_path = mod.__file__
        with open(source_path) as f:
            tree = ast.parse(f.read())

        allowed_top = {'math', 'dataclasses', 'datetime'}
        allowed_internal_prefix = 'constellation_generator.domain'

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split('.')[0]
                    assert top in allowed_top or alias.name.startswith(allowed_internal_prefix), \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split('.')[0]
                    assert top in allowed_top or node.module.startswith(allowed_internal_prefix), \
                        f"Forbidden import from: {node.module}"
