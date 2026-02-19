# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for interactive viewer server.

HTTP server serving Cesium viewer with on-demand CZML generation,
dynamic constellation management, and analysis layer control.
"""

import ast
import json
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from http.client import HTTPResponse
from unittest.mock import patch

import pytest

from humeris.domain.constellation import (
    ShellConfig,
    generate_walker_shell,
)
from humeris.domain.propagation import derive_orbital_state


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_states(n_planes=2, n_sats=2, altitude_km=550):
    """Create a small set of orbital states for testing."""
    shell = ShellConfig(
        altitude_km=altitude_km, inclination_deg=53,
        num_planes=n_planes, sats_per_plane=n_sats,
        phase_factor=1, raan_offset_deg=0, shell_name="Test",
    )
    sats = generate_walker_shell(shell)
    return [derive_orbital_state(s, EPOCH) for s in sats]


# ---------------------------------------------------------------------------
# Layer state management
# ---------------------------------------------------------------------------


class TestLayerState:
    """Tests for LayerState dataclass."""

    def test_layer_state_creation(self):
        from humeris.adapters.viewer_server import LayerState
        layer = LayerState(
            layer_id="walker-1",
            name="Constellation:Walker-550",
            category="Constellation",
            layer_type="walker",
            mode="animated",
            visible=True,
            states=_make_states(),
            params={"altitude_km": 550},
            czml=[{"id": "document", "version": "1.0"}],
        )
        assert layer.layer_id == "walker-1"
        assert layer.name == "Constellation:Walker-550"
        assert layer.category == "Constellation"
        assert layer.layer_type == "walker"
        assert layer.mode == "animated"
        assert layer.visible is True
        assert len(layer.states) == 4
        assert layer.params == {"altitude_km": 550}
        assert len(layer.czml) == 1

    def test_layer_state_defaults(self):
        from humeris.adapters.viewer_server import LayerState
        layer = LayerState(
            layer_id="test",
            name="Test",
            category="Test",
            layer_type="walker",
            mode="snapshot",
            visible=True,
            states=[],
            params={},
            czml=[],
        )
        assert layer.czml == []
        assert layer.states == []


class TestLayerManager:
    """Tests for layer management functions."""

    def test_add_layer(self):
        from humeris.adapters.viewer_server import (
            LayerManager,
            LayerState,
        )
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Constellation:Walker",
            category="Constellation",
            layer_type="walker",
            states=states,
            params={"altitude_km": 550},
        )
        assert layer_id is not None
        assert layer_id in mgr.layers

    def test_add_layer_auto_mode_animated_small(self):
        """<=100 sats default to animated mode."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states(n_planes=2, n_sats=2)  # 4 sats
        layer_id = mgr.add_layer(
            name="Constellation:Small",
            category="Constellation",
            layer_type="walker",
            states=states,
            params={},
        )
        assert mgr.layers[layer_id].mode == "animated"

    def test_add_layer_auto_mode_snapshot_large(self):
        """>100 sats default to snapshot mode."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states(n_planes=6, n_sats=20)  # 120 sats
        layer_id = mgr.add_layer(
            name="Constellation:Large",
            category="Constellation",
            layer_type="walker",
            states=states,
            params={},
        )
        assert mgr.layers[layer_id].mode == "snapshot"

    def test_add_layer_explicit_mode(self):
        """Explicit mode overrides auto-detection."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Test",
            category="Constellation",
            layer_type="walker",
            states=states,
            params={},
            mode="snapshot",
        )
        assert mgr.layers[layer_id].mode == "snapshot"

    def test_add_layer_generates_czml(self):
        """Adding a layer generates CZML packets."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Constellation:Walker",
            category="Constellation",
            layer_type="walker",
            states=states,
            params={},
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 0
        assert czml[0]["id"] == "document"

    def test_remove_layer(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Test", category="Test", layer_type="walker",
            states=states, params={},
        )
        assert layer_id in mgr.layers
        mgr.remove_layer(layer_id)
        assert layer_id not in mgr.layers

    def test_remove_nonexistent_layer_raises(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        with pytest.raises(KeyError):
            mgr.remove_layer("nonexistent")

    def test_update_layer_visibility(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Test", category="Test", layer_type="walker",
            states=states, params={},
        )
        mgr.update_layer(layer_id, visible=False)
        assert mgr.layers[layer_id].visible is False

    def test_update_layer_mode_regenerates_czml(self):
        """Switching mode regenerates CZML."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Constellation:Test",
            category="Constellation",
            layer_type="walker",
            states=states,
            params={},
            mode="animated",
        )
        old_czml = mgr.layers[layer_id].czml
        mgr.update_layer(layer_id, mode="snapshot")
        new_czml = mgr.layers[layer_id].czml
        assert mgr.layers[layer_id].mode == "snapshot"
        # CZML should be different after mode change
        assert new_czml != old_czml

    def test_update_nonexistent_layer_raises(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        with pytest.raises(KeyError):
            mgr.update_layer("nonexistent", visible=False)

    def test_get_state_returns_all_layers_metadata(self):
        """get_state() returns layer metadata without CZML data."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        mgr.add_layer(
            name="Constellation:A", category="Constellation",
            layer_type="walker", states=states, params={},
        )
        mgr.add_layer(
            name="Analysis:Eclipse", category="Analysis",
            layer_type="eclipse", states=states, params={},
        )
        state = mgr.get_state()
        assert len(state["layers"]) == 2
        for layer_info in state["layers"]:
            assert "layer_id" in layer_info
            assert "name" in layer_info
            assert "category" in layer_info
            assert "mode" in layer_info
            assert "visible" in layer_info
            assert "num_entities" in layer_info
            # No CZML data in state response
            assert "czml" not in layer_info

    def test_get_czml_for_layer(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Test", category="Test", layer_type="walker",
            states=states, params={},
        )
        czml = mgr.get_czml(layer_id)
        assert isinstance(czml, list)
        assert len(czml) > 0
        assert czml[0]["id"] == "document"

    def test_get_czml_nonexistent_raises(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        with pytest.raises(KeyError):
            mgr.get_czml("nonexistent")

    def test_unique_layer_ids(self):
        """Each add_layer call produces a unique ID."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        ids = set()
        for i in range(5):
            layer_id = mgr.add_layer(
                name=f"Test:{i}", category="Test", layer_type="walker",
                states=states, params={},
            )
            ids.add(layer_id)
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# CZML generation dispatch
# ---------------------------------------------------------------------------


class TestCzmlDispatch:
    """Tests for CZML generation dispatch based on layer type and mode."""

    def test_walker_snapshot_dispatch(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Constellation:Walker",
            category="Constellation",
            layer_type="walker",
            states=states,
            params={},
            mode="snapshot",
        )
        czml = mgr.layers[layer_id].czml
        # Snapshot packets: document + 1 per sat, no path/interpolation
        assert czml[0]["id"] == "document"
        # Snapshot points use point.pixelSize, not path
        sat_packet = czml[1]
        assert "point" in sat_packet
        assert "position" in sat_packet

    def test_walker_animated_dispatch(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Constellation:Walker",
            category="Constellation",
            layer_type="walker",
            states=states,
            params={},
            mode="animated",
        )
        czml = mgr.layers[layer_id].czml
        assert czml[0]["id"] == "document"
        # Animated packets have time-varying position (cartographicDegrees array)
        sat_packet = czml[1]
        assert "position" in sat_packet

    def test_eclipse_snapshot_dispatch(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Eclipse",
            category="Analysis",
            layer_type="eclipse",
            states=states,
            params={},
            mode="snapshot",
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 0
        assert czml[0]["id"] == "document"

    def test_eclipse_animated_dispatch(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Eclipse",
            category="Analysis",
            layer_type="eclipse",
            states=states,
            params={},
            mode="animated",
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 0
        assert czml[0]["id"] == "document"

    def test_coverage_dispatch(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Coverage",
            category="Analysis",
            layer_type="coverage",
            states=states,
            params={"lat_step_deg": 30.0, "lon_step_deg": 30.0},
            mode="snapshot",
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 0
        assert czml[0]["id"] == "document"

    def test_ground_track_dispatch(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:GroundTrack",
            category="Analysis",
            layer_type="ground_track",
            states=states,
            params={},
            mode="snapshot",
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 0
        assert czml[0]["id"] == "document"

    def test_ground_station_layer(self):
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_ground_station(
            name="Svalbard",
            lat_deg=78.23,
            lon_deg=15.39,
            source_states=states,
        )
        assert layer_id in mgr.layers
        layer = mgr.layers[layer_id]
        assert layer.category == "Ground Station"
        assert len(layer.czml) > 0

    def test_sensor_dispatch(self):
        """Sensor footprint layer generates CZML with entity packets."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Sensor", category="Analysis",
            layer_type="sensor", states=states, params={},
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 1
        assert czml[0]["id"] == "document"

    def test_isl_dispatch(self):
        """ISL topology layer generates CZML with entity packets."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:ISL", category="Analysis",
            layer_type="isl", states=states, params={},
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 1
        assert czml[0]["id"] == "document"

    def test_fragility_dispatch(self):
        """Fragility layer generates CZML with entity packets."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Fragility", category="Analysis",
            layer_type="fragility", states=states, params={},
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 1
        assert czml[0]["id"] == "document"

    def test_hazard_dispatch(self):
        """Hazard evolution layer generates CZML with entity packets."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Hazard", category="Analysis",
            layer_type="hazard", states=states, params={},
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 1
        assert czml[0]["id"] == "document"

    def test_network_eclipse_dispatch(self):
        """Network eclipse layer generates CZML with entity packets."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Network Eclipse", category="Analysis",
            layer_type="network_eclipse", states=states, params={},
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 1
        assert czml[0]["id"] == "document"

    def test_coverage_connectivity_dispatch(self):
        """Coverage connectivity layer generates CZML (doc + possible entities)."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Coverage Connectivity", category="Analysis",
            layer_type="coverage_connectivity", states=states, params={},
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) >= 1
        assert czml[0]["id"] == "document"

    def test_precession_dispatch(self):
        """Precession layer generates CZML with entity packets."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Precession", category="Analysis",
            layer_type="precession", states=states, params={},
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 1
        assert czml[0]["id"] == "document"

    def test_conjunction_dispatch(self):
        """Conjunction replay layer generates CZML with entity packets."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Conjunction", category="Analysis",
            layer_type="conjunction", states=states, params={},
        )
        czml = mgr.layers[layer_id].czml
        assert len(czml) > 1
        assert czml[0]["id"] == "document"


# ---------------------------------------------------------------------------
# HTTP server + API
# ---------------------------------------------------------------------------


def _start_server(port):
    """Start viewer server on given port, return (server, thread)."""
    from humeris.adapters.viewer_server import (
        create_viewer_server,
        LayerManager,
    )
    mgr = LayerManager(epoch=EPOCH)
    server = create_viewer_server(mgr, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Wait for server to be ready
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/api/state", timeout=1)
            break
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(0.05)
    return server, mgr


def _api_request(port, method, path, body=None):
    """Make HTTP request to server, return (status, parsed_json_or_text)."""
    url = f"http://localhost:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        content = resp.read().decode()
        try:
            return resp.status, json.loads(content)
        except json.JSONDecodeError:
            return resp.status, content
    except urllib.error.HTTPError as e:
        content = e.read().decode()
        try:
            return e.code, json.loads(content)
        except json.JSONDecodeError:
            return e.code, content


@pytest.fixture
def server_port():
    """Find a free port for testing."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def running_server(server_port):
    """Start server, yield (port, manager), shutdown after test."""
    server, mgr = _start_server(server_port)
    yield server_port, mgr
    server.shutdown()


class TestHttpApi:
    """Tests for HTTP API endpoints."""

    def test_get_root_returns_html(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "GET", "/")
        assert status == 200
        assert "<!DOCTYPE html>" in body

    def test_get_state_empty(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "GET", "/api/state")
        assert status == 200
        assert body["layers"] == []

    def test_post_constellation_walker(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550,
                "inclination_deg": 53,
                "num_planes": 2,
                "sats_per_plane": 2,
                "phase_factor": 1,
                "raan_offset_deg": 0,
                "shell_name": "Test-Walker",
            },
        })
        assert status == 201
        assert "layer_id" in body

        # Verify it appears in state
        status, state = _api_request(port, "GET", "/api/state")
        assert len(state["layers"]) == 1
        assert state["layers"][0]["category"] == "Constellation"

    def test_post_analysis_eclipse(self, running_server):
        port, mgr = running_server
        # First add a constellation as source
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Source",
            },
        })
        source_id = body["layer_id"]

        # Add eclipse analysis
        status, body = _api_request(port, "POST", "/api/analysis", {
            "type": "eclipse",
            "source_layer": source_id,
            "params": {},
        })
        assert status == 201
        assert "layer_id" in body

    def test_post_ground_station(self, running_server):
        port, mgr = running_server
        # Need a constellation for ground station access computation
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Src",
            },
        })
        status, body = _api_request(port, "POST", "/api/ground-station", {
            "name": "Svalbard",
            "lat": 78.23,
            "lon": 15.39,
        })
        assert status == 201
        assert "layer_id" in body

    def test_get_czml_for_layer(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Test",
            },
        })
        layer_id = body["layer_id"]
        status, czml = _api_request(port, "GET", f"/api/czml/{layer_id}")
        assert status == 200
        assert isinstance(czml, list)
        assert czml[0]["id"] == "document"

    def test_put_layer_update_mode(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Test",
            },
        })
        layer_id = body["layer_id"]
        status, body = _api_request(port, "PUT", f"/api/layer/{layer_id}", {
            "mode": "snapshot",
        })
        assert status == 200
        assert body["mode"] == "snapshot"

    def test_put_layer_update_visibility(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Test",
            },
        })
        layer_id = body["layer_id"]
        status, body = _api_request(port, "PUT", f"/api/layer/{layer_id}", {
            "visible": False,
        })
        assert status == 200
        assert body["visible"] is False

    def test_delete_layer(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Test",
            },
        })
        layer_id = body["layer_id"]
        status, body = _api_request(port, "DELETE", f"/api/layer/{layer_id}")
        assert status == 200

        # Verify removed
        status, state = _api_request(port, "GET", "/api/state")
        assert len(state["layers"]) == 0

    def test_get_czml_nonexistent_returns_404(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "GET", "/api/czml/nonexistent")
        assert status == 404

    def test_delete_nonexistent_returns_404(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "DELETE", "/api/layer/nonexistent")
        assert status == 404

    def test_post_constellation_invalid_type_returns_400(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "invalid_type",
            "params": {},
        })
        assert status == 400

    def test_post_analysis_missing_source_returns_400(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/analysis", {
            "type": "eclipse",
            "source_layer": "nonexistent",
            "params": {},
        })
        assert status == 404

    def test_cors_headers_present(self, running_server):
        """CORS headers allow local browser access."""
        port, mgr = running_server
        url = f"http://localhost:{port}/api/state"
        resp = urllib.request.urlopen(url, timeout=5)
        assert resp.headers.get("Access-Control-Allow-Origin") == f"http://localhost:{port}"

    def test_state_includes_epoch(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "GET", "/api/state")
        assert "epoch" in body


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


class TestThreadedServer:
    """Verify the server uses ThreadingMixIn for concurrent requests."""

    def test_server_uses_threading_mixin(self):
        """Server should use ThreadingMixIn so analysis doesn't block UI."""
        from humeris.adapters.viewer_server import create_viewer_server, LayerManager
        import socketserver
        mgr = LayerManager(epoch=EPOCH)
        server = create_viewer_server(mgr, port=0)
        assert isinstance(server, socketserver.ThreadingMixIn), \
            "Server should use ThreadingMixIn"
        server.server_close()


class TestCapMetadata:
    """Verify cap metadata is surfaced when satellite count is capped."""

    def test_state_includes_capped_from_for_isl(self):
        """When ISL caps satellite count, state response shows original count."""
        from humeris.adapters.viewer_server import LayerManager, _MAX_TOPOLOGY_SATS
        mgr = LayerManager(epoch=EPOCH)
        # Create enough states to trigger capping
        states = _make_states(n_planes=6, n_sats=20)  # 120 > _MAX_TOPOLOGY_SATS
        assert len(states) > _MAX_TOPOLOGY_SATS

        layer_id = mgr.add_layer(
            name="Analysis:ISL", category="Analysis",
            layer_type="isl", states=states, params={},
        )
        state = mgr.get_state()
        layer_info = [l for l in state["layers"] if l["layer_id"] == layer_id][0]
        assert "capped_from" in layer_info, \
            "Layer state should include capped_from when satellite count is capped"
        assert layer_info["capped_from"] == len(states)

    def test_state_includes_capped_from_for_precession(self):
        """When precession caps satellite count, state response shows original count."""
        from humeris.adapters.viewer_server import LayerManager, _MAX_PRECESSION_SATS
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states(n_planes=6, n_sats=20)  # 120 > _MAX_PRECESSION_SATS
        assert len(states) > _MAX_PRECESSION_SATS

        layer_id = mgr.add_layer(
            name="Analysis:Precession", category="Analysis",
            layer_type="precession", states=states, params={},
        )
        state = mgr.get_state()
        layer_info = [l for l in state["layers"] if l["layer_id"] == layer_id][0]
        assert "capped_from" in layer_info
        assert layer_info["capped_from"] == len(states)

    def test_no_capped_from_when_under_limit(self):
        """No capped_from when satellite count is under the cap."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states(n_planes=2, n_sats=2)  # 4 sats
        layer_id = mgr.add_layer(
            name="Analysis:ISL", category="Analysis",
            layer_type="isl", states=states, params={},
        )
        state = mgr.get_state()
        layer_info = [l for l in state["layers"] if l["layer_id"] == layer_id][0]
        assert "capped_from" not in layer_info


class TestErrorForwarding:
    """Verify analysis errors include actual error message."""

    def test_analysis_error_includes_details(self, running_server):
        """Analysis failure should include the actual error message, not generic text."""
        port, mgr = running_server
        # Add a source constellation first
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Err-Test",
            },
        })
        source_id = body["layer_id"]

        # Patch _generate_czml to raise an error
        with patch(
            "humeris.adapters.viewer_server._generate_czml",
            side_effect=ValueError("test error detail"),
        ):
            status, body = _api_request(port, "POST", "/api/analysis", {
                "type": "eclipse",
                "source_layer": source_id,
                "params": {},
            })
        assert status == 500
        assert "test error detail" in body.get("error", ""), \
            f"Error response should contain actual error: {body}"


class TestAnalysisParamsPassthrough:
    """Verify analysis params from request are passed to _generate_czml."""

    def test_coverage_params_passed(self):
        """Coverage analysis params (lat_step, lon_step, min_elev) should pass through."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        # First add a constellation as source
        source_id = mgr.add_layer(
            name="Constellation:Source", category="Constellation",
            layer_type="walker", states=states, params={},
        )
        # Now add coverage with explicit params
        layer_id = mgr.add_layer(
            name="Analysis:Coverage", category="Analysis",
            layer_type="coverage", states=states,
            params={"lat_step_deg": 20.0, "lon_step_deg": 20.0, "min_elevation_deg": 15.0},
        )
        layer = mgr.layers[layer_id]
        # The params should have been stored and used
        assert layer.params.get("lat_step_deg") == 20.0
        assert layer.params.get("lon_step_deg") == 20.0
        assert layer.params.get("min_elevation_deg") == 15.0

    def test_isl_max_range_passed(self):
        """ISL analysis max_range_km param should pass through."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:ISL", category="Analysis",
            layer_type="isl", states=states,
            params={"max_range_km": 3000.0},
        )
        layer = mgr.layers[layer_id]
        assert layer.params.get("max_range_km") == 3000.0

    def test_analysis_api_forwards_params(self, running_server):
        """POST /api/analysis should forward params to the layer."""
        port, mgr = running_server
        # Add source
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "ParamTest",
            },
        })
        source_id = body["layer_id"]

        # Add coverage with params
        status, body = _api_request(port, "POST", "/api/analysis", {
            "type": "coverage",
            "source_layer": source_id,
            "params": {"lat_step_deg": 20.0, "lon_step_deg": 20.0},
        })
        assert status == 201
        layer_id = body["layer_id"]

        # Check state includes the params
        status, state = _api_request(port, "GET", "/api/state")
        layer_info = [l for l in state["layers"] if l["layer_id"] == layer_id][0]
        assert layer_info["params"].get("lat_step_deg") == 20.0


class TestColorLegendData:
    """Verify color legend metadata is available in state response."""

    def test_state_includes_legend_for_eclipse(self):
        """Eclipse analysis layers should include legend color mapping."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Eclipse", category="Analysis",
            layer_type="eclipse", states=states, params={},
        )
        state = mgr.get_state()
        layer_info = [l for l in state["layers"] if l["layer_id"] == layer_id][0]
        assert "legend" in layer_info, \
            "Eclipse layer should include legend data in state response"

    def test_legend_has_entries(self):
        """Legend should have label+color entries."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        layer_id = mgr.add_layer(
            name="Analysis:Eclipse", category="Analysis",
            layer_type="eclipse", states=states, params={},
        )
        state = mgr.get_state()
        layer_info = [l for l in state["layers"] if l["layer_id"] == layer_id][0]
        legend = layer_info["legend"]
        assert len(legend) > 0
        assert "label" in legend[0]
        assert "color" in legend[0]


class TestExportEndpoint:
    """Verify CZML export endpoint."""

    def test_export_czml_returns_downloadable(self, running_server):
        """GET /api/export/{layer_id} should return CZML as downloadable JSON."""
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Export-Test",
            },
        })
        layer_id = body["layer_id"]
        status, czml = _api_request(port, "GET", f"/api/export/{layer_id}")
        assert status == 200
        assert isinstance(czml, list)
        assert czml[0]["id"] == "document"

    def test_export_nonexistent_returns_404(self, running_server):
        """GET /api/export/nonexistent should return 404."""
        port, mgr = running_server
        status, body = _api_request(port, "GET", "/api/export/nonexistent")
        assert status == 404


class TestGroundStationSatLimit:
    """Verify ground station satellite limit is configurable."""

    def test_ground_station_uses_more_than_six_sats(self, running_server):
        """Ground station should use configurable sat limit, not hardcoded 6."""
        port, mgr = running_server
        # Add a large constellation
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 4, "sats_per_plane": 5,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "GS-Test",
            },
        })
        source_id = body["layer_id"]
        source_layer = mgr.layers[source_id]
        assert len(source_layer.states) == 20

        # Add ground station — should use more than 6 sats
        status, body = _api_request(port, "POST", "/api/ground-station", {
            "name": "Test", "lat": 0.0, "lon": 0.0,
        })
        assert status == 201
        gs_layer = mgr.layers[body["layer_id"]]
        # The ground station should have access to more than 6 source states
        assert len(gs_layer.states) > 6, \
            f"Ground station only got {len(gs_layer.states)} sats, should be >6"


class TestSessionSaveLoad:
    """Verify session save/load endpoints."""

    def test_save_session_returns_data(self, running_server):
        """POST /api/session/save should return serializable session data."""
        port, mgr = running_server
        # Add a constellation
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Save-Test",
            },
        })
        status, body = _api_request(port, "POST", "/api/session/save")
        assert status == 200
        assert "session" in body
        assert "layers" in body["session"]

    def test_load_session_restores_layers(self, running_server):
        """POST /api/session/load should restore previously saved session."""
        port, mgr = running_server
        # Add a constellation
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Load-Test",
            },
        })
        # Save
        _, save_resp = _api_request(port, "POST", "/api/session/save")
        session_data = save_resp["session"]

        # Clear all layers
        _, state = _api_request(port, "GET", "/api/state")
        for layer in state["layers"]:
            _api_request(port, "DELETE", f"/api/layer/{layer['layer_id']}")

        # Verify empty
        _, state = _api_request(port, "GET", "/api/state")
        assert len(state["layers"]) == 0

        # Load
        status, body = _api_request(port, "POST", "/api/session/load", {
            "session": session_data,
        })
        assert status == 200

        # Verify restored
        _, state = _api_request(port, "GET", "/api/state")
        assert len(state["layers"]) >= 1


class TestSessionLoadClears:
    """BUG-007: Session load must clear existing layers before loading."""

    def test_load_session_replaces_existing_layers(self, running_server):
        """Loading a session should clear old layers, not append."""
        port, mgr = running_server
        # Add two constellations
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Existing-1",
            },
        })
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 800, "inclination_deg": 97,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Existing-2",
            },
        })
        _, state = _api_request(port, "GET", "/api/state")
        assert len(state["layers"]) == 2

        # Load a session with just one layer
        session_data = {
            "epoch": EPOCH.isoformat(),
            "duration_s": 7200,
            "step_s": 60,
            "layers": [{
                "name": "Constellation:New",
                "category": "Constellation",
                "layer_type": "walker",
                "mode": "animated",
                "visible": True,
                "params": {
                    "altitude_km": 600, "inclination_deg": 45,
                    "num_planes": 2, "sats_per_plane": 2,
                    "phase_factor": 1, "raan_offset_deg": 0,
                    "shell_name": "New",
                },
            }],
        }
        status, body = _api_request(port, "POST", "/api/session/load", {
            "session": session_data,
        })
        assert status == 200

        # Should have exactly 1 layer, not 3
        _, state = _api_request(port, "GET", "/api/state")
        assert len(state["layers"]) == 1, \
            f"Expected 1 layer after load, got {len(state['layers'])}"


class TestAnalysisRecompute:
    """Verify analysis recomputation with updated params."""

    def test_put_analysis_updates_params_and_regenerates(self, running_server):
        """PUT /api/analysis/{layer_id} should update params and regenerate CZML."""
        port, mgr = running_server
        # Add source
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Recomp-Test",
            },
        })
        source_id = body["layer_id"]

        # Add coverage analysis
        status, body = _api_request(port, "POST", "/api/analysis", {
            "type": "coverage",
            "source_layer": source_id,
            "params": {"lat_step_deg": 10.0, "lon_step_deg": 10.0},
        })
        analysis_id = body["layer_id"]

        # Get original CZML
        _, original_czml = _api_request(port, "GET", f"/api/czml/{analysis_id}")

        # Recompute with different params
        status, body = _api_request(port, "PUT", f"/api/analysis/{analysis_id}", {
            "params": {"lat_step_deg": 30.0, "lon_step_deg": 30.0},
        })
        assert status == 200

        # CZML should be different
        _, new_czml = _api_request(port, "GET", f"/api/czml/{analysis_id}")
        assert new_czml != original_czml, "CZML should change after param update"


class TestDurationStepSettings:
    """Verify global simulation duration/step settings endpoint."""

    def test_state_includes_duration_step(self, running_server):
        """GET /api/state should include current duration and step settings."""
        port, mgr = running_server
        status, state = _api_request(port, "GET", "/api/state")
        assert "duration_s" in state, "State should include duration_s"
        assert "step_s" in state, "State should include step_s"

    def test_put_settings_updates_duration(self, running_server):
        """PUT /api/settings should update duration and step."""
        port, mgr = running_server
        status, body = _api_request(port, "PUT", "/api/settings", {
            "duration_s": 14400,  # 4 hours
            "step_s": 120,  # 2 minutes
        })
        assert status == 200

        # Verify state reflects new settings
        _, state = _api_request(port, "GET", "/api/state")
        assert state["duration_s"] == 14400
        assert state["step_s"] == 120


class TestInputValidation:
    """BUG-004/005/012/016/027: Server-side input validation."""

    # BUG-004: Content-Length validation
    def test_non_numeric_content_length_returns_400(self, running_server):
        """Non-numeric Content-Length should return 400, not crash."""
        port, _ = running_server
        import http.client
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        conn.putrequest("POST", "/api/constellation")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", "notanumber")
        conn.endheaders(b'{}')
        resp = conn.getresponse()
        assert resp.status == 400

    # BUG-005: URL param sanitization
    def test_path_traversal_in_layer_id_rejected(self, running_server):
        """Layer ID with path traversal should be rejected."""
        port, _ = running_server
        status, body = _api_request(port, "GET", "/api/czml/../../etc/passwd")
        assert status in (400, 404)

    def test_newline_in_layer_id_rejected(self, running_server):
        """Layer ID with newlines should be rejected."""
        port, _ = running_server
        import http.client
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        conn.request("GET", "/api/czml/layer-1%0d%0aInjected:header")
        resp = conn.getresponse()
        assert resp.status in (400, 404)

    # BUG-012: Walker params validation
    def test_walker_negative_altitude_rejected(self, running_server):
        """Negative altitude should be rejected."""
        port, _ = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": -100,
                "inclination_deg": 53,
                "num_planes": 2,
                "sats_per_plane": 2,
            },
        })
        assert status == 400

    def test_walker_excessive_sat_count_rejected(self, running_server):
        """Excessively large satellite count should be rejected."""
        port, _ = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550,
                "inclination_deg": 53,
                "num_planes": 200,
                "sats_per_plane": 200,
            },
        })
        assert status == 400

    def test_walker_zero_planes_rejected(self, running_server):
        """Zero planes should be rejected."""
        port, _ = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550,
                "inclination_deg": 53,
                "num_planes": 0,
                "sats_per_plane": 2,
            },
        })
        assert status == 400

    # BUG-016: Settings validation
    def test_settings_zero_duration_rejected(self, running_server):
        """Zero duration should be rejected."""
        port, _ = running_server
        status, body = _api_request(port, "PUT", "/api/settings", {
            "duration_s": 0,
            "step_s": 60,
        })
        assert status == 400

    def test_settings_negative_step_rejected(self, running_server):
        """Negative step should be rejected."""
        port, _ = running_server
        status, body = _api_request(port, "PUT", "/api/settings", {
            "duration_s": 7200,
            "step_s": -10,
        })
        assert status == 400

    # BUG-027: Ground station lat/lon validation
    def test_ground_station_invalid_latitude_rejected(self, running_server):
        """Latitude outside [-90, 90] should be rejected."""
        port, _ = running_server
        status, body = _api_request(port, "POST", "/api/ground-station", {
            "name": "Bad",
            "lat": 91.0,
            "lon": 0.0,
        })
        assert status == 400

    def test_ground_station_invalid_longitude_rejected(self, running_server):
        """Longitude outside [-180, 180] should be rejected."""
        port, _ = running_server
        status, body = _api_request(port, "POST", "/api/ground-station", {
            "name": "Bad",
            "lat": 0.0,
            "lon": 181.0,
        })
        assert status == 400


class TestSessionRestoreBugs:
    """Bug fixes for session restore: CelesTrak and analysis layers."""

    def test_session_restore_celestrak_layers(self, running_server):
        """Session load should restore celestrak layers, not just walker."""
        port, mgr = running_server
        # Create a session with a celestrak-type layer in it
        session_data = {
            "epoch": EPOCH.isoformat(),
            "duration_s": 7200,
            "step_s": 60,
            "layers": [
                {
                    "name": "Constellation:Walker",
                    "category": "Constellation",
                    "layer_type": "walker",
                    "mode": "animated",
                    "visible": True,
                    "params": {
                        "altitude_km": 550,
                        "inclination_deg": 53,
                        "num_planes": 2,
                        "sats_per_plane": 2,
                        "phase_factor": 1,
                        "raan_offset_deg": 0,
                        "shell_name": "Test",
                    },
                },
            ],
        }
        # Load session
        status, body = _api_request(port, "POST", "/api/session/load", {
            "session": session_data,
        })
        assert status == 200
        assert body["restored"] >= 1, "Should restore the walker layer"

        # Now test celestrak restore: use the internal handler directly.
        # The bug: `if lt in ("walker", "celestrak") and lt == "walker":` never
        # matches celestrak. After fix, celestrak layers with walker-compatible
        # params should restore. We simulate by creating a session with a
        # celestrak layer that has walker params (since we can't fetch CelesTrak
        # in tests, we use a walker shell as the underlying params).
        celestrak_session = {
            "epoch": EPOCH.isoformat(),
            "duration_s": 7200,
            "step_s": 60,
            "layers": [
                {
                    "name": "Constellation:GPS-OPS",
                    "category": "Constellation",
                    "layer_type": "celestrak",
                    "mode": "animated",
                    "visible": True,
                    "params": {
                        "group": "GPS-OPS",
                        "altitude_km": 20200,
                        "inclination_deg": 55,
                        "num_planes": 6,
                        "sats_per_plane": 4,
                        "phase_factor": 1,
                        "raan_offset_deg": 0,
                        "shell_name": "GPS",
                    },
                },
            ],
        }
        # Clear existing layers
        _, state = _api_request(port, "GET", "/api/state")
        for layer in state["layers"]:
            _api_request(port, "DELETE", f"/api/layer/{layer['layer_id']}")

        # Load celestrak session
        status, body = _api_request(port, "POST", "/api/session/load", {
            "session": celestrak_session,
        })
        assert status == 200
        assert body["restored"] >= 1, \
            "CelesTrak layer should be restored (bug: condition never matches celestrak)"

    def test_session_restore_analysis_layers(self, running_server):
        """Session load should restore analysis layers, not just constellations."""
        port, mgr = running_server
        # Add a constellation first
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "AnalysisSrc",
            },
        })
        source_id = body["layer_id"]

        # Add an analysis layer
        status, body = _api_request(port, "POST", "/api/analysis", {
            "type": "eclipse",
            "source_layer": source_id,
            "params": {},
        })
        assert status == 201

        # Save session
        _, save_resp = _api_request(port, "POST", "/api/session/save")
        session_data = save_resp["session"]

        # Verify analysis layer is in saved data
        analysis_layers = [l for l in session_data["layers"]
                          if l["layer_type"] not in ("walker", "celestrak")]
        assert len(analysis_layers) >= 1, \
            "Saved session should include analysis layers"

        # Clear all layers
        _, state = _api_request(port, "GET", "/api/state")
        for layer in state["layers"]:
            _api_request(port, "DELETE", f"/api/layer/{layer['layer_id']}")

        # Load session
        status, body = _api_request(port, "POST", "/api/session/load", {
            "session": session_data,
        })
        assert status == 200

        # Verify analysis layers are restored (not just constellations)
        _, state = _api_request(port, "GET", "/api/state")
        restored_types = [l["layer_type"] for l in state["layers"]]
        assert "eclipse" in restored_types, \
            f"Analysis layers should be restored. Got types: {restored_types}"

    def test_save_session_includes_analysis_source_index(self, running_server):
        """Saved analysis layers should include source_layer_index for restore."""
        port, mgr = running_server
        # Add constellation + analysis
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "SaveSrc",
            },
        })
        source_id = body["layer_id"]
        _api_request(port, "POST", "/api/analysis", {
            "type": "coverage",
            "source_layer": source_id,
            "params": {"lat_step_deg": 20.0},
        })

        # Save and check
        _, save_resp = _api_request(port, "POST", "/api/session/save")
        session = save_resp["session"]
        analysis_layers = [l for l in session["layers"]
                          if l["layer_type"] == "coverage"]
        assert len(analysis_layers) == 1
        # Must have source_layer_index to know which constellation to use on restore
        assert "source_layer_index" in analysis_layers[0], \
            "Analysis layer save must include source_layer_index for restore"


class TestCapToastInResponse:
    """Verify capped_from info is included in analysis creation response."""

    def test_analysis_creation_includes_capped_from(self, running_server):
        """POST /api/analysis response should include capped_from when truncated."""
        port, mgr = running_server
        # Add a large constellation
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 6, "sats_per_plane": 20,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "CapTest",
            },
        })
        source_id = body["layer_id"]

        # Add ISL analysis (triggers capping at _MAX_TOPOLOGY_SATS=100)
        status, body = _api_request(port, "POST", "/api/analysis", {
            "type": "isl",
            "source_layer": source_id,
            "params": {},
        })
        assert status == 201

        # Check state for the new layer
        _, state = _api_request(port, "GET", "/api/state")
        isl_layers = [l for l in state["layers"] if l["layer_type"] == "isl"]
        assert len(isl_layers) == 1
        assert "capped_from" in isl_layers[0], \
            "ISL layer should show capped_from in state"
        assert isl_layers[0]["capped_from"] == 120


class TestMalformedInput:
    """BUG-018: Tests for malformed input handling."""

    def test_malformed_json_body_returns_400(self, running_server):
        """Non-JSON body with valid Content-Length should return 400."""
        port, _ = running_server
        import http.client
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        body = b"not valid json"
        conn.request("POST", "/api/constellation",
                     body=body,
                     headers={"Content-Type": "application/json",
                              "Content-Length": str(len(body))})
        resp = conn.getresponse()
        assert resp.status == 400

    def test_options_preflight(self, running_server):
        """OPTIONS request should return 204 with CORS headers."""
        port, _ = running_server
        import http.client
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        conn.request("OPTIONS", "/api/state")
        resp = conn.getresponse()
        assert resp.status == 204

    def test_unknown_method_returns_error(self, running_server):
        """PATCH request should get an error response."""
        port, _ = running_server
        import http.client
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        conn.request("PATCH", "/api/state")
        resp = conn.getresponse()
        # Should get 501 (not implemented) or similar error
        assert resp.status >= 400

    def test_put_layer_rename(self, running_server):
        """PUT /api/layer/{id} with name field should rename."""
        port, _ = running_server
        status, body = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "RenameTest",
            },
        })
        layer_id = body["layer_id"]
        status, body = _api_request(port, "PUT", f"/api/layer/{layer_id}", {
            "name": "NewName",
        })
        assert status == 200
        # Verify name changed in state
        _, state = _api_request(port, "GET", "/api/state")
        assert state["layers"][0]["name"] == "NewName"

    def test_unknown_analysis_type_returns_400(self, running_server):
        """Unknown analysis type should return 400."""
        port, _ = running_server
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 2, "sats_per_plane": 2,
            },
        })
        _, state = _api_request(port, "GET", "/api/state")
        source_id = state["layers"][0]["layer_id"]
        status, body = _api_request(port, "POST", "/api/analysis", {
            "type": "nonexistent_type",
            "source_layer": source_id,
        })
        assert status == 400

    def test_analysis_with_invalid_source_returns_404(self, running_server):
        """Analysis with missing source layer should return 404."""
        port, _ = running_server
        status, body = _api_request(port, "POST", "/api/analysis", {
            "type": "eclipse",
            "source_layer": "layer-999",
        })
        assert status == 404

    def test_delete_nonexistent_layer_returns_404(self, running_server):
        """DELETE on nonexistent layer should return 404."""
        port, _ = running_server
        status, body = _api_request(port, "DELETE", "/api/layer/layer-999")
        assert status == 404

    def test_ground_station_without_constellation(self, running_server):
        """Ground station without any constellation should still succeed."""
        port, _ = running_server
        status, body = _api_request(port, "POST", "/api/ground-station", {
            "name": "NoConst",
            "lat": 51.5,
            "lon": -0.12,
        })
        # Should succeed (empty access windows) or fail gracefully
        assert status in (201, 400, 500)

    def test_max_body_size_enforcement(self, running_server):
        """Body exceeding 10MB should be rejected."""
        port, _ = running_server
        import http.client
        conn = http.client.HTTPConnection("localhost", port, timeout=5)
        # Claim a body of 11MB but only send a small amount
        conn.putrequest("POST", "/api/constellation")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(11 * 1024 * 1024))
        conn.endheaders(b'{}')
        resp = conn.getresponse()
        assert resp.status == 400


class TestThreadSafety:
    """BUG-001/002: LayerManager must be thread-safe under concurrent access."""

    def test_layer_manager_has_lock(self):
        """LayerManager must have an RLock for thread safety."""
        from humeris.adapters.viewer_server import LayerManager
        import threading

        lm = LayerManager(EPOCH)
        assert hasattr(lm, "_lock"), "LayerManager must have a _lock attribute"
        assert isinstance(lm._lock, type(threading.RLock())), \
            "LayerManager._lock must be an RLock"

    def test_concurrent_add_unique_ids(self):
        """Concurrent add_layer calls must produce unique layer IDs."""
        from humeris.adapters.viewer_server import LayerManager

        lm = LayerManager(EPOCH)
        states = _make_states()
        ids: list[str] = []
        errors: list[Exception] = []

        def add_layer(n):
            try:
                lid = lm.add_layer(
                    name=f"Test-{n}", category="Constellation",
                    layer_type="walker", states=states, params={},
                )
                ids.append(lid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_layer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent add: {errors}"
        assert len(ids) == 20, f"Expected 20 IDs, got {len(ids)}"
        assert len(set(ids)) == 20, f"Duplicate IDs: {ids}"

    def test_concurrent_add_remove_no_crash(self):
        """Concurrent add + remove must not raise RuntimeError."""
        from humeris.adapters.viewer_server import LayerManager

        lm = LayerManager(EPOCH)
        states = _make_states()
        errors: list[Exception] = []

        # Pre-populate some layers
        pre_ids = []
        for i in range(10):
            pre_ids.append(lm.add_layer(
                name=f"Pre-{i}", category="Constellation",
                layer_type="walker", states=states, params={},
            ))

        def add_layers():
            for i in range(10):
                try:
                    lm.add_layer(
                        name=f"Add-{i}", category="Constellation",
                        layer_type="walker", states=states, params={},
                    )
                except Exception as e:
                    errors.append(e)

        def remove_layers():
            for lid in pre_ids:
                try:
                    lm.remove_layer(lid)
                except KeyError:
                    pass  # Already removed
                except Exception as e:
                    errors.append(e)

        def read_state():
            for _ in range(20):
                try:
                    lm.get_state()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=add_layers),
            threading.Thread(target=remove_layers),
            threading.Thread(target=read_state),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent operations: {errors}"


class TestSecondPassFixes:
    """Tests for second-pass audit fixes (BUG-031 through Gap-7)."""

    def test_update_layer_mode_does_not_block_get_state(self, running_server):
        """BUG-031: update_layer mode switch should not hold the lock during CZML gen."""
        port, mgr = running_server
        status, data = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 1, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "LockTest",
            },
        })
        assert status == 201
        lid = data["layer_id"]

        # Launch mode switch in a thread
        errors = []
        def switch_mode():
            try:
                _api_request(port, "PUT", f"/api/layer/{lid}", {"mode": "snapshot"})
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=switch_mode)
        t.start()
        # get_state should not be blocked for the full duration of CZML gen
        status, state = _api_request(port, "GET", "/api/state")
        t.join(timeout=30)
        assert status == 200
        assert "layers" in state
        assert not errors

    def test_ground_station_restored_on_session_load(self, running_server):
        """BUG-035: Ground stations should survive save/load cycle."""
        port, mgr = running_server
        # Add constellation first
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 1, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "GSTest",
            },
        })

        # Add ground station
        _api_request(port, "POST", "/api/ground-station", {
            "name": "TestGS", "lat": 52.0, "lon": 4.9,
        })

        # Save session
        _, save_resp = _api_request(port, "POST", "/api/session/save")
        session = save_resp["session"]

        # Verify ground station is in saved data
        gs_layers = [l for l in session["layers"]
                     if l.get("layer_type") == "ground_station"]
        assert len(gs_layers) == 1, "Ground station should be in saved session"

        # Load session (clears and restores)
        status, result = _api_request(port, "POST", "/api/session/load",
                                      {"session": session})
        assert status == 200

        # Check state — should have constellation + ground station
        _, state = _api_request(port, "GET", "/api/state")
        gs_restored = [l for l in state["layers"]
                       if l.get("layer_type") == "ground_station"]
        assert len(gs_restored) == 1, \
            f"Ground station not restored. Layers: {[l.get('layer_type') for l in state['layers']]}"

    def test_visible_state_preserved_on_session_load(self, running_server):
        """Gap-7: Hidden layers should stay hidden after save/load."""
        port, mgr = running_server
        # Add constellation
        status, data = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 1, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "VisTest",
            },
        })
        lid = data["layer_id"]

        # Hide the layer
        _api_request(port, "PUT", f"/api/layer/{lid}", {"visible": False})

        # Save session
        _, save_resp = _api_request(port, "POST", "/api/session/save")
        session = save_resp["session"]
        assert session["layers"][0]["visible"] is False

        # Load session
        _api_request(port, "POST", "/api/session/load", {"session": session})

        # Check state — layer should still be hidden
        _, state = _api_request(port, "GET", "/api/state")
        assert state["layers"][0]["visible"] is False, \
            "Layer visibility should be preserved across save/load"


class TestParamsMutationIsolation:
    """R2-01/BUG-037: _generate_czml must not mutate stored params."""

    def test_add_layer_params_not_mutated_by_czml_gen(self):
        """Params stored in LayerState must not contain _capped_from after add_layer."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(EPOCH)
        states = _make_states()
        original_params = {"max_range_km": 5000.0}
        lid = mgr.add_layer(
            name="ISL", category="Analysis", layer_type="isl",
            states=states, params=original_params,
        )
        # The caller's dict must not have been mutated
        assert "_capped_from" not in original_params, \
            "Caller's params dict was mutated by _generate_czml"

    def test_recompute_does_not_mutate_caller_params(self):
        """Recompute must not leak _capped_from into the caller's params dict."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(EPOCH)
        states = _make_states()
        lid = mgr.add_layer(
            name="Precession", category="Analysis", layer_type="precession",
            states=states, params={},
        )
        caller_params = {"some_key": "value"}
        mgr.recompute_analysis(lid, caller_params)
        assert "_capped_from" not in caller_params, \
            "Caller's params dict was mutated by recompute_analysis"


class TestExportCORS:
    """R2-02/BUG-039: Export endpoint must include full CORS headers."""

    def test_export_includes_cors_methods_header(self, running_server):
        """Export response must include Access-Control-Allow-Methods."""
        port, mgr = running_server
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 1, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "CORSTest",
            },
        })
        _, state = _api_request(port, "GET", "/api/state")
        lid = state["layers"][0]["layer_id"]
        # Direct request to export endpoint to check headers
        url = f"http://localhost:{port}/api/export/{lid}"
        req = urllib.request.Request(url, method="GET")
        resp = urllib.request.urlopen(req, timeout=10)
        allow_methods = resp.headers.get("Access-Control-Allow-Methods", "")
        allow_headers = resp.headers.get("Access-Control-Allow-Headers", "")
        assert "GET" in allow_methods, \
            f"Export missing Allow-Methods, got: '{allow_methods}'"
        assert "Content-Type" in allow_headers, \
            f"Export missing Allow-Headers, got: '{allow_headers}'"


class TestSessionLoadValidation:
    """R2-04/Gap-4: Malformed session data must return clean errors."""

    def test_non_list_layers_returns_400(self, running_server):
        """Session with layers as a string should return 400."""
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/session/load", {
            "session": {"layers": "not-a-list"},
        })
        assert status == 400, f"Expected 400 for non-list layers, got {status}"

    def test_non_dict_layer_entry_skipped(self, running_server):
        """Session with non-dict entries in layers should skip them gracefully."""
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/session/load", {
            "session": {"layers": ["not-a-dict", 42, None]},
        })
        # Should not crash — restores 0 layers
        assert status == 200
        assert body.get("restored", 0) == 0


class TestConcurrentSessionLoad:
    """R2-05/Gap-5: Concurrent session load must not crash."""

    def test_concurrent_load_and_add(self, running_server):
        """Simultaneous session load and constellation add must not crash."""
        port, mgr = running_server
        # Create a simple session to load
        session = {
            "layers": [{
                "name": "ConcTest", "category": "Constellation",
                "layer_type": "walker", "mode": "animated",
                "params": {
                    "altitude_km": 550, "inclination_deg": 53,
                    "num_planes": 1, "sats_per_plane": 2,
                    "phase_factor": 1, "raan_offset_deg": 0,
                    "shell_name": "ConcLoad",
                },
            }],
        }
        errors = []

        def load_session():
            try:
                _api_request(port, "POST", "/api/session/load", {"session": session})
            except Exception as e:
                errors.append(("load", e))

        def add_constellation():
            try:
                _api_request(port, "POST", "/api/constellation", {
                    "type": "walker",
                    "params": {
                        "altitude_km": 800, "inclination_deg": 97,
                        "num_planes": 1, "sats_per_plane": 2,
                        "phase_factor": 1, "raan_offset_deg": 0,
                        "shell_name": "ConcAdd",
                    },
                })
            except Exception as e:
                errors.append(("add", e))

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=load_session))
            threads.append(threading.Thread(target=add_constellation))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, f"Concurrent operations crashed: {errors}"


class TestCelesTrakRestoreFidelity:
    """R2-03/Gap-3: CelesTrak layer params must survive save/load."""

    def test_celestrak_params_preserved_on_restore(self, running_server):
        """CelesTrak layer should restore with original params."""
        port, mgr = running_server
        # Manually create a session with a CelesTrak layer that has specific params
        session = {
            "layers": [{
                "name": "CelesTrak:GPS-OPS", "category": "Constellation",
                "layer_type": "celestrak", "mode": "animated",
                "visible": True,
                "params": {
                    "group": "GPS-OPS",
                    "altitude_km": 20200, "inclination_deg": 55,
                    "num_planes": 6, "sats_per_plane": 5,
                    "phase_factor": 1, "raan_offset_deg": 0,
                    "shell_name": "GPS-OPS",
                },
            }],
        }
        status, body = _api_request(port, "POST", "/api/session/load",
                                    {"session": session})
        assert status == 200
        assert body["restored"] == 1

        # Save and check params are preserved
        _, save_resp = _api_request(port, "POST", "/api/session/save")
        restored_layer = save_resp["session"]["layers"][0]
        assert restored_layer["params"]["altitude_km"] == 20200
        assert restored_layer["params"]["inclination_deg"] == 55
        assert restored_layer["params"]["group"] == "GPS-OPS"


class TestLayerManagerLoadSession:
    """LayerManager.load_session() restores session state directly."""

    def test_load_session_method_exists(self):
        """LayerManager must have a load_session method."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        assert callable(getattr(mgr, "load_session", None))

    def test_load_session_restores_walker(self):
        """load_session with a walker layer should restore it."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        session_data = {
            "epoch": EPOCH.isoformat(),
            "duration_s": 7200,
            "step_s": 60,
            "layers": [{
                "name": "Constellation:LoadTest",
                "category": "Constellation",
                "layer_type": "walker",
                "mode": "snapshot",
                "visible": True,
                "params": {
                    "altitude_km": 550, "inclination_deg": 53,
                    "num_planes": 2, "sats_per_plane": 2,
                    "phase_factor": 1, "raan_offset_deg": 0,
                    "shell_name": "LoadTest",
                },
            }],
        }
        restored = mgr.load_session(session_data)
        assert restored == 1
        assert len(mgr.layers) == 1
        layer = list(mgr.layers.values())[0]
        assert layer.name == "Constellation:LoadTest"
        assert layer.layer_type == "walker"

    def test_load_session_clears_existing(self):
        """load_session must clear existing layers before restoring."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states()
        mgr.add_layer(
            name="Old", category="Constellation",
            layer_type="walker", states=states, params={},
        )
        assert len(mgr.layers) == 1
        session_data = {
            "layers": [{
                "name": "New",
                "category": "Constellation",
                "layer_type": "walker",
                "mode": "snapshot",
                "params": {
                    "altitude_km": 800, "inclination_deg": 97,
                    "num_planes": 1, "sats_per_plane": 2,
                    "phase_factor": 1, "raan_offset_deg": 0,
                    "shell_name": "New",
                },
            }],
        }
        mgr.load_session(session_data)
        assert len(mgr.layers) == 1
        layer = list(mgr.layers.values())[0]
        assert layer.name == "New"

    def test_load_session_restores_duration_step(self):
        """load_session should restore duration and step from session data."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        session_data = {
            "duration_s": 3600,
            "step_s": 30,
            "layers": [],
        }
        mgr.load_session(session_data)
        assert mgr.duration == timedelta(seconds=3600)
        assert mgr.step == timedelta(seconds=30)

    def test_load_session_returns_count(self):
        """load_session must return the number of restored layers."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        session_data = {"layers": []}
        result = mgr.load_session(session_data)
        assert result == 0


class TestCliLoadSession:
    """CLI --load-session flag loads a session file at startup."""

    def test_load_session_argument_accepted(self):
        """CLI parser must accept --load-session argument."""
        import subprocess
        result = subprocess.run(
            ["venv/bin/python3", "-m", "humeris.cli", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "--load-session" in result.stdout

    def test_load_session_file_not_found(self, tmp_path):
        """--load-session with nonexistent file should error."""
        import subprocess
        result = subprocess.run(
            [
                "venv/bin/python3", "-m", "humeris.cli",
                "--serve", "--load-session", "/nonexistent/session.json",
            ],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()

    def test_load_session_invalid_json(self, tmp_path):
        """--load-session with invalid JSON should error."""
        import subprocess
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all")
        result = subprocess.run(
            [
                "venv/bin/python3", "-m", "humeris.cli",
                "--serve", "--load-session", str(bad_file),
            ],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0


class TestFidelityModeRemoved:
    """F-06: Fidelity was dead code — verify it has been removed."""

    def test_no_fidelity_attribute(self):
        """LayerManager must not have a fidelity attribute (dead code removed)."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        assert not hasattr(mgr, "fidelity"), "fidelity was dead code and should be removed"

    def test_no_fidelity_in_save_session(self):
        """save_session must not include fidelity key."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        session = mgr.save_session()
        assert "fidelity" not in session

    def test_settings_api_ignores_fidelity(self, running_server):
        """PUT /api/settings with fidelity should not error (ignored)."""
        port, mgr = running_server
        status, body = _api_request(port, "PUT", "/api/settings", {
            "fidelity": "high",
        })
        assert status == 200
        assert "fidelity" not in body


class TestGroundStationPresets:
    """Pre-defined ground station networks can be added via API."""

    def test_presets_available_via_api(self, running_server):
        """GET /api/ground-station-presets should return preset list."""
        port, mgr = running_server
        status, body = _api_request(port, "GET", "/api/ground-station-presets")
        assert status == 200
        assert "presets" in body
        assert len(body["presets"]) >= 3  # At least DSN, ESTRACK, US

    def test_preset_has_required_fields(self, running_server):
        """Each preset must have name, description, stations."""
        port, mgr = running_server
        _, body = _api_request(port, "GET", "/api/ground-station-presets")
        for preset in body["presets"]:
            assert "name" in preset
            assert "description" in preset
            assert "stations" in preset
            assert len(preset["stations"]) >= 1
            for st in preset["stations"]:
                assert "name" in st
                assert "lat_deg" in st
                assert "lon_deg" in st

    def test_add_preset_network(self, running_server):
        """POST /api/ground-station-network should add all stations."""
        port, mgr = running_server
        # First add a constellation for access computation
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 1, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "GS-Test",
            },
        })
        status, body = _api_request(port, "POST", "/api/ground-station-network", {
            "preset": "NASA DSN",
        })
        assert status == 201 or status == 200
        assert body.get("added", 0) >= 3  # DSN has 3 stations

    def test_add_unknown_preset_returns_404(self, running_server):
        """POST with unknown preset name should return 404."""
        port, mgr = running_server
        status, body = _api_request(port, "POST", "/api/ground-station-network", {
            "preset": "Nonexistent Network",
        })
        assert status == 404


class TestMultiLayerCzmlExport:
    """GET /api/export-all merges visible layers into one CZML document."""

    def test_export_all_returns_czml(self, running_server):
        """GET /api/export-all should return merged CZML."""
        port, mgr = running_server
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 1, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "ExportAll-1",
            },
        })
        _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 800, "inclination_deg": 97,
                "num_planes": 1, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "ExportAll-2",
            },
        })
        status, body = _api_request(port, "GET", "/api/export-all")
        assert status == 200
        # Should be a list of CZML packets
        assert isinstance(body, list)
        # First packet must be a document packet
        assert body[0].get("id") == "document"
        # Should have satellites from both constellations
        assert len(body) > 3  # document + at least 2 sats from each

    def test_export_all_skips_hidden_layers(self, running_server):
        """Hidden layers should not be included in export-all."""
        port, mgr = running_server
        _, add_resp = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 550, "inclination_deg": 53,
                "num_planes": 1, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Visible",
            },
        })
        _, add_resp2 = _api_request(port, "POST", "/api/constellation", {
            "type": "walker",
            "params": {
                "altitude_km": 800, "inclination_deg": 97,
                "num_planes": 1, "sats_per_plane": 2,
                "phase_factor": 1, "raan_offset_deg": 0,
                "shell_name": "Hidden",
            },
        })
        # Hide the second layer
        layer_id = add_resp2["layer_id"]
        _api_request(port, "PUT", f"/api/layer/{layer_id}", {"visible": False})

        _, all_czml = _api_request(port, "GET", "/api/export-all")
        _, visible_czml = _api_request(port, "GET", f"/api/czml/{add_resp['layer_id']}")
        # All-export should have same count as just the visible layer
        # (document packet + satellite packets from visible only)
        assert len(all_czml) == len(visible_czml)

    def test_export_all_empty_returns_document_only(self, running_server):
        """No layers should return just a document packet."""
        port, mgr = running_server
        status, body = _api_request(port, "GET", "/api/export-all")
        assert status == 200
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0].get("id") == "document"

    def test_export_all_unique_entity_ids(self):
        """F-01: Merged CZML must not have duplicate entity IDs across layers."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states1 = _make_states(n_planes=1, n_sats=3)
        states2 = _make_states(n_planes=1, n_sats=3)
        mgr.add_layer(
            name="Walker-A", category="Constellation",
            layer_type="walker", states=states1, params={},
        )
        mgr.add_layer(
            name="Walker-B", category="Constellation",
            layer_type="walker", states=states2, params={},
        )
        merged = mgr.export_all_czml(visible_only=False)
        ids = [pkt["id"] for pkt in merged if "id" in pkt]
        assert len(ids) == len(set(ids)), (
            f"Duplicate entity IDs in merged CZML: "
            f"{[x for x in ids if ids.count(x) > 1]}"
        )

    def test_export_all_does_not_alias_packets(self):
        """F-07: Modifying returned packets must not change layer data."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        states = _make_states(n_planes=1, n_sats=2)
        mgr.add_layer(
            name="Alias-Test", category="Constellation",
            layer_type="walker", states=states, params={},
        )
        merged = mgr.export_all_czml(visible_only=False)
        # Mutate a returned non-document packet
        for pkt in merged:
            if pkt.get("id") != "document":
                pkt["INJECTED"] = True
                break
        # Original layer data must be unaffected
        layer = list(mgr.layers.values())[0]
        for pkt in layer.czml:
            assert "INJECTED" not in pkt, "export_all_czml must copy packets"

    def test_export_czml_layers_method_exists(self):
        """F-04: LayerManager must have export_czml_layers for headless use."""
        from humeris.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        assert callable(getattr(mgr, "export_czml_layers", None))


class TestCliHeadlessMode:
    """CLI --headless mode exports CZML without starting server."""

    def test_headless_argument_accepted(self):
        """CLI parser must accept --headless argument."""
        import subprocess
        result = subprocess.run(
            ["venv/bin/python3", "-m", "humeris.cli", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "--headless" in result.stdout

    def test_headless_requires_load_session(self):
        """--headless without --load-session should error."""
        import subprocess
        result = subprocess.run(
            [
                "venv/bin/python3", "-m", "humeris.cli",
                "--serve", "--headless",
            ],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
        assert "load-session" in result.stderr.lower() or "load-session" in result.stdout.lower()

    def test_headless_export_czml(self, tmp_path):
        """--headless --export-czml should export CZML files and exit."""
        import subprocess
        import json as _json
        # Create a minimal session file
        session = {
            "layers": [{
                "name": "Constellation:HeadlessTest",
                "category": "Constellation",
                "layer_type": "walker",
                "mode": "snapshot",
                "params": {
                    "altitude_km": 550, "inclination_deg": 53,
                    "num_planes": 1, "sats_per_plane": 2,
                    "phase_factor": 1, "raan_offset_deg": 0,
                    "shell_name": "HeadlessTest",
                },
            }],
        }
        session_file = tmp_path / "session.json"
        session_file.write_text(_json.dumps(session))
        output_dir = tmp_path / "czml_output"

        result = subprocess.run(
            [
                "venv/bin/python3", "-m", "humeris.cli",
                "--serve", "--headless",
                "--load-session", str(session_file),
                "--export-czml", str(output_dir),
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_dir.exists()
        czml_files = list(output_dir.glob("*.czml"))
        assert len(czml_files) >= 1


class TestViewerServerPurity:
    """Adapter purity: only stdlib + internal imports allowed."""

    def test_no_external_deps(self):
        import humeris.adapters.viewer_server as mod

        with open(mod.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {
            "json", "http", "threading", "datetime", "dataclasses",
            "urllib", "functools", "math", "numpy", "logging", "typing",
            "socketserver", "os",
        }
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
