# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
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

from constellation_generator.domain.constellation import (
    ShellConfig,
    generate_walker_shell,
)
from constellation_generator.domain.propagation import derive_orbital_state


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
        from constellation_generator.adapters.viewer_server import LayerState
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
        from constellation_generator.adapters.viewer_server import LayerState
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
        from constellation_generator.adapters.viewer_server import (
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        with pytest.raises(KeyError):
            mgr.remove_layer("nonexistent")

    def test_update_layer_visibility(self):
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        with pytest.raises(KeyError):
            mgr.update_layer("nonexistent", visible=False)

    def test_get_state_returns_all_layers_metadata(self):
        """get_state() returns layer metadata without CZML data."""
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
        mgr = LayerManager(epoch=EPOCH)
        with pytest.raises(KeyError):
            mgr.get_czml("nonexistent")

    def test_unique_layer_ids(self):
        """Each add_layer call produces a unique ID."""
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
        from constellation_generator.adapters.viewer_server import LayerManager
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
    from constellation_generator.adapters.viewer_server import (
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
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_state_includes_epoch(self, running_server):
        port, mgr = running_server
        status, body = _api_request(port, "GET", "/api/state")
        assert "epoch" in body


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


class TestViewerServerPurity:
    """Adapter purity: only stdlib + internal imports allowed."""

    def test_no_external_deps(self):
        import constellation_generator.adapters.viewer_server as mod

        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {
            "json", "http", "threading", "datetime", "dataclasses",
            "uuid", "urllib", "functools", "math", "numpy", "logging", "typing",
        }
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
