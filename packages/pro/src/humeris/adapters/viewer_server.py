# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Interactive viewer server for Cesium-based constellation visualization.

Local HTTP server (stdlib only) serving a Cesium viewer with on-demand
CZML generation, dynamic constellation management, and analysis layers.

Usage:
    python view_constellation.py --serve           # start server, open browser
    python view_constellation.py --serve --port 8765
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from humeris.domain.constellation import (
    ShellConfig,
    Satellite,
    generate_walker_shell,
)
from humeris.domain.propagation import (
    OrbitalState,
    derive_orbital_state,
)
from humeris.domain.observation import GroundStation
from humeris.adapters.czml_exporter import (
    constellation_packets,
    snapshot_packets,
    coverage_packets,
    ground_track_packets,
)
from humeris.adapters.czml_visualization import (
    eclipse_constellation_packets,
    eclipse_snapshot_packets,
    sensor_footprint_packets,
    ground_station_packets,
    coverage_evolution_packets,
    isl_topology_packets,
    fragility_constellation_packets,
    hazard_evolution_packets,
    coverage_connectivity_packets,
    network_eclipse_packets,
    precession_constellation_packets,
    conjunction_replay_packets,
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
from humeris.domain.sensor import SensorConfig, SensorType
from humeris.domain.atmosphere import DragConfig
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00


logger = logging.getLogger(__name__)


def _make_density_func(epoch: datetime) -> Any:
    """Create a density_func closure for the given epoch."""
    def density_func(altitude_km: float, ep: datetime) -> float:
        return atmospheric_density_nrlmsise00(altitude_km, ep)
    return density_func

_DEFAULT_DURATION = timedelta(hours=2)
_DEFAULT_STEP = timedelta(seconds=60)
_SNAPSHOT_THRESHOLD = 100  # sats above this default to snapshot mode

_DEFAULT_LINK_CONFIG = LinkConfig(
    frequency_hz=26e9, transmit_power_w=10.0,
    tx_antenna_gain_dbi=35.0, rx_antenna_gain_dbi=35.0,
    system_noise_temp_k=500.0, bandwidth_hz=100e6,
    additional_losses_db=2.0, required_snr_db=10.0,
)
_DEFAULT_SENSOR = SensorConfig(sensor_type=SensorType.CIRCULAR, half_angle_deg=30.0)
_DEFAULT_DRAG = DragConfig(cd=2.2, area_m2=0.01, mass_kg=4.0)
_MAX_TOPOLOGY_SATS = 100   # Cap for O(n²) computations
_MAX_PRECESSION_SATS = 24  # Subset for week-long precession view


@dataclass
class LayerState:
    """State for a single visualization layer."""

    layer_id: str
    name: str
    category: str
    layer_type: str
    mode: str
    visible: bool
    states: list[OrbitalState]
    params: dict[str, Any]
    czml: list[dict[str, Any]]


def _generate_czml(
    layer_type: str,
    mode: str,
    states: list[OrbitalState],
    epoch: datetime,
    name: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Dispatch CZML generation based on layer type and mode."""
    duration = params.get("duration", _DEFAULT_DURATION)
    step = params.get("step", _DEFAULT_STEP)

    if layer_type == "walker" or layer_type == "celestrak":
        if mode == "snapshot":
            return snapshot_packets(states, epoch, name=name)
        return constellation_packets(states, epoch, duration, step, name=name)

    if layer_type == "eclipse":
        if mode == "snapshot":
            return eclipse_snapshot_packets(states, epoch, name=name)
        return eclipse_constellation_packets(
            states, epoch, duration, step, name=name,
        )

    if layer_type == "coverage":
        lat_step = params.get("lat_step_deg", 10.0)
        lon_step = params.get("lon_step_deg", 10.0)
        min_elev = params.get("min_elevation_deg", 10.0)
        if mode == "animated":
            coverage_step = params.get("coverage_step", step)
            return coverage_evolution_packets(
                states, epoch, duration, coverage_step,
                lat_step_deg=lat_step, lon_step_deg=lon_step,
                min_elevation_deg=min_elev, name=name,
            )
        # Snapshot: compute coverage at epoch, use static packets
        from humeris.domain.coverage import (
            compute_coverage_snapshot,
        )
        points = compute_coverage_snapshot(
            states, epoch,
            lat_step_deg=lat_step, lon_step_deg=lon_step,
            min_elevation_deg=min_elev,
        )
        return coverage_packets(
            points, lat_step_deg=lat_step, lon_step_deg=lon_step, name=name,
        )

    if layer_type == "ground_track":
        from humeris.domain.ground_track import (
            compute_ground_track,
        )
        from humeris.domain.constellation import Satellite as Sat
        # Use first orbital state to compute ground track
        if not states:
            return [{"id": "document", "name": name, "version": "1.0"}]
        state = states[0]
        # Create a minimal Satellite from OrbitalState for ground_track
        sat = Sat(
            name="sat-0",
            position_eci=_eci_from_state(state),
            velocity_eci=_velocity_from_state(state),
            plane_index=0,
            sat_index=0,
            raan_deg=_rad_to_deg(state.raan_rad),
            true_anomaly_deg=_rad_to_deg(state.true_anomaly_rad),
            epoch=state.reference_epoch,
        )
        track = compute_ground_track(sat, epoch, duration, step)
        return ground_track_packets(track, name=name)

    if layer_type == "ground_station":
        # params must contain station + source_states info
        station = params.get("_station")
        if station is None:
            return [{"id": "document", "name": name, "version": "1.0"}]
        return ground_station_packets(
            station, states, epoch, duration, step, name=name,
        )

    if layer_type == "sensor":
        sensor = params.get("_sensor", _DEFAULT_SENSOR)
        return sensor_footprint_packets(
            states, epoch, duration, step, sensor, name=name,
        )

    if layer_type == "isl":
        link = params.get("_link_config", _DEFAULT_LINK_CONFIG)
        max_range = params.get("max_range_km", 5000.0)
        capped = states[:_MAX_TOPOLOGY_SATS]
        dur_s = duration.total_seconds()
        step_s = step.total_seconds()
        return isl_topology_packets(
            capped, epoch, link, epoch, dur_s, step_s,
            max_range_km=max_range, name=name,
        )

    if layer_type == "fragility":
        import math as _math
        link = params.get("_link_config", _DEFAULT_LINK_CONFIG)
        capped = states[:_MAX_TOPOLOGY_SATS]
        if capped:
            sma = capped[0].semi_major_axis_m
            n_rad_s = _math.sqrt(OrbitalConstants.MU_EARTH / sma ** 3)
            period_s = 2.0 * _math.pi / n_rad_s
        else:
            n_rad_s = 0.001
            period_s = 5400.0
        ctrl_s = params.get("control_duration_s", period_s)
        dur_s = duration.total_seconds()
        step_s = step.total_seconds()
        return fragility_constellation_packets(
            capped, epoch, link, n_rad_s, ctrl_s, dur_s, step_s, name=name,
        )

    if layer_type == "hazard":
        from humeris.domain.lifetime import compute_orbit_lifetime
        from humeris.domain.statistical_analysis import (
            compute_lifetime_survival_curve,
        )
        if not states:
            return [{"id": "document", "name": name, "version": "1.0"}]
        rep = states[0]
        drag = params.get("_drag_config", _DEFAULT_DRAG)
        density_fn = _make_density_func(epoch)
        lifetime_result = compute_orbit_lifetime(
            rep.semi_major_axis_m, rep.eccentricity, drag, epoch,
            density_func=density_fn,
        )
        curve = compute_lifetime_survival_curve(lifetime_result)
        # Duration: half lifetime capped at 1 year, daily steps
        half_life_s = curve.mean_remaining_life_days * 86400.0 / 2.0
        max_dur_s = min(half_life_s, 365.25 * 86400.0)
        hazard_step_s = 86400.0  # daily
        return hazard_evolution_packets(
            states, curve, epoch, max_dur_s, hazard_step_s, name=name,
        )

    if layer_type == "network_eclipse":
        link = params.get("_link_config", _DEFAULT_LINK_CONFIG)
        max_range = params.get("max_range_km", 5000.0)
        capped = states[:_MAX_TOPOLOGY_SATS]
        dur_s = duration.total_seconds()
        step_s = step.total_seconds()
        return network_eclipse_packets(
            capped, link, epoch, dur_s, step_s,
            max_range_km=max_range, name=name,
        )

    if layer_type == "coverage_connectivity":
        link = params.get("_link_config", _DEFAULT_LINK_CONFIG)
        capped = states[:_MAX_TOPOLOGY_SATS]
        dur_s = duration.total_seconds()
        step_s = step.total_seconds()
        return coverage_connectivity_packets(
            capped, link, epoch, dur_s, step_s, name=name,
        )

    if layer_type == "precession":
        subset = states[:_MAX_PRECESSION_SATS]
        prec_duration = timedelta(days=7)
        prec_step = timedelta(minutes=15)
        return constellation_packets(
            subset, epoch, prec_duration, prec_step, name=name,
        )

    if layer_type == "kessler_heatmap":
        return kessler_heatmap_packets(states, epoch, duration, step, name=name)

    if layer_type == "conjunction_hazard":
        capped = states[:_MAX_TOPOLOGY_SATS]
        return conjunction_hazard_packets(
            capped, epoch, duration, step, name=name,
        )

    if layer_type == "conjunction":
        if len(states) < 2:
            return [{"id": "document", "name": name, "version": "1.0"}]
        state_a = states[0]
        state_b = states[len(states) // 2]
        window = timedelta(minutes=30)
        conj_step = timedelta(seconds=10)
        return conjunction_replay_packets(
            state_a, state_b, epoch, window, conj_step,
        )

    if layer_type == "dop_grid":
        lat_step = params.get("lat_step_deg", 10.0)
        lon_step = params.get("lon_step_deg", 10.0)
        return dop_grid_packets(
            states, epoch, lat_step_deg=lat_step, lon_step_deg=lon_step,
            name=name,
        )

    if layer_type == "radiation":
        return radiation_coloring_packets(
            states, epoch, duration, step, name=name,
        )

    if layer_type == "beta_angle":
        return beta_angle_packets(states, epoch, name=name)

    if layer_type == "deorbit":
        drag = params.get("_drag_config", _DEFAULT_DRAG)
        return deorbit_compliance_packets(
            states, epoch, drag, name=name,
        )

    if layer_type == "station_keeping":
        drag = params.get("_drag_config", _DEFAULT_DRAG)
        density_fn = _make_density_func(epoch)
        return station_keeping_packets(
            states, epoch, drag_config=drag, density_func=density_fn, name=name,
        )

    if layer_type == "cascade_sir":
        dur_s = duration.total_seconds()
        step_s = step.total_seconds()
        return cascade_evolution_packets(
            states, epoch, dur_s, step_s, name=name,
        )

    if layer_type == "relative_motion":
        if len(states) < 2:
            return [{"id": "document", "name": name, "version": "1.0"}]
        return relative_motion_packets(
            states[0], states[1], epoch, duration, step, name=name,
        )

    if layer_type == "maintenance":
        drag = params.get("_drag_config", _DEFAULT_DRAG)
        density_fn = _make_density_func(epoch)
        return maintenance_schedule_packets(
            states, epoch, drag_config=drag, density_func=density_fn, name=name,
        )

    # Fallback: return empty document
    return [{"id": "document", "name": name, "version": "1.0"}]


def _eci_from_state(state: OrbitalState) -> tuple[float, float, float]:
    """Extract ECI position from orbital state at reference epoch."""
    from humeris.domain.propagation import propagate_to
    pos, _ = propagate_to(state, state.reference_epoch)
    return (pos[0], pos[1], pos[2])


def _velocity_from_state(state: OrbitalState) -> tuple[float, float, float]:
    """Extract ECI velocity from orbital state at reference epoch."""
    from humeris.domain.propagation import propagate_to
    _, vel = propagate_to(state, state.reference_epoch)
    return (vel[0], vel[1], vel[2])


def _rad_to_deg(rad: float) -> float:
    import math
    return rad * 180.0 / math.pi


class LayerManager:
    """Manages visualization layers and their CZML generation."""

    def __init__(self, epoch: datetime) -> None:
        self.epoch = epoch
        self.layers: dict[str, LayerState] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"layer-{self._counter}"

    def add_layer(
        self,
        name: str,
        category: str,
        layer_type: str,
        states: list[OrbitalState],
        params: dict[str, Any],
        mode: str | None = None,
    ) -> str:
        """Add a visualization layer. Returns the layer ID."""
        if mode is None:
            mode = "snapshot" if len(states) > _SNAPSHOT_THRESHOLD else "animated"

        layer_id = self._next_id()
        czml = _generate_czml(
            layer_type, mode, states, self.epoch, name, params,
        )
        self.layers[layer_id] = LayerState(
            layer_id=layer_id,
            name=name,
            category=category,
            layer_type=layer_type,
            mode=mode,
            visible=True,
            states=states,
            params=params,
            czml=czml,
        )
        return layer_id

    def add_ground_station(
        self,
        name: str,
        lat_deg: float,
        lon_deg: float,
        source_states: list[OrbitalState],
        alt_m: float = 0.0,
    ) -> str:
        """Add a ground station layer. Returns the layer ID."""
        station = GroundStation(
            name=name, lat_deg=lat_deg, lon_deg=lon_deg, alt_m=alt_m,
        )
        params = {
            "name": name,
            "lat_deg": lat_deg,
            "lon_deg": lon_deg,
            "_station": station,
        }
        return self.add_layer(
            name=f"Ground Station:{name}",
            category="Ground Station",
            layer_type="ground_station",
            states=source_states,
            params=params,
        )

    def remove_layer(self, layer_id: str) -> None:
        """Remove a layer by ID. Raises KeyError if not found."""
        if layer_id not in self.layers:
            raise KeyError(f"Layer not found: {layer_id}")
        del self.layers[layer_id]

    def update_layer(
        self,
        layer_id: str,
        mode: str | None = None,
        visible: bool | None = None,
    ) -> None:
        """Update layer mode and/or visibility. Raises KeyError if not found."""
        if layer_id not in self.layers:
            raise KeyError(f"Layer not found: {layer_id}")
        layer = self.layers[layer_id]
        if visible is not None:
            layer.visible = visible
        if mode is not None and mode != layer.mode:
            layer.mode = mode
            layer.czml = _generate_czml(
                layer.layer_type, mode, layer.states,
                self.epoch, layer.name, layer.params,
            )

    def get_state(self) -> dict[str, Any]:
        """Return all layer metadata (no CZML data)."""
        layers_info = []
        for layer in self.layers.values():
            layers_info.append({
                "layer_id": layer.layer_id,
                "name": layer.name,
                "category": layer.category,
                "layer_type": layer.layer_type,
                "mode": layer.mode,
                "visible": layer.visible,
                "num_entities": max(0, len(layer.czml) - 1),
                "params": {
                    k: v for k, v in layer.params.items()
                    if not k.startswith("_")
                },
            })
        return {
            "epoch": self.epoch.isoformat(),
            "layers": layers_info,
        }

    def get_czml(self, layer_id: str) -> list[dict[str, Any]]:
        """Return CZML packets for a layer. Raises KeyError if not found."""
        if layer_id not in self.layers:
            raise KeyError(f"Layer not found: {layer_id}")
        return self.layers[layer_id].czml


class ConstellationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the interactive viewer API."""

    # Set by create_viewer_server
    layer_manager: LayerManager
    html_content: str = ""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging."""
        logger.debug(format, *args)

    def _set_headers(self, status: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, data: Any, status: int = 200) -> None:
        self._set_headers(status, "application/json")
        self.wfile.write(json.dumps(data).encode())

    def _error_response(self, status: int, message: str) -> None:
        self._json_response({"error": message}, status)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode())

    def _route_path(self) -> tuple[str, str]:
        """Parse path into (base, param). E.g. /api/czml/foo -> ('/api/czml', 'foo')."""
        parts = self.path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "api":
            return f"/api/{parts[1]}", "/".join(parts[2:])
        if len(parts) >= 2 and parts[0] == "api":
            return f"/api/{parts[1]}", ""
        return f"/{'/'.join(parts)}" if parts[0] else "/", ""

    # --- GET ---

    def do_GET(self) -> None:
        base, param = self._route_path()

        if base == "/" or self.path == "/":
            self._set_headers(200, "text/html")
            self.wfile.write(self.html_content.encode())
            return

        if base == "/api/state":
            self._json_response(self.layer_manager.get_state())
            return

        if base == "/api/czml" and param:
            try:
                czml = self.layer_manager.get_czml(param)
                self._json_response(czml)
            except KeyError:
                self._error_response(404, f"Layer not found: {param}")
            return

        self._error_response(404, "Not found")

    # --- POST ---

    def do_POST(self) -> None:
        base, param = self._route_path()

        if base == "/api/constellation":
            self._handle_add_constellation()
            return

        if base == "/api/analysis":
            self._handle_add_analysis()
            return

        if base == "/api/ground-station":
            self._handle_add_ground_station()
            return

        self._error_response(404, "Not found")

    def _handle_add_constellation(self) -> None:
        body = self._read_body()
        ctype = body.get("type")
        params = body.get("params", {})

        if ctype == "walker":
            try:
                config = ShellConfig(
                    altitude_km=params["altitude_km"],
                    inclination_deg=params["inclination_deg"],
                    num_planes=params["num_planes"],
                    sats_per_plane=params["sats_per_plane"],
                    phase_factor=params.get("phase_factor", 1),
                    raan_offset_deg=params.get("raan_offset_deg", 0.0),
                    shell_name=params.get("shell_name", "Walker"),
                )
                sats = generate_walker_shell(config)
                states = [
                    derive_orbital_state(s, self.layer_manager.epoch, include_j2=True)
                    for s in sats
                ]
                layer_id = self.layer_manager.add_layer(
                    name=f"Constellation:{config.shell_name}",
                    category="Constellation",
                    layer_type="walker",
                    states=states,
                    params=params,
                )
                self._json_response({"layer_id": layer_id}, 201)
            except (KeyError, TypeError, ValueError) as e:
                self._error_response(400, f"Invalid walker params: {e}")
            return

        if ctype == "celestrak":
            try:
                from humeris.adapters.celestrak import (
                    CelesTrakAdapter,
                )
                adapter = CelesTrakAdapter()
                group = params.get("group")
                name = params.get("name")
                sats = adapter.fetch_satellites(
                    group=group, name=name,
                    epoch=self.layer_manager.epoch,
                )
                if not sats:
                    self._error_response(400, "No satellites found")
                    return
                states = [
                    derive_orbital_state(s, self.layer_manager.epoch, include_j2=True)
                    for s in sats
                ]
                display_name = group or name or "CelesTrak"
                layer_id = self.layer_manager.add_layer(
                    name=f"Constellation:{display_name}",
                    category="Constellation",
                    layer_type="celestrak",
                    states=states,
                    params=params,
                )
                self._json_response({"layer_id": layer_id}, 201)
            except Exception as e:
                self._error_response(400, f"CelesTrak fetch failed: {e}")
            return

        self._error_response(400, f"Unknown constellation type: {ctype}")

    def _handle_add_analysis(self) -> None:
        body = self._read_body()
        analysis_type = body.get("type")
        source_layer_id = body.get("source_layer")
        params = body.get("params", {})

        if not source_layer_id or source_layer_id not in self.layer_manager.layers:
            self._error_response(404, f"Source layer not found: {source_layer_id}")
            return

        source = self.layer_manager.layers[source_layer_id]
        valid_types = {
            "eclipse", "coverage", "sensor", "isl", "fragility",
            "hazard", "network_eclipse", "coverage_connectivity",
            "ground_track", "conjunction", "precession",
            "kessler_heatmap", "conjunction_hazard",
            "dop_grid", "radiation", "beta_angle", "deorbit",
            "station_keeping", "cascade_sir", "relative_motion",
            "maintenance",
        }
        if analysis_type not in valid_types:
            self._error_response(400, f"Unknown analysis type: {analysis_type}")
            return

        try:
            layer_id = self.layer_manager.add_layer(
                name=f"Analysis:{analysis_type.replace('_', ' ').title()}",
                category="Analysis",
                layer_type=analysis_type,
                states=source.states,
                params=params,
            )
            self._json_response({"layer_id": layer_id}, 201)
        except Exception as e:
            self._error_response(500, f"Analysis generation failed: {e}")

    def _handle_add_ground_station(self) -> None:
        body = self._read_body()
        name = body.get("name", "Station")
        lat = body.get("lat", 0.0)
        lon = body.get("lon", 0.0)

        # Find first constellation layer for access computation
        source_states: list[OrbitalState] = []
        for layer in self.layer_manager.layers.values():
            if layer.category == "Constellation":
                source_states = layer.states
                break

        if not source_states:
            # Use empty states — station will render without access windows
            source_states = []

        try:
            layer_id = self.layer_manager.add_ground_station(
                name=name,
                lat_deg=lat,
                lon_deg=lon,
                source_states=source_states[:6],  # limit for performance
            )
            self._json_response({"layer_id": layer_id}, 201)
        except Exception as e:
            self._error_response(500, f"Ground station failed: {e}")

    # --- PUT ---

    def do_PUT(self) -> None:
        base, param = self._route_path()

        if base == "/api/layer" and param:
            body = self._read_body()
            try:
                self.layer_manager.update_layer(
                    param,
                    mode=body.get("mode"),
                    visible=body.get("visible"),
                )
                layer = self.layer_manager.layers[param]
                self._json_response({
                    "layer_id": layer.layer_id,
                    "mode": layer.mode,
                    "visible": layer.visible,
                })
            except KeyError:
                self._error_response(404, f"Layer not found: {param}")
            return

        self._error_response(404, "Not found")

    # --- DELETE ---

    def do_DELETE(self) -> None:
        base, param = self._route_path()

        if base == "/api/layer" and param:
            try:
                self.layer_manager.remove_layer(param)
                self._json_response({"status": "removed"})
            except KeyError:
                self._error_response(404, f"Layer not found: {param}")
            return

        self._error_response(404, "Not found")

    # --- OPTIONS (CORS preflight) ---

    def do_OPTIONS(self) -> None:
        self._set_headers(204)


def create_viewer_server(
    layer_manager: LayerManager,
    port: int = 8765,
    cesium_token: str = "",
) -> HTTPServer:
    """Create an HTTP server for the interactive viewer.

    Args:
        layer_manager: LayerManager instance with layer state.
        port: Port to serve on.
        cesium_token: Cesium Ion access token (optional).

    Returns:
        HTTPServer ready to serve_forever().
    """
    from humeris.adapters.cesium_viewer import (
        generate_interactive_html,
    )

    html_content = generate_interactive_html(
        title="Constellation Viewer",
        cesium_token=cesium_token,
        port=port,
    )

    # Create handler class with shared state
    handler = type(
        "BoundHandler",
        (ConstellationHandler,),
        {
            "layer_manager": layer_manager,
            "html_content": html_content,
        },
    )

    server = HTTPServer(("localhost", port), handler)
    return server
