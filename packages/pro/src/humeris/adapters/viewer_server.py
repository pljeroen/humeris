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
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import socketserver
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
from humeris.domain.eclipse import compute_beta_angle
from humeris.domain.constellation_metrics import (
    compute_coverage_statistics,
    compute_eclipse_statistics,
)
from humeris.domain.coverage import compute_coverage_snapshot
from humeris.domain.station_keeping import compute_station_keeping_budget, StationKeepingConfig
from humeris.domain.lifetime import compute_orbit_lifetime


logger = logging.getLogger(__name__)


def _make_density_func(epoch: datetime) -> Any:
    """Create a density_func closure for the given epoch."""
    def density_func(altitude_km: float, ep: datetime) -> float:
        return atmospheric_density_nrlmsise00(altitude_km, ep)
    return density_func

_MAX_SWEEP_ITERATIONS = 1000
_VALID_CONSTRAINT_OPS = {">=", "<=", ">", "<", "=="}


def _sanitize_for_json(obj: Any) -> Any:
    """Replace float('inf') and float('nan') with None for JSON safety."""
    import math
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


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
_GROUND_STATION_SAT_LIMIT = 20  # Max sats for ground station access computation

# Pre-defined ground station network presets
_GROUND_STATION_PRESETS: list[dict[str, Any]] = [
    {
        "name": "NASA DSN",
        "description": "NASA Deep Space Network — three 70m antenna complexes",
        "stations": [
            {"name": "Goldstone", "lat_deg": 35.4267, "lon_deg": -116.89},
            {"name": "Madrid", "lat_deg": 40.4314, "lon_deg": -4.2481},
            {"name": "Canberra", "lat_deg": -35.4014, "lon_deg": 148.9817},
        ],
    },
    {
        "name": "ESA ESTRACK",
        "description": "ESA European Space Tracking network",
        "stations": [
            {"name": "Kourou", "lat_deg": 5.2522, "lon_deg": -52.7764},
            {"name": "Redu", "lat_deg": 50.0019, "lon_deg": 5.1464},
            {"name": "Cebreros", "lat_deg": 40.4528, "lon_deg": -4.3678},
            {"name": "New Norcia", "lat_deg": -31.0483, "lon_deg": 116.1917},
            {"name": "Malargue", "lat_deg": -35.7758, "lon_deg": -69.3983},
        ],
    },
    {
        "name": "US NEN",
        "description": "NASA Near Earth Network ground stations",
        "stations": [
            {"name": "Wallops", "lat_deg": 37.9333, "lon_deg": -75.4667},
            {"name": "Fairbanks", "lat_deg": 64.8594, "lon_deg": -147.8536},
            {"name": "McMurdo", "lat_deg": -77.8461, "lon_deg": 166.6689},
            {"name": "Svalbard", "lat_deg": 78.2306, "lon_deg": 15.3894},
            {"name": "Singapore", "lat_deg": 1.3521, "lon_deg": 103.8198},
        ],
    },
    {
        "name": "KSAT Lite",
        "description": "Kongsberg Satellite Services — polar and equatorial sites",
        "stations": [
            {"name": "Svalbard", "lat_deg": 78.2306, "lon_deg": 15.3894},
            {"name": "Troll", "lat_deg": -72.0117, "lon_deg": 2.5350},
            {"name": "Puertollano", "lat_deg": 38.6722, "lon_deg": -4.1500},
        ],
    },
    {
        "name": "Global Coverage",
        "description": "Minimal set for near-continuous LEO coverage",
        "stations": [
            {"name": "Svalbard", "lat_deg": 78.2306, "lon_deg": 15.3894},
            {"name": "Fairbanks", "lat_deg": 64.8594, "lon_deg": -147.8536},
            {"name": "Santiago", "lat_deg": -33.4489, "lon_deg": -70.6693},
            {"name": "Hartebeesthoek", "lat_deg": -25.8872, "lon_deg": 27.7075},
            {"name": "Tokyo", "lat_deg": 35.6762, "lon_deg": 139.6503},
            {"name": "McMurdo", "lat_deg": -77.8461, "lon_deg": 166.6689},
        ],
    },
]

# Color legends for analysis types
_LEGENDS: dict[str, list[dict[str, str]]] = {
    "eclipse": [
        {"label": "Sunlit", "color": "#FFD700"},
        {"label": "Penumbra", "color": "#FF8C00"},
        {"label": "Umbra", "color": "#8B0000"},
    ],
    "coverage": [
        {"label": "0 sats", "color": "#000080"},
        {"label": "1-2 sats", "color": "#0000FF"},
        {"label": "3-5 sats", "color": "#00FF00"},
        {"label": "6+ sats", "color": "#FF0000"},
    ],
    "isl": [
        {"label": "High SNR", "color": "#00FF00"},
        {"label": "Medium SNR", "color": "#FFFF00"},
        {"label": "Low SNR", "color": "#FF0000"},
    ],
    "fragility": [
        {"label": "Robust", "color": "#00FF00"},
        {"label": "Moderate", "color": "#FFFF00"},
        {"label": "Fragile", "color": "#FF0000"},
    ],
    "hazard": [
        {"label": "Long lifetime", "color": "#00FF00"},
        {"label": "Medium", "color": "#FFFF00"},
        {"label": "Short lifetime", "color": "#FF0000"},
    ],
    "radiation": [
        {"label": "Low dose", "color": "#00FF00"},
        {"label": "Medium dose", "color": "#FFFF00"},
        {"label": "High dose", "color": "#FF0000"},
    ],
    "kessler_heatmap": [
        {"label": "Low density", "color": "#000080"},
        {"label": "Medium density", "color": "#FFFF00"},
        {"label": "High density", "color": "#FF0000"},
    ],
    "conjunction_hazard": [
        {"label": "Green", "color": "#00FF00"},
        {"label": "Yellow", "color": "#FFFF00"},
        {"label": "Red", "color": "#FF0000"},
    ],
    "deorbit": [
        {"label": "Compliant", "color": "#00FF00"},
        {"label": "Non-compliant", "color": "#FF0000"},
    ],
    "station_keeping": [
        {"label": "Low dV", "color": "#00FF00"},
        {"label": "Medium dV", "color": "#FFFF00"},
        {"label": "High dV", "color": "#FF0000"},
    ],
    "maintenance": [
        {"label": "Nominal", "color": "#00FF00"},
        {"label": "Due soon", "color": "#FFFF00"},
        {"label": "Overdue", "color": "#FF0000"},
    ],
    "beta_angle": [
        {"label": "Low beta", "color": "#0000FF"},
        {"label": "High beta", "color": "#FF8C00"},
    ],
    "network_eclipse": [
        {"label": "Both sunlit", "color": "#00FF00"},
        {"label": "One eclipsed", "color": "#FFFF00"},
        {"label": "Both eclipsed", "color": "#FF0000"},
    ],
    "cascade_sir": [
        {"label": "Susceptible", "color": "#00FF00"},
        {"label": "Infected", "color": "#FF0000"},
        {"label": "Removed", "color": "#808080"},
    ],
}


def _merge_clocks(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge two CZML clock objects, taking the widest time range.

    CZML clock interval format: "start_iso/end_iso".
    """
    if existing is None:
        return dict(incoming)
    merged = dict(existing)
    try:
        e_start, e_end = existing["interval"].split("/")
        i_start, i_end = incoming["interval"].split("/")
        start = min(e_start, i_start)  # ISO8601 strings sort lexically
        end = max(e_end, i_end)
        merged["interval"] = f"{start}/{end}"
        merged["currentTime"] = start
    except (KeyError, ValueError):
        pass
    return merged


def _compute_metrics(
    layer_type: str,
    states: list[OrbitalState],
    epoch: datetime,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Compute quantitative metrics for an analysis layer.

    Returns a dict of metric key-value pairs, or None for layer types
    that don't have meaningful metrics.
    """
    import math
    try:
        if layer_type == "coverage":
            lat_step = params.get("lat_step_deg", 10.0)
            lon_step = params.get("lon_step_deg", 10.0)
            min_elev = params.get("min_elevation_deg", 10.0)
            grid = compute_coverage_snapshot(
                states, epoch,
                lat_step_deg=lat_step, lon_step_deg=lon_step,
                min_elevation_deg=min_elev,
            )
            stats = compute_coverage_statistics(grid, n_fold_levels=[1, 2, 4])
            return {
                "mean_visible": round(stats.mean_visible, 2),
                "max_visible": stats.max_visible,
                "min_visible": stats.min_visible,
                "percent_covered": round(stats.percent_covered, 1),
                "n_fold_coverage": {str(k): round(v, 1) for k, v in stats.n_fold_coverage.items()},
            }

        if layer_type == "eclipse":
            duration = params.get("duration", _DEFAULT_DURATION)
            step = params.get("step", _DEFAULT_STEP)
            if isinstance(duration, (int, float)):
                duration = timedelta(seconds=duration)
            if isinstance(step, (int, float)):
                step = timedelta(seconds=step)
            sunlit_pcts = []
            max_eclipse_s = 0.0
            for state in states:
                es = compute_eclipse_statistics(state, epoch, duration, step)
                sunlit_pcts.append((1.0 - es.eclipse_fraction) * 100.0)
                if es.max_duration_s > max_eclipse_s:
                    max_eclipse_s = es.max_duration_s
            avg_sunlit = sum(sunlit_pcts) / len(sunlit_pcts) if sunlit_pcts else 0.0
            return {
                "avg_sunlit_pct": round(avg_sunlit, 1),
                "max_eclipse_min": round(max_eclipse_s / 60.0, 1),
            }

        if layer_type == "beta_angle":
            betas = []
            for state in states:
                beta = compute_beta_angle(
                    state.raan_rad, state.inclination_rad, epoch,
                )
                betas.append(beta)
            if not betas:
                return {"min_beta_deg": 0.0, "max_beta_deg": 0.0, "avg_beta_deg": 0.0}
            return {
                "min_beta_deg": round(min(betas), 1),
                "max_beta_deg": round(max(betas), 1),
                "avg_beta_deg": round(sum(betas) / len(betas), 1),
            }

        if layer_type == "deorbit":
            drag = params.get("_drag_config", _DEFAULT_DRAG)
            compliant = 0
            total = len(states)
            for state in states:
                try:
                    result = compute_orbit_lifetime(
                        state.semi_major_axis_m, state.eccentricity,
                        drag, epoch,
                    )
                    lifetime_years = result.lifetime_days / 365.25
                except (ValueError, ArithmeticError):
                    lifetime_years = 999.0
                if lifetime_years <= 25.0:
                    compliant += 1
            return {
                "compliant": compliant,
                "total": total,
                "compliance_pct": round(100.0 * compliant / total, 1) if total > 0 else 0.0,
            }

        if layer_type == "station_keeping":
            drag = params.get("_drag_config", _DEFAULT_DRAG)
            density_fn = _make_density_func(epoch)
            dvs = []
            for state in states:
                alt_km = (state.semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
                inc_deg = math.degrees(state.inclination_rad)
                try:
                    sk_config = StationKeepingConfig(
                        target_altitude_km=alt_km,
                        inclination_deg=inc_deg,
                        drag_config=drag,
                        isp_s=220.0,
                        dry_mass_kg=4.0,
                        propellant_mass_kg=0.5,
                    )
                    budget = compute_station_keeping_budget(
                        sk_config, density_func=density_fn, epoch=epoch,
                    )
                    dvs.append(budget.total_dv_per_year_ms)
                except (ValueError, ArithmeticError):
                    pass
            if dvs:
                return {
                    "avg_dv_m_s": round(sum(dvs) / len(dvs), 3),
                    "max_dv_m_s": round(max(dvs), 3),
                    "min_dv_m_s": round(min(dvs), 3),
                }
            return {"avg_dv_m_s": 0.0, "max_dv_m_s": 0.0, "min_dv_m_s": 0.0}

    except Exception:
        logger.debug("Metrics computation failed for %s", layer_type, exc_info=True)
        return None

    return None


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
    source_layer_id: str = ""  # BUG-028: explicit source reference for analysis layers
    sat_names: list[str] | None = None
    metrics: dict[str, Any] | None = None


def _generate_czml(
    layer_type: str,
    mode: str,
    states: list[OrbitalState],
    epoch: datetime,
    name: str,
    params: dict[str, Any],
    sat_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Dispatch CZML generation based on layer type and mode."""
    duration = params.get("duration", _DEFAULT_DURATION)
    step = params.get("step", _DEFAULT_STEP)

    if layer_type == "walker" or layer_type == "celestrak":
        if mode == "snapshot":
            return snapshot_packets(states, epoch, name=name, sat_names=sat_names)
        return constellation_packets(states, epoch, duration, step, name=name, sat_names=sat_names)

    if layer_type == "eclipse":
        if mode == "snapshot":
            return eclipse_snapshot_packets(states, epoch, name=name, sat_names=sat_names)
        return eclipse_constellation_packets(
            states, epoch, duration, step, name=name, sat_names=sat_names,
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
            station, states, epoch, duration, step, name=name, sat_names=sat_names,
        )

    if layer_type == "sensor":
        sensor = params.get("_sensor", _DEFAULT_SENSOR)
        return sensor_footprint_packets(
            states, epoch, duration, step, sensor, name=name, sat_names=sat_names,
        )

    if layer_type == "isl":
        link = params.get("_link_config", _DEFAULT_LINK_CONFIG)
        max_range = params.get("max_range_km", 5000.0)
        capped = states[:_MAX_TOPOLOGY_SATS]
        if len(states) > _MAX_TOPOLOGY_SATS:
            params["_capped_from"] = len(states)
        dur_s = duration.total_seconds()
        step_s = step.total_seconds()
        return isl_topology_packets(
            capped, epoch, link, epoch, dur_s, step_s,
            max_range_km=max_range, name=name, sat_names=sat_names,
        )

    if layer_type == "fragility":
        import math as _math
        link = params.get("_link_config", _DEFAULT_LINK_CONFIG)
        capped = states[:_MAX_TOPOLOGY_SATS]
        if len(states) > _MAX_TOPOLOGY_SATS:
            params["_capped_from"] = len(states)
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
            capped, epoch, link, n_rad_s, ctrl_s, dur_s, step_s, name=name, sat_names=sat_names,
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
            states, curve, epoch, max_dur_s, hazard_step_s, name=name, sat_names=sat_names,
        )

    if layer_type == "network_eclipse":
        link = params.get("_link_config", _DEFAULT_LINK_CONFIG)
        max_range = params.get("max_range_km", 5000.0)
        capped = states[:_MAX_TOPOLOGY_SATS]
        if len(states) > _MAX_TOPOLOGY_SATS:
            params["_capped_from"] = len(states)
        dur_s = duration.total_seconds()
        step_s = step.total_seconds()
        return network_eclipse_packets(
            capped, link, epoch, dur_s, step_s,
            max_range_km=max_range, name=name, sat_names=sat_names,
        )

    if layer_type == "coverage_connectivity":
        link = params.get("_link_config", _DEFAULT_LINK_CONFIG)
        capped = states[:_MAX_TOPOLOGY_SATS]
        if len(states) > _MAX_TOPOLOGY_SATS:
            params["_capped_from"] = len(states)
        dur_s = duration.total_seconds()
        step_s = step.total_seconds()
        return coverage_connectivity_packets(
            capped, link, epoch, dur_s, step_s, name=name,  # grid-based, no per-sat names
        )

    if layer_type == "precession":
        subset = states[:_MAX_PRECESSION_SATS]
        if len(states) > _MAX_PRECESSION_SATS:
            params["_capped_from"] = len(states)
        prec_duration = timedelta(days=7)
        prec_step = timedelta(minutes=15)
        return constellation_packets(
            subset, epoch, prec_duration, prec_step, name=name, sat_names=sat_names,
        )

    if layer_type == "kessler_heatmap":
        return kessler_heatmap_packets(states, epoch, duration, step, name=name, sat_names=sat_names)

    if layer_type == "conjunction_hazard":
        capped = states[:_MAX_TOPOLOGY_SATS]
        if len(states) > _MAX_TOPOLOGY_SATS:
            params["_capped_from"] = len(states)
        return conjunction_hazard_packets(
            capped, epoch, duration, step, name=name, sat_names=sat_names,
        )

    if layer_type == "conjunction":
        if len(states) < 2:
            return [{"id": "document", "name": name, "version": "1.0"}]
        state_a = states[0]
        state_b = states[len(states) // 2]
        name_a = sat_names[0] if sat_names else "Sat-A"
        name_b = sat_names[len(states) // 2] if sat_names else "Sat-B"
        window = timedelta(minutes=30)
        conj_step = timedelta(seconds=10)
        return conjunction_replay_packets(
            state_a, state_b, epoch, window, conj_step,
            name_a=name_a, name_b=name_b,
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
            states, epoch, duration, step, name=name, sat_names=sat_names,
        )

    if layer_type == "beta_angle":
        return beta_angle_packets(states, epoch, name=name, sat_names=sat_names)

    if layer_type == "deorbit":
        return deorbit_compliance_packets(
            states, epoch, name=name, sat_names=sat_names,
        )

    if layer_type == "station_keeping":
        drag = params.get("_drag_config", _DEFAULT_DRAG)
        density_fn = _make_density_func(epoch)
        return station_keeping_packets(
            states, epoch, drag_config=drag, density_func=density_fn, name=name, sat_names=sat_names,
        )

    if layer_type == "cascade_sir":
        return cascade_evolution_packets(
            states, epoch, duration, step, name=name, sat_names=sat_names,
        )

    if layer_type == "relative_motion":
        if len(states) < 2:
            return [{"id": "document", "name": name, "version": "1.0"}]
        return relative_motion_packets(
            states[0], states[1], epoch, duration, step, name=name, sat_names=sat_names,
        )

    if layer_type == "maintenance":
        drag = params.get("_drag_config", _DEFAULT_DRAG)
        density_fn = _make_density_func(epoch)
        return maintenance_schedule_packets(
            states, epoch, drag_config=drag, density_func=density_fn, name=name, sat_names=sat_names,
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
        self.duration = _DEFAULT_DURATION
        self.step = _DEFAULT_STEP
        self._lock = threading.RLock()
        self.constraints: list[dict[str, Any]] = []

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
        source_layer_id: str = "",
        visible: bool = True,
        sat_names: list[str] | None = None,
    ) -> str:
        """Add a visualization layer. Returns the layer ID."""
        if mode is None:
            mode = "snapshot" if len(states) > _SNAPSHOT_THRESHOLD else "animated"

        # Reserve ID under lock, then generate CZML outside (can be slow)
        with self._lock:
            layer_id = self._next_id()

        # Copy params so _generate_czml mutations don't leak to caller
        gen_params = dict(params)
        czml = _generate_czml(layer_type, mode, states, self.epoch, name, gen_params, sat_names=sat_names)
        metrics = _compute_metrics(layer_type, states, self.epoch, gen_params)

        with self._lock:
            self.layers[layer_id] = LayerState(
                layer_id=layer_id,
                name=name,
                category=category,
                layer_type=layer_type,
                mode=mode,
                visible=visible,
                states=states,
                params=gen_params,
                czml=czml,
                source_layer_id=source_layer_id,
                sat_names=sat_names,
                metrics=metrics,
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
        with self._lock:
            if layer_id not in self.layers:
                raise KeyError(f"Layer not found: {layer_id}")
            del self.layers[layer_id]

    def update_layer(
        self,
        layer_id: str,
        mode: str | None = None,
        visible: bool | None = None,
        name: str | None = None,
    ) -> None:
        """Update layer mode, visibility, and/or name. Raises KeyError if not found."""
        regen_args = None
        with self._lock:
            if layer_id not in self.layers:
                raise KeyError(f"Layer not found: {layer_id}")
            layer = self.layers[layer_id]
            if visible is not None:
                layer.visible = visible
            if name is not None:
                layer.name = name
            if mode is not None and mode != layer.mode:
                layer.mode = mode
                regen_args = (layer.layer_type, mode, layer.states,
                              self.epoch, layer.name, layer.params, layer.sat_names)
        if regen_args is not None:
            new_czml = _generate_czml(*regen_args)
            with self._lock:
                if layer_id in self.layers:
                    self.layers[layer_id].czml = new_czml

    def get_state(self) -> dict[str, Any]:
        """Return all layer metadata (no CZML data)."""
        with self._lock:
            layers_info = []
            for layer in self.layers.values():
                info: dict[str, Any] = {
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
                }
                info["editable"] = layer.layer_type == "walker"
                if layer.metrics is not None:
                    info["metrics"] = layer.metrics
                if "_capped_from" in layer.params:
                    info["capped_from"] = layer.params["_capped_from"]
                if layer.layer_type in _LEGENDS:
                    info["legend"] = _LEGENDS[layer.layer_type]
                layers_info.append(info)
            return {
                "epoch": self.epoch.isoformat(),
                "duration_s": self.duration.total_seconds(),
                "step_s": self.step.total_seconds(),
                "layers": layers_info,
            }

    def recompute_analysis(self, layer_id: str, params: dict[str, Any]) -> None:
        """Recompute an analysis layer with updated params."""
        with self._lock:
            if layer_id not in self.layers:
                raise KeyError(f"Layer not found: {layer_id}")
            layer = self.layers[layer_id]
            # Merge new params (preserve internal _ params)
            internal = {k: v for k, v in layer.params.items() if k.startswith("_")}
            new_params = {**params, **internal}
            layer_type = layer.layer_type
            mode = layer.mode
            states = layer.states
            name = layer.name
            layer_sat_names = layer.sat_names

        # Generate CZML outside lock (can be slow)
        new_czml = _generate_czml(layer_type, mode, states, self.epoch, name, new_params, sat_names=layer_sat_names)

        # Swap atomically (BUG-015)
        with self._lock:
            if layer_id in self.layers:
                self.layers[layer_id].params = new_params
                self.layers[layer_id].czml = new_czml

    def reconfigure_constellation(
        self, layer_id: str, new_params: dict[str, Any],
    ) -> None:
        """Reconfigure a walker constellation with new parameters.

        Merges *new_params* into the existing walker params, regenerates the
        constellation (ShellConfig -> generate_walker_shell -> derive_orbital_state),
        updates the layer states/czml/sat_names, and cascades the change to any
        analysis layers that use this constellation as their source.

        Raises KeyError if the layer doesn't exist.
        Raises ValueError if the layer is not a walker constellation.
        """
        with self._lock:
            if layer_id not in self.layers:
                raise KeyError(f"Layer not found: {layer_id}")
            layer = self.layers[layer_id]
            if layer.layer_type != "walker":
                raise ValueError(
                    f"Only walker layers can be reconfigured, got {layer.layer_type}"
                )
            # Merge: new values override old, preserve anything not specified
            merged = {**layer.params, **new_params}

        # Validate bounds (same as POST /api/constellation handler)
        alt = merged.get("altitude_km", 550)
        inc = merged.get("inclination_deg", 53)
        nplanes = merged.get("num_planes", 6)
        spp = merged.get("sats_per_plane", 10)
        if not (100 <= alt <= 100000):
            raise ValueError(f"altitude_km must be in [100, 100000], got {alt}")
        if not (0 <= inc <= 180):
            raise ValueError(f"inclination_deg must be in [0, 180], got {inc}")
        if not (1 <= nplanes <= 100):
            raise ValueError(f"num_planes must be in [1, 100], got {nplanes}")
        if not (1 <= spp <= 100):
            raise ValueError(f"sats_per_plane must be in [1, 100], got {spp}")
        if nplanes * spp > 10000:
            raise ValueError(f"Total satellites ({nplanes * spp}) exceeds 10000")

        # Build ShellConfig from merged params (outside lock — pure computation)
        config = ShellConfig(
            altitude_km=merged["altitude_km"],
            inclination_deg=merged["inclination_deg"],
            num_planes=merged["num_planes"],
            sats_per_plane=merged["sats_per_plane"],
            phase_factor=merged.get("phase_factor", 1),
            raan_offset_deg=merged.get("raan_offset_deg", 0.0),
            shell_name=merged.get("shell_name", "Walker"),
        )
        sats = generate_walker_shell(config)
        sat_names = [s.name for s in sats]
        states = [
            derive_orbital_state(s, self.epoch, include_j2=True)
            for s in sats
        ]

        # Regenerate CZML for the constellation
        with self._lock:
            if layer_id not in self.layers:
                return
            layer = self.layers[layer_id]
            mode = layer.mode
            name = layer.name

        new_czml = _generate_czml(
            "walker", mode, states, self.epoch, name, merged,
            sat_names=sat_names,
        )

        # Update the constellation layer atomically
        with self._lock:
            if layer_id not in self.layers:
                return
            layer = self.layers[layer_id]
            layer.states = states
            layer.params = merged
            layer.sat_names = sat_names
            layer.czml = new_czml

        # Cascade: regenerate any analysis layers sourced from this constellation
        with self._lock:
            dependent_ids = [
                lid for lid, lyr in self.layers.items()
                if lyr.source_layer_id == layer_id
            ]

        for dep_id in dependent_ids:
            with self._lock:
                if dep_id not in self.layers:
                    continue
                dep = self.layers[dep_id]
                dep_type = dep.layer_type
                dep_mode = dep.mode
                dep_name = dep.name
                dep_params = dep.params

            dep_czml = _generate_czml(
                dep_type, dep_mode, states, self.epoch, dep_name,
                dep_params, sat_names=sat_names,
            )
            dep_metrics = _compute_metrics(dep_type, states, self.epoch, dep_params)
            with self._lock:
                if dep_id in self.layers:
                    self.layers[dep_id].states = states
                    self.layers[dep_id].sat_names = sat_names
                    self.layers[dep_id].czml = dep_czml
                    self.layers[dep_id].metrics = dep_metrics

    def generate_report(
        self, name: str = "Constellation Report", description: str = "",
    ) -> str:
        """Generate a self-contained HTML report of the current session.

        Includes: scenario metadata, layer parameters, metrics tables,
        constraint pass/fail results.
        """
        from html import escape
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        parts: list[str] = []
        parts.append(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{escape(name)}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; color: #333; }}
h1 {{ color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 8px; }}
h2 {{ color: #2d5f8a; margin-top: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
th, td {{ padding: 6px 12px; border: 1px solid #ddd; text-align: left; font-size: 13px; }}
th {{ background: #f0f4f8; font-weight: 600; }}
.pass {{ color: #2a7; }} .fail {{ color: #c44; }}
.meta {{ color: #666; font-size: 12px; margin-bottom: 16px; }}
</style></head><body>
<h1>{escape(name)}</h1>
<div class="meta">Generated: {ts} | Epoch: {self.epoch.isoformat()}</div>""")
        if description:
            parts.append(f"<p>{escape(description)}</p>")

        with self._lock:
            # Layers
            if not self.layers:
                parts.append("<p>No layers configured.</p>")
            for layer in self.layers.values():
                parts.append(f"<h2>{escape(layer.name)}</h2>")
                parts.append(f"<p>Type: {layer.layer_type} | Category: {layer.category}</p>")
                # Parameters
                visible_params = {k: v for k, v in layer.params.items() if not k.startswith("_")}
                if visible_params:
                    parts.append("<table><tr><th>Parameter</th><th>Value</th></tr>")
                    for k, v in visible_params.items():
                        parts.append(f"<tr><td>{escape(k)}</td><td>{escape(str(v))}</td></tr>")
                    parts.append("</table>")
                # Metrics
                if layer.metrics:
                    parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
                    for k, v in layer.metrics.items():
                        parts.append(f"<tr><td>{escape(k)}</td><td>{escape(str(v))}</td></tr>")
                    parts.append("</table>")

            # Constraints
            if self.constraints:
                parts.append("<h2>Constraints</h2>")
                # Find first constellation for evaluation
                const_id = None
                for lid, layer in self.layers.items():
                    if layer.category == "Constellation":
                        const_id = lid
                        break
                if const_id:
                    results = self.evaluate_constraints(const_id)
                    passed = sum(1 for r in results if r["passed"])
                    parts.append(f"<p>{passed}/{len(results)} constraints met</p>")
                    parts.append("<table><tr><th>Metric</th><th>Operator</th><th>Threshold</th><th>Actual</th><th>Result</th></tr>")
                    for r in results:
                        icon = '<span class="pass">PASS</span>' if r["passed"] else '<span class="fail">FAIL</span>'
                        actual = str(r["actual"]) if r["actual"] is not None else "N/A"
                        parts.append(f'<tr><td>{escape(r["metric"])}</td><td>{escape(r["operator"])}</td>'
                                     f'<td>{escape(str(r["threshold"]))}</td><td>{escape(actual)}</td><td>{icon}</td></tr>')
                    parts.append("</table>")

        parts.append("</body></html>")
        return "\n".join(parts)

    def add_constraint(self, constraint: dict[str, Any]) -> None:
        """Add a metric constraint {metric, operator, threshold}.

        Raises ValueError if required keys are missing or invalid.
        """
        if "metric" not in constraint:
            raise ValueError("Constraint must include 'metric' key")
        if "operator" not in constraint or constraint["operator"] not in _VALID_CONSTRAINT_OPS:
            raise ValueError(
                f"Constraint 'operator' must be one of {_VALID_CONSTRAINT_OPS}, "
                f"got {constraint.get('operator')!r}"
            )
        if "threshold" not in constraint or not isinstance(constraint["threshold"], (int, float)):
            raise ValueError("Constraint 'threshold' must be numeric")
        with self._lock:
            self.constraints.append(constraint)

    def evaluate_constraints(
        self, constellation_layer_id: str,
    ) -> list[dict[str, Any]]:
        """Evaluate all constraints against a constellation's analysis metrics.

        Returns list of {metric, operator, threshold, actual, passed}.
        """
        _ops = {
            ">=": lambda a, t: a >= t,
            "<=": lambda a, t: a <= t,
            ">": lambda a, t: a > t,
            "<": lambda a, t: a < t,
            "==": lambda a, t: abs(a - t) < 1e-9,
        }
        with self._lock:
            # Collect all metrics from analysis layers sourced from this constellation
            merged: dict[str, Any] = {}
            for layer in self.layers.values():
                if layer.source_layer_id == constellation_layer_id and layer.metrics:
                    for k, v in layer.metrics.items():
                        key = f"{layer.layer_type}_{k}"
                        merged[key] = v

            results = []
            for c in self.constraints:
                try:
                    metric_key = c["metric"]
                    op = c["operator"]
                    threshold = c["threshold"]
                except KeyError:
                    continue  # Skip malformed constraints
                actual = merged.get(metric_key)
                if actual is not None and isinstance(actual, (int, float)):
                    op_fn = _ops.get(op)
                    passed = op_fn(actual, threshold) if op_fn else False
                else:
                    passed = False
                results.append({
                    "metric": metric_key,
                    "operator": op,
                    "threshold": threshold,
                    "actual": actual,
                    "passed": passed,
                })
            return results

    def compare_layers(
        self, layer_id_a: str, layer_id_b: str,
    ) -> dict[str, Any]:
        """Compare two constellation layers on their analysis metrics.

        Collects metrics from analysis layers sourced from each constellation,
        computes delta (B - A) for all numeric metric values.

        Raises KeyError if either layer_id not found.
        """
        with self._lock:
            layer_a = self.layers[layer_id_a]
            layer_b = self.layers[layer_id_b]

            def _collect_metrics(source_id: str) -> dict[str, Any]:
                merged: dict[str, Any] = {}
                for layer in self.layers.values():
                    if layer.source_layer_id == source_id and layer.metrics:
                        for k, v in layer.metrics.items():
                            key = f"{layer.layer_type}_{k}"
                            merged[key] = v
                return merged

            metrics_a = _collect_metrics(layer_id_a)
            metrics_b = _collect_metrics(layer_id_b)

            # Compute delta for numeric values — only when both sides have the metric
            common_keys = set(metrics_a.keys()) & set(metrics_b.keys())
            delta: dict[str, float] = {}
            for k in common_keys:
                va = metrics_a[k]
                vb = metrics_b[k]
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    delta[k] = round(vb - va, 6)

            return {
                "config_a": {
                    "name": layer_a.name,
                    "params": layer_a.params,
                    "metrics": metrics_a,
                },
                "config_b": {
                    "name": layer_b.name,
                    "params": layer_b.params,
                    "metrics": metrics_b,
                },
                "delta": delta,
            }

    def run_sweep(
        self,
        base_params: dict[str, Any],
        sweep_param: str,
        sweep_min: float,
        sweep_max: float,
        sweep_step: float,
        metric_type: str,
    ) -> list[dict[str, Any]]:
        """Run a parameter sweep computing metrics for each configuration.

        Generates a Walker constellation for each sweep value, computes
        metrics of the specified type, returns list of {params, metrics}.
        """
        if sweep_step <= 0:
            raise ValueError(f"sweep_step must be > 0, got {sweep_step}")
        if sweep_min > sweep_max:
            raise ValueError(
                f"sweep_min ({sweep_min}) must be <= sweep_max ({sweep_max})"
            )
        n_iterations = int((sweep_max - sweep_min) / sweep_step) + 1
        if n_iterations > _MAX_SWEEP_ITERATIONS:
            raise ValueError(
                f"Sweep would require {n_iterations} iterations "
                f"(max {_MAX_SWEEP_ITERATIONS})"
            )

        results: list[dict[str, Any]] = []
        for i in range(n_iterations):
            val = sweep_min + i * sweep_step
            if val > sweep_max + 1e-9:
                break
            params = dict(base_params)
            params[sweep_param] = round(val, 6) if isinstance(val, float) else val
            # Generate constellation
            shell = ShellConfig(
                altitude_km=params.get("altitude_km", 550),
                inclination_deg=params.get("inclination_deg", 53),
                num_planes=int(params.get("num_planes", 6)),
                sats_per_plane=int(params.get("sats_per_plane", 10)),
                phase_factor=int(params.get("phase_factor", 0)),
                raan_offset_deg=params.get("raan_offset_deg", 0),
                shell_name="Sweep",
            )
            sats = generate_walker_shell(shell)
            states = [derive_orbital_state(s, self.epoch) for s in sats]
            metrics = _compute_metrics(metric_type, states, self.epoch, params)
            results.append({
                "params": params,
                "metrics": metrics or {},
            })
        return results

    def get_satellite_table(self, layer_id: str) -> dict[str, Any]:
        """Return tabular per-satellite data for a constellation layer.

        Columns: name, plane, altitude_km, inclination_deg, raan_deg,
        period_min, beta_angle_deg, eclipse_pct.

        Raises KeyError if layer_id not found.
        """
        import math
        with self._lock:
            layer = self.layers[layer_id]
            states = layer.states or []
            names = layer.sat_names
            params = layer.params
            sats_per_plane = params.get("sats_per_plane", len(states))
            rows = []
            for idx, state in enumerate(states):
                name = names[idx] if names and idx < len(names) else f"Sat-{idx}"
                alt_km = (state.semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
                inc_deg = math.degrees(state.inclination_rad)
                raan_deg = math.degrees(state.raan_rad)
                period_s = 2.0 * math.pi / state.mean_motion_rad_s if state.mean_motion_rad_s > 0 else 0.0
                period_min = period_s / 60.0
                plane = idx // sats_per_plane if sats_per_plane > 0 else 0
                try:
                    beta_deg = compute_beta_angle(state.raan_rad, state.inclination_rad, self.epoch)
                except Exception:
                    beta_deg = 0.0
                try:
                    es = compute_eclipse_statistics(state, self.epoch, _DEFAULT_DURATION, _DEFAULT_STEP)
                    eclipse_pct = round(es.eclipse_fraction * 100.0, 1)
                except Exception:
                    eclipse_pct = 0.0
                rows.append({
                    "name": name,
                    "plane": plane,
                    "altitude_km": round(alt_km, 2),
                    "inclination_deg": round(inc_deg, 2),
                    "raan_deg": round(raan_deg, 2),
                    "period_min": round(period_min, 2),
                    "beta_angle_deg": round(beta_deg, 1),
                    "eclipse_pct": eclipse_pct,
                })
            return {
                "columns": [
                    "name", "plane", "altitude_km", "inclination_deg",
                    "raan_deg", "period_min", "beta_angle_deg", "eclipse_pct",
                ],
                "rows": rows,
            }

    def save_session(
        self,
        name: str = "Untitled",
        description: str = "",
    ) -> dict[str, Any]:
        """Serialize current session state for save/restore.

        Args:
            name: Scenario name (shown in load dialog and recent list).
            description: Optional scenario description.
        """
        with self._lock:
            layers = []
            layer_ids = list(self.layers.keys())
            categories: dict[str, int] = {}
            for layer in self.layers.values():
                entry: dict[str, Any] = {
                    "name": layer.name,
                    "category": layer.category,
                    "layer_type": layer.layer_type,
                    "mode": layer.mode,
                    "visible": layer.visible,
                    "params": {
                        k: v for k, v in layer.params.items()
                        if not k.startswith("_")
                    },
                }
                if layer.sat_names is not None:
                    entry["sat_names"] = layer.sat_names
                if layer.metrics is not None:
                    entry["metrics"] = layer.metrics
                if layer.category == "Analysis" and layer.source_layer_id:
                    if layer.source_layer_id in layer_ids:
                        entry["source_layer_index"] = layer_ids.index(layer.source_layer_id)
                    else:
                        for idx, lid in enumerate(layer_ids):
                            other = self.layers[lid]
                            if other.category == "Constellation" and other.states is layer.states:
                                entry["source_layer_index"] = idx
                                break
                categories[layer.category] = categories.get(layer.category, 0) + 1
                layers.append(entry)
            return {
                "version": 1,
                "name": name,
                "description": description,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "epoch": self.epoch.isoformat(),
                "duration_s": self.duration.total_seconds(),
                "step_s": self.step.total_seconds(),
                "layers": layers,
                "layer_summary": {
                    "total": len(layers),
                    **categories,
                },
                "constraints": list(self.constraints),
            }

    def load_session(self, session_data: dict[str, Any]) -> int:
        """Restore session state from save_session() output.

        Clears existing layers, then restores in three passes:
        Pass 1: constellation layers (walker/celestrak)
        Pass 1.5: ground station layers
        Pass 2: analysis layers (with source constellation references)

        The entire operation holds the lock to prevent TOCTOU races
        (RLock allows re-entry from add_layer/add_ground_station).

        Returns the number of layers restored.
        """
        layers_data = session_data.get("layers", [])
        if not isinstance(layers_data, list):
            raise ValueError("Session 'layers' must be a list")
        layers_data = [ld for ld in layers_data if isinstance(ld, dict)]

        with self._lock:
            # Clear existing layers atomically with restore
            self.layers.clear()
            self._counter = 0
            self.constraints = list(session_data.get("constraints", []))
            if "duration_s" in session_data:
                self.duration = timedelta(seconds=session_data["duration_s"])
            if "step_s" in session_data:
                self.step = timedelta(seconds=session_data["step_s"])

            return self._restore_layers(layers_data)

    def _restore_layers(self, layers_data: list[dict[str, Any]]) -> int:
        """Restore layers from session data. Caller must hold self._lock."""
        # Pass 1: Restore constellation layers (walker and celestrak)
        restored = 0
        restored_layers: list[str] = []
        for layer_data in layers_data:
            lt = layer_data.get("layer_type", "")
            params = layer_data.get("params", {})
            if lt in ("walker", "celestrak"):
                try:
                    config = ShellConfig(
                        altitude_km=params.get("altitude_km", 550),
                        inclination_deg=params.get("inclination_deg", 53),
                        num_planes=params.get("num_planes", 6),
                        sats_per_plane=params.get("sats_per_plane", 10),
                        phase_factor=params.get("phase_factor", 1),
                        raan_offset_deg=params.get("raan_offset_deg", 0),
                        shell_name=params.get("shell_name", "Walker"),
                    )
                    sats = generate_walker_shell(config)
                    restored_sat_names = layer_data.get("sat_names") or [s.name for s in sats]
                    states = [
                        derive_orbital_state(s, self.epoch, include_j2=True)
                        for s in sats
                    ]
                    layer_id = self.add_layer(
                        name=layer_data.get("name", f"Constellation:{config.shell_name}"),
                        category=layer_data.get("category", "Constellation"),
                        layer_type=lt,
                        states=states,
                        params=params,
                        mode=layer_data.get("mode"),
                        visible=layer_data.get("visible", True),
                        sat_names=restored_sat_names,
                    )
                    restored_layers.append(layer_id)
                    restored += 1
                except (KeyError, TypeError, ValueError):
                    restored_layers.append("")
            else:
                restored_layers.append("")

        # Pass 1.5: Restore ground station layers
        for idx, layer_data in enumerate(layers_data):
            lt = layer_data.get("layer_type", "")
            if lt != "ground_station":
                continue
            params = layer_data.get("params", {})
            gs_source_states: list[OrbitalState] = []
            for lid in restored_layers:
                if lid and lid in self.layers:
                    layer = self.layers[lid]
                    if layer.category == "Constellation":
                        gs_source_states = layer.states
                        break
            try:
                gs_lid = self.add_ground_station(
                    name=params.get("name", layer_data.get("name", "Station")),
                    lat_deg=params.get("lat_deg", 0.0),
                    lon_deg=params.get("lon_deg", 0.0),
                    source_states=gs_source_states[:_GROUND_STATION_SAT_LIMIT],
                )
                restored_layers[idx] = gs_lid
                restored += 1
            except (KeyError, TypeError, ValueError):
                pass

        # Pass 2: Restore analysis layers using source constellation reference
        for idx, layer_data in enumerate(layers_data):
            lt = layer_data.get("layer_type", "")
            if lt in ("walker", "celestrak", "ground_station"):
                continue
            params = layer_data.get("params", {})
            source_idx = layer_data.get("source_layer_index")
            source_states: list[OrbitalState] = []
            source_sat_names: list[str] | None = None
            if source_idx is not None and 0 <= source_idx < len(restored_layers):
                source_lid = restored_layers[source_idx]
                if source_lid and source_lid in self.layers:
                    source_states = self.layers[source_lid].states
                    source_sat_names = self.layers[source_lid].sat_names
            if not source_states:
                for lid in restored_layers:
                    if lid and lid in self.layers:
                        layer = self.layers[lid]
                        if layer.category == "Constellation":
                            source_states = layer.states
                            source_sat_names = layer.sat_names
                            break
            if not source_states:
                continue
            try:
                self.add_layer(
                    name=layer_data.get("name", f"Analysis:{lt}"),
                    category=layer_data.get("category", "Analysis"),
                    layer_type=lt,
                    states=source_states,
                    params=params,
                    mode=layer_data.get("mode"),
                    visible=layer_data.get("visible", True),
                    sat_names=source_sat_names,
                )
                restored += 1
            except (KeyError, TypeError, ValueError, ArithmeticError):
                pass

        return restored

    def export_all_czml(self, visible_only: bool = True) -> list[dict[str, Any]]:
        """Merge all (visible) layers into a single CZML document.

        Entity IDs are prefixed with the layer name to prevent collisions.
        The merged clock encompasses all layers' time ranges.
        Packets are shallow-copied to prevent aliasing.
        """
        doc_packet: dict[str, Any] = {
            "id": "document",
            "name": "Constellation Export",
            "version": "1.0",
        }
        merged: list[dict[str, Any]] = [doc_packet]
        merged_clock: dict[str, Any] | None = None
        with self._lock:
            for layer in self.layers.values():
                if visible_only and not layer.visible:
                    continue
                # Sanitize layer name for use as ID prefix
                prefix = layer.name.replace(" ", "_").replace("/", "_").replace(":", "_")
                for pkt in layer.czml:
                    if pkt.get("id") == "document":
                        if "clock" in pkt:
                            merged_clock = _merge_clocks(merged_clock, pkt["clock"])
                        continue
                    # Shallow copy + prefix entity ID to avoid collisions
                    copied = dict(pkt)
                    if "id" in copied:
                        copied["id"] = f"{prefix}/{copied['id']}"
                    merged.append(copied)
        if merged_clock is not None:
            doc_packet["clock"] = merged_clock
        return merged

    def export_czml_layers(self, output_dir: str) -> int:
        """Export each layer's CZML to a separate file in output_dir.

        Returns the number of files written.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        exported = 0
        with self._lock:
            layer_snapshot = list(self.layers.values())
        for layer in layer_snapshot:
            safe_name = layer.name.replace("/", "_").replace(":", "_").replace(" ", "_")
            filename = f"{safe_name}.czml"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(layer.czml, f, indent=2)
            exported += 1
        return exported

    def get_czml(self, layer_id: str) -> list[dict[str, Any]]:
        """Return CZML packets for a layer. Raises KeyError if not found."""
        with self._lock:
            if layer_id not in self.layers:
                raise KeyError(f"Layer not found: {layer_id}")
            return self.layers[layer_id].czml


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a new thread."""
    daemon_threads = True


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
        port = self.server.server_address[1]
        self.send_header("Access-Control-Allow-Origin", f"http://localhost:{port}")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, data: Any, status: int = 200) -> None:
        self._set_headers(status, "application/json")
        self.wfile.write(json.dumps(_sanitize_for_json(data)).encode())

    def _error_response(self, status: int, message: str) -> None:
        self._json_response({"error": message}, status)

    _MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB

    def _read_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid Content-Length: {raw_length}")
        if length == 0:
            return {}
        if length < 0 or length > self._MAX_BODY_SIZE:
            raise ValueError(
                f"Request body too large: {length} bytes "
                f"(max {self._MAX_BODY_SIZE})"
            )
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

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a string for use in Content-Disposition filename."""
        import re
        return re.sub(r'[^a-zA-Z0-9._-]', '_', name)

    def do_GET(self) -> None:
        try:
            self._do_GET()
        except Exception:
            logger.exception("Unhandled error in GET %s", self.path)
            self._error_response(500, "Internal server error")

    def _do_GET(self) -> None:
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

        if base == "/api/export" and param:
            try:
                czml = self.layer_manager.get_czml(param)
                safe_name = self._sanitize_filename(param)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{safe_name}.czml"',
                )
                port = self.server.server_address[1]
                self.send_header("Access-Control-Allow-Origin", f"http://localhost:{port}")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()
                self.wfile.write(json.dumps(czml, indent=2).encode())
            except KeyError:
                self._error_response(404, f"Layer not found: {param}")
            return

        if base == "/api/ground-station-presets":
            self._json_response({"presets": _GROUND_STATION_PRESETS})
            return

        if base == "/api/export-all":
            merged = self.layer_manager.export_all_czml()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header(
                "Content-Disposition",
                'attachment; filename="constellation-all.czml"',
            )
            port = self.server.server_address[1]
            self.send_header("Access-Control-Allow-Origin", f"http://localhost:{port}")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(json.dumps(merged, indent=2).encode())
            return

        if base == "/api/table" and param:
            try:
                table = self.layer_manager.get_satellite_table(param)
                self._json_response(table)
            except KeyError:
                self._error_response(404, f"Layer not found: {param}")
            return

        if base == "/api/report":
            html_report = self.layer_manager.generate_report(
                name="Constellation Report",
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Disposition", 'attachment; filename="report.html"')
            port = self.server.server_address[1]
            self.send_header("Access-Control-Allow-Origin", f"http://localhost:{port}")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(html_report.encode())
            return

        self._error_response(404, "Not found")

    # --- POST ---

    def do_POST(self) -> None:
        try:
            self._do_POST()
        except Exception:
            logger.exception("Unhandled error in POST %s", self.path)
            self._error_response(500, "Internal server error")

    def _do_POST(self) -> None:
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

        if base == "/api/ground-station-network":
            self._handle_add_ground_station_network()
            return

        if base == "/api/session" and param == "save":
            try:
                body = self._read_body()
            except (ValueError, json.JSONDecodeError):
                body = {}
            save_name = body.get("name", "Untitled")
            save_desc = body.get("description", "")
            session = self.layer_manager.save_session(
                name=save_name, description=save_desc,
            )
            self._json_response({"session": session})
            return

        if base == "/api/session" and param == "load":
            self._handle_load_session()
            return

        if base == "/api/sweep":
            try:
                body = self._read_body()
            except (ValueError, json.JSONDecodeError) as e:
                self._error_response(400, f"Bad request body: {e}")
                return
            try:
                results = self.layer_manager.run_sweep(
                    base_params=body.get("base_params", {}),
                    sweep_param=body["sweep_param"],
                    sweep_min=float(body["sweep_min"]),
                    sweep_max=float(body["sweep_max"]),
                    sweep_step=float(body["sweep_step"]),
                    metric_type=body.get("metric_type", "coverage"),
                )
                self._json_response({"results": results})
            except (KeyError, ValueError, TypeError) as e:
                self._error_response(400, f"Sweep error: {e}")
            return

        if base == "/api/constraints" and param == "add":
            try:
                body = self._read_body()
            except (ValueError, json.JSONDecodeError) as e:
                self._error_response(400, f"Bad request: {e}")
                return
            self.layer_manager.add_constraint(body)
            self._json_response({"ok": True, "count": len(self.layer_manager.constraints)})
            return

        if base == "/api/constraints" and param == "evaluate":
            try:
                body = self._read_body()
            except (ValueError, json.JSONDecodeError) as e:
                self._error_response(400, f"Bad request: {e}")
                return
            try:
                results = self.layer_manager.evaluate_constraints(body["layer_id"])
                passed = sum(1 for r in results if r["passed"])
                self._json_response({
                    "results": results,
                    "summary": f"{passed}/{len(results)} constraints met",
                })
            except KeyError as e:
                self._error_response(404, f"Layer not found: {e}")
            return

        if base == "/api/constraints" and param == "list":
            with self.layer_manager._lock:
                constraints_copy = list(self.layer_manager.constraints)
            self._json_response({"constraints": constraints_copy})
            return

        if base == "/api/compare":
            try:
                body = self._read_body()
            except (ValueError, json.JSONDecodeError) as e:
                self._error_response(400, f"Bad request body: {e}")
                return
            try:
                result = self.layer_manager.compare_layers(
                    body["layer_a"], body["layer_b"],
                )
                self._json_response(result)
            except KeyError as e:
                self._error_response(404, f"Layer not found: {e}")
            return

        self._error_response(404, "Not found")

    def _handle_add_constellation(self) -> None:
        try:
            body = self._read_body()
        except (ValueError, json.JSONDecodeError) as e:
            self._error_response(400, f"Bad request body: {e}")
            return
        ctype = body.get("type")
        params = body.get("params", {})

        if ctype == "walker":
            try:
                alt = params["altitude_km"]
                inc = params["inclination_deg"]
                nplanes = params["num_planes"]
                spp = params["sats_per_plane"]
                if not (100 <= alt <= 100000):
                    raise ValueError(f"altitude_km must be in [100, 100000], got {alt}")
                if not (0 <= inc <= 180):
                    raise ValueError(f"inclination_deg must be in [0, 180], got {inc}")
                if not (1 <= nplanes <= 100):
                    raise ValueError(f"num_planes must be in [1, 100], got {nplanes}")
                if not (1 <= spp <= 100):
                    raise ValueError(f"sats_per_plane must be in [1, 100], got {spp}")
                if nplanes * spp > 10000:
                    raise ValueError(f"Total satellites ({nplanes * spp}) exceeds 10000")
                config = ShellConfig(
                    altitude_km=alt,
                    inclination_deg=inc,
                    num_planes=nplanes,
                    sats_per_plane=spp,
                    phase_factor=params.get("phase_factor", 1),
                    raan_offset_deg=params.get("raan_offset_deg", 0.0),
                    shell_name=params.get("shell_name", "Walker"),
                )
                sats = generate_walker_shell(config)
                sat_names = [s.name for s in sats]
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
                    sat_names=sat_names,
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
                sat_names = [s.name for s in sats]
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
                    sat_names=sat_names,
                )
                self._json_response({"layer_id": layer_id}, 201)
            except (ConnectionError, OSError, ValueError, KeyError, json.JSONDecodeError) as e:
                self._error_response(400, f"CelesTrak fetch failed: {e}")
            return

        self._error_response(400, f"Unknown constellation type: {ctype}")

    def _handle_add_analysis(self) -> None:
        try:
            body = self._read_body()
        except (ValueError, json.JSONDecodeError) as e:
            self._error_response(400, f"Bad request body: {e}")
            return
        analysis_type = body.get("type")
        source_layer_id = body.get("source_layer")
        params = body.get("params", {})

        with self.layer_manager._lock:
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
                source_layer_id=source_layer_id,
                sat_names=source.sat_names,
            )
            self._json_response({"layer_id": layer_id}, 201)
        except (ValueError, TypeError, KeyError, ArithmeticError) as e:
            logger.exception("Analysis generation failed")
            self._error_response(500, f"Analysis generation failed: {e}")

    def _handle_add_ground_station(self) -> None:
        try:
            body = self._read_body()
        except (ValueError, json.JSONDecodeError) as e:
            self._error_response(400, f"Bad request body: {e}")
            return
        name = body.get("name", "Station")
        lat = body.get("lat", 0.0)
        lon = body.get("lon", 0.0)

        if not (-90 <= lat <= 90):
            self._error_response(400, f"lat must be in [-90, 90], got {lat}")
            return
        if not (-180 <= lon <= 180):
            self._error_response(400, f"lon must be in [-180, 180], got {lon}")
            return

        # Find first constellation layer for access computation (under lock)
        source_states: list[OrbitalState] = []
        with self.layer_manager._lock:
            for layer in self.layer_manager.layers.values():
                if layer.category == "Constellation":
                    source_states = layer.states
                    break

        try:
            layer_id = self.layer_manager.add_ground_station(
                name=name,
                lat_deg=lat,
                lon_deg=lon,
                source_states=source_states[:_GROUND_STATION_SAT_LIMIT],
            )
            self._json_response({"layer_id": layer_id}, 201)
        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Ground station failed")
            self._error_response(500, f"Ground station failed: {e}")

    def _handle_add_ground_station_network(self) -> None:
        try:
            body = self._read_body()
        except (ValueError, json.JSONDecodeError) as e:
            self._error_response(400, f"Bad request body: {e}")
            return
        preset_name = body.get("preset", "")
        preset = None
        for p in _GROUND_STATION_PRESETS:
            if p["name"] == preset_name:
                preset = p
                break
        if preset is None:
            self._error_response(
                404,
                f"Unknown preset: {preset_name!r}. "
                f"Available: {[p['name'] for p in _GROUND_STATION_PRESETS]}",
            )
            return

        # Find first constellation layer for access computation (under lock)
        source_states: list[OrbitalState] = []
        with self.layer_manager._lock:
            for layer in self.layer_manager.layers.values():
                if layer.category == "Constellation":
                    source_states = layer.states
                    break

        added = 0
        for st in preset["stations"]:
            try:
                self.layer_manager.add_ground_station(
                    name=st["name"],
                    lat_deg=st["lat_deg"],
                    lon_deg=st["lon_deg"],
                    source_states=source_states[:_GROUND_STATION_SAT_LIMIT],
                )
                added += 1
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("Failed to add station %s: %s", st["name"], e)
        self._json_response({"added": added, "preset": preset_name}, 201)

    def _handle_load_session(self) -> None:
        try:
            body = self._read_body()
        except (ValueError, json.JSONDecodeError) as e:
            self._error_response(400, f"Bad request body: {e}")
            return
        session = body.get("session", {})
        try:
            restored = self.layer_manager.load_session(session)
        except ValueError as e:
            self._error_response(400, str(e))
            return
        self._json_response({"restored": restored})

    # --- PUT ---

    def do_PUT(self) -> None:
        try:
            self._do_PUT()
        except Exception:
            logger.exception("Unhandled error in PUT %s", self.path)
            self._error_response(500, "Internal server error")

    def _do_PUT(self) -> None:
        base, param = self._route_path()

        try:
            body = self._read_body()
        except (ValueError, json.JSONDecodeError) as e:
            self._error_response(400, f"Bad request body: {e}")
            return

        if base == "/api/layer" and param:
            try:
                self.layer_manager.update_layer(
                    param,
                    mode=body.get("mode"),
                    visible=body.get("visible"),
                    name=body.get("name"),
                )
                with self.layer_manager._lock:
                    layer = self.layer_manager.layers[param]
                    self._json_response({
                        "layer_id": layer.layer_id,
                        "mode": layer.mode,
                        "visible": layer.visible,
                    })
            except KeyError:
                self._error_response(404, f"Layer not found: {param}")
            return

        if base == "/api/constellation" and param:
            params = body.get("params", {})
            try:
                self.layer_manager.reconfigure_constellation(param, params)
                self._json_response({"status": "reconfigured", "layer_id": param})
            except KeyError:
                self._error_response(404, f"Layer not found: {param}")
            except ValueError as e:
                self._error_response(400, str(e))
            return

        if base == "/api/analysis" and param:
            params = body.get("params", {})
            try:
                self.layer_manager.recompute_analysis(param, params)
                self._json_response({"status": "recomputed", "layer_id": param})
            except KeyError:
                self._error_response(404, f"Layer not found: {param}")
            except (ValueError, TypeError, ArithmeticError) as e:
                self._error_response(500, f"Recomputation failed: {e}")
            return

        if base == "/api/settings":
            dur = body.get("duration_s")
            step = body.get("step_s")
            if dur is not None:
                if not isinstance(dur, (int, float)) or dur <= 0 or dur > 604800:
                    self._error_response(
                        400, f"duration_s must be in (0, 604800], got {dur}",
                    )
                    return
                self.layer_manager.duration = timedelta(seconds=dur)
            if step is not None:
                if not isinstance(step, (int, float)) or step <= 0 or step > 86400:
                    self._error_response(
                        400, f"step_s must be in (0, 86400], got {step}",
                    )
                    return
                self.layer_manager.step = timedelta(seconds=step)
            self._json_response({
                "duration_s": self.layer_manager.duration.total_seconds(),
                "step_s": self.layer_manager.step.total_seconds(),
            })
            return

        self._error_response(404, "Not found")

    # --- DELETE ---

    def do_DELETE(self) -> None:
        try:
            self._do_DELETE()
        except Exception:
            logger.exception("Unhandled error in DELETE %s", self.path)
            self._error_response(500, "Internal server error")

    def _do_DELETE(self) -> None:
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

    server = ThreadingHTTPServer(("localhost", port), handler)
    return server
