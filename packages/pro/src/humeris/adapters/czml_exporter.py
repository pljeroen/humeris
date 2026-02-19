# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""CZML exporter adapter for CesiumJS visualization.

Generates CZML JSON packets for animated satellite orbits, ground tracks,
and coverage heatmaps. Output opens directly in Cesium viewer.

Uses only stdlib json/math/datetime + domain imports.
"""

import json
import math
from datetime import datetime, timedelta, timezone

from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)
from humeris.domain.ground_track import GroundTrackPoint
from humeris.domain.coverage import CoveragePoint
from humeris.domain.numerical_propagation import NumericalPropagationResult
from humeris.domain.orbit_properties import (
    compute_orbital_velocity,
    check_sun_synchronous,
    compute_ltan,
    compute_ground_track_repeat,
)
from humeris.domain.eclipse import compute_beta_angle
from humeris.domain.atmosphere import atmospheric_density


_PLANE_COLORS = [
    (255, 82, 82, 255),     # Red
    (66, 165, 245, 255),    # Blue
    (102, 187, 106, 255),   # Green
    (255, 167, 38, 255),    # Orange
    (171, 71, 188, 255),    # Purple
    (38, 198, 218, 255),    # Cyan
    (255, 238, 88, 255),    # Yellow
    (236, 64, 122, 255),    # Pink
    (129, 199, 132, 255),   # Light green
    (79, 195, 247, 255),    # Light blue
    (149, 117, 205, 255),   # Light purple
    (255, 138, 101, 255),   # Light orange
]


def _assign_plane_indices(states: list[OrbitalState]) -> list[int]:
    """Group states by RAAN to assign plane indices for coloring."""
    raan_to_plane: dict[int, int] = {}
    result: list[int] = []
    for state in states:
        raan_key = round(math.degrees(state.raan_rad) * 10)
        if raan_key not in raan_to_plane:
            raan_to_plane[raan_key] = len(raan_to_plane)
        result.append(raan_to_plane[raan_key])
    return result


def _satellite_description(state: OrbitalState, epoch: datetime) -> str:
    """HTML description for Cesium info box with enriched orbital data."""
    alt_km = (state.semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
    incl_deg = math.degrees(state.inclination_rad)
    raan_deg = math.degrees(state.raan_rad)
    argp_deg = math.degrees(state.arg_perigee_rad)

    vel = compute_orbital_velocity(state)
    period_min = vel.orbital_period_s / 60.0
    vel_kms = vel.circular_velocity_ms / 1000.0

    beta_deg = compute_beta_angle(state.raan_rad, state.inclination_rad, epoch)

    ltan_hours = compute_ltan(state.raan_rad, epoch)
    ltan_h = int(ltan_hours)
    ltan_m = int((ltan_hours - ltan_h) * 60)

    sso = check_sun_synchronous(state)
    if sso.is_sun_synchronous:
        sso_text = "Yes"
    else:
        sso_text = f"No (error: {sso.error_deg_day:.2f}&deg;/day)"

    try:
        density = atmospheric_density(alt_km)
    except ValueError:
        density = 0.0

    gt = compute_ground_track_repeat(state)

    hdr = 'style="padding-top:6px;font-size:14px;color:#4fc3f7"'
    cell = 'style="padding:1px 8px 1px 0"'

    return (
        f'<table style="font-size:13px;border-collapse:collapse">'
        f'<tr><td colspan="2" {hdr}><b>Orbit</b></td></tr>'
        f"<tr><td {cell}>Altitude</td><td>{alt_km:.1f} km</td></tr>"
        f"<tr><td {cell}>Inclination</td><td>{incl_deg:.1f}&deg;</td></tr>"
        f"<tr><td {cell}>RAAN</td><td>{raan_deg:.1f}&deg;</td></tr>"
        f"<tr><td {cell}>Arg. perigee</td><td>{argp_deg:.1f}&deg;</td></tr>"
        f"<tr><td {cell}>Eccentricity</td><td>{state.eccentricity:.4f}</td></tr>"
        f"<tr><td {cell}>Period</td><td>{period_min:.1f} min</td></tr>"
        f'<tr><td colspan="2" {hdr}><b>Velocity</b></td></tr>'
        f"<tr><td {cell}>Orbital</td><td>{vel_kms:.2f} km/s</td></tr>"
        f"<tr><td {cell}>Ground speed</td><td>{vel.ground_speed_kmh:.0f} km/h</td></tr>"
        f'<tr><td colspan="2" {hdr}><b>Solar Geometry</b></td></tr>'
        f"<tr><td {cell}>Beta angle</td><td>{beta_deg:.1f}&deg;</td></tr>"
        f"<tr><td {cell}>LTAN</td><td>{ltan_h:02d}:{ltan_m:02d}</td></tr>"
        f"<tr><td {cell}>Sun-sync</td><td>{sso_text}</td></tr>"
        f'<tr><td colspan="2" {hdr}><b>Environment</b></td></tr>'
        f"<tr><td {cell}>Atm. density</td><td>{density:.2e} kg/m&sup3;</td></tr>"
        f'<tr><td colspan="2" {hdr}><b>Ground Track</b></td></tr>'
        f"<tr><td {cell}>Rev/day</td><td>{gt.revs_per_day:.2f}</td></tr>"
        f"<tr><td {cell}>Near repeat</td><td>{gt.near_repeat_revs} rev / {gt.near_repeat_days} day</td></tr>"
        f"</table>"
    )


def _iso(dt: datetime) -> str:
    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_step(step: timedelta) -> float:
    """Validate step size and return total seconds. Raises ValueError if <= 0."""
    step_seconds = step.total_seconds()
    if step_seconds <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return step_seconds


def _interpolation_degree(num_points: int) -> int:
    """LAGRANGE interpolation degree: min(5, num_points - 1), at least 1."""
    if num_points <= 1:
        return 1
    return min(5, num_points - 1)


def _document_packet(
    name: str,
    epoch: datetime | None = None,
    duration: timedelta | None = None,
) -> dict:
    pkt: dict = {
        "id": "document",
        "name": name,
        "version": "1.0",
    }
    if epoch is not None and duration is not None:
        end = epoch + duration
        pkt["clock"] = {
            "interval": f"{_iso(epoch)}/{_iso(end)}",
            "currentTime": _iso(epoch),
            "multiplier": 60,
            "range": "LOOP_START",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        }
    return pkt


def snapshot_packets(
    states: list[OrbitalState],
    epoch: datetime,
    name: str = "Constellation",
) -> list[dict]:
    """Static point representation: one position per satellite at epoch.

    No animation, no path, no label. ~200 bytes/sat instead of ~10KB.
    Suitable for dense constellations (thousands of satellites).

    Args:
        states: List of orbital states.
        epoch: Evaluation time for positions.
        name: Document name.

    Returns:
        List of CZML packets (document + N point packets).
    """
    packets: list[dict] = [_document_packet(name)]

    if not states:
        return packets

    plane_indices = _assign_plane_indices(states)

    for idx, state in enumerate(states):
        pos_eci, vel_eci = propagate_to(state, epoch)
        gmst = gmst_rad(epoch)
        pos_ecef, _ = eci_to_ecef(
            (pos_eci[0], pos_eci[1], pos_eci[2]),
            (vel_eci[0], vel_eci[1], vel_eci[2]),
            gmst,
        )
        lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)

        plane_color = _PLANE_COLORS[plane_indices[idx] % len(_PLANE_COLORS)]

        pkt: dict = {
            "id": f"snapshot-{idx}",
            "name": f"Sat-{idx}",
            "description": _satellite_description(state, epoch),
            "position": {
                "cartographicDegrees": [lon_deg, lat_deg, alt_m],
            },
            "point": {
                "pixelSize": 3,
                "color": {"rgba": list(plane_color)},
            },
        }
        packets.append(pkt)

    return packets


def constellation_packets(
    orbital_states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    name: str = "Constellation",
) -> list[dict]:
    """Document packet + 1 packet per satellite with time-varying position.

    Position: epoch + cartographicDegrees [seconds, lon, lat, height_m, ...]
    Interpolation: LAGRANGE degree 5.
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not orbital_states:
        return packets

    total_seconds = duration.total_seconds()
    step_seconds = _validate_step(step)
    num_steps = math.ceil(total_seconds / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    plane_indices = _assign_plane_indices(orbital_states)

    for idx, state in enumerate(orbital_states):
        coords: list[float] = []
        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            pos_eci, vel_eci = propagate_to(state, target_time)
            gmst = gmst_rad(target_time)
            pos_ecef, _ = eci_to_ecef(
                (pos_eci[0], pos_eci[1], pos_eci[2]),
                (vel_eci[0], vel_eci[1], vel_eci[2]),
                gmst,
            )
            lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])

        plane_color = _PLANE_COLORS[plane_indices[idx] % len(_PLANE_COLORS)]
        path_color = [plane_color[0], plane_color[1], plane_color[2], 128]

        sat_id = f"satellite-{idx}"
        pkt: dict = {
            "id": sat_id,
            "name": f"Sat-{idx}",
            "description": _satellite_description(state, epoch),
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "point": {
                "pixelSize": 5,
                "color": {"rgba": list(plane_color)},
            },
            "label": {
                "text": f"Sat-{idx}",
                "font": "11pt sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 200]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [12, 0]},
            },
            "path": {
                "leadTime": 3600,
                "trailTime": 3600,
                "resolution": 120,
                "material": {
                    "solidColor": {
                        "color": {"rgba": path_color},
                    },
                },
                "width": 1,
            },
        }
        packets.append(pkt)

    return packets


def ground_track_packets(
    track: list[GroundTrackPoint],
    name: str = "Ground Track",
    color: tuple[int, int, int, int] = (255, 255, 0, 255),
) -> list[dict]:
    """Document packet + polyline packet (clampToGround, positions from track)."""
    packets: list[dict] = [_document_packet(name)]

    if not track:
        return packets

    coords: list[float] = []
    for pt in track:
        coords.extend([pt.lon_deg, pt.lat_deg, 0.0])

    pkt: dict = {
        "id": "ground-track",
        "name": name,
        "polyline": {
            "positions": {
                "cartographicDegrees": coords,
            },
            "clampToGround": True,
            "material": {
                "solidColor": {
                    "color": {"rgba": list(color)},
                },
            },
            "width": 2,
        },
    }
    packets.append(pkt)

    return packets


def coverage_packets(
    points: list[CoveragePoint],
    lat_step_deg: float,
    lon_step_deg: float,
    name: str = "Coverage",
) -> list[dict]:
    """Document packet + 1 rectangle per grid point with visible_count > 0.

    Color: green intensity mapped from visible_count / max_count.
    Rectangle: wsenDegrees = [lon, lat, lon+step, lat+step].
    """
    packets: list[dict] = [_document_packet(name)]

    nonzero = [p for p in points if p.visible_count > 0]
    if not nonzero:
        return packets

    max_count = max(p.visible_count for p in nonzero)

    for idx, pt in enumerate(nonzero):
        intensity = int(255 * pt.visible_count / max_count)
        w = pt.lon_deg
        s = pt.lat_deg
        e = pt.lon_deg + lon_step_deg
        n = pt.lat_deg + lat_step_deg

        pkt: dict = {
            "id": f"coverage-{idx}",
            "name": f"Coverage ({pt.lat_deg}, {pt.lon_deg})",
            "rectangle": {
                "coordinates": {
                    "wsenDegrees": [w, s, e, n],
                },
                "fill": True,
                "material": {
                    "solidColor": {
                        "color": {"rgba": [0, intensity, 0, 180]},
                    },
                },
            },
        }
        packets.append(pkt)

    return packets


def constellation_packets_numerical(
    results: list[NumericalPropagationResult],
    name: str = "Constellation",
    sat_names: list[str] | None = None,
) -> list[dict]:
    """CZML packets from numerical propagation results.

    Same structure as constellation_packets but reads PropagationStep
    positions directly instead of re-propagating analytically.

    Args:
        results: List of NumericalPropagationResult (one per satellite).
        name: Document name.
        sat_names: Optional custom satellite names. Defaults to Sat-0, Sat-1, ...

    Returns:
        List of CZML packets (document + N satellites).
    """
    if not results:
        return [_document_packet(name)]

    if not results[0].steps:
        raise ValueError("First result has empty steps; cannot determine epoch")

    if sat_names is not None and len(sat_names) < len(results):
        raise ValueError(
            f"sat_names has {len(sat_names)} entries but there are {len(results)} results"
        )

    epoch = results[0].steps[0].time
    end_time = results[0].steps[-1].time
    duration = end_time - epoch

    packets: list[dict] = [_document_packet(name, epoch, duration)]

    for idx, result in enumerate(results):
        num_points = len(result.steps)
        interp_degree = _interpolation_degree(num_points)
        coords: list[float] = []
        for step_item in result.steps:
            t_offset = (step_item.time - epoch).total_seconds()
            gmst = gmst_rad(step_item.time)
            pos_ecef, _ = eci_to_ecef(
                step_item.position_eci,
                step_item.velocity_eci,
                gmst,
            )
            lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])

        sat_name = sat_names[idx] if sat_names else f"Sat-{idx}"
        sat_id = f"satellite-{idx}"
        pkt: dict = {
            "id": sat_id,
            "name": sat_name,
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "point": {
                "pixelSize": 5,
                "color": {"rgba": [255, 255, 255, 255]},
            },
            "label": {
                "text": sat_name,
                "font": "11pt sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 200]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [12, 0]},
            },
            "path": {
                "leadTime": 3600,
                "trailTime": 3600,
                "resolution": 120,
                "material": {
                    "solidColor": {
                        "color": {"rgba": [0, 255, 255, 128]},
                    },
                },
                "width": 1,
            },
        }
        packets.append(pkt)

    return packets


def write_czml(packets: list[dict], path: str) -> int:
    """Write JSON array to file. Returns len(packets)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(packets, f, indent=2, ensure_ascii=False)
    return len(packets)
