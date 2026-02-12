"""CZML exporter adapter for CesiumJS visualization.

Generates CZML JSON packets for animated satellite orbits, ground tracks,
and coverage heatmaps. Output opens directly in Cesium viewer.

Uses only stdlib json/math/datetime + domain imports.
"""

import json
import math
from datetime import datetime, timedelta

from constellation_generator.domain.propagation import OrbitalState, propagate_to
from constellation_generator.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)
from constellation_generator.domain.ground_track import GroundTrackPoint
from constellation_generator.domain.coverage import CoveragePoint


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


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
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        }
    return pkt


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
    step_seconds = step.total_seconds()
    num_steps = int(total_seconds / step_seconds) + 1

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

        sat_id = f"satellite-{idx}"
        pkt: dict = {
            "id": sat_id,
            "name": f"Sat-{idx}",
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 5,
            },
            "point": {
                "pixelSize": 5,
                "color": {"rgba": [255, 255, 255, 255]},
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
                        "color": {"rgba": [0, 255, 255, 128]},
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
                        "color": {"rgba": [0, intensity, 0, 128]},
                    },
                },
            },
        }
        packets.append(pkt)

    return packets


def write_czml(packets: list[dict], path: str) -> int:
    """Write JSON array to file. Returns len(packets)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(packets, f, indent=2, ensure_ascii=False)
    return len(packets)
