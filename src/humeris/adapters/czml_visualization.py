# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Advanced CZML visualization packets for CesiumJS.

Eclipse-aware coloring, sensor footprints, ground station access,
conjunction replay, coverage evolution, and J2 precession timelapse.

Uses only stdlib math/datetime + internal domain/adapter imports.
"""

import math
from datetime import datetime, timedelta

from humeris.domain.propagation import (
    OrbitalState,
    derive_orbital_state,
    propagate_to,
)
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)
from humeris.domain.solar import sun_position_eci
from humeris.domain.eclipse import is_eclipsed, EclipseType
from humeris.domain.sensor import SensorConfig, compute_swath_width
from humeris.domain.observation import GroundStation, compute_observation
from humeris.domain.access_windows import compute_access_windows
from humeris.domain.coverage import compute_coverage_snapshot
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.link_budget import LinkConfig, compute_link_budget
from humeris.domain.inter_satellite_links import compute_isl_topology
from humeris.domain.graph_analysis import compute_topology_resilience
from humeris.domain.revisit import compute_single_coverage_fraction
from humeris.domain.kessler_heatmap import compute_kessler_heatmap
from humeris.domain.conjunction import (
    screen_conjunctions,
    assess_conjunction,
)
from humeris.domain.hazard_reporting import (
    classify_hazard,
    HazardLevel,
)
from humeris.adapters.czml_exporter import (
    _PLANE_COLORS,
    _assign_plane_indices,
    _satellite_description,
    _document_packet,
    _iso,
    _validate_step,
    _interpolation_degree,
    constellation_packets,
)


_ECLIPSE_COLOR = (80, 80, 80, 255)

# Eclipse snapshot colors
_SUNLIT_COLOR = [102, 187, 106, 255]    # Green
_PENUMBRA_COLOR = [255, 167, 38, 255]   # Orange
_UMBRA_COLOR = [255, 82, 82, 255]       # Red


def eclipse_snapshot_packets(
    states: list[OrbitalState],
    epoch: datetime,
    name: str = "Eclipse State",
) -> list[dict]:
    """Static snapshot colored by eclipse state at epoch.

    Green = sunlit, orange = penumbra, red = umbra.
    No animation, no path, no label — just colored points.

    Args:
        states: List of orbital states.
        epoch: Evaluation time.
        name: Document name.

    Returns:
        List of CZML packets (document + N point packets).
    """
    packets: list[dict] = [_document_packet(name)]

    if not states:
        return packets

    sun = sun_position_eci(epoch)

    for idx, state in enumerate(states):
        pos_eci, vel_eci = propagate_to(state, epoch)
        gmst = gmst_rad(epoch)
        pos_ecef, _ = eci_to_ecef(
            (pos_eci[0], pos_eci[1], pos_eci[2]),
            (vel_eci[0], vel_eci[1], vel_eci[2]),
            gmst,
        )
        lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)

        sat_pos = (pos_eci[0], pos_eci[1], pos_eci[2])
        eclipse_type = is_eclipsed(sat_pos, sun.position_eci_m)

        if eclipse_type == EclipseType.UMBRA:
            color = _UMBRA_COLOR
        elif eclipse_type == EclipseType.PENUMBRA:
            color = _PENUMBRA_COLOR
        else:
            color = _SUNLIT_COLOR

        pkt: dict = {
            "id": f"eclipse-snap-{idx}",
            "name": f"Sat-{idx}",
            "position": {
                "cartographicDegrees": [lon_deg, lat_deg, alt_m],
            },
            "point": {
                "pixelSize": 3,
                "color": {"rgba": list(color)},
            },
        }
        packets.append(pkt)

    return packets


def _propagate_geodetic(
    state: OrbitalState, target_time: datetime,
) -> tuple[list[float], list[float], float, float, float]:
    """Propagate and convert to geodetic. Returns (pos_eci, vel_eci, lat, lon, alt)."""
    pos_eci, vel_eci = propagate_to(state, target_time)
    gmst = gmst_rad(target_time)
    pos_ecef, _ = eci_to_ecef(
        (pos_eci[0], pos_eci[1], pos_eci[2]),
        (vel_eci[0], vel_eci[1], vel_eci[2]),
        gmst,
    )
    lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)
    return pos_eci, vel_eci, lat_deg, lon_deg, alt_m


def _build_eclipse_intervals(
    step_times: list[datetime],
    eclipse_states: list[bool],
    sunlit_rgba: list[int],
    eclipse_rgba: list[int],
    epoch: datetime,
    duration: timedelta,
) -> list[dict]:
    """Build interval-based CZML color from eclipse state transitions."""
    end_time = epoch + duration

    if not step_times:
        return [{"interval": f"{_iso(epoch)}/{_iso(end_time)}", "rgba": sunlit_rgba}]

    intervals: list[dict] = []
    current_eclipsed = eclipse_states[0]
    interval_start = step_times[0]

    for i in range(1, len(step_times)):
        if eclipse_states[i] != current_eclipsed:
            color = eclipse_rgba if current_eclipsed else sunlit_rgba
            intervals.append({
                "interval": f"{_iso(interval_start)}/{_iso(step_times[i])}",
                "rgba": color,
            })
            interval_start = step_times[i]
            current_eclipsed = eclipse_states[i]

    color = eclipse_rgba if current_eclipsed else sunlit_rgba
    intervals.append({
        "interval": f"{_iso(interval_start)}/{_iso(end_time)}",
        "rgba": color,
    })

    return intervals


def eclipse_constellation_packets(
    orbital_states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    name: str = "Eclipse Constellation",
) -> list[dict]:
    """Constellation packets with eclipse-aware satellite coloring.

    Satellites change color between their orbital plane color (sunlit)
    and dark gray (eclipsed). Uses CZML interval-based color property.

    Args:
        orbital_states: List of orbital states.
        epoch: Start time.
        duration: Total duration.
        step: Time step for propagation.
        name: Document name.

    Returns:
        List of CZML packets.
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not orbital_states:
        return packets

    total_seconds = duration.total_seconds()
    step_seconds = _validate_step(step)
    num_steps = int(total_seconds / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    plane_indices = _assign_plane_indices(orbital_states)

    for idx, state in enumerate(orbital_states):
        coords: list[float] = []
        eclipse_flags: list[bool] = []
        step_times: list[datetime] = []

        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            pos_eci, vel_eci, lat_deg, lon_deg, alt_m = _propagate_geodetic(
                state, target_time,
            )
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])

            sun = sun_position_eci(target_time)
            sat_pos = (pos_eci[0], pos_eci[1], pos_eci[2])
            eclipse_type = is_eclipsed(sat_pos, sun.position_eci_m)
            eclipse_flags.append(eclipse_type != EclipseType.NONE)
            step_times.append(target_time)

        plane_color = _PLANE_COLORS[plane_indices[idx] % len(_PLANE_COLORS)]
        sunlit_rgba = list(plane_color)
        eclipse_rgba = list(_ECLIPSE_COLOR)

        color_intervals = _build_eclipse_intervals(
            step_times, eclipse_flags, sunlit_rgba, eclipse_rgba, epoch, duration,
        )

        path_color = [plane_color[0], plane_color[1], plane_color[2], 128]

        pkt: dict = {
            "id": f"satellite-{idx}",
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
                "color": color_intervals,
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


def sensor_footprint_packets(
    orbital_states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    sensor: SensorConfig,
    name: str = "Sensor Footprint",
) -> list[dict]:
    """Sensor FOV footprints sweeping the ground as satellites orbit.

    Creates ground-level ellipse entities that follow each satellite's
    sub-satellite point, sized by the sensor swath width.

    Args:
        orbital_states: List of orbital states.
        epoch: Start time.
        duration: Total duration.
        step: Time step.
        sensor: Sensor configuration for FOV geometry.
        name: Document name.

    Returns:
        List of CZML packets.
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not orbital_states:
        return packets

    total_seconds = duration.total_seconds()
    step_seconds = _validate_step(step)
    num_steps = int(total_seconds / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    for idx, state in enumerate(orbital_states):
        coords: list[float] = []
        alt_km_first: float | None = None

        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_deg, lat_deg, 0.0])
            if alt_km_first is None:
                alt_km_first = alt_m / 1000.0

        alt_km = alt_km_first if alt_km_first is not None else 550.0
        swath_km = compute_swath_width(alt_km, sensor.half_angle_deg)
        footprint_radius_m = swath_km * 1000.0 / 2.0

        pkt: dict = {
            "id": f"footprint-{idx}",
            "name": f"Footprint Sat-{idx}",
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "ellipse": {
                "semiMajorAxis": footprint_radius_m,
                "semiMinorAxis": footprint_radius_m,
                "height": 0,
                "material": {
                    "solidColor": {
                        "color": {"rgba": [0, 255, 0, 64]},
                    },
                },
            },
        }
        packets.append(pkt)

    return packets


def ground_station_packets(
    station: GroundStation,
    orbital_states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    min_elevation_deg: float = 10.0,
    name: str = "Ground Station",
) -> list[dict]:
    """Ground station marker with visibility circle and access event tracks.

    Creates:
    - Station point marker with label.
    - Visibility circle (ground range at min elevation).
    - Satellite track entities during access windows (with availability).

    Args:
        station: Ground station.
        orbital_states: Satellite orbital states.
        epoch: Start time.
        duration: Total duration.
        step: Time step.
        min_elevation_deg: Minimum elevation for visibility.
        name: Document name.

    Returns:
        List of CZML packets.
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    station_id = f"station-{station.name}"

    # Station marker
    station_pkt: dict = {
        "id": station_id,
        "name": station.name,
        "position": {
            "cartographicDegrees": [station.lon_deg, station.lat_deg, station.alt_m],
        },
        "point": {
            "pixelSize": 10,
            "color": {"rgba": [255, 255, 0, 255]},
        },
        "label": {
            "text": station.name,
            "font": "13pt sans-serif",
            "fillColor": {"rgba": [255, 255, 0, 255]},
            "outlineWidth": 2,
            "style": "FILL_AND_OUTLINE",
            "horizontalOrigin": "LEFT",
            "pixelOffset": {"cartesian2": [14, 0]},
        },
    }
    packets.append(station_pkt)

    # Visibility circle based on representative altitude
    if orbital_states:
        alt_m = orbital_states[0].semi_major_axis_m - OrbitalConstants.R_EARTH
        alt_km = alt_m / 1000.0
        r_earth_km = OrbitalConstants.R_EARTH / 1000.0
        el_rad = math.radians(min_elevation_deg)

        sin_angle_s = r_earth_km * math.cos(el_rad) / (r_earth_km + alt_km)
        if sin_angle_s < 1.0:
            angle_o = math.pi / 2 - el_rad - math.asin(sin_angle_s)
            max_range_m = OrbitalConstants.R_EARTH * angle_o
        else:
            max_range_m = 0.0

        if max_range_m > 0:
            circle_pkt: dict = {
                "id": f"{station_id}-visibility",
                "name": f"{station.name} Visibility",
                "position": {
                    "cartographicDegrees": [station.lon_deg, station.lat_deg, 0.0],
                },
                "ellipse": {
                    "semiMajorAxis": max_range_m,
                    "semiMinorAxis": max_range_m,
                    "height": 0,
                    "material": {
                        "solidColor": {
                            "color": {"rgba": [255, 255, 0, 32]},
                        },
                    },
                    "outline": True,
                    "outlineColor": {"rgba": [255, 255, 0, 128]},
                },
            }
            packets.append(circle_pkt)

    # Access window satellite tracks
    for idx, state in enumerate(orbital_states):
        windows = compute_access_windows(
            station, state, epoch, duration, step, min_elevation_deg,
        )
        for w_idx, window in enumerate(windows):
            access_coords: list[float] = []
            t = window.rise_time
            while t <= window.set_time:
                _, _, lat_deg, lon_deg, alt_m_sat = _propagate_geodetic(state, t)
                t_offset = (t - epoch).total_seconds()
                access_coords.extend([t_offset, lon_deg, lat_deg, alt_m_sat])
                t = t + step

            if access_coords:
                num_access_pts = len(access_coords) // 4
                access_pkt: dict = {
                    "id": f"access-{idx}-{w_idx}",
                    "name": f"Access: Sat-{idx} Pass {w_idx + 1}",
                    "availability": f"{_iso(window.rise_time)}/{_iso(window.set_time)}",
                    "position": {
                        "epoch": _iso(epoch),
                        "cartographicDegrees": access_coords,
                        "interpolationAlgorithm": "LAGRANGE",
                        "interpolationDegree": _interpolation_degree(num_access_pts),
                    },
                    "point": {
                        "pixelSize": 8,
                        "color": {"rgba": [0, 255, 0, 255]},
                    },
                    "path": {
                        "leadTime": 0,
                        "trailTime": window.duration_seconds,
                        "resolution": 30,
                        "material": {
                            "solidColor": {
                                "color": {"rgba": [0, 255, 0, 128]},
                            },
                        },
                        "width": 2,
                    },
                }
                packets.append(access_pkt)

    return packets


def conjunction_replay_packets(
    state_a: OrbitalState,
    state_b: OrbitalState,
    event_time: datetime,
    window: timedelta,
    step: timedelta,
    name_a: str = "Sat-A",
    name_b: str = "Sat-B",
) -> list[dict]:
    """Conjunction event 3D replay with two highlighted satellites.

    Creates two satellite entities with highlighted colors and a
    proximity polyline connecting them via CZML position references.

    Args:
        state_a: First satellite orbital state.
        state_b: Second satellite orbital state.
        event_time: Approximate conjunction time (center of replay).
        window: Half-width of replay window (before and after event).
        step: Time step for propagation.
        name_a: First satellite name.
        name_b: Second satellite name.

    Returns:
        List of CZML packets.
    """
    start = event_time - window
    total_duration = timedelta(seconds=window.total_seconds() * 2)

    packets: list[dict] = [_document_packet("Conjunction Replay", start, total_duration)]

    total_seconds = total_duration.total_seconds()
    step_seconds = _validate_step(step)
    num_steps = int(total_seconds / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    sat_colors = [
        [255, 50, 50, 255],    # Red for satellite A
        [50, 100, 255, 255],   # Blue for satellite B
    ]

    for sat_idx, (state, sat_name) in enumerate([(state_a, name_a), (state_b, name_b)]):
        coords: list[float] = []

        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = start + timedelta(seconds=t_offset)
            _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])

        color = sat_colors[sat_idx]
        path_color = [c // 2 for c in color[:3]] + [128]
        sat_id = f"conjunction-sat-{sat_idx}"

        pkt: dict = {
            "id": sat_id,
            "name": sat_name,
            "position": {
                "epoch": _iso(start),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "point": {
                "pixelSize": 8,
                "color": {"rgba": color},
            },
            "label": {
                "text": sat_name,
                "font": "12pt sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 220]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [12, 0]},
            },
            "path": {
                "leadTime": 300,
                "trailTime": 300,
                "resolution": 30,
                "material": {
                    "solidColor": {
                        "color": {"rgba": path_color},
                    },
                },
                "width": 2,
            },
        }
        packets.append(pkt)

    # Proximity line connecting both satellites via CZML references
    packets.append({
        "id": "conjunction-proximity-line",
        "name": "Proximity",
        "polyline": {
            "positions": {
                "references": [
                    "conjunction-sat-0#position",
                    "conjunction-sat-1#position",
                ],
            },
            "material": {
                "solidColor": {
                    "color": {"rgba": [255, 255, 0, 200]},
                },
            },
            "width": 2,
            "arcType": "NONE",
        },
    })

    return packets


def coverage_evolution_packets(
    orbital_states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    coverage_step: timedelta,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    min_elevation_deg: float = 10.0,
    name: str = "Coverage Evolution",
) -> list[dict]:
    """Time-varying coverage heatmap with interval-based colors.

    Computes coverage snapshots at each coverage_step and builds
    CZML rectangles with time-varying green intensity.

    Args:
        orbital_states: List of orbital states.
        epoch: Start time.
        duration: Total duration.
        coverage_step: Time between coverage snapshots.
        lat_step_deg: Latitude grid spacing.
        lon_step_deg: Longitude grid spacing.
        min_elevation_deg: Minimum elevation for visibility.
        name: Document name.

    Returns:
        List of CZML packets.
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not orbital_states:
        return packets

    step_seconds = _validate_step(coverage_step)
    total_seconds = duration.total_seconds()
    num_steps = int(total_seconds / step_seconds) + 1

    times = [epoch + timedelta(seconds=i * step_seconds) for i in range(num_steps)]

    # Compute coverage at each time step
    snapshots: list[dict[tuple[float, float], int]] = []
    for t in times:
        snap = compute_coverage_snapshot(
            orbital_states, t, lat_step_deg, lon_step_deg, min_elevation_deg,
        )
        snapshots.append({(p.lat_deg, p.lon_deg): p.visible_count for p in snap})

    # Find global max count
    all_counts = [c for snap in snapshots for c in snap.values()]
    max_count = max(all_counts) if all_counts else 1
    if max_count == 0:
        max_count = 1

    # Get grid cells from first snapshot
    if not snapshots:
        return packets

    grid_cells = list(snapshots[0].keys())

    for cell_idx, (lat, lon) in enumerate(grid_cells):
        any_nonzero = any(snap.get((lat, lon), 0) > 0 for snap in snapshots)
        if not any_nonzero:
            continue

        color_intervals: list[dict] = []
        for step_idx in range(len(times)):
            t_start = times[step_idx]
            t_end = times[step_idx + 1] if step_idx + 1 < len(times) else epoch + duration
            count = snapshots[step_idx].get((lat, lon), 0)
            intensity = int(255 * count / max_count)
            color_intervals.append({
                "interval": f"{_iso(t_start)}/{_iso(t_end)}",
                "rgba": [0, intensity, 0, 128],
            })

        w = lon
        s_bound = lat
        e = lon + lon_step_deg
        n = lat + lat_step_deg

        pkt: dict = {
            "id": f"coverage-evo-{cell_idx}",
            "name": f"Coverage ({lat}, {lon})",
            "rectangle": {
                "coordinates": {"wsenDegrees": [w, s_bound, e, n]},
                "fill": True,
                "material": {
                    "solidColor": {
                        "color": color_intervals,
                    },
                },
            },
        }
        packets.append(pkt)

    return packets


def precession_constellation_packets(
    satellites: list,
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    name: str = "J2 Precession",
) -> list[dict]:
    """Constellation with J2 RAAN precession over long duration.

    Derives orbital states with J2 secular corrections enabled,
    then delegates to constellation_packets for CZML generation.

    Args:
        satellites: List of Satellite objects from generate_walker_shell.
        epoch: Start time.
        duration: Total duration (typically days to weeks).
        step: Time step.
        name: Document name.

    Returns:
        List of CZML packets with J2-precessed orbits.
    """
    states = [derive_orbital_state(s, epoch, include_j2=True) for s in satellites]
    return constellation_packets(states, epoch, duration, step, name=name)


def _snr_color(snr_db: float, min_snr: float = 0.0, max_snr: float = 30.0) -> list[int]:
    """Map SNR to green->yellow->red color gradient."""
    t = max(0.0, min(1.0, (snr_db - min_snr) / (max_snr - min_snr)))
    if t > 0.5:
        # Green to yellow
        f = (t - 0.5) * 2.0
        return [int(255 * (1 - f)), 255, 0, 200]
    else:
        # Yellow to red
        f = t * 2.0
        return [255, int(255 * f), 0, 200]


def _health_color(value: float) -> list[int]:
    """Map 0..1 health value to green->yellow->red."""
    t = max(0.0, min(1.0, value))
    if t > 0.5:
        f = (t - 0.5) * 2.0
        return [int(255 * (1 - f)), 255, 0, 200]
    else:
        f = t * 2.0
        return [255, int(255 * f), 0, 200]


def isl_topology_packets(
    states: list[OrbitalState],
    time: datetime,
    link_config: LinkConfig,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    max_range_km: float = 5000.0,
    name: str = "ISL Topology",
) -> list[dict]:
    """ISL topology visualization with SNR-colored polylines.

    Satellite points (plane-colored) + polylines between ISL-linked pairs.
    Polyline color: green (high SNR) -> yellow -> red (low margin).

    Args:
        states: Satellite orbital states.
        time: Evaluation time for initial topology.
        link_config: RF link configuration.
        epoch: Animation start time.
        duration_s: Animation duration (seconds).
        step_s: Time step (seconds).
        max_range_km: Maximum ISL range (km).
        name: Document name.

    Returns:
        List of CZML packets.
    """
    duration = timedelta(seconds=duration_s)
    step = timedelta(seconds=step_s)
    step_seconds = _validate_step(step)
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not states:
        return packets

    num_steps = int(duration_s / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)
    plane_indices = _assign_plane_indices(states)

    # Satellite point packets
    for idx, state in enumerate(states):
        coords: list[float] = []
        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])

        plane_color = _PLANE_COLORS[plane_indices[idx] % len(_PLANE_COLORS)]
        packets.append({
            "id": f"isl-sat-{idx}",
            "name": f"Sat-{idx}",
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
        })

    # ISL link polylines using position references
    topology = compute_isl_topology(states, time, max_range_km=max_range_km)
    for link_idx, link in enumerate(topology.links):
        if link.is_blocked:
            continue
        budget = compute_link_budget(link_config, link.distance_m)
        color = _snr_color(budget.snr_db)
        packets.append({
            "id": f"isl-link-{link_idx}",
            "name": f"ISL {link.sat_idx_a}-{link.sat_idx_b}",
            "polyline": {
                "positions": {
                    "references": [
                        f"isl-sat-{link.sat_idx_a}#position",
                        f"isl-sat-{link.sat_idx_b}#position",
                    ],
                },
                "material": {
                    "solidColor": {"color": {"rgba": color}},
                },
                "width": 2,
                "arcType": "NONE",
            },
        })

    return packets


def fragility_constellation_packets(
    states: list[OrbitalState],
    epoch: datetime,
    link_config: LinkConfig,
    n_rad_s: float,
    control_duration_s: float,
    duration_s: float,
    step_s: float,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    name: str = "Spectral Fragility",
) -> list[dict]:
    """Constellation colored by spectral fragility.

    Green (high fragility/healthy) -> yellow -> red (fragile).
    Coarse time step for performance.

    Args:
        states: Satellite orbital states.
        epoch: Start time.
        link_config: RF link configuration.
        n_rad_s: Mean motion (rad/s).
        control_duration_s: Control horizon (s).
        duration_s: Total duration (s).
        step_s: Time step for position (s).
        lat_deg: Reference latitude for DOP.
        lon_deg: Reference longitude for DOP.
        name: Document name.

    Returns:
        List of CZML packets.
    """
    from humeris.domain.design_sensitivity import compute_spectral_fragility

    duration = timedelta(seconds=duration_s)
    step = timedelta(seconds=step_s)
    step_seconds = _validate_step(step)
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not states:
        return packets

    num_steps = int(duration_s / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    # Compute fragility at epoch for coloring
    fragility = compute_spectral_fragility(
        states, epoch, link_config, n_rad_s, control_duration_s, lat_deg, lon_deg,
    )
    # Map composite to color (higher = healthier = greener)
    frag_val = min(1.0, fragility.composite_fragility * 1000.0)
    frag_color = _health_color(frag_val)

    for idx, state in enumerate(states):
        coords: list[float] = []
        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            _, _, lat_d, lon_d, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_d, lat_d, alt_m])

        packets.append({
            "id": f"frag-sat-{idx}",
            "name": f"Sat-{idx} (fragility)",
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "point": {
                "pixelSize": 6,
                "color": {"rgba": frag_color},
            },
        })

    return packets


def hazard_evolution_packets(
    states: list[OrbitalState],
    survival_curve,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    name: str = "Hazard Evolution",
) -> list[dict]:
    """Satellites colored by hazard rate / survival fraction.

    Green (healthy) -> yellow -> red (near EOL).
    Long-duration timeline (days/months).

    Args:
        states: Satellite orbital states.
        survival_curve: LifetimeSurvivalCurve from statistical_analysis.
        epoch: Start time.
        duration_s: Total duration (s).
        step_s: Time step (s).
        name: Document name.

    Returns:
        List of CZML packets.
    """
    duration = timedelta(seconds=duration_s)
    step = timedelta(seconds=step_s)
    step_seconds = _validate_step(step)
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not states:
        return packets

    num_steps = int(duration_s / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    # Build interval colors from survival curve
    surv_fracs = list(survival_curve.survival_fraction) if survival_curve.survival_fraction else [1.0]
    surv_times = list(survival_curve.times) if survival_curve.times else [epoch]

    for idx, state in enumerate(states):
        coords: list[float] = []
        step_times: list[datetime] = []
        health_vals: list[float] = []

        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])
            step_times.append(target_time)

            # Find survival fraction at this time
            fraction = 1.0
            if surv_times and surv_fracs:
                elapsed_days = t_offset / 86400.0
                total_days = survival_curve.mean_remaining_life_days
                if total_days > 0:
                    fraction = max(0.0, 1.0 - elapsed_days / total_days)
            health_vals.append(fraction)

        # Build interval-based color
        end_time = epoch + duration
        color_intervals: list[dict] = []
        for i in range(len(step_times)):
            t_start = step_times[i]
            t_end = step_times[i + 1] if i + 1 < len(step_times) else end_time
            color = _health_color(health_vals[i])
            color_intervals.append({
                "interval": f"{_iso(t_start)}/{_iso(t_end)}",
                "rgba": color,
            })

        packets.append({
            "id": f"hazard-sat-{idx}",
            "name": f"Sat-{idx} (hazard)",
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "point": {
                "pixelSize": 6,
                "color": color_intervals,
            },
        })

    return packets


def coverage_connectivity_packets(
    states: list[OrbitalState],
    link_config: LinkConfig,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    name: str = "Coverage-Connectivity",
) -> list[dict]:
    """Ground rectangles colored by coverage_count * fiedler_value.

    Shows where coverage is backed by network connectivity.

    Args:
        states: Satellite orbital states.
        link_config: RF link configuration.
        epoch: Start time.
        duration_s: Total duration (s).
        step_s: Coverage snapshot interval (s).
        lat_step_deg: Latitude grid spacing.
        lon_step_deg: Longitude grid spacing.
        name: Document name.

    Returns:
        List of CZML packets.
    """
    duration = timedelta(seconds=duration_s)
    coverage_step = timedelta(seconds=step_s)
    step_seconds = _validate_step(coverage_step)
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not states:
        return packets

    num_steps = int(duration_s / step_seconds) + 1
    times = [epoch + timedelta(seconds=i * step_seconds) for i in range(num_steps)]

    # At each time, compute coverage and fiedler
    snapshots: list[dict] = []
    for t in times:
        snap = compute_coverage_snapshot(
            states, t, lat_step_deg, lon_step_deg, 10.0,
        )
        resilience = compute_topology_resilience(states, t, link_config)
        fiedler = resilience.fiedler_value
        cell_values = {}
        for p in snap:
            cell_values[(p.lat_deg, p.lon_deg)] = p.visible_count * fiedler
        snapshots.append(cell_values)

    # Find global max
    all_vals = [v for s in snapshots for v in s.values()]
    max_val = max(all_vals) if all_vals else 1.0
    if max_val == 0:
        max_val = 1.0

    if not snapshots:
        return packets

    grid_cells = list(snapshots[0].keys())

    for cell_idx, (lat, lon) in enumerate(grid_cells):
        any_nonzero = any(s.get((lat, lon), 0) > 0 for s in snapshots)
        if not any_nonzero:
            continue

        color_intervals: list[dict] = []
        for step_idx in range(len(times)):
            t_start = times[step_idx]
            t_end = times[step_idx + 1] if step_idx + 1 < len(times) else epoch + duration
            val = snapshots[step_idx].get((lat, lon), 0)
            intensity = int(255 * val / max_val)
            color_intervals.append({
                "interval": f"{_iso(t_start)}/{_iso(t_end)}",
                "rgba": [0, intensity, intensity // 2, 128],
            })

        packets.append({
            "id": f"cov-conn-{cell_idx}",
            "name": f"CovConn ({lat}, {lon})",
            "rectangle": {
                "coordinates": {
                    "wsenDegrees": [lon, lat, lon + lon_step_deg, lat + lat_step_deg],
                },
                "fill": True,
                "material": {
                    "solidColor": {"color": color_intervals},
                },
            },
        })

    return packets


def network_eclipse_packets(
    states: list[OrbitalState],
    link_config: LinkConfig,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    max_range_km: float = 5000.0,
    name: str = "Network Eclipse",
) -> list[dict]:
    """ISL polylines with eclipse-dependent color.

    Green (both sunlit) -> orange (one eclipsed) -> hidden (both eclipsed).

    Args:
        states: Satellite orbital states.
        link_config: RF link configuration.
        epoch: Start time.
        duration_s: Total duration (s).
        step_s: Time step (s).
        max_range_km: Maximum ISL range (km).
        name: Document name.

    Returns:
        List of CZML packets.
    """
    duration = timedelta(seconds=duration_s)
    step = timedelta(seconds=step_s)
    step_seconds = _validate_step(step)
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not states:
        return packets

    num_steps = int(duration_s / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    # Satellite positions
    for idx, state in enumerate(states):
        coords: list[float] = []
        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])

        packets.append({
            "id": f"netecl-sat-{idx}",
            "name": f"Sat-{idx}",
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "point": {"pixelSize": 4, "color": {"rgba": [200, 200, 200, 200]}},
        })

    # ISL links with eclipse-aware color at epoch
    topology = compute_isl_topology(states, epoch, max_range_km=max_range_km)
    sun = sun_position_eci(epoch)

    eclipsed = []
    for state in states:
        pos_eci, _ = propagate_to(state, epoch)
        pos_tuple = (pos_eci[0], pos_eci[1], pos_eci[2])
        sun_tuple = (sun.position_eci_m[0], sun.position_eci_m[1], sun.position_eci_m[2])
        ecl_type = is_eclipsed(pos_tuple, sun_tuple)
        eclipsed.append(ecl_type != EclipseType.NONE)

    for link_idx, link in enumerate(topology.links):
        if link.is_blocked:
            continue
        a_ecl = eclipsed[link.sat_idx_a] if link.sat_idx_a < len(eclipsed) else False
        b_ecl = eclipsed[link.sat_idx_b] if link.sat_idx_b < len(eclipsed) else False

        if a_ecl and b_ecl:
            color = [100, 100, 100, 80]
        elif a_ecl or b_ecl:
            color = [255, 165, 0, 200]
        else:
            color = [0, 255, 0, 200]

        packets.append({
            "id": f"netecl-link-{link_idx}",
            "name": f"ISL {link.sat_idx_a}-{link.sat_idx_b}",
            "polyline": {
                "positions": {
                    "references": [
                        f"netecl-sat-{link.sat_idx_a}#position",
                        f"netecl-sat-{link.sat_idx_b}#position",
                    ],
                },
                "material": {
                    "solidColor": {"color": {"rgba": color}},
                },
                "width": 2,
                "arcType": "NONE",
            },
        })

    return packets


# ── Risk / hazard color maps ─────────────────────────────────────────

_KESSLER_RISK_COLORS: dict[str, list[int]] = {
    "low": [102, 187, 106, 255],
    "moderate": [255, 235, 59, 255],
    "high": [255, 152, 0, 255],
    "critical": [244, 67, 54, 255],
}

_HAZARD_LEVEL_COLORS: dict[HazardLevel, list[int]] = {
    HazardLevel.ROUTINE: [102, 187, 106, 180],
    HazardLevel.WARNING: [255, 235, 59, 220],
    HazardLevel.CRITICAL: [244, 67, 54, 255],
}


def kessler_heatmap_packets(
    states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    name: str = "Kessler Heatmap",
) -> list[dict]:
    """Satellites colored by Kessler cascade risk level.

    Computes spatial density heatmap from orbital states and colors
    each satellite by the risk level of its altitude/inclination cell:
    green (low), yellow (moderate), orange (high), red (critical).

    Args:
        states: Satellite orbital states.
        epoch: Start time.
        duration: Total duration.
        step: Time step.
        name: Document name.

    Returns:
        List of CZML packets.
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if not states:
        return packets

    heatmap = compute_kessler_heatmap(states)

    # Map each satellite to its cell's risk level
    sat_risks: list[str] = []
    for state in states:
        alt_km = (state.semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
        inc_deg = abs(math.degrees(state.inclination_rad))
        risk = "low"
        for cell in heatmap.cells:
            if (cell.altitude_min_km <= alt_km < cell.altitude_max_km
                    and cell.inclination_min_deg <= inc_deg < cell.inclination_max_deg):
                risk = cell.risk_level
                break
        sat_risks.append(risk)

    total_seconds = duration.total_seconds()
    step_seconds = _validate_step(step)
    num_steps = int(total_seconds / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    for idx, state in enumerate(states):
        coords: list[float] = []
        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])

        color = _KESSLER_RISK_COLORS.get(sat_risks[idx], _KESSLER_RISK_COLORS["low"])

        packets.append({
            "id": f"kessler-sat-{idx}",
            "name": f"Sat-{idx} ({sat_risks[idx]})",
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "point": {
                "pixelSize": 6,
                "color": {"rgba": color},
            },
            "label": {
                "text": sat_risks[idx][0].upper(),
                "font": "9pt sans-serif",
                "fillColor": {"rgba": color},
                "outlineWidth": 1,
                "style": "FILL_AND_OUTLINE",
                "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [8, 0]},
            },
        })

    return packets


def conjunction_hazard_packets(
    states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    screening_threshold_m: float = 100_000.0,
    name: str = "Conjunction Hazard",
) -> list[dict]:
    """Constellation with conjunction events colored by hazard level.

    Screens for conjunctions and shows polylines between close pairs,
    colored by NASA-STD-8719.14 hazard level:
    green (ROUTINE), yellow (WARNING), red (CRITICAL).

    Args:
        states: Satellite orbital states.
        epoch: Start time.
        duration: Total duration.
        step: Time step.
        screening_threshold_m: Distance threshold for screening (meters).
        name: Document name.

    Returns:
        List of CZML packets.
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]

    if len(states) < 2:
        return packets

    total_seconds = duration.total_seconds()
    step_seconds = _validate_step(step)
    num_steps = int(total_seconds / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    # Satellite position packets
    for idx, state in enumerate(states):
        coords: list[float] = []
        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])

        packets.append({
            "id": f"hazconj-sat-{idx}",
            "name": f"Sat-{idx}",
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "point": {
                "pixelSize": 4,
                "color": {"rgba": [200, 200, 200, 200]},
            },
        })

    # Screen for conjunctions
    sat_names = [f"Sat-{i}" for i in range(len(states))]
    screening_step = timedelta(seconds=60)
    candidates = screen_conjunctions(
        states, sat_names, epoch, duration, screening_step, screening_threshold_m,
    )

    # Assess and classify top candidates
    for c_idx, (i, j, tca, dist_m) in enumerate(candidates[:20]):
        event = assess_conjunction(states[i], sat_names[i], states[j], sat_names[j], tca)
        hazard = classify_hazard(event)
        color = _HAZARD_LEVEL_COLORS.get(hazard, _HAZARD_LEVEL_COLORS[HazardLevel.ROUTINE])

        # Show polyline during a window around TCA
        window_half = timedelta(minutes=10)
        avail_start = max(epoch, tca - window_half)
        avail_end = min(epoch + duration, tca + window_half)

        packets.append({
            "id": f"hazconj-line-{c_idx}",
            "name": f"{hazard.value}: Sat-{i} - Sat-{j} ({dist_m:.0f}m)",
            "availability": f"{_iso(avail_start)}/{_iso(avail_end)}",
            "polyline": {
                "positions": {
                    "references": [
                        f"hazconj-sat-{i}#position",
                        f"hazconj-sat-{j}#position",
                    ],
                },
                "material": {
                    "solidColor": {"color": {"rgba": color}},
                },
                "width": 3,
                "arcType": "NONE",
            },
        })

    return packets
