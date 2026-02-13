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
from humeris.domain.dilution_of_precision import compute_dop_grid
from humeris.domain.radiation import compute_l_shell
from humeris.domain.eclipse import compute_beta_angle
from humeris.domain.deorbit import assess_deorbit_compliance, DeorbitRegulation
from humeris.domain.atmosphere import DragConfig
from humeris.domain.station_keeping import (
    compute_station_keeping_budget,
    StationKeepingConfig,
)
from humeris.domain.cascade_analysis import compute_cascade_sir
from humeris.domain.relative_motion import compute_relative_state
from humeris.domain.maintenance_planning import compute_perturbation_budget
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

    # Extend last interval to far future so tiles persist after clock stops
    far_future = epoch + timedelta(days=365)

    for cell_idx, (lat, lon) in enumerate(grid_cells):
        any_nonzero = any(snap.get((lat, lon), 0) > 0 for snap in snapshots)
        if not any_nonzero:
            continue

        color_intervals: list[dict] = []
        for step_idx in range(len(times)):
            t_start = times[step_idx]
            if step_idx + 1 < len(times):
                t_end = times[step_idx + 1]
            else:
                # Last interval extends to far future — holds last color
                t_end = far_future
            count = snapshots[step_idx].get((lat, lon), 0)
            intensity = int(255 * count / max_count)
            color_intervals.append({
                "interval": f"{_iso(t_start)}/{_iso(t_end)}",
                "rgba": [0, intensity, 0, 180],
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

    # Extend last interval to far future so tiles persist after clock stops
    far_future = epoch + timedelta(days=365)

    for cell_idx, (lat, lon) in enumerate(grid_cells):
        any_nonzero = any(s.get((lat, lon), 0) > 0 for s in snapshots)
        if not any_nonzero:
            continue

        color_intervals: list[dict] = []
        for step_idx in range(len(times)):
            t_start = times[step_idx]
            if step_idx + 1 < len(times):
                t_end = times[step_idx + 1]
            else:
                t_end = far_future
            val = snapshots[step_idx].get((lat, lon), 0)
            intensity = int(255 * val / max_val)
            color_intervals.append({
                "interval": f"{_iso(t_start)}/{_iso(t_end)}",
                "rgba": [0, intensity, intensity // 2, 180],
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


def dop_grid_packets(
    states: list[OrbitalState],
    epoch: datetime,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    min_elevation_deg: float = 10.0,
    name: str = "DOP Grid",
) -> list[dict]:
    """Grid heatmap showing Geometric Dilution of Precision at epoch.

    Colors each grid cell by GDOP value: green (< 3, good),
    yellow (3-6, moderate), red (> 6, poor). Gray if no satellites visible.

    Args:
        states: List of orbital states.
        epoch: Evaluation time.
        lat_step_deg: Latitude grid spacing (degrees).
        lon_step_deg: Longitude grid spacing (degrees).
        min_elevation_deg: Minimum elevation for visibility (degrees).
        name: Document name.

    Returns:
        List of CZML packets (document + rectangle packets).
    """
    packets: list[dict] = [_document_packet(name)]

    if not states:
        return packets

    grid = compute_dop_grid(
        states, epoch, lat_step_deg, lon_step_deg, min_elevation_deg,
    )

    for cell_idx, point in enumerate(grid):
        gdop = point.dop.gdop
        num_vis = point.dop.num_visible

        if num_vis == 0:
            color = [160, 160, 160, 128]
        elif gdop < 3.0:
            # Green — good geometry
            t = gdop / 3.0
            color = [int(255 * t), 255, 0, 160]
        elif gdop < 6.0:
            # Yellow to red transition
            t = (gdop - 3.0) / 3.0
            color = [255, int(255 * (1.0 - t)), 0, 160]
        else:
            color = [255, 0, 0, 160]

        lat = point.lat_deg
        lon = point.lon_deg
        w = lon
        s_bound = lat
        e = lon + lon_step_deg
        n = lat + lat_step_deg

        pkt: dict = {
            "id": f"dop-grid-{cell_idx}",
            "name": f"DOP ({lat:.0f}, {lon:.0f}) GDOP={gdop:.1f}",
            "rectangle": {
                "coordinates": {"wsenDegrees": [w, s_bound, e, n]},
                "fill": True,
                "material": {
                    "solidColor": {
                        "color": {"rgba": color},
                    },
                },
            },
        }
        packets.append(pkt)

    return packets


def radiation_coloring_packets(
    states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    name: str = "Radiation",
) -> list[dict]:
    """Animated satellite coloring by McIlwain L-shell radiation proxy.

    Green (L < 2, low radiation), yellow (2-4, moderate), red (> 4, high).
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]
    if not states:
        return packets

    total_seconds = duration.total_seconds()
    step_seconds = _validate_step(step)
    num_steps = int(total_seconds / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)
    end_time = epoch + duration

    for idx, state in enumerate(states):
        coords: list[float] = []
        step_times: list[datetime] = []
        l_shells: list[float] = []

        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])
            step_times.append(target_time)
            l_shells.append(compute_l_shell(lat_deg, lon_deg, alt_m / 1000.0))

        color_intervals: list[dict] = []
        for i in range(len(step_times)):
            t_end = step_times[i + 1] if i + 1 < len(step_times) else end_time
            l_val = l_shells[i]
            if l_val < 2.0:
                color = [102, 187, 106, 255]
            elif l_val < 4.0:
                t = (l_val - 2.0) / 2.0
                color = [int(102 + 153 * t), int(187 + 68 * t), int(106 * (1 - t)), 255]
            else:
                color = [255, 82, 82, 255]
            color_intervals.append({
                "interval": f"{_iso(step_times[i])}/{_iso(t_end)}", "rgba": color,
            })

        packets.append({
            "id": f"rad-sat-{idx}",
            "name": f"Sat-{idx}",
            "position": {
                "epoch": _iso(epoch),
                "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": interp_degree,
            },
            "point": {"pixelSize": 5, "color": color_intervals},
        })

    return packets


def beta_angle_packets(
    states: list[OrbitalState],
    epoch: datetime,
    name: str = "Beta Angle",
) -> list[dict]:
    """Snapshot coloring by orbital beta angle at epoch.

    Green (|beta| > 60 deg, minimal eclipses), yellow (20-60 deg),
    red (|beta| < 20 deg, maximum eclipse exposure).

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

    for idx, state in enumerate(states):
        _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, epoch)

        beta_deg = compute_beta_angle(
            state.raan_rad, state.inclination_rad, epoch,
        )
        abs_beta = abs(beta_deg)

        if abs_beta > 60.0:
            color = [102, 187, 106, 255]   # Green — minimal eclipses
        elif abs_beta > 20.0:
            t = (abs_beta - 20.0) / 40.0
            color = [int(255 * (1.0 - t) + 102 * t),
                     int(235 * (1.0 - t) + 187 * t),
                     int(59 * (1.0 - t) + 106 * t), 255]
        else:
            color = [255, 82, 82, 255]     # Red — deep eclipse zone

        pkt: dict = {
            "id": f"beta-snap-{idx}",
            "name": f"Sat-{idx} (beta={beta_deg:.1f} deg)",
            "position": {
                "cartographicDegrees": [lon_deg, lat_deg, alt_m],
            },
            "point": {
                "pixelSize": 5,
                "color": {"rgba": color},
            },
            "label": {
                "text": f"{beta_deg:.0f} deg",
                "font": "9pt sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 200]},
                "outlineWidth": 1,
                "style": "FILL_AND_OUTLINE",
                "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [10, 0]},
            },
        }
        packets.append(pkt)

    return packets


def deorbit_compliance_packets(
    states: list[OrbitalState],
    epoch: datetime,
    name: str = "Deorbit Compliance",
) -> list[dict]:
    """Snapshot coloring by 25-year deorbit compliance status.

    Green (compliant), yellow (marginal — natural lifetime < 30 years),
    red (non-compliant under FCC 5-year rule).

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

    drag_config = DragConfig(cd=2.2, area_m2=0.01, mass_kg=4.0)

    for idx, state in enumerate(states):
        _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, epoch)

        alt_km = (state.semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
        assessment = assess_deorbit_compliance(
            alt_km, drag_config, epoch, eccentricity=state.eccentricity,
        )

        remaining_years = assessment.natural_lifetime_days / 365.25

        if assessment.compliant:
            color = [102, 187, 106, 255]   # Green — compliant
            label_text = "OK"
        elif remaining_years < 30.0:
            color = [255, 235, 59, 255]    # Yellow — marginal
            label_text = f"{remaining_years:.0f}yr"
        else:
            color = [255, 82, 82, 255]     # Red — non-compliant
            label_text = f"{remaining_years:.0f}yr"

        pkt: dict = {
            "id": f"deorbit-snap-{idx}",
            "name": f"Sat-{idx} ({label_text})",
            "position": {
                "cartographicDegrees": [lon_deg, lat_deg, alt_m],
            },
            "point": {
                "pixelSize": 5,
                "color": {"rgba": color},
            },
            "label": {
                "text": label_text,
                "font": "9pt sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 200]},
                "outlineWidth": 1,
                "style": "FILL_AND_OUTLINE",
                "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [10, 0]},
            },
        }
        packets.append(pkt)

    return packets


def station_keeping_packets(
    states: list[OrbitalState],
    epoch: datetime,
    drag_config: DragConfig | None = None,
    density_func: object | None = None,
    name: str = "Station Keeping",
) -> list[dict]:
    """Snapshot coloring by annual station-keeping delta-V budget.

    Green (< 5 m/s/yr), yellow (5-20 m/s/yr), red (> 20 m/s/yr).

    Args:
        states: List of orbital states.
        epoch: Evaluation time.
        drag_config: Drag configuration (optional, uses internal default).
        density_func: Optional density callback for variable atmosphere.
        name: Document name.

    Returns:
        List of CZML packets (document + N point packets).
    """
    packets: list[dict] = [_document_packet(name)]

    if not states:
        return packets

    default_drag = DragConfig(cd=2.2, area_m2=0.01, mass_kg=4.0)

    for idx, state in enumerate(states):
        _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, epoch)

        alt_km = (state.semi_major_axis_m - OrbitalConstants.R_EARTH) / 1000.0
        inc_deg = math.degrees(state.inclination_rad)

        sk_config = StationKeepingConfig(
            target_altitude_km=alt_km,
            inclination_deg=inc_deg,
            drag_config=drag_config or default_drag,
            isp_s=220.0,
            dry_mass_kg=4.0,
            propellant_mass_kg=0.5,
        )
        budget = compute_station_keeping_budget(
            sk_config, density_func=density_func, epoch=epoch,
        )
        dv_yr = budget.total_dv_per_year_ms

        if dv_yr < 5.0:
            color = [102, 187, 106, 255]   # Green
        elif dv_yr < 20.0:
            t = (dv_yr - 5.0) / 15.0
            color = [int(102 + 153 * t), int(187 + 48 * t), int(106 * (1 - t)), 255]
        else:
            color = [255, 82, 82, 255]     # Red

        label_text = f"{dv_yr:.1f} m/s/yr"

        pkt: dict = {
            "id": f"sk-snap-{idx}",
            "name": f"Sat-{idx} ({label_text})",
            "position": {
                "cartographicDegrees": [lon_deg, lat_deg, alt_m],
            },
            "point": {
                "pixelSize": 5,
                "color": {"rgba": color},
            },
            "label": {
                "text": label_text,
                "font": "9pt sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 200]},
                "outlineWidth": 1,
                "style": "FILL_AND_OUTLINE",
                "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [10, 0]},
            },
        }
        packets.append(pkt)

    return packets


def cascade_evolution_packets(
    states: list[OrbitalState],
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    name: str = "Cascade SIR",
) -> list[dict]:
    """Animated SIR cascade model evolution shown as satellite coloring.

    Green (susceptible/operational), orange (infected/at risk), red (removed/debris).
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]
    if not states:
        return packets

    total_seconds = duration.total_seconds()
    step_seconds = _validate_step(step)
    num_steps = int(total_seconds / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)
    n_sats = len(states)

    # Shell parameters for SIR model
    mean_alt_km = sum(s.semi_major_axis_m - OrbitalConstants.R_EARTH for s in states) / (n_sats * 1000.0)
    shell_r = (OrbitalConstants.R_EARTH / 1000.0) + mean_alt_km
    shell_vol = 4.0 * math.pi * shell_r ** 2 * 50.0
    duration_yr = max(0.1, total_seconds / (365.25 * 86400.0))
    step_yr = max(1e-6, duration_yr / max(num_steps, 1))

    sir = compute_cascade_sir(
        shell_volume_km3=shell_vol, spatial_density_per_km3=n_sats / shell_vol,
        mean_collision_velocity_ms=10000.0, satellite_count=n_sats,
        duration_years=duration_yr, step_years=step_yr,
    )
    sir_times, sir_s, sir_i = sir.time_series_years, sir.susceptible, sir.infected
    total_pop = [sir_s[k] + sir_i[k] for k in range(len(sir_times))]
    end_time = epoch + duration

    for idx, state in enumerate(states):
        coords: list[float] = []
        step_times_dt: list[datetime] = []
        sat_colors: list[list[int]] = []
        for s in range(num_steps):
            t_offset = s * step_seconds
            target_time = epoch + timedelta(seconds=t_offset)
            _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, target_time)
            coords.extend([t_offset, lon_deg, lat_deg, alt_m])
            step_times_dt.append(target_time)
            # Find SIR index for this time
            t_yr = t_offset / (365.25 * 86400.0)
            si = 0
            for k in range(len(sir_times) - 1):
                if sir_times[k + 1] > t_yr:
                    break
                si = k + 1
            n_s = int(sir_s[si] / max(total_pop[si], 1.0) * n_sats)
            n_i = int(sir_i[si] / max(total_pop[si], 1.0) * n_sats)
            if idx < n_s:
                sat_colors.append([102, 187, 106, 255])
            elif idx < n_s + n_i:
                sat_colors.append([255, 165, 0, 255])
            else:
                sat_colors.append([255, 82, 82, 255])

        color_intervals = [
            {"interval": f"{_iso(step_times_dt[i])}/{_iso(step_times_dt[i + 1] if i + 1 < len(step_times_dt) else end_time)}",
             "rgba": sat_colors[i]}
            for i in range(len(step_times_dt))
        ]
        packets.append({
            "id": f"cascade-sat-{idx}", "name": f"Sat-{idx}",
            "position": {
                "epoch": _iso(epoch), "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE", "interpolationDegree": interp_degree,
            },
            "point": {"pixelSize": 5, "color": color_intervals},
        })

    return packets


def relative_motion_packets(
    state_a: OrbitalState,
    state_b: OrbitalState,
    epoch: datetime,
    duration: timedelta,
    step: timedelta,
    name: str = "Relative Motion",
) -> list[dict]:
    """Two-satellite trajectory with distance-annotated proximity polyline.

    Green (close) to red (far) polyline connecting both satellites.
    """
    packets: list[dict] = [_document_packet(name, epoch, duration)]
    total_seconds = duration.total_seconds()
    step_seconds = _validate_step(step)
    num_steps = int(total_seconds / step_seconds) + 1
    interp_degree = _interpolation_degree(num_steps)

    all_coords: list[list[float]] = [[], []]
    distances_km: list[float] = []
    for s in range(num_steps):
        t_offset = s * step_seconds
        target_time = epoch + timedelta(seconds=t_offset)
        for si, st in enumerate([state_a, state_b]):
            _, _, lat, lon, alt = _propagate_geodetic(st, target_time)
            all_coords[si].extend([t_offset, lon, lat, alt])
        rel = compute_relative_state(state_a, state_b, target_time)
        distances_km.append(math.sqrt(rel.x ** 2 + rel.y ** 2 + rel.z ** 2) / 1000.0)

    sat_colors = [[255, 50, 50, 255], [50, 100, 255, 255]]
    for sat_idx, (coords, sat_name) in enumerate(
        [(all_coords[0], "Sat-A"), (all_coords[1], "Sat-B")],
    ):
        color = sat_colors[sat_idx]
        packets.append({
            "id": f"relmotion-sat-{sat_idx}", "name": sat_name,
            "position": {
                "epoch": _iso(epoch), "cartographicDegrees": coords,
                "interpolationAlgorithm": "LAGRANGE", "interpolationDegree": interp_degree,
            },
            "point": {"pixelSize": 8, "color": {"rgba": color}},
            "label": {
                "text": sat_name, "font": "12pt sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 220]}, "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE", "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [12, 0]},
            },
            "path": {
                "leadTime": 300, "trailTime": 300, "resolution": 30,
                "material": {"solidColor": {"color": {"rgba": [color[0] // 2, color[1] // 2, color[2] // 2, 128]}}},
                "width": 1,
            },
        })

    packets.append({
        "id": "relmotion-proximity-line", "name": "Relative Distance",
        "polyline": {
            "positions": {"references": ["relmotion-sat-0#position", "relmotion-sat-1#position"]},
            "material": {"solidColor": {"color": {"rgba": [255, 255, 0, 200]}}},
            "width": 2, "arcType": "NONE",
        },
    })

    return packets


def maintenance_schedule_packets(
    states: list[OrbitalState],
    epoch: datetime,
    drag_config: DragConfig | None = None,
    density_func: object | None = None,
    name: str = "Maintenance",
) -> list[dict]:
    """Snapshot showing perturbation budget and annual dV per satellite.

    Green (low perturbation), yellow (moderate), red (high). Labels show dV budget.
    """
    packets: list[dict] = [_document_packet(name)]
    if not states:
        return packets

    default_drag = DragConfig(cd=2.2, area_m2=0.01, mass_kg=4.0)
    dc = drag_config or default_drag
    budgets = [
        compute_perturbation_budget(
            s, dc, density_func=density_func, epoch=epoch,
        )
        for s in states
    ]
    total_rates = [sum(e.total_rate for e in b.elements) for b in budgets]
    max_rate = max(total_rates) if total_rates else 1.0
    if max_rate == 0:
        max_rate = 1.0

    for idx, state in enumerate(states):
        _, _, lat_deg, lon_deg, alt_m = _propagate_geodetic(state, epoch)
        normalized = total_rates[idx] / max_rate
        if normalized < 0.33:
            color = [102, 187, 106, 255]
        elif normalized < 0.66:
            t = (normalized - 0.33) / 0.33
            color = [int(102 + 153 * t), int(187 + 48 * t), int(106 * (1 - t)), 255]
        else:
            color = [255, 82, 82, 255]

        sk_config = StationKeepingConfig(
            target_altitude_km=budgets[idx].altitude_km,
            inclination_deg=math.degrees(state.inclination_rad),
            drag_config=dc, isp_s=220.0,
            dry_mass_kg=4.0, propellant_mass_kg=0.5,
        )
        label_text = f"{compute_station_keeping_budget(sk_config, density_func=density_func, epoch=epoch).total_dv_per_year_ms:.1f} m/s/yr"

        packets.append({
            "id": f"maint-snap-{idx}",
            "name": f"Sat-{idx} ({label_text})",
            "position": {"cartographicDegrees": [lon_deg, lat_deg, alt_m]},
            "point": {"pixelSize": 5, "color": {"rgba": color}},
            "label": {
                "text": label_text, "font": "9pt sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 200]}, "outlineWidth": 1,
                "style": "FILL_AND_OUTLINE", "horizontalOrigin": "LEFT",
                "pixelOffset": {"cartesian2": [10, 0]},
            },
        })

    return packets
