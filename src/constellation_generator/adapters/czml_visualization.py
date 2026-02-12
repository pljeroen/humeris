"""Advanced CZML visualization packets for CesiumJS.

Eclipse-aware coloring, sensor footprints, ground station access,
conjunction replay, coverage evolution, and J2 precession timelapse.

Uses only stdlib math/datetime + internal domain/adapter imports.
"""

import math
from datetime import datetime, timedelta

from constellation_generator.domain.propagation import (
    OrbitalState,
    derive_orbital_state,
    propagate_to,
)
from constellation_generator.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    ecef_to_geodetic,
)
from constellation_generator.domain.solar import sun_position_eci
from constellation_generator.domain.eclipse import is_eclipsed, EclipseType
from constellation_generator.domain.sensor import SensorConfig, compute_swath_width
from constellation_generator.domain.observation import GroundStation
from constellation_generator.domain.access_windows import compute_access_windows
from constellation_generator.domain.coverage import compute_coverage_snapshot
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.adapters.czml_exporter import (
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
            "description": _satellite_description(state),
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
