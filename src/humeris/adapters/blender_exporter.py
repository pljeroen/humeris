# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Blender Python script exporter.

Generates a standalone .py script that creates a constellation
visualization when run inside Blender. Does NOT import bpy itself —
it writes Python source code that uses the Blender API.

External dependencies (file I/O) are confined to this adapter.
"""
import math
from datetime import datetime

from humeris.domain.constellation import Satellite
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.ports.export import SatelliteExporter


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """Cross product of two 3-vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Dot product of two 3-vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: tuple[float, float, float]) -> float:
    """Magnitude of a 3-vector."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _scale(v: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    """Scale a 3-vector."""
    return (v[0] * s, v[1] * s, v[2] * s)


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """Add two 3-vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _rotate_around_axis(
    v: tuple[float, float, float],
    axis: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float]:
    """Rotate vector v around unit axis by angle_rad (Rodrigues' formula)."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dot_av = _dot(axis, v)
    cross_av = _cross(axis, v)
    return _add(
        _add(_scale(v, cos_a), _scale(cross_av, sin_a)),
        _scale(axis, dot_av * (1.0 - cos_a)),
    )


def _orbit_points_km(
    position_eci: tuple[float, float, float],
    velocity_eci: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """Generate 37 orbit ring points by rotating position around angular momentum axis.

    Points are at 10-degree steps from 0 to 360 inclusive. Positions are
    returned in km (ECI metres divided by 1000).
    """
    h = _cross(position_eci, velocity_eci)
    h_mag = _norm(h)
    if h_mag == 0.0:
        pos_km = (
            position_eci[0] / 1000.0,
            position_eci[1] / 1000.0,
            position_eci[2] / 1000.0,
        )
        return [pos_km] * 37

    h_unit = _scale(h, 1.0 / h_mag)

    r_mag = _norm(position_eci)
    r_unit = _scale(position_eci, 1.0 / r_mag) if r_mag > 0 else position_eci
    r_orbit = _scale(r_unit, r_mag)

    points: list[tuple[float, float, float]] = []
    for step in range(37):
        angle_rad = math.radians(step * 10.0)
        rotated = _rotate_around_axis(r_orbit, h_unit, angle_rad)
        points.append((
            rotated[0] / 1000.0,
            rotated[1] / 1000.0,
            rotated[2] / 1000.0,
        ))
    return points


class BlenderExporter(SatelliteExporter):
    """Generates a .py script for Blender constellation visualization.

    The generated script uses the Blender Python API (bpy) to create
    Earth, satellite spheres, and orbit curves in a 3D scene.

    Args:
        earth_radius_units: Blender radius for Earth sphere.
        sat_radius_units: Blender radius for satellite ico spheres.
    """

    def __init__(
        self,
        earth_radius_units: float = 6.371,
        sat_radius_units: float = 0.05,
    ) -> None:
        self._earth_radius = earth_radius_units
        self._sat_radius = sat_radius_units

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        lines: list[str] = []

        # Header
        lines.append("import bpy")
        lines.append("import math")
        lines.append("")

        # Clear scene
        lines.append("# Clear scene")
        lines.append("bpy.ops.object.select_all(action='SELECT')")
        lines.append("bpy.ops.object.delete()")
        lines.append("")

        # Create Earth
        lines.append("# Create Earth")
        lines.append(
            f"bpy.ops.mesh.primitive_uv_sphere_add("
            f"radius={self._earth_radius}, "
            f"location=(0, 0, 0), "
            f"segments=64, ring_count=32)"
        )
        lines.append("earth = bpy.context.active_object")
        lines.append('earth.name = "Earth"')
        lines.append("")

        # Create satellites and orbit curves
        for sat in satellites:
            pos_km = (
                sat.position_eci[0] / 1000.0,
                sat.position_eci[1] / 1000.0,
                sat.position_eci[2] / 1000.0,
            )

            # Satellite ico sphere
            lines.append(f"# Create satellite: {sat.name}")
            lines.append(
                f"bpy.ops.mesh.primitive_ico_sphere_add("
                f"radius={self._sat_radius}, "
                f"location=({pos_km[0]:.6f}, {pos_km[1]:.6f}, {pos_km[2]:.6f}), "
                f"subdivisions=1)"
            )
            lines.append("sat = bpy.context.active_object")
            lines.append(f'sat.name = "{sat.name}"')
            lines.append("")

            # Orbit curve
            orbit_pts = _orbit_points_km(sat.position_eci, sat.velocity_eci)
            safe_name = sat.name.replace('"', '\\"')
            curve_var = f"orbit_{sat.name}"

            lines.append(f"# Create orbit curve for: {sat.name}")
            lines.append(
                f"curve_data = bpy.data.curves.new('{curve_var}', 'CURVE')"
            )
            lines.append("curve_data.dimensions = '3D'")
            lines.append("spline = curve_data.splines.new('NURBS')")
            lines.append("spline.points.add(36)")
            for idx, pt in enumerate(orbit_pts):
                lines.append(
                    f"spline.points[{idx}].co = ({pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f}, 1.0)"
                )
            lines.append("spline.use_cyclic_u = True")
            lines.append(
                f"orbit_obj = bpy.data.objects.new('{curve_var}', curve_data)"
            )
            lines.append("bpy.context.collection.objects.link(orbit_obj)")
            lines.append("")

        script = "\n".join(lines) + "\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(script)

        return len(satellites)
