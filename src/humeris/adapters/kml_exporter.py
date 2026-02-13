# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
KML satellite exporter for Google Earth.

Exports satellite positions and orbit paths as KML with Placemarks.
Uses spherical Earth approximation for ECI-to-geodetic conversion.
External dependencies (xml, file I/O) are confined to this adapter.
"""
import math
import xml.etree.ElementTree as ET
from datetime import datetime

from humeris.domain.constellation import Satellite
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.ports.export import SatelliteExporter


_KML_NS = "http://www.opengis.net/kml/2.2"

_R_EARTH = OrbitalConstants.R_EARTH


def _eci_to_geodetic_spherical(
    x: float, y: float, z: float,
) -> tuple[float, float, float]:
    """
    Convert ECI position to geodetic (lat, lon, alt) using spherical Earth.

    Args:
        x, y, z: ECI position in metres.

    Returns:
        (lat_deg, lon_deg, alt_m) — latitude/longitude in degrees,
        altitude in metres above mean spherical Earth.
    """
    r = math.sqrt(x * x + y * y + z * z)
    lat_deg = math.degrees(math.asin(z / r))
    lon_deg = math.degrees(math.atan2(y, x))
    alt_m = r - _R_EARTH
    return lat_deg, lon_deg, alt_m


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """Cross product of two 3-vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return unit vector."""
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def _orbit_path_coords(
    position_eci: tuple[float, float, float],
    velocity_eci: tuple[float, float, float],
) -> list[str]:
    """
    Compute 37 coordinate strings (36 orbit points + closing point).

    Uses the orbital plane defined by r and v to rotate the position
    vector in 10-degree steps around the full orbit.

    Returns:
        List of "lon,lat,alt" strings.
    """
    r_vec = position_eci
    v_vec = velocity_eci

    r_mag = math.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2)

    # Orbital plane unit vectors
    e_r = _normalize(r_vec)
    h = _cross(r_vec, v_vec)
    e_h = _normalize(h)
    e_t = _cross(e_h, e_r)

    coords: list[str] = []
    # 0, 10, 20, ..., 350, then 0 again to close
    angles_deg = list(range(0, 360, 10)) + [0]

    for angle_deg in angles_deg:
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        x = r_mag * (cos_t * e_r[0] + sin_t * e_t[0])
        y = r_mag * (cos_t * e_r[1] + sin_t * e_t[1])
        z = r_mag * (cos_t * e_r[2] + sin_t * e_t[2])

        lat_deg, lon_deg, alt_m = _eci_to_geodetic_spherical(x, y, z)
        coords.append(f"{lon_deg:.6f},{lat_deg:.6f},{alt_m:.1f}")

    return coords


def _sub_element(parent: ET.Element, tag: str, text: str | None = None, **attribs: str) -> ET.Element:
    """Create a sub-element with optional text and attributes."""
    elem = ET.SubElement(parent, tag, **attribs)
    if text is not None:
        elem.text = text
    return elem


class KmlExporter(SatelliteExporter):
    """Exports satellite positions and orbit paths as KML for Google Earth."""

    def __init__(self, name: str = "Constellation") -> None:
        self._name = name

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        # Register namespace to avoid ns0: prefix in output
        ET.register_namespace("", _KML_NS)

        kml = ET.Element(f"{{{_KML_NS}}}kml")
        doc = _sub_element(kml, f"{{{_KML_NS}}}Document")
        _sub_element(doc, f"{{{_KML_NS}}}name", self._name)

        # Shared style
        style = _sub_element(doc, f"{{{_KML_NS}}}Style", id="sat-style")
        icon_style = _sub_element(style, f"{{{_KML_NS}}}IconStyle")
        _sub_element(icon_style, f"{{{_KML_NS}}}scale", "0.5")
        line_style = _sub_element(style, f"{{{_KML_NS}}}LineStyle")
        _sub_element(line_style, f"{{{_KML_NS}}}color", "ff0000ff")
        _sub_element(line_style, f"{{{_KML_NS}}}width", "1")

        for sat in satellites:
            folder = _sub_element(doc, f"{{{_KML_NS}}}Folder")
            _sub_element(folder, f"{{{_KML_NS}}}name", sat.name)

            # Position Placemark
            pm_pos = _sub_element(folder, f"{{{_KML_NS}}}Placemark")
            _sub_element(pm_pos, f"{{{_KML_NS}}}name", sat.name)
            _sub_element(pm_pos, f"{{{_KML_NS}}}styleUrl", "#sat-style")

            px, py, pz = sat.position_eci
            lat_deg, lon_deg, alt_m = _eci_to_geodetic_spherical(px, py, pz)

            point = _sub_element(pm_pos, f"{{{_KML_NS}}}Point")
            _sub_element(point, f"{{{_KML_NS}}}altitudeMode", "absolute")
            _sub_element(
                point,
                f"{{{_KML_NS}}}coordinates",
                f"{lon_deg:.6f},{lat_deg:.6f},{alt_m:.1f}",
            )

            # Orbit path Placemark
            pm_orbit = _sub_element(folder, f"{{{_KML_NS}}}Placemark")
            _sub_element(pm_orbit, f"{{{_KML_NS}}}name", f"{sat.name} orbit")
            _sub_element(pm_orbit, f"{{{_KML_NS}}}styleUrl", "#sat-style")

            orbit_coords = _orbit_path_coords(sat.position_eci, sat.velocity_eci)

            linestring = _sub_element(pm_orbit, f"{{{_KML_NS}}}LineString")
            _sub_element(linestring, f"{{{_KML_NS}}}altitudeMode", "absolute")
            _sub_element(
                linestring,
                f"{{{_KML_NS}}}coordinates",
                " ".join(orbit_coords),
            )

        tree = ET.ElementTree(kml)
        ET.indent(tree, space="  ")
        tree.write(
            path,
            encoding="UTF-8",
            xml_declaration=True,
        )

        return len(satellites)
