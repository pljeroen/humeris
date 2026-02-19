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
from datetime import datetime, timezone

from humeris.domain.constellation import Satellite
from humeris.domain.coordinate_frames import gmst_rad
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.ports.export import SatelliteExporter
from humeris.adapters.enrichment import compute_satellite_enrichment


_KML_NS = "http://www.opengis.net/kml/2.2"

_R_EARTH = OrbitalConstants.R_EARTH


def _eci_to_geodetic_spherical(
    x: float, y: float, z: float,
    gmst_angle_rad: float = 0.0,
) -> tuple[float, float, float]:
    """
    Convert ECI position to geodetic (lat, lon, alt) using spherical Earth.

    Applies GMST rotation to convert from ECI to ECEF before extracting
    geographic longitude.

    Args:
        x, y, z: ECI position in metres.
        gmst_angle_rad: Greenwich Mean Sidereal Time in radians.

    Returns:
        (lat_deg, lon_deg, alt_m) — latitude/longitude in degrees,
        altitude in metres above mean spherical Earth.
    """
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        return 0.0, 0.0, -_R_EARTH
    lat_deg = math.degrees(math.asin(z / r))
    # Rotate ECI → ECEF by subtracting GMST from the inertial longitude
    lon_eci_rad = math.atan2(y, x)
    lon_ecef_rad = lon_eci_rad - gmst_angle_rad
    lon_deg = math.degrees(lon_ecef_rad)
    # Normalize to [-180, 180]
    lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0
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
    gmst_angle_rad: float = 0.0,
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

        lat_deg, lon_deg, alt_m = _eci_to_geodetic_spherical(x, y, z, gmst_angle_rad)
        coords.append(f"{lon_deg:.6f},{lat_deg:.6f},{alt_m:.1f}")

    return coords


def _sub_element(parent: ET.Element, tag: str, text: str | None = None, **attribs: str) -> ET.Element:
    """Create a sub-element with optional text and attributes."""
    elem = ET.SubElement(parent, tag, **attribs)
    if text is not None:
        elem.text = text
    return elem


class KmlExporter(SatelliteExporter):
    """Exports satellite positions and orbit paths as KML for Google Earth.

    Args:
        name: Document name in KML output.
        include_orbits: Include orbit path LineStrings (default True).
        include_planes: Organize Placemarks into per-plane Folder elements.
        include_isl: Include ISL topology as LineStrings between satellites.
        max_isl_range_km: Maximum ISL range in km for topology computation.
    """

    def __init__(
        self,
        name: str = "Constellation",
        include_orbits: bool = True,
        include_planes: bool = False,
        include_isl: bool = False,
        max_isl_range_km: float = 5000.0,
    ) -> None:
        self._name = name
        self._include_orbits = include_orbits
        self._include_planes = include_planes
        self._include_isl = include_isl
        self._max_isl_range_km = max_isl_range_km

    def _build_sat_folder(
        self,
        sat: Satellite,
        epoch: datetime | None,
        gmst_angle_rad: float = 0.0,
    ) -> ET.Element:
        """Build a Folder element for one satellite."""
        folder = ET.Element(f"{{{_KML_NS}}}Folder")
        _sub_element(folder, f"{{{_KML_NS}}}name", sat.name)

        # Position Placemark
        pm_pos = _sub_element(folder, f"{{{_KML_NS}}}Placemark")
        _sub_element(pm_pos, f"{{{_KML_NS}}}name", sat.name)
        _sub_element(pm_pos, f"{{{_KML_NS}}}styleUrl", "#sat-style")

        px, py, pz = sat.position_eci
        lat_deg, lon_deg, alt_m = _eci_to_geodetic_spherical(px, py, pz, gmst_angle_rad)

        # Enrichment data
        enrich = compute_satellite_enrichment(sat, epoch)

        # Description table
        desc_html = (
            f"<![CDATA["
            f"<table>"
            f"<tr><td>Altitude</td><td>{enrich.altitude_km:.1f} km</td></tr>"
            f"<tr><td>Inclination</td><td>{enrich.inclination_deg:.2f}&deg;</td></tr>"
            f"<tr><td>Period</td><td>{enrich.orbital_period_min:.2f} min</td></tr>"
            f"<tr><td>Beta angle</td><td>{enrich.beta_angle_deg:.2f}&deg;</td></tr>"
            f"<tr><td>Atm. density</td><td>{enrich.atmospheric_density_kg_m3:.3e} kg/m&sup3;</td></tr>"
            f"<tr><td>L-shell</td><td>{enrich.l_shell:.2f}</td></tr>"
            f"</table>"
            f"]]>"
        )
        _sub_element(pm_pos, f"{{{_KML_NS}}}description", desc_html)

        # ExtendedData block
        ext_data = _sub_element(pm_pos, f"{{{_KML_NS}}}ExtendedData")
        for field_name, field_val in [
            ("altitude_km", f"{enrich.altitude_km:.3f}"),
            ("inclination_deg", f"{enrich.inclination_deg:.4f}"),
            ("orbital_period_min", f"{enrich.orbital_period_min:.4f}"),
            ("beta_angle_deg", f"{enrich.beta_angle_deg:.4f}"),
            ("atmospheric_density_kg_m3", f"{enrich.atmospheric_density_kg_m3:.6e}"),
            ("l_shell", f"{enrich.l_shell:.4f}"),
        ]:
            data_elem = _sub_element(ext_data, f"{{{_KML_NS}}}Data", name=field_name)
            _sub_element(data_elem, f"{{{_KML_NS}}}value", field_val)

        point = _sub_element(pm_pos, f"{{{_KML_NS}}}Point")
        _sub_element(point, f"{{{_KML_NS}}}altitudeMode", "absolute")
        _sub_element(
            point,
            f"{{{_KML_NS}}}coordinates",
            f"{lon_deg:.6f},{lat_deg:.6f},{alt_m:.1f}",
        )

        # Orbit path Placemark (optional)
        if self._include_orbits:
            pm_orbit = _sub_element(folder, f"{{{_KML_NS}}}Placemark")
            _sub_element(pm_orbit, f"{{{_KML_NS}}}name", f"{sat.name} orbit")
            _sub_element(pm_orbit, f"{{{_KML_NS}}}styleUrl", "#sat-style")

            orbit_coords = _orbit_path_coords(sat.position_eci, sat.velocity_eci, gmst_angle_rad)

            linestring = _sub_element(pm_orbit, f"{{{_KML_NS}}}LineString")
            _sub_element(linestring, f"{{{_KML_NS}}}altitudeMode", "absolute")
            _sub_element(
                linestring,
                f"{{{_KML_NS}}}coordinates",
                " ".join(orbit_coords),
            )

        return folder

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        ET.register_namespace("", _KML_NS)

        _j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        effective_epoch = epoch or _j2000
        gmst = gmst_rad(effective_epoch)

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

        if self._include_planes:
            # Group satellites by plane_index
            planes: dict[int, list[Satellite]] = {}
            for sat in satellites:
                planes.setdefault(sat.plane_index, []).append(sat)
            for plane_idx in sorted(planes):
                plane_folder = _sub_element(doc, f"{{{_KML_NS}}}Folder")
                _sub_element(plane_folder, f"{{{_KML_NS}}}name", f"Plane {plane_idx}")
                for sat in planes[plane_idx]:
                    sat_folder = self._build_sat_folder(sat, epoch, gmst)
                    plane_folder.append(sat_folder)
        else:
            for sat in satellites:
                sat_folder = self._build_sat_folder(sat, epoch, gmst)
                doc.append(sat_folder)

        # ISL topology layer (optional)
        if self._include_isl and satellites:
            from humeris.domain.propagation import derive_orbital_state
            from humeris.domain.inter_satellite_links import compute_isl_topology

            try:
                states = [derive_orbital_state(s, effective_epoch) for s in satellites]
            except (ValueError, ZeroDivisionError):
                states = []
            topology = compute_isl_topology(
                states, effective_epoch, max_range_km=self._max_isl_range_km,
            ) if len(states) == len(satellites) else None

            isl_folder = _sub_element(doc, f"{{{_KML_NS}}}Folder")
            _sub_element(isl_folder, f"{{{_KML_NS}}}name", "ISL Topology")

            # ISL style
            isl_style = _sub_element(doc, f"{{{_KML_NS}}}Style", id="isl-style")
            isl_line = _sub_element(isl_style, f"{{{_KML_NS}}}LineStyle")
            _sub_element(isl_line, f"{{{_KML_NS}}}color", "ff00ff00")
            _sub_element(isl_line, f"{{{_KML_NS}}}width", "1")

            for link in (topology.links if topology is not None else []):
                if link.distance_m / 1000.0 > self._max_isl_range_km:
                    continue
                if link.is_blocked:
                    continue
                sat_a = satellites[link.sat_idx_a]
                sat_b = satellites[link.sat_idx_b]
                a_lat, a_lon, a_alt = _eci_to_geodetic_spherical(*sat_a.position_eci, gmst)
                b_lat, b_lon, b_alt = _eci_to_geodetic_spherical(*sat_b.position_eci, gmst)

                pm_isl = _sub_element(isl_folder, f"{{{_KML_NS}}}Placemark")
                _sub_element(pm_isl, f"{{{_KML_NS}}}name",
                             f"ISL {sat_a.name} - {sat_b.name}")
                _sub_element(pm_isl, f"{{{_KML_NS}}}styleUrl", "#isl-style")
                ls = _sub_element(pm_isl, f"{{{_KML_NS}}}LineString")
                _sub_element(ls, f"{{{_KML_NS}}}altitudeMode", "absolute")
                _sub_element(
                    ls, f"{{{_KML_NS}}}coordinates",
                    f"{a_lon:.6f},{a_lat:.6f},{a_alt:.1f} "
                    f"{b_lon:.6f},{b_lat:.6f},{b_alt:.1f}",
                )

        tree = ET.ElementTree(kml)
        ET.indent(tree, space="  ")
        tree.write(
            path,
            encoding="UTF-8",
            xml_declaration=True,
        )

        return len(satellites)
