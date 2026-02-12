# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Universe Sandbox .ubox exporter.

Exports constellation satellites as a Universe Sandbox simulation file.
The .ubox format is a ZIP archive containing XML that defines bodies
with Keplerian orbital elements orbiting Earth.

Optional enrichment with physical properties (mass, diameter) from
DragConfig when provided.

No external dependencies — only stdlib xml/zipfile/math/io.
"""
import io
import math
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone

from constellation_generator.domain.constellation import Satellite
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.ports.export import SatelliteExporter


_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_R_EARTH_M = OrbitalConstants.R_EARTH
_MU = OrbitalConstants.MU_EARTH


class UboxExporter(SatelliteExporter):
    """Exports satellites as a Universe Sandbox .ubox simulation file.

    Basic mode: Keplerian orbital elements (semi-major axis, eccentricity,
    inclination, RAAN, argument of perigee, mean anomaly) for each
    satellite orbiting Earth.

    Enhanced mode: When drag_config is provided, adds physical properties
    (mass in 10^20 kg, diameter in km derived from cross-sectional area).
    """

    def __init__(self, drag_config=None):
        """Initialize exporter.

        Args:
            drag_config: Optional DragConfig for physical properties
                (mass_kg, area_m2 → diameter).
        """
        self._drag_config = drag_config

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        effective_epoch = epoch or _J2000
        xml_bytes = self._build_xml(satellites, effective_epoch)
        self._write_ubox(xml_bytes, path)
        return len(satellites)

    def _build_xml(
        self,
        satellites: list[Satellite],
        epoch: datetime,
    ) -> bytes:
        root = ET.Element("Simulation")

        # Settings
        date_str = epoch.strftime("%Y-%m-%d")
        ET.SubElement(root, "Settings",
                      date=date_str,
                      focus="Earth",
                      trailsegments="500",
                      labels="on")

        # Earth as central body
        earth = ET.SubElement(root, "Body")
        ET.SubElement(earth, "Object").text = "Earth"

        # Satellite bodies
        for sat in satellites:
            self._add_satellite_body(root, sat)

        tree = ET.ElementTree(root)
        buf = io.BytesIO()
        tree.write(buf, encoding="utf-8", xml_declaration=True)
        return buf.getvalue()

    def _add_satellite_body(
        self,
        root: ET.Element,
        sat: Satellite,
    ) -> None:
        body = ET.SubElement(root, "Body")
        ET.SubElement(body, "Name").text = sat.name

        # Derive Keplerian elements from ECI state
        px, py, pz = sat.position_eci
        r_mag = math.sqrt(px**2 + py**2 + pz**2)

        # Semi-major axis (circular orbit assumption: a = r)
        a_km = r_mag / 1000.0

        # Eccentricity (Walker shells are circular)
        e = 0.0

        # Inclination from angular momentum vector
        vx, vy, vz = sat.velocity_eci
        hx = py * vz - pz * vy
        hy = pz * vx - px * vz
        hz = px * vy - py * vx
        h_mag = math.sqrt(hx**2 + hy**2 + hz**2)
        inc_deg = math.degrees(math.acos(hz / h_mag)) if h_mag > 0 else 0.0

        # RAAN and true anomaly from satellite metadata
        raan_deg = sat.raan_deg % 360.0
        nu_deg = sat.true_anomaly_deg % 360.0

        # Argument of perigee (0 for circular orbits)
        arg_perigee_deg = 0.0

        # Mean anomaly = true anomaly for circular orbits
        mean_anomaly_deg = nu_deg

        ET.SubElement(body, "Orbit",
                      body="Earth",
                      a=f"{a_km:.3f}",
                      e=f"{e:.6f}",
                      i=f"{inc_deg:.4f}",
                      node=f"{raan_deg:.4f}",
                      peri=f"{arg_perigee_deg:.4f}",
                      m=f"{mean_anomaly_deg:.4f}")

        # Physical properties from DragConfig
        if self._drag_config is not None:
            # Mass: Universe Sandbox uses 10^20 kg
            mass_1e20 = self._drag_config.mass_kg / 1e20
            ET.SubElement(body, "Mass").text = f"{mass_1e20:.6e}"

            # Diameter: derive from cross-sectional area (A = pi * r^2)
            # diameter_m = 2 * sqrt(area / pi)
            # Convert to km for Universe Sandbox
            diameter_m = 2.0 * math.sqrt(self._drag_config.area_m2 / math.pi)
            diameter_km = diameter_m / 1000.0
            ET.SubElement(body, "Diameter").text = f"{diameter_km:.6e}"

    def _write_ubox(self, xml_bytes: bytes, path: str) -> None:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("simulation.xml", xml_bytes)
