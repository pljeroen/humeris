# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Celestia .ssc catalog exporter.

Exports constellation satellites as a Celestia Solar System Catalog
file (.ssc). Each satellite is a spacecraft object orbiting Earth
with Keplerian orbital elements in Celestia units (km, days, degrees).

Optional enrichment with physical properties (mass, radius) from
DragConfig when provided.

No external dependencies — only stdlib math.
"""
import math
from datetime import datetime, timezone

from humeris.domain.constellation import Satellite
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.ports.export import SatelliteExporter
from humeris.adapters.enrichment import compute_satellite_enrichment


_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_MU = OrbitalConstants.MU_EARTH


class CelestiaExporter(SatelliteExporter):
    """Exports satellites as a Celestia .ssc catalog file.

    Each satellite becomes a spacecraft object parented to Sol/Earth
    with Keplerian orbital elements in an EllipticalOrbit block.

    When drag_config is provided, adds Mass (in kg) and derives
    Radius (in km) from cross-sectional area. Without drag_config
    the default Radius is 0.001 km.
    """

    def __init__(self, drag_config=None):
        self._drag_config = drag_config

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        lines: list[str] = []
        for sat in satellites:
            enrich = compute_satellite_enrichment(sat, epoch)
            lines.append(
                f"# Altitude: {enrich.altitude_km:.1f} km\n"
                f"# Inclination: {enrich.inclination_deg:.2f} deg\n"
                f"# Period: {enrich.orbital_period_min:.2f} min\n"
                f"# Beta angle: {enrich.beta_angle_deg:.2f} deg\n"
                f"# Atm. density: {enrich.atmospheric_density_kg_m3:.3e} kg/m3\n"
                f"# L-shell: {enrich.l_shell:.2f}"
            )
            lines.append(self._format_satellite(sat, epoch))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return len(satellites)

    def _format_satellite(self, sat: Satellite, epoch: datetime | None) -> str:
        # Derive orbital elements from ECI state vectors
        px, py, pz = sat.position_eci
        r_mag = math.sqrt(px**2 + py**2 + pz**2)

        # Semi-major axis in km
        a_km = r_mag / 1000.0

        # Eccentricity (circular orbit assumption)
        e = 0.0

        # Inclination from angular momentum vector
        vx, vy, vz = sat.velocity_eci
        hx = py * vz - pz * vy
        hy = pz * vx - px * vz
        hz = px * vy - py * vx
        h_mag = math.sqrt(hx**2 + hy**2 + hz**2)
        inc_deg = math.degrees(math.acos(hz / h_mag)) if h_mag > 0 else 0.0

        # RAAN and true anomaly from metadata
        raan_deg = sat.raan_deg % 360.0
        nu_deg = sat.true_anomaly_deg % 360.0

        # Orbital period in days (Kepler's 3rd law)
        period_s = 2 * math.pi * math.sqrt(r_mag**3 / _MU)
        period_days = period_s / 86400.0

        # Epoch as Julian Date
        effective_epoch = epoch if epoch is not None else _J2000
        jd = 2451545.0 + (effective_epoch - _J2000).total_seconds() / 86400.0

        # Radius in km
        if self._drag_config is not None:
            radius_m = math.sqrt(self._drag_config.area_m2 / math.pi)
            radius_km = radius_m / 1000.0
        else:
            radius_km = 0.001

        # Build the .ssc block
        parts = [
            f'"{sat.name}" "Sol/Earth" {{',
            f'    Class "spacecraft"',
            f'    Mesh ""',
            f'    Radius {radius_km:.6e}',
        ]

        # Optional mass from DragConfig
        if self._drag_config is not None:
            parts.append(f"    Mass {self._drag_config.mass_kg:.2f}")

        parts.extend([
            "",
            "    EllipticalOrbit {",
            f"        Epoch {jd:.4f}",
            f"        Period {period_days:.10e}",
            f"        SemiMajorAxis {a_km:.4f}",
            f"        Eccentricity {e:.6f}",
            f"        Inclination {inc_deg:.4f}",
            f"        AscendingNode {raan_deg:.4f}",
            f"        ArgOfPericenter 0.0000",
            f"        MeanAnomaly {nu_deg:.4f}",
            "    }",
            "}",
            "",
        ])

        return "\n".join(parts)
