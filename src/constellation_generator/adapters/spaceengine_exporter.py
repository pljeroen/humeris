# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""SpaceEngine .sc catalog exporter.

Exports constellation satellites as a SpaceEngine catalog file (.sc).
Each satellite is a Moon object orbiting Earth with Keplerian orbital
elements in SpaceEngine units (AU, years, degrees).

Optional enrichment with physical properties (mass, radius) from
DragConfig when provided.

No external dependencies — only stdlib math.
"""
import math
from datetime import datetime, timezone

from constellation_generator.domain.constellation import Satellite
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.ports.export import SatelliteExporter


_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_R_EARTH_M = OrbitalConstants.R_EARTH
_MU = OrbitalConstants.MU_EARTH
_AU_M = 1.496e11  # 1 AU in metres
_YEAR_S = 365.25 * 86400  # Julian year in seconds
_EARTH_MASS_KG = 5.972e24


class SpaceEngineExporter(SatelliteExporter):
    """Exports satellites as a SpaceEngine .sc catalog file.

    Each satellite becomes a Moon object parented to Earth with
    Keplerian orbital elements. The RefPlane is "Equator" (Earth's
    equatorial plane) since orbital elements are Earth-centric.

    When drag_config is provided, adds Mass (in Earth masses) and
    Radius (in km, derived from cross-sectional area).
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
            lines.append(self._format_satellite(sat))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return len(satellites)

    def _format_satellite(self, sat: Satellite) -> str:
        # Derive orbital elements
        px, py, pz = sat.position_eci
        r_mag = math.sqrt(px**2 + py**2 + pz**2)

        # Semi-major axis in AU
        a_au = r_mag / _AU_M

        # Eccentricity
        e = 0.0

        # Inclination from angular momentum
        vx, vy, vz = sat.velocity_eci
        hx = py * vz - pz * vy
        hy = pz * vx - px * vz
        hz = px * vy - py * vx
        h_mag = math.sqrt(hx**2 + hy**2 + hz**2)
        inc_deg = math.degrees(math.acos(hz / h_mag)) if h_mag > 0 else 0.0

        # RAAN and true anomaly from metadata
        raan_deg = sat.raan_deg % 360.0
        nu_deg = sat.true_anomaly_deg % 360.0

        # Orbital period in years
        period_s = 2 * math.pi * math.sqrt(r_mag**3 / _MU)
        period_yr = period_s / _YEAR_S

        # Build the .sc block
        parts = [
            f'Moon "{sat.name}"',
            "{",
            f'\tParentBody "Earth"',
        ]

        # Physical properties from DragConfig
        if self._drag_config is not None:
            mass_earth = self._drag_config.mass_kg / _EARTH_MASS_KG
            parts.append(f"\tMass {mass_earth:.6e}")
            radius_m = math.sqrt(self._drag_config.area_m2 / math.pi)
            radius_km = radius_m / 1000.0
            parts.append(f"\tRadius {radius_km:.6e}")

        parts.extend([
            "",
            "\tOrbit",
            "\t{",
            f'\t\tRefPlane "Equator"',
            f"\t\tSemiMajorAxis {a_au:.10e}",
            f"\t\tPeriod {period_yr:.10e}",
            f"\t\tEccentricity {e:.6f}",
            f"\t\tInclination {inc_deg:.4f}",
            f"\t\tAscendingNode {raan_deg:.4f}",
            f"\t\tArgOfPericenter 0.0000",
            f"\t\tMeanAnomaly {nu_deg:.4f}",
            "\t}",
            "}",
            "",
        ])

        return "\n".join(parts)
