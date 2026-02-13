# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Kerbal Space Program .sfs exporter.

Exports constellation satellites as VESSEL blocks in KSP's ConfigNode
format. The output file can be pasted into a persistent.sfs save file
inside the GAME { FLIGHTSTATE { } } node.

Orbital elements are scaled from Earth to Kerbin by default so
constellations appear at proportionally correct altitudes. Orbits that
would fall below Kerbin's atmosphere (70 km) are clamped to 80 km.

No external dependencies — only stdlib math.
"""
import math
from datetime import datetime, timezone

from humeris.domain.constellation import Satellite
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.ports.export import SatelliteExporter
from humeris.adapters.enrichment import compute_satellite_enrichment


_R_EARTH_M = OrbitalConstants.R_EARTH
_MU_EARTH = OrbitalConstants.MU_EARTH
_R_KERBIN_M = 600_000.0
_MU_KERBIN = 3.5316000e12
_KERBIN_ATMO_M = 70_000.0
_MIN_ALT_M = 80_000.0
_KERBIN_REF = 1
_SCALE = _R_KERBIN_M / _R_EARTH_M


def _build_vessel(
    sat: Satellite,
    index: int,
    sma_m: float,
    inc_deg: float,
    mass_tons: float | None,
) -> str:
    """Build a single VESSEL block in ConfigNode format."""
    raan_deg = sat.raan_deg % 360.0
    mna_rad = math.radians(sat.true_anomaly_deg % 360.0)

    pid = f"00000000-0000-0000-0000-{index + 1:012d}"
    uid = 1_000_000_000 + index

    part_mass = f"\t\tmass = {mass_tons:.6f}" if mass_tons is not None else "\t\tmass = 0.100000"

    return (
        "VESSEL\n"
        "{\n"
        f"\tpid = {pid}\n"
        f"\tname = {sat.name}\n"
        "\ttype = Probe\n"
        "\tsit = ORBITING\n"
        "\tlanded = False\n"
        "\tsplashed = False\n"
        "\tmet = 0\n"
        "\tlct = 0\n"
        "\troot = 0\n"
        "\tlat = 0\n"
        "\tlon = 0\n"
        "\talt = 0\n"
        "\tnrm = 0,0,0\n"
        "\trot = 0,0,0,1\n"
        "\tCoM = 0,0,0\n"
        "\tstg = 0\n"
        "\tprst = True\n"
        "\tref = 0\n"
        "\tctrl = True\n"
        "\tORBIT\n"
        "\t{\n"
        f"\t\tSMA = {sma_m:.1f}\n"
        "\t\tECC = 0.0000000\n"
        f"\t\tINC = {inc_deg:.7f}\n"
        "\t\tLPE = 0.0000000\n"
        f"\t\tLAN = {raan_deg:.7f}\n"
        f"\t\tMNA = {mna_rad:.7f}\n"
        "\t\tEPH = 0\n"
        f"\t\tREF = {_KERBIN_REF}\n"
        "\t}\n"
        "\tPART\n"
        "\t{\n"
        "\t\tname = probeCoreCube\n"
        f"\t\tcid = {uid}\n"
        f"\t\tuid = {uid}\n"
        f"\t\tmid = {uid}\n"
        "\t\tlaunchID = 0\n"
        "\t\tparent = 0\n"
        "\t\tposition = 0,0,0\n"
        "\t\trotation = 0,0,0,1\n"
        "\t\tmirror = 1,1,1\n"
        "\t\tsymMethod = Radial\n"
        "\t\tistg = 0\n"
        "\t\tdstg = 0\n"
        "\t\tsqor = -1\n"
        "\t\tsepI = 0\n"
        "\t\tsidx = -1\n"
        "\t\tattm = 0\n"
        "\t\tsrfN = None, -1\n"
        f"{part_mass}\n"
        "\t\tshielded = False\n"
        "\t\ttemp = 290\n"
        "\t\texpt = 0.5\n"
        "\t\tstate = 0\n"
        "\t\tconnected = True\n"
        "\t\tattached = True\n"
        "\t\tflag = \n"
        "\t\trTrf = probeCoreCube\n"
        "\t\tmodCost = 0\n"
        "\t\tMODULE\n"
        "\t\t{\n"
        "\t\t\tname = ModuleCommand\n"
        "\t\t\tisEnabled = True\n"
        "\t\t}\n"
        "\t}\n"
        "}"
    )


class KspExporter(SatelliteExporter):
    """Exports satellites as KSP VESSEL blocks for persistent.sfs.

    Generates ConfigNode-format VESSEL definitions with orbital elements
    and a minimal probeCoreCube part. Paste the output into your
    persistent.sfs inside GAME { FLIGHTSTATE { } }.

    By default, orbital elements are scaled from Earth to Kerbin so
    constellations appear at proportionally correct altitudes. Orbits
    below Kerbin's atmosphere are clamped to 80 km altitude.

    When drag_config is provided, satellite mass is set from mass_kg
    (converted to metric tons for KSP).
    """

    def __init__(self, drag_config=None, scale_to_kerbin: bool = True):
        self._drag_config = drag_config
        self._scale = scale_to_kerbin

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        mass_tons = None
        if self._drag_config:
            mass_tons = self._drag_config.mass_kg / 1000.0

        lines = [
            "// Humeris — KSP Vessel Export",
            "// Paste these VESSEL blocks into your persistent.sfs",
            "// inside the GAME { FLIGHTSTATE { } } node.",
            "",
        ]

        for i, sat in enumerate(satellites):
            sma_m, inc_deg = self._compute_orbit(sat)
            enrich = compute_satellite_enrichment(sat, epoch)
            lines.append(
                f"// Altitude: {enrich.altitude_km:.1f} km\n"
                f"// Inclination: {enrich.inclination_deg:.2f} deg\n"
                f"// Period: {enrich.orbital_period_min:.2f} min\n"
                f"// Beta angle: {enrich.beta_angle_deg:.2f} deg\n"
                f"// Atm. density: {enrich.atmospheric_density_kg_m3:.3e} kg/m3\n"
                f"// L-shell: {enrich.l_shell:.2f}"
            )
            lines.append(_build_vessel(sat, i, sma_m, inc_deg, mass_tons))
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return len(satellites)

    def _compute_orbit(self, sat: Satellite) -> tuple[float, float]:
        """Compute SMA and inclination, optionally scaled to Kerbin."""
        px, py, pz = sat.position_eci
        r_earth = math.sqrt(px**2 + py**2 + pz**2)

        vx, vy, vz = sat.velocity_eci
        hx = py * vz - pz * vy
        hy = pz * vx - px * vz
        hz = px * vy - py * vx
        h_mag = math.sqrt(hx**2 + hy**2 + hz**2)
        inc_deg = math.degrees(math.acos(hz / h_mag)) if h_mag > 0 else 0.0

        if self._scale:
            sma_m = r_earth * _SCALE
            alt_m = sma_m - _R_KERBIN_M
            if alt_m < _MIN_ALT_M:
                sma_m = _R_KERBIN_M + _MIN_ALT_M
        else:
            sma_m = r_earth

        return sma_m, inc_deg
