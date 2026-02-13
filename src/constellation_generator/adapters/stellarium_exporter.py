# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Stellarium-compatible TLE exporter.

Generates Two-Line Element sets that can be imported by Stellarium's
satellite plugin, as well as STK, GMAT, and any TLE-consuming software.

TLE format: https://celestrak.org/NORAD/documentation/tle-fmt.php
Each TLE entry is three lines: satellite name, line 1, line 2.
Lines 1 and 2 are exactly 69 characters with a modulo-10 checksum.
"""
import math
from datetime import datetime, timezone

from constellation_generator.domain.constellation import Satellite
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.ports.export import SatelliteExporter


_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _tle_checksum(line: str) -> int:
    """Compute TLE checksum: sum digits ('-' counts as 1), mod 10."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def _epoch_components(epoch: datetime) -> tuple[int, float]:
    """Extract 2-digit year and fractional day-of-year from epoch.

    Returns:
        (epoch_year_2digit, epoch_day_fractional)
    """
    epoch_year = epoch.year % 100
    day_of_year = epoch.timetuple().tm_yday
    fractional_day = (
        epoch.hour / 24.0
        + epoch.minute / 1440.0
        + epoch.second / 86400.0
    )
    return epoch_year, day_of_year + fractional_day


def _mean_motion(position_eci: tuple[float, float, float]) -> float:
    """Compute mean motion in revolutions per day from ECI position.

    Uses |position_eci| as orbital radius for a circular orbit.
    n = 86400 / period, where period = 2*pi*sqrt(r^3/mu).
    """
    r = math.sqrt(
        position_eci[0] ** 2
        + position_eci[1] ** 2
        + position_eci[2] ** 2
    )
    period = 2.0 * math.pi * math.sqrt(r**3 / OrbitalConstants.MU_EARTH)
    return 86400.0 / period


def _inclination_from_state(
    position_eci: tuple[float, float, float],
    velocity_eci: tuple[float, float, float],
) -> float:
    """Compute orbital inclination in degrees from ECI state vectors.

    Uses h = r x v, then i = arccos(h_z / |h|).
    """
    rx, ry, rz = position_eci
    vx, vy, vz = velocity_eci
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    h_mag = math.sqrt(hx**2 + hy**2 + hz**2)
    if h_mag == 0.0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, hz / h_mag))))


def _format_line1(
    catalog_num: int,
    epoch_year: int,
    epoch_day: float,
) -> str:
    """Format TLE line 1 (exactly 69 characters including checksum).

    Fixed-width columns per TLE specification.
    """
    launch_num = catalog_num - 99000

    line = (
        f"1 {catalog_num:05d}U "
        f"{epoch_year:02d}{launch_num:03d}A   "
        f"{epoch_year:02d}{epoch_day:012.8f} "
        f" .00000000 "
        f" 00000-0 "
        f" 00000-0 "
        f"0 "
        f" 999"
    )
    checksum = _tle_checksum(line)
    return line + str(checksum)


def _format_line2(
    catalog_num: int,
    inclination_deg: float,
    raan_deg: float,
    eccentricity: float,
    arg_perigee_deg: float,
    mean_anomaly_deg: float,
    mean_motion: float,
    rev_number: int = 0,
) -> str:
    """Format TLE line 2 (exactly 69 characters including checksum).

    Fixed-width columns per TLE specification.
    """
    ecc_str = f"{eccentricity:.7f}"[2:]  # "0.0000000" -> "0000000"

    line = (
        f"2 {catalog_num:05d} "
        f"{inclination_deg:8.4f} "
        f"{raan_deg:8.4f} "
        f"{ecc_str} "
        f"{arg_perigee_deg:8.4f} "
        f"{mean_anomaly_deg:8.4f} "
        f"{mean_motion:11.8f}"
        f"{rev_number:5d}"
    )
    checksum = _tle_checksum(line)
    return line + str(checksum)


class StellariumExporter(SatelliteExporter):
    """Exports satellite data as Stellarium-compatible TLE files.

    Generates Two-Line Element sets suitable for import into Stellarium's
    satellite plugin, STK, GMAT, and any TLE-consuming application.

    Catalog numbers start at catalog_start (default 99001) to avoid
    conflicts with real NORAD catalog entries.
    """

    def __init__(self, catalog_start: int = 99001) -> None:
        self._catalog_start = catalog_start

    def export(
        self,
        satellites: list[Satellite],
        path: str,
        epoch: datetime | None = None,
    ) -> int:
        """Export satellites as TLE to a file.

        Args:
            satellites: List of Satellite domain objects.
            path: Output file path.
            epoch: Fallback epoch for TLE epoch field when
                satellite.epoch is None.

        Returns:
            Number of satellites exported.
        """
        lines: list[str] = []

        for idx, sat in enumerate(satellites):
            sat_epoch = sat.epoch or epoch or _J2000
            catalog_num = self._catalog_start + idx

            epoch_year, epoch_day = _epoch_components(sat_epoch)
            n = _mean_motion(sat.position_eci)
            inc = _inclination_from_state(sat.position_eci, sat.velocity_eci)

            line0 = sat.name
            line1 = _format_line1(catalog_num, epoch_year, epoch_day)
            line2 = _format_line2(
                catalog_num=catalog_num,
                inclination_deg=inc,
                raan_deg=sat.raan_deg % 360.0,
                eccentricity=0.0,
                arg_perigee_deg=0.0,
                mean_anomaly_deg=sat.true_anomaly_deg % 360.0,
                mean_motion=n,
            )

            lines.append(line0)
            lines.append(line1)
            lines.append(line2)

        with open(path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines))
            # Empty satellites list produces empty file

        return len(satellites)
