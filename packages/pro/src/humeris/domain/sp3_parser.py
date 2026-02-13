# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""SP3 precise ephemeris parser.

Parses the SP3 (Standard Product #3) format used by the International
GNSS Service (IGS) for precise satellite ephemerides. These contain
post-processed, centimeter-level accurate positions and (optionally)
velocities for GNSS satellites.

SP3 format reference:
    https://files.igs.org/pub/data/format/sp3d.pdf

This parser handles SP3-c and SP3-d formats.
All inputs/outputs use SI units (meters, seconds).
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class SP3EphemerisPoint:
    """A single satellite position at a specific epoch."""
    satellite_id: str
    time: datetime
    x_m: float
    y_m: float
    z_m: float
    clock_us: float  # clock correction in microseconds (0 if unavailable)


@dataclass(frozen=True)
class SP3Ephemeris:
    """Parsed SP3 ephemeris file."""
    agency: str
    num_epochs: int
    num_satellites: int
    satellite_ids: tuple[str, ...]
    points: tuple[SP3EphemerisPoint, ...]
    interval_s: float
    start_time: datetime
    end_time: datetime
    coordinate_system: str
    orbit_type: str


def parse_sp3(content: str) -> SP3Ephemeris:
    """Parse an SP3 format string into an SP3Ephemeris object.

    Args:
        content: Complete SP3 file content as a string.

    Returns:
        SP3Ephemeris with all parsed position records.

    Raises:
        ValueError: If the content is not valid SP3 format.
    """
    lines = content.strip().split('\n')
    if not lines or not lines[0].startswith('#'):
        raise ValueError("Not a valid SP3 file: missing header line")

    header_line = lines[0]
    if len(header_line) < 60:
        raise ValueError("SP3 header line too short")

    # Parse header line 1: #cP2026  1 15  0  0  0.00000000      97 ORBIT IGS20 FIT  IGS
    version = header_line[1]  # 'c' or 'd'
    pos_vel = header_line[2]  # 'P' (position only) or 'V' (position+velocity)

    # Parse start epoch from header
    parts = header_line[3:].split()
    if len(parts) < 7:
        raise ValueError("SP3 header: cannot parse epoch")
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    hour = int(parts[3])
    minute = int(parts[4])
    second = float(parts[5])
    sec_int = int(second)
    microsecond = int((second - sec_int) * 1e6)
    start_time = datetime(year, month, day, hour, minute, sec_int,
                          microsecond, tzinfo=timezone.utc)

    num_epochs_str = parts[6]
    num_epochs = int(num_epochs_str)

    # Parse line 2: ## week seconds_of_week epoch_interval num_mjd frac_mjd
    line2 = lines[1]
    if not line2.startswith('##'):
        raise ValueError("SP3 missing second header line (##)")
    parts2 = line2[2:].split()
    interval_s = float(parts2[2]) if len(parts2) > 2 else 900.0

    # Parse satellite IDs from + lines (lines starting with '+')
    satellite_ids = []
    coord_system = "ITRF"
    orbit_type = "FIT"
    agency = "IGS"

    # Try to get orbit type and agency from header
    if "ORBIT" in header_line:
        idx = header_line.index("ORBIT")
        remaining = header_line[idx + 5:].strip().split()
        if remaining:
            coord_system = remaining[0]
        if len(remaining) > 1:
            orbit_type = remaining[1]
        if len(remaining) > 2:
            agency = remaining[2]

    for line in lines[2:]:
        if line.startswith('+') and not line.startswith('++'):
            # Satellite ID line: +   32   G01G02G03...
            tokens = line[1:].split()
            if not tokens:
                continue
            # First + line has count, subsequent just have IDs
            start_idx = 0
            if tokens[0].isdigit():
                start_idx = 1
            for token in tokens[start_idx:]:
                # Parse 3-character satellite IDs: G01, G02, R01, E01, etc.
                for i in range(0, len(token), 3):
                    sid = token[i:i + 3].strip()
                    if sid and sid != '0' and sid != '00' and sid != '000':
                        satellite_ids.append(sid)
        elif line.startswith('*'):
            break  # reached epoch records
        elif line.startswith('%'):
            continue  # format descriptor lines

    num_satellites = len(satellite_ids) if satellite_ids else 0

    # Parse epoch and position records
    points = []
    current_epoch = None

    for line in lines:
        if line.startswith('*'):
            # Epoch header: *  2026  1 15  0  0  0.00000000
            eparts = line[1:].split()
            if len(eparts) >= 6:
                ey = int(eparts[0])
                em = int(eparts[1])
                ed = int(eparts[2])
                eh = int(eparts[3])
                emin = int(eparts[4])
                esec = float(eparts[5])
                esec_int = int(esec)
                eus = int((esec - esec_int) * 1e6)
                current_epoch = datetime(ey, em, ed, eh, emin, esec_int,
                                         eus, tzinfo=timezone.utc)
        elif line.startswith('P') and current_epoch is not None:
            # Position record: PG01  x_km  y_km  z_km  clock_us
            sat_id = line[1:4].strip()
            pparts = line[4:].split()
            if len(pparts) >= 3:
                x_km = float(pparts[0])
                y_km = float(pparts[1])
                z_km = float(pparts[2])
                clock_us = float(pparts[3]) if len(pparts) > 3 else 0.0

                # SP3 positions are in km, convert to meters
                # SP3 uses 999999.999999 for missing/bad data
                if abs(x_km) > 999990 or abs(y_km) > 999990 or abs(z_km) > 999990:
                    continue

                points.append(SP3EphemerisPoint(
                    satellite_id=sat_id,
                    time=current_epoch,
                    x_m=x_km * 1000.0,
                    y_m=y_km * 1000.0,
                    z_m=z_km * 1000.0,
                    clock_us=clock_us,
                ))
        elif line.startswith('EOF'):
            break

    end_time = points[-1].time if points else start_time

    return SP3Ephemeris(
        agency=agency,
        num_epochs=num_epochs,
        num_satellites=num_satellites,
        satellite_ids=tuple(satellite_ids),
        points=tuple(points),
        interval_s=interval_s,
        start_time=start_time,
        end_time=end_time,
        coordinate_system=coord_system,
        orbit_type=orbit_type,
    )


def filter_satellite(ephemeris: SP3Ephemeris, satellite_id: str) -> list[SP3EphemerisPoint]:
    """Extract all points for a specific satellite.

    Args:
        ephemeris: Parsed SP3 ephemeris.
        satellite_id: 3-character satellite ID (e.g., "G01").

    Returns:
        List of SP3EphemerisPoint for the specified satellite, time-ordered.
    """
    return [p for p in ephemeris.points if p.satellite_id == satellite_id]
