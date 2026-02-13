# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see LICENSE-COMMERCIAL.md.
"""Earth Orientation Parameters (EOP): UT1-UTC, polar motion, and transformations.

Provides loading and interpolation of IERS finals2000A data for
high-fidelity GCRS → ITRS transformations.

References:
    IERS Conventions 2010, Chapter 5.
    IERS Technical Note 36.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from constellation_generator.domain.time_systems import datetime_to_jd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_ARCSEC_TO_RAD: float = math.pi / (180.0 * 3600.0)
_MJD_OFFSET: float = 2400000.5  # JD = MJD + 2400000.5

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EOPEntry:
    """A single EOP data point."""

    mjd: float
    ut1_utc: float  # seconds
    xp_arcsec: float  # polar motion x (arcseconds)
    yp_arcsec: float  # polar motion y (arcseconds)


@dataclass(frozen=True)
class EOPTable:
    """Ordered table of EOP entries for interpolation."""

    entries: tuple[EOPEntry, ...]


# --------------------------------------------------------------------------- #
# MJD conversion
# --------------------------------------------------------------------------- #


def datetime_to_mjd(dt) -> float:
    """Convert a datetime to Modified Julian Date.

    MJD = JD - 2400000.5.
    """
    from datetime import timezone as _tz
    if dt.tzinfo is None:
        from datetime import timezone as _tz2
        dt = dt.replace(tzinfo=_tz.utc)
    jd = datetime_to_jd(dt)
    return jd - _MJD_OFFSET


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

_CACHED_TABLE: Optional[EOPTable] = None


def load_eop(path: Optional[str] = None) -> EOPTable:
    """Load EOP data from bundled JSON or custom path.

    Parameters
    ----------
    path : str, optional
        Path to a custom EOP JSON file. If None, uses bundled data.

    Returns
    -------
    EOPTable with entries sorted by MJD.
    """
    global _CACHED_TABLE

    if path is None and _CACHED_TABLE is not None:
        return _CACHED_TABLE

    if path is None:
        data_path = Path(__file__).parent.parent / "data" / "eop_finals2000a.json"
    else:
        data_path = Path(path)

    with open(data_path) as f:
        data = json.load(f)

    entries = tuple(
        EOPEntry(
            mjd=float(e["mjd"]),
            ut1_utc=float(e["ut1_utc"]),
            xp_arcsec=float(e["xp"]),
            yp_arcsec=float(e["yp"]),
        )
        for e in data["entries"]
    )

    table = EOPTable(entries=entries)

    if path is None:
        _CACHED_TABLE = table

    return table


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #


def interpolate_eop(table: EOPTable, mjd: float) -> EOPEntry:
    """Linearly interpolate EOP values at a given MJD.

    Extrapolates using the nearest entry outside the table range.

    Parameters
    ----------
    table : EOPTable
        EOP data table.
    mjd : float
        Modified Julian Date to interpolate at.

    Returns
    -------
    EOPEntry with interpolated values.
    """
    entries = table.entries

    if not entries:
        return EOPEntry(mjd=mjd, ut1_utc=0.0, xp_arcsec=0.0, yp_arcsec=0.0)

    # Before range: clamp to first entry
    if mjd <= entries[0].mjd:
        e = entries[0]
        return EOPEntry(mjd=mjd, ut1_utc=e.ut1_utc,
                        xp_arcsec=e.xp_arcsec, yp_arcsec=e.yp_arcsec)

    # After range: clamp to last entry
    if mjd >= entries[-1].mjd:
        e = entries[-1]
        return EOPEntry(mjd=mjd, ut1_utc=e.ut1_utc,
                        xp_arcsec=e.xp_arcsec, yp_arcsec=e.yp_arcsec)

    # Binary search for bracketing entries
    lo, hi = 0, len(entries) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if entries[mid].mjd <= mjd:
            lo = mid
        else:
            hi = mid

    e0 = entries[lo]
    e1 = entries[hi]

    # Linear interpolation
    frac = (mjd - e0.mjd) / (e1.mjd - e0.mjd)

    return EOPEntry(
        mjd=mjd,
        ut1_utc=e0.ut1_utc + frac * (e1.ut1_utc - e0.ut1_utc),
        xp_arcsec=e0.xp_arcsec + frac * (e1.xp_arcsec - e0.xp_arcsec),
        yp_arcsec=e0.yp_arcsec + frac * (e1.yp_arcsec - e0.yp_arcsec),
    )


# --------------------------------------------------------------------------- #
# Polar motion matrix
# --------------------------------------------------------------------------- #


def polar_motion_matrix(
    xp_rad: float,
    yp_rad: float,
    s_prime_rad: float = 0.0,
) -> tuple[tuple[float, ...], ...]:
    """Compute the polar motion rotation matrix W.

    W = R3(-s') · R2(x_p) · R1(y_p)

    IERS Conventions 2010, Eq. 5.4.

    Parameters
    ----------
    xp_rad : float
        Polar motion x_p in radians.
    yp_rad : float
        Polar motion y_p in radians.
    s_prime_rad : float
        TIO locator s' in radians. Default 0 (negligible for most applications).

    Returns
    -------
    W : 3x3 rotation matrix (tuple of tuples).
    """
    cx, sx = math.cos(xp_rad), math.sin(xp_rad)
    cy, sy = math.cos(yp_rad), math.sin(yp_rad)
    cs, ss = math.cos(-s_prime_rad), math.sin(-s_prime_rad)

    # R1(y_p)
    R1 = (
        (1.0, 0.0, 0.0),
        (0.0, cy, sy),
        (0.0, -sy, cy),
    )

    # R2(x_p)
    R2 = (
        (cx, 0.0, -sx),
        (0.0, 1.0, 0.0),
        (sx, 0.0, cx),
    )

    # R3(-s')
    R3 = (
        (cs, ss, 0.0),
        (-ss, cs, 0.0),
        (0.0, 0.0, 1.0),
    )

    # W = R3(-s') · R2(x_p) · R1(y_p)
    # Compute R2 · R1 first
    R2R1 = _mat_mul(R2, R1)
    return _mat_mul(R3, R2R1)


def _mat_mul(A: tuple, B: tuple) -> tuple[tuple[float, ...], ...]:
    """Multiply two 3x3 matrices."""
    result = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return tuple(tuple(row) for row in result)
