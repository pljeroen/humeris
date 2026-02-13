# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Astronomical time systems: AstroTime value object with UTC/TAI/TT/TDB/GPS conversions.

Implements the IAU time scale chain: UTC → TAI → TT → TDB, plus GPS.
All conversions follow IERS Conventions 2010 and IAU resolutions.
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_TT_TAI_OFFSET: float = 32.184
"""TT = TAI + 32.184 s (exact, IAU 1991)."""

_GPS_TAI_OFFSET: float = 19.0
"""TAI = GPS + 19 s (exact, by definition)."""

_J2000_JD: float = 2451545.0
"""Julian Date of J2000.0 epoch."""

_SECONDS_PER_DAY: float = 86400.0

_SECONDS_PER_JULIAN_CENTURY: float = 36525.0 * _SECONDS_PER_DAY

# GPS epoch: 1980-01-06 00:00:00 UTC
_GPS_EPOCH_UTC = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

# J2000.0 TDB epoch in UTC (approximate): 2000-01-01 11:58:55.816
# More precisely: J2000.0 = 2000-01-01T11:58:55.816 TDB
# At J2000, TDB-TT ~ 0, TT = TAI + 32.184, TAI = UTC + 32
# So J2000 UTC ~ 11:58:55.816 - 32.184 - 32 = 11:57:51.632
# But we compute precisely via the chain.

# --------------------------------------------------------------------------- #
# Leap second table
# --------------------------------------------------------------------------- #

_LEAP_SECOND_TABLE: Optional[list[tuple[datetime, float]]] = None


def _load_leap_seconds() -> list[tuple[datetime, float]]:
    """Load and cache leap second table from bundled JSON."""
    global _LEAP_SECOND_TABLE
    if _LEAP_SECOND_TABLE is not None:
        return _LEAP_SECOND_TABLE

    data_path = Path(__file__).parent.parent / "data" / "tai_utc.json"
    with open(data_path) as f:
        data = json.load(f)

    table: list[tuple[datetime, float]] = []
    for entry in data["entries"]:
        parts = entry["date"].split("-")
        dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                      tzinfo=timezone.utc)
        table.append((dt, float(entry["tai_utc"])))

    _LEAP_SECOND_TABLE = table
    return _LEAP_SECOND_TABLE


def utc_to_tai_seconds(dt: datetime) -> float:
    """Return TAI-UTC offset (delta_AT) for a given UTC datetime.

    Uses binary search on the leap second table.
    Raises ValueError for dates before 1972-01-01.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    table = _load_leap_seconds()

    if dt < table[0][0]:
        raise ValueError(
            f"UTC date {dt.isoformat()} is before 1972-01-01; "
            "leap second table undefined"
        )

    # Binary search for the last entry <= dt
    lo, hi = 0, len(table) - 1
    result = table[0][1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if table[mid][0] <= dt:
            result = table[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1

    return result


# --------------------------------------------------------------------------- #
# Julian Date utilities
# --------------------------------------------------------------------------- #

def datetime_to_jd(dt: datetime) -> float:
    """Convert a UTC datetime to Julian Date.

    Uses the standard algorithm (Meeus, Astronomical Algorithms, Ch. 7).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    y = dt.year
    m = dt.month
    d = (dt.day
         + dt.hour / 24.0
         + dt.minute / 1440.0
         + dt.second / 86400.0
         + dt.microsecond / 86400_000_000.0)

    if m <= 2:
        y -= 1
        m += 12

    A = y // 100
    B = 2 - A + A // 4

    return (math.floor(365.25 * (y + 4716))
            + math.floor(30.6001 * (m + 1))
            + d + B - 1524.5)


def _jd_to_datetime(jd: float) -> datetime:
    """Convert Julian Date back to a UTC datetime.

    Inverse of datetime_to_jd. Meeus Ch. 7 inverse.
    """
    jd_plus = jd + 0.5
    Z = int(jd_plus)
    F = jd_plus - Z

    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - alpha // 4

    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)

    day_frac = B - D - int(30.6001 * E) + F

    if E < 14:
        month = E - 1
    else:
        month = E - 13

    if month > 2:
        year = C - 4716
    else:
        year = C - 4715

    day = int(day_frac)
    remainder = (day_frac - day) * _SECONDS_PER_DAY
    hour = int(remainder / 3600.0)
    remainder -= hour * 3600.0
    minute = int(remainder / 60.0)
    second_frac = remainder - minute * 60.0
    second = int(second_frac)
    microsecond = int((second_frac - second) * 1_000_000.0 + 0.5)

    if microsecond >= 1_000_000:
        microsecond -= 1_000_000
        second += 1
    if second >= 60:
        second -= 60
        minute += 1
    if minute >= 60:
        minute -= 60
        hour += 1

    return datetime(year, month, day, hour, minute, second, microsecond,
                    tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# TDB-TT conversion (Fairhead & Bretagnon 1990)
# --------------------------------------------------------------------------- #

def _tdb_minus_tt(t_tt_centuries: float) -> float:
    """Compute TDB - TT in seconds using Fairhead & Bretagnon 1990.

    Accurate to ~30 μs. The dominant term is the Earth's orbital eccentricity.

    Parameters
    ----------
    t_tt_centuries : float
        Julian centuries of TT from J2000.0.
    """
    # Mean anomaly of Earth (radians)
    M_E = float(np.radians(357.5277233 + 35999.160503 * t_tt_centuries))
    # Mean longitude of Jupiter (simplified)
    L_J = float(np.radians(246.11 + 3034.906 * t_tt_centuries))

    return float(0.001657 * np.sin(M_E)
            + 0.000022 * np.sin(L_J - M_E))


# --------------------------------------------------------------------------- #
# AstroTime value object
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, order=True)
class AstroTime:
    """Astronomical time as seconds since J2000.0 TDB.

    J2000.0 TDB = 2000-01-01T11:58:55.816 TDB = JD 2451545.0 TDB.

    This is the fundamental time representation for high-fidelity
    astrodynamics. Conversions to/from UTC, TT, TAI, GPS are provided.
    """

    tdb_j2000: float
    """Seconds since J2000.0 TDB epoch."""

    # -- Construction ------------------------------------------------------- #

    @staticmethod
    def from_utc(dt: datetime) -> "AstroTime":
        """Create AstroTime from a UTC datetime.

        Conversion chain: UTC → TAI → TT → TDB.
        Naive datetimes are treated as UTC.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Step 1: UTC → JD(UTC)
        jd_utc = datetime_to_jd(dt)

        # Step 2: UTC → TAI (add leap seconds)
        delta_at = utc_to_tai_seconds(dt)

        # Step 3: TAI → TT
        # TT = TAI + 32.184s = UTC + delta_AT + 32.184
        tt_offset_s = delta_at + _TT_TAI_OFFSET
        jd_tt = jd_utc + tt_offset_s / _SECONDS_PER_DAY

        # Step 4: TT → TDB (iterative, one pass sufficient for <30μs)
        t_tt = (jd_tt - _J2000_JD) / 36525.0  # Julian centuries TT
        tdb_minus_tt = _tdb_minus_tt(t_tt)

        # TDB seconds from J2000 TDB
        tt_seconds_from_j2000 = (jd_tt - _J2000_JD) * _SECONDS_PER_DAY
        tdb_seconds = tt_seconds_from_j2000 + tdb_minus_tt

        return AstroTime(tdb_j2000=tdb_seconds)

    @staticmethod
    def from_tt_seconds(tt_seconds_from_j2000: float) -> "AstroTime":
        """Create AstroTime from TT seconds since J2000.0 TT."""
        t_tt = tt_seconds_from_j2000 / _SECONDS_PER_JULIAN_CENTURY
        tdb_minus_tt = _tdb_minus_tt(t_tt)
        return AstroTime(tdb_j2000=tt_seconds_from_j2000 + tdb_minus_tt)

    @staticmethod
    def from_julian_date_tdb(jd_tdb: float) -> "AstroTime":
        """Create AstroTime from a TDB Julian Date."""
        return AstroTime(tdb_j2000=(jd_tdb - _J2000_JD) * _SECONDS_PER_DAY)

    @staticmethod
    def from_gps(gps_seconds: float) -> "AstroTime":
        """Create AstroTime from GPS seconds since GPS epoch (1980-01-06 00:00:00 UTC).

        TAI = GPS + 19s (exact). Then TAI → TT → TDB.
        """
        # GPS epoch in UTC
        gps_epoch_jd = datetime_to_jd(_GPS_EPOCH_UTC)

        # GPS seconds → TAI seconds from GPS epoch
        tai_seconds_from_gps_epoch = gps_seconds + _GPS_TAI_OFFSET

        # TAI → TT
        tt_seconds_from_gps_epoch = tai_seconds_from_gps_epoch + _TT_TAI_OFFSET

        # Convert to TT JD
        # At GPS epoch (1980-01-06 00:00:00 UTC), TAI-UTC = 19s
        # So TT at GPS epoch = UTC + 19 + 32.184 = UTC + 51.184s
        delta_at_gps_epoch = 19.0  # TAI-UTC at GPS epoch
        tt_offset_at_gps_epoch = delta_at_gps_epoch + _TT_TAI_OFFSET

        jd_tt = (gps_epoch_jd
                 + tt_offset_at_gps_epoch / _SECONDS_PER_DAY
                 + gps_seconds / _SECONDS_PER_DAY)

        # TT → TDB
        tt_seconds_from_j2000 = (jd_tt - _J2000_JD) * _SECONDS_PER_DAY
        t_tt = tt_seconds_from_j2000 / _SECONDS_PER_JULIAN_CENTURY
        tdb_minus_tt = _tdb_minus_tt(t_tt)

        return AstroTime(tdb_j2000=tt_seconds_from_j2000 + tdb_minus_tt)

    # -- Conversions -------------------------------------------------------- #

    def to_tt_seconds(self) -> float:
        """Return TT seconds since J2000.0 TT.

        Inverts TDB → TT using single-iteration Newton.
        """
        # First approximation: TT ≈ TDB
        tt_approx = self.tdb_j2000
        t_tt = tt_approx / _SECONDS_PER_JULIAN_CENTURY
        tdb_minus_tt = _tdb_minus_tt(t_tt)
        return self.tdb_j2000 - tdb_minus_tt

    def to_julian_date_tdb(self) -> float:
        """Return TDB Julian Date."""
        return _J2000_JD + self.tdb_j2000 / _SECONDS_PER_DAY

    def to_julian_centuries_tt(self) -> float:
        """Return Julian centuries of TT from J2000.0.

        This is the 'T' parameter used in precession/nutation formulae.
        """
        return self.to_tt_seconds() / _SECONDS_PER_JULIAN_CENTURY

    def to_mjd_tt(self) -> float:
        """Return Modified Julian Date in TT scale.

        MJD = JD - 2400000.5.
        """
        jd_tt = _J2000_JD + self.to_tt_seconds() / _SECONDS_PER_DAY
        return jd_tt - 2400000.5

    def to_utc_datetime(self) -> datetime:
        """Convert to UTC datetime.

        Inverts the UTC → TDB chain: TDB → TT → TAI → UTC.
        """
        # TDB → TT
        tt_seconds = self.to_tt_seconds()

        # TT JD
        jd_tt = _J2000_JD + tt_seconds / _SECONDS_PER_DAY

        # TT → UTC: we need to subtract (delta_AT + 32.184) / 86400
        # But delta_AT depends on UTC, which we don't know yet.
        # Iterative: start with approximate UTC, refine.
        # Usually converges in 1 iteration (leap seconds are integer).

        # Initial guess: subtract current max offset
        delta_at_guess = 37.0  # Start with most recent
        jd_utc = jd_tt - (delta_at_guess + _TT_TAI_OFFSET) / _SECONDS_PER_DAY
        dt_utc = _jd_to_datetime(jd_utc)

        # Refine with actual delta_AT at guessed UTC
        try:
            delta_at = utc_to_tai_seconds(dt_utc)
        except ValueError:
            delta_at = delta_at_guess

        jd_utc = jd_tt - (delta_at + _TT_TAI_OFFSET) / _SECONDS_PER_DAY
        dt_utc = _jd_to_datetime(jd_utc)

        # One more refinement for edge cases near leap second boundaries
        try:
            delta_at2 = utc_to_tai_seconds(dt_utc)
        except ValueError:
            delta_at2 = delta_at

        if delta_at2 != delta_at:
            jd_utc = jd_tt - (delta_at2 + _TT_TAI_OFFSET) / _SECONDS_PER_DAY
            dt_utc = _jd_to_datetime(jd_utc)

        return dt_utc

    def to_gps_seconds(self) -> float:
        """Convert to GPS seconds since GPS epoch (1980-01-06 00:00:00 UTC).

        GPS = TAI - 19s. TAI = TT - 32.184s.
        """
        tt_seconds = self.to_tt_seconds()
        jd_tt = _J2000_JD + tt_seconds / _SECONDS_PER_DAY

        # GPS epoch TT JD
        gps_epoch_jd_utc = datetime_to_jd(_GPS_EPOCH_UTC)
        delta_at_gps_epoch = 19.0
        gps_epoch_jd_tt = (gps_epoch_jd_utc
                           + (delta_at_gps_epoch + _TT_TAI_OFFSET)
                           / _SECONDS_PER_DAY)

        # TT elapsed since GPS epoch
        tt_elapsed = (jd_tt - gps_epoch_jd_tt) * _SECONDS_PER_DAY

        # TT elapsed = GPS elapsed + (TAI-GPS offset is constant = 19)
        # + (TT-TAI offset is constant = 32.184)
        # So GPS elapsed = TT elapsed (the constant offsets cancel in the difference)
        return tt_elapsed

    def to_datetime_utc(self) -> datetime:
        """Alias for to_utc_datetime() for API convenience."""
        return self.to_utc_datetime()

    # -- Arithmetic --------------------------------------------------------- #

    def __add__(self, seconds: float) -> "AstroTime":
        """Add seconds to this time, returning a new AstroTime."""
        if not isinstance(seconds, (int, float)):
            return NotImplemented
        return AstroTime(tdb_j2000=self.tdb_j2000 + seconds)

    def __radd__(self, seconds: float) -> "AstroTime":
        return self.__add__(seconds)

    def __sub__(self, other):
        """Subtract another AstroTime (returns seconds) or seconds (returns AstroTime)."""
        if isinstance(other, AstroTime):
            return self.tdb_j2000 - other.tdb_j2000
        if isinstance(other, (int, float)):
            return AstroTime(tdb_j2000=self.tdb_j2000 - other)
        return NotImplemented

    def __repr__(self) -> str:
        return f"AstroTime(tdb_j2000={self.tdb_j2000:.6f})"
