# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Analytical solar ephemeris.

Low-precision Sun position using Meeus "Astronomical Algorithms" Ch. 25 /
Vallado simplified algorithm. Accuracy ~1 arcminute, sufficient for
eclipse/illumination analysis.

No external dependencies — only stdlib math/dataclasses/datetime.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

AU_METERS: float = 1.495978707e11  # Astronomical unit in meters

# J2000.0 reference epoch
_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class SunPosition:
    """Sun position at a given epoch."""
    position_eci_m: tuple[float, float, float]  # ECI position in meters
    right_ascension_rad: float
    declination_rad: float
    distance_m: float  # Earth-Sun distance in meters


def julian_centuries_j2000(epoch: datetime) -> float:
    """Julian centuries since J2000.0 (2000-01-01 12:00:00 UTC)."""
    dt_seconds = (epoch - _J2000).total_seconds()
    return dt_seconds / (36525.0 * 86400.0)


def sun_position_eci(epoch: datetime) -> SunPosition:
    """Low-precision analytical solar ephemeris.

    Meeus "Astronomical Algorithms" Ch. 25 / Vallado simplified algorithm.
    Accuracy: ~1 arcminute, sufficient for eclipse/illumination analysis.

    Args:
        epoch: UTC datetime for Sun position computation.

    Returns:
        SunPosition with ECI coordinates, RA, Dec, and distance.
    """
    T = julian_centuries_j2000(epoch)

    # Mean anomaly (degrees)
    M_deg = (357.5291 + 35999.0503 * T) % 360.0
    M_rad = float(np.radians(M_deg))

    # Ecliptic longitude (degrees)
    L_deg = (280.4665 + 36000.7698 * T + 1.9146 * float(np.sin(M_rad))
             + 0.0200 * float(np.sin(2.0 * M_rad))) % 360.0
    L_rad = float(np.radians(L_deg))

    # Obliquity of the ecliptic (degrees)
    eps_deg = 23.4393 - 0.01300 * T
    eps_rad = float(np.radians(eps_deg))

    # Right ascension and declination
    ra_rad = float(np.arctan2(np.cos(eps_rad) * np.sin(L_rad), np.cos(L_rad)))
    dec_rad = float(np.arcsin(np.sin(eps_rad) * np.sin(L_rad)))

    # Distance in AU
    r_au = 1.00014 - 0.01671 * float(np.cos(M_rad)) - 0.00014 * float(np.cos(2.0 * M_rad))
    distance_m = r_au * AU_METERS

    # ECI position
    cos_ra = float(np.cos(ra_rad))
    sin_ra = float(np.sin(ra_rad))
    cos_dec = float(np.cos(dec_rad))
    sin_dec = float(np.sin(dec_rad))

    x = distance_m * cos_ra * cos_dec
    y = distance_m * sin_ra * cos_dec
    z = distance_m * sin_dec

    return SunPosition(
        position_eci_m=(x, y, z),
        right_ascension_rad=ra_rad,
        declination_rad=dec_rad,
        distance_m=distance_m,
    )


def solar_declination_rad(epoch: datetime) -> float:
    """Solar declination at given epoch. Convenience wrapper."""
    return sun_position_eci(epoch).declination_rad
