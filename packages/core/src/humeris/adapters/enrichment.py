# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Satellite enrichment adapter.

Computes orbital analysis data from a Satellite + epoch using domain
functions. Shared by all exporters that add enrichment fields.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from humeris.domain.constellation import Satellite
from humeris.domain.coordinate_frames import (
    ecef_to_geodetic,
    eci_to_ecef,
    gmst_rad,
)
from humeris.domain.orbital_mechanics import OrbitalConstants

try:
    from humeris.domain.eclipse import compute_beta_angle as _compute_beta_angle
    _HAS_ECLIPSE = True
except ImportError:
    _HAS_ECLIPSE = False

try:
    from humeris.domain.atmosphere import atmospheric_density as _atmospheric_density
    _HAS_ATMOSPHERE = True
except ImportError:
    _HAS_ATMOSPHERE = False

try:
    from humeris.domain.nrlmsise00 import atmospheric_density_nrlmsise00 as _nrlmsise00_density
    _HAS_NRLMSISE00 = True
except ImportError:
    _HAS_NRLMSISE00 = False

try:
    from humeris.domain.radiation import compute_l_shell as _compute_l_shell
    _HAS_RADIATION = True
except ImportError:
    _HAS_RADIATION = False


_J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_R_EARTH_M = OrbitalConstants.R_EARTH
_MU_EARTH = OrbitalConstants.MU_EARTH


@dataclass(frozen=True)
class SatelliteEnrichment:
    """Orbital analysis data computed from a Satellite + epoch."""

    altitude_km: float
    inclination_deg: float
    orbital_period_min: float
    beta_angle_deg: float
    atmospheric_density_kg_m3: float
    l_shell: float


def compute_satellite_enrichment(
    sat: Satellite,
    epoch: datetime | None,
) -> SatelliteEnrichment:
    """Compute orbital analysis data for a satellite.

    All computations are lightweight (microseconds per satellite).

    Args:
        sat: Satellite with ECI state vectors.
        epoch: Reference epoch for solar geometry. Falls back to J2000.

    Returns:
        Frozen dataclass with 6 analysis fields.
    """
    effective_epoch = epoch or _J2000

    px, py, pz = sat.position_eci
    vx, vy, vz = sat.velocity_eci

    # Orbital radius and altitude
    r_m = math.sqrt(px * px + py * py + pz * pz)
    alt_km = r_m / 1000.0 - _R_EARTH_M / 1000.0

    # Inclination from angular momentum vector h = r × v
    hx = py * vz - pz * vy
    hy = pz * vx - px * vz
    hz = px * vy - py * vx
    h_mag = math.sqrt(hx * hx + hy * hy + hz * hz)
    if h_mag > 0.0:
        inc_deg = math.degrees(math.acos(max(-1.0, min(1.0, hz / h_mag))))
    else:
        inc_deg = 0.0

    # Orbital period (Kepler's third law)
    period_min = 2.0 * math.pi * math.sqrt(r_m ** 3 / _MU_EARTH) / 60.0

    # Beta angle (Sun-orbit plane angle) — requires humeris-pro
    if _HAS_ECLIPSE:
        beta_deg = _compute_beta_angle(
            math.radians(sat.raan_deg),
            math.radians(inc_deg),
            effective_epoch,
        )
    else:
        beta_deg = 0.0

    # Atmospheric density at altitude — requires humeris-pro
    # Prefer NRLMSISE-00 (epoch-dependent) over static exponential model
    density = 0.0
    if _HAS_NRLMSISE00 and epoch is not None:
        try:
            density = _nrlmsise00_density(alt_km, effective_epoch)
        except (ValueError, Exception):
            density = 0.0
    if density == 0.0 and _HAS_ATMOSPHERE:
        try:
            density = _atmospheric_density(alt_km)
        except ValueError:
            density = 0.0

    # L-shell from geodetic position — requires humeris-pro
    if _HAS_RADIATION:
        gmst_angle = gmst_rad(effective_epoch)
        pos_ecef, _ = eci_to_ecef(sat.position_eci, sat.velocity_eci, gmst_angle)
        lat_deg, lon_deg, _ = ecef_to_geodetic(pos_ecef)
        l_val = _compute_l_shell(lat_deg, lon_deg, alt_km)
    else:
        l_val = 0.0

    return SatelliteEnrichment(
        altitude_km=alt_km,
        inclination_deg=inc_deg,
        orbital_period_min=period_min,
        beta_angle_deg=beta_deg,
        atmospheric_density_kg_m3=density,
        l_shell=l_val,
    )
