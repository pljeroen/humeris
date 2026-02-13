# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Pass analysis: Doppler shift, pass quality, contact statistics, data downlink, visual magnitude.

Derives pass-level and link-level metrics from existing access windows,
ground station geometry, and orbital state vectors.

"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from humeris.domain.observation import GroundStation, compute_observation
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
    geodetic_to_ecef,
)
from humeris.domain.access_windows import AccessWindow

_C_LIGHT = 299_792_458.0  # speed of light m/s
_SUN_MAG = -26.74  # apparent magnitude of the Sun


# --- Types ---

@dataclass(frozen=True)
class DopplerResult:
    """Doppler shift prediction at a single instant."""
    shift_hz: float
    range_rate_ms: float
    slant_range_km: float


@dataclass(frozen=True)
class PassClassification:
    """Pass quality classification based on maximum elevation."""
    quality: str  # "excellent", "good", "fair", "poor"
    max_elevation_deg: float
    duration_seconds: float


@dataclass(frozen=True)
class ContactSummary:
    """Aggregate contact statistics from a set of access windows."""
    total_contact_seconds: float
    num_passes: int
    max_gap_seconds: float
    mean_gap_seconds: float
    best_elevation_deg: float


@dataclass(frozen=True)
class DataDownlinkEstimate:
    """Estimated data downlink volume from access windows."""
    daily_contact_seconds: float
    daily_data_bytes: float
    human_readable: str


# --- Functions ---

def compute_doppler_shift(
    station: GroundStation,
    sat_pos_eci: list[float] | tuple[float, ...],
    sat_vel_eci: list[float] | tuple[float, ...],
    time: datetime,
    freq_hz: float,
) -> DopplerResult:
    """Compute Doppler frequency shift from relative velocity along line of sight.

    shift = -freq * range_rate / c

    where range_rate = d(range)/dt, computed from the ECI velocity projected
    onto the line-of-sight unit vector.

    Args:
        station: Ground station.
        sat_pos_eci: Satellite ECI position (m).
        sat_vel_eci: Satellite ECI velocity (m/s).
        time: Observation time (for GMST conversion).
        freq_hz: Transmit frequency in Hz.

    Returns:
        DopplerResult with shift (Hz), range rate (m/s), slant range (km).
    """
    # Convert satellite to ECEF
    gmst_angle = gmst_rad(time)
    pos_ecef, vel_ecef = eci_to_ecef(
        (sat_pos_eci[0], sat_pos_eci[1], sat_pos_eci[2]),
        (sat_vel_eci[0], sat_vel_eci[1], sat_vel_eci[2]),
        gmst_angle,
    )

    # Station ECEF
    station_ecef = geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)

    # Range vector (station → satellite)
    d_vec = np.array(pos_ecef) - np.array(station_ecef)

    slant_range_m = float(np.linalg.norm(d_vec))
    slant_range_km = slant_range_m / 1000.0

    if slant_range_m < 1e-10:
        return DopplerResult(shift_hz=0.0, range_rate_ms=0.0, slant_range_km=0.0)

    # Unit vector from station to satellite
    u_hat = d_vec / slant_range_m

    # Range rate: projection of satellite ECEF velocity onto line-of-sight.
    # Station velocity from Earth rotation (~465 m/s at equator) is
    # neglected. Error < 15% of total Doppler at equatorial stations.
    # The sign convention: positive range_rate means increasing range
    # (satellite moving away).
    range_rate = float(np.dot(vel_ecef, u_hat))

    # Doppler shift: approaching = positive shift, receding = negative
    shift_hz = -freq_hz * range_rate / _C_LIGHT

    return DopplerResult(
        shift_hz=shift_hz,
        range_rate_ms=range_rate,
        slant_range_km=slant_range_km,
    )


def classify_pass(window: AccessWindow) -> PassClassification:
    """Classify pass quality by maximum elevation.

    Categories:
        excellent: max_el >= 60°
        good: 40° <= max_el < 60°
        fair: 20° <= max_el < 40°
        poor: max_el < 20°

    Args:
        window: AccessWindow with max_elevation_deg.

    Returns:
        PassClassification with quality label.
    """
    el = window.max_elevation_deg
    if el >= 60:
        quality = "excellent"
    elif el >= 40:
        quality = "good"
    elif el >= 20:
        quality = "fair"
    else:
        quality = "poor"

    return PassClassification(
        quality=quality,
        max_elevation_deg=el,
        duration_seconds=window.duration_seconds,
    )


def compute_contact_summary(windows: list[AccessWindow]) -> ContactSummary:
    """Compute aggregate contact statistics from access windows.

    Args:
        windows: List of AccessWindow objects (should be chronologically sorted).

    Returns:
        ContactSummary with total contact, gaps, and best elevation.
    """
    if not windows:
        return ContactSummary(
            total_contact_seconds=0.0,
            num_passes=0,
            max_gap_seconds=0.0,
            mean_gap_seconds=0.0,
            best_elevation_deg=0.0,
        )

    total_contact = sum(w.duration_seconds for w in windows)
    best_el = max(w.max_elevation_deg for w in windows)

    # Compute gaps between consecutive windows
    gaps: list[float] = []
    for i in range(1, len(windows)):
        gap = (windows[i].rise_time - windows[i - 1].set_time).total_seconds()
        if gap > 0:
            gaps.append(gap)

    max_gap = max(gaps) if gaps else 0.0
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0

    return ContactSummary(
        total_contact_seconds=total_contact,
        num_passes=len(windows),
        max_gap_seconds=max_gap,
        mean_gap_seconds=mean_gap,
        best_elevation_deg=best_el,
    )


def estimate_data_downlink(
    windows: list[AccessWindow],
    data_rate_bps: float,
    efficiency: float = 0.8,
) -> DataDownlinkEstimate:
    """Estimate daily data downlink volume from access windows.

    data_bytes = contact_seconds * data_rate_bps * efficiency / 8

    Args:
        windows: List of AccessWindow objects.
        data_rate_bps: Data rate in bits per second.
        efficiency: Link efficiency factor (0 to 1, default 0.8).

    Returns:
        DataDownlinkEstimate with contact time, bytes, and human-readable string.
    """
    total_contact = sum(w.duration_seconds for w in windows)
    data_bytes = total_contact * data_rate_bps * efficiency / 8.0

    # Human-readable format
    if data_bytes >= 1e9:
        readable = f"{data_bytes / 1e9:.2f} GB/day"
    elif data_bytes >= 1e6:
        readable = f"{data_bytes / 1e6:.2f} MB/day"
    elif data_bytes >= 1e3:
        readable = f"{data_bytes / 1e3:.2f} KB/day"
    else:
        readable = f"{data_bytes:.0f} B/day"

    return DataDownlinkEstimate(
        daily_contact_seconds=total_contact,
        daily_data_bytes=data_bytes,
        human_readable=readable,
    )


def compute_visible_magnitude(
    slant_range_km: float,
    phase_angle_deg: float,
    cross_section_m2: float,
    albedo: float = 0.25,
) -> float:
    """Estimate apparent visual magnitude of a satellite.

    Uses the diffuse sphere (Lambertian) model:
    F_sat / F_sun = (albedo * cross_section * phase_function) / (pi * range²)
    mag = sun_mag - 2.5 * log10(F_sat / F_sun)

    Phase function for diffuse sphere:
    phi(alpha) = (2/(3*pi)) * ((pi - alpha) * cos(alpha) + sin(alpha))

    Ref: Hejduk & Snow (2019), simplified.

    Args:
        slant_range_km: Slant range from observer to satellite (km).
        phase_angle_deg: Sun-satellite-observer angle (degrees). 0=full, 180=new.
        cross_section_m2: Effective cross-section area (m²).
        albedo: Diffuse reflectivity (0 to 1, default 0.25).

    Returns:
        Apparent visual magnitude (lower = brighter).
    """
    if slant_range_km <= 0:
        raise ValueError(f"slant_range_km must be positive, got {slant_range_km}")
    if cross_section_m2 < 0:
        raise ValueError(f"cross_section_m2 must be non-negative, got {cross_section_m2}")

    range_m = slant_range_km * 1000.0
    alpha_rad = math.radians(phase_angle_deg)

    # Diffuse sphere phase function
    # phi(alpha) = (2/(3*pi)) * ((pi - alpha)*cos(alpha) + sin(alpha))
    phase_fn = (2.0 / (3.0 * math.pi)) * (
        (math.pi - alpha_rad) * math.cos(alpha_rad) + math.sin(alpha_rad)
    )

    # Avoid log of zero for fully dark (phase_angle = 180)
    if phase_fn < 1e-20:
        return 30.0  # effectively invisible

    # Flux ratio: F_sat / F_sun
    flux_ratio = albedo * cross_section_m2 * phase_fn / (math.pi * range_m**2)

    if flux_ratio < 1e-30:
        return 30.0

    # Apparent magnitude
    mag = _SUN_MAG - 2.5 * math.log10(flux_ratio)

    return mag
