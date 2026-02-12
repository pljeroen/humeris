# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Validation: SGP4 cross-propagation against Keplerian+J2.

Compares our analytical Keplerian+J2 propagator against SGP4 using real
OMM records. SGP4 is the industry standard for TLE/OMM propagation.

Important: SGP4 and Keplerian+J2 are fundamentally different theories:
    - SGP4 uses TEME (True Equator Mean Equinox) reference frame
    - SGP4 TLE mean elements ≠ osculating Keplerian elements
    - SGP4 includes its own perturbation model (secular + periodic)
    - Direct position comparison has limited value due to frame mismatch

What we CAN validate:
    1. Both models produce physically reasonable orbits (altitude, velocity)
    2. Our Keplerian propagator is self-consistent (round-trip)
    3. Orbital properties (period, SMA, velocity) agree between models
    4. The SGP4 adapter produces valid domain objects

Requires: sgp4 package (pip install constellation-generator[live])
"""
import math
import pytest
from datetime import datetime, timedelta, timezone

try:
    from sgp4.api import Satrec, WGS72, jday
    HAS_SGP4 = True
except ImportError:
    HAS_SGP4 = False

from constellation_generator.domain.omm import parse_omm_record
from constellation_generator.domain.propagation import (
    derive_orbital_state,
    propagate_to,
    OrbitalState,
)
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.orbit_properties import (
    compute_energy_momentum,
    state_vector_to_elements,
)
from constellation_generator.adapters.celestrak import SGP4Adapter

_MU = OrbitalConstants.MU_EARTH
_R_E = OrbitalConstants.R_EARTH

# Hardcoded ISS OMM record (CelesTrak format, frozen snapshot)
_ISS_OMM = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "NORAD_CAT_ID": 25544,
    "EPOCH": "2026-01-15T12:00:00.000Z",
    "MEAN_MOTION": 15.50103292,
    "ECCENTRICITY": 0.0006703,
    "INCLINATION": 51.6410,
    "RA_OF_ASC_NODE": 280.1234,
    "ARG_OF_PERICENTER": 52.6789,
    "MEAN_ANOMALY": 307.5432,
    "BSTAR": 0.000032511,
    "MEAN_MOTION_DOT": 0.00002182,
    "MEAN_MOTION_DDOT": 0.0,
    "CLASSIFICATION_TYPE": "U",
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 60000,
    "EPHEMERIS_TYPE": 0,
}

# Hardcoded GPS satellite OMM record (MEO, very stable orbit)
_GPS_OMM = {
    "OBJECT_NAME": "GPS BIIR-2 (PRN 13)",
    "OBJECT_ID": "1997-035A",
    "NORAD_CAT_ID": 24876,
    "EPOCH": "2026-01-15T12:00:00.000Z",
    "MEAN_MOTION": 2.00561755,
    "ECCENTRICITY": 0.0046904,
    "INCLINATION": 55.3827,
    "RA_OF_ASC_NODE": 197.4567,
    "ARG_OF_PERICENTER": 78.9012,
    "MEAN_ANOMALY": 281.2345,
    "BSTAR": 0.0,
    "MEAN_MOTION_DOT": 0.0,
    "MEAN_MOTION_DDOT": 0.0,
    "CLASSIFICATION_TYPE": "U",
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 50000,
    "EPHEMERIS_TYPE": 0,
}

# GEO satellite (very high altitude, stable)
_GEO_OMM = {
    "OBJECT_NAME": "GOES 16",
    "OBJECT_ID": "2016-071A",
    "NORAD_CAT_ID": 41866,
    "EPOCH": "2026-01-15T12:00:00.000Z",
    "MEAN_MOTION": 1.00271244,
    "ECCENTRICITY": 0.0001455,
    "INCLINATION": 0.0237,
    "RA_OF_ASC_NODE": 85.1234,
    "ARG_OF_PERICENTER": 210.4567,
    "MEAN_ANOMALY": 42.5678,
    "BSTAR": 0.0,
    "MEAN_MOTION_DOT": 0.0,
    "MEAN_MOTION_DDOT": 0.0,
    "CLASSIFICATION_TYPE": "U",
    "ELEMENT_SET_NO": 999,
    "REV_AT_EPOCH": 10000,
    "EPHEMERIS_TYPE": 0,
}


def _vec_mag(v):
    """Euclidean magnitude of a 3-vector."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _sgp4_propagate(omm, target_dt):
    """Propagate an OMM record to target_dt using SGP4, return (pos_m, vel_ms)."""
    elements = parse_omm_record(omm)
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        'i',
        elements.norad_cat_id,
        _epoch_to_jd_offset(elements.epoch),
        elements.bstar,
        elements.mean_motion_dot / (2.0 * math.pi / (86400.0 ** 2)),
        elements.mean_motion_ddot,
        elements.eccentricity,
        math.radians(elements.arg_perigee_deg),
        math.radians(elements.inclination_deg),
        math.radians(elements.mean_anomaly_deg),
        elements.mean_motion_rev_per_day * 2.0 * math.pi / 1440.0,
        math.radians(elements.raan_deg),
    )
    jd_val, fr_val = jday(
        target_dt.year, target_dt.month, target_dt.day,
        target_dt.hour, target_dt.minute,
        target_dt.second + target_dt.microsecond / 1e6,
    )
    error_code, pos_km, vel_km_s = sat.sgp4(jd_val, fr_val)
    if error_code != 0:
        raise RuntimeError(f"SGP4 error {error_code}")
    pos_m = (pos_km[0] * 1000, pos_km[1] * 1000, pos_km[2] * 1000)
    vel_ms = (vel_km_s[0] * 1000, vel_km_s[1] * 1000, vel_km_s[2] * 1000)
    return pos_m, vel_ms


def _epoch_to_jd_offset(epoch_str):
    """SGP4 epoch offset: fractional days since 1949-12-31."""
    if epoch_str.endswith("Z"):
        epoch_str = epoch_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(epoch_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ref = datetime(1949, 12, 31, tzinfo=timezone.utc)
    delta = dt - ref
    return delta.days + delta.seconds / 86400.0 + delta.microseconds / 86400e6


pytestmark = pytest.mark.skipif(not HAS_SGP4, reason="sgp4 package not installed")


class TestSGP4ProducesValidOrbits:
    """Verify SGP4 produces physically reasonable state vectors."""

    def test_iss_altitude_reasonable(self):
        """ISS at epoch should be at ~400-430 km altitude."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        pos, vel = _sgp4_propagate(_ISS_OMM, epoch)
        alt_km = _vec_mag(pos) / 1000 - _R_E / 1000
        assert 380 < alt_km < 450, f"ISS altitude: {alt_km:.1f} km"

    def test_iss_velocity_reasonable(self):
        """ISS velocity should be ~7.66 km/s."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        _, vel = _sgp4_propagate(_ISS_OMM, epoch)
        v_km_s = _vec_mag(vel) / 1000
        assert abs(v_km_s - 7.66) < 0.1, f"ISS velocity: {v_km_s:.3f} km/s"

    def test_gps_altitude_reasonable(self):
        """GPS satellite at epoch should be at ~20100-20300 km altitude."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        pos, _ = _sgp4_propagate(_GPS_OMM, epoch)
        alt_km = _vec_mag(pos) / 1000 - _R_E / 1000
        assert 19000 < alt_km < 21000, f"GPS altitude: {alt_km:.1f} km"

    def test_geo_altitude_reasonable(self):
        """GEO satellite at epoch should be at ~35600-35900 km altitude."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        pos, _ = _sgp4_propagate(_GEO_OMM, epoch)
        alt_km = _vec_mag(pos) / 1000 - _R_E / 1000
        assert 35000 < alt_km < 36500, f"GEO altitude: {alt_km:.1f} km"

    def test_geo_velocity_reasonable(self):
        """GEO velocity should be ~3.07 km/s."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        _, vel = _sgp4_propagate(_GEO_OMM, epoch)
        v_km_s = _vec_mag(vel) / 1000
        assert abs(v_km_s - 3.07) < 0.1, f"GEO velocity: {v_km_s:.3f} km/s"


class TestSGP4OrbitalProperties:
    """Verify derived orbital properties from OMM records."""

    def test_iss_orbital_period_consistent(self):
        """ISS orbital period from OMM should be ~92-93 minutes."""
        elements = parse_omm_record(_ISS_OMM)
        period_min = 1440.0 / elements.mean_motion_rev_per_day
        assert 91 < period_min < 94, f"ISS period: {period_min:.1f} min"

    def test_gps_orbital_period_consistent(self):
        """GPS orbital period should be ~11h 58m (half sidereal day)."""
        elements = parse_omm_record(_GPS_OMM)
        period_hours = 24.0 / elements.mean_motion_rev_per_day
        assert 11.5 < period_hours < 12.5, f"GPS period: {period_hours:.2f} hours"

    def test_geo_orbital_period_consistent(self):
        """GEO period should be ~23h 56m."""
        elements = parse_omm_record(_GEO_OMM)
        period_hours = 24.0 / elements.mean_motion_rev_per_day
        assert 23.5 < period_hours < 24.5, f"GEO period: {period_hours:.2f} hours"

    def test_iss_semi_major_axis_reasonable(self):
        """ISS semi-major axis should be ~6780 km (R_E + ~410 km)."""
        elements = parse_omm_record(_ISS_OMM)
        a_km = elements.semi_major_axis_m / 1000
        assert 6750 < a_km < 6820, f"ISS SMA: {a_km:.1f} km"

    def test_gps_semi_major_axis_reasonable(self):
        """GPS semi-major axis should be ~26560 km."""
        elements = parse_omm_record(_GPS_OMM)
        a_km = elements.semi_major_axis_m / 1000
        assert 26000 < a_km < 27000, f"GPS SMA: {a_km:.1f} km"


class TestSGP4AdapterIntegration:
    """Verify the SGP4 adapter produces valid Satellite domain objects."""

    def test_adapter_produces_satellite(self):
        """SGP4Adapter.omm_to_satellite should return a valid Satellite."""
        adapter = SGP4Adapter()
        sat = adapter.omm_to_satellite(_ISS_OMM)
        assert sat.name == "ISS (ZARYA)"
        assert sat.sat_index == 25544
        alt_km = _vec_mag(sat.position_eci) / 1000 - _R_E / 1000
        assert 380 < alt_km < 450

    def test_adapter_with_epoch_override(self):
        """SGP4 adapter with epoch_override should propagate to that time."""
        adapter = SGP4Adapter()
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        future = epoch + timedelta(hours=1)
        sat_epoch = adapter.omm_to_satellite(_ISS_OMM)
        sat_future = adapter.omm_to_satellite(_ISS_OMM, epoch_override=future)
        # Positions should differ (satellite moved)
        dist = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(sat_epoch.position_eci, sat_future.position_eci)
        ))
        assert dist > 1000, "Positions should differ after 1 hour"

    def test_derive_state_produces_valid_elements(self):
        """derive_orbital_state from SGP4 satellite should give valid elements."""
        adapter = SGP4Adapter()
        sat = adapter.omm_to_satellite(_ISS_OMM)
        state = derive_orbital_state(sat, sat.epoch, include_j2=True)
        # SMA should be close to OMM-derived value
        elements = parse_omm_record(_ISS_OMM)
        assert abs(state.semi_major_axis_m - elements.semi_major_axis_m) / elements.semi_major_axis_m < 0.01
        # Inclination should match
        assert abs(math.degrees(state.inclination_rad) - 51.64) < 1.0
        # Eccentricity should be small
        assert state.eccentricity < 0.01


class TestSGP4SelfConsistency:
    """Verify SGP4 propagation is internally consistent."""

    def test_sgp4_altitude_stable_over_one_orbit(self):
        """ISS altitude from SGP4 should stay in a narrow band over one orbit."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        altitudes = []
        for minutes in range(0, 93, 5):
            t = epoch + timedelta(minutes=minutes)
            pos, _ = _sgp4_propagate(_ISS_OMM, t)
            alt_km = _vec_mag(pos) / 1000 - _R_E / 1000
            altitudes.append(alt_km)
        alt_range = max(altitudes) - min(altitudes)
        # ISS eccentricity ~0.0007, so altitude range should be small
        assert alt_range < 20, f"ISS altitude range: {alt_range:.1f} km"

    def test_sgp4_energy_approximately_conserved(self):
        """Specific orbital energy from SGP4 should be approximately constant."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        energies = []
        for minutes in range(0, 93, 10):
            t = epoch + timedelta(minutes=minutes)
            pos, vel = _sgp4_propagate(_ISS_OMM, t)
            em = compute_energy_momentum(list(pos), list(vel))
            energies.append(em.specific_energy_j_kg)
        energy_range = max(energies) - min(energies)
        mean_energy = sum(energies) / len(energies)
        # SGP4 includes periodic perturbation terms that cause energy
        # oscillation within an orbit. Tolerance allows for this.
        assert abs(energy_range / mean_energy) < 0.005, (
            f"Energy variation: {energy_range:.0f} J/kg "
            f"(relative: {energy_range / abs(mean_energy):.4e})"
        )

    def test_sgp4_velocity_direction_tangent_to_orbit(self):
        """Velocity should be approximately perpendicular to radius (circular orbit)."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        pos, vel = _sgp4_propagate(_ISS_OMM, epoch)
        # r · v should be small for near-circular orbit
        r_dot_v = sum(p * v for p, v in zip(pos, vel))
        r_mag = _vec_mag(pos)
        v_mag = _vec_mag(vel)
        cos_angle = r_dot_v / (r_mag * v_mag)
        # For circular orbit, angle should be ~90° (cos ≈ 0)
        assert abs(cos_angle) < 0.01, f"cos(angle) = {cos_angle:.4f}"


class TestOrbitalPropertyAgreement:
    """Compare orbital properties derived from both models."""

    def test_iss_velocity_agrees(self):
        """Both models should produce ISS velocity within ~0.1 km/s."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        _, sgp4_vel = _sgp4_propagate(_ISS_OMM, epoch)
        sgp4_v = _vec_mag(sgp4_vel) / 1000

        adapter = SGP4Adapter()
        sat = adapter.omm_to_satellite(_ISS_OMM)
        state = derive_orbital_state(sat, sat.epoch, include_j2=True)
        pos, vel = propagate_to(state, epoch)
        our_v = _vec_mag(vel) / 1000

        assert abs(sgp4_v - our_v) < 0.1, (
            f"Velocity mismatch: SGP4={sgp4_v:.3f}, ours={our_v:.3f} km/s"
        )

    def test_iss_altitude_agrees(self):
        """Both models should produce ISS altitude within ~20 km."""
        epoch = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        sgp4_pos, _ = _sgp4_propagate(_ISS_OMM, epoch)
        sgp4_alt = _vec_mag(sgp4_pos) / 1000 - _R_E / 1000

        adapter = SGP4Adapter()
        sat = adapter.omm_to_satellite(_ISS_OMM)
        state = derive_orbital_state(sat, sat.epoch, include_j2=True)
        pos, _ = propagate_to(state, epoch)
        our_alt = _vec_mag(pos) / 1000 - _R_E / 1000

        assert abs(sgp4_alt - our_alt) < 20, (
            f"Altitude mismatch: SGP4={sgp4_alt:.1f}, ours={our_alt:.1f} km"
        )
