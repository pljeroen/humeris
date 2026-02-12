# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Validation: cross-checking between library modules.

These tests verify internal consistency — that different modules produce
compatible results when applied to the same physical scenario. This catches
integration bugs where individual modules work correctly in isolation
but produce contradictory results when composed.

Scenarios:
    1. Eclipse fraction from geometry vs eclipse windows integration
    2. Access windows vs observation module consistency
    3. Orbital velocity from orbit_properties vs analytical formula
    4. Propagation round-trip: Kepler elements → state vector → elements
    5. J2 RAAN drift vs SSO condition consistency
    6. Energy conservation during propagation
    7. Beta angle vs eclipse fraction correlation
    8. Coordinate frame round-trips
"""
import math
from datetime import datetime, timedelta, timezone

from constellation_generator.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
    j2_raan_rate,
    sso_inclination_deg,
)
from constellation_generator.domain.propagation import (
    OrbitalState,
    propagate_to,
)
from constellation_generator.domain.eclipse import (
    eclipse_fraction,
    compute_eclipse_windows,
    compute_beta_angle,
)
from constellation_generator.domain.observation import (
    compute_observation,
    GroundStation,
)
from constellation_generator.domain.access_windows import compute_access_windows
from constellation_generator.domain.coordinate_frames import (
    ecef_to_geodetic,
    eci_to_ecef,
    geodetic_to_ecef,
    gmst_rad,
)
from constellation_generator.domain.orbit_properties import (
    compute_orbital_velocity,
    compute_energy_momentum,
    state_vector_to_elements,
)
from constellation_generator.domain.orbit_design import design_sso_orbit

_MU = OrbitalConstants.MU_EARTH
_R_E = OrbitalConstants.R_EARTH
_EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_state(alt_km, inc_deg=0.0, ecc=0.0, raan_deg=0.0,
                arg_perigee_deg=0.0, ta_deg=0.0):
    """Helper to create an OrbitalState from altitude and angles."""
    a = _R_E + alt_km * 1000.0
    n = math.sqrt(_MU / a ** 3)
    return OrbitalState(
        semi_major_axis_m=a, eccentricity=ecc,
        inclination_rad=math.radians(inc_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=math.radians(arg_perigee_deg),
        true_anomaly_rad=math.radians(ta_deg),
        mean_motion_rad_s=n, reference_epoch=_EPOCH,
    )


class TestEclipseConsistency:
    """Eclipse fraction and eclipse windows should agree."""

    def test_eclipse_fraction_matches_windows_integration(self):
        """Eclipse fraction from geometry ≈ total eclipse time / orbit period.

        Both methods compute eclipse for the same orbit; results should agree
        within 5% (methods use different sampling strategies).
        """
        state = _make_state(500, inc_deg=51.6)
        frac = eclipse_fraction(state, _EPOCH)

        windows = compute_eclipse_windows(
            state, _EPOCH,
            duration=timedelta(hours=1.6),
            step=timedelta(seconds=10),
        )
        period = 2 * math.pi / state.mean_motion_rad_s
        total_eclipse_s = sum(w.duration_seconds for w in windows)
        frac_from_windows = total_eclipse_s / period

        assert abs(frac - frac_from_windows) < 0.05, (
            f"Fraction methods disagree: geometry={frac:.3f}, "
            f"windows={frac_from_windows:.3f}"
        )

    def test_high_beta_angle_means_less_eclipse(self):
        """When |beta| is large, eclipse fraction should be small or zero."""
        # SSO at equinox has beta ≈ 0 (max eclipse)
        state_equinox = _make_state(500, inc_deg=97.4, raan_deg=0)
        frac_equinox = eclipse_fraction(state_equinox, _EPOCH)

        # Orbit with RAAN giving high beta angle
        state_high_beta = _make_state(500, inc_deg=97.4, raan_deg=90)
        beta = compute_beta_angle(
            state_high_beta.raan_rad,
            state_high_beta.inclination_rad,
            _EPOCH,
        )
        frac_high_beta = eclipse_fraction(state_high_beta, _EPOCH)

        # If |beta| > 60°, eclipse fraction should be significantly less
        if abs(beta) > 60:
            assert frac_high_beta < frac_equinox


class TestVelocityConsistency:
    """Orbital velocity from different methods should agree."""

    def test_orbit_properties_vs_analytical(self):
        """compute_orbital_velocity should match v = sqrt(mu/r) for circular."""
        state = _make_state(500, inc_deg=45)
        ov = compute_orbital_velocity(state)
        v_analytical = math.sqrt(_MU / state.semi_major_axis_m)
        assert abs(ov.circular_velocity_ms - v_analytical) < 0.01

    def test_period_from_orbit_properties_vs_kepler(self):
        """Period from orbit_properties should match T = 2π√(a³/μ)."""
        state = _make_state(800, inc_deg=53)
        ov = compute_orbital_velocity(state)
        T_kepler = 2 * math.pi * math.sqrt(state.semi_major_axis_m ** 3 / _MU)
        assert abs(ov.orbital_period_s - T_kepler) < 0.01

    def test_energy_momentum_from_state_vector(self):
        """Specific energy from state vector should match -mu/(2a)."""
        state = _make_state(600, inc_deg=45)
        pos, vel = propagate_to(state, _EPOCH)
        em = compute_energy_momentum(pos, vel)
        expected_energy = -_MU / (2 * state.semi_major_axis_m)
        assert abs(em.specific_energy_j_kg - expected_energy) / abs(expected_energy) < 0.001


class TestPropagationRoundTrip:
    """Propagation should preserve orbital elements for circular orbits."""

    def test_elements_roundtrip_circular(self):
        """State → propagate → state_vector_to_elements should recover elements."""
        state = _make_state(500, inc_deg=51.6, raan_deg=45, ta_deg=30)
        pos, vel = propagate_to(state, _EPOCH)
        recovered = state_vector_to_elements(pos, vel)

        assert abs(recovered["semi_major_axis_m"] - state.semi_major_axis_m) < 10
        assert abs(recovered["eccentricity"]) < 0.001
        assert abs(recovered["inclination_deg"] - 51.6) < 0.1

    def test_propagation_preserves_altitude_circular(self):
        """For circular orbit, altitude should remain constant over one orbit."""
        state = _make_state(500, inc_deg=45)
        period = 2 * math.pi / state.mean_motion_rad_s
        altitudes = []
        for i in range(12):
            t = _EPOCH + timedelta(seconds=period * i / 12)
            pos, _ = propagate_to(state, t)
            r = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
            alt_km = (r - _R_E) / 1000
            altitudes.append(alt_km)
        alt_range = max(altitudes) - min(altitudes)
        assert alt_range < 1.0, f"Altitude range: {alt_range:.2f} km (expected < 1 km)"

    def test_energy_conserved_during_propagation(self):
        """Specific energy should be constant (Keplerian, no perturbations)."""
        state = _make_state(600, inc_deg=53, raan_deg=30)
        period = 2 * math.pi / state.mean_motion_rad_s
        energies = []
        for i in range(8):
            t = _EPOCH + timedelta(seconds=period * i / 8)
            pos, vel = propagate_to(state, t)
            em = compute_energy_momentum(pos, vel)
            energies.append(em.specific_energy_j_kg)
        energy_range = max(energies) - min(energies)
        mean_energy = sum(energies) / len(energies)
        assert energy_range / abs(mean_energy) < 1e-6, (
            f"Energy variation: {energy_range:.2f} J/kg "
            f"(relative: {energy_range / abs(mean_energy):.2e})"
        )


class TestJ2SSOConsistency:
    """J2 RAAN drift and SSO condition should be self-consistent."""

    def test_sso_inclination_gives_earth_rate_raan_drift(self):
        """At SSO inclination, RAAN drift should equal Earth's annual rate.

        SSO condition: dΩ/dt = 360°/365.25 days ≈ 0.9856°/day
        """
        alt_km = 600
        inc_deg = sso_inclination_deg(alt_km)
        a = _R_E + alt_km * 1000
        n = math.sqrt(_MU / a ** 3)
        inc_rad = math.radians(inc_deg)
        raan_rate = j2_raan_rate(n, a, 0.0, inc_rad)
        raan_rate_deg_day = math.degrees(raan_rate) * 86400
        expected_rate = 360.0 / 365.25  # ~0.9856 °/day
        assert abs(raan_rate_deg_day - expected_rate) < 0.02, (
            f"SSO RAAN rate: {raan_rate_deg_day:.4f}°/day, "
            f"expected: {expected_rate:.4f}°/day"
        )

    def test_sso_design_module_consistent(self):
        """design_sso_orbit and sso_inclination_deg should agree."""
        for alt_km in [400, 500, 600, 700, 800]:
            result = design_sso_orbit(alt_km, 10.5, _EPOCH)
            inc_direct = sso_inclination_deg(alt_km)
            assert abs(result.inclination_deg - inc_direct) < 0.01, (
                f"At {alt_km} km: design={result.inclination_deg:.4f}°, "
                f"direct={inc_direct:.4f}°"
            )


class TestCoordinateRoundTrips:
    """Coordinate transformations should be invertible."""

    def test_geodetic_to_ecef_roundtrip(self):
        """geodetic → ECEF → geodetic should recover original coordinates."""
        test_points = [
            (0.0, 0.0, 0.0),          # equator, prime meridian, sea level
            (52.0, 4.7, 0.0),         # Netherlands
            (-33.9, 18.4, 500_000.0),  # Cape Town, 500 km altitude
            (90.0, 0.0, 0.0),         # North pole
            (0.0, 180.0, 200_000.0),   # Date line, 200 km
        ]
        for lat, lon, alt in test_points:
            ecef = geodetic_to_ecef(lat, lon, alt)
            lat2, lon2, alt2 = ecef_to_geodetic(ecef)
            assert abs(lat2 - lat) < 0.01, f"Lat: {lat} → {lat2}"
            lon_diff = (lon2 - lon + 180) % 360 - 180  # handle wraparound
            assert abs(lon_diff) < 0.01, f"Lon: {lon} → {lon2}"
            assert abs(alt2 - alt) < 1.0, f"Alt: {alt} → {alt2}"

    def test_eci_ecef_roundtrip(self):
        """ECI → ECEF at GMST=0 should be identity transform."""
        pos = (7000000.0, 0.0, 0.0)
        vel = (0.0, 7500.0, 0.0)
        ecef_pos, ecef_vel = eci_to_ecef(pos, vel, 0.0)
        # At GMST=0, ECI = ECEF
        for i in range(3):
            assert abs(ecef_pos[i] - pos[i]) < 0.01
            assert abs(ecef_vel[i] - vel[i]) < 0.01


class TestAccessWindowsConsistency:
    """Access windows and observation module should produce consistent results."""

    def test_station_sees_overhead_satellite(self):
        """A satellite directly overhead should have elevation ~90°."""
        # Place station at equator, 0° longitude
        station = GroundStation(name="Equator", lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
        # Satellite directly above at 500 km
        sat_ecef = geodetic_to_ecef(0.0, 0.0, 500_000.0)
        obs = compute_observation(station, sat_ecef)
        assert obs.elevation_deg > 85, f"Overhead elevation: {obs.elevation_deg:.1f}°"

    def test_station_sees_horizon_satellite(self):
        """A satellite on the horizon should have elevation ~0°."""
        station = GroundStation(name="Test", lat_deg=0.0, lon_deg=0.0, alt_m=0.0)
        # Satellite at same altitude but ~24° away (approximate horizon for 500 km)
        sat_ecef = geodetic_to_ecef(0.0, 24.0, 500_000.0)
        obs = compute_observation(station, sat_ecef)
        assert -5 < obs.elevation_deg < 15, (
            f"Horizon elevation: {obs.elevation_deg:.1f}°"
        )

    def test_access_windows_have_positive_elevation(self):
        """All access windows should have max elevation above minimum threshold."""
        station = GroundStation(name="Delft", lat_deg=52.0, lon_deg=4.4, alt_m=0.0)
        state = _make_state(500, inc_deg=51.6)
        windows = compute_access_windows(
            station, state, _EPOCH,
            duration=timedelta(hours=24),
            step=timedelta(seconds=30),
            min_elevation_deg=10.0,
        )
        for w in windows:
            assert w.max_elevation_deg >= 10.0, (
                f"Window max elevation {w.max_elevation_deg:.1f}° < 10°"
            )
            assert w.duration_seconds > 0


class TestKeplerianIdentities:
    """Verify fundamental Keplerian relationships hold in our implementation."""

    def test_vis_viva_at_periapsis(self):
        """At periapsis of eccentric orbit: v = sqrt(mu * (1+e) / (a*(1-e)))."""
        a = _R_E + 500_000
        e = 0.1
        pos, vel = kepler_to_cartesian(a, e, 0.0, 0.0, 0.0, 0.0)
        v_mag = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        v_expected = math.sqrt(_MU * (1 + e) / (a * (1 - e)))
        assert abs(v_mag - v_expected) < 0.1

    def test_vis_viva_at_apoapsis(self):
        """At apoapsis of eccentric orbit: v = sqrt(mu * (1-e) / (a*(1+e)))."""
        a = _R_E + 500_000
        e = 0.1
        pos, vel = kepler_to_cartesian(a, e, 0.0, 0.0, 0.0, math.pi)
        v_mag = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
        v_expected = math.sqrt(_MU * (1 - e) / (a * (1 + e)))
        assert abs(v_mag - v_expected) < 0.1

    def test_angular_momentum_constant(self):
        """h = r × v should be constant at different true anomalies."""
        a = _R_E + 600_000
        e = 0.05
        h_values = []
        for ta_deg in range(0, 360, 45):
            ta_rad = math.radians(ta_deg)
            pos, vel = kepler_to_cartesian(a, e, math.radians(45), 0.0, 0.0, ta_rad)
            # h = |r × v|
            hx = pos[1] * vel[2] - pos[2] * vel[1]
            hy = pos[2] * vel[0] - pos[0] * vel[2]
            hz = pos[0] * vel[1] - pos[1] * vel[0]
            h = math.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
            h_values.append(h)
        h_range = max(h_values) - min(h_values)
        h_mean = sum(h_values) / len(h_values)
        assert h_range / h_mean < 1e-10, (
            f"Angular momentum variation: {h_range / h_mean:.2e}"
        )
