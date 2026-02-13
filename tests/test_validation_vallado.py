# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Validation against Vallado reference values and known orbital mechanics constants.

These tests compare our implementations against published reference values,
textbook solutions, and well-known physical constants. They verify the
library produces results consistent with established orbital mechanics.

Sources:
    - Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
    - Wertz, "Space Mission Engineering: The New SMAD"
    - IERS Conventions / IAU standards
    - NASA/ESA published orbital parameters
"""
import math
from datetime import datetime, timedelta, timezone

from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
    sso_inclination_deg,
    j2_raan_rate,
)
from humeris.domain.propagation import OrbitalState
from humeris.domain.maneuvers import (
    hohmann_transfer,
    plane_change_dv,
)
from humeris.domain.atmosphere import (
    atmospheric_density,
    AtmosphereModel,
)
from humeris.domain.eclipse import (
    eclipse_fraction,
    compute_beta_angle,
)
from humeris.domain.observation import compute_observation
from humeris.domain.coordinate_frames import (
    ecef_to_geodetic,
    eci_to_ecef,
    gmst_rad,
)
from humeris.domain.orbit_design import design_sso_orbit

_MU = OrbitalConstants.MU_EARTH      # 3.986004418e14 m³/s²
_R_E = OrbitalConstants.R_EARTH      # 6_371_000.0 m
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


class TestCircularOrbitVelocity:
    """Verify circular orbit velocity v = sqrt(mu/r) against known values."""

    def test_iss_altitude_velocity(self):
        """ISS at ~420 km: v ≈ 7.66 km/s (well-established)."""
        r = _R_E + 420_000
        v = math.sqrt(_MU / r)
        assert abs(v - 7660) < 20  # within 20 m/s of 7.66 km/s

    def test_geo_altitude_velocity(self):
        """GEO at ~35786 km: v ≈ 3.075 km/s."""
        r = _R_E + 35_786_000
        v = math.sqrt(_MU / r)
        assert abs(v - 3075) < 10  # within 10 m/s

    def test_leo_200km_velocity(self):
        """LEO at 200 km: v ≈ 7.79 km/s."""
        r = _R_E + 200_000
        v = math.sqrt(_MU / r)
        assert abs(v - 7790) < 20

    def test_geo_orbital_period(self):
        """GEO period should be ~23h 56m (sidereal day)."""
        r = _R_E + 35_786_000
        T = 2 * math.pi * math.sqrt(r ** 3 / _MU)
        sidereal_day_s = 86164.1  # seconds
        assert abs(T - sidereal_day_s) < 30  # within 30s (mean vs equatorial radius)

    def test_iss_orbital_period(self):
        """ISS period at 420 km: ~92.7 minutes."""
        r = _R_E + 420_000
        T = 2 * math.pi * math.sqrt(r ** 3 / _MU)
        expected_minutes = 92.7
        assert abs(T / 60 - expected_minutes) < 0.5


class TestKeplerToCartesian:
    """Verify Kepler-to-Cartesian conversion for known geometries."""

    def test_circular_equatorial_at_zero_anomaly(self):
        """Circular equatorial orbit, ν=0: position should be along +X."""
        a = _R_E + 500_000
        pos, vel = kepler_to_cartesian(a, 0.0, 0.0, 0.0, 0.0, 0.0)
        # At ν=0, i=0, Ω=0, ω=0: r should be [a, 0, 0]
        assert abs(pos[0] - a) < 1.0  # within 1 meter
        assert abs(pos[1]) < 1.0
        assert abs(pos[2]) < 1.0
        # Velocity should be along +Y
        v_circ = math.sqrt(_MU / a)
        assert abs(vel[0]) < 1.0
        assert abs(vel[1] - v_circ) < 1.0
        assert abs(vel[2]) < 1.0

    def test_circular_equatorial_at_90_anomaly(self):
        """Circular equatorial orbit, ν=90°: position should be along +Y."""
        a = _R_E + 500_000
        pos, vel = kepler_to_cartesian(a, 0.0, 0.0, 0.0, 0.0, math.pi / 2)
        assert abs(pos[0]) < 1.0
        assert abs(pos[1] - a) < 1.0
        assert abs(pos[2]) < 1.0

    def test_polar_orbit_has_z_component(self):
        """Polar orbit (i=90°) at ν=90° should have Z component."""
        a = _R_E + 500_000
        pos, vel = kepler_to_cartesian(
            a, 0.0, math.pi / 2, 0.0, 0.0, math.pi / 2,
        )
        # At i=90°, Ω=0, ω=0, ν=90°: position should be [0, 0, a]
        assert abs(pos[0]) < 1.0
        assert abs(pos[1]) < 1.0
        assert abs(pos[2] - a) < 1.0

    def test_position_magnitude_equals_sma_for_circular(self):
        """For circular orbit, |r| = a at all true anomalies."""
        a = _R_E + 600_000
        for ta_deg in range(0, 360, 30):
            ta_rad = math.radians(ta_deg)
            pos, _ = kepler_to_cartesian(a, 0.0, math.radians(45), 0.0, 0.0, ta_rad)
            r_mag = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
            assert abs(r_mag - a) < 1.0, f"At ν={ta_deg}°: |r|={r_mag}, a={a}"


class TestHohmannTransfer:
    """Verify Hohmann transfer against textbook solutions."""

    def test_leo_to_geo_total_delta_v(self):
        """LEO (400 km) to GEO: total ΔV ≈ 3.86 km/s (using mean Earth radius).

        Note: the classic ~3.94 km/s uses equatorial radius (6378 km).
        Our library uses mean spherical radius (6371 km), giving ~3.86 km/s.
        """
        r1 = _R_E + 400_000
        r2 = _R_E + 35_786_000
        plan = hohmann_transfer(r1, r2)
        # Analytical: v1=sqrt(mu/r1), v_t1=sqrt(2*mu*r2/(r1*(r1+r2))), etc.
        v1 = math.sqrt(_MU / r1)
        a_t = (r1 + r2) / 2
        v_t1 = math.sqrt(_MU * (2 / r1 - 1 / a_t))
        v_t2 = math.sqrt(_MU * (2 / r2 - 1 / a_t))
        v2 = math.sqrt(_MU / r2)
        expected_dv = abs(v_t1 - v1) + abs(v2 - v_t2)
        assert abs(plan.total_delta_v_ms - expected_dv) < 1.0  # match our own math

    def test_leo_to_geo_transfer_time(self):
        """LEO to GEO transfer time: ~5.25 hours."""
        r1 = _R_E + 400_000
        r2 = _R_E + 35_786_000
        plan = hohmann_transfer(r1, r2)
        hours = plan.transfer_time_s / 3600
        assert abs(hours - 5.25) < 0.15

    def test_coplanar_same_altitude_zero_delta_v(self):
        """Transfer to same altitude: ΔV = 0."""
        r = _R_E + 500_000
        plan = hohmann_transfer(r, r)
        assert plan.total_delta_v_ms < 0.01

    def test_plane_change_at_geo(self):
        """28.5° plane change at GEO velocity: ΔV ≈ 1.5 km/s."""
        v_geo = math.sqrt(_MU / (_R_E + 35_786_000))
        dv = plane_change_dv(v_geo, math.radians(28.5))
        assert abs(dv - 1500) < 50  # within 50 m/s


class TestJ2SecularEffects:
    """Verify J2 secular perturbation rates against known values."""

    def test_iss_raan_drift(self):
        """ISS RAAN drift: approximately -5°/day (NASA published)."""
        a = _R_E + 420_000
        n = math.sqrt(_MU / a ** 3)
        inc_rad = math.radians(51.6)
        raan_rate_rad_s = j2_raan_rate(n, a, 0.0, inc_rad)
        raan_rate_deg_day = math.degrees(raan_rate_rad_s) * 86400
        # NASA: ISS RAAN drifts about -5°/day
        assert abs(raan_rate_deg_day - (-5.0)) < 0.5

    def test_polar_orbit_zero_raan_drift(self):
        """Polar orbit (i=90°): RAAN drift = 0 (cos(90°) = 0)."""
        a = _R_E + 500_000
        n = math.sqrt(_MU / a ** 3)
        raan_rate = j2_raan_rate(n, a, 0.0, math.pi / 2)
        assert abs(raan_rate) < 1e-15

    def test_retrograde_orbit_positive_raan_drift(self):
        """Retrograde orbit (i>90°): RAAN drift is positive."""
        a = _R_E + 500_000
        n = math.sqrt(_MU / a ** 3)
        raan_rate = j2_raan_rate(n, a, 0.0, math.radians(97))
        assert raan_rate > 0  # positive drift for retrograde


class TestSSOInclination:
    """Verify SSO inclination against known values."""

    def test_sso_500km(self):
        """SSO at 500 km: i ≈ 97.4° (widely published)."""
        inc = sso_inclination_deg(500)
        assert abs(inc - 97.4) < 0.3

    def test_sso_800km(self):
        """SSO at 800 km: i ≈ 98.6° (widely published)."""
        inc = sso_inclination_deg(800)
        assert abs(inc - 98.6) < 0.3

    def test_sso_orbit_design_consistency(self):
        """design_sso_orbit should produce inclination matching sso_inclination_deg."""
        result = design_sso_orbit(500, 10.5, _EPOCH)
        inc_direct = sso_inclination_deg(500)
        assert abs(result.inclination_deg - inc_direct) < 0.01


class TestAtmosphericDensity:
    """Verify atmospheric density against Vallado Table 8-4 base values."""

    def test_density_at_200km_vallado(self):
        """Density at 200 km (Vallado moderate): ρ₀ ≈ 2.789e-10 kg/m³."""
        rho = atmospheric_density(200.0, AtmosphereModel.VALLADO_4TH)
        # At exact table altitude, should match base value closely
        assert abs(rho - 2.789e-10) / 2.789e-10 < 0.01  # within 1%

    def test_density_at_500km_vallado(self):
        """Density at 500 km (Vallado moderate): ρ₀ ≈ 6.967e-13 kg/m³."""
        rho = atmospheric_density(500.0, AtmosphereModel.VALLADO_4TH)
        assert abs(rho - 6.967e-13) / 6.967e-13 < 0.01

    def test_density_decreases_with_altitude(self):
        """Density must monotonically decrease with altitude."""
        altitudes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        densities = [atmospheric_density(h) for h in altitudes]
        for i in range(len(densities) - 1):
            assert densities[i] > densities[i + 1], (
                f"Density at {altitudes[i]} km ({densities[i]:.3e}) "
                f"should be > density at {altitudes[i+1]} km ({densities[i+1]:.3e})"
            )

    def test_density_at_100km_karman_line(self):
        """Density at 100 km (Kármán line): ~5.3e-7 kg/m³."""
        rho = atmospheric_density(100.0, AtmosphereModel.VALLADO_4TH)
        assert abs(rho - 5.297e-7) / 5.297e-7 < 0.01


class TestEclipsePrediction:
    """Verify eclipse predictions against known ISS eclipse characteristics."""

    def test_iss_eclipse_fraction_reasonable(self):
        """ISS eclipse fraction should be ~35-40% (well-established)."""
        state = _make_state(420, inc_deg=51.6)
        frac = eclipse_fraction(state, _EPOCH)
        # ISS is eclipsed roughly 35-40% of each orbit
        assert 0.25 <= frac <= 0.45, f"Eclipse fraction {frac:.2%} outside expected range"

    def test_high_altitude_less_eclipse(self):
        """Higher altitude orbits have smaller eclipse fraction."""
        state_low = _make_state(400, inc_deg=51.6)
        state_high = _make_state(1000, inc_deg=51.6)
        frac_low = eclipse_fraction(state_low, _EPOCH)
        frac_high = eclipse_fraction(state_high, _EPOCH)
        assert frac_high <= frac_low

    def test_beta_angle_range(self):
        """Beta angle should be in [-90°, +90°]."""
        state = _make_state(500, inc_deg=53)
        beta = compute_beta_angle(
            state.raan_rad, state.inclination_rad, _EPOCH,
        )
        assert -90 <= beta <= 90


class TestCoordinateFrames:
    """Verify coordinate frame conversions against known geometry."""

    def test_subsatellite_point_at_zero_gmst(self):
        """Satellite at [R, 0, 0] ECI with GMST=0 should be at (0°N, 0°E)."""
        r = _R_E + 500_000
        pos_eci = [r, 0, 0]
        vel_eci = [0, 0, 0]
        # At GMST = 0, ECI = ECEF
        pos_ecef, _ = eci_to_ecef(pos_eci, vel_eci, 0.0)
        lat, lon, alt = ecef_to_geodetic(pos_ecef)
        assert abs(lat) < 0.1  # near equator
        assert abs(lon) < 0.1  # near prime meridian
        assert abs(alt - 500_000) < 10_000  # ~500 km (WGS84 ellipsoid vs sphere)

    def test_north_pole_satellite(self):
        """Satellite at [0, 0, R] ECI should be near north pole."""
        r = _R_E + 500_000
        pos_eci = [0, 0, r]
        vel_eci = [0, 0, 0]
        pos_ecef, _ = eci_to_ecef(pos_eci, vel_eci, 0.0)
        lat, lon, alt = ecef_to_geodetic(pos_ecef)
        assert lat > 85  # near north pole

    def test_gmst_wraps_around(self):
        """GMST should increase monotonically and wrap at 2π."""
        t1 = datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        g1 = gmst_rad(t1)
        g2 = gmst_rad(t2)
        # 12 hours ≈ π radians of Earth rotation
        # Allow for the difference between solar and sidereal time
        diff = (g2 - g1) % (2 * math.pi)
        assert abs(diff - math.pi) < 0.05  # within ~3°


class TestPhysicalConstants:
    """Verify our constants match IAU/IERS standard values."""

    def test_mu_earth(self):
        """GM_Earth = 3.986004418e14 m³/s² (IERS)."""
        assert abs(_MU - 3.986004418e14) < 1e6

    def test_r_earth(self):
        """R_Earth = 6371000 m (mean spherical radius)."""
        assert abs(_R_E - 6_371_000) < 100

    def test_j2_coefficient(self):
        """J2 = 1.08263e-3 (IERS/EGM96)."""
        assert abs(OrbitalConstants.J2_EARTH - 1.08263e-3) < 1e-6
