# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Math verification test suite — permanent regression firewall.

Fast constant assertions, single-evaluation formula checks, and cross-module
consistency tests. Catches categories of bugs found in the v1.28 audit:
sign errors, missing factors, constant drift, integrator coefficient typos.

All tests are pure computation: no propagation runs, no network, no file I/O.
Runtime target: <2 seconds total.
"""

import math
from datetime import datetime, timezone

import pytest

from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
    sso_inclination_deg,
    j2_raan_rate,
)
from humeris.domain.coordinate_frames import (
    gmst_rad,
    eci_to_ecef,
)
from humeris.domain.numerical_propagation import (
    J2Perturbation,
    J3Perturbation,
    AtmosphericDragForce,
    _CS_COEFFS,
)
from humeris.domain.adaptive_integration import (
    DORMAND_PRINCE_A,
    DORMAND_PRINCE_B4,
    DORMAND_PRINCE_B5,
    DORMAND_PRINCE_C,
)
from humeris.domain.tidal_forces import (
    SolidTideForce,
    OceanTideForce,
    _K20,
    _K21,
    _K22,
    _GM_MOON as TIDAL_GM_MOON,
    _GM_SUN as TIDAL_GM_SUN,
    _R_EARTH as TIDAL_R_EARTH,
    _AU as TIDAL_AU,
)
from humeris.domain.atmosphere import (
    atmospheric_density,
    AtmosphereModel,
    DragConfig,
    _ATMOSPHERE_TABLE_VALLADO,
    _ATMOSPHERE_TABLE_HIGH,
)
from humeris.domain.relativistic_forces import (
    SchwarzschildForce,
    _C_LIGHT,
    _GM_SUN as REL_GM_SUN,
)
from humeris.domain.time_systems import (
    _TT_TAI_OFFSET,
    _GPS_TAI_OFFSET,
    _J2000_JD,
)


# ── Shared test fixtures ──────────────────────────────────────────

_LEO_EPOCH = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

# Off-axis LEO test point (not on any axis — catches sign bugs)
_LEO_POS = (4_000_000.0, 5_000_000.0, 3_000_000.0)  # ~7071 km radius
_LEO_VEL = (-3000.0, 4000.0, 2000.0)


# ═══════════════════════════════════════════════════════════════════
# A. Physical Constants
# ═══════════════════════════════════════════════════════════════════


class TestPhysicalConstants:
    """Assert each constant against its authoritative value."""

    def test_mu_earth(self):
        # WGS84 defining parameter
        assert OrbitalConstants.MU_EARTH == 3.986004418e14

    def test_r_earth_equatorial(self):
        # WGS84 defining semi-major axis
        assert OrbitalConstants.R_EARTH_EQUATORIAL == 6_378_137.0

    def test_r_earth_polar(self):
        # WGS84 derived semi-minor axis
        assert OrbitalConstants.R_EARTH_POLAR == 6_356_752.3142

    def test_flattening(self):
        # WGS84 defining flattening
        assert abs(OrbitalConstants.FLATTENING - 1.0 / 298.257223563) < 1e-15

    def test_e_squared(self):
        # WGS84 derived first eccentricity squared
        assert abs(OrbitalConstants.E_SQUARED - 0.00669437999014) < 1e-14

    def test_j2(self):
        # EGM96 J2 zonal harmonic
        assert OrbitalConstants.J2_EARTH == 1.08263e-3

    def test_j3_sign_and_value(self):
        # JGM-3/EGM96 J3 — MUST be negative (sign error caught in audit)
        assert OrbitalConstants.J3_EARTH == -2.53241e-6
        assert OrbitalConstants.J3_EARTH < 0, "J3 must be negative"

    def test_earth_rotation_rate(self):
        # IERS 2010 sidereal rotation rate
        assert OrbitalConstants.EARTH_ROTATION_RATE == 7.2921159e-5

    def test_earth_omega(self):
        # Mean motion around Sun: 2pi / tropical year
        assert OrbitalConstants.EARTH_OMEGA == 1.99106380e-7

    def test_c_light(self):
        # SI 2019 exact definition
        assert _C_LIGHT == 299792458.0

    def test_au(self):
        # IAU 2012 exact definition
        assert TIDAL_AU == 1.495978707e11

    def test_gm_sun(self):
        # DE405 Sun gravitational parameter — must agree across modules
        assert TIDAL_GM_SUN == 1.32712440041e20
        assert REL_GM_SUN == 1.32712440041e20

    def test_gm_moon(self):
        # DE405 Moon gravitational parameter
        assert TIDAL_GM_MOON == 4.9028e12

    def test_tt_tai_offset(self):
        # IAU 1991 exact definition
        assert _TT_TAI_OFFSET == 32.184

    def test_gps_tai_offset(self):
        # GPS ICD exact definition
        assert _GPS_TAI_OFFSET == 19.0


# ═══════════════════════════════════════════════════════════════════
# B. Cross-Module Consistency
# ═══════════════════════════════════════════════════════════════════


class TestCrossModuleConsistency:
    """Verify constants agree across modules that define them independently."""

    def test_j2_matches_cs_coeffs(self):
        # _CS_COEFFS[(2,0)][0] = -J2 (by convention C_n0 = -J_n)
        assert OrbitalConstants.J2_EARTH == -_CS_COEFFS[(2, 0)][0]

    def test_j3_matches_cs_coeffs(self):
        # _CS_COEFFS[(3,0)][0] = -J3 = +2.53241e-6 (positive, since J3 < 0)
        assert OrbitalConstants.J3_EARTH == -_CS_COEFFS[(3, 0)][0]

    def test_earth_omega_computation(self):
        # EARTH_OMEGA should match 2*pi / (365.2422 * 86400)
        computed = 2.0 * math.pi / (365.2422 * 86400.0)
        assert abs(OrbitalConstants.EARTH_OMEGA - computed) < 1e-12

    def test_love_numbers_iers_2010(self):
        # IERS 2010 Table 6.3 frequency-independent Love numbers
        assert _K20 == 0.30190
        assert _K21 == 0.29830
        assert _K22 == 0.30102

    def test_altitude_uses_equatorial_radius(self):
        # Altitude computations must use R_EARTH_EQUATORIAL, not R_EARTH (mean)
        # First verify these are distinct values (otherwise this test is vacuous)
        assert OrbitalConstants.R_EARTH != OrbitalConstants.R_EARTH_EQUATORIAL
        # Tidal forces module must use equatorial radius as its reference
        assert TIDAL_R_EARTH == OrbitalConstants.R_EARTH_EQUATORIAL


# ═══════════════════════════════════════════════════════════════════
# C. Force Model Gradient Verification
# ═══════════════════════════════════════════════════════════════════


class TestForceModelGradients:
    """Compare analytical acceleration to numerical gradient of potential."""

    @staticmethod
    def _j2_potential(x: float, y: float, z: float) -> float:
        """J2 geopotential: Phi = -(mu*J2*Re^2)/2 * (3z^2/r^5 - 1/r^3)."""
        mu = OrbitalConstants.MU_EARTH
        j2 = OrbitalConstants.J2_EARTH
        re = OrbitalConstants.R_EARTH_EQUATORIAL
        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        r3 = r * r2
        r5 = r3 * r2
        return -(mu * j2 * re * re) / 2.0 * (3.0 * z * z / r5 - 1.0 / r3)

    @staticmethod
    def _j3_potential(x: float, y: float, z: float) -> float:
        """J3 geopotential: Phi = -(mu*J3*Re^3)/2 * (5z^3/r^7 - 3z/r^5)."""
        mu = OrbitalConstants.MU_EARTH
        j3 = OrbitalConstants.J3_EARTH
        re = OrbitalConstants.R_EARTH_EQUATORIAL
        r2 = x * x + y * y + z * z
        r = math.sqrt(r2)
        r5 = r2 * r2 * r
        r7 = r5 * r2
        return -(mu * j3 * re ** 3) / 2.0 * (5.0 * z ** 3 / r7 - 3.0 * z / r5)

    @staticmethod
    def _numerical_gradient(potential_fn, x, y, z, h=1.0):
        """Central finite difference gradient with step h."""
        gx = (potential_fn(x + h, y, z) - potential_fn(x - h, y, z)) / (2.0 * h)
        gy = (potential_fn(x, y + h, z) - potential_fn(x, y - h, z)) / (2.0 * h)
        gz = (potential_fn(x, y, z + h) - potential_fn(x, y, z - h)) / (2.0 * h)
        return (gx, gy, gz)

    def test_j2_gradient(self):
        """J2 analytical acceleration matches numerical gradient of potential."""
        force = J2Perturbation()
        ax, ay, az = force.acceleration(_LEO_EPOCH, _LEO_POS, _LEO_VEL)
        gx, gy, gz = self._numerical_gradient(
            self._j2_potential, *_LEO_POS, h=1.0,
        )
        a_mag = math.sqrt(ax * ax + ay * ay + az * az)
        assert a_mag > 0, "J2 acceleration should be nonzero"
        assert abs(ax - gx) / a_mag < 1e-5
        assert abs(ay - gy) / a_mag < 1e-5
        assert abs(az - gz) / a_mag < 1e-5

    def test_j3_gradient(self):
        """J3 analytical acceleration matches numerical gradient — catches sign inversion."""
        force = J3Perturbation()
        ax, ay, az = force.acceleration(_LEO_EPOCH, _LEO_POS, _LEO_VEL)
        gx, gy, gz = self._numerical_gradient(
            self._j3_potential, *_LEO_POS, h=1.0,
        )
        a_mag = math.sqrt(ax * ax + ay * ay + az * az)
        assert a_mag > 0, "J3 acceleration should be nonzero"
        assert abs(ax - gx) / a_mag < 1e-5
        assert abs(ay - gy) / a_mag < 1e-5
        assert abs(az - gz) / a_mag < 1e-5

    def test_solid_tide_magnitude(self):
        """Solid Earth tide acceleration at LEO: [1e-8, 1e-6] m/s^2."""
        force = SolidTideForce()
        ax, ay, az = force.acceleration(_LEO_EPOCH, _LEO_POS, _LEO_VEL)
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        assert 1e-8 <= mag <= 1e-6, f"Solid tide magnitude {mag:.2e} out of range"

    def test_ocean_tide_smaller_than_solid(self):
        """Ocean tide acceleration magnitude < solid tide magnitude."""
        solid = SolidTideForce()
        ocean = OceanTideForce()
        a_solid = solid.acceleration(_LEO_EPOCH, _LEO_POS, _LEO_VEL)
        a_ocean = ocean.acceleration(_LEO_EPOCH, _LEO_POS, _LEO_VEL)
        mag_solid = math.sqrt(sum(c * c for c in a_solid))
        mag_ocean = math.sqrt(sum(c * c for c in a_ocean))
        assert mag_ocean < mag_solid, (
            f"Ocean tide ({mag_ocean:.2e}) >= solid tide ({mag_solid:.2e})"
        )

    def test_schwarzschild_magnitude(self):
        """Schwarzschild acceleration ~2e-8 m/s^2 at LEO (order of magnitude)."""
        force = SchwarzschildForce()
        r = math.sqrt(sum(c * c for c in _LEO_POS))
        v_circ = math.sqrt(OrbitalConstants.MU_EARTH / r)
        r_hat = tuple(c / r for c in _LEO_POS)
        # Velocity perpendicular to radius
        v_dir = (-r_hat[1], r_hat[0], 0.0)
        v_mag = math.sqrt(sum(c * c for c in v_dir))
        vel = tuple(v_circ * c / v_mag for c in v_dir)
        ax, ay, az = force.acceleration(_LEO_EPOCH, _LEO_POS, vel)
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        assert 1e-9 < mag < 1e-7, f"Schwarzschild magnitude {mag:.2e} out of range"

    def test_drag_opposes_velocity(self):
        """Atmospheric drag acceleration opposes relative velocity."""
        config = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0)
        force = AtmosphericDragForce(config)
        r_eq = OrbitalConstants.R_EARTH_EQUATORIAL
        pos = (r_eq + 400_000.0, 0.0, 0.0)  # 400 km altitude, on equator
        v_circ = math.sqrt(OrbitalConstants.MU_EARTH / (r_eq + 400_000.0))
        vel = (0.0, v_circ, 0.0)
        ax, ay, az = force.acceleration(_LEO_EPOCH, pos, vel)
        # Relative velocity accounts for atmosphere co-rotation
        omega_e = OrbitalConstants.EARTH_ROTATION_RATE
        vr_x = vel[0] + omega_e * pos[1]
        vr_y = vel[1] - omega_e * pos[0]
        vr_z = vel[2]
        dot_a_vrel = ax * vr_x + ay * vr_y + az * vr_z
        assert dot_a_vrel < 0, "Drag must oppose relative velocity"


# ═══════════════════════════════════════════════════════════════════
# D. Integrator Coefficients
# ═══════════════════════════════════════════════════════════════════


class TestIntegratorCoefficients:
    """Verify Butcher tableau and symplectic integrator coefficients."""

    def test_dp_fsal_property(self):
        """FSAL: B4 weights (excl. trailing zero) equal last row of A."""
        last_a_row = DORMAND_PRINCE_A[-1]
        b4_trimmed = DORMAND_PRINCE_B4[: len(last_a_row)]
        for i, (a_val, b_val) in enumerate(zip(last_a_row, b4_trimmed)):
            assert a_val == b_val, (
                f"FSAL mismatch at index {i}: A={a_val}, B4={b_val}"
            )

    def test_dp_b4_sum(self):
        """5th-order weights sum to 1."""
        assert abs(sum(DORMAND_PRINCE_B4) - 1.0) < 1e-14

    def test_dp_b5_sum(self):
        """4th-order weights sum to 1."""
        assert abs(sum(DORMAND_PRINCE_B5) - 1.0) < 1e-14

    def test_dp_stage_count(self):
        """Dormand-Prince: 7 stages, 7 nodes, 7 weights."""
        assert len(DORMAND_PRINCE_A) == 7
        assert len(DORMAND_PRINCE_C) == 7
        assert len(DORMAND_PRINCE_B4) == 7
        assert len(DORMAND_PRINCE_B5) == 7

    def test_yoshida_w_sum(self):
        """Yoshida 4th-order: w0 + 2*w1 = 1 (time symmetry)."""
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cbrt2)
        w0 = -cbrt2 / (2.0 - cbrt2)
        assert abs(w0 + 2.0 * w1 - 1.0) < 1e-14

    def test_yoshida_c_sum(self):
        """Yoshida position kicks sum to 1 (full step)."""
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cbrt2)
        w0 = -cbrt2 / (2.0 - cbrt2)
        c_half = (w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0)
        assert abs(sum(c_half) - 1.0) < 1e-14

    def test_yoshida_d_sum(self):
        """Yoshida velocity kicks sum to 1 (full step)."""
        cbrt2 = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - cbrt2)
        w0 = -cbrt2 / (2.0 - cbrt2)
        d = (w1, w0, w1)
        assert abs(sum(d) - 1.0) < 1e-14


# ═══════════════════════════════════════════════════════════════════
# E. Atmosphere Table Spot-Checks
# ═══════════════════════════════════════════════════════════════════


class TestAtmosphereSpotChecks:
    """Spot-check atmospheric density against Vallado Table 8-4."""

    def test_density_200km(self):
        """Vallado Table 8-4: rho(200 km) = 2.789e-10 kg/m^3 (1% tol)."""
        rho = atmospheric_density(200.0, model=AtmosphereModel.VALLADO_4TH)
        assert abs(rho - 2.789e-10) / 2.789e-10 < 0.01

    def test_density_500km(self):
        """Vallado Table 8-4: rho(500 km) = 6.967e-13 kg/m^3 (1% tol)."""
        rho = atmospheric_density(500.0, model=AtmosphereModel.VALLADO_4TH)
        assert abs(rho - 6.967e-13) / 6.967e-13 < 0.01

    def test_monotonic_decrease(self):
        """Density must decrease monotonically with altitude."""
        altitudes = [150.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0]
        for model in (AtmosphereModel.VALLADO_4TH, AtmosphereModel.HIGH_ACTIVITY):
            densities = [atmospheric_density(h, model=model) for h in altitudes]
            for i in range(len(densities) - 1):
                assert densities[i] > densities[i + 1], (
                    f"Non-monotonic at {altitudes[i]}-{altitudes[i + 1]} km "
                    f"({model.value}): {densities[i]:.3e} <= {densities[i + 1]:.3e}"
                )

    def test_scale_heights_positive(self):
        """All scale heights in atmosphere tables must be positive."""
        for table_name, table in [
            ("VALLADO_4TH", _ATMOSPHERE_TABLE_VALLADO),
            ("HIGH_ACTIVITY", _ATMOSPHERE_TABLE_HIGH),
        ]:
            for alt, _rho, H in table:
                assert H > 0, (
                    f"Non-positive scale height {H} at {alt} km ({table_name})"
                )


# ═══════════════════════════════════════════════════════════════════
# F. Derived Results / Golden Values
# ═══════════════════════════════════════════════════════════════════


class TestGoldenValues:
    """Verify derived physical quantities against known values."""

    def test_circular_velocity_500km(self):
        """v_circ at 500 km ~ 7613 m/s (+-10 m/s)."""
        r = OrbitalConstants.R_EARTH_EQUATORIAL + 500_000.0
        v = math.sqrt(OrbitalConstants.MU_EARTH / r)
        assert abs(v - 7613.0) < 10.0, f"v_circ = {v:.1f} m/s"

    def test_sso_inclination_550km(self):
        """SSO inclination at 550 km ~ 97.6 deg (+-0.5 deg)."""
        inc = sso_inclination_deg(550.0)
        assert abs(inc - 97.6) < 0.5, f"SSO inc = {inc:.2f} deg"

    def test_gmst_at_j2000(self):
        """GMST at J2000.0 ~ 280.46 deg (+-0.1 deg)."""
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        gmst = math.degrees(gmst_rad(j2000))
        assert abs(gmst - 280.46) < 0.1, f"GMST = {gmst:.4f} deg"

    def test_geo_altitude(self):
        """GEO altitude ~ 35786 km (+-10 km)."""
        omega = OrbitalConstants.EARTH_ROTATION_RATE
        a_geo = (OrbitalConstants.MU_EARTH / (omega * omega)) ** (1.0 / 3.0)
        alt_km = (a_geo - OrbitalConstants.R_EARTH_EQUATORIAL) / 1000.0
        assert abs(alt_km - 35786.0) < 10.0, f"GEO alt = {alt_km:.1f} km"

    def test_sidereal_day(self):
        """Sidereal day = 2*pi/omega ~ 86164.09 s (+-1 s)."""
        T_sid = 2.0 * math.pi / OrbitalConstants.EARTH_ROTATION_RATE
        assert abs(T_sid - 86164.09) < 1.0, f"T_sid = {T_sid:.2f} s"

    def test_j2_raan_rate_leo(self):
        """J2 RAAN rate at 500 km, 53 deg ~ -5 deg/day (+-0.5 deg/day)."""
        a = OrbitalConstants.R_EARTH_EQUATORIAL + 500_000.0
        n = math.sqrt(OrbitalConstants.MU_EARTH / a ** 3)
        i_rad = math.radians(53.0)
        raan_rate = j2_raan_rate(n, a, 0.0, i_rad)
        raan_rate_deg_day = math.degrees(raan_rate) * 86400.0
        assert abs(raan_rate_deg_day - (-5.0)) < 0.5, (
            f"RAAN rate = {raan_rate_deg_day:.3f} deg/day"
        )

    def test_kepler_cartesian_roundtrip(self):
        """|r| matches a for circular orbit (< 1 m)."""
        a = OrbitalConstants.R_EARTH_EQUATORIAL + 500_000.0
        pos, _vel = kepler_to_cartesian(
            a=a, e=0.0, i_rad=math.radians(45.0),
            omega_big_rad=0.0, omega_small_rad=0.0, nu_rad=0.0,
        )
        r = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        assert abs(r - a) < 1.0, f"|r| - a = {abs(r - a):.6f} m"

    def test_eci_ecef_roundtrip(self):
        """ECI -> ECEF -> ECI position round-trip < 0.001 m."""
        pos_eci = (7_000_000.0, 1_000_000.0, 500_000.0)
        vel_eci = (-1000.0, 7000.0, 500.0)
        epoch = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        gmst = gmst_rad(epoch)

        # Forward: ECI -> ECEF
        pos_ecef, _vel_ecef = eci_to_ecef(pos_eci, vel_eci, gmst)

        # Reverse: ECEF -> ECI via R_z(-gmst)
        cos_g = math.cos(gmst)
        sin_g = math.sin(gmst)
        x_back = cos_g * pos_ecef[0] - sin_g * pos_ecef[1]
        y_back = sin_g * pos_ecef[0] + cos_g * pos_ecef[1]
        z_back = pos_ecef[2]

        dx = x_back - pos_eci[0]
        dy = y_back - pos_eci[1]
        dz = z_back - pos_eci[2]
        err = math.sqrt(dx * dx + dy * dy + dz * dz)
        assert err < 0.001, f"Round-trip error = {err:.6f} m"


# ═══════════════════════════════════════════════════════════════════
# G. Time System Offsets
# ═══════════════════════════════════════════════════════════════════


class TestTimeSystemOffsets:
    """Verify fundamental time system offset constants."""

    def test_tt_equals_tai_plus_32_184(self):
        """TT = TAI + 32.184 s (IAU 1991, exact by definition)."""
        assert _TT_TAI_OFFSET == 32.184

    def test_tai_equals_gps_plus_19(self):
        """TAI = GPS + 19 s (exact by definition)."""
        assert _GPS_TAI_OFFSET == 19.0

    def test_j2000_julian_date(self):
        """J2000.0 epoch = JD 2451545.0."""
        assert _J2000_JD == 2451545.0
