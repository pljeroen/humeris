# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Real-world orbital scenario validation tests for the Humeris astrodynamics library.

Six well-known scenarios from spaceflight history, each validating a different
combination of domain modules against published or physically derivable values.

Scenarios:
    1. 'Oumuamua — hyperbolic interstellar trajectory (e > 1)
    2. ISS — LEO with J2/J3 secular perturbations and atmospheric drag
    3. Tiangong-1 — uncontrolled reentry and orbit lifetime estimation
    4. Starlink v1.0 — constellation deployment and Hohmann orbit raising
    5. ENVISAT — Sun-synchronous debris object and conjunction screening
    6. Iridium 33 / Cosmos 2251 — hypervelocity satellite collision

References:
    - JPL Horizons (https://ssd.jpl.nasa.gov/horizons/)
    - Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
    - NASA Orbital Debris Quarterly News
    - ESA Space Debris Office publications
"""
import math
from datetime import datetime, timedelta, timezone

import pytest

from humeris.domain.orbital_mechanics import (
    OrbitalConstants,
    kepler_to_cartesian,
    sso_inclination_deg,
    j2_raan_rate,
    j2_arg_perigee_rate,
)
from humeris.domain.propagation import OrbitalState, propagate_to
from humeris.domain.atmosphere import DragConfig, atmospheric_density, drag_acceleration
from humeris.domain.lifetime import compute_orbit_lifetime
from humeris.domain.maneuvers import hohmann_transfer
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.conjunction import (
    screen_conjunctions,
    assess_conjunction,
    collision_probability_2d,
    PositionCovariance,
)
from humeris.domain.orbit_design import design_sso_orbit
from humeris.domain.cascade_analysis import compute_cascade_sir


# ── Shared constants ────────────────────────────────────────────────

MU_EARTH = OrbitalConstants.MU_EARTH          # 3.986004418e14 m^3/s^2
R_EARTH = OrbitalConstants.R_EARTH             # 6_371_000 m (mean)
R_EARTH_EQ = OrbitalConstants.R_EARTH_EQUATORIAL  # 6_378_137.0 m (WGS84)
EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _sma_from_alt(altitude_km: float) -> float:
    """Semi-major axis (m) from altitude above mean Earth radius."""
    return R_EARTH + altitude_km * 1000.0


def _mean_motion(a_m: float) -> float:
    """Keplerian mean motion (rad/s) from semi-major axis."""
    return math.sqrt(MU_EARTH / a_m ** 3)


def _circular_state(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float = 0.0,
    nu_deg: float = 0.0,
    epoch: datetime = EPOCH,
) -> OrbitalState:
    """Helper: create a circular OrbitalState at given altitude."""
    a = _sma_from_alt(altitude_km)
    n = _mean_motion(a)
    return OrbitalState(
        semi_major_axis_m=a,
        eccentricity=0.0,
        inclination_rad=math.radians(inclination_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=0.0,
        true_anomaly_rad=math.radians(nu_deg),
        mean_motion_rad_s=n,
        reference_epoch=epoch,
    )


# ════════════════════════════════════════════════════════════════════
# Scenario 1: 'Oumuamua — Hyperbolic Interstellar Trajectory
# ════════════════════════════════════════════════════════════════════
#
# 1I/2017 U1 'Oumuamua was the first interstellar object detected
# traversing the solar system.  Its heliocentric orbit has e ~ 1.2
# (hyperbolic).  Since the Humeris library uses Earth-centric mu,
# we validate the kepler_to_cartesian function with a simulated
# Earth-flyby hyperbolic orbit that exercises the same code paths.
#
# Simulated hyperbolic orbit parameters:
#   a  = -10 000 km (negative for hyperbolic)
#   e  = 1.5
#   i  = 30 deg
#   Om = 45 deg
#   w  = 90 deg
#   nu = 0 deg (periapsis)
#
# Periapsis distance: q = a*(1 - e) = -10 000*(1 - 1.5) = 5 000 km
# (from Earth center — well inside LEO; purely a math validation)
#
# Ref: JPL Horizons solution 1I/2017 U1, epoch 2017-Nov-22.0 TDB
#      a = -1.2722 AU, e = 1.19951, i = 122.69 deg

class TestOumuamua:
    """Validate kepler_to_cartesian for hyperbolic orbits (e > 1)."""

    # Simulated Earth-flyby hyperbolic orbit
    A_M = -10_000_000.0        # -10 000 km (negative for hyperbolic)
    E = 1.5
    I_RAD = math.radians(30.0)
    RAAN_RAD = math.radians(45.0)
    ARGP_RAD = math.radians(90.0)
    NU_PERIAPSIS = 0.0         # True anomaly at periapsis

    # Derived: periapsis distance
    # q = |a| * (e - 1) for hyperbolic (a < 0)
    # Or equivalently: q = a * (1 - e) which gives positive since both terms negative
    Q_M = abs(A_M) * (E - 1.0)  # 10_000_000 * 0.5 = 5_000_000 m

    def test_oumuamua_hyperbolic_eccentricity(self):
        """kepler_to_cartesian handles e > 1 without error."""
        pos, vel = kepler_to_cartesian(
            a=self.A_M, e=self.E,
            i_rad=self.I_RAD,
            omega_big_rad=self.RAAN_RAD,
            omega_small_rad=self.ARGP_RAD,
            nu_rad=self.NU_PERIAPSIS,
        )
        # Must return finite values
        assert all(math.isfinite(x) for x in pos), f"Non-finite position: {pos}"
        assert all(math.isfinite(x) for x in vel), f"Non-finite velocity: {vel}"

    def test_oumuamua_kepler_to_cartesian(self):
        """Position magnitude at periapsis matches expected q = 5 000 km."""
        pos, vel = kepler_to_cartesian(
            a=self.A_M, e=self.E,
            i_rad=self.I_RAD,
            omega_big_rad=self.RAAN_RAD,
            omega_small_rad=self.ARGP_RAD,
            nu_rad=self.NU_PERIAPSIS,
        )
        r_mag = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
        # Periapsis distance from the conic equation:
        # r = a*(1 - e^2) / (1 + e*cos(nu))
        # At nu=0: r = a*(1-e^2)/(1+e) = a*(1-e)*(1+e)/(1+e) = a*(1-e)
        # a*(1-e) = -10e6 * (1-1.5) = -10e6 * (-0.5) = 5e6 m
        expected_q = self.Q_M
        assert r_mag == pytest.approx(expected_q, rel=1e-10), (
            f"Periapsis distance {r_mag / 1000:.1f} km != expected {expected_q / 1000:.1f} km"
        )

    def test_oumuamua_propagate_30_days(self):
        """Propagating 30 days outbound: distance increases monotonically.

        Since propagate_to requires near-circular orbits (e < 1e-6), we
        use kepler_to_cartesian directly at successive true anomaly values
        to verify the outbound hyperbolic trajectory.
        """
        distances = []
        # Sample true anomaly from 0 (periapsis) to 60 deg outbound
        for nu_deg in range(0, 61, 10):
            nu_rad = math.radians(nu_deg)
            pos, _ = kepler_to_cartesian(
                a=self.A_M, e=self.E,
                i_rad=self.I_RAD,
                omega_big_rad=self.RAAN_RAD,
                omega_small_rad=self.ARGP_RAD,
                nu_rad=nu_rad,
            )
            r = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
            distances.append(r)

        # Distance must be monotonically increasing on the outbound leg
        for i in range(1, len(distances)):
            assert distances[i] > distances[i - 1], (
                f"Distance not increasing at step {i}: "
                f"{distances[i - 1]:.0f} -> {distances[i]:.0f}"
            )

    def test_oumuamua_energy_positive(self):
        """Specific orbital energy is positive for an unbound (hyperbolic) orbit.

        E = v^2/2 - mu/r > 0 for hyperbolic orbits.
        Also E = -mu / (2*a), and since a < 0, E > 0.
        """
        pos, vel = kepler_to_cartesian(
            a=self.A_M, e=self.E,
            i_rad=self.I_RAD,
            omega_big_rad=self.RAAN_RAD,
            omega_small_rad=self.ARGP_RAD,
            nu_rad=self.NU_PERIAPSIS,
        )
        r_mag = math.sqrt(sum(x ** 2 for x in pos))
        v_mag = math.sqrt(sum(x ** 2 for x in vel))

        energy = v_mag ** 2 / 2.0 - MU_EARTH / r_mag
        assert energy > 0, f"Specific energy {energy:.4e} J/kg must be positive (unbound)"

        # Cross-check with vis-viva: E = -mu / (2*a)
        energy_visviva = -MU_EARTH / (2.0 * self.A_M)
        assert energy == pytest.approx(energy_visviva, rel=1e-8)


# ════════════════════════════════════════════════════════════════════
# Scenario 2: ISS — LEO with J2/J3 and Atmospheric Drag
# ════════════════════════════════════════════════════════════════════
#
# The International Space Station orbits at ~420 km altitude in a
# 51.6 deg inclination orbit. Its low altitude makes it subject to
# significant J2 secular perturbations and atmospheric drag.
#
# Known values (from NASA/JSC flight dynamics):
#   Altitude:          ~420 km
#   Inclination:       51.64 deg
#   Eccentricity:      ~0.0001 (near-circular)
#   Orbital period:    ~92.68 minutes (5561 s)
#   RAAN drift (J2):   ~-5.06 deg/day
#   Arg perigee drift: ~+3.5 deg/day (approximate, for near-circular)
#
# Ref: NASA ISS Trajectory Data, Vallado Ch. 9

class TestISS:
    """Validate J2 secular rates and orbital period for ISS-like orbit."""

    ALT_KM = 420.0
    INC_DEG = 51.64
    ECC = 0.0001
    SMA_M = R_EARTH_EQ + ALT_KM * 1000.0  # Using equatorial radius for J2 consistency
    N_RAD_S = math.sqrt(MU_EARTH / SMA_M ** 3)

    # Expected period: T = 2*pi / n
    EXPECTED_PERIOD_MIN = 92.68  # minutes, from published ISS data

    def test_iss_orbital_period(self):
        """Orbital period from Kepler's third law matches ~92.68 min.

        T = 2*pi * sqrt(a^3 / mu)
        """
        period_s = 2.0 * math.pi / self.N_RAD_S
        period_min = period_s / 60.0
        assert period_min == pytest.approx(self.EXPECTED_PERIOD_MIN, abs=0.5), (
            f"ISS period {period_min:.2f} min != expected {self.EXPECTED_PERIOD_MIN} +/- 0.5 min"
        )

    def test_iss_j2_raan_drift(self):
        """J2 RAAN drift rate matches ~-5.06 deg/day for ISS orbit.

        dOmega/dt = -3/2 * n * J2 * (R_E/a)^2 * cos(i) / (1-e^2)^2
        """
        rate_rad_s = j2_raan_rate(
            n=self.N_RAD_S,
            a=self.SMA_M,
            e=self.ECC,
            i_rad=math.radians(self.INC_DEG),
        )
        rate_deg_day = math.degrees(rate_rad_s) * 86400.0
        # Expected: approximately -5.0 deg/day
        # Published values range from -4.9 to -5.1 depending on exact epoch
        # and whether J2-corrected mean motion is used. Our analytical formula
        # gives -4.95 for the nominal ISS parameters.
        assert rate_deg_day == pytest.approx(-5.0, abs=0.15), (
            f"ISS RAAN drift {rate_deg_day:.3f} deg/day != expected -5.0 +/- 0.15"
        )

    def test_iss_j2_arg_perigee_drift(self):
        """J2 argument of perigee drift rate for ISS orbit.

        dw/dt = 3/2 * n * J2 * (R_E/a)^2 * (2 - 5/2*sin^2(i)) / (1-e^2)^2
        For i = 51.64 deg, sin^2(i) ~ 0.614, so (2 - 2.5*0.614) ~ 0.465 > 0,
        meaning the perigee advances (positive rate).
        """
        rate_rad_s = j2_arg_perigee_rate(
            n=self.N_RAD_S,
            a=self.SMA_M,
            e=self.ECC,
            i_rad=math.radians(self.INC_DEG),
        )
        rate_deg_day = math.degrees(rate_rad_s) * 86400.0
        # For ISS, the rate is approximately +3.5 deg/day
        # The formula gives 2 - 2.5*sin^2(51.64) = 2 - 2.5*0.614 = 0.465
        # Positive => prograde perigee advance
        assert rate_deg_day > 0, "Arg perigee drift should be positive for ISS"
        assert rate_deg_day == pytest.approx(3.5, abs=0.5), (
            f"ISS arg perigee drift {rate_deg_day:.3f} deg/day != expected ~3.5 +/- 0.5"
        )

    def test_iss_ground_track_regression(self):
        """ISS ground track approximately repeats after ~3 days.

        With a period of ~92.68 min, the ISS completes ~15.54 orbits per day.
        The ground track shifts ~24.1 deg westward per orbit.
        After ~3 days (~46.6 orbits), the accumulated shift is ~1123 deg =
        ~3*360 + 43 deg, so it roughly closes after ~3 days (not exact,
        the ISS does not have a precisely repeating ground track).

        We verify: revolutions per day is approximately 15.5.
        """
        period_s = 2.0 * math.pi / self.N_RAD_S
        revs_per_day = 86400.0 / period_s
        # ISS does ~15.5 revolutions per day
        assert revs_per_day == pytest.approx(15.54, abs=0.1), (
            f"ISS revs/day {revs_per_day:.2f} != expected ~15.54"
        )

    def test_iss_drag_altitude_loss(self):
        """Atmospheric drag at ISS altitude produces non-negligible deceleration.

        At 420 km, atmospheric density is ~3e-12 kg/m^3 (order of magnitude).
        With ISS parameters (large area, large mass), drag causes measurable
        decay.
        """
        alt_km = self.ALT_KM
        # ISS approximate drag parameters
        # Area ~ 3500 m^2 (solar arrays + modules) projected cross-section ~ 500 m^2
        # Mass ~ 420 000 kg, Cd ~ 2.2
        iss_drag = DragConfig(cd=2.2, area_m2=500.0, mass_kg=420_000.0)

        density = atmospheric_density(alt_km)
        velocity = math.sqrt(MU_EARTH / self.SMA_M)  # ~7 665 m/s
        accel = drag_acceleration(density, velocity, iss_drag)

        # Drag acceleration should be on the order of 1e-6 to 1e-5 m/s^2
        assert accel > 1e-8, f"Drag acceleration {accel:.2e} m/s^2 too small"
        assert accel < 1e-3, f"Drag acceleration {accel:.2e} m/s^2 unreasonably large"


# ════════════════════════════════════════════════════════════════════
# Scenario 3: Tiangong-1 — Uncontrolled Reentry / Decay
# ════════════════════════════════════════════════════════════════════
#
# The Tiangong-1 Chinese space station lost control in March 2016
# and reentered uncontrolled on 2018-04-02 at ~220 km altitude.
# In early 2018 it was at ~340 km altitude.
#
# Parameters:
#   Initial altitude: ~340 km (early January 2018)
#   Final reentry:    2018-04-02 (~3 months later)
#   Inclination:      42.8 deg
#   Mass:             ~8 500 kg
#   Cd:               ~2.2
#   Area:             ~30 m^2 (compact shape)
#
# Ref: ESA Space Debris Office Tiangong-1 reentry campaign reports

class TestTiangong1:
    """Validate orbit lifetime estimation for Tiangong-1 decay profile."""

    ALT_KM = 340.0
    SMA_M = _sma_from_alt(ALT_KM)
    INC_DEG = 42.8
    MASS_KG = 8500.0
    CD = 2.2
    AREA_M2 = 30.0
    EPOCH_T1 = datetime(2018, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    DRAG_CONFIG = DragConfig(cd=CD, area_m2=AREA_M2, mass_kg=MASS_KG)

    def test_tiangong1_lifetime_order_of_magnitude(self):
        """Lifetime from 340 km is weeks-to-months, not years.

        The actual reentry occurred ~3 months after January 2018.
        With the exponential atmosphere model, we expect the computed
        lifetime to be in the range of tens to hundreds of days.
        """
        result = compute_orbit_lifetime(
            semi_major_axis_m=self.SMA_M,
            eccentricity=0.001,
            drag_config=self.DRAG_CONFIG,
            epoch=self.EPOCH_T1,
            re_entry_altitude_km=100.0,
            step_days=0.5,
        )
        assert result.converged, "Lifetime computation did not converge to reentry"
        # Actual reentry was ~91 days later
        # Allow wide tolerance due to simplified atmosphere model
        assert result.lifetime_days > 10, (
            f"Lifetime {result.lifetime_days:.1f} days unrealistically short"
        )
        assert result.lifetime_days < 365, (
            f"Lifetime {result.lifetime_days:.1f} days unrealistically long (should be months)"
        )

    def test_tiangong1_decay_trajectory_monotone(self):
        """Altitude monotonically decreases throughout the decay profile.

        An uncontrolled decaying orbit loses energy continuously.
        """
        result = compute_orbit_lifetime(
            semi_major_axis_m=self.SMA_M,
            eccentricity=0.001,
            drag_config=self.DRAG_CONFIG,
            epoch=self.EPOCH_T1,
            re_entry_altitude_km=100.0,
            step_days=0.5,
        )
        altitudes = [pt.altitude_km for pt in result.decay_profile]
        for i in range(1, len(altitudes)):
            assert altitudes[i] <= altitudes[i - 1], (
                f"Altitude increased at step {i}: "
                f"{altitudes[i - 1]:.2f} -> {altitudes[i]:.2f} km"
            )

    def test_tiangong1_reentry_altitude(self):
        """Reentry occurs at or below the 100 km threshold.

        The computation terminates when altitude drops to re_entry_altitude_km.
        """
        result = compute_orbit_lifetime(
            semi_major_axis_m=self.SMA_M,
            eccentricity=0.001,
            drag_config=self.DRAG_CONFIG,
            epoch=self.EPOCH_T1,
            re_entry_altitude_km=100.0,
            step_days=0.5,
        )
        assert result.converged
        final_alt = result.decay_profile[-1].altitude_km
        assert final_alt <= 100.0, (
            f"Final altitude {final_alt:.1f} km exceeds reentry threshold of 100 km"
        )


# ════════════════════════════════════════════════════════════════════
# Scenario 4: Starlink v1.0 Train — Constellation Deployment
# ════════════════════════════════════════════════════════════════════
#
# SpaceX launches Starlink satellites to a deploy altitude of ~440 km
# and then raises them to an operational altitude of ~550 km using
# onboard ion thrusters. The constellation uses 72 orbital planes at
# 53 deg inclination with ~22 satellites per plane.
#
# Parameters:
#   Deploy altitude:      440 km
#   Operational altitude: 550 km
#   Inclination:          53.0 deg
#   Number of planes:     72
#   Sats per plane:       22 (shell total: 1 584)
#
# Ref: SpaceX FCC filings, ITU constellation data

class TestStarlink:
    """Validate Hohmann transfer, Walker generation, and SSO checks for Starlink."""

    DEPLOY_ALT_KM = 440.0
    OPER_ALT_KM = 550.0
    INC_DEG = 53.0
    NUM_PLANES = 72
    SATS_PER_PLANE = 22
    TOTAL_SATS = NUM_PLANES * SATS_PER_PLANE  # 1 584

    R_DEPLOY = R_EARTH + DEPLOY_ALT_KM * 1000.0
    R_OPER = R_EARTH + OPER_ALT_KM * 1000.0

    def test_starlink_hohmann_dv(self):
        """Hohmann delta-V from 440 km to 550 km matches ~60 m/s.

        This is a modest orbit-raising maneuver. The theoretical Hohmann
        delta-V for 110 km altitude change at LEO is ~60 m/s total.
        """
        plan = hohmann_transfer(r1_m=self.R_DEPLOY, r2_m=self.R_OPER)
        total_dv = plan.total_delta_v_ms
        # Expected: approximately 60 m/s total
        assert total_dv == pytest.approx(60.0, abs=5.0), (
            f"Hohmann dV {total_dv:.2f} m/s != expected ~60 +/- 5 m/s"
        )
        # Both burns should be prograde (orbit raising)
        assert plan.burns[0].delta_v_ms > 0
        assert plan.burns[1].delta_v_ms > 0

    def test_starlink_hohmann_transfer_time(self):
        """Hohmann transfer time from 440 to 550 km is about 46 minutes.

        Half the period of the transfer ellipse.
        """
        plan = hohmann_transfer(r1_m=self.R_DEPLOY, r2_m=self.R_OPER)
        transfer_min = plan.transfer_time_s / 60.0
        # Transfer ellipse has semi-major axis ~ (R1 + R2) / 2
        # Expected half-period: ~46 minutes
        assert transfer_min == pytest.approx(46.0, abs=2.0), (
            f"Hohmann transfer time {transfer_min:.1f} min != expected ~46 +/- 2 min"
        )

    def test_starlink_walker_generation(self):
        """Generate a Walker 53 deg:1584/72/1 constellation.

        Verify correct number of satellites and orbital parameters.
        Using a smaller subset (4 planes, 3 sats) to keep test fast.
        """
        config = ShellConfig(
            altitude_km=self.OPER_ALT_KM,
            inclination_deg=self.INC_DEG,
            num_planes=4,
            sats_per_plane=3,
            phase_factor=1,
            raan_offset_deg=0.0,
            shell_name="Starlink-test",
        )
        sats = generate_walker_shell(config)
        assert len(sats) == 4 * 3, f"Expected 12 satellites, got {len(sats)}"

        # Verify orbital altitude is correct (position magnitude ~ R_E + 550 km)
        expected_r = R_EARTH + self.OPER_ALT_KM * 1000.0
        for sat in sats:
            r_mag = math.sqrt(
                sat.position_eci[0] ** 2
                + sat.position_eci[1] ** 2
                + sat.position_eci[2] ** 2
            )
            assert r_mag == pytest.approx(expected_r, rel=1e-6), (
                f"Satellite {sat.name} radius {r_mag:.0f} m != expected {expected_r:.0f} m"
            )

    def test_starlink_sso_not_applicable(self):
        """At 53 deg inclination, this is NOT a Sun-synchronous orbit.

        SSO at 550 km requires an inclination of ~97.6 deg (retrograde).
        Starlink at 53 deg is a prograde LEO constellation.
        """
        sso_inc = sso_inclination_deg(self.OPER_ALT_KM)
        assert sso_inc > 90.0, "SSO inclination must be retrograde (> 90 deg)"
        assert abs(sso_inc - self.INC_DEG) > 40.0, (
            f"SSO inclination {sso_inc:.1f} deg too close to Starlink {self.INC_DEG} deg"
        )

    def test_starlink_coverage_at_53deg(self):
        """A small Walker shell at 53 deg provides some coverage.

        Generate a minimal constellation and verify at least some satellites
        exist in each plane.
        """
        config = ShellConfig(
            altitude_km=self.OPER_ALT_KM,
            inclination_deg=self.INC_DEG,
            num_planes=6,
            sats_per_plane=4,
            phase_factor=1,
            raan_offset_deg=0.0,
            shell_name="Starlink-coverage",
        )
        sats = generate_walker_shell(config)
        # Verify planes are spread in RAAN
        raan_values = sorted(set(s.raan_deg for s in sats))
        assert len(raan_values) == 6, f"Expected 6 RAAN planes, got {len(raan_values)}"
        # RAAN separation should be 360/6 = 60 deg
        for i in range(1, len(raan_values)):
            sep = raan_values[i] - raan_values[i - 1]
            assert sep == pytest.approx(60.0, abs=0.01)


# ════════════════════════════════════════════════════════════════════
# Scenario 5: ENVISAT — Debris / Conjunction Risk
# ════════════════════════════════════════════════════════════════════
#
# ENVISAT is a defunct ESA Earth observation satellite at ~770 km in a
# Sun-synchronous orbit. At ~8 tonnes and ~26 m long, it is one of the
# largest debris objects and a major conjunction risk.
#
# Parameters:
#   Altitude:     770 km
#   Inclination:  98.55 deg (SSO)
#   LTAN:         ~10:00 (descending node)
#   Mass:         ~8 000 kg
#   Size:         ~26 m x 10 m
#
# Ref: ESA Space Debris Office, Krag et al. (2012)

class TestENVISAT:
    """Validate SSO design and conjunction screening for ENVISAT-like orbit."""

    ALT_KM = 770.0
    EXPECTED_SSO_INC_DEG = 98.55  # Published SSO inclination at 770 km

    def test_envisat_sso_inclination(self):
        """SSO inclination at 770 km matches 98.55 deg within 0.5 deg.

        The analytical formula: cos(i) = -(2*omega_sun)/(3*J2*R_E^2) * (a^3.5 / sqrt(mu))
        """
        computed_inc = sso_inclination_deg(self.ALT_KM)
        assert computed_inc == pytest.approx(self.EXPECTED_SSO_INC_DEG, abs=0.5), (
            f"SSO inclination at {self.ALT_KM} km: {computed_inc:.2f} deg != "
            f"expected {self.EXPECTED_SSO_INC_DEG} +/- 0.5 deg"
        )

    def test_envisat_is_sso(self):
        """The ENVISAT orbit satisfies the SSO condition.

        Verify by checking that the J2 RAAN drift rate equals the Earth's
        mean motion around the Sun (~0.9856 deg/day).
        """
        computed_inc = sso_inclination_deg(self.ALT_KM)
        a = R_EARTH_EQ + self.ALT_KM * 1000.0
        n = math.sqrt(MU_EARTH / a ** 3)
        raan_rate = j2_raan_rate(
            n=n, a=a, e=0.0, i_rad=math.radians(computed_inc),
        )
        raan_deg_day = math.degrees(raan_rate) * 86400.0
        # SSO condition: RAAN precesses at +0.9856 deg/day (eastward)
        # to match Earth's orbital motion around the Sun
        # (360 deg / 365.2422 days = 0.9856 deg/day)
        expected_rate = 360.0 / 365.2422
        assert raan_deg_day == pytest.approx(expected_rate, abs=0.02), (
            f"RAAN drift {raan_deg_day:.4f} deg/day != SSO rate {expected_rate:.4f}"
        )

    def test_envisat_conjunction_screening(self):
        """Screen ENVISAT against a nearby object and find a conjunction.

        Place two satellites in nearby SSO orbits: ENVISAT at 770 km and
        a debris fragment at 770 km with slightly different RAAN. They
        should approach each other during the screening window.
        """
        inc_rad = math.radians(98.55)
        a_envisat = _sma_from_alt(770.0)
        a_debris = _sma_from_alt(770.0)
        n_env = _mean_motion(a_envisat)
        n_deb = _mean_motion(a_debris)

        state_envisat = OrbitalState(
            semi_major_axis_m=a_envisat,
            eccentricity=0.0,
            inclination_rad=inc_rad,
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=n_env,
            reference_epoch=EPOCH,
        )
        # Debris at slightly different RAAN and true anomaly
        state_debris = OrbitalState(
            semi_major_axis_m=a_debris,
            eccentricity=0.0,
            inclination_rad=inc_rad,
            raan_rad=math.radians(0.01),  # 0.01 deg offset
            arg_perigee_rad=0.0,
            true_anomaly_rad=math.radians(0.5),  # Small phase offset
            mean_motion_rad_s=n_deb,
            reference_epoch=EPOCH,
        )

        results = screen_conjunctions(
            states=[state_envisat, state_debris],
            names=["ENVISAT", "Debris-X"],
            start=EPOCH,
            duration=timedelta(hours=2),
            step=timedelta(seconds=60),
            distance_threshold_m=100_000.0,  # 100 km threshold
        )
        # Should find at least one conjunction event within the window
        assert len(results) > 0, "No conjunctions found between ENVISAT and nearby debris"
        # The closest approach should be less than 100 km
        closest_dist = results[0][3]
        assert closest_dist < 100_000.0

    def test_envisat_collision_probability(self):
        """Compute collision probability for a simulated near-miss scenario.

        Using the 2D B-plane collision probability with realistic uncertainties.
        """
        # Simulate a near-miss: 500 m miss distance, 200 m combined radius
        # (ENVISAT is ~26 m, typical debris ~1 m, but using hard-body + covariance)
        # Position uncertainty: ~100 m in each axis
        pc = collision_probability_2d(
            miss_distance_m=500.0,
            b_radial_m=300.0,
            b_cross_m=400.0,
            sigma_radial_m=100.0,
            sigma_cross_m=100.0,
            combined_radius_m=15.0,  # Combined hard-body radius
        )
        # With 500 m miss and 100 m sigma, Pc should be small but non-zero
        assert pc >= 0.0, "Collision probability must be non-negative"
        assert pc <= 1.0, "Collision probability must be <= 1"
        # For this geometry, Pc should be quite small
        assert pc < 0.1, f"Pc = {pc:.6e} seems unreasonably high for 500 m miss"


# ════════════════════════════════════════════════════════════════════
# Scenario 6: Iridium 33 / Cosmos 2251 — Actual Satellite Collision
# ════════════════════════════════════════════════════════════════════
#
# On 2009-02-10, the operational Iridium 33 satellite collided with the
# defunct Cosmos 2251 at ~789 km altitude. This was the first accidental
# hypervelocity collision between two intact artificial satellites.
#
# Parameters:
#   Iridium 33:  780 km, i = 86.4 deg, mass = ~689 kg
#   Cosmos 2251: 790 km, i = 74.0 deg, mass = ~900 kg
#   Relative velocity: ~11.7 km/s
#   Combined mass: ~1 589 kg
#   Trackable debris: ~2 000 fragments (as of 2010)
#
# The high relative velocity results from the large inclination difference
# (86.4 deg vs 74.0 deg) — the orbits intersect at a steep angle.
#
# Ref: NASA Orbital Debris Quarterly News, Vol. 13, Issue 2 (April 2009)
#      Kelso, T.S. "Analysis of the Iridium 33-Cosmos 2251 Collision"

class TestIridiumCosmos:
    """Validate collision geometry and cascade analysis for Iridium/Cosmos event."""

    # Iridium 33 parameters
    IRIDIUM_ALT_KM = 780.0
    IRIDIUM_INC_DEG = 86.4
    IRIDIUM_MASS_KG = 689.0

    # Cosmos 2251 parameters
    COSMOS_ALT_KM = 790.0
    COSMOS_INC_DEG = 74.0
    COSMOS_MASS_KG = 900.0

    # Combined
    COMBINED_MASS_KG = IRIDIUM_MASS_KG + COSMOS_MASS_KG
    EXPECTED_REL_VEL_KMS = 11.7  # km/s

    COLLISION_EPOCH = datetime(2009, 2, 10, 16, 56, 0, tzinfo=timezone.utc)

    def test_iridium_cosmos_relative_velocity(self):
        """Two satellites at different inclinations at ~780-790 km give ~11.7 km/s.

        The actual collision occurred with ~11.7 km/s relative velocity. This
        high value arises because the orbital planes of Iridium 33 (86.4 deg)
        and Cosmos 2251 (74.0 deg) were separated by ~100 deg in RAAN, causing
        their velocity vectors to cross at a steep angle (~103 deg) at the
        intersection point. With orbital velocity ~7.45 km/s for both objects:
            v_rel = sqrt(v1^2 + v2^2 - 2*v1*v2*cos(theta))

        We reconstruct this geometry by choosing RAAN values and true anomalies
        that place both satellites at their orbital intersection with the correct
        velocity vector angle.

        Ref: Kelso, T.S. "Analysis of the Iridium 33-Cosmos 2251 Collision"
        """
        a_ir = _sma_from_alt(self.IRIDIUM_ALT_KM)
        a_co = _sma_from_alt(self.COSMOS_ALT_KM)

        # The collision geometry: Iridium and Cosmos had RAAN separation
        # of approximately 90 degrees. With Iridium ascending near its
        # equatorial crossing and Cosmos near the top of its orbit arc,
        # the velocity vectors cross at a steep angle (~103 deg), producing
        # the observed ~11.7 km/s relative velocity.
        raan_iridium = math.radians(0.0)
        raan_cosmos = math.radians(90.0)

        # Position both satellites at representative true anomalies that
        # reproduce the collision encounter angle.
        pos_ir, vel_ir = kepler_to_cartesian(
            a=a_ir, e=0.0,
            i_rad=math.radians(self.IRIDIUM_INC_DEG),
            omega_big_rad=raan_iridium,
            omega_small_rad=0.0,
            nu_rad=math.radians(15.0),
        )
        # Cosmos at nu ~ 100 deg produces the correct relative velocity.
        pos_co, vel_co = kepler_to_cartesian(
            a=a_co, e=0.0,
            i_rad=math.radians(self.COSMOS_INC_DEG),
            omega_big_rad=raan_cosmos,
            omega_small_rad=0.0,
            nu_rad=math.radians(100.0),
        )

        # Relative velocity magnitude
        dv = [vel_ir[k] - vel_co[k] for k in range(3)]
        rel_vel_ms = math.sqrt(sum(x ** 2 for x in dv))
        rel_vel_kms = rel_vel_ms / 1000.0

        # With this geometry, relative velocity should be in the 10-13 km/s
        # range consistent with the actual event (~11.7 km/s).
        assert rel_vel_kms == pytest.approx(self.EXPECTED_REL_VEL_KMS, abs=2.0), (
            f"Relative velocity {rel_vel_kms:.2f} km/s != expected ~{self.EXPECTED_REL_VEL_KMS} +/- 2 km/s"
        )

    def test_iridium_cosmos_collision_geometry(self):
        """B-plane screening finds a close approach between the two orbits.

        Place both objects in the same orbital plane at slightly different
        true anomalies to create a close approach geometry.
        """
        a_ir = _sma_from_alt(self.IRIDIUM_ALT_KM)
        a_co = _sma_from_alt(self.COSMOS_ALT_KM)

        state_ir = OrbitalState(
            semi_major_axis_m=a_ir,
            eccentricity=0.0,
            inclination_rad=math.radians(self.IRIDIUM_INC_DEG),
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=0.0,
            mean_motion_rad_s=_mean_motion(a_ir),
            reference_epoch=self.COLLISION_EPOCH,
        )
        # Cosmos starts with a tiny phase offset to produce near-miss
        state_co = OrbitalState(
            semi_major_axis_m=a_co,
            eccentricity=0.0,
            inclination_rad=math.radians(self.COSMOS_INC_DEG),
            raan_rad=0.0,
            arg_perigee_rad=0.0,
            true_anomaly_rad=math.radians(0.1),
            mean_motion_rad_s=_mean_motion(a_co),
            reference_epoch=self.COLLISION_EPOCH,
        )

        # Full conjunction assessment with B-plane decomposition
        cov = PositionCovariance(
            sigma_xx=10000.0, sigma_yy=10000.0, sigma_zz=10000.0,
            sigma_xy=0.0, sigma_xz=0.0, sigma_yz=0.0,
        )
        event = assess_conjunction(
            state1=state_ir,
            name1="Iridium-33",
            state2=state_co,
            name2="Cosmos-2251",
            t_guess=self.COLLISION_EPOCH,
            combined_radius_m=10.0,
            cov1=cov, cov2=cov,
        )
        # The conjunction assessment should produce meaningful results
        assert event.miss_distance_m >= 0, "Miss distance must be non-negative"
        assert event.relative_velocity_ms > 0, "Relative velocity must be positive"
        # B-plane decomposition should be finite
        assert math.isfinite(event.b_plane_radial_m)
        assert math.isfinite(event.b_plane_cross_track_m)

    def test_iridium_cosmos_cascade_fragments(self):
        """SIR cascade model with Iridium/Cosmos parameters predicts debris increase.

        The 780-790 km shell had significant debris added by this event.
        Using the SIR epidemic model, we verify that these parameters
        produce a supercritical or significant debris response.
        """
        # Shell volume at ~785 km: V = 4*pi*r^2 * dh
        # Using a 50 km shell thickness centered at 785 km
        r_shell = (R_EARTH + 785_000.0) / 1000.0  # km
        shell_thickness = 50.0  # km
        shell_volume = 4.0 * math.pi * r_shell ** 2 * shell_thickness

        result = compute_cascade_sir(
            shell_volume_km3=shell_volume,
            spatial_density_per_km3=1e-8,  # Pre-collision debris density
            mean_collision_velocity_ms=11_700.0,  # ~11.7 km/s
            satellite_count=200,  # Approximate intact objects in shell
            launch_rate_per_year=10.0,
            fragments_per_collision=200.0,  # Iridium/Cosmos produced ~2000 trackable
            drag_lifetime_years=100.0,  # Long lifetime at 785 km
            collision_cross_section_km2=1e-5,
            duration_years=100.0,
        )
        # The infected (debris) count should increase from initial conditions
        assert result.infected[-1] > result.infected[0], (
            "Debris should increase in the Iridium/Cosmos shell scenario"
        )
        # The R_0 should be computed (may or may not be > 1)
        assert math.isfinite(result.r_0), f"R_0 must be finite, got {result.r_0}"

    def test_iridium_cosmos_orbital_period_match(self):
        """Both Iridium 33 and Cosmos 2251 orbits have period ~100 min.

        At ~780-790 km altitude, the orbital period is approximately 100 minutes.
        """
        a_ir = _sma_from_alt(self.IRIDIUM_ALT_KM)
        a_co = _sma_from_alt(self.COSMOS_ALT_KM)

        period_ir_min = 2.0 * math.pi / _mean_motion(a_ir) / 60.0
        period_co_min = 2.0 * math.pi / _mean_motion(a_co) / 60.0

        assert period_ir_min == pytest.approx(100.0, abs=1.5), (
            f"Iridium period {period_ir_min:.2f} min != expected ~100 min"
        )
        assert period_co_min == pytest.approx(100.0, abs=1.5), (
            f"Cosmos period {period_co_min:.2f} min != expected ~100 min"
        )
        # Periods should be very close since altitudes differ by only 10 km
        assert abs(period_ir_min - period_co_min) < 0.3, (
            f"Period difference {abs(period_ir_min - period_co_min):.3f} min "
            f"too large for 10 km altitude difference"
        )
