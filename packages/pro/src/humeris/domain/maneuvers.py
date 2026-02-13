# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Orbital transfer maneuver planning.

Hohmann, bi-elliptic, plane change, combined, and phasing maneuvers
with propellant estimation via Tsiolkovsky equation.

"""
import math
from dataclasses import dataclass

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants

_MU = OrbitalConstants.MU_EARTH
_G0 = 9.80665  # standard gravity m/s²


@dataclass(frozen=True)
class ManeuverBurn:
    """Single impulsive burn in a transfer sequence."""
    delta_v_ms: float      # delta-V magnitude (m/s)
    description: str       # e.g. "Hohmann burn 1 (periapsis)"


@dataclass(frozen=True)
class TransferPlan:
    """Complete orbital transfer maneuver plan."""
    burns: tuple[ManeuverBurn, ...]
    total_delta_v_ms: float
    transfer_time_s: float
    propellant_mass_kg: float | None = None


def hohmann_transfer(r1_m: float, r2_m: float) -> TransferPlan:
    """Two-impulse Hohmann transfer between circular orbits.

    Ref: Vallado Ch. 6, Bate/Mueller/White Ch. 6.

    Args:
        r1_m: Initial circular orbit radius (m).
        r2_m: Final circular orbit radius (m).

    Returns:
        TransferPlan with 2 burns.

    Raises:
        ValueError: If r1 or r2 <= 0.
    """
    if r1_m <= 0:
        raise ValueError(f"r1_m must be positive, got {r1_m}")
    if r2_m <= 0:
        raise ValueError(f"r2_m must be positive, got {r2_m}")

    if r1_m == r2_m:
        return TransferPlan(
            burns=(
                ManeuverBurn(delta_v_ms=0.0, description="Hohmann burn 1 (no change)"),
                ManeuverBurn(delta_v_ms=0.0, description="Hohmann burn 2 (no change)"),
            ),
            total_delta_v_ms=0.0,
            transfer_time_s=0.0,
        )

    raising = r2_m > r1_m
    direction = "prograde" if raising else "retrograde"

    v1 = float(np.sqrt(_MU / r1_m))
    v2 = float(np.sqrt(_MU / r2_m))

    dv1 = abs(v1 * (float(np.sqrt(2.0 * r2_m / (r1_m + r2_m))) - 1.0))
    dv2 = abs(v2 * (1.0 - float(np.sqrt(2.0 * r1_m / (r1_m + r2_m)))))

    t_transfer = np.pi * float(np.sqrt((r1_m + r2_m) ** 3 / (8.0 * _MU)))

    return TransferPlan(
        burns=(
            ManeuverBurn(
                delta_v_ms=dv1,
                description=f"Hohmann burn 1 (departure, {direction})",
            ),
            ManeuverBurn(
                delta_v_ms=dv2,
                description=f"Hohmann burn 2 (arrival, {direction})",
            ),
        ),
        total_delta_v_ms=dv1 + dv2,
        transfer_time_s=t_transfer,
    )


def bielliptic_transfer(
    r1_m: float, r2_m: float, r_intermediate_m: float,
) -> TransferPlan:
    """Three-impulse bi-elliptic transfer via intermediate apogee.

    More efficient than Hohmann when r2/r1 > 11.94.

    Args:
        r1_m: Initial circular orbit radius (m).
        r2_m: Final circular orbit radius (m).
        r_intermediate_m: Intermediate apogee radius (m).

    Returns:
        TransferPlan with 3 burns.

    Raises:
        ValueError: If radii invalid or intermediate < max(r1, r2).
    """
    if r1_m <= 0 or r2_m <= 0 or r_intermediate_m <= 0:
        raise ValueError("All radii must be positive")
    if r_intermediate_m <= max(r1_m, r2_m):
        raise ValueError(
            f"Intermediate radius {r_intermediate_m} must exceed "
            f"max(r1, r2) = {max(r1_m, r2_m)}"
        )

    v1 = float(np.sqrt(_MU / r1_m))
    # Burn 1: raise apogee from r1 to r_intermediate
    v_transfer1_peri = float(np.sqrt(2.0 * _MU * r_intermediate_m / (r1_m * (r1_m + r_intermediate_m))))
    dv1 = abs(v_transfer1_peri - v1)

    # Burn 2: at r_intermediate, change orbit from (r1, r_int) to (r2, r_int)
    v_transfer1_apo = float(np.sqrt(2.0 * _MU * r1_m / (r_intermediate_m * (r1_m + r_intermediate_m))))
    v_transfer2_apo = float(np.sqrt(2.0 * _MU * r2_m / (r_intermediate_m * (r2_m + r_intermediate_m))))
    dv2 = abs(v_transfer2_apo - v_transfer1_apo)

    # Burn 3: circularize at r2
    v_transfer2_peri = float(np.sqrt(2.0 * _MU * r_intermediate_m / (r2_m * (r2_m + r_intermediate_m))))
    v2 = float(np.sqrt(_MU / r2_m))
    dv3 = abs(v2 - v_transfer2_peri)

    # Transfer time: half-period of first transfer + half-period of second
    t1 = np.pi * float(np.sqrt((r1_m + r_intermediate_m) ** 3 / (8.0 * _MU)))
    t2 = np.pi * float(np.sqrt((r2_m + r_intermediate_m) ** 3 / (8.0 * _MU)))

    return TransferPlan(
        burns=(
            ManeuverBurn(delta_v_ms=dv1, description="Bi-elliptic burn 1 (raise apogee)"),
            ManeuverBurn(delta_v_ms=dv2, description="Bi-elliptic burn 2 (adjust at apogee)"),
            ManeuverBurn(delta_v_ms=dv3, description="Bi-elliptic burn 3 (circularize)"),
        ),
        total_delta_v_ms=dv1 + dv2 + dv3,
        transfer_time_s=t1 + t2,
    )


def plane_change_dv(velocity_ms: float, inclination_change_rad: float) -> float:
    """Delta-V for pure inclination change.

    dV = 2 * v * sin(di/2)

    Args:
        velocity_ms: Orbital velocity magnitude (m/s).
        inclination_change_rad: Inclination change (radians).

    Returns:
        Delta-V in m/s.
    """
    return 2.0 * velocity_ms * abs(float(np.sin(inclination_change_rad / 2.0)))


def combined_plane_and_altitude(
    r1_m: float, r2_m: float, inclination_change_rad: float,
) -> TransferPlan:
    """Combined altitude change + plane change at apogee.

    Uses law of cosines for optimal combined burn at apogee.
    dV_combined = sqrt(v1^2 + v2^2 - 2*v1*v2*cos(di))

    Args:
        r1_m: Initial circular orbit radius (m).
        r2_m: Final circular orbit radius (m).
        inclination_change_rad: Inclination change (radians).

    Returns:
        TransferPlan with 2 burns.
    """
    if r1_m <= 0 or r2_m <= 0:
        raise ValueError("Radii must be positive")

    v1_circ = float(np.sqrt(_MU / r1_m))

    # Hohmann transfer ellipse velocities
    a_transfer = (r1_m + r2_m) / 2.0
    v_depart = float(np.sqrt(_MU * (2.0 / r1_m - 1.0 / a_transfer)))
    v_arrive = float(np.sqrt(_MU * (2.0 / r2_m - 1.0 / a_transfer)))
    v2_circ = float(np.sqrt(_MU / r2_m))

    # Burn 1: tangential at departure (no plane change)
    dv1 = abs(v_depart - v1_circ)

    # Burn 2: combined circularization + plane change at arrival
    dv2 = float(np.sqrt(
        v_arrive**2 + v2_circ**2
        - 2.0 * v_arrive * v2_circ * np.cos(inclination_change_rad)
    ))

    t_transfer = np.pi * float(np.sqrt(a_transfer**3 / _MU))

    return TransferPlan(
        burns=(
            ManeuverBurn(delta_v_ms=dv1, description="Departure burn (tangential)"),
            ManeuverBurn(delta_v_ms=dv2, description="Arrival burn (circularize + plane change)"),
        ),
        total_delta_v_ms=dv1 + dv2,
        transfer_time_s=t_transfer,
    )


def phasing_maneuver(
    a_m: float, phase_angle_rad: float, n_orbits: int = 1,
) -> TransferPlan:
    """In-plane phasing maneuver to close a phase angle gap.

    Uses a phasing orbit with adjusted period.

    Args:
        a_m: Nominal circular orbit semi-major axis (m).
        phase_angle_rad: Phase angle to close (radians).
        n_orbits: Number of phasing orbits.

    Returns:
        TransferPlan with 2 burns.

    Raises:
        ValueError: If a_m <= 0 or n_orbits < 1.
    """
    if a_m <= 0:
        raise ValueError(f"a_m must be positive, got {a_m}")
    if n_orbits < 1:
        raise ValueError(f"n_orbits must be >= 1, got {n_orbits}")

    T_nominal = 2.0 * np.pi * float(np.sqrt(a_m**3 / _MU))

    # Phasing orbit period: adjusted so after n_orbits, the phase gap is closed
    T_phasing = T_nominal * (1.0 - phase_angle_rad / (2.0 * np.pi * n_orbits))

    # Semi-major axis of phasing orbit from Kepler's 3rd law
    a_phasing = ((_MU * (T_phasing / (2.0 * np.pi))**2))**(1.0 / 3.0)

    # Delta-V to enter and exit phasing orbit
    v_nominal = float(np.sqrt(_MU / a_m))
    v_phasing = float(np.sqrt(_MU * (2.0 / a_m - 1.0 / a_phasing)))
    dv = abs(v_phasing - v_nominal)

    transfer_time = T_phasing * n_orbits

    return TransferPlan(
        burns=(
            ManeuverBurn(delta_v_ms=dv, description="Enter phasing orbit"),
            ManeuverBurn(delta_v_ms=dv, description="Return to nominal orbit"),
        ),
        total_delta_v_ms=2.0 * dv,
        transfer_time_s=transfer_time,
    )


@dataclass(frozen=True)
class FiniteBurnConfig:
    """Configuration for finite-duration burns."""
    thrust_n: float           # thrust force (Newtons)
    isp_s: float              # specific impulse (seconds)
    initial_mass_kg: float    # wet mass at burn start


@dataclass(frozen=True)
class FiniteBurnResult:
    """Result of a finite-duration burn computation."""
    delta_v_ms: float         # achieved delta-V
    burn_duration_s: float    # engine-on time
    propellant_mass_kg: float # mass consumed
    final_mass_kg: float      # mass after burn
    thrust_arc_deg: float     # angular arc traversed during burn


def compute_finite_burn(
    delta_v_target_ms: float,
    config: FiniteBurnConfig,
    orbital_period_s: float = 5400.0,
) -> FiniteBurnResult:
    """Compute finite burn parameters for a target delta-V.

    Uses Tsiolkovsky equation: dv = Isp * g0 * ln(m0/mf)
    Solves for final mass, propellant, burn duration, and thrust arc.

    Args:
        delta_v_target_ms: Target delta-V (m/s).
        config: Finite burn configuration.
        orbital_period_s: Orbital period for arc calculation (seconds).

    Returns:
        FiniteBurnResult with burn parameters.

    Raises:
        ValueError: If thrust <= 0, Isp <= 0, or mass <= 0.
    """
    if config.thrust_n <= 0:
        raise ValueError(f"thrust_n must be > 0, got {config.thrust_n}")
    if config.isp_s <= 0:
        raise ValueError(f"isp_s must be > 0, got {config.isp_s}")
    if config.initial_mass_kg <= 0:
        raise ValueError(f"initial_mass_kg must be > 0, got {config.initial_mass_kg}")

    exhaust_vel = config.isp_s * _G0

    # Tsiolkovsky: mf = m0 * exp(-dv / ve)
    final_mass = config.initial_mass_kg * float(np.exp(-delta_v_target_ms / exhaust_vel))
    propellant_mass = config.initial_mass_kg - final_mass

    # Burn duration: T = propellant * ve / thrust
    burn_duration = propellant_mass * exhaust_vel / config.thrust_n

    # Thrust arc: fraction of orbit × 360°
    thrust_arc = (burn_duration / orbital_period_s) * 360.0

    return FiniteBurnResult(
        delta_v_ms=delta_v_target_ms,
        burn_duration_s=burn_duration,
        propellant_mass_kg=propellant_mass,
        final_mass_kg=final_mass,
        thrust_arc_deg=thrust_arc,
    )


def finite_burn_loss(
    delta_v_ms: float,
    burn_duration_s: float,
    orbital_velocity_ms: float,
) -> float:
    """Compute effective delta-V after finite burn gravity loss.

    Uses the cosine loss approximation: effective_dv = dv * cos(alpha/2)
    where alpha is the burn arc in radians.

    Note:
        The burn arc alpha = burn_duration * v / r assumes a circular orbit
        (constant v and r). For eccentric orbits, the burn arc varies over
        the orbit and this approximation underestimates the gravity loss
        at perigee and overestimates at apogee. For high-eccentricity orbits,
        integrate the finite burn equations numerically.

    Args:
        delta_v_ms: Impulsive delta-V (m/s).
        burn_duration_s: Burn duration (seconds).
        orbital_velocity_ms: Orbital velocity (m/s).

    Returns:
        Effective delta-V after accounting for gravity loss (m/s).
    """
    if orbital_velocity_ms <= 0 or burn_duration_s <= 0:
        return delta_v_ms

    # Burn arc in radians: angular rate * burn time
    # angular_rate ≈ v / r, burn arc ≈ burn_duration * angular_rate
    # For circular orbit: angular_rate = v / r = v^2 / (v * r) = mu / (r^2 * v)
    # Simpler: alpha = burn_duration * v / r, but we don't know r
    # Use alpha = delta_v / v as a reasonable approximation
    alpha = burn_duration_s * orbital_velocity_ms / (
        _MU / orbital_velocity_ms
    )  # burn_duration * omega = burn_duration * v / r

    # Cosine loss (clamp arc to avoid negative effective dv for very long burns)
    alpha = min(alpha, np.pi)
    effective_dv = delta_v_ms * float(np.cos(alpha / 2.0))
    return effective_dv


def low_thrust_spiral(
    r1_m: float,
    r2_m: float,
    config: FiniteBurnConfig,
) -> FiniteBurnResult:
    """Edelbaum approximation for continuous low-thrust orbit raising.

    Spiral transfer between circular orbits.
    dv ≈ |v1 - v2| where v = sqrt(mu/r).

    Args:
        r1_m: Initial circular orbit radius (m).
        r2_m: Final circular orbit radius (m).
        config: Finite burn configuration.

    Returns:
        FiniteBurnResult with spiral parameters.

    Raises:
        ValueError: If radii <= 0 or config invalid.
    """
    if r1_m <= 0:
        raise ValueError(f"r1_m must be > 0, got {r1_m}")
    if r2_m <= 0:
        raise ValueError(f"r2_m must be > 0, got {r2_m}")
    if config.thrust_n <= 0:
        raise ValueError(f"thrust_n must be > 0, got {config.thrust_n}")
    if config.isp_s <= 0:
        raise ValueError(f"isp_s must be > 0, got {config.isp_s}")
    if config.initial_mass_kg <= 0:
        raise ValueError(f"initial_mass_kg must be > 0, got {config.initial_mass_kg}")

    v1 = float(np.sqrt(_MU / r1_m))
    v2 = float(np.sqrt(_MU / r2_m))
    delta_v = abs(v1 - v2)

    exhaust_vel = config.isp_s * _G0
    final_mass = config.initial_mass_kg * float(np.exp(-delta_v / exhaust_vel))
    propellant_mass = config.initial_mass_kg - final_mass

    # Exact burn duration from constant-thrust mass flow
    burn_duration = propellant_mass * exhaust_vel / config.thrust_n

    # Total arc traversed (spiral, so many orbits)
    avg_r = (r1_m + r2_m) / 2.0
    avg_period = 2.0 * np.pi * float(np.sqrt(avg_r**3 / _MU))
    thrust_arc = (burn_duration / avg_period) * 360.0

    return FiniteBurnResult(
        delta_v_ms=delta_v,
        burn_duration_s=burn_duration,
        propellant_mass_kg=propellant_mass,
        final_mass_kg=final_mass,
        thrust_arc_deg=thrust_arc,
    )


def add_propellant_estimate(
    transfer: TransferPlan, isp_s: float, dry_mass_kg: float,
) -> TransferPlan:
    """Compute propellant mass for a transfer using Tsiolkovsky equation.

    Args:
        transfer: Existing TransferPlan.
        isp_s: Specific impulse (seconds).
        dry_mass_kg: Dry mass of satellite (kg).

    Returns:
        New TransferPlan with propellant_mass_kg filled in.
    """
    prop_mass = dry_mass_kg * (
        float(np.exp(transfer.total_delta_v_ms / (isp_s * _G0))) - 1.0
    )
    return TransferPlan(
        burns=transfer.burns,
        total_delta_v_ms=transfer.total_delta_v_ms,
        transfer_time_s=transfer.transfer_time_s,
        propellant_mass_kg=prop_mass,
    )


# ── P13: Pontryagin Minimum-Fuel Plane Change ─────────────────────


@dataclass(frozen=True)
class OptimalPlaneChange:
    """Result of optimal plane change maneuver timing.

    For an eccentric orbit the dV for a plane change depends on the velocity
    magnitude at the maneuver point.  The classical formula dV = 2v*sin(di/2)
    is minimised where v is smallest, i.e. at apoapsis.  On top of that the
    Gauss variational equation for di shows that the sensitivity of inclination
    to a cross-track impulse also varies with position: di = dV_n * r*cos(theta)/h,
    so the *efficiency* (di per unit dV) is maximised where r*cos(theta)/h is
    largest.

    The Pontryagin minimum principle for the impulsive case reduces to finding
    the true anomaly that minimises dV = di_target / (r*cos(theta) / (h*v))
    = di_target * h * v / (r * cos(theta)), which we solve numerically.

    Attributes:
        optimal_true_anomaly_rad: True anomaly at which the plane change burn
            should be executed for minimum dV.
        min_delta_v_ms: Minimum delta-V achievable at the optimal position.
        node_delta_v_ms: Delta-V if the maneuver were at the ascending node
            (nu such that theta = 0, i.e. nu = -omega).
        savings_percent: Percentage savings of optimal vs ascending-node burn.
    """
    optimal_true_anomaly_rad: float
    min_delta_v_ms: float
    node_delta_v_ms: float
    savings_percent: float
    validity_warning: str = ""
    nonlinear_node_delta_v_ms: float = 0.0


def optimal_plane_change(
    semi_major_axis_m: float,
    eccentricity: float,
    inclination_change_rad: float,
    arg_perigee_rad: float = 0.0,
    num_samples: int = 720,
) -> OptimalPlaneChange:
    """Find the true anomaly minimising dV for a given inclination change.

    Uses Pontryagin's minimum principle for the impulsive case: the optimal
    maneuver point minimises dV = |delta_i| * h * v / (r * |cos(theta)|),
    where theta = omega + nu is the argument of latitude.

    The GVE for inclination under a cross-track impulse:
        delta_i = (r * cos(theta) / h) * dV_n
    Rearranging:
        dV_n = delta_i * h / (r * cos(theta))

    The factor h / (r * cos(theta)) is the inverse of the inclination
    sensitivity. We evaluate it around the orbit and pick the minimum.

    For circular orbits the minimum is at theta = 0 or pi (the nodes),
    recovering the classical result.  For eccentric orbits the minimum
    shifts toward apoapsis because r is larger there.

    Args:
        semi_major_axis_m: Semi-major axis (m).
        eccentricity: Eccentricity (0 = circular).
        inclination_change_rad: Required inclination change (radians, positive).
        arg_perigee_rad: Argument of perigee (radians).
        num_samples: Number of true anomaly samples.

    Note:
        The linearized GVE formula dv = di * h / (r * |cos(theta)|) is only
        valid for small inclination changes (< ~5 deg). For larger changes,
        the nonlinear formula dv = 2*v*sin(di/2) should be used. When
        di > 5 deg, a validity_warning is set and nonlinear_node_delta_v_ms
        provides the nonlinear comparison value.

        The optimal position is found by grid search over num_samples
        true anomaly values. For analytical optimization, the stationarity
        condition d(dV)/d(nu) = 0 could be solved, but the grid search is
        robust to all eccentricities and avoids root-finding edge cases.

    Returns:
        OptimalPlaneChange with optimal position and savings.

    Raises:
        ValueError: If semi_major_axis_m <= 0, eccentricity < 0 or >= 1,
            inclination_change_rad == 0, or num_samples < 4.
    """
    if semi_major_axis_m <= 0:
        raise ValueError(
            f"semi_major_axis_m must be positive, got {semi_major_axis_m}"
        )
    if eccentricity < 0 or eccentricity >= 1.0:
        raise ValueError(f"eccentricity must be in [0, 1), got {eccentricity}")
    if abs(inclination_change_rad) < 1e-15:
        raise ValueError("inclination_change_rad must be non-zero")
    if num_samples < 4:
        raise ValueError(f"num_samples must be >= 4, got {num_samples}")

    mu = _MU
    a = semi_major_axis_m
    e = eccentricity
    omega = arg_perigee_rad
    di = abs(inclination_change_rad)

    p = a * (1.0 - e * e)
    h = math.sqrt(mu * p)

    nu_values = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
    best_dv = float('inf')
    best_nu = 0.0

    for nu in nu_values:
        cos_nu = float(np.cos(nu))
        theta = omega + nu
        cos_theta = float(np.cos(theta))

        # Skip positions near the equatorial crossing where cos(theta) ~ 0
        # (di sensitivity goes to infinity, dV goes to infinity)
        if abs(cos_theta) < 1e-10:
            continue

        r = p / (1.0 + e * cos_nu)
        # dV = di * h / (r * |cos(theta)|)
        dv = di * h / (r * abs(cos_theta))

        if dv < best_dv:
            best_dv = dv
            best_nu = float(nu)

    # Node burn: at ascending node theta = 0, so nu_node = -omega (mod 2pi)
    nu_node = (-omega) % (2.0 * math.pi)
    cos_nu_node = math.cos(nu_node)
    r_node = p / (1.0 + e * cos_nu_node)
    # At the ascending node, cos(theta) = cos(omega + nu_node) = cos(0) = 1
    dv_node = di * h / (r_node * 1.0)

    # Nonlinear node dV: dv = 2*v*sin(di/2) (exact for impulsive plane change)
    v_node = math.sqrt(mu * (2.0 / r_node - 1.0 / a))
    nonlinear_dv_node = 2.0 * v_node * math.sin(di / 2.0)

    # Validity warning: linearized GVE formula dv = di*h/(r*|cos(theta)|)
    # is only valid for small di (< ~5 deg). For larger changes, the
    # nonlinear formula dv = 2*v*sin(di/2) should be used.
    di_deg = math.degrees(di)
    if di_deg > 5.0:
        validity_warning = (
            f"Linearized GVE formula is only valid for small inclination "
            f"changes (<5 deg). Current di = {di_deg:.1f} deg. "
            f"Nonlinear node dV = {nonlinear_dv_node:.2f} m/s vs "
            f"linearized node dV = {dv_node:.2f} m/s."
        )
    else:
        validity_warning = ""

    if dv_node > 1e-15:
        savings = (dv_node - best_dv) / dv_node * 100.0
    else:
        savings = 0.0

    return OptimalPlaneChange(
        optimal_true_anomaly_rad=best_nu,
        min_delta_v_ms=best_dv,
        node_delta_v_ms=dv_node,
        savings_percent=max(savings, 0.0),
        validity_warning=validity_warning,
        nonlinear_node_delta_v_ms=nonlinear_dv_node,
    )


# ── P12: Low-Thrust Trajectory via Euler-Lagrange ─────────────────


@dataclass(frozen=True)
class LowThrustTrajectory:
    """Result of low-thrust spiral trajectory computation.

    Extends the Edelbaum approximation to eccentric orbits using the
    Euler-Lagrange equations of the variational problem.

    Attributes:
        total_dv_ms: Total delta-V for the transfer.
        transfer_time_s: Total transfer time.
        propellant_mass_kg: Propellant consumed.
        final_mass_kg: Mass after transfer.
        num_revolutions: Number of orbit revolutions during transfer.
        edelbaum_dv_ms: Edelbaum circular approximation dV for comparison.
    """
    total_dv_ms: float
    transfer_time_s: float
    propellant_mass_kg: float
    final_mass_kg: float
    num_revolutions: float
    edelbaum_dv_ms: float


def low_thrust_transfer(
    a_initial_m: float,
    e_initial: float,
    a_final_m: float,
    e_final: float,
    thrust_n: float,
    isp_s: float,
    mass_kg: float,
    num_segments: int = 100,
) -> LowThrustTrajectory:
    """Compute low-thrust spiral transfer using Euler-Lagrange optimal control.

    For the minimum-fuel problem with continuous thrust:
        min integral ||u||^2 dt
    subject to the Gauss variational equations, the Euler-Lagrange
    optimality condition gives the optimal thrust direction along-track
    with magnitude proportional to the costate.

    For the case of orbit raising/lowering with optional eccentricity change,
    the analytical solution shows that the optimal strategy is:
    1. Tangential thrust for semi-major axis change.
    2. Radial thrust component for eccentricity correction, applied at specific
       orbital positions (perigee/apogee).

    The total dV is computed by integrating the GVE-derived thrust requirements
    over a quasi-circular spiral, generalised from the Edelbaum formula to
    account for non-zero eccentricity.

    For circular-to-circular: reduces to Edelbaum (dV = |v1 - v2|).
    For eccentric cases: adds the eccentricity change cost.

    Args:
        a_initial_m: Initial semi-major axis (m).
        e_initial: Initial eccentricity.
        a_final_m: Final semi-major axis (m).
        e_final: Final eccentricity.
        thrust_n: Continuous thrust level (N).
        isp_s: Specific impulse (s).
        mass_kg: Initial wet mass (kg).
        num_segments: Number of integration segments.

    Returns:
        LowThrustTrajectory with transfer parameters.

    Raises:
        ValueError: If physical parameters are non-positive or eccentricities
            are out of [0, 1).
    """
    if a_initial_m <= 0:
        raise ValueError(f"a_initial_m must be positive, got {a_initial_m}")
    if a_final_m <= 0:
        raise ValueError(f"a_final_m must be positive, got {a_final_m}")
    if thrust_n <= 0:
        raise ValueError(f"thrust_n must be positive, got {thrust_n}")
    if isp_s <= 0:
        raise ValueError(f"isp_s must be positive, got {isp_s}")
    if mass_kg <= 0:
        raise ValueError(f"mass_kg must be positive, got {mass_kg}")
    if e_initial < 0 or e_initial >= 1.0:
        raise ValueError(f"e_initial must be in [0, 1), got {e_initial}")
    if e_final < 0 or e_final >= 1.0:
        raise ValueError(f"e_final must be in [0, 1), got {e_final}")

    mu = _MU

    # Edelbaum dV (circular approximation): |v1 - v2|
    v1_circ = math.sqrt(mu / a_initial_m)
    v2_circ = math.sqrt(mu / a_final_m)
    edelbaum_dv = abs(v1_circ - v2_circ)

    # For eccentric orbits, compute dV by integrating the spiral.
    # The strategy: step through SMA changes in num_segments steps,
    # at each step compute the local circular velocity and the tangential
    # thrust needed. For eccentricity change, add the radial correction cost.

    # SMA change component: integrate |dv/da| * da along the spiral.
    # For a quasi-circular spiral: dv/da = -v/(2a) (from vis-viva differentiation)
    # so dV_a = integral_{a1}^{a2} v/(2a) da = integral sqrt(mu)/(2 a^{3/2}) da
    #         = sqrt(mu) * [- a^{-1/2}]_{a1}^{a2} = |v1 - v2|
    # This reproduces Edelbaum for e=0.

    # For nonzero eccentricity, the vis-viva average velocity over an orbit is
    # higher than the circular velocity. The orbit-averaged velocity for an
    # eccentric orbit is: <v> = v_circ * sqrt(1 + e^2/2) to first order.
    # More precisely, the orbit-averaged specific energy relation gives us:
    # <v^2> = mu * (2/<r> - 1/a), where <1/r> = 1/(a*sqrt(1-e^2)).
    # We use the exact integration below.

    a_values = np.linspace(a_initial_m, a_final_m, num_segments + 1)
    e_values = np.linspace(e_initial, e_final, num_segments + 1)
    da = (a_final_m - a_initial_m) / num_segments

    total_dv_a = 0.0
    for k in range(num_segments):
        a_mid = (a_values[k] + a_values[k + 1]) / 2.0
        e_mid = (e_values[k] + e_values[k + 1]) / 2.0
        # Orbit-averaged velocity for an elliptic orbit:
        # <v> = sqrt(mu/a) * sqrt(1 + e^2/2) (first-order in e^2)
        v_avg = math.sqrt(mu / a_mid) * math.sqrt(1.0 + e_mid * e_mid / 2.0)
        # dV for this SMA segment: tangential thrust requirement
        # dV = v/(2a) * |da| (from vis-viva linearization)
        total_dv_a += v_avg / (2.0 * a_mid) * abs(da)

    # Eccentricity change component.
    # The GVE for de under tangential thrust, orbit-averaged:
    #   de_per_dV_t ~ 2e/(n*a) for small e
    # Inverting: dV for eccentricity change = |de| * n * a / (2 * max(e, small))
    # We use a more careful derivation:
    # Optimal eccentricity change uses radial impulses at specific anomalies.
    # For perigee burn: delta_e = (2*p/h) * dV_r (from GVE for e, radial component)
    # where the perigee term gives: delta_e = 2*(1-e)/v_p * dV_r
    # So dV_r = |delta_e| * v_p / (2*(1-e))
    de = abs(e_final - e_initial)
    if de > 1e-12:
        # Use mean orbit parameters for the eccentricity correction cost
        a_mean = (a_initial_m + a_final_m) / 2.0
        e_mean = (e_initial + e_final) / 2.0
        n_mean = math.sqrt(mu / a_mean ** 3)
        # From GVE (tangential at perigee): de = 2(1-e^2)/(n*a*e) * dV_t
        # for nearly circular: de ~ 2/(n*a) * dV_t
        # We use the general formula:
        if e_mean > 1e-6:
            dv_e = de * n_mean * a_mean * e_mean / (2.0 * (1.0 - e_mean * e_mean))
        else:
            # For near-circular, eccentricity changes are very cheap
            dv_e = de * n_mean * a_mean / 2.0
    else:
        dv_e = 0.0

    total_dv = total_dv_a + dv_e

    # Transfer time and propellant from thrust and mass
    exhaust_vel = isp_s * _G0
    final_mass = mass_kg * math.exp(-total_dv / exhaust_vel)
    propellant = mass_kg - final_mass
    transfer_time = propellant * exhaust_vel / thrust_n

    # Number of revolutions
    a_mean = (a_initial_m + a_final_m) / 2.0
    mean_period = 2.0 * math.pi * math.sqrt(a_mean ** 3 / mu)
    num_revolutions = transfer_time / mean_period if mean_period > 0 else 0.0

    return LowThrustTrajectory(
        total_dv_ms=total_dv,
        transfer_time_s=transfer_time,
        propellant_mass_kg=propellant,
        final_mass_kg=final_mass,
        num_revolutions=num_revolutions,
        edelbaum_dv_ms=edelbaum_dv,
    )
