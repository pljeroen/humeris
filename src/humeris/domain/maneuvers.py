# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Orbital transfer maneuver planning.

Hohmann, bi-elliptic, plane change, combined, and phasing maneuvers
with propellant estimation via Tsiolkovsky equation.

No external dependencies — only stdlib math/dataclasses.
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

    v1 = float(np.sqrt(_MU / r1_m))
    v2 = float(np.sqrt(_MU / r2_m))

    dv1 = abs(v1 * (float(np.sqrt(2.0 * r2_m / (r1_m + r2_m))) - 1.0))
    dv2 = abs(v2 * (1.0 - float(np.sqrt(2.0 * r1_m / (r1_m + r2_m)))))

    t_transfer = np.pi * float(np.sqrt((r1_m + r2_m) ** 3 / (8.0 * _MU)))

    return TransferPlan(
        burns=(
            ManeuverBurn(delta_v_ms=dv1, description="Hohmann burn 1 (departure)"),
            ManeuverBurn(delta_v_ms=dv2, description="Hohmann burn 2 (arrival)"),
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
