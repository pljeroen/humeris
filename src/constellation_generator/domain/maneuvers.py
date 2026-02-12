# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""
Orbital transfer maneuver planning.

Hohmann, bi-elliptic, plane change, combined, and phasing maneuvers
with propellant estimation via Tsiolkovsky equation.

No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass

from constellation_generator.domain.orbital_mechanics import OrbitalConstants

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

    v1 = math.sqrt(_MU / r1_m)
    v2 = math.sqrt(_MU / r2_m)

    dv1 = abs(v1 * (math.sqrt(2.0 * r2_m / (r1_m + r2_m)) - 1.0))
    dv2 = abs(v2 * (1.0 - math.sqrt(2.0 * r1_m / (r1_m + r2_m))))

    t_transfer = math.pi * math.sqrt((r1_m + r2_m) ** 3 / (8.0 * _MU))

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

    v1 = math.sqrt(_MU / r1_m)
    # Burn 1: raise apogee from r1 to r_intermediate
    v_transfer1_peri = math.sqrt(2.0 * _MU * r_intermediate_m / (r1_m * (r1_m + r_intermediate_m)))
    dv1 = abs(v_transfer1_peri - v1)

    # Burn 2: at r_intermediate, change orbit from (r1, r_int) to (r2, r_int)
    v_transfer1_apo = math.sqrt(2.0 * _MU * r1_m / (r_intermediate_m * (r1_m + r_intermediate_m)))
    v_transfer2_apo = math.sqrt(2.0 * _MU * r2_m / (r_intermediate_m * (r2_m + r_intermediate_m)))
    dv2 = abs(v_transfer2_apo - v_transfer1_apo)

    # Burn 3: circularize at r2
    v_transfer2_peri = math.sqrt(2.0 * _MU * r_intermediate_m / (r2_m * (r2_m + r_intermediate_m)))
    v2 = math.sqrt(_MU / r2_m)
    dv3 = abs(v2 - v_transfer2_peri)

    # Transfer time: half-period of first transfer + half-period of second
    t1 = math.pi * math.sqrt((r1_m + r_intermediate_m) ** 3 / (8.0 * _MU))
    t2 = math.pi * math.sqrt((r2_m + r_intermediate_m) ** 3 / (8.0 * _MU))

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
    return 2.0 * velocity_ms * abs(math.sin(inclination_change_rad / 2.0))


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

    v1_circ = math.sqrt(_MU / r1_m)

    # Hohmann transfer ellipse velocities
    a_transfer = (r1_m + r2_m) / 2.0
    v_depart = math.sqrt(_MU * (2.0 / r1_m - 1.0 / a_transfer))
    v_arrive = math.sqrt(_MU * (2.0 / r2_m - 1.0 / a_transfer))
    v2_circ = math.sqrt(_MU / r2_m)

    # Burn 1: tangential at departure (no plane change)
    dv1 = abs(v_depart - v1_circ)

    # Burn 2: combined circularization + plane change at arrival
    dv2 = math.sqrt(
        v_arrive**2 + v2_circ**2
        - 2.0 * v_arrive * v2_circ * math.cos(inclination_change_rad)
    )

    t_transfer = math.pi * math.sqrt(a_transfer**3 / _MU)

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

    T_nominal = 2.0 * math.pi * math.sqrt(a_m**3 / _MU)

    # Phasing orbit period: adjusted so after n_orbits, the phase gap is closed
    T_phasing = T_nominal * (1.0 - phase_angle_rad / (2.0 * math.pi * n_orbits))

    # Semi-major axis of phasing orbit from Kepler's 3rd law
    a_phasing = ((_MU * (T_phasing / (2.0 * math.pi))**2))**(1.0 / 3.0)

    # Delta-V to enter and exit phasing orbit
    v_nominal = math.sqrt(_MU / a_m)
    v_phasing = math.sqrt(_MU * (2.0 / a_m - 1.0 / a_phasing))
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
        math.exp(transfer.total_delta_v_ms / (isp_s * _G0)) - 1.0
    )
    return TransferPlan(
        burns=transfer.burns,
        total_delta_v_ms=transfer.total_delta_v_ms,
        transfer_time_s=transfer.transfer_time_s,
        propellant_mass_kg=prop_mass,
    )
