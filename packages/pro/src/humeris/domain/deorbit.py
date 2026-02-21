# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Deorbit compliance assessment.

Evaluates whether a satellite meets FCC 5-year or ESA 25-year
deorbit regulations, computes required delta-V if non-compliant.

"""
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.atmosphere import DragConfig
from humeris.domain.lifetime import compute_orbit_lifetime
from humeris.domain.maneuvers import hohmann_transfer

_MU = OrbitalConstants.MU_EARTH
_R_EARTH = OrbitalConstants.R_EARTH_EQUATORIAL
_G0 = 9.80665


class DeorbitRegulation(Enum):
    FCC_5YEAR = "fcc_5year"    # 5-year rule (effective Sept 2024)
    ESA_25YEAR = "esa_25year"  # ESA 25-year guideline (legacy)


_REGULATION_THRESHOLDS = {
    DeorbitRegulation.FCC_5YEAR: 5.0 * 365.25,    # days
    DeorbitRegulation.ESA_25YEAR: 25.0 * 365.25,   # days
}


@dataclass(frozen=True)
class DeorbitAssessment:
    """Result of deorbit compliance assessment."""
    compliant: bool
    regulation: DeorbitRegulation
    natural_lifetime_days: float
    threshold_days: float
    maneuver_required: bool
    deorbit_delta_v_ms: float       # 0.0 if naturally compliant
    target_perigee_km: float | None  # perigee after deorbit burn
    propellant_mass_kg: float | None  # if isp and mass provided


def assess_deorbit_compliance(
    altitude_km: float,
    drag_config: DragConfig,
    epoch: datetime,
    regulation: DeorbitRegulation = DeorbitRegulation.FCC_5YEAR,
    eccentricity: float = 0.0,
    isp_s: float | None = None,
    dry_mass_kg: float | None = None,
) -> DeorbitAssessment:
    """Assess whether a satellite meets deorbit regulations.

    1. Compute natural lifetime via compute_orbit_lifetime()
    2. If lifetime <= threshold: compliant, no maneuver needed
    3. If not: bisection search for target perigee that gives compliant lifetime
    4. Compute Hohmann dV for perigee lowering

    Args:
        altitude_km: Orbital altitude (km).
        drag_config: Satellite drag configuration.
        epoch: Reference epoch.
        regulation: Deorbit regulation to assess.
        eccentricity: Orbital eccentricity.
        isp_s: Specific impulse (s) for propellant estimate.
        dry_mass_kg: Dry mass (kg) for propellant estimate.

    Returns:
        DeorbitAssessment with compliance result.
    """
    threshold_days = _REGULATION_THRESHOLDS[regulation]
    a = _R_EARTH + altitude_km * 1000.0

    # Compute natural lifetime
    lifetime_result = compute_orbit_lifetime(
        semi_major_axis_m=a,
        eccentricity=eccentricity,
        drag_config=drag_config,
        epoch=epoch,
        max_years=max(50.0, threshold_days / 365.25 + 5.0),
    )
    natural_lifetime_days = lifetime_result.lifetime_days

    if natural_lifetime_days <= threshold_days:
        return DeorbitAssessment(
            compliant=True,
            regulation=regulation,
            natural_lifetime_days=natural_lifetime_days,
            threshold_days=threshold_days,
            maneuver_required=False,
            deorbit_delta_v_ms=0.0,
            target_perigee_km=None,
            propellant_mass_kg=None,
        )

    # Not compliant — find target perigee via bisection
    # Lower perigee → higher drag at perigee → shorter lifetime
    perigee_lo_km = 100.0  # minimum survivable perigee
    perigee_hi_km = altitude_km  # current altitude

    for _ in range(50):
        perigee_mid_km = (perigee_lo_km + perigee_hi_km) / 2.0
        # Compute lifetime for orbit with lowered perigee
        # Use mean altitude (perigee + apogee) / 2 as effective circular altitude
        # Apogee stays at original altitude, perigee lowered
        effective_alt_km = (perigee_mid_km + altitude_km) / 2.0
        a_eff = _R_EARTH + effective_alt_km * 1000.0

        try:
            lt = compute_orbit_lifetime(
                semi_major_axis_m=a_eff,
                eccentricity=0.0,
                drag_config=drag_config,
                epoch=epoch,
                max_years=max(50.0, threshold_days / 365.25 + 5.0),
            )
            lt_days = lt.lifetime_days
        except ValueError:
            lt_days = 0.0

        if lt_days <= threshold_days:
            perigee_lo_km = perigee_mid_km
        else:
            perigee_hi_km = perigee_mid_km

        if abs(perigee_hi_km - perigee_lo_km) < 0.1:
            break

    target_perigee_km = perigee_hi_km

    # Compute delta-V for Hohmann maneuver to lower perigee
    r_circular = a
    # Target orbit: apogee at current altitude, perigee at target
    r_perigee = _R_EARTH + target_perigee_km * 1000.0
    a_target = (r_circular + r_perigee) / 2.0
    # dV = v_circular - v_at_apogee_of_target_orbit
    v_circular = float(np.sqrt(_MU / r_circular))
    v_at_apogee = float(np.sqrt(_MU * (2.0 / r_circular - 1.0 / a_target)))
    delta_v = abs(v_circular - v_at_apogee)

    # Propellant estimate
    propellant_mass = None
    if isp_s is not None and dry_mass_kg is not None and delta_v > 0:
        propellant_mass = dry_mass_kg * (
            float(np.exp(delta_v / (isp_s * _G0))) - 1.0
        )

    return DeorbitAssessment(
        compliant=False,
        regulation=regulation,
        natural_lifetime_days=natural_lifetime_days,
        threshold_days=threshold_days,
        maneuver_required=True,
        deorbit_delta_v_ms=delta_v,
        target_perigee_km=target_perigee_km,
        propellant_mass_kg=propellant_mass,
    )
