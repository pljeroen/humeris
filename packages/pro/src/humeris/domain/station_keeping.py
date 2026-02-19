# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Station-keeping delta-V budgets and propellant computation.

Drag compensation, plane maintenance, Tsiolkovsky rocket equation,
and combined operational lifetime budgets.

"""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.atmosphere import (
    AtmosphereModel,
    DragConfig,
    atmospheric_density,
    drag_acceleration,
    semi_major_axis_decay_rate,
)


_G0 = 9.80665  # standard gravity m/s²
_SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass(frozen=True)
class StationKeepingConfig:
    """Configuration for station-keeping budget computation."""
    target_altitude_km: float
    inclination_deg: float
    drag_config: DragConfig
    isp_s: float
    dry_mass_kg: float
    propellant_mass_kg: float


@dataclass(frozen=True)
class StationKeepingBudget:
    """Station-keeping delta-V and propellant budget."""
    drag_dv_per_year_ms: float
    plane_dv_per_year_ms: float
    total_dv_per_year_ms: float
    propellant_per_year_kg: float
    operational_lifetime_years: float
    total_dv_capacity_ms: float


def drag_compensation_dv_per_year(
    altitude_km: float,
    drag_config: DragConfig,
    atmosphere_model: AtmosphereModel | None = None,
    density_func: Optional[Callable[[float, datetime], float]] = None,
    epoch: datetime | None = None,
) -> float:
    """Annual delta-V for drag compensation at given altitude.

    Uses vis-viva linearization: dV/dt = |da/dt| * n / 2, annualized.
    Ref: Wertz SMAD Ch. 6.

    Args:
        altitude_km: Target orbit altitude (km).
        drag_config: Satellite drag configuration.
        atmosphere_model: Exponential atmosphere model selection.
        density_func: Optional callback (altitude_km, epoch) -> density_kg_m3.
            When provided, overrides atmosphere_model.
        epoch: Reference epoch for density_func evaluation. Required when
            density_func is provided; ignored otherwise.

    Returns:
        Delta-V in m/s per year.
    """
    a = OrbitalConstants.R_EARTH + altitude_km * 1000.0
    n = float(np.sqrt(OrbitalConstants.MU_EARTH / a ** 3))

    if density_func is not None:
        if epoch is None:
            from datetime import timezone
            epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rho = density_func(altitude_km, epoch)
        v = float(np.sqrt(OrbitalConstants.MU_EARTH / a))
        da_dt = -rho * v * drag_config.ballistic_coefficient * a
    else:
        decay_kwargs: dict = {}
        if atmosphere_model is not None:
            decay_kwargs["model"] = atmosphere_model
        da_dt = semi_major_axis_decay_rate(a, 0.0, drag_config, **decay_kwargs)

    dv_per_second = abs(da_dt) * n / 2.0
    return dv_per_second * _SECONDS_PER_YEAR


def plane_maintenance_dv_per_year(
    inclination_deg: float,
    altitude_km: float,
    delta_inclination_deg: float = 0.05,
) -> float:
    """Annual delta-V for orbital plane maintenance.

    dV = 2 * v * sin(di/2) for inclination correction budget.

    Args:
        inclination_deg: Orbit inclination (degrees) — unused in formula
            but kept for API consistency and future J2 drift models.
        altitude_km: Orbit altitude (km).
        delta_inclination_deg: Annual inclination correction budget (degrees).

    Returns:
        Delta-V in m/s per year.
    """
    a = OrbitalConstants.R_EARTH + altitude_km * 1000.0
    v = float(np.sqrt(OrbitalConstants.MU_EARTH / a))
    di_rad = float(np.radians(delta_inclination_deg))
    return 2.0 * v * float(np.sin(di_rad / 2.0))


def tsiolkovsky_dv(
    isp_s: float,
    dry_mass_kg: float,
    propellant_mass_kg: float,
) -> float:
    """Tsiolkovsky rocket equation: delta-V from propellant mass.

    dV = Isp * g0 * ln(m0 / mf)

    Args:
        isp_s: Specific impulse (seconds).
        dry_mass_kg: Dry mass (kg).
        propellant_mass_kg: Propellant mass (kg).

    Returns:
        Delta-V capacity in m/s.

    Raises:
        ValueError: If dry_mass <= 0 or propellant < 0.
    """
    if dry_mass_kg <= 0:
        raise ValueError(f"dry_mass_kg must be positive, got {dry_mass_kg}")
    if propellant_mass_kg < 0:
        raise ValueError(f"propellant_mass_kg must be non-negative, got {propellant_mass_kg}")
    if propellant_mass_kg == 0.0:
        return 0.0
    m0 = dry_mass_kg + propellant_mass_kg
    return isp_s * _G0 * float(np.log(m0 / dry_mass_kg))


def propellant_mass_for_dv(
    isp_s: float,
    dry_mass_kg: float,
    dv_ms: float,
) -> float:
    """Propellant mass required for a given delta-V.

    m_prop = m_dry * (exp(dV / (Isp * g0)) - 1)

    Args:
        isp_s: Specific impulse (seconds).
        dry_mass_kg: Dry mass (kg).
        dv_ms: Required delta-V (m/s).

    Returns:
        Propellant mass in kg.
    """
    if isp_s <= 0:
        raise ValueError(f"isp_s must be positive, got {isp_s}")
    if dry_mass_kg <= 0:
        raise ValueError(f"dry_mass_kg must be positive, got {dry_mass_kg}")
    return dry_mass_kg * (float(np.exp(dv_ms / (isp_s * _G0))) - 1.0)


def compute_station_keeping_budget(
    config: StationKeepingConfig,
    density_func: Optional[Callable[[float, datetime], float]] = None,
    epoch: datetime | None = None,
) -> StationKeepingBudget:
    """Compute combined station-keeping budget.

    Combines drag compensation and plane maintenance delta-V,
    computes total capacity via Tsiolkovsky, derives operational lifetime.

    Args:
        config: Station-keeping configuration.
        density_func: Optional density callback for variable atmosphere.
        epoch: Reference epoch for density_func evaluation.

    Returns:
        StationKeepingBudget with annual budgets and lifetime.
    """
    drag_dv = drag_compensation_dv_per_year(
        config.target_altitude_km, config.drag_config,
        density_func=density_func, epoch=epoch,
    )
    plane_dv = plane_maintenance_dv_per_year(config.inclination_deg, config.target_altitude_km)
    total_dv_year = drag_dv + plane_dv

    total_capacity = tsiolkovsky_dv(
        config.isp_s, config.dry_mass_kg, config.propellant_mass_kg,
    )

    if total_dv_year > 0:
        lifetime_years = total_capacity / total_dv_year
    else:
        lifetime_years = float('inf')

    propellant_year = propellant_mass_for_dv(
        config.isp_s, config.dry_mass_kg, total_dv_year,
    )

    return StationKeepingBudget(
        drag_dv_per_year_ms=drag_dv,
        plane_dv_per_year_ms=plane_dv,
        total_dv_per_year_ms=total_dv_year,
        propellant_per_year_kg=propellant_year,
        operational_lifetime_years=lifetime_years,
        total_dv_capacity_ms=total_capacity,
    )


# ── P14: Gauss Variational Equations for Analytic Station-Keeping ──


@dataclass(frozen=True)
class GVEStationKeepingBudget:
    """Station-keeping budget from Gauss Variational Equations.

    Integrates GVE over a full orbit to capture eccentricity effects
    that the linearized vis-viva model ignores.

    Attributes:
        drag_dv_along_track_ms_per_year: Along-track dV to compensate SMA drag decay.
        eccentricity_correction_dv_ms_per_year: dV to counteract drag-induced
            eccentricity growth (invisible to the linearized model).
        total_dv_per_year_ms: Total annual dV budget (along-track + eccentricity).
        linearized_dv_per_year_ms: Value from the linearized vis-viva model for
            direct comparison.
        linearized_error_percent: Overestimate of the linearized model relative to GVE.
    """
    drag_dv_along_track_ms_per_year: float
    eccentricity_correction_dv_ms_per_year: float
    total_dv_per_year_ms: float
    linearized_dv_per_year_ms: float
    linearized_error_percent: float


def compute_gve_station_keeping_budget(
    altitude_km: float,
    eccentricity: float,
    drag_config: DragConfig,
    num_orbit_samples: int = 360,
    atmosphere_model: AtmosphereModel | None = None,
    density_func: Optional[Callable[[float, datetime], float]] = None,
    epoch: datetime | None = None,
) -> GVEStationKeepingBudget:
    """Compute station-keeping budget via Gauss Variational Equations.

    Integrates the GVE for da/dt and de/dt due to drag around a full orbit,
    properly accounting for eccentricity (perigee sees higher drag than apogee).
    The linearized model dV = |da/dt| * n / 2 assumes circular orbits and misses
    the eccentricity maintenance term entirely.

    Gauss VOP for drag (along-track perturbation a_t, opposing velocity):
        da/dt = (2 * a^2 / h) * (p / r) * a_t
        de/dt = (1/h) * ((p+r)*cos(nu) + r*e) * a_t

    where h = sqrt(mu*p), p = a(1-e^2), r = p/(1+e*cos(nu)), and
    a_t = -drag_acceleration (negative because drag opposes velocity).

    The orbit-averaged integrals give the secular da and de per orbit.
    dV_a = |da_avg| * n / 2 annualized (semi-major axis maintenance).
    dV_e = |de_avg| * v * (proportional eccentricity correction).

    Note:
        True anomaly samples where the altitude drops below 100 km are
        skipped because the atmosphere model table does not extend below
        100 km. For very low or highly eccentric orbits with perigee
        below 100 km, the drag integral is therefore incomplete and the
        budget will be underestimated.

    Args:
        altitude_km: Mean orbit altitude (km).
        eccentricity: Orbit eccentricity (0 = circular).
        drag_config: Satellite drag configuration.
        num_orbit_samples: Quadrature points around the orbit.
        atmosphere_model: Atmosphere model (None = default).

    Returns:
        GVEStationKeepingBudget with GVE-derived budget and linearized comparison.

    Raises:
        ValueError: If altitude_km <= 0, eccentricity < 0 or >= 1,
            or num_orbit_samples < 4.
    """
    if altitude_km <= 0:
        raise ValueError(f"altitude_km must be positive, got {altitude_km}")
    if eccentricity < 0 or eccentricity >= 1.0:
        raise ValueError(f"eccentricity must be in [0, 1), got {eccentricity}")
    if num_orbit_samples < 4:
        raise ValueError(
            f"num_orbit_samples must be >= 4, got {num_orbit_samples}"
        )

    mu = OrbitalConstants.MU_EARTH
    r_earth = OrbitalConstants.R_EARTH
    a = r_earth + altitude_km * 1000.0
    e = eccentricity
    p = a * (1.0 - e * e)
    h = math.sqrt(mu * p)
    n = math.sqrt(mu / a ** 3)
    period_s = 2.0 * math.pi / n

    atm_kwargs: dict = {}
    if atmosphere_model is not None:
        atm_kwargs["model"] = atmosphere_model

    # Sample true anomaly around the orbit
    nu_values = np.linspace(0.0, 2.0 * np.pi, num_orbit_samples, endpoint=False)
    d_nu = 2.0 * np.pi / num_orbit_samples

    da_integral = 0.0
    de_integral = 0.0

    for nu in nu_values:
        cos_nu = float(np.cos(nu))
        sin_nu = float(np.sin(nu))

        r = p / (1.0 + e * cos_nu)
        altitude_at_nu_km = (r - r_earth) / 1000.0

        # Skip if altitude outside atmosphere table range
        if altitude_at_nu_km < 100.0 or altitude_at_nu_km > 2000.0:
            continue

        v_at_nu = math.sqrt(mu * (2.0 / r - 1.0 / a))
        if density_func is not None:
            if epoch is not None:
                ref_epoch = epoch
            else:
                from datetime import timezone
                ref_epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
            rho = density_func(altitude_at_nu_km, ref_epoch)
        else:
            rho = atmospheric_density(altitude_at_nu_km, **atm_kwargs)
        a_drag = drag_acceleration(rho, v_at_nu, drag_config)

        # Drag opposes velocity: tangential component = -a_drag
        a_t = -a_drag

        # GVE: da/dt for tangential perturbation
        da_dt_nu = (2.0 * a * a / h) * (p / r) * a_t

        # GVE: de/dt for tangential perturbation
        de_dt_nu = (1.0 / h) * ((p + r) * cos_nu + r * e) * a_t

        # Convert from time-integral to anomaly-integral:
        # dt = (r^2 / h) * d_nu  (from angular momentum: h = r^2 * d_nu/dt)
        dt_d_nu = r * r / h

        da_integral += da_dt_nu * dt_d_nu * d_nu
        de_integral += de_dt_nu * dt_d_nu * d_nu

    # da_integral and de_integral are per-orbit changes
    orbits_per_year = _SECONDS_PER_YEAR / period_s

    da_per_year = da_integral * orbits_per_year  # m/year
    de_per_year = de_integral * orbits_per_year  # 1/year

    # dV for SMA maintenance: dV_a = |da/year| * n / 2
    dv_a_per_year = abs(da_per_year) * n / 2.0

    # dV for eccentricity correction.
    # A tangential impulse at perigee changes e by: delta_e = 2 * (1+e*cos(nu)) / (n*a) * dV_t
    # For correction at perigee (nu=0): delta_e = 2*(1+e)/(n*a) * dV_t
    # So dV_t = |de| * n * a / (2*(1+e))
    v_circ = math.sqrt(mu / a)
    if e > 1e-12:
        dv_e_per_year = abs(de_per_year) * n * a / (2.0 * (1.0 + e))
    else:
        dv_e_per_year = 0.0

    total_dv = dv_a_per_year + dv_e_per_year

    # Linearized comparison
    linearized_dv = drag_compensation_dv_per_year(
        altitude_km, drag_config, atmosphere_model=atmosphere_model,
        density_func=density_func, epoch=epoch,
    )

    if total_dv > 1e-15:
        error_pct = (linearized_dv - total_dv) / total_dv * 100.0
    else:
        error_pct = 0.0

    return GVEStationKeepingBudget(
        drag_dv_along_track_ms_per_year=dv_a_per_year,
        eccentricity_correction_dv_ms_per_year=dv_e_per_year,
        total_dv_per_year_ms=total_dv,
        linearized_dv_per_year_ms=linearized_dv,
        linearized_error_percent=error_pct,
    )


# ── P40: Compartmental Pharmacokinetic Propellant Model ────────────
#
# 3 compartments: P_stored, P_committed, P_expended.
# Models propellant flow through planning → execution → consumption
# with altitude-dependent rates from drag compensation requirements.


@dataclass(frozen=True)
class PropellantPharmacokinetics:
    """Compartmental pharmacokinetic propellant model results.

    Attributes:
        stored_trajectory: P_stored(t) time series (kg).
        committed_trajectory: P_committed(t) time series (kg).
        expended_trajectory: P_expended(t) time series (kg).
        half_life: Time for stored propellant to halve (years).
        depletion_time: Time until stored propellant < 1% of initial (years).
        margin_at_eol: Fraction of initial propellant remaining (stored+committed)
            at end of simulation.
    """
    stored_trajectory: tuple
    committed_trajectory: tuple
    expended_trajectory: tuple
    half_life: float
    depletion_time: float
    margin_at_eol: float


def compute_propellant_pharmacokinetics(
    initial_propellant_kg: float,
    altitude_km: float,
    isp_s: float,
    dry_mass_kg: float,
    drag_cd: float = 2.2,
    drag_area_m2: float = 1.0,
    emergency_rate_per_year: float = 0.01,
    execution_rate_per_year: float = 4.0,
    duration_years: float = 15.0,
    step_years: float = 0.1,
    atmosphere_model: AtmosphereModel | None = None,
    density_func: Optional[Callable[[float, datetime], float]] = None,
    epoch: datetime | None = None,
) -> PropellantPharmacokinetics:
    """Compute compartmental pharmacokinetic propellant depletion model.

    Models propellant in three compartments:
        P_stored: unallocated propellant reserve
        P_committed: propellant allocated to planned maneuvers
        P_expended: propellant consumed (burned)

    Transfer rates:
        k12(h) = dV_drag(h) / (Isp * g0) per year  (drag-driven commitment)
        k13(h) = emergency_rate  (emergency burns directly from stored)
        k23 = execution_rate  (committed burns executed)

    dP_stored/dt = -k12*P_stored - k13*P_stored
    dP_committed/dt = k12*P_stored - k23*P_committed
    dP_expended/dt = k13*P_stored + k23*P_committed

    Conservation: P_stored + P_committed + P_expended = P_initial (always).

    Args:
        initial_propellant_kg: Initial propellant mass (kg).
        altitude_km: Orbit altitude (km).
        isp_s: Specific impulse (seconds).
        dry_mass_kg: Satellite dry mass (kg).
        drag_cd: Drag coefficient.
        drag_area_m2: Cross-sectional area (m^2).
        emergency_rate_per_year: Emergency burn rate k13 (1/year).
        execution_rate_per_year: Committed-to-expended rate k23 (1/year).
        duration_years: Simulation duration (years).
        step_years: Integration time step (years).
        atmosphere_model: Atmosphere model (None = default).

    Returns:
        PropellantPharmacokinetics with compartment trajectories and diagnostics.
    """
    # Compute drag compensation dV rate at this altitude
    drag_config = DragConfig(cd=drag_cd, area_m2=drag_area_m2, mass_kg=dry_mass_kg)
    dv_per_year = drag_compensation_dv_per_year(
        altitude_km, drag_config, atmosphere_model=atmosphere_model,
        density_func=density_func, epoch=epoch,
    )

    # Transfer rate k12: fraction of stored propellant committed per year
    # based on drag dV requirement relative to total capacity
    # k12 = (dV_annual / (Isp * g0)) is the mass fraction consumed per year
    # for a first-order compartment model
    ve = isp_s * _G0  # exhaust velocity m/s
    if ve > 0 and initial_propellant_kg > 0:
        # Mass flow rate for drag compensation (linearized for small dV):
        # dm/dt = m_total * dV/dt / ve (linearized Tsiolkovsky)
        k12 = dv_per_year / ve
    else:
        k12 = 0.0

    k13 = emergency_rate_per_year
    k23 = execution_rate_per_year
    k_total = k12 + k13

    # Half-life of stored propellant
    if k_total > 0:
        half_life = math.log(2.0) / k_total
    else:
        half_life = float('inf')

    # Forward Euler integration
    n_steps = int(duration_years / step_years) + 1
    dt = step_years

    p_stored = initial_propellant_kg
    p_committed = 0.0
    p_expended = 0.0

    stored_arr = [p_stored]
    committed_arr = [p_committed]
    expended_arr = [p_expended]

    depletion_time_found = False
    depletion_time = 0.0

    for step in range(1, n_steps):
        dp_stored = -(k12 + k13) * p_stored
        dp_committed = k12 * p_stored - k23 * p_committed
        dp_expended = k13 * p_stored + k23 * p_committed

        p_stored = p_stored + dp_stored * dt
        p_committed = p_committed + dp_committed * dt
        p_expended = p_expended + dp_expended * dt

        # Clamp: stored and committed cannot go negative
        p_stored = max(p_stored, 0.0)
        p_committed = max(p_committed, 0.0)
        p_expended = max(p_expended, 0.0)

        stored_arr.append(p_stored)
        committed_arr.append(p_committed)
        expended_arr.append(p_expended)

        # Track depletion time (stored < 1% of initial)
        if not depletion_time_found and p_stored < 0.01 * initial_propellant_kg:
            depletion_time = step * dt
            depletion_time_found = True

    # If not depleted during simulation, compute analytically:
    # P_stored(t) = P0 * exp(-k_total * t), so 0.01*P0 = P0*exp(-k_total*t_d)
    # => t_d = ln(100) / k_total
    if not depletion_time_found:
        if k_total > 0:
            depletion_time = math.log(100.0) / k_total
        else:
            depletion_time = float('inf')

    # Margin at EOL: fraction of initial propellant that is still usable
    margin = (stored_arr[-1] + committed_arr[-1]) / initial_propellant_kg if initial_propellant_kg > 0 else 0.0
    margin = max(0.0, min(1.0, margin))

    return PropellantPharmacokinetics(
        stored_trajectory=tuple(stored_arr),
        committed_trajectory=tuple(committed_arr),
        expended_trajectory=tuple(expended_arr),
        half_life=half_life,
        depletion_time=depletion_time,
        margin_at_eol=margin,
    )


# ── P57: Geometric (Berry) Phase for Cyclic Maneuvers ────────────


@dataclass(frozen=True)
class GeometricPhaseResult:
    """Geometric (Berry) phase analysis for cyclic orbital maneuvers.

    When a satellite executes a cycle of maneuvers that returns it to
    the original orbital elements, it acquires a geometric phase (Hannay
    angle) in addition to the dynamic phase from normal orbital motion.

    For a Hohmann-like raise-wait-lower cycle:
    - Dynamic phase: accumulated mean anomaly from orbital motion.
    - Geometric phase: area enclosed in action-angle space.
    - Hannay angle: the geometric correction to the angular position.

    Attributes:
        dynamic_phase_rad: Accumulated dynamic phase from orbital motion.
        geometric_phase_rad: Berry/geometric phase from the cycle contour.
        total_phase_rad: Sum of dynamic and geometric phases.
        hannay_angle_rad: Hannay angle (geometric phase in action-angle).
        mean_anomaly_shift_rad: Net mean anomaly shift after the cycle.
        raan_shift_rad: Net RAAN shift from J2 during the maneuver cycle.
    """
    dynamic_phase_rad: float
    geometric_phase_rad: float
    total_phase_rad: float
    hannay_angle_rad: float
    mean_anomaly_shift_rad: float
    raan_shift_rad: float


def compute_geometric_phase(
    semi_major_axis_m: float,
    delta_a_m: float,
    wait_time_s: float,
    inclination_rad: float = 0.0,
    eccentricity: float = 0.0,
) -> GeometricPhaseResult:
    """Compute geometric (Berry) phase for a raise-wait-lower maneuver cycle.

    The cycle consists of:
    1. Hohmann raise: a -> a + delta_a (instantaneous)
    2. Wait at a + delta_a for wait_time_s
    3. Hohmann lower: a + delta_a -> a (instantaneous)

    Dynamic phase: integral of n(a) * dt over the cycle.
    For the wait segment: phi_dyn = n(a + delta_a) * wait_time_s

    Geometric phase: from the contour integral in action-angle (L, l) space.
    For a Hohmann-like cycle with small delta_a:
        Delta_M_geo = -3*pi * (delta_a / a) * (T_wait / T_orbit)

    This is the Hannay angle -- the geometric correction arising from the
    area enclosed in the (action, angle) = (L, l) phase space.

    RAAN shift from J2 during the cycle (useful for constellation phasing).

    Args:
        semi_major_axis_m: Nominal semi-major axis (m).
        delta_a_m: SMA change for the raise maneuver (m). Can be negative
            for a lowering cycle.
        wait_time_s: Duration spent at the altered altitude (s).
        inclination_rad: Orbit inclination for J2 RAAN calculation (rad).
        eccentricity: Eccentricity (default 0 for circular).

    Returns:
        GeometricPhaseResult with phase breakdown.

    Raises:
        ValueError: If semi_major_axis_m <= 0 or wait_time_s < 0.
    """
    if semi_major_axis_m <= 0:
        raise ValueError(
            f"semi_major_axis_m must be positive, got {semi_major_axis_m}"
        )
    if wait_time_s < 0:
        raise ValueError(f"wait_time_s must be non-negative, got {wait_time_s}")

    mu = OrbitalConstants.MU_EARTH
    a = semi_major_axis_m
    a2 = a + delta_a_m

    if a2 <= 0:
        raise ValueError(
            f"a + delta_a must be positive, got {a2}"
        )

    n1 = math.sqrt(mu / a ** 3)
    n2 = math.sqrt(mu / a2 ** 3)
    T_orbit = 2.0 * math.pi / n1

    # Dynamic phase: accumulated mean anomaly during wait
    dynamic_phase = n2 * wait_time_s

    # Geometric phase (Hannay angle) for Hohmann-like cycle
    # From adiabatic invariant theory in action-angle variables:
    # The action L = sqrt(mu * a), angle l = M (mean anomaly)
    # Area enclosed in (L, l) space gives the geometric phase.
    # For small delta_a: Delta_M_geo = -3*pi * (delta_a/a) * (T_wait/T_orbit)
    # This comes from: d(n)/d(a) = -3n/(2a), and the area in action-angle
    # space = integral p dq around the contour.
    geometric_phase = -3.0 * math.pi * (delta_a_m / a) * (wait_time_s / T_orbit)

    # Hannay angle is the geometric phase
    hannay_angle = geometric_phase

    # Total mean anomaly shift = dynamic + geometric
    total_phase = dynamic_phase + geometric_phase

    # Mean anomaly shift modulo 2*pi
    mean_anomaly_shift = total_phase % (2.0 * math.pi)

    # RAAN shift from J2 during the maneuver cycle
    from humeris.domain.orbital_mechanics import j2_raan_rate
    n2_mean = math.sqrt(mu / a2 ** 3)
    raan_rate = j2_raan_rate(n2_mean, a2, eccentricity, inclination_rad)
    raan_shift = raan_rate * wait_time_s

    return GeometricPhaseResult(
        dynamic_phase_rad=dynamic_phase,
        geometric_phase_rad=geometric_phase,
        total_phase_rad=total_phase,
        hannay_angle_rad=hannay_angle,
        mean_anomaly_shift_rad=mean_anomaly_shift,
        raan_shift_rad=raan_shift,
    )


# ── P62: Stochastic Resonance for Maneuver Timing ───────────────


@dataclass(frozen=True)
class StochasticResonanceManeuver:
    """Stochastic resonance analysis for maneuver timing optimisation.

    In a bistable potential V(x) = -ax^2/2 + bx^4/4, noise-induced
    transitions between the two wells can synchronise with a periodic
    signal (maneuver window frequency), achieving stochastic resonance.

    Attributes:
        optimal_noise_intensity: Noise intensity D_opt at resonance peak.
        peak_snr: Signal-to-noise ratio at optimal noise.
        resonance_frequency_hz: Kramers rate at optimal noise (Hz).
        barrier_height: Potential barrier DeltaV between wells.
        kramers_rate: Kramers escape rate at optimal noise (1/s).
        maneuver_effectiveness_gain: Ratio of SNR at optimal noise to
            SNR at half the optimal noise (resonance enhancement factor).
    """
    optimal_noise_intensity: float
    peak_snr: float
    resonance_frequency_hz: float
    barrier_height: float
    kramers_rate: float
    maneuver_effectiveness_gain: float


def compute_stochastic_resonance(
    potential_a: float,
    potential_b: float,
    signal_amplitude: float,
    signal_frequency_hz: float,
    num_noise_samples: int = 200,
) -> StochasticResonanceManeuver:
    """Compute stochastic resonance parameters for maneuver timing.

    For a bistable potential V(x) = -a*x^2/2 + b*x^4/4:
    - Minima at x = +/- sqrt(a/b)
    - Barrier height: DeltaV = a^2 / (4b)
    - Angular frequency at minimum: omega_0 = sqrt(2a)

    Kramers escape rate: r_K = (omega_0 / (2*pi)) * exp(-DeltaV / D)

    SNR for weak periodic signal A*cos(2*pi*f*t):
        SNR = (pi * A^2) / (4 * D) * exp(-DeltaV / D)

    Optimal noise: D_opt = DeltaV (from d(SNR)/dD = 0).

    Args:
        potential_a: Coefficient a (curvature parameter, > 0 for bistable).
        potential_b: Coefficient b (quartic parameter, > 0).
        signal_amplitude: Amplitude A of the periodic forcing.
        signal_frequency_hz: Frequency of the maneuver window signal (Hz).
        num_noise_samples: Grid resolution for SNR evaluation.

    Returns:
        StochasticResonanceManeuver with resonance parameters.

    Raises:
        ValueError: If parameters are non-positive.
    """
    if potential_a <= 0:
        raise ValueError(f"potential_a must be positive, got {potential_a}")
    if potential_b <= 0:
        raise ValueError(f"potential_b must be positive, got {potential_b}")
    if signal_amplitude <= 0:
        raise ValueError(
            f"signal_amplitude must be positive, got {signal_amplitude}"
        )
    if signal_frequency_hz <= 0:
        raise ValueError(
            f"signal_frequency_hz must be positive, got {signal_frequency_hz}"
        )

    a = potential_a
    b = potential_b

    # Barrier height
    delta_v = a ** 2 / (4.0 * b)

    # Angular frequency at well minimum
    omega_0 = math.sqrt(2.0 * a)

    # Analytical optimal noise: d(SNR)/dD = 0 => D_opt = DeltaV
    d_opt = delta_v

    # Peak SNR at optimal noise
    if d_opt > 0:
        peak_snr = (math.pi * signal_amplitude ** 2) / (4.0 * d_opt) * math.exp(-1.0)
    else:
        peak_snr = 0.0

    # Kramers rate at optimal noise
    kramers_rate = (omega_0 / (2.0 * math.pi)) * math.exp(-delta_v / d_opt)
    resonance_freq = kramers_rate

    # Effectiveness gain: SNR(D_opt) / SNR(D_opt/2)
    d_half = d_opt / 2.0
    if d_half > 0:
        snr_half = (
            (math.pi * signal_amplitude ** 2) / (4.0 * d_half)
            * math.exp(-delta_v / d_half)
        )
        if snr_half > 0:
            effectiveness_gain = peak_snr / snr_half
        else:
            effectiveness_gain = float('inf')
    else:
        effectiveness_gain = 1.0

    return StochasticResonanceManeuver(
        optimal_noise_intensity=d_opt,
        peak_snr=peak_snr,
        resonance_frequency_hz=resonance_freq,
        barrier_height=delta_v,
        kramers_rate=kramers_rate,
        maneuver_effectiveness_gain=effectiveness_gain,
    )
