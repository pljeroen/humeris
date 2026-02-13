# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Competing-risks satellite population dynamics.

Multi-risk survival analysis for satellite constellations. Models the
simultaneous hazards of drag decay, collision, component failure, and
deliberate deorbit as competing risks in a unified framework.

Uses cause-specific hazard functions h_k(t) where the overall survival
is S(t) = exp(-integral sum h_k(t) dt). The cumulative incidence
function CIF_k(t) gives the probability of failure from cause k.

Ref: Prentice et al. (1978) competing risks; adapted for orbital dynamics.

"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskProfile:
    """Hazard rate profile for a single risk."""
    name: str  # e.g., "drag_decay", "collision", "component_failure", "deorbit"
    hazard_rates: tuple  # Hazard rate h_k(t) at each time step (per day)
    is_constant: bool  # Whether hazard rate is time-invariant


@dataclass(frozen=True)
class CompetingRisksResult:
    """Result of competing risks analysis."""
    times_years: tuple  # Time grid (years)
    overall_survival: tuple  # S(t) = P(surviving all risks)
    cause_specific_cif: tuple  # Tuple of (risk_name, CIF_k(t)) — cumulative incidence per cause
    cause_specific_hazard: tuple  # Tuple of (risk_name, h_k(t)) hazard per cause
    dominant_risk_at_time: tuple  # Which risk dominates at each time step
    median_lifetime_years: float  # Time when S(t) = 0.5
    mean_lifetime_years: float  # Integral of S(t)
    risk_attribution: tuple  # Tuple of (risk_name, fraction) — what fraction of failures from each cause
    expected_population: tuple  # Expected number of surviving satellites at each time


@dataclass(frozen=True)
class PopulationProjection:
    """Population dynamics with launch replenishment."""
    times_years: tuple
    active_population: tuple  # Active satellites at each time
    cumulative_launches: tuple  # Total launched by each time
    cumulative_failures: tuple  # Tuple of (risk_name, count(t)) failures per cause
    replacement_rate: tuple  # Required launch rate to maintain target population
    steady_state_population: float  # Equilibrium population
    cost_per_year: tuple  # If launch cost provided: cost at each time


_DAYS_PER_YEAR = 365.25
_SECONDS_PER_DAY = 86400.0
_SECONDS_PER_YEAR = _DAYS_PER_YEAR * _SECONDS_PER_DAY


def _resample_hazard(hazard_rates: np.ndarray, original_n: int,
                     target_n: int) -> np.ndarray:
    """Resample hazard rates to target grid length via linear interpolation."""
    if original_n == target_n:
        return hazard_rates[:target_n]
    original_t = np.linspace(0.0, 1.0, original_n)
    target_t = np.linspace(0.0, 1.0, target_n)
    return np.interp(target_t, original_t, hazard_rates[:original_n])


def create_drag_risk(
    altitude_km: float,
    drag_decay_rate_km_per_year: float,
    reentry_altitude_km: float = 200.0,
) -> RiskProfile:
    """Create drag decay risk profile.

    Hazard rate h_drag(t) = decay_rate / (altitude(t) - reentry_altitude),
    where altitude(t) = altitude_km - decay_rate * t. Time grid covers
    the full interval from initial altitude to reentry.

    Args:
        altitude_km: Initial orbital altitude in km.
        drag_decay_rate_km_per_year: Rate of altitude loss in km/year.
        reentry_altitude_km: Altitude at which reentry occurs (km).

    Returns:
        RiskProfile with time-varying drag hazard.

    """
    if altitude_km <= reentry_altitude_km:
        return RiskProfile(
            name="drag_decay",
            hazard_rates=(1.0,),
            is_constant=False,
        )

    total_years = (altitude_km - reentry_altitude_km) / drag_decay_rate_km_per_year
    n_steps = max(int(np.ceil(total_years)), 2)
    times_years = np.linspace(0.0, total_years, n_steps)

    altitude_t = altitude_km - drag_decay_rate_km_per_year * times_years
    remaining = altitude_t - reentry_altitude_km
    remaining = np.maximum(remaining, 1e-6)  # avoid division by zero

    # h_drag(t) = decay_rate / remaining_altitude (units: 1/year)
    # Convert to per-day for consistency
    hazard_per_year = drag_decay_rate_km_per_year / remaining
    hazard_per_day = hazard_per_year / _DAYS_PER_YEAR

    return RiskProfile(
        name="drag_decay",
        hazard_rates=tuple(hazard_per_day.tolist()),
        is_constant=False,
    )


def create_collision_risk(
    spatial_density_per_km3: float,
    relative_velocity_ms: float,
    collision_cross_section_m2: float = 10.0,
) -> RiskProfile:
    """Create collision risk profile with constant hazard rate.

    h_coll = spatial_density * relative_velocity * cross_section,
    converted to per-day units.

    Args:
        spatial_density_per_km3: Debris/object density in objects/km^3.
        relative_velocity_ms: Mean relative velocity in m/s.
        collision_cross_section_m2: Effective collision cross-section in m^2.

    Returns:
        RiskProfile with constant collision hazard.

    """
    # density: /km^3 → /m^3
    density_per_m3 = spatial_density_per_km3 * 1e-9

    # flux: density * velocity * cross_section → collisions/second
    flux_per_second = density_per_m3 * relative_velocity_ms * collision_cross_section_m2

    # Convert to per-day
    hazard_per_day = flux_per_second * _SECONDS_PER_DAY

    return RiskProfile(
        name="collision",
        hazard_rates=(hazard_per_day,),
        is_constant=True,
    )


def create_component_risk(
    mtbf_years: float,
    wear_factor: float = 0.0,
) -> RiskProfile:
    """Create component failure risk profile.

    h_comp(t) = (1/mtbf) * (1 + wear_factor * t), where t is in years.
    With wear_factor = 0, this is a constant exponential failure rate.
    With wear_factor > 0, the hazard increases over time (Weibull-like).

    The returned hazard rates are in per-day units. For time-varying
    hazards, a grid of 250 steps over 25 years is generated.

    Args:
        mtbf_years: Mean time between failures in years.
        wear_factor: Rate of hazard increase per year (0 = constant).

    Returns:
        RiskProfile with component failure hazard.

    """
    base_rate_per_year = 1.0 / mtbf_years

    if wear_factor == 0.0:
        hazard_per_day = base_rate_per_year / _DAYS_PER_YEAR
        return RiskProfile(
            name="component_failure",
            hazard_rates=(hazard_per_day,),
            is_constant=True,
        )

    # Time-varying: generate over 25-year grid
    duration_years = 25.0
    n_steps = 250
    times = np.linspace(0.0, duration_years, n_steps)
    hazard_per_year = base_rate_per_year * (1.0 + wear_factor * times)
    hazard_per_day = hazard_per_year / _DAYS_PER_YEAR

    return RiskProfile(
        name="component_failure",
        hazard_rates=tuple(hazard_per_day.tolist()),
        is_constant=False,
    )


def create_deorbit_risk(
    planned_lifetime_years: float,
    compliance_probability: float = 0.9,
) -> RiskProfile:
    """Create planned deorbit risk profile.

    Step function: h_deorbit(t) = 0 for t < planned_lifetime, then
    a high rate scaled by compliance_probability. The high rate is
    set to 10/year (i.e., rapid deorbit once mission life is reached).

    Args:
        planned_lifetime_years: Mission design lifetime in years.
        compliance_probability: Probability that deorbit is actually executed.

    Returns:
        RiskProfile with step-function deorbit hazard.

    """
    duration_years = max(planned_lifetime_years * 2.0, 25.0)
    n_steps = max(int(np.ceil(duration_years * 10)), 100)
    times = np.linspace(0.0, duration_years, n_steps)

    # High rate after planned lifetime: 10/year * compliance
    post_mission_rate_per_year = 10.0 * compliance_probability
    hazard_per_year = np.where(
        times < planned_lifetime_years,
        0.0,
        post_mission_rate_per_year,
    )
    hazard_per_day = hazard_per_year / _DAYS_PER_YEAR

    return RiskProfile(
        name="deorbit",
        hazard_rates=tuple(hazard_per_day.tolist()),
        is_constant=False,
    )


def _build_hazard_arrays(
    risks: list,
    n_steps: int,
) -> tuple:
    """Build aligned hazard rate arrays for all risks.

    Each risk's hazard_rates are resampled to the common n_steps grid.
    Constant risks are broadcast to the full grid.

    Returns:
        Tuple of (risk_names, hazard_matrix) where hazard_matrix has
        shape (n_risks, n_steps) with values in per-day units.

    """
    names = []
    rows = []
    for risk in risks:
        names.append(risk.name)
        rates = np.array(risk.hazard_rates, dtype=np.float64)
        if risk.is_constant:
            row = np.full(n_steps, rates[0])
        else:
            row = _resample_hazard(rates, len(rates), n_steps)
        rows.append(row)
    return tuple(names), np.array(rows, dtype=np.float64)


def compute_competing_risks(
    risks: list,
    duration_years: float = 25.0,
    population: int = 1,
    dt_years: float = 0.1,
) -> CompetingRisksResult:
    """Compute competing risks survival analysis.

    Combined hazard: H(t) = sum_k h_k(t).
    Overall survival: S(t) = exp(-integral_0^t H(tau) dtau).
    Cause-specific CIF: CIF_k(t) = integral_0^t h_k(tau) * S(tau) dtau.
    Risk attribution: CIF_k(T_max) / (1 - S(T_max)).

    Uses trapezoidal integration for all integrals.

    Args:
        risks: List of RiskProfile objects.
        duration_years: Analysis duration in years.
        population: Initial number of satellites.
        dt_years: Time step size in years.

    Returns:
        CompetingRisksResult with survival, CIF, and attribution data.

    """
    n_steps = max(int(np.ceil(duration_years / dt_years)) + 1, 2)
    times_years = np.linspace(0.0, duration_years, n_steps)
    dt_days = dt_years * _DAYS_PER_YEAR

    risk_names, hazard_matrix = _build_hazard_arrays(risks, n_steps)

    # Combined hazard H(t) = sum of all cause-specific hazards
    combined_hazard = hazard_matrix.sum(axis=0)

    # Cumulative hazard via trapezoidal integration: Lambda(t) = int_0^t H(tau) dtau
    cumulative_hazard = np.zeros(n_steps)
    for i in range(1, n_steps):
        cumulative_hazard[i] = cumulative_hazard[i - 1] + 0.5 * (
            combined_hazard[i - 1] + combined_hazard[i]
        ) * dt_days

    # Overall survival S(t) = exp(-Lambda(t))
    survival = np.exp(-cumulative_hazard)

    # Cause-specific CIF using discrete decomposition to ensure
    # sum(CIF_k) = 1 - S(t) exactly.
    # At each step, total failure increment = S(t_{i-1}) - S(t_i).
    # Cause k gets fraction h_k / H (midpoint) of that increment.
    cause_cifs = [np.zeros(n_steps) for _ in range(len(risks))]
    for i in range(1, n_steps):
        delta_failure = survival[i - 1] - survival[i]
        h_mid = 0.5 * (combined_hazard[i - 1] + combined_hazard[i])
        if h_mid > 0.0:
            for k in range(len(risks)):
                h_k_mid = 0.5 * (hazard_matrix[k, i - 1] + hazard_matrix[k, i])
                cause_cifs[k][i] = cause_cifs[k][i - 1] + delta_failure * h_k_mid / h_mid
        else:
            for k in range(len(risks)):
                cause_cifs[k][i] = cause_cifs[k][i - 1]

    # Cause-specific hazard tuples
    cause_specific_hazard = tuple(
        (risk_names[k], tuple(hazard_matrix[k].tolist()))
        for k in range(len(risks))
    )

    # CIF tuples
    cause_specific_cif = tuple(
        (risk_names[k], tuple(cause_cifs[k].tolist()))
        for k in range(len(risks))
    )

    # Dominant risk at each time step: argmax h_k(t)
    dominant_indices = np.argmax(hazard_matrix, axis=0)
    dominant_risk = tuple(risk_names[i] for i in dominant_indices)

    # Median lifetime: interpolate when S(t) crosses 0.5
    median_lifetime = duration_years  # default if S never drops to 0.5
    below_half = np.where(survival <= 0.5)[0]
    if len(below_half) > 0:
        idx = below_half[0]
        if idx > 0:
            # Linear interpolation between idx-1 and idx
            s0 = survival[idx - 1]
            s1 = survival[idx]
            t0 = times_years[idx - 1]
            t1 = times_years[idx]
            if s0 != s1:
                median_lifetime = t0 + (0.5 - s0) * (t1 - t0) / (s1 - s0)
            else:
                median_lifetime = t0
        else:
            median_lifetime = times_years[0]

    # Mean lifetime: trapezoidal integral of S(t)
    mean_lifetime = float(np.trapezoid(survival, times_years))

    # Risk attribution: CIF_k(T_max) / (1 - S(T_max))
    total_failure_prob = 1.0 - survival[-1]
    if total_failure_prob > 1e-15:
        attribution = tuple(
            (risk_names[k], float(cause_cifs[k][-1] / total_failure_prob))
            for k in range(len(risks))
        )
    else:
        # No failures occurred — equal attribution
        equal = 1.0 / max(len(risks), 1)
        attribution = tuple(
            (risk_names[k], equal) for k in range(len(risks))
        )

    # Expected population
    expected_pop = tuple((population * survival).tolist())

    return CompetingRisksResult(
        times_years=tuple(times_years.tolist()),
        overall_survival=tuple(survival.tolist()),
        cause_specific_cif=cause_specific_cif,
        cause_specific_hazard=cause_specific_hazard,
        dominant_risk_at_time=dominant_risk,
        median_lifetime_years=float(median_lifetime),
        mean_lifetime_years=float(mean_lifetime),
        risk_attribution=attribution,
        expected_population=expected_pop,
    )


def project_population(
    risks: list,
    initial_population: int,
    launch_rate_per_year: float = 0.0,
    target_population: int = 0,
    duration_years: float = 25.0,
    dt_years: float = 0.1,
    cost_per_launch: float = 0.0,
) -> PopulationProjection:
    """Project population dynamics with launch replenishment.

    At each time step, the population decays according to the competing
    risks survival function and new satellites are launched either at a
    fixed rate or to maintain a target population.

    Args:
        risks: List of RiskProfile objects.
        initial_population: Starting number of active satellites.
        launch_rate_per_year: Fixed launch rate (satellites per year).
        target_population: If > 0, compute required launch rate to
            maintain this population level.
        duration_years: Projection duration in years.
        dt_years: Time step in years.
        cost_per_launch: Cost per satellite launch (monetary units).

    Returns:
        PopulationProjection with population trajectory and costs.

    """
    n_steps = max(int(np.ceil(duration_years / dt_years)) + 1, 2)
    times_years = np.linspace(0.0, duration_years, n_steps)
    dt_days = dt_years * _DAYS_PER_YEAR

    risk_names, hazard_matrix = _build_hazard_arrays(risks, n_steps)
    combined_hazard = hazard_matrix.sum(axis=0)

    # Per-step survival fraction: P(survive this step) = exp(-H(t) * dt)
    step_survival = np.exp(-combined_hazard * dt_days)

    # Cause-specific failure fraction per step: proportional to hazard
    # fraction of combined hazard
    hazard_fractions = np.zeros_like(hazard_matrix)
    nonzero = combined_hazard > 0
    for k in range(len(risks)):
        hazard_fractions[k, nonzero] = (
            hazard_matrix[k, nonzero] / combined_hazard[nonzero]
        )

    # Population simulation
    active = np.zeros(n_steps)
    active[0] = float(initial_population)
    cum_launches = np.zeros(n_steps)
    cum_failures = np.zeros((len(risks), n_steps))
    replacement_rate = np.zeros(n_steps)
    cost_series = np.zeros(n_steps)

    for i in range(1, n_steps):
        # Survivors from previous step
        survivors = active[i - 1] * step_survival[i]
        step_failures = active[i - 1] - survivors

        # Allocate failures to causes
        for k in range(len(risks)):
            cause_failures = step_failures * hazard_fractions[k, i]
            cum_failures[k, i] = cum_failures[k, i - 1] + cause_failures

        # Launches this step
        if target_population > 0:
            # Launch enough to reach target
            deficit = max(0.0, target_population - survivors)
            launches_this_step = deficit
            replacement_rate[i] = deficit / dt_years
        else:
            launches_this_step = launch_rate_per_year * dt_years
            replacement_rate[i] = launch_rate_per_year

        active[i] = survivors + launches_this_step
        cum_launches[i] = cum_launches[i - 1] + launches_this_step
        cost_series[i] = launches_this_step * cost_per_launch / dt_years

    # Steady state population: launch_rate / hazard_rate (for constant hazard)
    mean_hazard_per_year = float(combined_hazard.mean()) * _DAYS_PER_YEAR
    effective_launch_rate = launch_rate_per_year
    if target_population > 0:
        # In maintenance mode, effective launch rate is the mean replacement rate
        effective_launch_rate = float(replacement_rate[1:].mean()) if n_steps > 1 else 0.0
    if mean_hazard_per_year > 1e-15:
        steady_state = effective_launch_rate / mean_hazard_per_year
    else:
        steady_state = float(initial_population)

    # Build cumulative failure tuples
    cumulative_failures = tuple(
        (risk_names[k], tuple(cum_failures[k].tolist()))
        for k in range(len(risks))
    )

    return PopulationProjection(
        times_years=tuple(times_years.tolist()),
        active_population=tuple(active.tolist()),
        cumulative_launches=tuple(cum_launches.tolist()),
        cumulative_failures=cumulative_failures,
        replacement_rate=tuple(replacement_rate.tolist()),
        steady_state_population=float(steady_state),
        cost_per_year=tuple(cost_series.tolist()),
    )


def compute_risk_sensitivity(
    base_risks: list,
    risk_index: int,
    multipliers: tuple,
    duration_years: float = 25.0,
) -> tuple:
    """Sensitivity analysis: how does varying one risk affect outcomes?

    Scales the hazard rates of the risk at risk_index by each multiplier
    and re-runs the competing risks analysis.

    Args:
        base_risks: List of RiskProfile objects.
        risk_index: Index of the risk to vary.
        multipliers: Tuple of scaling factors to apply.
        duration_years: Analysis duration in years.

    Returns:
        Tuple of (multiplier, median_lifetime, dominant_risk_at_end,
        risk_attribution) for each multiplier value.

    """
    results = []
    for mult in multipliers:
        modified_risks = []
        for i, risk in enumerate(base_risks):
            if i == risk_index:
                scaled_rates = tuple(
                    r * mult for r in risk.hazard_rates
                )
                modified_risks.append(RiskProfile(
                    name=risk.name,
                    hazard_rates=scaled_rates,
                    is_constant=risk.is_constant,
                ))
            else:
                modified_risks.append(risk)

        cr = compute_competing_risks(modified_risks, duration_years=duration_years)
        results.append((
            float(mult),
            cr.median_lifetime_years,
            cr.dominant_risk_at_time[-1],
            cr.risk_attribution,
        ))

    return tuple(results)
