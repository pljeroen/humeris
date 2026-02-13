# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Failure cascade prediction.

Composes DFT of Fiedler value time series with hazard rate peakiness
from orbit lifetime survival curves to predict cascading failures.

"""
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from humeris.domain.propagation import OrbitalState
from humeris.domain.link_budget import LinkConfig
from humeris.domain.atmosphere import DragConfig
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.graph_analysis import compute_fragmentation_timeline
from humeris.domain.linalg import naive_dft
from humeris.domain.lifetime import compute_orbit_lifetime
from humeris.domain.statistical_analysis import compute_lifetime_survival_curve


@dataclass(frozen=True)
class CascadeIndicator:
    """Failure cascade risk indicator."""
    spectral_power_at_orbital: float
    hazard_peakiness: float
    cascade_indicator: float
    fiedler_dominant_freq_hz: float
    orbital_frequency_hz: float
    is_cascade_risk: bool


def compute_cascade_indicator(
    states: list,
    link_config: LinkConfig,
    epoch: datetime,
    drag_config: DragConfig,
    fragmentation_duration_s: float = 5400.0,
    fragmentation_step_s: float = 300.0,
    max_range_km: float = 5000.0,
    eclipse_power_fraction: float = 0.5,
) -> CascadeIndicator:
    """Compute failure cascade indicator.

    CI = spectral_power(fiedler, f_orbital) * max(hazard) / mean(hazard)

    High CI means eclipse-driven connectivity oscillations coincide
    with high hazard-rate regimes.
    """
    if not states:
        return CascadeIndicator(
            spectral_power_at_orbital=0.0, hazard_peakiness=1.0,
            cascade_indicator=0.0, fiedler_dominant_freq_hz=0.0,
            orbital_frequency_hz=0.0, is_cascade_risk=False,
        )

    # Fragmentation timeline → DFT
    timeline = compute_fragmentation_timeline(
        states, link_config, epoch,
        fragmentation_duration_s, fragmentation_step_s,
        max_range_km=max_range_km,
        eclipse_power_fraction=eclipse_power_fraction,
    )

    fiedler_signal = [e.fiedler_value for e in timeline.events]
    sample_rate = 1.0 / fragmentation_step_s
    dft_result = naive_dft(fiedler_signal, sample_rate)

    # Orbital frequency
    orbital_freq = states[0].mean_motion_rad_s / (2.0 * np.pi)

    # Find spectral power at/near orbital frequency
    spectral_at_orbital = 0.0
    half_n = max(1, len(fiedler_signal) // 2)
    for k in range(1, half_n):
        freq = dft_result.frequencies_hz[k]
        if abs(freq - orbital_freq) < orbital_freq * 0.2:
            mag = dft_result.magnitudes[k]
            spectral_at_orbital = max(spectral_at_orbital, mag * mag)

    # Find dominant frequency
    if half_n > 1:
        peak_idx = max(range(1, half_n),
                       key=lambda k: dft_result.magnitudes[k])
        dom_freq = dft_result.frequencies_hz[peak_idx]
    else:
        dom_freq = 0.0

    # Hazard rate peakiness from survival curve
    ref_state = states[0]
    a_m = ref_state.semi_major_axis_m

    try:
        lifetime = compute_orbit_lifetime(
            a_m, 0.0, drag_config, epoch,
            step_days=10.0, max_years=25.0,
        )
        survival = compute_lifetime_survival_curve(lifetime)
        hazard_rates = [h for h in survival.hazard_rate_per_day if h > 0]
    except (ValueError, ZeroDivisionError):
        hazard_rates = []

    if hazard_rates:
        mean_hazard = sum(hazard_rates) / len(hazard_rates)
        max_hazard = max(hazard_rates)
        if mean_hazard > 1e-20:
            peakiness = max_hazard / mean_hazard
        else:
            peakiness = 1.0
    else:
        peakiness = 1.0

    ci = spectral_at_orbital * peakiness
    is_risk = ci > 0.01

    return CascadeIndicator(
        spectral_power_at_orbital=spectral_at_orbital,
        hazard_peakiness=peakiness,
        cascade_indicator=ci,
        fiedler_dominant_freq_hz=dom_freq,
        orbital_frequency_hz=orbital_freq,
        is_cascade_risk=is_risk,
    )


# ── SIR Epidemic Model for Debris Cascade ──────────────────────────

_SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass(frozen=True)
class CascadeSIR:
    """SIR epidemic model results for debris cascade dynamics."""
    r_0: float                    # Basic reproduction number
    time_to_peak_years: float     # Time to peak debris population
    equilibrium_debris: float     # Steady-state debris count (0 if subcritical)
    is_supercritical: bool        # R_0 > 1
    time_series_years: tuple      # Time points
    susceptible: tuple            # S(t) — intact satellites
    infected: tuple               # I(t) — active debris
    recovered: tuple              # R(t) — deorbited debris


def compute_cascade_sir(
    shell_volume_km3: float,
    spatial_density_per_km3: float,
    mean_collision_velocity_ms: float,
    satellite_count: int,
    launch_rate_per_year: float = 0.0,
    fragments_per_collision: float = 100.0,
    drag_lifetime_years: float = 25.0,
    collision_cross_section_km2: float = 1e-5,
    duration_years: float = 100.0,
    step_years: float = 0.1,
) -> CascadeSIR:
    """Compute SIR epidemic model for debris cascade dynamics.

    Models debris cascade as an epidemic: intact satellites are susceptible,
    debris fragments are infected, deorbited fragments are recovered.

    Uses forward Euler integration.

    Args:
        shell_volume_km3: Volume of the orbital shell (km^3).
        spatial_density_per_km3: Current debris spatial density (objects/km^3).
        mean_collision_velocity_ms: Mean relative collision velocity (m/s).
        satellite_count: Number of intact satellites (S_0).
        launch_rate_per_year: New satellite launches per year (replenishment).
        fragments_per_collision: Average fragments generated per collision.
        drag_lifetime_years: Mean drag lifetime for debris (years).
        collision_cross_section_km2: Effective collision cross-section (km^2).
        duration_years: Simulation duration (years).
        step_years: Integration time step (years).

    Returns:
        CascadeSIR with R_0, time series, and equilibrium analysis.
    """
    # Compute beta: collision rate per object per year
    # Convert velocity from m/s to km/s, multiply by seconds_per_year
    velocity_km_s = mean_collision_velocity_ms * 0.001
    beta = (collision_cross_section_km2 * velocity_km_s
            * _SECONDS_PER_YEAR / shell_volume_km3)

    # Recovery rate
    gamma = 1.0 / drag_lifetime_years

    # Basic reproduction number
    r_0 = fragments_per_collision * beta * satellite_count / gamma

    is_supercritical = r_0 > 1.0

    # Initial conditions
    s_0 = float(satellite_count)
    i_0 = spatial_density_per_km3 * shell_volume_km3  # existing debris count
    recovered_count = 0.0

    # Time to peak (exponential growth phase estimate)
    if is_supercritical and i_0 > 0:
        growth_rate = beta * s_0 * fragments_per_collision - gamma
        if growth_rate > 0 and i_0 > 0:
            time_to_peak = math.log(s_0 / i_0) / growth_rate
        else:
            time_to_peak = 0.0
    else:
        time_to_peak = 0.0

    # Equilibrium debris
    if is_supercritical and launch_rate_per_year > 0 and s_0 > 0:
        equilibrium_debris = (launch_rate_per_year * (r_0 - 1.0)
                              / (beta * fragments_per_collision * s_0))
    else:
        equilibrium_debris = 0.0

    # Forward Euler integration
    n_steps = int(duration_years / step_years) + 1
    dt = step_years

    time_arr = []
    s_arr = []
    i_arr = []
    r_arr = []

    s = s_0
    i = i_0
    r = recovered_count

    for k in range(n_steps):
        t = k * dt
        time_arr.append(t)
        s_arr.append(s)
        i_arr.append(i)
        r_arr.append(r)

        # Forward Euler stability check (N-04):
        # If beta * s * fragments_per_collision * dt > 1.0, the explicit Euler
        # step can become unstable. Use adaptive sub-stepping in that case.
        growth_metric = beta * s * fragments_per_collision * dt
        if growth_metric > 1.0:
            n_sub = int(math.ceil(growth_metric * 2.0))
            sub_dt = dt / n_sub
        else:
            n_sub = 1
            sub_dt = dt

        for _sub in range(n_sub):
            # Derivatives
            ds_dt = launch_rate_per_year - beta * s * i
            di_dt = fragments_per_collision * beta * s * i - gamma * i
            dr_dt = gamma * i

            # Euler step
            s = s + ds_dt * sub_dt
            i = i + di_dt * sub_dt
            r = r + dr_dt * sub_dt

            # Clamp to non-negative (physical constraint)
            s = max(s, 0.0)
            i = max(i, 0.0)
            r = max(r, 0.0)

    return CascadeSIR(
        r_0=r_0,
        time_to_peak_years=time_to_peak,
        equilibrium_debris=equilibrium_debris,
        is_supercritical=is_supercritical,
        time_series_years=tuple(time_arr),
        susceptible=tuple(s_arr),
        infected=tuple(i_arr),
        recovered=tuple(r_arr),
    )


# ── Birth-Death Chain for Debris Population Dynamics ────────────────
#
# Discrete stochastic model: collisions birth N fragments, drag removes
# them. Provides full probability distribution of debris count, not just
# the mean trajectory. See SESSION_MINING_R1_CREATIVE.md P11.


@dataclass(frozen=True)
class BirthDeathDebris:
    """Birth-death chain results for debris population dynamics.

    Attributes:
        mean_equilibrium_debris: Expected debris count at equilibrium.
        variance_equilibrium_debris: Variance of debris count at equilibrium.
        extinction_probability: Probability that debris eventually reaches 0.
        time_to_threshold_years: Time until P(N > threshold) > 0.5.
        is_supercritical: True if birth rate exceeds death rate (no equilibrium).
        stationary_distribution: P(N=k) for k = 0, 1, ..., max_n.
        birth_rate: Effective birth rate per debris item per year.
        death_rate: Drag removal rate per debris item per year.
        threshold_exceedance_probability: P(N > 2 * current_debris) at equilibrium.
    """
    mean_equilibrium_debris: float
    variance_equilibrium_debris: float
    extinction_probability: float
    time_to_threshold_years: float
    is_supercritical: bool
    stationary_distribution: tuple
    birth_rate: float
    death_rate: float
    threshold_exceedance_probability: float


def compute_debris_birth_death(
    shell_volume_km3: float,
    spatial_density_per_km3: float,
    mean_collision_velocity_ms: float,
    satellite_count: int,
    fragments_per_collision: float = 100.0,
    drag_lifetime_years: float = 25.0,
    collision_cross_section_km2: float = 1e-5,
    max_debris_count: int = 500,
) -> BirthDeathDebris:
    """Compute birth-death chain model for debris population.

    Birth rate: lambda_n = beta * n * S * N_frag (collision rate proportional
    to debris count times satellite count times fragments per collision).
    Death rate: mu_n = gamma * n (drag removal proportional to debris count).

    When birth/death ratio < 1 (subcritical), the stationary distribution
    is Poisson with parameter lambda/mu. When supercritical, no stationary
    distribution exists and population grows without bound.

    Args:
        shell_volume_km3: Volume of the orbital shell (km^3).
        spatial_density_per_km3: Current debris spatial density (objects/km^3).
        mean_collision_velocity_ms: Mean relative collision velocity (m/s).
        satellite_count: Number of intact satellites.
        fragments_per_collision: Average fragments per collision.
        drag_lifetime_years: Mean drag lifetime for debris (years).
        collision_cross_section_km2: Effective collision cross-section (km^2).
        max_debris_count: Maximum debris count for distribution truncation.

    Returns:
        BirthDeathDebris with equilibrium statistics and distribution.
    """
    # Compute collision rate coefficient beta (collisions per pair per year)
    velocity_km_s = mean_collision_velocity_ms * 0.001
    beta = (collision_cross_section_km2 * velocity_km_s
            * _SECONDS_PER_YEAR / shell_volume_km3)

    # Death rate per debris item
    gamma = 1.0 / drag_lifetime_years

    # Effective birth rate per debris item (net fragments entering population)
    # A debris item collides with S satellites at rate beta*S; each collision
    # creates N_frag new fragments.
    effective_birth_rate = beta * satellite_count * fragments_per_collision

    # Birth/death ratio (rho = lambda / mu per item)
    rho = effective_birth_rate / gamma if gamma > 0 else float("inf")
    is_supercritical = rho > 1.0

    # Current debris count
    current_debris = spatial_density_per_km3 * shell_volume_km3

    # Stationary distribution: for subcritical birth-death chain (rho < 1),
    # the quasi-stationary distribution is geometric with parameter rho.
    # Mean = 1/(1 - rho), Variance = rho / (1 - rho)^2.
    if not is_supercritical and rho > 0:
        mean_eq = 1.0 / (1.0 - rho)
        var_eq = rho / (1.0 - rho) ** 2

        # Compute truncated geometric distribution: P(N=k) = (1-rho)*rho^k
        max_n = min(max_debris_count, max(int(mean_eq * 5) + 10, 20))
        dist = np.zeros(max_n + 1)
        for k in range(max_n + 1):
            dist[k] = (1.0 - rho) * rho ** k
        # Normalize truncated distribution
        total = float(np.sum(dist))
        if total > 0:
            dist = dist / total

        # P(N > 2 * current_debris)
        threshold_idx = min(int(2 * current_debris), max_n)
        if threshold_idx < max_n:
            threshold_exceed = float(np.sum(dist[threshold_idx + 1:]))
        else:
            threshold_exceed = 0.0

        # Extinction probability: for subcritical birth-death, P(extinction) = 1
        extinction_prob = 1.0

        # Time to threshold: rare event estimate
        if threshold_exceed > 0.5:
            time_to_threshold = 0.0
        elif threshold_exceed > 1e-20:
            time_to_threshold = -math.log(threshold_exceed) / gamma
        else:
            time_to_threshold = float("inf")

    elif is_supercritical:
        # Supercritical: no stationary distribution, exponential growth
        growth_rate = effective_birth_rate - gamma
        mean_eq = float("inf")
        var_eq = float("inf")

        # Extinction probability: (gamma / effective_birth_rate)^N_0
        n_0 = max(1, int(current_debris))
        extinction_prob = (gamma / effective_birth_rate) ** min(n_0, 100)

        # Time to double: T ~ ln(2) / growth_rate
        if growth_rate > 0:
            time_to_threshold = math.log(2.0) / growth_rate
        else:
            time_to_threshold = float("inf")

        dist = np.array([0.0])
        threshold_exceed = 1.0

    else:
        # rho == 0: no collisions, debris only decays
        mean_eq = 0.0
        var_eq = 0.0
        extinction_prob = 1.0
        time_to_threshold = float("inf")
        dist = np.array([1.0])
        threshold_exceed = 0.0

    return BirthDeathDebris(
        mean_equilibrium_debris=mean_eq,
        variance_equilibrium_debris=var_eq,
        extinction_probability=extinction_prob,
        time_to_threshold_years=time_to_threshold,
        is_supercritical=is_supercritical,
        stationary_distribution=tuple(float(x) for x in dist),
        birth_rate=effective_birth_rate,
        death_rate=gamma,
        threshold_exceedance_probability=threshold_exceed,
    )


# ── Lotka-Volterra Multi-Species Debris Model (P24) ────────────────
#
# 3 species: rocket bodies (N1), mission debris (N2), fragments (N3).
# Competitive Lotka-Volterra with collision coupling and drag removal.
# See SESSION_MINING_R2_CREATIVE.md P24.


@dataclass(frozen=True)
class LotkaVolterraDebris:
    """Lotka-Volterra multi-species debris model results.

    Attributes:
        species: Names of the three species.
        equilibrium_counts: Equilibrium population of each species (last state).
        jacobian_eigenvalues: Eigenvalues of the Jacobian at equilibrium.
        is_stable: True if all eigenvalue real parts are negative.
        dominant_interaction: Description of the strongest inter-species coupling.
        populations: Tuple of 3 tuples: (N1(t), N2(t), N3(t)) time series.
    """
    species: tuple
    equilibrium_counts: tuple
    jacobian_eigenvalues: tuple
    is_stable: bool
    dominant_interaction: str
    populations: tuple


def compute_lotka_volterra_debris(
    initial_rocket_bodies: float,
    initial_mission_debris: float,
    initial_fragments: float,
    gamma: np.ndarray | None = None,
    collision_rates: np.ndarray | None = None,
    fragment_multipliers: np.ndarray | None = None,
    launch_rates: np.ndarray | None = None,
    duration_years: float = 50.0,
    step_years: float = 0.1,
) -> LotkaVolterraDebris:
    """Compute Lotka-Volterra multi-species debris dynamics.

    Models three debris species interacting through collisions:
        dN1/dt = -gamma1*N1 + L1 - sum_j a1j*N1*Nj
        dN2/dt = -gamma2*N2 + L2 - sum_j a2j*N2*Nj
        dN3/dt = -gamma3*N3 + sum_ij fij*aij*Ni*Nj

    Species: rocket bodies (N1), mission debris (N2), fragments (N3).

    Args:
        initial_rocket_bodies: Initial rocket body count (N1_0).
        initial_mission_debris: Initial mission debris count (N2_0).
        initial_fragments: Initial fragment count (N3_0).
        gamma: Drag removal rates [gamma1, gamma2, gamma3] (1/year).
            Default: [0.04, 0.05, 0.02] (25yr, 20yr, 50yr lifetimes).
        collision_rates: 3x3 collision rate coefficient matrix a_ij (1/year).
            Default: weak coupling.
        fragment_multipliers: 3x3 fragment multiplier matrix f_ij.
            Default: moderate fragmentation.
        launch_rates: Launch rates [L1, L2, L3] (objects/year).
            Default: [10.0, 20.0, 0.0].
        duration_years: Simulation duration (years).
        step_years: Integration time step (years).

    Returns:
        LotkaVolterraDebris with populations, equilibrium, and stability.
    """
    species_names = ("rocket_bodies", "mission_debris", "fragments")

    # Defaults
    if gamma is None:
        gamma = np.array([0.04, 0.05, 0.02])
    if collision_rates is None:
        collision_rates = np.array([
            [0.0, 1e-5, 1e-6],
            [1e-5, 0.0, 1e-6],
            [1e-6, 1e-6, 0.0],
        ])
    if fragment_multipliers is None:
        fragment_multipliers = np.array([
            [0.0, 50.0, 20.0],
            [50.0, 0.0, 20.0],
            [20.0, 20.0, 0.0],
        ])
    if launch_rates is None:
        launch_rates = np.array([10.0, 20.0, 0.0])

    n_steps = int(duration_years / step_years) + 1
    dt = step_years

    # Population arrays
    n = np.array([initial_rocket_bodies, initial_mission_debris, initial_fragments])
    history = [[], [], []]
    for k in range(3):
        history[k].append(n[k])

    # Forward Euler integration
    for _ in range(n_steps - 1):
        dn = np.zeros(3)

        # Species 0 (rocket bodies): dN1/dt = -gamma1*N1 + L1 - sum_j a1j*N1*Nj
        dn[0] = -gamma[0] * n[0] + launch_rates[0]
        for j in range(3):
            dn[0] -= collision_rates[0, j] * n[0] * n[j]

        # Species 1 (mission debris): dN2/dt = -gamma2*N2 + L2 - sum_j a2j*N2*Nj
        dn[1] = -gamma[1] * n[1] + launch_rates[1]
        for j in range(3):
            dn[1] -= collision_rates[1, j] * n[1] * n[j]

        # Species 2 (fragments): dN3/dt = -gamma3*N3 + sum_ij fij*aij*Ni*Nj
        dn[2] = -gamma[2] * n[2]
        for i in range(3):
            for j in range(3):
                dn[2] += fragment_multipliers[i, j] * collision_rates[i, j] * n[i] * n[j]

        n = n + dn * dt
        # Clamp to non-negative
        n = np.maximum(n, 0.0)

        for k in range(3):
            history[k].append(float(n[k]))

    # Equilibrium: use last state
    equilibrium = tuple(float(n[k]) for k in range(3))

    # Jacobian at equilibrium
    # J_ij = d(dNi/dt)/dNj evaluated at equilibrium
    ne = n.copy()
    jac = np.zeros((3, 3))

    # For species 0:
    # dN1/dt = -gamma1*N1 + L1 - sum_j a1j*N1*Nj
    # d/dN1 = -gamma1 - sum_j a1j*Nj - a11*N1 (self-interaction doubled)
    jac[0, 0] = -gamma[0] - sum(collision_rates[0, j] * ne[j] for j in range(3)) - collision_rates[0, 0] * ne[0]
    for j in range(1, 3):
        jac[0, j] = -collision_rates[0, j] * ne[0]

    # For species 1:
    jac[1, 1] = -gamma[1] - sum(collision_rates[1, j] * ne[j] for j in range(3)) - collision_rates[1, 1] * ne[1]
    for j in [0, 2]:
        jac[1, j] = -collision_rates[1, j] * ne[1]

    # For species 2:
    # dN3/dt = -gamma3*N3 + sum_ij fij*aij*Ni*Nj
    jac[2, 2] = -gamma[2]
    for i in range(3):
        jac[2, 2] += fragment_multipliers[i, 2] * collision_rates[i, 2] * ne[i]
        jac[2, 2] += fragment_multipliers[2, i] * collision_rates[2, i] * ne[i]
    for j in range(3):
        if j != 2:
            # d/dN_j of sum_{i,k} f[i,k]*a[i,k]*N_i*N_k
            # = sum_i f[i,j]*a[i,j]*N_i + sum_k f[j,k]*a[j,k]*N_k
            for i in range(3):
                jac[2, j] += fragment_multipliers[i, j] * collision_rates[i, j] * ne[i]
            for k in range(3):
                jac[2, j] += fragment_multipliers[j, k] * collision_rates[j, k] * ne[k]

    eigenvalues = np.linalg.eigvals(jac)
    is_stable = bool(np.all(np.real(eigenvalues) < 1e-10))

    # Dominant interaction: find largest collision flux a_ij * Ni * Nj
    max_flux = 0.0
    dominant = "none"
    for i in range(3):
        for j in range(3):
            flux = collision_rates[i, j] * ne[i] * ne[j]
            if flux > max_flux:
                max_flux = flux
                dominant = f"{species_names[i]}-{species_names[j]}"

    return LotkaVolterraDebris(
        species=species_names,
        equilibrium_counts=equilibrium,
        jacobian_eigenvalues=tuple(complex(ev) for ev in eigenvalues),
        is_stable=is_stable,
        dominant_interaction=dominant,
        populations=tuple(tuple(h) for h in history),
    )


# ── P58: Coupled Debris-Operations Feedback ────────────────────────
#
# 3-ODE system coupling satellite operations, debris population, and
# propellant consumption. Hill function for maneuver-induced death rate.
# Forward Euler integration + Jacobian stability analysis.


@dataclass(frozen=True)
class CoupledDebrisOperations:
    """Coupled debris-operations feedback model results.

    Attributes:
        equilibrium_satellites: Satellite count at equilibrium.
        equilibrium_debris: Debris count at equilibrium.
        equilibrium_propellant_rate: Propellant consumption rate at equilibrium (kg/year).
        jacobian_eigenvalues: Eigenvalues of Jacobian at final state.
        is_stable: True if all eigenvalue real parts are negative.
        oscillation_period_years: Period from imaginary parts (inf if no oscillation).
        populations: (time, N_sat, N_debris, P_rate) time series.
    """
    equilibrium_satellites: float
    equilibrium_debris: float
    equilibrium_propellant_rate: float
    jacobian_eigenvalues: tuple
    is_stable: bool
    oscillation_period_years: float
    populations: tuple


def compute_coupled_debris_operations(
    initial_satellites: float = 1000.0,
    initial_debris: float = 5000.0,
    initial_propellant_rate: float = 50.0,
    launch_rate: float = 100.0,
    natural_death_rate: float = 0.04,
    collision_rate: float = 1e-5,
    fragments_per_collision: float = 100.0,
    drag_removal_rate: float = 0.04,
    mu_0: float = 0.1,
    k_m: float = 10000.0,
    propellant_per_maneuver: float = 0.5,
    propellant_recovery_rate: float = 0.1,
    duration_years: float = 100.0,
    step_years: float = 0.1,
) -> CoupledDebrisOperations:
    """Compute coupled debris-operations feedback model.

    Three-ODE system:
        dN_sat/dt = L - mu_nat*N_sat - mu(N_d)*N_sat - alpha*N_sat*N_d
        dN_d/dt = alpha*f*N_sat*N_d + mu(N_d)*N_sat*f_man - gamma*N_d
        dP/dt = mu(N_d)*N_sat*C_man - kappa*P

    where:
        mu(N_d) = mu_0 * N_d^2 / (N_d^2 + K_m^2)  [Hill function]
            Models maneuver-induced death rate that saturates at high debris.

    Args:
        initial_satellites: Initial active satellite count.
        initial_debris: Initial debris count.
        initial_propellant_rate: Initial propellant consumption rate (kg/year).
        launch_rate: Satellite launch rate (satellites/year).
        natural_death_rate: Natural satellite death rate (1/year).
        collision_rate: Collision rate coefficient (1/(object*year)).
        fragments_per_collision: Fragments per collision event.
        drag_removal_rate: Debris drag removal rate (1/year).
        mu_0: Maximum maneuver-induced death rate (1/year).
        k_m: Hill function half-saturation debris count.
        propellant_per_maneuver: Propellant per maneuver event (kg).
        propellant_recovery_rate: Propellant rate decay constant (1/year).
        duration_years: Simulation duration (years).
        step_years: Integration time step (years).

    Returns:
        CoupledDebrisOperations with populations, equilibrium, and stability.

    Raises:
        ValueError: If initial counts or rates are negative.
    """
    if initial_satellites < 0 or initial_debris < 0 or initial_propellant_rate < 0:
        raise ValueError("Initial values must be non-negative")

    n_steps = int(duration_years / step_years) + 1
    dt = step_years

    # State: [N_sat, N_debris, P_rate]
    n_sat = initial_satellites
    n_d = initial_debris
    p_rate = initial_propellant_rate

    times = []
    sat_hist = []
    deb_hist = []
    prop_hist = []

    for k in range(n_steps):
        t = k * dt
        times.append(t)
        sat_hist.append(n_sat)
        deb_hist.append(n_d)
        prop_hist.append(p_rate)

        # Hill function: maneuver-induced death rate
        mu_man = mu_0 * n_d ** 2 / (n_d ** 2 + k_m ** 2)

        # Derivatives
        dn_sat = launch_rate - natural_death_rate * n_sat - mu_man * n_sat - collision_rate * n_sat * n_d
        dn_d = (collision_rate * fragments_per_collision * n_sat * n_d
                + mu_man * n_sat * 0.1  # maneuver debris fraction
                - drag_removal_rate * n_d)
        dp_rate = mu_man * n_sat * propellant_per_maneuver - propellant_recovery_rate * p_rate

        # Forward Euler
        n_sat = max(n_sat + dn_sat * dt, 0.0)
        n_d = max(n_d + dn_d * dt, 0.0)
        p_rate = max(p_rate + dp_rate * dt, 0.0)

    # Jacobian at final state
    ns, nd, pr = n_sat, n_d, p_rate

    # Partial derivatives of Hill function
    # mu(N_d) = mu_0 * N_d^2 / (N_d^2 + K_m^2)
    # dmu/dN_d = mu_0 * 2*N_d*K_m^2 / (N_d^2 + K_m^2)^2
    denom = nd ** 2 + k_m ** 2
    mu_val = mu_0 * nd ** 2 / denom if denom > 0 else 0.0
    dmu_dnd = mu_0 * 2.0 * nd * k_m ** 2 / (denom ** 2) if denom > 0 else 0.0

    jac = np.zeros((3, 3))

    # d(dN_sat)/dN_sat
    jac[0, 0] = -natural_death_rate - mu_val - collision_rate * nd
    # d(dN_sat)/dN_d
    jac[0, 1] = -dmu_dnd * ns - collision_rate * ns
    # d(dN_sat)/dP
    jac[0, 2] = 0.0

    # d(dN_d)/dN_sat
    jac[1, 0] = collision_rate * fragments_per_collision * nd + mu_val * 0.1
    # d(dN_d)/dN_d
    jac[1, 1] = collision_rate * fragments_per_collision * ns + dmu_dnd * ns * 0.1 - drag_removal_rate
    # d(dN_d)/dP
    jac[1, 2] = 0.0

    # d(dP)/dN_sat
    jac[2, 0] = mu_val * propellant_per_maneuver
    # d(dP)/dN_d
    jac[2, 1] = dmu_dnd * ns * propellant_per_maneuver
    # d(dP)/dP
    jac[2, 2] = -propellant_recovery_rate

    eigenvalues = np.linalg.eigvals(jac)
    is_stable = bool(np.all(np.real(eigenvalues) < 1e-10))

    # Detect oscillation period from imaginary parts
    imag_parts = np.abs(np.imag(eigenvalues))
    max_imag = float(np.max(imag_parts))
    if max_imag > 1e-10:
        oscillation_period = 2.0 * math.pi / max_imag
    else:
        oscillation_period = float('inf')

    populations = (
        tuple(times),
        tuple(sat_hist),
        tuple(deb_hist),
        tuple(prop_hist),
    )

    return CoupledDebrisOperations(
        equilibrium_satellites=n_sat,
        equilibrium_debris=n_d,
        equilibrium_propellant_rate=p_rate,
        jacobian_eigenvalues=tuple(complex(ev) for ev in eigenvalues),
        is_stable=is_stable,
        oscillation_period_years=oscillation_period,
        populations=populations,
    )
