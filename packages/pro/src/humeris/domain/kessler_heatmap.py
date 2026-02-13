# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Kessler syndrome heat map: spatial density by altitude and inclination.

Computes a 2D grid of debris/object spatial density over altitude x
inclination bands. Each cell contains object count, volumetric spatial
density, mean collision velocity, and a risk classification.

Spatial density uses correct spherical zone volume for each cell:
    V_cell = (2/3) * pi * (r_max^3 - r_min^3) * (cos(i_min) - cos(i_max))

where the cosine difference accounts for the solid angle fraction of the
spherical shell accessible to orbits in that inclination band.

Mean collision velocity uses the random-RAAN crossing angle:
    V_rel = V_circ * sqrt(2) * sin(i_mid)

where i_mid is the mid-inclination of the band (Kessler 1978).

Cross-disciplinary metrics (all derived from verified structural rhymes):
    Population entropy: Shannon entropy of density distribution (Information Theory).
    Percolation fraction: Fraction of cells at high/critical risk vs
        2D percolation threshold 0.59 (Physics, Stauffer & Aharony 1994).
    Cascade multiplication factor k_eff: Analogous to nuclear criticality
        (Nuclear Engineering). k_eff > 1 = supercritical cascade regime.
    Temporal persistence: Track chronic vs transient risk concentrations
        (Criminology hotspot persistence analysis).
    Lyapunov density trend: Rate of change of peak density as a cascade
        proximity indicator (Dynamical Systems bifurcation theory).

References:
    Kessler (1978). Collision frequency of artificial satellites.
    Kessler (1991). Collisional cascading: the limits of population growth.
    Stauffer & Aharony (1994). Introduction to Percolation Theory, 2nd ed.
    Bak, Tang & Wiesenfeld (1987). Self-organized criticality.
"""
import math
from dataclasses import dataclass

import numpy as np

from humeris.domain.propagation import OrbitalState
from humeris.domain.orbital_mechanics import OrbitalConstants


@dataclass(frozen=True)
class KesslerCell:
    """A single cell in the Kessler heat map grid."""
    altitude_min_km: float
    altitude_max_km: float
    inclination_min_deg: float
    inclination_max_deg: float
    object_count: int
    spatial_density_per_km3: float
    mean_collision_velocity_ms: float
    risk_level: str  # "low", "moderate", "high", "critical"


@dataclass(frozen=True)
class KesslerHeatMap:
    """Complete Kessler syndrome heat map with cross-disciplinary metrics."""
    cells: tuple
    altitude_bins_km: tuple
    inclination_bins_deg: tuple
    total_objects: int
    peak_density_altitude_km: float
    peak_density_inclination_deg: float
    peak_density_per_km3: float
    population_entropy: float = 0.0
    percolation_fraction: float = 0.0
    is_percolation_risk: bool = False
    cascade_k_eff: float = 0.0
    is_supercritical: bool = False


@dataclass(frozen=True)
class KesslerPersistence:
    """Temporal persistence tracking across successive evaluations."""
    altitude_min_km: float
    altitude_max_km: float
    inclination_min_deg: float
    inclination_max_deg: float
    consecutive_high_count: int
    is_chronic: bool  # True if high/critical for >= 3 consecutive evals


def classify_kessler_risk(spatial_density_per_km3: float) -> str:
    """Classify Kessler cascade risk from spatial density.

    Thresholds based on Kessler 1991 critical density estimates:
    - Critical: > 1e-7 /km^3 (self-sustaining cascade likely)
    - High:     > 1e-8 /km^3 (significant collision risk)
    - Moderate: > 1e-9 /km^3 (elevated collision risk)
    - Low:      <= 1e-9 /km^3

    Args:
        spatial_density_per_km3: Volumetric spatial density (objects/km^3).

    Returns:
        Risk level string.
    """
    if spatial_density_per_km3 > 1e-7:
        return "critical"
    if spatial_density_per_km3 > 1e-8:
        return "high"
    if spatial_density_per_km3 > 1e-9:
        return "moderate"
    return "low"


def _shell_volume_km3(
    alt_min_km: float,
    alt_max_km: float,
    inc_min_deg: float,
    inc_max_deg: float,
) -> float:
    """Compute the spherical shell volume for an altitude-inclination cell.

    Uses correct spherical zone geometry: the solid angle fraction for
    an inclination band [i_min, i_max] is:
        (cos(i_min) - cos(i_max)) / 2

    This accounts for the non-uniform distribution of solid angle with
    inclination (more volume near 90 deg than near 0 or 180 deg).
    """
    r_min = OrbitalConstants.R_EARTH / 1000.0 + alt_min_km
    r_max = OrbitalConstants.R_EARTH / 1000.0 + alt_max_km
    full_shell = (4.0 / 3.0) * math.pi * (r_max ** 3 - r_min ** 3)

    # Solid angle fraction (correct spherical zone geometry)
    inc_min_rad = math.radians(inc_min_deg)
    inc_max_rad = math.radians(inc_max_deg)
    inc_fraction = (math.cos(inc_min_rad) - math.cos(inc_max_rad)) / 2.0

    return full_shell * inc_fraction


def _mean_collision_velocity(
    alt_min_km: float,
    alt_max_km: float,
    inc_min_deg: float,
    inc_max_deg: float,
) -> float:
    """Estimate mean collision velocity for objects in this cell.

    Uses random-RAAN crossing angle formula (Kessler 1978):
        V_rel = V_circ * sqrt(2) * sin(i_mid)

    For near-equatorial orbits (i < 5 deg), uses a minimum relative
    velocity from orbital mechanics (differential node rates).
    """
    mid_alt_m = ((alt_min_km + alt_max_km) / 2.0) * 1000.0
    r = OrbitalConstants.R_EARTH + mid_alt_m
    v_circ = float(np.sqrt(OrbitalConstants.MU_EARTH / r))

    mid_inc_rad = math.radians((inc_min_deg + inc_max_deg) / 2.0)

    # Kessler (1978): V_rel = V_circ * sqrt(2) * sin(i)
    # Clip to minimum sin(1 deg) to avoid zero velocity at equatorial orbits
    sin_i_eff = max(math.sin(mid_inc_rad), math.sin(math.radians(1.0)))
    v_rel = v_circ * math.sqrt(2.0) * sin_i_eff
    return v_rel


def _compute_population_entropy(
    counts: list[list[int]],
    total: int,
) -> float:
    """Compute Shannon entropy of object distribution across cells.

    H = -sum(p_i * log2(p_i)) for all cells with objects.

    Low entropy = concentrated risk (few cells hold most objects).
    High entropy = spread out (risk distributed evenly).
    Maximum entropy = log2(n_cells) when all cells have equal objects.
    """
    if total <= 0:
        return 0.0

    entropy = 0.0
    for row in counts:
        for count in row:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
    return entropy


def _compute_percolation_fraction(cells: list[KesslerCell]) -> float:
    """Compute fraction of cells at high or critical risk level.

    Percolation theory (Stauffer & Aharony 1994): when the fraction
    of occupied sites exceeds the percolation threshold (~0.59 for
    2D square lattice), a spanning cluster forms and cascade propagates
    across altitude bands.
    """
    if not cells:
        return 0.0
    high_critical = sum(
        1 for c in cells if c.risk_level in ("high", "critical")
    )
    return high_critical / len(cells)


_PERCOLATION_THRESHOLD_2D = 0.5927  # 2D square lattice site percolation


def _compute_cascade_k_eff(
    cells: list[KesslerCell],
    mean_fragments_per_collision: float = 100.0,
    mean_cross_section_km2: float = 1e-6,
    orbital_lifetime_years: float = 25.0,
) -> float:
    """Compute cascade multiplication factor k_eff from peak-density cell.

    Analogous to nuclear criticality (neutron multiplication factor):
        k_eff = mean_fragments * P_collision_per_fragment

    where P_collision_per_fragment is the probability that a single
    fragment itself collides within one orbital lifetime.

    k_eff < 1: subcritical (debris removed faster than produced)
    k_eff = 1: critical (steady state)
    k_eff > 1: supercritical (runaway cascade)

    Uses spatial density and collision velocity from cells to estimate
    P_collision = density * sigma * V_rel * lifetime.

    Note: this uses only the peak-density cell for a worst-case
    single-cell estimate. For a more accurate multi-cell analysis
    that accounts for inter-cell debris migration via drag, see
    compute_spectral_kessler() which computes the Perron-Frobenius
    eigenvalue of the full migration matrix.
    """
    if not cells:
        return 0.0

    # Use peak-density cell for worst-case k_eff
    peak_cell = max(cells, key=lambda c: c.spatial_density_per_km3)
    if peak_cell.spatial_density_per_km3 <= 0:
        return 0.0

    # Collision probability for one fragment per lifetime
    # P_col = n * sigma * V_rel * T
    n = peak_cell.spatial_density_per_km3  # objects/km^3
    sigma = mean_cross_section_km2  # km^2
    v_rel_km_s = peak_cell.mean_collision_velocity_ms / 1000.0  # km/s
    t_seconds = orbital_lifetime_years * 365.25 * 86400.0  # seconds
    p_collision = n * sigma * v_rel_km_s * t_seconds

    return mean_fragments_per_collision * p_collision


def _compute_lyapunov_estimate(
    current_peak_density: float,
    previous_peak_density: float,
    time_interval_years: float,
) -> float:
    """Estimate Lyapunov exponent of debris density dynamics.

    lambda = ln(density_new / density_old) / delta_t

    lambda > 0: density growing (approaching cascade bifurcation)
    lambda = 0: stable (critical point)
    lambda < 0: density declining (subcritical)

    Requires two successive evaluations.
    """
    if (
        time_interval_years <= 0
        or current_peak_density <= 0
        or previous_peak_density <= 0
    ):
        return 0.0
    return (
        math.log(current_peak_density / previous_peak_density)
        / time_interval_years
    )


def update_persistence(
    current_cells: list[KesslerCell],
    previous_persistence: list[KesslerPersistence] | None = None,
) -> list[KesslerPersistence]:
    """Update temporal persistence tracking.

    Tracks how many consecutive evaluations each cell has been at
    high or critical risk level. Cells persistent for >= 3 evaluations
    are classified as chronic hotspots (Criminology hotspot persistence).

    Args:
        current_cells: Current heatmap cells.
        previous_persistence: Previous persistence records (None if first).

    Returns:
        Updated persistence records.
    """
    # Build lookup from previous persistence
    prev_map: dict[tuple[float, float, float, float], int] = {}
    if previous_persistence:
        for p in previous_persistence:
            key = (
                p.altitude_min_km, p.altitude_max_km,
                p.inclination_min_deg, p.inclination_max_deg,
            )
            prev_map[key] = p.consecutive_high_count

    result = []
    for cell in current_cells:
        key = (
            cell.altitude_min_km, cell.altitude_max_km,
            cell.inclination_min_deg, cell.inclination_max_deg,
        )
        is_elevated = cell.risk_level in ("high", "critical")
        prev_count = prev_map.get(key, 0)
        new_count = (prev_count + 1) if is_elevated else 0

        result.append(KesslerPersistence(
            altitude_min_km=cell.altitude_min_km,
            altitude_max_km=cell.altitude_max_km,
            inclination_min_deg=cell.inclination_min_deg,
            inclination_max_deg=cell.inclination_max_deg,
            consecutive_high_count=new_count,
            is_chronic=new_count >= 3,
        ))

    return result


def compute_kessler_heatmap(
    states: list,
    altitude_step_km: float = 50.0,
    inclination_step_deg: float = 10.0,
    altitude_min_km: float = 200.0,
    altitude_max_km: float = 2000.0,
    mean_fragments_per_collision: float = 100.0,
    mean_cross_section_km2: float = 1e-6,
    orbital_lifetime_years: float = 25.0,
) -> KesslerHeatMap:
    """Compute Kessler syndrome heat map from orbital states.

    Creates a 2D grid of altitude x inclination bins and counts objects
    in each cell. Computes spatial density, collision velocity, and
    cross-disciplinary cascade risk metrics.

    Args:
        states: List of OrbitalState objects (constellation + debris).
        altitude_step_km: Altitude bin width (km).
        inclination_step_deg: Inclination bin width (degrees).
        altitude_min_km: Lower altitude bound (km).
        altitude_max_km: Upper altitude bound (km).
        mean_fragments_per_collision: Average trackable fragments per
            collision event (default 100, NASA Standard Breakup Model).
        mean_cross_section_km2: Average collision cross-section (km^2).
        orbital_lifetime_years: Assumed orbital lifetime for k_eff.

    Returns:
        KesslerHeatMap with spatial density grid, peak identification,
        and cross-disciplinary cascade metrics.
    """
    # Build bins
    alt_edges = []
    a = altitude_min_km
    while a < altitude_max_km:
        alt_edges.append(a)
        a += altitude_step_km
    alt_edges.append(altitude_max_km)

    inc_edges = []
    i = 0.0
    while i < 180.0:
        inc_edges.append(i)
        i += inclination_step_deg
    inc_edges.append(180.0)

    n_alt = len(alt_edges) - 1
    n_inc = len(inc_edges) - 1

    # Count objects per cell
    counts = [[0] * n_inc for _ in range(n_alt)]
    r_earth_m = OrbitalConstants.R_EARTH

    for s in states:
        alt_km = (s.semi_major_axis_m - r_earth_m) / 1000.0
        inc_deg = math.degrees(s.inclination_rad)

        # Find altitude bin
        alt_idx = -1
        for ai in range(n_alt):
            if alt_edges[ai] <= alt_km < alt_edges[ai + 1]:
                alt_idx = ai
                break
        if alt_idx < 0:
            if abs(alt_km - altitude_max_km) < 0.01:
                alt_idx = n_alt - 1
            else:
                continue

        # Find inclination bin
        inc_idx = -1
        for ii in range(n_inc):
            if inc_edges[ii] <= inc_deg < inc_edges[ii + 1]:
                inc_idx = ii
                break
        if inc_idx < 0:
            if abs(inc_deg - 180.0) < 0.01:
                inc_idx = n_inc - 1
            else:
                continue

        counts[alt_idx][inc_idx] += 1

    # Build cells
    cells = []
    peak_density = 0.0
    peak_alt = 0.0
    peak_inc = 0.0

    for ai in range(n_alt):
        for ii in range(n_inc):
            alt_lo = alt_edges[ai]
            alt_hi = alt_edges[ai + 1]
            inc_lo = inc_edges[ii]
            inc_hi = inc_edges[ii + 1]
            count = counts[ai][ii]

            vol = _shell_volume_km3(alt_lo, alt_hi, inc_lo, inc_hi)
            density = count / vol if vol > 0 else 0.0
            v_coll = _mean_collision_velocity(alt_lo, alt_hi, inc_lo, inc_hi)
            risk = classify_kessler_risk(density)

            cells.append(KesslerCell(
                altitude_min_km=alt_lo,
                altitude_max_km=alt_hi,
                inclination_min_deg=inc_lo,
                inclination_max_deg=inc_hi,
                object_count=count,
                spatial_density_per_km3=density,
                mean_collision_velocity_ms=v_coll,
                risk_level=risk,
            ))

            if density > peak_density:
                peak_density = density
                peak_alt = (alt_lo + alt_hi) / 2.0
                peak_inc = (inc_lo + inc_hi) / 2.0

    alt_bins = tuple(alt_edges)
    inc_bins = tuple(inc_edges)
    total = sum(sum(row) for row in counts)

    # Cross-disciplinary metrics
    entropy = _compute_population_entropy(counts, total)
    perc_frac = _compute_percolation_fraction(cells)
    k_eff = _compute_cascade_k_eff(
        cells,
        mean_fragments_per_collision,
        mean_cross_section_km2,
        orbital_lifetime_years,
    )

    return KesslerHeatMap(
        cells=tuple(cells),
        altitude_bins_km=alt_bins,
        inclination_bins_deg=inc_bins,
        total_objects=total,
        peak_density_altitude_km=peak_alt,
        peak_density_inclination_deg=peak_inc,
        peak_density_per_km3=peak_density,
        population_entropy=entropy,
        percolation_fraction=perc_frac,
        is_percolation_risk=perc_frac >= _PERCOLATION_THRESHOLD_2D,
        cascade_k_eff=k_eff,
        is_supercritical=k_eff > 1.0,
    )


# ── Poisson Conjunction Arrival Process ─────────────────────────────
#
# Non-homogeneous Poisson process for conjunction arrivals.
# Intensity derived from Kessler heatmap spatial density and relative
# velocity fields. See SESSION_MINING_R1_CREATIVE.md P8.


@dataclass(frozen=True)
class ConjunctionPoissonModel:
    """Non-homogeneous Poisson model for conjunction arrivals.

    Attributes:
        intensity_per_year: Instantaneous conjunction rate at each time step.
        cumulative_intensity: Cumulative intensity Lambda(t) = integral lambda ds.
        survival_probability: P(no conjunction in [0, t]) = exp(-Lambda(t)).
        expected_time_to_first_years: E[T_1] via numerical integration.
        is_increasing_rate: True if intensity trend is positive (worsening).
        peak_intensity_per_year: Maximum intensity over the time horizon.
    """
    intensity_per_year: tuple
    cumulative_intensity: tuple
    survival_probability: tuple
    expected_time_to_first_years: float
    is_increasing_rate: bool
    peak_intensity_per_year: float


def compute_conjunction_intensity(
    spatial_density_per_km3: float,
    mean_collision_velocity_ms: float,
    collision_cross_section_km2: float = 1e-5,
) -> float:
    """Compute conjunction intensity from spatial density and relative velocity.

    lambda = rho * sigma * v_rel (encounters per unit time per unit volume,
    but since the satellite sweeps through the volume, this gives rate per
    satellite).

    Units: density [1/km^3] * cross-section [km^2] * velocity [km/s] * [s/year]
    = [1/year].

    Args:
        spatial_density_per_km3: Volumetric spatial density (objects/km^3).
        mean_collision_velocity_ms: Mean relative collision velocity (m/s).
        collision_cross_section_km2: Effective collision cross-section (km^2).

    Returns:
        Conjunction intensity in events per year.
    """
    v_rel_km_s = mean_collision_velocity_ms / 1000.0
    seconds_per_year = 365.25 * 86400.0
    return (spatial_density_per_km3
            * collision_cross_section_km2
            * v_rel_km_s
            * seconds_per_year)


def conjunction_poisson_rate(
    heatmap: KesslerHeatMap,
    satellite_altitude_km: float,
    satellite_inclination_deg: float,
    collision_cross_section_km2: float = 1e-5,
) -> float:
    """Expected conjunctions per year from heatmap for a given orbit.

    Identifies the heatmap cell containing the satellite and computes
    the Poisson intensity from that cell's spatial density and collision
    velocity.

    Args:
        heatmap: Kessler heatmap with spatial density grid.
        satellite_altitude_km: Satellite altitude (km).
        satellite_inclination_deg: Satellite inclination (degrees).
        collision_cross_section_km2: Collision cross-section (km^2).

    Returns:
        Expected conjunctions per year.
    """
    best_cell = None
    for cell in heatmap.cells:
        if (cell.altitude_min_km <= satellite_altitude_km < cell.altitude_max_km
                and cell.inclination_min_deg <= satellite_inclination_deg < cell.inclination_max_deg):
            best_cell = cell
            break

    if best_cell is None:
        return 0.0

    return compute_conjunction_intensity(
        best_cell.spatial_density_per_km3,
        best_cell.mean_collision_velocity_ms,
        collision_cross_section_km2,
    )


def conjunction_probability_window(
    rate_per_year: float,
    duration_years: float,
) -> float:
    """Probability of at least one conjunction in a time window.

    P(N >= 1) = 1 - exp(-lambda * T) for a homogeneous Poisson process.

    Args:
        rate_per_year: Conjunction rate (events/year).
        duration_years: Time window duration (years).

    Returns:
        Probability of at least one conjunction (0 to 1).
    """
    if rate_per_year <= 0.0 or duration_years <= 0.0:
        return 0.0
    return 1.0 - math.exp(-rate_per_year * duration_years)


def compute_conjunction_poisson_model(
    heatmap: KesslerHeatMap,
    satellite_altitude_km: float,
    satellite_inclination_deg: float,
    collision_cross_section_km2: float = 1e-5,
    duration_years: float = 25.0,
    step_years: float = 0.1,
    density_growth_rate_per_year: float = 0.0,
) -> ConjunctionPoissonModel:
    """Compute non-homogeneous Poisson conjunction model.

    Models conjunction arrivals as NHPP with intensity that may vary
    over time (e.g., due to debris environment growth). The base rate
    comes from the Kessler heatmap; temporal variation is modelled via
    an exponential growth/decay factor.

    Args:
        heatmap: Kessler heatmap with spatial density grid.
        satellite_altitude_km: Satellite altitude (km).
        satellite_inclination_deg: Satellite inclination (degrees).
        collision_cross_section_km2: Collision cross-section (km^2).
        duration_years: Time horizon (years).
        step_years: Time step (years).
        density_growth_rate_per_year: Annual fractional growth rate of
            spatial density (0.0 = constant, 0.05 = 5% per year growth).

    Returns:
        ConjunctionPoissonModel with intensity, survival, and statistics.
    """
    base_rate = conjunction_poisson_rate(
        heatmap, satellite_altitude_km, satellite_inclination_deg,
        collision_cross_section_km2,
    )

    n_steps = max(1, int(duration_years / step_years))
    dt = step_years

    intensities = []
    cumulative = []
    survival = []
    cum = 0.0

    for k in range(n_steps + 1):
        t = k * dt
        # Non-homogeneous: density grows/decays exponentially
        lam = base_rate * math.exp(density_growth_rate_per_year * t)
        intensities.append(lam)
        cum += lam * dt if k > 0 else 0.0
        cumulative.append(cum)
        survival.append(math.exp(-cum))

    # Expected time to first conjunction via numerical integration
    # E[T_1] = integral_0^inf S(t) dt, approximated over our time grid
    expected_t1 = sum(s * dt for s in survival)

    # Analytical tail correction: if S(T) > 0.01 at the end of the grid,
    # the numerical integral is truncated and underestimates E[T1].
    # For small rates where E[T1] >> duration, add tail estimate.
    # For constant rate lambda: S(t) = exp(-lambda*t), tail integral
    # from T to inf = S(T) / lambda.
    final_survival = survival[-1] if survival else 0.0
    if final_survival > 0.01:
        # Use the final intensity as the tail rate
        final_rate = intensities[-1] if intensities else 0.0
        if final_rate > 0:
            tail_correction = final_survival / final_rate
            expected_t1 += tail_correction

    peak_intensity = max(intensities)
    is_increasing = density_growth_rate_per_year > 0.0

    return ConjunctionPoissonModel(
        intensity_per_year=tuple(intensities),
        cumulative_intensity=tuple(cumulative),
        survival_probability=tuple(survival),
        expected_time_to_first_years=expected_t1,
        is_increasing_rate=is_increasing,
        peak_intensity_per_year=peak_intensity,
    )


# ── Spectral Kessler via Perron-Frobenius Eigenvalue (P18) ────────

@dataclass(frozen=True)
class SpectralKessler:
    """Spectral cascade analysis using Perron-Frobenius eigenvalue.

    The debris migration matrix T describes how collision fragments
    from one altitude-inclination cell migrate to adjacent cells via
    atmospheric drag. The dominant eigenvalue of T (Perron-Frobenius
    eigenvalue) is the true cascade multiplication factor k_eff,
    accounting for inter-cell debris transport.

    This is more accurate than the per-cell k_eff in _compute_cascade_k_eff
    which uses only the peak-density cell and ignores migration.

    The spectral gap (lambda_1 - lambda_2) indicates how quickly the
    debris distribution converges to the dominant mode: larger gap means
    faster convergence and a more predictable cascade pattern.

    References:
        Perron (1907). Zur Theorie der Matrices.
        Frobenius (1912). Uber Matrizen aus nicht negativen Elementen.
        Seneta (2006). Non-negative Matrices and Markov Chains. Springer.
    """
    migration_eigenvalue: float        # Perron-Frobenius eigenvalue (true k_eff)
    dominant_mode: tuple[float, ...]   # PF eigenvector (amplification pattern)
    convergence_time: float            # 1 / spectral_gap (characteristic time)
    k_eff_peak_cell: float             # Single-cell estimate for comparison
    spectral_gap: float                # lambda_1 - lambda_2
    most_dangerous_cell_idx: int       # Cell with highest PF eigenvector component
    is_supercritical: bool             # migration_eigenvalue > 1.0


def compute_spectral_kessler(
    heatmap: KesslerHeatMap,
    mean_fragments_per_collision: float = 100.0,
    collision_cross_section_km2: float = 1e-6,
    orbital_lifetime_years: float = 25.0,
    drag_scale_height_km: float = 50.0,
) -> SpectralKessler:
    """Compute spectral cascade analysis from Kessler heatmap.

    Builds a debris migration matrix between altitude-inclination cells
    and computes the Perron-Frobenius eigenvalue as the true k_eff.

    The migration matrix T has entries:
        T_ij = N_frag * P_collision_j * P_migration_{j->i}

    where:
        P_collision_j = rho_j * sigma * v_rel_j * lifetime
        P_migration_{j->i} = exp(-|alt_j - alt_i| / scale_height)
            (fragments preferentially migrate to nearby altitude bands via drag)

    For the inclination dimension, migration is minimal (drag doesn't
    change inclination significantly), so off-diagonal inclination
    entries are near zero.

    Args:
        heatmap: Kessler heatmap with cells.
        mean_fragments_per_collision: Average trackable fragments per collision.
        collision_cross_section_km2: Average collision cross-section (km^2).
        orbital_lifetime_years: Assumed orbital lifetime for collision probability.
        drag_scale_height_km: Atmospheric scale height for altitude migration.

    Returns:
        SpectralKessler with Perron-Frobenius eigenvalue and dominant mode.
    """
    cells = list(heatmap.cells)
    n_cells = len(cells)

    if n_cells == 0:
        return SpectralKessler(
            migration_eigenvalue=0.0,
            dominant_mode=(),
            convergence_time=float('inf'),
            k_eff_peak_cell=0.0,
            spectral_gap=0.0,
            most_dangerous_cell_idx=0,
            is_supercritical=False,
        )

    # Single-cell k_eff for comparison
    k_eff_peak = _compute_cascade_k_eff(
        cells, mean_fragments_per_collision,
        collision_cross_section_km2, orbital_lifetime_years,
    )

    # Build the debris migration matrix
    t_seconds = orbital_lifetime_years * 365.25 * 86400.0

    # Precompute cell mid-altitudes and mid-inclinations
    mid_alts = np.array([(c.altitude_min_km + c.altitude_max_km) / 2.0 for c in cells])
    mid_incs = np.array([(c.inclination_min_deg + c.inclination_max_deg) / 2.0 for c in cells])
    densities = np.array([c.spatial_density_per_km3 for c in cells])
    v_rels = np.array([c.mean_collision_velocity_ms / 1000.0 for c in cells])  # km/s

    # Collision probability per fragment in each source cell
    p_collision = densities * collision_cross_section_km2 * v_rels * t_seconds

    # Migration probability: fragments from cell j migrate to cell i
    # based on altitude difference (drag-driven) and inclination similarity
    transfer = np.zeros((n_cells, n_cells))

    for j in range(n_cells):
        for i in range(n_cells):
            # Altitude migration: exponential decay with distance
            alt_diff = abs(mid_alts[i] - mid_alts[j])
            p_alt = float(np.exp(-alt_diff / drag_scale_height_km))

            # Inclination: drag doesn't change inclination, so
            # same-inclination band gets most of the fragments
            inc_diff = abs(mid_incs[i] - mid_incs[j])
            # Tight Gaussian: most fragments stay in same inclination band
            inc_scale = 5.0  # degrees
            p_inc = float(np.exp(-0.5 * (inc_diff / inc_scale) ** 2))

            # Combined migration probability (not normalized — the matrix
            # T is not stochastic, it's a reproduction matrix)
            p_migration = p_alt * p_inc

            # Transfer matrix entry: fragments produced in j that end up
            # colliding in cell i
            transfer[i, j] = mean_fragments_per_collision * p_collision[j] * p_migration

    # Normalize migration probabilities column-wise so that sum_i P_migration_{j->i} = 1
    # This ensures T_ij = N_frag * P_collision_j * (normalized migration)
    # Without normalization, the raw migration weights over-count fragments.
    col_sums = np.sum(transfer, axis=0)
    for j in range(n_cells):
        if col_sums[j] > 0:
            transfer[:, j] /= col_sums[j]
            # Re-scale: T_ij = N_frag * P_collision_j * P_migration_norm_{j->i}
            transfer[:, j] *= mean_fragments_per_collision * p_collision[j]

    # Compute eigenvalues using numpy
    eigenvalues = np.linalg.eigvals(transfer)

    # Perron-Frobenius: dominant eigenvalue is the one with largest magnitude
    # For a non-negative matrix, it is real and positive
    magnitudes = np.abs(eigenvalues)
    sorted_idx = np.argsort(-magnitudes)  # descending

    lambda_1 = float(np.real(eigenvalues[sorted_idx[0]]))
    lambda_1 = max(lambda_1, 0.0)  # PF eigenvalue is non-negative

    if n_cells > 1:
        lambda_2 = float(np.abs(eigenvalues[sorted_idx[1]]))
    else:
        lambda_2 = 0.0

    spectral_gap = lambda_1 - lambda_2

    # Convergence time: characteristic time for debris distribution to reach
    # dominant mode. 1/spectral_gap in units of "generations" (orbital lifetimes)
    convergence_time = 1.0 / spectral_gap if spectral_gap > 1e-15 else float('inf')

    # Dominant eigenvector: compute from the transfer matrix
    # Use power iteration for robustness (real, non-negative eigenvector)
    dominant_mode = _power_iteration(transfer, max_iter=100)

    # Most dangerous cell: highest component in dominant mode
    most_dangerous = int(np.argmax(dominant_mode))

    return SpectralKessler(
        migration_eigenvalue=lambda_1,
        dominant_mode=tuple(float(x) for x in dominant_mode),
        convergence_time=convergence_time,
        k_eff_peak_cell=k_eff_peak,
        spectral_gap=spectral_gap,
        most_dangerous_cell_idx=most_dangerous,
        is_supercritical=lambda_1 > 1.0,
    )


def _power_iteration(matrix: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
    """Power iteration to find dominant eigenvector of a non-negative matrix.

    Args:
        matrix: Square non-negative matrix.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Normalized dominant eigenvector (non-negative).
    """
    n = matrix.shape[0]
    if n == 0:
        return np.array([])

    # Start with uniform vector
    v = np.ones(n) / n

    for _ in range(max_iter):
        v_new = matrix @ v
        norm = float(np.linalg.norm(v_new))
        if norm < 1e-20:
            return np.ones(n) / n  # Matrix is effectively zero
        v_new = v_new / norm

        # Check convergence
        if float(np.linalg.norm(v_new - v)) < tol:
            break
        v = v_new

    # Ensure non-negative (PF eigenvector should be)
    v = np.abs(v)
    norm = float(np.sum(v))
    if norm > 0:
        v = v / norm

    return v


# ── P33: Fokker-Planck Density Evolution ────────────────────────────
#
# Debris density rho(h, t) evolves via advection-diffusion PDE:
#   drho/dt = -d(F*rho)/dh + D * d2rho/dh2 + S(h,t)
#
# Explicit Euler with CFL stability check.


@dataclass(frozen=True)
class DebrisDensityEvolution:
    """Fokker-Planck debris density evolution results.

    Attributes:
        altitude_bins: Mid-altitudes of each altitude bin (km).
        time_steps: Time points (years).
        density_evolution: Density at each (time, altitude) pair.
            Tuple of tuples: outer = time steps, inner = altitude bins.
        peak_density_trajectory: Peak density at each time step.
        total_mass_trajectory: Integral of density over altitude at each time step.
    """
    altitude_bins: tuple
    time_steps: tuple
    density_evolution: tuple
    peak_density_trajectory: tuple
    total_mass_trajectory: tuple


def compute_debris_density_evolution(
    altitude_min_km: float = 300.0,
    altitude_max_km: float = 1000.0,
    n_altitude_bins: int = 50,
    duration_years: float = 10.0,
    step_years: float = 0.1,
    initial_density_per_km3: float = 1e-8,
    source_altitude_km: float = 800.0,
    source_width_km: float = 50.0,
    source_rate_per_km3_per_year: float = 1e-10,
    drag_scale_height_km: float = 50.0,
    drag_base_velocity_km_per_year: float = 0.5,
    diffusion_coefficient: float = 10.0,
) -> DebrisDensityEvolution:
    """Compute Fokker-Planck debris density evolution.

    Models debris density rho(h, t) as an advection-diffusion PDE:
        drho/dt = -d(F*rho)/dh + D * d2rho/dh2 + S(h, t)

    where F(h) is altitude-dependent drag drift (downward), D is diffusion
    from atmospheric density variability, and S is collision source injection.

    Uses explicit Euler with CFL-constrained time stepping.

    Args:
        altitude_min_km: Minimum altitude (km).
        altitude_max_km: Maximum altitude (km).
        n_altitude_bins: Number of altitude bins.
        duration_years: Simulation duration (years).
        step_years: Output time step (years). Internal step may be smaller for CFL.
        initial_density_per_km3: Uniform initial density (objects/km^3).
        source_altitude_km: Center altitude for collision source injection (km).
        source_width_km: Width of source injection Gaussian (km).
        source_rate_per_km3_per_year: Peak source injection rate.
        drag_scale_height_km: Atmospheric scale height for drag drift profile.
        drag_base_velocity_km_per_year: Base drift velocity at reference altitude.
        diffusion_coefficient: Diffusion coefficient D (km^2/year).

    Returns:
        DebrisDensityEvolution with density grid, peak trajectory, and mass trajectory.
    """
    dh = (altitude_max_km - altitude_min_km) / n_altitude_bins
    alt_mids = np.array([altitude_min_km + (i + 0.5) * dh for i in range(n_altitude_bins)])

    # Drift velocity F(h): drag pushes debris downward, stronger at lower altitudes
    # F(h) = -v_base * exp(-(h - h_min) / H_scale)
    drift = -drag_base_velocity_km_per_year * np.exp(
        -(alt_mids - altitude_min_km) / drag_scale_height_km
    )

    # Source term: Gaussian injection at source_altitude_km
    source = source_rate_per_km3_per_year * np.exp(
        -0.5 * ((alt_mids - source_altitude_km) / max(source_width_km, 1.0)) ** 2
    )

    # CFL stability conditions
    max_abs_f = float(np.max(np.abs(drift))) if float(np.max(np.abs(drift))) > 0 else 1e-10
    d_eff = max(diffusion_coefficient, 1e-20)

    dt_cfl_advection = dh / max_abs_f
    dt_cfl_diffusion = dh * dh / (2.0 * d_eff) if d_eff > 0 else float('inf')
    dt_cfl = 0.5 * min(dt_cfl_advection, dt_cfl_diffusion)  # safety factor 0.5

    # Initialize density
    rho = np.full(n_altitude_bins, initial_density_per_km3)

    # Output collection
    n_output_steps = int(duration_years / step_years) + 1
    output_times = []
    output_densities = []
    peak_densities = []
    total_masses = []

    current_time = 0.0
    next_output_time = 0.0
    output_idx = 0

    # Record initial state
    output_times.append(0.0)
    output_densities.append(tuple(float(x) for x in rho))
    peak_densities.append(float(np.max(rho)))
    total_masses.append(float(np.sum(rho) * dh))
    output_idx = 1
    next_output_time = step_years

    # Time integration
    while output_idx < n_output_steps:
        # Sub-step to respect CFL
        time_to_next = next_output_time - current_time
        dt_actual = min(dt_cfl, time_to_next)
        if dt_actual <= 0:
            dt_actual = dt_cfl

        # Explicit Euler step
        rho_new = rho.copy()

        for i in range(n_altitude_bins):
            # Advection: -d(F*rho)/dh via upwind scheme
            if i > 0 and i < n_altitude_bins - 1:
                flux_right = drift[i] * rho[i] if drift[i] >= 0 else drift[i] * rho[min(i + 1, n_altitude_bins - 1)]
                flux_left = drift[max(i - 1, 0)] * rho[max(i - 1, 0)] if drift[max(i - 1, 0)] >= 0 else drift[max(i - 1, 0)] * rho[i]
                advection = -(flux_right - flux_left) / dh
            elif i == 0:
                flux_right = drift[0] * rho[0] if drift[0] >= 0 else drift[0] * rho[min(1, n_altitude_bins - 1)]
                advection = -flux_right / dh
            else:
                flux_left = drift[i - 1] * rho[i - 1] if drift[i - 1] >= 0 else drift[i - 1] * rho[i]
                advection = flux_left / dh

            # Diffusion: D * d2rho/dh2
            if 0 < i < n_altitude_bins - 1:
                diffusion_term = diffusion_coefficient * (
                    rho[i + 1] - 2.0 * rho[i] + rho[i - 1]
                ) / (dh * dh)
            else:
                diffusion_term = 0.0

            rho_new[i] = rho[i] + (advection + diffusion_term + source[i]) * dt_actual

        # Clamp to non-negative
        rho = np.maximum(rho_new, 0.0)
        current_time += dt_actual

        # Check if it is time to output
        if current_time >= next_output_time - 1e-12:
            output_times.append(float(next_output_time))
            output_densities.append(tuple(float(x) for x in rho))
            peak_densities.append(float(np.max(rho)))
            total_masses.append(float(np.sum(rho) * dh))
            output_idx += 1
            next_output_time = output_idx * step_years

    return DebrisDensityEvolution(
        altitude_bins=tuple(float(x) for x in alt_mids),
        time_steps=tuple(output_times),
        density_evolution=tuple(output_densities),
        peak_density_trajectory=tuple(peak_densities),
        total_mass_trajectory=tuple(total_masses),
    )


# ── P32: Hausdorff Dimension of Debris Cloud ──────────────────────
#
# Box-counting dimension: D = lim log(N(eps)) / log(1/eps).
# Linear regression of log(N) vs log(1/eps) gives D.


@dataclass(frozen=True)
class DebrisCloudDimension:
    """Box-counting dimension analysis of a debris cloud.

    Attributes:
        box_counting_dimension: Fractal dimension from box-counting.
        filling_fraction: Fraction of smallest boxes that are occupied.
        concentration_factor: Ratio of peak density to mean density.
        avoidance_duration_orbits: Estimated orbits before cloud fills D~2.
        dimension_uncertainty: R^2 of the log-log linear regression.
    """
    box_counting_dimension: float
    filling_fraction: float
    concentration_factor: float
    avoidance_duration_orbits: float
    dimension_uncertainty: float


def compute_debris_cloud_dimension(
    positions: np.ndarray,
    n_scales: int = 10,
) -> DebrisCloudDimension:
    """Compute box-counting (Hausdorff) dimension of a debris cloud.

    Uses the box-counting method: grid the space into boxes of
    decreasing size epsilon, count occupied boxes N(epsilon), and
    fit D = -slope of log(N) vs log(epsilon).

    Args:
        positions: Nx3 array of debris positions (any consistent units).
        n_scales: Number of box sizes to evaluate.

    Returns:
        DebrisCloudDimension with fractal dimension and diagnostics.
    """
    if positions.shape[0] <= 1:
        return DebrisCloudDimension(
            box_counting_dimension=0.0,
            filling_fraction=0.0 if positions.shape[0] == 0 else 1.0,
            concentration_factor=1.0,
            avoidance_duration_orbits=0.0,
            dimension_uncertainty=0.0,
        )

    # Normalize positions to [0, 1]^3 for scale-independent analysis
    mins = np.min(positions, axis=0)
    maxs = np.max(positions, axis=0)
    spans = maxs - mins

    # Handle degenerate dimensions (all points have same coordinate)
    effective_dims = int(np.sum(spans > 1e-15))
    if effective_dims == 0:
        return DebrisCloudDimension(
            box_counting_dimension=0.0,
            filling_fraction=1.0,
            concentration_factor=1.0,
            avoidance_duration_orbits=0.0,
            dimension_uncertainty=0.0,
        )

    # Normalize only non-degenerate dimensions; leave degenerate at 0
    normed = np.zeros_like(positions, dtype=float)
    for d in range(3):
        if spans[d] > 1e-15:
            normed[:, d] = (positions[:, d] - mins[d]) / spans[d]

    # Box-counting at multiple scales
    epsilons = []
    box_counts = []
    n_points = positions.shape[0]

    for k in range(n_scales):
        # Box sizes from 1/2 to 1/2^n_scales
        n_boxes_per_dim = 2 ** (k + 1)
        eps = 1.0 / n_boxes_per_dim
        epsilons.append(eps)

        # Assign each point to a box (only non-degenerate dimensions)
        box_indices = np.floor(normed / max(eps, 1e-15)).astype(int)
        # Clamp to valid range
        box_indices = np.clip(box_indices, 0, n_boxes_per_dim - 1)

        # Count unique boxes using only non-degenerate dimensions
        active_dims = [d for d in range(3) if spans[d] > 1e-15]
        unique_boxes = set()
        for i in range(n_points):
            key = tuple(box_indices[i, d] for d in active_dims)
            unique_boxes.add(key)
        box_counts.append(len(unique_boxes))

    # Filter out saturated scales: box count can't exceed n_points
    # Only use scales where box count is still growing (below saturation)
    valid_scales = []
    for i in range(len(box_counts)):
        if box_counts[i] < n_points * 0.95:
            valid_scales.append(i)
    # Need at least 2 points for regression; if too few non-saturated, use first half
    if len(valid_scales) < 2:
        valid_scales = list(range(min(n_scales, max(2, n_scales // 2))))

    # Linear regression: log(N) = D * log(1/eps) + c
    log_inv_eps = np.array([math.log(1.0 / epsilons[i]) for i in valid_scales])
    log_n = np.array([math.log(max(box_counts[i], 1)) for i in valid_scales])

    # Ordinary least squares
    x_mean = float(np.mean(log_inv_eps))
    y_mean = float(np.mean(log_n))

    ss_xy = float(np.sum((log_inv_eps - x_mean) * (log_n - y_mean)))
    ss_xx = float(np.sum((log_inv_eps - x_mean) ** 2))
    ss_yy = float(np.sum((log_n - y_mean) ** 2))

    if ss_xx > 1e-20:
        dimension = ss_xy / ss_xx
    else:
        dimension = 0.0

    # R^2 (coefficient of determination)
    if ss_yy > 1e-20 and ss_xx > 1e-20:
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)
    else:
        r_squared = 0.0

    # Clamp dimension to [0, 3]
    dimension = max(0.0, min(3.0, dimension))

    # Filling fraction: fraction of smallest-scale boxes occupied
    max_possible_boxes = (2 ** n_scales) ** effective_dims
    filling_fraction = box_counts[-1] / max(max_possible_boxes, 1)
    filling_fraction = min(filling_fraction, 1.0)

    # Concentration factor: peak vs mean density in box counts
    if len(box_counts) > 0 and box_counts[-1] > 0:
        # Use the ratio of total points to number of boxes as proxy
        concentration_factor = positions.shape[0] / box_counts[-1]
    else:
        concentration_factor = 1.0

    # Avoidance duration: estimated orbits until D reaches ~2
    # Model: D(t) ~ 0 at t=0, D ~ 1 after 1 orbit, D ~ 2 after 100 orbits
    # Inverse: t_avoid ~ 100^((2 - D_current) / 2.0) if D < 2
    if dimension < 2.0 and dimension > 0.0:
        avoidance_orbits = 100.0 ** ((2.0 - dimension) / 2.0)
    elif dimension >= 2.0:
        avoidance_orbits = 0.0
    else:
        avoidance_orbits = float('inf')

    return DebrisCloudDimension(
        box_counting_dimension=dimension,
        filling_fraction=filling_fraction,
        concentration_factor=concentration_factor,
        avoidance_duration_orbits=avoidance_orbits,
        dimension_uncertainty=r_squared,
    )


# ── P39: Renormalization Group for Multi-Scale Debris ──────────────
#
# Coarse-grain debris density from fine to coarse altitude bins.
# At each level: block-average density, recompute k_eff.
# RG flow: iterate to find fixed point k_eff*.


@dataclass(frozen=True)
class RenormalizationGroupAnalysis:
    """Renormalization group analysis for multi-scale debris dynamics.

    Attributes:
        fixed_point_k_eff: Fixed-point cascade multiplication factor.
        critical_exponent_nu: Critical exponent from linearized RG flow.
        correlation_length_cells: Correlation length in fine-grid cells.
        is_critical: True if k_eff is within 10% of 1.0.
        scale_levels: Tuple of (n_cells, k_eff) at each coarse-graining level.
    """
    fixed_point_k_eff: float
    critical_exponent_nu: float
    correlation_length_cells: float
    is_critical: bool
    scale_levels: tuple


def _compute_k_eff_from_density(
    density_profile: np.ndarray,
    mean_fragments: float,
    cross_section_km2: float,
    velocity_ms: float,
    lifetime_years: float,
) -> float:
    """Compute k_eff from a density profile using peak density.

    k_eff = N_frag * rho_peak * sigma * v_rel * T
    """
    if len(density_profile) == 0:
        return 0.0
    rho_peak = float(np.max(density_profile))
    v_rel_km_s = velocity_ms / 1000.0
    t_seconds = lifetime_years * 365.25 * 86400.0
    p_collision = rho_peak * cross_section_km2 * v_rel_km_s * t_seconds
    return mean_fragments * p_collision


def compute_renormalization_group(
    density_profile: np.ndarray,
    mean_fragments_per_collision: float = 100.0,
    collision_cross_section_km2: float = 1e-6,
    mean_collision_velocity_ms: float = 10000.0,
    orbital_lifetime_years: float = 25.0,
    block_factor: int = 2,
) -> RenormalizationGroupAnalysis:
    """Compute renormalization group analysis of debris density.

    Applies iterative coarse-graining (block averaging) to a 1D density
    profile and tracks how k_eff changes across scales. The fixed point
    of the RG flow determines whether the system is subcritical, critical,
    or supercritical at all scales simultaneously.

    Near the critical point, the RG eigenvalue lambda determines the
    critical exponent nu = ln(b) / ln(lambda), which governs the
    divergence of the correlation length.

    Args:
        density_profile: 1D array of debris spatial density per altitude bin.
        mean_fragments_per_collision: Fragments per collision event.
        collision_cross_section_km2: Collision cross-section (km^2).
        mean_collision_velocity_ms: Mean relative velocity (m/s).
        orbital_lifetime_years: Fragment orbital lifetime (years).
        block_factor: Coarse-graining block size (default 2).

    Returns:
        RenormalizationGroupAnalysis with fixed point, critical exponent,
        and scale-level diagnostics.
    """
    profile = np.array(density_profile, dtype=float)
    scale_levels = []

    # Compute k_eff at each scale level
    current = profile.copy()
    while len(current) >= 1:
        k_eff = _compute_k_eff_from_density(
            current, mean_fragments_per_collision,
            collision_cross_section_km2, mean_collision_velocity_ms,
            orbital_lifetime_years,
        )
        scale_levels.append((len(current), k_eff))

        if len(current) < block_factor:
            break

        # Coarse-grain: block average
        n = len(current)
        n_blocks = n // block_factor
        if n_blocks == 0:
            break
        truncated = current[:n_blocks * block_factor]
        current = truncated.reshape(n_blocks, block_factor).mean(axis=1)

    # Fixed point: the k_eff that the RG flow converges to
    # Use the last level (coarsest) as the fixed point estimate
    if len(scale_levels) >= 2:
        k_effs = [kv for _, kv in scale_levels]
        fixed_point = k_effs[-1]

        # RG eigenvalue: delta_k at level n+1 / delta_k at level n
        # near the fixed point
        deltas = [abs(k - fixed_point) for k in k_effs[:-1]]
        lambdas = []
        for i in range(len(deltas) - 1):
            if deltas[i] > 1e-20:
                lambdas.append(deltas[i + 1] / deltas[i])

        if lambdas and len(lambdas) > 0:
            avg_lambda = sum(lambdas) / len(lambdas)
        else:
            avg_lambda = 1.0

        # Critical exponent: nu = ln(b) / ln(lambda)
        if avg_lambda > 1e-20 and abs(math.log(max(avg_lambda, 1e-20))) > 1e-15:
            nu = abs(math.log(block_factor) / math.log(max(avg_lambda, 1e-20)))
        else:
            nu = 0.0

        # Correlation length: diverges as |k - 1|^(-nu)
        delta_k = abs(fixed_point - 1.0)
        if delta_k > 1e-10 and nu > 0:
            correlation_length = delta_k ** (-nu)
            # Cap at the original profile length
            correlation_length = min(correlation_length, float(len(profile)))
        else:
            correlation_length = float(len(profile))
    else:
        fixed_point = scale_levels[0][1] if scale_levels else 0.0
        nu = 0.0
        correlation_length = 0.0

    # Critical: k_eff within 10% of 1.0
    is_critical = 0.9 <= fixed_point <= 1.1

    return RenormalizationGroupAnalysis(
        fixed_point_k_eff=fixed_point,
        critical_exponent_nu=nu,
        correlation_length_cells=correlation_length,
        is_critical=is_critical,
        scale_levels=tuple(scale_levels),
    )


# ── P46: Reynolds Decomposition of Debris Density ──────────────────
#
# Decomposes debris density evolution into time-mean and fluctuation
# components. Computes turbulent kinetic energy analogue, turbulence
# intensity, and eddy diffusivity from the density gradient.


@dataclass(frozen=True)
class ReynoldsDebrisDecomposition:
    """Reynolds decomposition of debris density evolution.

    Attributes:
        mean_density: Time-mean density rho_bar(h) at each altitude bin.
        fluctuation_rms: RMS of density fluctuations rho'(h) at each altitude.
        turbulence_intensity: I(h) = sqrt(TKE(h)) / rho_bar(h) at each altitude.
        eddy_diffusivity: D_eddy(h) at each interior altitude bin (km^2/year).
        turbulent_kinetic_energy: TKE(h) = 0.5 * <rho'^2>_t at each altitude.
        mean_eddy_diffusivity: Spatial mean of eddy diffusivity (km^2/year).
    """
    mean_density: tuple[float, ...]
    fluctuation_rms: tuple[float, ...]
    turbulence_intensity: tuple[float, ...]
    eddy_diffusivity: tuple[float, ...]
    turbulent_kinetic_energy: tuple[float, ...]
    mean_eddy_diffusivity: float


def compute_reynolds_decomposition(
    density_evolution: DebrisDensityEvolution,
) -> ReynoldsDebrisDecomposition:
    """Compute Reynolds decomposition of debris density evolution.

    Decomposes the density field rho(h, t) into:
        rho(h, t) = rho_bar(h) + rho'(h, t)

    where rho_bar(h) is the time-mean and rho'(h, t) is the fluctuation.

    Turbulent kinetic energy analogue:
        TKE(h) = 0.5 * <rho'^2>_t

    Turbulence intensity:
        I(h) = sqrt(TKE(h)) / rho_bar(h)

    Eddy diffusivity from gradient-diffusion hypothesis:
        D_eddy(h) = -<rho' * v'> / (d rho_bar / dh)
    Approximated as: D_eddy(h) ~ <rho'^2>_t / |d rho_bar / dh|

    Args:
        density_evolution: Output from compute_debris_density_evolution.

    Returns:
        ReynoldsDebrisDecomposition with mean, fluctuation, and transport.

    Raises:
        ValueError: If density_evolution has fewer than 2 time steps.
    """
    n_times = len(density_evolution.time_steps)
    n_alt = len(density_evolution.altitude_bins)

    if n_times < 2:
        raise ValueError(
            f"Need >= 2 time steps for decomposition, got {n_times}"
        )

    # Build density array: shape (n_times, n_alt)
    rho = np.array(density_evolution.density_evolution, dtype=np.float64)

    # Time-mean density: rho_bar(h) = mean over time axis
    rho_bar = np.mean(rho, axis=0)

    # Fluctuation: rho'(h, t) = rho(h, t) - rho_bar(h)
    rho_prime = rho - rho_bar[np.newaxis, :]

    # Variance of fluctuations: <rho'^2>_t at each altitude
    rho_prime_sq_mean = np.mean(rho_prime ** 2, axis=0)

    # Turbulent kinetic energy: TKE(h) = 0.5 * <rho'^2>_t
    tke = 0.5 * rho_prime_sq_mean

    # Fluctuation RMS: sqrt(<rho'^2>_t)
    fluctuation_rms = np.sqrt(rho_prime_sq_mean)

    # Turbulence intensity: I(h) = sqrt(TKE(h)) / rho_bar(h)
    turbulence_intensity = np.zeros(n_alt)
    for i in range(n_alt):
        if rho_bar[i] > 1e-30:
            turbulence_intensity[i] = math.sqrt(tke[i]) / rho_bar[i]

    # Eddy diffusivity from gradient-diffusion hypothesis:
    # D_eddy(h) ~ <rho'^2>_t / |d rho_bar / dh|
    # Compute gradient of mean density using central differences
    alt = np.array(density_evolution.altitude_bins, dtype=np.float64)
    eddy_diffusivity = np.zeros(n_alt)
    for i in range(1, n_alt - 1):
        dh = alt[i + 1] - alt[i - 1]
        if dh > 1e-15:
            grad_rho = abs((rho_bar[i + 1] - rho_bar[i - 1]) / dh)
            if grad_rho > 1e-30:
                eddy_diffusivity[i] = rho_prime_sq_mean[i] / grad_rho

    # Mean eddy diffusivity (over interior points only)
    interior = eddy_diffusivity[1:-1] if n_alt > 2 else eddy_diffusivity
    mean_eddy = float(np.mean(interior)) if len(interior) > 0 else 0.0

    return ReynoldsDebrisDecomposition(
        mean_density=tuple(float(x) for x in rho_bar),
        fluctuation_rms=tuple(float(x) for x in fluctuation_rms),
        turbulence_intensity=tuple(float(x) for x in turbulence_intensity),
        eddy_diffusivity=tuple(float(x) for x in eddy_diffusivity),
        turbulent_kinetic_energy=tuple(float(x) for x in tke),
        mean_eddy_diffusivity=mean_eddy,
    )
