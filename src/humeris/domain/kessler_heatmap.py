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
    # with minimum for near-equatorial orbits
    v_rel = v_circ * math.sqrt(2.0) * max(
        math.sin(mid_inc_rad), 0.05
    )
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
    """Compute cascade multiplication factor k_eff.

    Analogous to nuclear criticality (neutron multiplication factor):
        k_eff = mean_fragments * P_collision_per_fragment

    where P_collision_per_fragment is the probability that a single
    fragment itself collides within one orbital lifetime.

    k_eff < 1: subcritical (debris removed faster than produced)
    k_eff = 1: critical (steady state)
    k_eff > 1: supercritical (runaway cascade)

    Uses spatial density and collision velocity from cells to estimate
    P_collision = density * sigma * V_rel * lifetime.
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
