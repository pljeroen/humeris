# Copyright (c) 2026 Jeroen. All rights reserved.
"""Design optimization — DOP/Fisher information, coverage drift, mass efficiency frontier.

Reinterprets DOP geometry as Fisher Information Matrix, computes RAAN drift
sensitivity for coverage maintenance, and maps the Tsiolkovsky mass wall.

No external dependencies — only stdlib + domain modules.
"""
import math
from dataclasses import dataclass
from datetime import datetime

from constellation_generator.domain.propagation import OrbitalState, propagate_ecef_to
from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.dilution_of_precision import compute_dop
from constellation_generator.domain.orbital_mechanics import OrbitalConstants, j2_raan_rate
from constellation_generator.domain.station_keeping import drag_compensation_dv_per_year, propellant_mass_for_dv
from constellation_generator.domain.maneuvers import hohmann_transfer
from constellation_generator.domain.revisit import compute_single_coverage_fraction


@dataclass(frozen=True)
class PositioningInformationMetric:
    """DOP reinterpreted as Fisher Information."""
    gdop: float
    pdop: float
    fisher_determinant: float
    d_optimal_criterion: float
    crlb_position_m: float
    information_efficiency: float


@dataclass(frozen=True)
class CoverageDriftAnalysis:
    """Sensitivity of coverage to RAAN drift from altitude errors."""
    raan_sensitivity_rad_s_per_m: float
    coverage_drift_rate_per_s: float
    coverage_half_life_s: float
    maintenance_interval_s: float


@dataclass(frozen=True)
class MassEfficiencyPoint:
    """Single point on the mass efficiency frontier."""
    altitude_km: float
    total_dv_ms: float
    wet_mass_kg: float
    constellation_mass_kg: float
    mass_efficiency: float


@dataclass(frozen=True)
class MassEfficiencyFrontier:
    """Tsiolkovsky mass efficiency across altitude range."""
    points: tuple
    optimal_altitude_km: float
    peak_efficiency: float
    mass_wall_altitude_km: float


def compute_positioning_information(
    lat_deg: float,
    lon_deg: float,
    sat_positions_ecef: list,
    sigma_measurement_m: float = 1.0,
    min_elevation_deg: float = 10.0,
) -> PositioningInformationMetric:
    """Reinterpret DOP geometry as Fisher Information Matrix.

    FIM = H^T H, where H is the geometry matrix from DOP computation.
    D-optimality = det(FIM)^(1/N) where N=4 unknowns.
    CRLB_position = sigma_meas * PDOP.
    Information efficiency = 1 / GDOP^2.
    """
    dop = compute_dop(lat_deg, lon_deg, sat_positions_ecef, min_elevation_deg)

    # FIM ≈ inverse of covariance. det(Q) = GDOP^2-related quantity.
    # For 4x4 Q = (H^T H)^-1, det(FIM) = 1/det(Q).
    # det(Q) can be approximated as GDOP^4 (for 4 unknowns, Q has 4 dimensions).
    # More precisely: GDOP^2 = tr(Q) and det(Q) ≤ (tr(Q)/4)^4

    if dop.gdop > 0 and dop.gdop < float('inf'):
        # Approximate det(FIM) from DOP components
        # Q diagonal approx: σ_x^2, σ_y^2, σ_z^2, σ_t^2
        # σ_x^2 + σ_y^2 ≈ HDOP^2, σ_z^2 ≈ VDOP^2, σ_t^2 ≈ TDOP^2
        # det(Q) ≈ (HDOP^2/2)^2 * VDOP^2 * TDOP^2 (rough)
        # Use simpler: det(FIM) ≈ (1/GDOP)^8 is too aggressive
        # Best available: det(FIM) = 1/(PDOP^2 * TDOP^2) × geometric factor
        # Simplify: use GDOP directly
        gdop_sq = dop.gdop ** 2
        fisher_det = 1.0 / (gdop_sq ** 2)  # 1/GDOP^4 for 4D
        d_optimal = fisher_det ** 0.25  # det^(1/4)
        crlb = sigma_measurement_m * dop.pdop
        efficiency = 1.0 / gdop_sq
    else:
        fisher_det = 0.0
        d_optimal = 0.0
        crlb = float('inf')
        efficiency = 0.0

    return PositioningInformationMetric(
        gdop=dop.gdop,
        pdop=dop.pdop,
        fisher_determinant=fisher_det,
        d_optimal_criterion=d_optimal,
        crlb_position_m=crlb,
        information_efficiency=min(1.0, efficiency),
    )


def compute_coverage_drift(
    states: list,
    epoch: datetime,
    altitude_error_m: float = 100.0,
    coverage_threshold: float = 0.05,
    duration_s: float = 5400.0,
    step_s: float = 60.0,
) -> CoverageDriftAnalysis:
    """Compute coverage sensitivity to J2 RAAN drift from altitude errors.

    dΩ/dt sensitivity to semi-major axis:
    ∂(dΩ/dt)/∂a = -7/2 · (dΩ/dt)_nominal / a

    Coverage half-life: time until coverage degrades by threshold.
    """
    if not states:
        return CoverageDriftAnalysis(
            raan_sensitivity_rad_s_per_m=0.0,
            coverage_drift_rate_per_s=0.0,
            coverage_half_life_s=float('inf'),
            maintenance_interval_s=float('inf'),
        )

    # Use first satellite as reference
    ref = states[0]
    a = ref.semi_major_axis_m
    n = ref.mean_motion_rad_s
    inc = ref.inclination_rad

    # J2 RAAN rate and its sensitivity to semi-major axis
    raan_rate = j2_raan_rate(n, a, 0.0, inc)
    # ∂(dΩ/dt)/∂a = -7/2 · (dΩ/dt) / a
    raan_sensitivity = -3.5 * raan_rate / a

    # Differential RAAN drift from altitude error
    d_raan_rate = raan_sensitivity * altitude_error_m

    # Coverage drift: approximate as linear degradation
    # Compute baseline coverage
    cov_baseline = compute_single_coverage_fraction(
        states, epoch, lat_step_deg=30.0, lon_step_deg=30.0,
    )

    if cov_baseline < 1e-10:
        return CoverageDriftAnalysis(
            raan_sensitivity_rad_s_per_m=raan_sensitivity,
            coverage_drift_rate_per_s=0.0,
            coverage_half_life_s=float('inf'),
            maintenance_interval_s=float('inf'),
        )

    # Coverage drift rate: proportional to differential RAAN drift
    # One radian of RAAN shift ≈ 1/N_planes of coverage cell width
    n_planes = max(1, len(set(s.raan_rad for s in states)))
    coverage_sensitivity = 1.0 / (2.0 * math.pi / n_planes)
    coverage_drift = abs(d_raan_rate) * coverage_sensitivity

    # Half-life: time until coverage degrades by threshold fraction
    if coverage_drift > 1e-20:
        half_life = coverage_threshold / coverage_drift
    else:
        half_life = float('inf')

    # Maintenance interval: half of half-life (for proactive maintenance)
    maintenance = half_life / 2.0 if half_life < float('inf') else float('inf')

    return CoverageDriftAnalysis(
        raan_sensitivity_rad_s_per_m=raan_sensitivity,
        coverage_drift_rate_per_s=coverage_drift,
        coverage_half_life_s=half_life,
        maintenance_interval_s=maintenance,
    )


def compute_mass_efficiency_frontier(
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    injection_altitude_km: float,
    mission_years: float,
    num_sats: int,
    alt_min_km: float = 300.0,
    alt_max_km: float = 800.0,
    alt_step_km: float = 25.0,
) -> MassEfficiencyFrontier:
    """Map the Tsiolkovsky mass efficiency frontier across altitudes.

    ΔV_total(alt) = ΔV_raise(alt) + ΔV_SK(alt) * T_mission
    M_wet(alt) = m_dry * exp(ΔV_total / (Isp*g0))
    Efficiency(alt) = 1 / (M_wet * num_sats)  [higher = better]
    Mass wall: altitude where M_wet grows 10x from minimum.
    """
    g0 = 9.80665
    r_e = OrbitalConstants.R_EARTH
    points = []

    num_alt_steps = max(0, int(round((alt_max_km - alt_min_km) / alt_step_km)))
    for alt_idx in range(num_alt_steps + 1):
        alt = alt_min_km + alt_idx * alt_step_km
        r_injection = r_e + injection_altitude_km * 1000.0
        r_operational = r_e + alt * 1000.0

        # Raise maneuver dV
        if abs(alt - injection_altitude_km) > 0.1:
            transfer = hohmann_transfer(r_injection, r_operational)
            dv_raise = transfer.total_delta_v_ms
        else:
            dv_raise = 0.0

        # Station-keeping dV over mission
        dv_sk_year = drag_compensation_dv_per_year(alt, drag_config)
        max_dv_sk = isp_s * g0 * 15.0  # Cap to prevent overflow
        dv_sk_total = min(dv_sk_year, max_dv_sk) * mission_years

        dv_total = dv_raise + dv_sk_total

        # Tsiolkovsky: m_wet = m_dry * exp(dv/(Isp*g0))
        exponent = dv_total / (isp_s * g0)
        if exponent > 20.0:
            exponent = 20.0  # Cap to prevent overflow
        wet_mass = dry_mass_kg * math.exp(exponent)
        constellation_mass = wet_mass * num_sats

        # Efficiency = inverse of total mass (higher = better)
        efficiency = 1.0 / constellation_mass if constellation_mass > 0 else 0.0

        points.append(MassEfficiencyPoint(
            altitude_km=alt,
            total_dv_ms=dv_total,
            wet_mass_kg=wet_mass,
            constellation_mass_kg=constellation_mass,
            mass_efficiency=efficiency,
        ))

    if not points:
        return MassEfficiencyFrontier(
            points=(), optimal_altitude_km=0.0,
            peak_efficiency=0.0, mass_wall_altitude_km=0.0,
        )

    # Find optimal altitude (peak efficiency)
    best_idx = max(range(len(points)), key=lambda i: points[i].mass_efficiency)
    optimal_alt = points[best_idx].altitude_km
    peak_eff = points[best_idx].mass_efficiency

    # Mass wall: lowest altitude where wet mass > 10x minimum wet mass
    min_wet = min(p.wet_mass_kg for p in points)
    mass_wall_alt = points[0].altitude_km
    for p in points:
        if p.wet_mass_kg > 10.0 * min_wet:
            mass_wall_alt = p.altitude_km
            break

    # If no mass wall found (all within 10x), set to min altitude
    if mass_wall_alt == points[0].altitude_km:
        mass_wall_found = False
        for p in points:
            if p.wet_mass_kg > 10.0 * min_wet:
                mass_wall_alt = p.altitude_km
                mass_wall_found = True
                break
        if not mass_wall_found:
            mass_wall_alt = points[0].altitude_km

    return MassEfficiencyFrontier(
        points=tuple(points),
        optimal_altitude_km=optimal_alt,
        peak_efficiency=peak_eff,
        mass_wall_altitude_km=mass_wall_alt,
    )
