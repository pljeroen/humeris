# CONTRACT: v1.17.0 — Cross-Domain Structural Analysis Layer

## Scope

7 new domain modules, ~25 functions, ~35 types, ~105 tests.
Version 1.16.0 → 1.17.0. Target: 1014 + 105 = ~1119 tests.
All functions compose existing domain capabilities with math from information theory,
graph theory, control theory, statistics, and reliability engineering. No new physics.
stdlib only. One linear algebra infrastructure module unlocks four findings.

## Findings → Functions Map

| Finding | Function | Module |
|---------|----------|--------|
| Infrastructure | `mat_multiply`, `mat_transpose`, `mat_eigenvalues_symmetric`, `mat_determinant`, `mat_inverse`, `naive_dft` | linalg |
| R1-1 ISL Algebraic Connectivity | `compute_topology_resilience` | graph_analysis |
| R2-1 Eclipse-Weighted Fragmentation | `compute_fragmentation_timeline` | graph_analysis |
| R1-2 Eclipse as BEC Channel | `compute_eclipse_channel_capacity` | information_theory |
| R1-4 Coverage Spectral Analysis | `compute_coverage_spectrum` | information_theory |
| R2-2 Marginal Satellite Value | `compute_marginal_satellite_value` | information_theory |
| R1-3 CW Controllability Gramian | `compute_cw_controllability` | control_analysis |
| R1-5 Analytical Collision Probability | `compute_analytical_collision_probability` | statistical_analysis |
| R1-6 Lifetime Survival Curve | `compute_lifetime_survival_curve` | statistical_analysis |
| R2-3 Mission Availability | `compute_mission_availability` | statistical_analysis |
| R2-5 Radiation-Eclipse Correlation | `compute_radiation_eclipse_correlation` | statistical_analysis |
| R1-7 DOP Fisher Information | `compute_positioning_information` | design_optimization |
| R2-4 Coverage Drift Rate | `compute_coverage_drift` | design_optimization |
| R2-6 Mass Efficiency Frontier | `compute_mass_efficiency_frontier` | design_optimization |

---

## Module 1: `domain/linalg.py`

**Purpose**: Pure linear algebra infrastructure. No orbital concepts. Unlocks eigenvalue
decomposition for graph analysis (R1-1, R2-1), control analysis (R1-3), DOP/Fisher
information (R1-7), and DFT for spectral analysis (R1-4).

### Types

```
Matrix = list[list[float]]  # type alias, NxN or NxM
EigenDecomposition(eigenvalues: tuple[float,...], eigenvectors: tuple[tuple[float,...],...])
DFTResult(frequencies_hz: tuple[float,...], magnitudes: tuple[float,...], phases_rad: tuple[float,...])
```

### Functions

```python
mat_zeros(n: int, m: int) → Matrix
mat_identity(n: int) → Matrix
mat_multiply(a: Matrix, b: Matrix) → Matrix
mat_transpose(a: Matrix) → Matrix
mat_add(a: Matrix, b: Matrix) → Matrix
mat_scale(a: Matrix, scalar: float) → Matrix
mat_eigenvalues_symmetric(a: Matrix, max_iterations: int = 100, tolerance: float = 1e-10) → EigenDecomposition
    # Jacobi eigenvalue algorithm for symmetric matrices
mat_determinant(a: Matrix) → float
    # LU decomposition for NxN
mat_inverse(a: Matrix) → Matrix
    # Gauss-Jordan elimination
mat_trace(a: Matrix) → float
naive_dft(signal: list[float], sample_rate_hz: float) → DFTResult
    # O(N²) DFT: C[k] = Σⱼ x[j]·exp(-2πijk/N)
```

### Math

**Jacobi eigenvalue algorithm** for real symmetric matrix A:
1. Find largest off-diagonal |A[p][q]|
2. Compute rotation angle θ = 0.5·atan2(2·A[p][q], A[p][p] - A[q][q])
3. Apply Givens rotation: A' = GᵀAG where G is rotation in (p,q) plane
4. Repeat until off-diagonal elements < tolerance
5. Diagonal elements are eigenvalues; accumulated rotations give eigenvectors

Convergence: quadratic for distinct eigenvalues. O(N³) per sweep, typically 5-10 sweeps.

**Naive DFT**:
```
C[k] = Σⱼ₌₀ᴺ⁻¹ x[j] · (cos(2πjk/N) - i·sin(2πjk/N))
magnitude[k] = √(Re(C[k])² + Im(C[k])²) / N
phase[k] = atan2(Im(C[k]), Re(C[k]))
frequency[k] = k · sample_rate / N
```

### Tests (18)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_identity_multiply | I·A = A |
| 2 | test_transpose_involution | (Aᵀ)ᵀ = A |
| 3 | test_multiply_dimensions | (NxM)·(MxK) = NxK |
| 4 | test_determinant_identity | det(I) = 1 |
| 5 | test_determinant_singular | det(singular) = 0 |
| 6 | test_determinant_2x2 | det([[a,b],[c,d]]) = ad-bc |
| 7 | test_inverse_identity | I⁻¹ = I |
| 8 | test_inverse_roundtrip | A·A⁻¹ ≈ I |
| 9 | test_eigenvalues_diagonal | eig(diag(1,2,3)) = (1,2,3) |
| 10 | test_eigenvalues_symmetric_2x2 | Known analytical result |
| 11 | test_eigenvalues_sorted | Returned sorted ascending |
| 12 | test_eigenvectors_orthogonal | vᵢ·vⱼ ≈ 0 for i≠j |
| 13 | test_eigendecomposition_reconstruct | A ≈ V·Λ·Vᵀ |
| 14 | test_trace_equals_eigenvalue_sum | tr(A) = Σλ |
| 15 | test_dft_constant_signal | DFT of constant = DC component only |
| 16 | test_dft_pure_sine | DFT of sin(2πft) peaks at f |
| 17 | test_dft_parseval | Σ|x|² = Σ|X|² (energy conservation) |
| 18 | test_module_pure | stdlib only |

---

## Module 2: `domain/graph_analysis.py`

**Findings**: R1-1, R2-1

**Core isomorphism**: ISL topology IS a weighted graph. The Laplacian eigenvalue λ₂
(Fiedler value) measures algebraic connectivity — the bottleneck of information flow.
Same eigenvalue computation as vibrational mode analysis of spring-mass networks.

### Math

Given ISL topology with N satellites and active links:
```
W[i][j] = SNR_linear(i,j) for active unblocked link between i and j
W[i][j] = W[i][j] × eclipse_power_fraction  if one endpoint eclipsed
W[i][j] = 0  otherwise

L[i][i] = Σⱼ W[i][j]     (degree)
L[i][j] = -W[i][j]         (i ≠ j)

Eigenvalues of L: 0 = λ₁ ≤ λ₂ ≤ ... ≤ λ_N
λ₂ = Fiedler value = algebraic connectivity
λ₂ > 0 iff graph is connected
Fiedler vector = eigenvector of λ₂ → identifies optimal bisection
```

### Types

```
TopologyResilience(
    fiedler_value: float,
    fiedler_vector: tuple[float, ...],
    num_components: int,
    is_connected: bool,
    spectral_gap: float,               # λ₃ - λ₂ (robustness of connectivity)
    total_capacity: float,              # sum of all link capacities
)

FragmentationEvent(
    time: datetime,
    fiedler_value: float,
    eclipsed_count: int,
    active_links: int,
)

FragmentationTimeline(
    events: tuple[FragmentationEvent, ...],
    min_fiedler_value: float,
    min_fiedler_time: datetime,
    fragmentation_count: int,           # number of times λ₂ ≤ 0
    mean_fiedler_value: float,
    resilience_margin: float,           # min λ₂ / mean λ₂
)
```

### Functions

```python
compute_topology_resilience(
    states: list[OrbitalState],
    time: datetime,
    link_config: LinkConfig,
    max_range_km: float = 5000.0,
    eclipse_power_fraction: float = 0.5,
) → TopologyResilience
    # Computes ISL topology → SNR weights → Laplacian → eigendecomposition → Fiedler value

compute_fragmentation_timeline(
    states: list[OrbitalState],
    link_config: LinkConfig,
    epoch: datetime,
    duration_s: float,
    step_s: float,
    max_range_km: float = 5000.0,
    eclipse_power_fraction: float = 0.5,
) → FragmentationTimeline
    # Time series of λ₂(t) with eclipse-degraded weights
```

### Composes

`compute_isl_topology` + `compute_link_budget` + `is_eclipsed` + `sun_position_eci` + `propagate_to` + `mat_eigenvalues_symmetric` from linalg

### Tests (12)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_resilience_returns_type | TopologyResilience type |
| 2 | test_two_connected_sats_positive_fiedler | 2 sats in range → λ₂ > 0 |
| 3 | test_single_sat_zero_fiedler | 1 sat → λ₂ = 0 |
| 4 | test_fiedler_vector_length | len(vector) = N satellites |
| 5 | test_connected_graph_flag | is_connected = True when λ₂ > 0 |
| 6 | test_spectral_gap_nonneg | spectral_gap ≥ 0 |
| 7 | test_fragmentation_returns_type | FragmentationTimeline type |
| 8 | test_fragmentation_event_count | len(events) = expected steps |
| 9 | test_min_fiedler_leq_mean | min ≤ mean |
| 10 | test_resilience_margin_range | margin ∈ [0, 1] or 0 if mean = 0 |
| 11 | test_fragmentation_time_set | min_fiedler_time is a datetime |
| 12 | test_module_pure | stdlib + domain only |

---

## Module 3: `domain/information_theory.py`

**Findings**: R1-2, R1-4, R2-2

### Math

**BEC Channel Capacity (R1-2)**:
```
ε = eclipse_fraction(state, epoch)
C_awgn = compute_link_budget(config, distance).max_data_rate_bps
C_bec = (1 - ε) · C_awgn
Scheduling gain = C_scheduled / C_bec  (≥ 1 with eclipse foreknowledge)
```

**Coverage Spectrum (R1-4)**:
Binary coverage signal c(t) at a grid point → DFT → power spectral density.
```
c[j] = 1 if any satellite visible at time step j, 0 otherwise
PSD[k] = |DFT(c)[k]|² / N²
Dominant frequency reveals orbital resonance.
resonance_ratio = dominant_freq / orbital_freq
```

**Marginal Satellite Value (R2-2)**:
Information gain from adding satellite N+1:
```
I_coverage = H(coverage_{N+1}) - H(coverage_N)
  where H = -Σ p_i · log₂(p_i)  (Shannon entropy of coverage distribution)
I_positioning = log₂(det(FIM_{N+1})) - log₂(det(FIM_N))
  where FIM = HᵀH from DOP geometry matrix
I_total = α · I_coverage + β · I_positioning
```

### Types

```
EclipseChannelCapacity(
    awgn_capacity_bps: float,
    erasure_fraction: float,
    bec_capacity_bps: float,
    scheduled_throughput_bps: float,
    scheduling_gain: float,
)

CoverageSpectrum(
    lat_deg: float,
    lon_deg: float,
    frequencies_hz: tuple[float, ...],
    power_density: tuple[float, ...],
    dominant_frequency_hz: float,
    dominant_period_s: float,
    orbital_frequency_hz: float,
    resonance_ratio: float,
)

MarginalSatelliteValue(
    coverage_entropy_before: float,
    coverage_entropy_after: float,
    coverage_information_gain: float,
    positioning_info_gain: float,
    total_information_value: float,
)
```

### Functions

```python
compute_eclipse_channel_capacity(
    state: OrbitalState,
    epoch: datetime,
    link_config: LinkConfig,
    distance_m: float,
) → EclipseChannelCapacity

compute_coverage_spectrum(
    states: list[OrbitalState],
    start: datetime,
    duration_s: float,
    step_s: float,
    lat_deg: float,
    lon_deg: float,
    min_elevation_deg: float = 10.0,
) → CoverageSpectrum

compute_marginal_satellite_value(
    states: list[OrbitalState],
    candidate: OrbitalState,
    epoch: datetime,
    duration_s: float = 5400.0,
    step_s: float = 60.0,
    min_elevation_deg: float = 10.0,
    lat_step_deg: float = 30.0,
    lon_step_deg: float = 30.0,
) → MarginalSatelliteValue
```

### Composes

`eclipse_fraction` + `compute_link_budget` + `predict_eclipse_seasons` + `compute_eclipse_windows` + `propagate_ecef_to` + `compute_observation` + `compute_dop` + `compute_revisit` + `naive_dft` from linalg + `mat_determinant` from linalg

### Tests (14)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_bec_capacity_returns_type | EclipseChannelCapacity type |
| 2 | test_bec_capacity_leq_awgn | bec_capacity ≤ awgn_capacity |
| 3 | test_bec_zero_eclipse_equals_awgn | ε=0 → bec = awgn |
| 4 | test_scheduling_gain_geq_one | gain ≥ 1.0 |
| 5 | test_spectrum_returns_type | CoverageSpectrum type |
| 6 | test_spectrum_has_orbital_peak | dominant near orbital frequency |
| 7 | test_spectrum_frequencies_positive | all f ≥ 0 |
| 8 | test_spectrum_parseval | energy conservation in DFT |
| 9 | test_resonance_ratio_positive | resonance_ratio > 0 |
| 10 | test_marginal_value_returns_type | MarginalSatelliteValue type |
| 11 | test_marginal_coverage_gain_nonneg | coverage_information_gain ≥ 0 |
| 12 | test_marginal_positioning_gain_nonneg | positioning_info_gain ≥ 0 |
| 13 | test_total_value_positive | total_information_value > 0 when adding useful sat |
| 14 | test_module_pure | stdlib + domain only |

---

## Module 4: `domain/control_analysis.py`

**Finding**: R1-3

**Core isomorphism**: The CW equations ARE a linear time-invariant system.
The state transition matrix Φ(t) from `cw_propagate_state` IS the system
propagator. The controllability Gramian eigenvalues reveal fuel cost anisotropy
for proximity operations.

### Math

```
CW system: ẋ = A·x + B·u,  B = I₆

State transition matrix Φ(t) is computed by cw_propagate_state with unit initial conditions.

Controllability Gramian:
W_c(T) = ∫₀ᵀ Φ(τ)·Φᵀ(τ) dτ ≈ Σᵢ Φ(iΔt)·Φᵀ(iΔt)·Δt

Properties:
- rank(W_c) = 6 → fully controllable
- eigenvalues → control effort per direction
- condition_number = λ_max / λ_min → fuel cost anisotropy
- Minimum-fuel transfer to state x: cost ∝ xᵀ · W_c⁻¹ · x
```

### Types

```
ControllabilityAnalysis(
    gramian_eigenvalues: tuple[float, ...],
    gramian_eigenvectors: tuple[tuple[float, ...], ...],
    is_controllable: bool,
    condition_number: float,
    min_energy_direction: tuple[float, ...],
    max_energy_direction: tuple[float, ...],
    gramian_trace: float,
)
```

### Functions

```python
compute_cw_controllability(
    n_rad_s: float,
    duration_s: float,
    step_s: float = 10.0,
) → ControllabilityAnalysis
    # Numerically integrates Φ(τ)·Φᵀ(τ) using cw_propagate_state, then eigendecompose
```

### Composes

`cw_propagate_state` + `mat_multiply` + `mat_transpose` + `mat_eigenvalues_symmetric` + `mat_inverse` from linalg

### Tests (10)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_controllability_returns_type | ControllabilityAnalysis type |
| 2 | test_controllable_full_rank | is_controllable = True (6 nonzero eigenvalues) |
| 3 | test_eigenvalues_positive | All eigenvalues > 0 |
| 4 | test_eigenvalues_sorted | Ascending order |
| 5 | test_condition_number_geq_one | condition_number ≥ 1.0 |
| 6 | test_min_energy_is_along_track | Smallest eigenvalue ≈ along-track (CW free drift) |
| 7 | test_longer_duration_better_controllability | More time → lower condition number |
| 8 | test_gramian_symmetric | W_c eigenvalues all real (symmetric matrix) |
| 9 | test_trace_equals_eigenvalue_sum | tr(W_c) = Σλ |
| 10 | test_module_pure | stdlib + domain only |

---

## Module 5: `domain/statistical_analysis.py`

**Findings**: R1-5, R1-6, R2-3, R2-5

### Math

**Analytical Collision Probability (R1-5)** — Marcum Q approximation:
```
d = √(b_r² + b_c²)     (miss distance from B-plane)
σ = √((σ_r² + σ_c²)/2) (average sigma)
For centered equal-sigma case: P_c = 1 - exp(-r²/(2σ²))
General: P_c via series expansion of modified Bessel I₀
normalized_miss = d / σ   (the single governing parameter)
```

**Lifetime Survival Curve (R1-6)**:
```
Given decay profile {(t_i, a_i)} from compute_orbit_lifetime:
S(t) = (T_total - t) / T_total           (fraction of lifetime remaining)
h(t) = |da/dt| / (a(t) - a_reentry)      (hazard rate — instantaneous failure rate)
Half-life altitude: a where S = 0.5
```

**Mission Availability (R2-3)**:
```
A(t) = P_fuel(t) · P_power(t) · P_conjunction(t)
P_fuel = max(0, 1 - cumulative_prop(t) / budget)
P_power = 1 - eclipse_fraction(t)
P_conjunction = exp(-λ_conj · t)  (Poisson survival)
Mission reliability R = (1/T)∫₀ᵀ A(t) dt
```

**Radiation-Eclipse Correlation (R2-5)**:
```
Pearson r = [NΣxy - ΣxΣy] / √[(NΣx² - (Σx)²)(NΣy² - (Σy)²)]
x = monthly dose rates, y = monthly eclipse fractions
r > 0 → benign (eclipse shields during high-radiation)
r < 0 → dangerous (high radiation with low eclipse shielding)
```

### Types

```
CollisionProbabilityAnalytical(
    numerical_pc: float,
    analytical_pc: float,
    relative_error: float,
    normalized_miss_distance: float,
)

LifetimeSurvivalCurve(
    times: tuple[datetime, ...],
    altitudes_km: tuple[float, ...],
    survival_fraction: tuple[float, ...],
    hazard_rate_per_day: tuple[float, ...],
    half_life_altitude_km: float,
    mean_remaining_life_days: float,
)

MissionAvailabilityProfile(
    times: tuple[datetime, ...],
    fuel_availability: tuple[float, ...],
    power_availability: tuple[float, ...],
    conjunction_survival: tuple[float, ...],
    total_availability: tuple[float, ...],
    mission_reliability: float,
    critical_factor: str,
)

RadiationEclipseCorrelation(
    monthly_doses_rad_s: tuple[float, ...],
    monthly_eclipse_fractions: tuple[float, ...],
    monthly_beta_angles_deg: tuple[float, ...],
    dose_eclipse_correlation: float,
    dose_beta_correlation: float,
    is_benign_correlation: bool,
    worst_month: int,
)
```

### Functions

```python
compute_analytical_collision_probability(
    miss_distance_m: float,
    b_radial_m: float,
    b_cross_m: float,
    sigma_radial_m: float,
    sigma_cross_m: float,
    combined_radius_m: float,
) → CollisionProbabilityAnalytical

compute_lifetime_survival_curve(
    lifetime_result: OrbitLifetimeResult,
) → LifetimeSurvivalCurve

compute_mission_availability(
    state: OrbitalState,
    drag_config: DragConfig,
    epoch: datetime,
    isp_s: float,
    dry_mass_kg: float,
    propellant_budget_kg: float,
    mission_years: float = 5.0,
    conjunction_rate_per_year: float = 0.1,
) → MissionAvailabilityProfile

compute_radiation_eclipse_correlation(
    state: OrbitalState,
    epoch: datetime,
    num_months: int = 12,
) → RadiationEclipseCorrelation
```

### Composes

`collision_probability_2d` + `compute_b_plane` + `compute_orbit_lifetime` + `compute_propellant_profile` + `eclipse_fraction` + `compute_orbit_radiation_summary` + `compute_beta_angle` + `drag_compensation_dv_per_year` + `semi_major_axis_decay_rate`

### Tests (18)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_analytical_pc_returns_type | CollisionProbabilityAnalytical type |
| 2 | test_analytical_pc_centered_known | d=0, equal σ → P_c = 1-exp(-r²/2σ²) |
| 3 | test_analytical_pc_far_miss_low | Large miss → low probability |
| 4 | test_normalized_miss_positive | normalized_miss_distance ≥ 0 |
| 5 | test_survival_returns_type | LifetimeSurvivalCurve type |
| 6 | test_survival_monotone_decreasing | S(t) decreases over time |
| 7 | test_survival_starts_at_one | S(0) ≈ 1.0 |
| 8 | test_hazard_rate_positive | All hazard rates > 0 |
| 9 | test_hazard_rate_increases | h(t) increases as altitude drops |
| 10 | test_half_life_in_range | half_life within altitude bounds |
| 11 | test_availability_returns_type | MissionAvailabilityProfile type |
| 12 | test_availability_starts_high | A(0) near 1.0 |
| 13 | test_availability_decreases | A(t) generally decreases over mission |
| 14 | test_reliability_range | 0 ≤ mission_reliability ≤ 1 |
| 15 | test_critical_factor_valid | critical_factor in {"fuel", "power", "conjunction"} |
| 16 | test_correlation_returns_type | RadiationEclipseCorrelation type |
| 17 | test_correlation_range | -1 ≤ r ≤ 1 |
| 18 | test_module_pure | stdlib + domain only |

---

## Module 6: `domain/design_optimization.py`

**Findings**: R1-7, R2-4, R2-6

### Math

**DOP as Fisher Information (R1-7)**:
```
FIM = HᵀH  (already computed inside compute_dop)
D-optimality = det(FIM)^(1/N)  where N = 4 (unknowns)
CRLB_position = σ_meas · PDOP
Information efficiency = 1 / GDOP² (relative to ideal geometry)
```

**Coverage Drift Rate (R2-4)**:
```
Differential RAAN rate sensitivity:
∂(dΩ/dt)/∂a = -7/2 · (dΩ/dt)_nominal / a

For altitude error Δa between planes:
d(ΔRAAN)/dt = ∂(dΩ/dt)/∂a · Δa

Coverage half-life: T_half = Δcov_threshold / |d(cov)/dt|
estimated via finite-difference coverage evaluation
```

**Mass Efficiency Frontier (R2-6)**:
```
ΔV_total(alt) = ΔV_raise(alt) + ΔV_SK(alt)·T_mission
M_wet(alt) = m_dry · exp(ΔV_total / (Isp·g₀))   [Tsiolkovsky]
M_constellation(alt) = N_sats · M_wet(alt)
Efficiency(alt) = Coverage(alt) / M_constellation(alt)
Mass wall: altitude where M_wet grows 10× from minimum
```

### Types

```
PositioningInformationMetric(
    dop_result: DOPResult,
    fisher_determinant: float,
    d_optimal_criterion: float,
    crlb_position_m: float,
    information_efficiency: float,
)

CoverageDriftAnalysis(
    raan_sensitivity_rad_s_per_m: float,
    coverage_drift_rate_per_s: float,
    coverage_half_life_s: float,
    maintenance_interval_s: float,
)

MassEfficiencyPoint(
    altitude_km: float,
    total_dv_ms: float,
    wet_mass_kg: float,
    constellation_mass_kg: float,
    mass_efficiency: float,
)

MassEfficiencyFrontier(
    points: tuple[MassEfficiencyPoint, ...],
    optimal_altitude_km: float,
    peak_efficiency: float,
    mass_wall_altitude_km: float,
)
```

### Functions

```python
compute_positioning_information(
    lat_deg: float,
    lon_deg: float,
    sat_positions_ecef: list[tuple[float, float, float]],
    sigma_measurement_m: float = 1.0,
    min_elevation_deg: float = 10.0,
) → PositioningInformationMetric

compute_coverage_drift(
    states: list[OrbitalState],
    epoch: datetime,
    altitude_error_m: float = 100.0,
    coverage_threshold: float = 0.05,
    duration_s: float = 5400.0,
    step_s: float = 60.0,
) → CoverageDriftAnalysis

compute_mass_efficiency_frontier(
    drag_config: DragConfig,
    isp_s: float,
    dry_mass_kg: float,
    injection_altitude_km: float,
    mission_years: float,
    num_sats: int,
    alt_min_km: float = 300.0,
    alt_max_km: float = 800.0,
    alt_step_km: float = 25.0,
) → MassEfficiencyFrontier
```

### Composes

`compute_dop` + `mat_determinant` from linalg + `j2_raan_rate` + `compute_revisit` or `compute_single_coverage_fraction` + `hohmann_transfer` + `drag_compensation_dv_per_year` + `propellant_mass_for_dv` + `pareto_front_indices`

### Tests (15)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_positioning_info_returns_type | PositioningInformationMetric type |
| 2 | test_fisher_det_positive | fisher_determinant > 0 with ≥4 sats |
| 3 | test_d_optimal_positive | d_optimal_criterion > 0 |
| 4 | test_crlb_scales_with_sigma | Higher σ → higher CRLB |
| 5 | test_information_efficiency_range | 0 < efficiency ≤ 1 |
| 6 | test_drift_returns_type | CoverageDriftAnalysis type |
| 7 | test_drift_sensitivity_negative | ∂(dΩ/dt)/∂a < 0 (higher a → slower precession) |
| 8 | test_drift_half_life_positive | coverage_half_life_s > 0 |
| 9 | test_larger_error_faster_drift | Bigger altitude error → shorter half-life |
| 10 | test_mass_frontier_returns_type | MassEfficiencyFrontier type |
| 11 | test_mass_frontier_has_peak | peak_efficiency > efficiency at endpoints |
| 12 | test_mass_wall_below_optimal | mass_wall_altitude_km < optimal_altitude_km |
| 13 | test_wet_mass_increases_at_extremes | Extreme altitudes → high wet mass |
| 14 | test_efficiency_positive | All mass_efficiency > 0 |
| 15 | test_module_pure | stdlib + domain only |

---

## Module 7: Purity Enforcement

Each of the 6 domain modules above has a purity test (counted in module tests above).
The linalg module has a standalone purity test.

Total purity tests: 7 (one per module).

---

## Integration

### `__init__.py` additions

New imports from 7 modules. New entries in `__all__`.
Total new exports: ~35 types + ~25 functions ≈ 60 new symbols.

### `pyproject.toml`

Version: `"1.16.0"` → `"1.17.0"`

### `__init__.py` `__version__`

`"1.16.0"` → `"1.17.0"`

---

## Implementation Order

1. Write `tests/test_linalg.py` (18 tests) → RED
2. Implement `domain/linalg.py` → GREEN
3. Write `tests/test_graph_analysis.py` (12 tests) → RED
4. Implement `domain/graph_analysis.py` → GREEN
5. Write `tests/test_information_theory.py` (14 tests) → RED
6. Implement `domain/information_theory.py` → GREEN
7. Write `tests/test_control_analysis.py` (10 tests) → RED
8. Implement `domain/control_analysis.py` → GREEN
9. Write `tests/test_statistical_analysis.py` (18 tests) → RED
10. Implement `domain/statistical_analysis.py` → GREEN
11. Write `tests/test_design_optimization.py` (15 tests) → RED
12. Implement `domain/design_optimization.py` → GREEN
13. Update `__init__.py` exports + version
14. Update `pyproject.toml` version
15. Full regression: ~1119 tests pass

---

## Test Count

| Module | Tests |
|--------|-------|
| linalg | 18 |
| graph_analysis | 12 |
| information_theory | 14 |
| control_analysis | 10 |
| statistical_analysis | 18 |
| design_optimization | 15 |
| **Total new** | **87** |
| **Existing** | **1014** |
| **Target** | **~1101** |

---

## Dependency Graph

```
linalg (foundation — no orbital imports)
  ├── graph_analysis (uses linalg eigenvalues)
  ├── information_theory (uses linalg DFT + determinant)
  ├── control_analysis (uses linalg eigenvalues + matrix ops)
  └── design_optimization (uses linalg determinant)

statistical_analysis (no linalg dependency — pure statistics)
```

All 7 modules depend only on stdlib + existing domain modules. No circular dependencies.
