# CONTRACT: v1.16.0 — Cross-Domain Composition Layer

## Scope

6 new domain modules, 24 composition functions, ~50 types, ~76 tests.
Version 1.15.0 → 1.16.0. Target: 944 + 76 = 1020 tests.
All functions compose existing domain capabilities. No new physics. stdlib only.

## Findings → Functions Map

| Finding | Function | Module |
|---------|----------|--------|
| 1.1 Lifetime + SK → propellant curve | `compute_propellant_profile` | mission_analysis |
| 1.2 Beta + Thermal → seasonal thermal | `compute_seasonal_thermal_profile` | environment_analysis |
| 1.3 GT Crossings + Access → station siting | `compute_optimal_ground_stations` | coverage_optimization |
| 1.4 Relative Motion + Conjunction → triage | `triage_conjunction` | conjunction_management |
| 1.5 Radiation + Orbit Design → LTAN | `compute_radiation_optimized_ltan` | environment_analysis |
| 1.6 Ascending Nodes + ISL → distance | `predict_isl_distances` | communication_analysis |
| 1.7 Sensor + DOP → quality coverage | `compute_quality_weighted_coverage` | coverage_optimization |
| 1.8 Doppler + Link Budget → true rate | `compute_pass_data_throughput` | communication_analysis |
| 1.9 Eclipse Seasons + Orbit Design | `compute_eclipse_free_windows` | environment_analysis |
| 1.10 Torques + Eclipse → worst-case | `compute_worst_case_torque_timing` | environment_analysis |
| 2.1 Eclipse + Radiation + Beta → dose | `compute_seasonal_dose_profile` | environment_analysis |
| 2.2 ISL + Link + Eclipse → degraded | `compute_eclipse_degraded_topology` | communication_analysis |
| 2.3 Drag + Lifetime + Hohmann → alt | `compute_optimal_altitude` | mission_analysis |
| 2.4 Crossings + Sensor + Revisit | `compute_crossing_revisit` | coverage_optimization |
| 2.5 Trade + Deorbit + Lifetime | `compute_compliant_trade_study` | coverage_optimization |
| 2.6 CW + Phasing + Conjunction | `compute_avoidance_maneuver` | conjunction_management |
| 2.7 Doppler + Link + Access → throughput | `compute_pass_data_throughput` | communication_analysis |
| 2.8 All Forces + Elements → budget | `compute_perturbation_budget` | maintenance_planning |
| 3.1 Health timeline (6 modules) | `compute_health_timeline` | mission_analysis |
| 3.2 Network capacity (5 modules) | `compute_network_capacity_timeline` | communication_analysis |
| 3.3 Mission cost model (6 modules) | `compute_mission_cost_metric` | mission_analysis |
| 3.4 Optimal EO design (6 modules) | `compute_optimal_eo_ltan` | coverage_optimization |
| 3.5 Conjunction pipeline (5 modules) | `run_conjunction_decision_pipeline` | conjunction_management |
| 3.6 Maintenance schedule (7 modules) | `compute_maintenance_schedule` | maintenance_planning |

---

## Module 1: `domain/mission_analysis.py`

**Findings**: 1.1, 2.3, 3.1, 3.3

### Types

```
PropellantPoint(time, altitude_km, dv_per_year_ms, propellant_per_year_kg, cumulative_propellant_kg)
PropellantProfile(points: tuple[PropellantPoint,...], total_propellant_kg, depletion_time: datetime|None)
HealthSnapshot(time, altitude_km, cumulative_dose_rad, cumulative_thermal_cycles, cumulative_propellant_kg)
HealthTimeline(snapshots: tuple[HealthSnapshot,...], limiting_factor: str, end_of_life_time: datetime|None)
AltitudeTradePoint(altitude_km, raising_dv_ms, sk_dv_ms, total_dv_ms)
AltitudeOptimization(points: tuple[AltitudeTradePoint,...], optimal_altitude_km, minimum_dv_ms)
MissionCostMetric(total_dv_ms, wet_mass_per_sat_kg, total_constellation_mass_kg, coverage_pct, cost_per_coverage_point)
```

### Functions

```python
compute_propellant_profile(decay_profile: tuple[DecayPoint,...], drag_config, isp_s, dry_mass_kg, propellant_budget_kg) → PropellantProfile
compute_health_timeline(state, drag_config, epoch, mission_years=5.0, radiation_limit_rad=10000.0, thermal_limit_cycles=50000) → HealthTimeline
compute_optimal_altitude(drag_config, isp_s, dry_mass_kg, injection_altitude_km, mission_years, alt_min_km=300, alt_max_km=800, alt_step_km=25) → AltitudeOptimization
compute_mission_cost_metric(config: WalkerConfig, drag_config, isp_s, dry_mass_kg, injection_altitude_km, mission_years, coverage_result: CoverageResult) → MissionCostMetric
```

### Tests (13)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_propellant_profile_returns_type | PropellantProfile type |
| 2 | test_propellant_accelerates | Later points have higher dv_per_year |
| 3 | test_propellant_depletion_detected | depletion_time set when budget exceeded |
| 4 | test_health_timeline_returns_type | HealthTimeline type |
| 5 | test_health_dose_increases | Cumulative dose monotonically increases |
| 6 | test_health_limiting_factor | Identifies correct limiting factor |
| 7 | test_optimal_altitude_returns_type | AltitudeOptimization type |
| 8 | test_optimal_altitude_finds_minimum | optimal_altitude_km at minimum total_dv |
| 9 | test_optimal_low_alt_high_sk | Low altitude → high SK dV |
| 10 | test_mission_cost_returns_type | MissionCostMetric type |
| 11 | test_mission_cost_positive | All costs > 0 |
| 12 | test_mission_cost_higher_alt_more_raising | Higher alt → more raising dV |
| 13 | test_module_pure | stdlib + domain only |

---

## Module 2: `domain/conjunction_management.py`

**Findings**: 1.4, 2.6, 3.5

### Types

```
ConjunctionAction(Enum): NO_ACTION, MANEUVER, ACCEPT_RISK
ConjunctionTriage(action: ConjunctionAction, is_passively_safe: bool, relative_state: RelativeState, maneuver_dv_ms: float)
AvoidanceManeuver(delta_v_ms, along_track_dv_ms, lead_time_s, post_maneuver_miss_m)
ConjunctionDecision(event: ConjunctionEvent, triage: ConjunctionTriage, maneuver: AvoidanceManeuver|None, fuel_sufficient: bool)
```

### Functions

```python
triage_conjunction(state1, state2, tca: datetime, min_safe_distance_m=1000.0, check_periods=2) → ConjunctionTriage
compute_avoidance_maneuver(state1, state2, tca: datetime, target_miss_m=1000.0) → AvoidanceManeuver
run_conjunction_decision_pipeline(events: list[ConjunctionEvent], states: dict[str, OrbitalState], fuel_budgets_ms: dict[str, float], min_safe_distance_m=1000.0) → list[ConjunctionDecision]
```

### Tests (9)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_triage_returns_type | ConjunctionTriage type |
| 2 | test_triage_safe_no_action | Well-separated → NO_ACTION |
| 3 | test_triage_close_maneuver | Close approach → MANEUVER |
| 4 | test_avoidance_returns_type | AvoidanceManeuver type |
| 5 | test_avoidance_dv_positive | delta_v > 0 |
| 6 | test_avoidance_larger_miss_less_dv | Closer target miss → more dV |
| 7 | test_pipeline_returns_list | list[ConjunctionDecision] |
| 8 | test_pipeline_empty_events | Empty list → empty result |
| 9 | test_module_pure | stdlib + domain only |

---

## Module 3: `domain/communication_analysis.py`

**Findings**: 1.6, 1.8, 2.2, 2.7, 3.2

### Types

```
DegradedLink(link: ISLLink, budget: LinkBudgetResult, is_eclipsed_a: bool, is_eclipsed_b: bool, has_positive_margin: bool)
EclipseDegradedTopology(links: tuple[DegradedLink,...], active_link_count, degraded_link_count, total_link_count, capacity_fraction)
PassDataPoint(time: datetime, snr_db, doppler_hz, data_rate_bps, slant_range_km)
PassThroughput(window: AccessWindow, data_points: tuple[PassDataPoint,...], total_bytes, effective_rate_fraction, peak_data_rate_bps)
ISLDistancePrediction(plane_pairs: tuple[tuple[int,int],...], predicted_distances_m: tuple[float,...], node_spacing_deg)
NetworkCapacitySnapshot(time: datetime, active_isl_count, degraded_isl_count, ground_contact_count, eclipsed_sat_count)
NetworkCapacityTimeline(snapshots: tuple[NetworkCapacitySnapshot,...], min_active_isl_count, min_capacity_time: datetime, mean_active_isl_count)
```

### Functions

```python
compute_eclipse_degraded_topology(states: list[OrbitalState], time: datetime, link_config: LinkConfig, eclipse_power_fraction=0.5, max_range_km=5000.0) → EclipseDegradedTopology
compute_pass_data_throughput(station: GroundStation, state: OrbitalState, window: AccessWindow, link_config: LinkConfig, freq_hz: float, step_s=10.0) → PassThroughput
predict_isl_distances(node_longitudes_deg: list[float], altitude_km: float) → ISLDistancePrediction
compute_network_capacity_timeline(states: list[OrbitalState], stations: list[GroundStation], link_config: LinkConfig, epoch: datetime, duration_s: float, step_s: float, max_range_km=5000.0) → NetworkCapacityTimeline
```

### Tests (12)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_degraded_topology_returns_type | EclipseDegradedTopology type |
| 2 | test_degraded_capacity_leq_full | capacity_fraction ≤ 1.0 |
| 3 | test_degraded_counts_consistent | active + degraded ≤ total |
| 4 | test_pass_throughput_returns_type | PassThroughput type |
| 5 | test_pass_throughput_bytes_positive | total_bytes > 0 |
| 6 | test_pass_effective_rate_leq_one | effective_rate_fraction ≤ 1.0 |
| 7 | test_isl_distances_returns_type | ISLDistancePrediction type |
| 8 | test_isl_distances_positive | All distances > 0 |
| 9 | test_isl_spacing_matches_input | node_spacing_deg consistent |
| 10 | test_network_timeline_returns_type | NetworkCapacityTimeline type |
| 11 | test_network_timeline_snapshot_count | Correct number of snapshots |
| 12 | test_module_pure | stdlib + domain only |

---

## Module 4: `domain/coverage_optimization.py`

**Findings**: 1.3, 1.7, 2.4, 2.5, 3.4

### Types

```
QualityCoveragePoint(lat_deg, lon_deg, num_visible: int, gdop: float, is_usable: bool)
QualityCoverageResult(points: tuple[QualityCoveragePoint,...], usable_fraction, mean_gdop)
CrossingRevisitResult(crossing_revisit_s, elsewhere_revisit_s, improvement_factor, num_crossings: int)
CompliantTradePoint(trade_point: TradePoint, lifetime_days, is_compliant: bool, deorbit_dv_ms)
CompliantTradeResult(all_points: tuple[CompliantTradePoint,...], compliant_count: int, pareto_indices: tuple[int,...])
EOLTANPoint(ltan_hours, effective_revisit_s, eclipse_free_days_per_year, annual_dose_rad)
EOLTANOptimization(points: tuple[EOLTANPoint,...], optimal_ltan_hours)
StationCandidate(lat_deg, lon_deg, total_contact_s, num_passes: int)
GroundStationOptimization(candidates: tuple[StationCandidate,...], ranked_indices: tuple[int,...])
```

### Functions

```python
compute_quality_weighted_coverage(states: list[OrbitalState], time: datetime, sensor: SensorConfig, lat_step_deg=10.0, lon_step_deg=10.0, gdop_threshold=6.0) → QualityCoverageResult
compute_crossing_revisit(track: list[GroundTrackPoint], states: list[OrbitalState], start: datetime, duration: timedelta, step: timedelta, min_elevation_deg=10.0) → CrossingRevisitResult
compute_compliant_trade_study(trade_result: TradeStudyResult, drag_config: DragConfig, epoch: datetime, max_lifetime_years=5.0) → CompliantTradeResult
compute_optimal_eo_ltan(altitude_km: float, epoch: datetime, ltan_values: list[float], analysis_duration_s=5400.0, analysis_step_s=60.0) → EOLTANOptimization
compute_optimal_ground_stations(track: list[GroundTrackPoint], states: list[OrbitalState], start: datetime, duration: timedelta, step: timedelta, min_elevation_deg=10.0, num_candidates=10) → GroundStationOptimization
```

### Tests (15)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_quality_coverage_returns_type | QualityCoverageResult type |
| 2 | test_quality_usable_fraction_range | 0 ≤ usable_fraction ≤ 1 |
| 3 | test_quality_gdop_positive | mean_gdop > 0 |
| 4 | test_crossing_revisit_returns_type | CrossingRevisitResult type |
| 5 | test_crossing_improvement_geq_one | improvement_factor ≥ 1 (or 0 if no crossings) |
| 6 | test_crossing_count_nonnegative | num_crossings ≥ 0 |
| 7 | test_compliant_trade_returns_type | CompliantTradeResult type |
| 8 | test_compliant_leq_all | compliant_count ≤ len(all_points) |
| 9 | test_compliant_pareto_subset | pareto_indices are valid indices |
| 10 | test_eo_ltan_returns_type | EOLTANOptimization type |
| 11 | test_eo_ltan_in_range | optimal_ltan within provided values |
| 12 | test_ground_stations_returns_type | GroundStationOptimization type |
| 13 | test_ground_stations_ranked | ranked_indices valid |
| 14 | test_ground_stations_contact_positive | total_contact_s > 0 for at least one |
| 15 | test_module_pure | stdlib + domain only |

---

## Module 5: `domain/environment_analysis.py`

**Findings**: 1.2, 1.5, 1.9, 1.10, 2.1

### Types

```
ThermalMonth(month: int, mean_beta_deg, cycle_count: int, mean_eclipse_duration_s, mean_sunlit_duration_s)
SeasonalThermalProfile(months: tuple[ThermalMonth,...], total_annual_cycles: int, max_cycle_month: int, min_cycle_month: int)
DoseSnapshot(time: datetime, dose_rate_rad_s, cumulative_dose_rad, eclipse_fraction, beta_deg)
SeasonalDoseProfile(snapshots: tuple[DoseSnapshot,...], annual_dose_rad, max_dose_rate_rad_s, mean_eclipse_fraction)
RadiationLTANPoint(ltan_hours, annual_dose_rad, saa_fraction)
RadiationLTANResult(points: tuple[RadiationLTANPoint,...], optimal_ltan_hours, min_annual_dose_rad)
EclipseFreeLTANPoint(ltan_hours, eclipse_free_days, total_eclipse_season_days)
EclipseFreeResult(points: tuple[EclipseFreeLTANPoint,...], optimal_ltan_hours, max_eclipse_free_days)
TorqueBoundary(time: datetime, is_eclipse_entry: bool, gg_torque_before_nm: float, gg_torque_after_nm: float, aero_torque_before_nm: float, aero_torque_after_nm: float, total_discontinuity_nm: float)
TorqueTimingResult(boundaries: tuple[TorqueBoundary,...], max_discontinuity_nm, worst_case_time: datetime)
```

### Functions

```python
compute_seasonal_thermal_profile(state: OrbitalState, epoch: datetime, raan_drift_rad_s=0.0) → SeasonalThermalProfile
compute_seasonal_dose_profile(state: OrbitalState, epoch: datetime, duration_days=365.0, step_days=30.0) → SeasonalDoseProfile
compute_radiation_optimized_ltan(altitude_km: float, epoch: datetime, ltan_values: list[float] | None = None) → RadiationLTANResult
compute_eclipse_free_windows(altitude_km: float, epoch: datetime, ltan_values: list[float] | None = None) → EclipseFreeResult
compute_worst_case_torque_timing(state: OrbitalState, inertia: InertiaTensor, drag_config: DragConfig, cp_offset_m: tuple[float,float,float], epoch: datetime, duration_s: float, step_s: float) → TorqueTimingResult
```

### Tests (14)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_seasonal_thermal_returns_type | SeasonalThermalProfile type |
| 2 | test_seasonal_thermal_12_months | len(months) == 12 |
| 3 | test_seasonal_thermal_cycles_vary | Not all months same cycles |
| 4 | test_seasonal_dose_returns_type | SeasonalDoseProfile type |
| 5 | test_seasonal_dose_positive | annual_dose_rad > 0 |
| 6 | test_seasonal_dose_cumulative_increases | Monotonically increasing |
| 7 | test_radiation_ltan_returns_type | RadiationLTANResult type |
| 8 | test_radiation_ltan_in_range | optimal_ltan in provided values |
| 9 | test_eclipse_free_returns_type | EclipseFreeResult type |
| 10 | test_eclipse_free_days_range | 0 ≤ days ≤ 365 |
| 11 | test_torque_timing_returns_type | TorqueTimingResult type |
| 12 | test_torque_discontinuity_positive | max_discontinuity_nm > 0 |
| 13 | test_torque_boundaries_at_eclipse | Boundaries correspond to eclipse times |
| 14 | test_module_pure | stdlib + domain only |

---

## Module 6: `domain/maintenance_planning.py`

**Findings**: 2.8, 3.6

### Types

```
ElementPerturbation(element_name: str, j2_rate: float, drag_rate: float, total_rate: float, dominant_source: str)
PerturbationBudget(elements: tuple[ElementPerturbation,...], altitude_km: float)
MaintenanceBurn(time: datetime, element: str, delta_v_ms: float, description: str)
MaintenanceSchedule(burns: tuple[MaintenanceBurn,...], total_dv_per_year_ms, burn_frequency_per_year: int, dominant_correction: str)
```

### Functions

```python
compute_perturbation_budget(state: OrbitalState, drag_config: DragConfig | None = None) → PerturbationBudget
compute_maintenance_schedule(state: OrbitalState, drag_config: DragConfig, epoch: datetime, altitude_tolerance_km=5.0, mission_duration_days=365.0) → MaintenanceSchedule
```

### Tests (7)

| # | Test | Verifies |
|---|------|----------|
| 1 | test_perturbation_budget_returns_type | PerturbationBudget type |
| 2 | test_perturbation_has_elements | At least RAAN, arg_perigee, SMA |
| 3 | test_perturbation_dominant_identified | dominant_source is set |
| 4 | test_maintenance_returns_type | MaintenanceSchedule type |
| 5 | test_maintenance_burns_exist | len(burns) > 0 for low alt |
| 6 | test_maintenance_dv_positive | total_dv_per_year_ms > 0 |
| 7 | test_module_pure | stdlib + domain only |

---

## Integration

### `__init__.py` additions

New imports from 6 modules. New entries in `__all__`.
Total new exports: ~50 (24 functions + ~26 types).

### `pyproject.toml`

Version: `"1.15.0"` → `"1.16.0"`

### `__init__.py` `__version__`

`"1.15.0"` → `"1.16.0"`

---

## Implementation Order

1. Write all 6 test files → RED
2. Implement all 6 domain modules → GREEN
3. Update __init__.py exports + version
4. Update pyproject.toml version
5. Full regression: 1020 tests pass
