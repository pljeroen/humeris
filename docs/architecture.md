# Architecture

## Hexagonal design

The project follows hexagonal (ports and adapters) architecture. Business
logic lives in the domain layer with zero external dependencies. All I/O
flows through port interfaces implemented by adapters.

```
┌─────────────────────────────────────────────────┐
│                   Adapters                      │
│  CelesTrak · CesiumJS · CSV · GeoJSON · CZML    │
│  ViewerServer · JSON I/O                        │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │              Ports (Protocols)            │  │
│  │  SimulationReader · SimulationWriter      │  │
│  │  OrbitalDataSource · SatelliteExporter    │  │
│  │                                           │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │           Domain (stdlib only)      │  │  │
│  │  │                                     │  │  │
│  │  │  76 modules · math/datetime only    │  │  │
│  │  │  Zero external dependencies         │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Domain layer

Pure business logic. Python stdlib + NumPy (`math`, `datetime`, `dataclasses`,
`enum`, `typing`, `numpy`). Enforced by purity tests that parse AST and reject
any imports beyond stdlib and NumPy.

### Module categories (76 modules)

**MIT core (10 modules)**:

| Module | Purpose |
|--------|---------|
| `orbital_mechanics` | Kepler → Cartesian, SSO inclination, J2/J3 secular corrections |
| `constellation` | Walker shells, SSO bands, ShellConfig, Satellite |
| `coordinate_frames` | ECI ↔ ECEF ↔ Geodetic (GMST, Bowring, WGS84) |
| `propagation` | Keplerian + J2 analytical propagation |
| `coverage` | Grid-based visibility coverage analysis |
| `access_windows` | Satellite rise/set window detection |
| `ground_track` | Sub-satellite ground track computation |
| `observation` | Topocentric azimuth/elevation/range |
| `omm` | CelesTrak OMM record parsing |
| `serialization` | Simulation format (Y/Z swap, precision) |

**Commercial modules (66 modules)** — free for personal/educational/academic use:

| Category | Modules |
|----------|---------|
| Propagation | `numerical_propagation` (RK4, pluggable forces), `functorial_composition` (functorial force composition) |
| Analysis | `revisit`, `conjunction`, `eclipse`, `sensor`, `pass_analysis`, `constellation_metrics`, `dilution_of_precision`, `koopman_conjunction` (Koopman-spectral screening) |
| Design | `orbit_design`, `trade_study`, `multi_objective_design`, `design_optimization`, `design_sensitivity`, `orbit_properties`, `gramian_reconfiguration` (Gramian-guided reconfiguration) |
| Environment | `atmosphere`, `lifetime`, `station_keeping`, `deorbit`, `radiation`, `torques`, `third_body`, `solar` |
| Topology | `inter_satellite_links`, `link_budget`, `graph_analysis`, `information_theory`, `spectral_topology`, `hodge_cusum` (Hodge-CUSUM topology detection) |
| Composition | `mission_analysis`, `conjunction_management`, `communication_analysis`, `coverage_optimization`, `environment_analysis`, `maintenance_planning`, `mission_economics`, `constellation_operability`, `cascade_analysis`, `competing_risks` (competing-risks population dynamics) |
| Math | `linalg`, `control_analysis`, `statistical_analysis`, `relative_motion` |
| Research | `decay_analysis`, `temporal_correlation`, `operational_prediction`, `sp3_parser` |
| Maneuvers | `maneuvers` |
| Early Warning | `orbit_determination` (EKF), `maneuver_detection` (CUSUM/EWMA/chi-squared), `hazard_reporting` (NASA-STD-8719.14), `kessler_heatmap` (spatial density + cascade) |
| Fidelity | `time_systems` (AstroTime), `precession_nutation` (IAU 2006/2000B), `earth_orientation`, `planetary_ephemeris` (Chebyshev Sun/Moon), `nrlmsise00`, `adaptive_integration` (Dormand-Prince), `gravity_field` (EGM96), `relativistic_forces`, `tidal_forces`, `albedo_srp` |
| Propagation | `numerical_propagation` (RK4), `koopman_propagation` (DMD), `adaptive_integration` (RK4(5)) |

## Ports layer

Protocol interfaces (structural typing via `Protocol`). One port per concept.

| Port | Purpose | Implemented by |
|------|---------|----------------|
| `SimulationReader` | Read simulation template files | `JsonSimulationReader` |
| `SimulationWriter` | Write simulation output files | `JsonSimulationWriter` |
| `OrbitalDataSource` | Fetch orbital elements from external sources | `CelesTrakAdapter` |
| `SatelliteExporter` | Export satellite positions to file | `CsvSatelliteExporter`, `GeoJsonSatelliteExporter` |

## Adapters layer

External integrations. Import domain types only.

| Adapter | Purpose |
|---------|---------|
| `celestrak` | CelesTrak OMM API + SGP4 propagation |
| `concurrent_celestrak` | Threaded SGP4 propagation for large groups |
| `csv_exporter` | CSV export (lat/lon/alt) |
| `geojson_exporter` | GeoJSON FeatureCollection export |
| `czml_exporter` | CZML packets for CesiumJS |
| `czml_visualization` | Advanced CZML (eclipse, ISL, fragility, hazard, etc.) |
| `cesium_viewer` | Self-contained HTML viewer generation |
| `viewer_server` | Interactive HTTP server with analysis dispatch |

## Key constraints

1. **Domain purity**: No external dependencies in domain beyond stdlib + NumPy. Enforced by AST-parsing purity tests.
2. **Port isolation**: Adapters depend on domain, never the reverse.
3. **Immutable value objects**: Domain types use frozen dataclasses where appropriate.
4. **Result types**: No exceptions for control flow. Functions return success/failure values.
5. **Absolute imports**: Always `from humeris.domain.X import Y`, never relative.

## Dependency graph

```
cli.py
  └── adapters/
        ├── celestrak.py         → domain/omm, domain/propagation, domain/constellation
        ├── concurrent_celestrak → adapters/celestrak
        ├── csv_exporter         → domain/constellation, domain/coordinate_frames
        ├── geojson_exporter     → domain/constellation, domain/coordinate_frames
        ├── czml_exporter        → domain/propagation, domain/coordinate_frames
        ├── czml_visualization   → domain/* (many analysis modules)
        ├── cesium_viewer        → (standalone HTML generation)
        └── viewer_server        → adapters/czml_*, domain/constellation, domain/propagation
```
