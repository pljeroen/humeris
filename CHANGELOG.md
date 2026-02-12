# Changelog

All notable changes to this project are documented here.

## [Unreleased]

- Documentation suite (`docs/` folder)
- GitHub license detection fix (standard MIT LICENSE)
- Simulation JSON schema documentation
- CelesTrak reproducibility documentation

## [1.19.0] - 2026-02-12

- Interactive viewer server with 13 analysis layer types
- 5 research domain modules: decay analysis, temporal correlation,
  operational prediction, design sensitivity, SP3 parser
- 81 validation tests (Vallado, SGP4, SP3, internal cross-checks)
- Viewer server HTTP API (REST, CORS-enabled)
- All 8 analysis dispatch types wired (sensor, ISL, fragility, hazard,
  network eclipse, coverage connectivity, precession, conjunction)
- Default RF link config (Ka-band 26 GHz), sensor config (30deg circular),
  drag config, performance caps for O(n^2) analyses

**Tests**: 1384 passing

## [1.18.0] - 2026-02-12

- 27 new domain modules: ISL topology, DOP, pass analysis, constellation
  metrics, relative motion, link budget, third-body perturbations, torques,
  radiation environment, linalg (Jacobi eigensolver, DFT), graph analysis,
  information theory, control analysis, statistical analysis, design
  optimization, spectral topology, constellation operability, mission
  economics, multi-objective design, cascade analysis, and 9 cross-domain
  composition modules (mission analysis, conjunction management,
  communication analysis, coverage optimization, environment analysis,
  maintenance planning)

**Tests**: 1146 passing

## [1.12.0] - 2026-02-12

- Self-contained Cesium HTML viewer with plane coloring
- CZML visualization: eclipse, sensor footprint, ground station, conjunction
  replay, ISL topology, coverage evolution, precession, fragility, hazard,
  network eclipse, coverage connectivity

## [1.11.0] - 2026-02-12

- Sensor/payload FOV modeling (circular, rectangular, custom polygon)
- 163 formal invariant tests (energy, momentum, vis-viva, frame round-trips)

**Tests**: 467 passing (630 including invariants)

## [1.10.0] - 2026-02-12

- Numerical propagation bridges (analytical-to-numerical state handoff)
- Shadow-aware solar radiation pressure force model

**Tests**: 436 passing

## [1.9.0] - 2026-02-12

- RK4 numerical propagator with pluggable force models
- Force models: two-body, J2, J3, atmospheric drag, SRP

## [1.8.0] - 2026-02-12

- CZML exporter for CesiumJS 3D visualization
- Animated constellation, snapshot, ground track, coverage CZML packets

## [1.7.0] - 2026-02-12

- Time-domain revisit analysis (mean/max revisit, coverage fraction,
  mean response time)
- Parametric Walker trade studies with Pareto front extraction

## [1.6.0] - 2026-02-12

- Solar ephemeris (analytical Sun position, declination)
- Eclipse prediction (shadow geometry, beta angle, eclipse windows)
- Orbit transfer maneuvers (Hohmann, bi-elliptic, plane change, phasing)
- Deorbit compliance estimation (FCC 5-year / ESA 25-year)
- Orbit design (SSO/LTAN, frozen orbit, repeat ground track)

## [1.5.0] - 2026-02-12

- Atmospheric drag model (exponential density, configurable solar activity)
- Orbit lifetime prediction with decay profile
- Station-keeping delta-V budgets (drag compensation, plane maintenance)
- Conjunction screening, TCA refinement, B-plane geometry, collision
  probability

## [1.4.0] - 2026-02-12

- J2/J3 secular perturbations (RAAN drift, argument of perigee drift)
- Analytical propagation (Keplerian + J2)
- Topocentric observation (azimuth, elevation, slant range)
- Access window detection (satellite rise/set from ground station)
- Grid-based coverage analysis

## [1.3.0] - 2026-02-12

- Ground track computation (sub-satellite trace with optional J2)
- CSV and GeoJSON satellite position export
- Coordinate frame conversions (ECI, ECEF, Geodetic via Bowring/WGS84)
- Concurrent CelesTrak adapter (threaded SGP4 propagation)

## [1.0.0] - 2026-02-11

- Initial release
- Walker constellation shell generation
- CelesTrak OMM fetching with SGP4 propagation
- Simulation JSON I/O (Y/Z axis swap)
- Hexagonal architecture (domain/ports/adapters)
