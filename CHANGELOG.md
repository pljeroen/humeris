# Changelog

All notable changes to this project are documented here.

## [1.26.0] - 2026-02-14

### NRLMSISE-00 fidelity improvements

- **Kp↔Ap conversion** — `kp_to_ap()` and `ap_to_kp()` using Bartels (1957)
  28-entry quasi-logarithmic table with linear interpolation. Clamped to valid ranges.
- **Solar cycle uncertainty bounds** — `SolarCyclePrediction` gains `f107_upper` and
  `f107_lower` fields (+1σ/−1σ envelope). `_SolarCycleParams` gains `amplitude_upper`
  and `amplitude_lower`. Published ±1σ ranges for cycles 23-26.
- **7-element Ap array** — `SpaceWeather` gains optional `ap_array` tuple for
  time-weighted geomagnetic history (Picone 2002 g0/sg0/sumex weighting in
  `_exospheric_temperature()`). Falls back to scalar g0(Ap) when not provided.
  Magnetic activity factor uses 3h current Ap with g0 nonlinear saturation.
- **Nonlinear geomagnetic saturation** — Implemented Picone (2002) g0/sg0/sumex
  functions from NRLMSISE-00 reference. g0(Ap) provides nonlinear saturation at
  extreme Ap values (prevents unrealistic density enhancement). Parameters from
  reference pt[] array: α=0.00513, β=0.0867. Applied to both exospheric temperature
  and magnetic activity factor.
- **Hathaway (2015) full form** — Solar cycle shape function upgraded from simplified
  `x³·exp(-x²)` (c=0) to full `x³/(exp(x²)-c)` with asymmetry parameter c=0.8.
  Numerical peak finding via bisection (60 iterations, ~18 digits precision).
  Produces heavier declining-phase tail matching observed cycle behavior.
- **ITU-R P.371 coefficient fix** — SSN→F10.7 proxy linear coefficient corrected
  from 0.727 to 0.728 per published standard. Attribution corrected from
  "Tapping (2013)" to "ITU-R P.371-8".
- **Calibrated synthetic space weather data** — Replaced prior generated data with
  calibrated synthetic dataset including 27-day Bartels rotation modulation,
  day-to-day variability, and Poisson-like geomagnetic storm episodes. Uses full
  Hathaway c=0.8 model. Deterministic generator (seed=20260214). Marked as
  "calibrated_synthetic" provenance.
- **36 new tests** — Kp↔Ap (8), uncertainty bounds (7), 7-element Ap (8),
  calibrated data (7), nonlinear geomagnetic (6). Total: 3226 passing.

## [1.25.0] - 2026-02-13

### Monorepo split

- **Two-package layout** — Split `src/` into `packages/core/` (MIT, humeris-core) and
  `packages/pro/` (Commercial, humeris-pro) using PEP 420 implicit namespace packages.
  Both install into the shared `humeris` namespace via `pip install -e ./packages/core -e ./packages/pro`
- **Package pyproject.toml** — Each package has its own pyproject.toml with correct
  metadata, dependencies, and entry points

### Solar activity model (STK/GMAT parity)

- **Hathaway solar cycle prediction** — `predict_solar_activity()` in `solar.py` using
  Hathaway (2015) parametric shape function SSN(t) = A * x³ * exp(-x²) / peak_norm.
  Cycle parameters for cycles 23-26 from NOAA/SWPC observations
- **Tapping (2013) SSN→F10.7 proxy** — F10.7 = 63.7 + 0.727·SSN + 0.00089·SSN²
- **SpaceWeatherProvider protocol** — Structural typing for pluggable space weather
  sources (historical, predicted, composite)
- **Composite provider** — Seamless historical→predicted transition at configurable
  crossover date, with module-level caching for performance
- **atmospheric_density_nrlmsise00()** — One-call convenience function with automatic
  space weather lookup, cached model instance
- **density_func callback** — `Callable[[float, datetime], float]` wired through
  `compute_orbit_lifetime()`, `compute_altitude_at_time()`, `compute_stochastic_lifetime()`,
  `drag_compensation_dv_per_year()`, `compute_station_keeping_budget()`,
  `compute_gve_station_keeping_budget()`, `compute_propellant_pharmacokinetics()`,
  `compute_exponential_scale_map()`, `compute_perturbation_budget()`,
  `compute_maintenance_schedule()` — all with proper epoch propagation
- **Enrichment adapter** — Prefers NRLMSISE-00 when available with graceful fallback
  to exponential model
- **Viewer server** — Variable atmosphere wired into hazard, station-keeping, and
  maintenance layer dispatches
- **Space weather data** — Daily-resolution historical data (11,323 entries, 2000-2030)

### Research modules (from 5-pass cross-disciplinary exploration)

- **Functorial force composition** — Category-theoretic framework for composing
  force models with functorial guarantees (identity, composition, associativity)
- **Hodge-CUSUM topology detector** — Detects ISL topology changes via Hodge Laplacian
  spectral tracking with CUSUM change-point detection
- **Gramian-guided reconfiguration** — Controllability Gramian-based satellite
  reconfiguration planning with energy-optimal maneuver sequencing
- **Koopman-spectral conjunction screening** — DMD-based spectral screening for
  conjunction events using Koopman operator eigenfunction proximity
- **Competing-risks population dynamics** — Multi-risk (drag, collision, component
  failure, fuel depletion) population model with cause-specific hazard rates and
  cumulative incidence functions

### Bug fixes

- Fixed viewer_server dispatch crash: `station_keeping_packets()` and
  `maintenance_schedule_packets()` received positional arg that conflicted with `name`
  parameter (TypeError at runtime)
- Fixed hardcoded epoch (`datetime(2024,1,1)`) in density_func paths — epoch now
  properly propagated through station_keeping, maintenance_planning, and decay_analysis
- Fixed enrichment.py epoch check (`effective_epoch != _J2000` → `epoch is not None`)

### Documentation

- 15 research papers across 3 tiers documenting novel algorithmic contributions
- Research algorithms reference guide
- Updated API reference, architecture docs, and getting started guide

### Infrastructure

- 3 rounds of math verification (fixes to Coriolis, R_EARTH, decay formulas,
  Gramian eigendecomposition, coverage quadrature, graph Laplacian)
- 3 rounds of cross-disciplinary creative exploration (30 proposals evaluated,
  5 Tier 1 + 5 Tier 2 + 5 Tier 3 implemented)
- Module-level caching for SpaceWeatherProvider and NRLMSISE00Model singletons

**Tests**: 3190 passing (+1033 from v1.23.0)

## [1.23.0] - 2026-02-13

### Mathematical corrections (from 3-pass verification against Vallado, Montenbruck & Gill, IERS)

- **Coriolis term** — ECI-to-ECEF velocity transformation now includes Earth rotation
  correction: `v_ECEF = R * v_ECI - omega_E × r_ECEF` (was missing ~465 m/s term)
- **Equatorial radius** — J2/J3/SSO perturbation formulas now use WGS84 equatorial
  radius (6378.137 km) instead of mean radius (6371 km), fixing ~0.1% systematic error
- **Propagation guard** — `propagate_to` raises `ValueError` for eccentricity > 1e-6
  (linear true anomaly advance is only valid for circular orbits)
- **EARTH_OMEGA** — SSO rate constant updated to `1.99098659e-7` (was truncated to `1.99e-7`)

### New capabilities (from cross-disciplinary creative exploration)

- **SIR cascade model** — Epidemiological (SIR) dynamics for debris cascade prediction
  with R_0, time-to-peak, equilibrium debris population, and full S/I/R time series.
  Maps Kessler heatmap parameters to epidemic rates (beta from collision cross-section,
  gamma from drag lifetime)
- **FTLE conjunction risk** — Finite-Time Lyapunov Exponent for conjunction predictability
  classification. High FTLE = chaotic sensitivity (widen margins), low FTLE = reliable
  prediction (reduce margins). Uses finite-difference STM via SVD
- **Koopman propagator** — Dynamic Mode Decomposition for fast long-term propagation.
  Fits Koopman operator from numerical propagation snapshots, predicts via matrix power.
  100-1000x speedup after training for J2-dominated orbits
- **Hodge Laplacian** — Higher-order ISL topology analysis. L1 edge Laplacian from
  triangle boundary operators, beta_1 (independent routing cycles), L1 spectral gap
  (routing redundancy). Distinguishes "connected but fragile" from "connected and resilient"
- **Energy monitoring** — Numerical propagation tracks specific orbital energy (v²/2 - μ/r)
  per step with initial/final energy, max drift, and relative energy drift in results
- **Thermal equilibrium** — Beta-angle driven spacecraft thermal analysis using
  Stefan-Boltzmann energy balance

### Project

- Renamed from `constellation-generator` to `humeris`
- Added origin story in README

**Tests**: 2157 passing (+97 from v1.22.0)

## [1.22.0] - 2026-02-13

### Early warning and hazard analysis

- **Hazard reporting** — NASA-STD-8719.14 hazard classification (ROUTINE/WARNING/CRITICAL),
  covariance-unavailable path with tighter miss-distance thresholds, secondary maneuver
  escalation, circuit-breaker hysteresis for de-escalation, catastrophic breakup potential
  (E > 40 kJ/kg), Conjunction Weather Index (FWI-inspired composite), HMAC-SHA256
  provenance signing
- **Maneuver detection** — Two-sided CUSUM (Page 1954, Montgomery 2013 parameters h=5.0
  k=0.5), EWMA detector for low-thrust maneuvers (Roberts 1959), chi-squared windowed
  variance test with correct DOF, self-starting rolling baseline (Hawkins 1987), EKF
  innovation variance support, ARL₀ estimation (Siegmund 1985), d' sensitivity index
- **Kessler heatmap** — Altitude × inclination spatial density grid, corrected collision
  velocity (Kessler 1978: V_circ·√2·sin(i_mid)), spherical zone volume fraction,
  Shannon population entropy, percolation fraction with 2D threshold (Stauffer & Aharony),
  nuclear criticality k_eff cascade indicator, Lyapunov exponent estimate, temporal
  persistence tracking for chronic hotspots
- **Orbit determination** — Extended Kalman Filter with Joseph-stabilized covariance
  update, innovation variance field for downstream CUSUM

### Cross-disciplinary patterns implemented (from 90-discipline analysis)

- Statistical Process Control (CUSUM/EWMA parameters from manufacturing quality control)
- Percolation theory (2D lattice threshold from condensed matter physics)
- Nuclear criticality (k_eff multiplication factor from reactor physics)
- Signal detection theory (d' sensitivity from psychophysics)
- Circuit breaker pattern (hysteresis from electrical engineering)
- Forensic accounting (HMAC chain-of-custody from audit science)
- Fire Weather Index structure (composite risk index from forestry)
- STPA hazard analysis (covariance gap from systems safety)

### Viewer integration

- `kessler_heatmap` analysis layer type (altitude × inclination grid visualization)
- `conjunction_hazard` analysis layer type (hazard-classified conjunction screening)
- 15 total analysis layer types (was 13)

### Infrastructure

- NumPy vectorization across all domain modules
- Domain purity allowlist extended for `hmac` (stdlib)

**Tests**: 2060 passing

## [1.21.0] - 2026-02-13

- NumPy big-bang upgrade across all 71 domain modules
- Vectorized linear algebra, orbital mechanics, and analysis computations

**Tests**: 1883 passing

## [1.20.0] - 2026-02-13

### NASA-grade fidelity upgrade (7 phases)

- **Time systems** — AstroTime value object with UTC/TAI/TT/TDB/GPS conversions,
  leap second table, Fairhead & Bretagnon 1990 TDB approximation
- **Precession-nutation** — IAU 2006 Fukushima-Williams precession, IAU 2000B 77-term
  nutation, GCRS↔ITRS frame chain, Earth Rotation Angle
- **Earth orientation** — EOP loader/interpolator (IERS finals2000A 2000-2030),
  UT1-UTC, polar motion, full GCRS→ITRS with W·R3(ERA)·N·P·B
- **Planetary ephemeris** — Compact Chebyshev Sun/Moon (2000-2050, ~120KB),
  Clenshaw recurrence, ~100m accuracy vs DE440
- **NRLMSISE-00** — Full atmosphere model with solar activity (F10.7, Ap),
  species densities, diurnal/seasonal variations
- **Adaptive integration** — Dormand-Prince RK4(5) with FSAL, PI step controller,
  dense output via Hermite interpolation
- **Force models** — Schwarzschild + Lense-Thirring + de Sitter relativistic,
  solid Earth tides (IERS 2010 Love numbers), FES2004 ocean tides,
  Earth albedo + IR radiation pressure

### Simulator integrations

- Celestia .ssc exporter
- Google Earth KML exporter
- Blender Python script exporter
- Stellarium TLE exporter
- Kerbal Space Program exporter (Kerbin scaling)

**Tests**: 1882 passing

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
