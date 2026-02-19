# Licensing

## Dual-license model

This project uses a dual-license model: MIT open core with commercial
extended modules. The two licenses are distributed as separate pip packages
sharing the `humeris` namespace via PEP 420 implicit namespace packages.

### Package structure

| Package | License | Install |
|---------|---------|---------|
| `humeris-core` | MIT | `pip install humeris-core` |
| `humeris-pro` | Commercial | `pip install humeris-pro` |

`humeris-pro` depends on `humeris-core`. Installing pro automatically
installs core. Core works standalone for constellation generation,
Keplerian propagation, coverage analysis, and export.

### MIT (core) — `packages/core/`

The foundation is MIT-licensed. Copyright "Jeroen Visser".

Covers 10 domain modules, 13 adapters, 3 ports, CLI, and their tests:

**Domain**: `orbital_mechanics`, `constellation`, `coordinate_frames`,
`propagation`, `coverage`, `access_windows`, `ground_track`, `observation`,
`omm`, `serialization`

**Adapters**: `json_io`, `enrichment`, `celestrak`, `concurrent_celestrak`,
`csv_exporter`, `geojson_exporter`, `kml_exporter`, `blender_exporter`,
`stellarium_exporter`, `celestia_exporter`, `spaceengine_exporter`,
`ksp_exporter`, `ubox_exporter`

**Ports**: `SimulationReader`, `SimulationWriter`, `OrbitalDataSource`,
`SatelliteExporter`

Use freely for any purpose. See [LICENSE](../LICENSE).

### Commercial (extended modules) — `packages/pro/`

66 domain modules and 4 adapters. Copyright "Jeroen Visser".

**Free for**: personal use, educational use, academic research.

**Requires paid license for**: commercial use by companies. Starting at
EUR 2,000.

See [COMMERCIAL-LICENSE.md](../COMMERCIAL-LICENSE.md) for full terms.

## What's commercial

| Category | Count | Examples |
|----------|-------|---------|
| Domain — propagation | 4 | numerical propagation (RK4), adaptive integration (Dormand-Prince), Koopman propagation (DMD), functorial force composition |
| Domain — analysis | 9 | revisit, conjunction, eclipse, sensor, pass analysis, metrics, DOP, thermal, Koopman-spectral conjunction |
| Domain — design | 7 | orbit design, trade studies, multi-objective, optimization, sensitivity, orbit properties, Gramian reconfiguration |
| Domain — environment | 8 | atmosphere, NRLMSISE-00, lifetime, station-keeping, deorbit, radiation, torques, third-body, solar |
| Domain — topology | 6 | ISL, link budget, graph analysis, information theory, spectral topology, Hodge-CUSUM |
| Domain — composition | 10 | mission analysis, conjunction management, communication, coverage optimization, environment, maintenance, economics, operability, cascade, competing risks |
| Domain — math | 4 | linalg, control analysis, statistical analysis, relative motion |
| Domain — research | 4 | decay analysis, temporal correlation, operational prediction, SP3 parser |
| Domain — early warning | 4 | orbit determination (EKF), maneuver detection, hazard reporting, Kessler heatmap |
| Domain — fidelity | 10 | time systems, precession/nutation, earth orientation, planetary ephemeris, gravity field, relativistic forces, tidal forces, albedo/SRP |
| Domain — maneuvers | 1 | Hohmann, bi-elliptic, plane change, phasing |
| Adapters | 4 | czml_exporter, czml_visualization, cesium_viewer, viewer_server |

## Identifying license type

Check the copyright line at the top of any file:

```python
# MIT:
# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.

# Commercial:
# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
```

## What you get with the commercial modules

The 66 commercial modules extend the MIT core into a broader analysis
toolkit. A few things worth knowing:

**Analytical and numerical in one place.** The MIT core gives you Keplerian
and J2 secular propagation. The commercial modules add RK4 numerical
integration with pluggable forces — drag, SRP, third-body, J2/J3,
relativistic, tidal, albedo. You can switch between fast analytical
estimates and higher-fidelity numerical runs without changing your workflow.

**Things compose.** Conjunction screening flows into collision probability,
which flows into avoidance maneuver planning. Coverage analysis combines
with eclipse prediction, link budgets, and lifetime estimates into a
single mission assessment. These compositions encode domain knowledge that
would take time to build from scratch.

**Pure Python, inspectable.** No C extensions, no compiled binaries, no
platform-specific builds. Every computation — the RK4 integrator, the
Jacobi eigensolver, the NRLMSISE-00 atmosphere model — is plain Python
you can step through in your debugger.

**What it is not.** This library is not certified for operational flight
decisions, regulatory compliance determination, or safety-of-flight
assessment. It provides engineering analysis tools for research, education,
and design exploration. Operational use requires independent validation
against authoritative sources.

## Contact

For commercial licensing: see COMMERCIAL-LICENSE.md for contact details.
