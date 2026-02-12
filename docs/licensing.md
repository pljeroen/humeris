# Licensing

## Dual-license model

This project uses a dual-license model: MIT open core with commercial
extended modules.

### MIT (core)

The foundation is MIT-licensed. Copyright "Jeroen Visser".

Covers 10 domain modules, 5 adapters, 3 ports, and their tests:

**Domain**: `orbital_mechanics`, `constellation`, `coordinate_frames`,
`propagation`, `coverage`, `access_windows`, `ground_track`, `observation`,
`omm`, `serialization`

**Adapters**: `celestrak`, `concurrent_celestrak`, `csv_exporter`,
`geojson_exporter`, JSON I/O

**Ports**: `SimulationReader`, `SimulationWriter`, `OrbitalDataSource`,
`SatelliteExporter`

Use freely for any purpose. See [LICENSE](../LICENSE).

### Commercial (extended modules)

49 domain modules and 4 adapters. Copyright "Jeroen Michaël Visser".

**Free for**: personal use, educational use, academic research.

**Requires paid license for**: commercial use by companies. Starting at
EUR 2,000.

See [LICENSE-COMMERCIAL.md](../LICENSE-COMMERCIAL.md) for full terms.

## Boundary rule

The licensing boundary is defined by what's on the git remote:

- **On the remote** = the license stated in the file header applies
- **Not on the remote** = commercial

In practice: MIT core modules carry MIT headers, commercial modules carry
commercial headers. The file header is the source of truth.

## What's commercial

| Category | Count | Examples |
|----------|-------|---------|
| Domain — propagation | 1 | numerical propagation (RK4 + force models) |
| Domain — analysis | 7 | revisit, conjunction, eclipse, sensor, pass analysis, metrics, DOP |
| Domain — design | 6 | orbit design, trade studies, multi-objective, optimization, sensitivity |
| Domain — environment | 7 | atmosphere, lifetime, station-keeping, deorbit, radiation, torques, third-body |
| Domain — topology | 5 | ISL, link budget, graph analysis, information theory, spectral topology |
| Domain — composition | 9 | mission analysis, conjunction management, communication, coverage optimization, environment, maintenance, economics, operability, cascade |
| Domain — math | 4 | linalg, control analysis, statistical analysis, relative motion |
| Domain — research | 4 | decay analysis, temporal correlation, operational prediction, SP3 parser |
| Domain — other | 6 | solar, maneuvers, constellation operability, design optimization, multi-objective design, mission economics |
| Adapters | 4 | czml_exporter, czml_visualization, cesium_viewer, viewer_server |

## Identifying license type

Check the copyright line at the top of any file:

```python
# MIT:
# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.

# Commercial:
# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
```

## Contact

For commercial licensing: see LICENSE-COMMERCIAL.md for contact details.
