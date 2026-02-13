# Trade Study and Decision Support

This guide walks through a complete constellation design workflow: sweep
the parameter space, find the Pareto-optimal configurations, export for
downstream tools, and screen for conjunction risk.

## 1. Pareto Optimization

Find the sweet spot between cost (number of satellites) and performance
(max revisit time) by sweeping altitude, plane count, and satellites
per plane.

```python
from datetime import datetime, timedelta, timezone
from constellation_generator import (
    generate_walker_configs, run_walker_trade_study, pareto_front_indices
)

# Define the design space — explicit values for each parameter
configs = generate_walker_configs(
    altitude_range=(500.0, 550.0, 600.0),
    inclination_range=(53.0,),
    planes_range=(4, 6, 8, 10),
    sats_per_plane_range=(10, 15, 20),
)

# Run coverage analysis (revisit metrics for each configuration)
epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
result = run_walker_trade_study(
    configs, epoch, timedelta(hours=12), timedelta(seconds=60),
    min_elevation_deg=10, lat_step_deg=15, lon_step_deg=15
)

# Extract cost vs performance
costs = [pt.total_satellites for pt in result.points]
revisit = [pt.coverage.max_revisit_s / 60 for pt in result.points]  # minutes

# Identify Pareto front (configurations where no other design is
# both cheaper AND has better revisit)
front_idx = pareto_front_indices(costs, revisit)

for i in front_idx:
    pt = result.points[i]
    c = pt.config
    print(f"{c.num_planes}x{c.sats_per_plane} @ {c.altitude_km} km: "
          f"{pt.total_satellites} sats, "
          f"max revisit {pt.coverage.max_revisit_s / 60:.1f} min")
```

The Pareto front tells you: "if you want max revisit below X minutes,
you need at least Y satellites." Every point on the front is optimal —
you can only improve one metric by sacrificing the other.

## 2. Exporting for Downstream Analysis

Once you've identified an optimal shell, export it for GIS tools,
ground station planning, or mission planning software.

### CSV export

```python
from constellation_generator import ShellConfig, generate_walker_shell
from constellation_generator.adapters.csv_exporter import CsvSatelliteExporter

best_shell = ShellConfig(
    altitude_km=550, inclination_deg=53,
    num_planes=8, sats_per_plane=15,
    phase_factor=1, raan_offset_deg=0,
    shell_name="Optimized-LEO",
)
sats = generate_walker_shell(best_shell)

CsvSatelliteExporter().export(sats, "optimized_constellation.csv")
```

The CSV contains state vectors for each satellite:

| Field | Use |
|-------|-----|
| Position (ECI x, y, z) | Orbit determination, ground track prediction |
| Velocity (ECI vx, vy, vz) | Doppler shift estimation, link budget timing |
| RAAN | Plane phasing, collision avoidance within the shell |
| True anomaly | Satellite spacing within each plane |

### GeoJSON export

```python
from constellation_generator.adapters.geojson_exporter import GeoJsonExporter

GeoJsonExporter().export(sats, "optimized_constellation.geojson")
```

Opens directly in QGIS, Mapbox, or any GIS tool for geographic analysis.

### Simulator export

See [Simulator Integrations](simulator-integrations.md) for exporting to
Universe Sandbox, SpaceEngine, KSP, Celestia, Google Earth, Blender, or
Stellarium.

## 3. Conjunction Screening

Screen your constellation against itself (or other objects) for close
approaches. Essential for regulatory filings and operational safety.

```python
from constellation_generator import (
    ShellConfig, generate_walker_shell,
    derive_orbital_state, screen_conjunctions,
)

shell = ShellConfig(
    altitude_km=550, inclination_deg=53,
    num_planes=8, sats_per_plane=15,
    phase_factor=1, raan_offset_deg=0,
    shell_name="Optimized-LEO",
)
sats = generate_walker_shell(shell)
epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

# Derive orbital states for propagation
states = [derive_orbital_state(s, epoch) for s in sats]
names = [s.name for s in sats]

# Screen for close approaches within 2 hours
events = screen_conjunctions(
    states, names, epoch, timedelta(hours=2),
    timedelta(seconds=10), distance_threshold_m=5000
)

print(f"Detected {len(events)} potential close approaches.")
for i, j, t, dist in events[:5]:
    print(f"  {names[i]} - {names[j]}: {dist:.0f} m at {t}")
```

## 4. Putting It All Together

A complete design workflow combines all three steps:

1. **Sweep** the design space with `generate_walker_configs` +
   `run_walker_trade_study`
2. **Select** the optimal configuration from the Pareto front
3. **Export** the selected design to CSV, GeoJSON, or simulator formats
4. **Screen** for conjunction risk
5. **Iterate** — adjust parameters, re-run, compare

This replaces manual spreadsheet-based constellation sizing with a
reproducible, scriptable pipeline. The Pareto front provides
mathematical justification for constellation size decisions —
useful for investor presentations, regulatory filings, and
engineering trade reviews.
