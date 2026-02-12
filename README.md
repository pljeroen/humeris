# Constellation Generator

Generate Walker constellation satellite shells and fetch live orbital data for orbit simulation tools.

## Install

```bash
# Core (synthetic constellations only, no external deps)
pip install .

# With live CelesTrak support
pip install ".[live]"

# Development
pip install ".[dev]"
```

## Usage

### Synthetic constellations (Walker shells)

```bash
# Default: 3 Walker shells + SSO band → 7200 satellites
constellation-generator -i simulation_old.json -o simulation.json

# Custom base ID
constellation-generator -i sim_old.json -o sim.json --base-id 200
```

### Live data from CelesTrak

```bash
# Real GPS constellation (32 satellites)
constellation-generator -i sim.json -o out.json --live-group GPS-OPS

# All Starlink satellites (~6000+) with concurrent SGP4 propagation
constellation-generator -i sim.json -o out.json --live-group STARLINK --concurrent

# Search by name
constellation-generator -i sim.json -o out.json --live-name "ISS (ZARYA)"

# By NORAD catalog number
constellation-generator -i sim.json -o out.json --live-catnr 25544
```

#### Available CelesTrak groups

`STATIONS`, `GPS-OPS`, `STARLINK`, `ONEWEB`, `ACTIVE`, `WEATHER`,
`GALILEO`, `BEIDOU`, `IRIDIUM-NEXT`, `PLANET`, `SPIRE`, `GEO`,
`INTELSAT`, `SES`, `TELESAT`, `AMATEUR`, `SCIENCE`, `NOAA`, `GOES`

### Export formats

Export satellite positions as geodetic coordinates (lat/lon/alt) alongside
the simulation JSON:

```bash
# CSV export
constellation-generator -i sim.json -o out.json --export-csv satellites.csv

# GeoJSON export
constellation-generator -i sim.json -o out.json --export-geojson satellites.geojson

# Both at once
constellation-generator -i sim.json -o out.json --export-csv sats.csv --export-geojson sats.geojson
```

CSV columns: `name`, `lat_deg`, `lon_deg`, `alt_km`, `epoch`, `plane_index`,
`sat_index`, `raan_deg`, `true_anomaly_deg`.

GeoJSON produces a FeatureCollection with Point geometries. Coordinates
follow the GeoJSON spec: `[longitude, latitude, altitude_km]`.

### Programmatic

#### Walker shell generation

```python
from constellation_generator import ShellConfig, generate_walker_shell

shell = ShellConfig(
    altitude_km=550, inclination_deg=53,
    num_planes=10, sats_per_plane=20,
    phase_factor=1, raan_offset_deg=0,
    shell_name="Custom-Shell",
)
satellites = generate_walker_shell(shell)
```

#### Live data from CelesTrak

```python
from constellation_generator.adapters.celestrak import CelesTrakAdapter

celestrak = CelesTrakAdapter()
gps_sats = celestrak.fetch_satellites(group="GPS-OPS")
iss = celestrak.fetch_satellites(name="ISS (ZARYA)")
```

#### Concurrent mode

SGP4 propagation is the bottleneck when fetching large groups (not HTTP).
`ConcurrentCelesTrakAdapter` parallelizes propagation across threads
using `ThreadPoolExecutor`:

```python
from constellation_generator.adapters.concurrent_celestrak import ConcurrentCelesTrakAdapter

concurrent = ConcurrentCelesTrakAdapter(max_workers=16)
starlink = concurrent.fetch_satellites(group="STARLINK")
```

#### Coordinate frames

ECI to ECEF conversion applies a Z-axis rotation by the Greenwich Mean
Sidereal Time (GMST) angle. ECEF to Geodetic uses the iterative Bowring
method on the WGS84 ellipsoid:

```python
from datetime import datetime, timezone
from constellation_generator import gmst_rad, eci_to_ecef, ecef_to_geodetic

sat = gps_sats[0]
gmst = gmst_rad(sat.epoch)
pos_ecef, vel_ecef = eci_to_ecef(sat.position_eci, sat.velocity_eci, gmst)
lat, lon, alt = ecef_to_geodetic(pos_ecef)
print(f"{sat.name}: {lat:.4f}°N, {lon:.4f}°E, {alt/1000:.1f} km")
```

#### Ground track

Compute the sub-satellite ground track over time using Keplerian two-body
propagation, optionally with J2 secular perturbations. Appropriate for
synthetic Walker shell satellites. For TLE data, SGP4 propagation via
the adapter layer gives more accurate results.

```python
from datetime import datetime, timedelta, timezone
from constellation_generator import compute_ground_track, generate_walker_shell, ShellConfig

shell = ShellConfig(
    altitude_km=500, inclination_deg=53,
    num_planes=2, sats_per_plane=3,
    phase_factor=1, raan_offset_deg=0,
    shell_name="Demo",
)
sats = generate_walker_shell(shell)

start = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
track = compute_ground_track(sats[0], start, timedelta(minutes=90), timedelta(minutes=1))
track_j2 = compute_ground_track(sats[0], start, timedelta(hours=6), timedelta(minutes=1), include_j2=True)
print(f"Ground track: {len(track)} points")
print(f"Ground track (J2): {len(track_j2)} points")
```

#### Topocentric observation

Compute azimuth, elevation, and slant range from a ground station to a
satellite:

```python
from constellation_generator import (
    GroundStation, derive_orbital_state, propagate_ecef_to, compute_observation,
    generate_walker_shell, ShellConfig,
)
from datetime import datetime, timezone

shell = ShellConfig(altitude_km=500, inclination_deg=53, num_planes=1,
                    sats_per_plane=1, phase_factor=0, raan_offset_deg=0, shell_name="Demo")
sat = generate_walker_shell(shell)[0]
epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

station = GroundStation(name="Delft", lat_deg=52.0, lon_deg=4.4, alt_m=0.0)
state = derive_orbital_state(sat, epoch)
sat_ecef = propagate_ecef_to(state, epoch)
obs = compute_observation(station, sat_ecef)
print(f"Az={obs.azimuth_deg:.1f}°, El={obs.elevation_deg:.1f}°, Range={obs.slant_range_m/1000:.0f} km")
```

#### Access windows

Predict satellite visibility windows (rise/set times) from a ground station:

```python
from constellation_generator import (
    GroundStation, derive_orbital_state, compute_access_windows,
    generate_walker_shell, ShellConfig,
)
from datetime import datetime, timedelta, timezone

shell = ShellConfig(altitude_km=420, inclination_deg=51.6, num_planes=1,
                    sats_per_plane=1, phase_factor=0, raan_offset_deg=0, shell_name="ISS-like")
sat = generate_walker_shell(shell)[0]
epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

station = GroundStation(name="Delft", lat_deg=52.0, lon_deg=4.4)
state = derive_orbital_state(sat, epoch)
windows = compute_access_windows(station, state, epoch, timedelta(hours=24), timedelta(seconds=30))
for w in windows:
    print(f"Rise: {w.rise_time}, Set: {w.set_time}, Max el: {w.max_elevation_deg:.1f}°")
```

#### Coverage analysis

Compute a grid-based coverage snapshot showing how many satellites are
visible from each point:

```python
from constellation_generator import (
    derive_orbital_state, compute_coverage_snapshot,
    generate_walker_shell, ShellConfig,
)
from datetime import datetime, timezone

shell = ShellConfig(altitude_km=500, inclination_deg=53, num_planes=6,
                    sats_per_plane=10, phase_factor=1, raan_offset_deg=0, shell_name="Constellation")
sats = generate_walker_shell(shell)
epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

states = [derive_orbital_state(s, epoch) for s in sats]
grid = compute_coverage_snapshot(states, epoch, lat_step_deg=10, lon_step_deg=10)
max_vis = max(p.visible_count for p in grid)
print(f"Grid: {len(grid)} points, max visible: {max_vis}")
```

#### Revisit time analysis

Compute time-domain coverage FoMs (mean/max revisit, coverage fraction,
mean response time) over an analysis window:

```python
from constellation_generator import (
    derive_orbital_state, compute_revisit,
    generate_walker_shell, ShellConfig,
)
from datetime import datetime, timedelta, timezone

shell = ShellConfig(altitude_km=550, inclination_deg=53, num_planes=6,
                    sats_per_plane=10, phase_factor=1, raan_offset_deg=0, shell_name="Test")
sats = generate_walker_shell(shell)
epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

states = [derive_orbital_state(s, epoch) for s in sats]
result = compute_revisit(
    states, epoch, timedelta(hours=24), timedelta(seconds=60),
    min_elevation_deg=10, lat_step_deg=10, lon_step_deg=10,
)
print(f"Mean coverage: {result.mean_coverage_fraction:.1%}")
print(f"Max revisit: {result.max_revisit_s/60:.0f} min")
print(f"Mean response time: {result.mean_response_time_s/60:.0f} min")
```

#### Parametric trade studies

Sweep Walker constellation parameters and compare coverage metrics:

```python
from constellation_generator import (
    generate_walker_configs, run_walker_trade_study, pareto_front_indices,
)
from datetime import datetime, timedelta, timezone

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
configs = generate_walker_configs(
    altitude_range=(500.0, 600.0),
    inclination_range=(53.0,),
    planes_range=(4, 6),
    sats_per_plane_range=(8,),
)
result = run_walker_trade_study(
    configs, epoch, timedelta(hours=6), timedelta(seconds=120),
    min_elevation_deg=10, lat_step_deg=20, lon_step_deg=20,
)
for pt in result.points:
    print(f"  {pt.config.altitude_km}km, {pt.config.num_planes}P: "
          f"max_revisit={pt.coverage.max_revisit_s/60:.0f}min, "
          f"coverage={pt.coverage.mean_coverage_fraction:.1%}, "
          f"sats={pt.total_satellites}")

costs = [float(pt.total_satellites) for pt in result.points]
metrics = [pt.coverage.max_revisit_s for pt in result.points]
front = pareto_front_indices(costs, metrics)
print(f"Pareto front: {len(front)} points")
```

#### Atmospheric drag and orbit lifetime

Model atmospheric density, compute orbit lifetime under drag decay,
and predict altitude at future times:

```python
from constellation_generator import (
    DragConfig, atmospheric_density, compute_orbit_lifetime,
    compute_altitude_at_time, OrbitalConstants,
)
from datetime import datetime, timedelta, timezone

drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
print(f"B_c = {drag.ballistic_coefficient:.4f} m²/kg")
print(f"Density at 550 km: {atmospheric_density(550.0):.3e} kg/m³")

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
a = OrbitalConstants.R_EARTH + 550_000
result = compute_orbit_lifetime(a, 0.0, drag, epoch)
print(f"Lifetime at 550 km: {result.lifetime_days:.0f} days ({result.lifetime_days/365.25:.1f} yr)")
print(f"Converged: {result.converged}, profile points: {len(result.decay_profile)}")

alt_1yr = compute_altitude_at_time(a, 0.0, drag, epoch, epoch + timedelta(days=365))
print(f"Altitude after 1 year: {alt_1yr:.1f} km")
```

#### Station-keeping delta-V budgets

Compute annual delta-V for drag compensation and plane maintenance,
total propellant budget, and operational lifetime:

```python
from constellation_generator import (
    StationKeepingConfig, compute_station_keeping_budget, DragConfig,
)

drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
config = StationKeepingConfig(
    target_altitude_km=550, inclination_deg=53,
    drag_config=drag, isp_s=300,
    dry_mass_kg=250, propellant_mass_kg=10,
)
budget = compute_station_keeping_budget(config)
print(f"Drag ΔV: {budget.drag_dv_per_year_ms:.2f} m/s/yr")
print(f"Plane ΔV: {budget.plane_dv_per_year_ms:.2f} m/s/yr")
print(f"Total ΔV capacity: {budget.total_dv_capacity_ms:.1f} m/s")
print(f"Operational lifetime: {budget.operational_lifetime_years:.1f} yr")
```

#### Conjunction screening and collision probability

Screen a constellation for close approaches, refine TCA, compute
B-plane geometry and collision probability:

```python
from constellation_generator import (
    screen_conjunctions, assess_conjunction, PositionCovariance,
    generate_walker_shell, ShellConfig, derive_orbital_state,
)
from datetime import datetime, timedelta, timezone

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
shell = ShellConfig(altitude_km=550, inclination_deg=53, num_planes=6,
                    sats_per_plane=10, phase_factor=1, raan_offset_deg=0, shell_name="Test")
sats = generate_walker_shell(shell)
states = [derive_orbital_state(s, reference_epoch=epoch) for s in sats]
names = [s.name for s in sats]

events = screen_conjunctions(states, names, epoch, timedelta(hours=2),
                             timedelta(seconds=30), distance_threshold_m=100_000)
print(f"Conjunction candidates in 2h: {len(events)}")
if events:
    i, j, t, d = events[0]
    event = assess_conjunction(states[i], names[i], states[j], names[j], t)
    print(f"Closest: {event.miss_distance_m:.0f} m between {event.sat1_name} and {event.sat2_name}")
```

#### Solar ephemeris

Compute Sun position in ECI coordinates at any epoch:

```python
from constellation_generator import sun_position_eci, solar_declination_rad
from datetime import datetime, timezone
import math

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
sun = sun_position_eci(epoch)
print(f"Sun RA={math.degrees(sun.right_ascension_rad):.1f}°, Dec={math.degrees(sun.declination_rad):.1f}°")
print(f"Distance: {sun.distance_m/1.496e11:.4f} AU")
```

#### Eclipse prediction

Determine shadow conditions, beta angle, and eclipse windows:

```python
from constellation_generator import (
    eclipse_fraction, compute_beta_angle, compute_eclipse_windows,
    derive_orbital_state, generate_walker_shell, ShellConfig,
)
from datetime import datetime, timedelta, timezone
import math

shell = ShellConfig(altitude_km=500, inclination_deg=53, num_planes=1,
                    sats_per_plane=1, phase_factor=0, raan_offset_deg=0, shell_name="Test")
sat = generate_walker_shell(shell)[0]
epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
state = derive_orbital_state(sat, epoch)

frac = eclipse_fraction(state, epoch)
print(f"Eclipse fraction: {frac:.1%}")

beta = compute_beta_angle(state.raan_rad, state.inclination_rad, epoch)
print(f"Beta angle: {beta:.1f}°")

windows = compute_eclipse_windows(state, epoch, timedelta(hours=3), timedelta(seconds=30))
print(f"Eclipse events in 3h: {len(windows)}")
```

#### Orbit transfer maneuvers

Plan Hohmann, bi-elliptic, plane change, and phasing maneuvers:

```python
from constellation_generator import (
    hohmann_transfer, bielliptic_transfer, plane_change_dv,
    add_propellant_estimate, OrbitalConstants,
)
import math

R_E = OrbitalConstants.R_EARTH
r_leo = R_E + 400_000
r_geo = R_E + 35_786_000

plan = hohmann_transfer(r_leo, r_geo)
print(f"LEO→GEO: {plan.total_delta_v_ms:.0f} m/s, {plan.transfer_time_s/3600:.1f} h")

plan_prop = add_propellant_estimate(plan, isp_s=300, dry_mass_kg=500)
print(f"Propellant: {plan_prop.propellant_mass_kg:.1f} kg")

dv_plane = plane_change_dv(7500, math.radians(28.5))
print(f"28.5° plane change: {dv_plane:.0f} m/s")
```

#### Deorbit compliance

Assess FCC 5-year / ESA 25-year deorbit regulations:

```python
from constellation_generator import (
    DragConfig, assess_deorbit_compliance, DeorbitRegulation,
)
from datetime import datetime, timezone

epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
drag = DragConfig(cd=2.2, area_m2=4.0, mass_kg=400.0)

result = assess_deorbit_compliance(800, drag, epoch, isp_s=300, dry_mass_kg=390)
print(f"Compliant: {result.compliant}, lifetime: {result.natural_lifetime_days:.0f} d")
if result.maneuver_required:
    print(f"Deorbit ΔV: {result.deorbit_delta_v_ms:.1f} m/s")
    print(f"Propellant: {result.propellant_mass_kg:.2f} kg")
```

#### Orbit design

Design sun-synchronous, frozen, and repeat ground track orbits:

```python
from constellation_generator import (
    design_sso_orbit, design_frozen_orbit, design_repeat_ground_track,
)
from datetime import datetime, timezone

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

sso = design_sso_orbit(500, 10.5, epoch)
print(f"SSO 500km LTAN 10:30: inc={sso.inclination_deg:.1f}°, RAAN={sso.raan_deg:.1f}°")

frozen = design_frozen_orbit(800, 98.6)
print(f"Frozen 800km: e={frozen.eccentricity:.6f}, ω={frozen.arg_perigee_deg}°")

rgt = design_repeat_ground_track(97.0, 1, 15)
print(f"Repeat GT 1d/15rev: alt={rgt.altitude_km:.1f} km")
```

#### Numerical propagation (RK4 + pluggable force models)

High-fidelity orbit propagation with composable perturbation forces:

```python
from datetime import datetime, timedelta, timezone
from constellation_generator import (
    ShellConfig, generate_walker_shell, derive_orbital_state,
    TwoBodyGravity, J2Perturbation, J3Perturbation,
    AtmosphericDragForce, SolarRadiationPressureForce,
    DragConfig, propagate_numerical,
)

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
shell = ShellConfig(altitude_km=500, inclination_deg=53, num_planes=1,
                    sats_per_plane=1, phase_factor=0, raan_offset_deg=0, shell_name="Test")
sat = generate_walker_shell(shell)[0]
state = derive_orbital_state(sat, epoch)

# Two-body + J2 + J3
result = propagate_numerical(
    state, timedelta(hours=2), timedelta(seconds=30),
    [TwoBodyGravity(), J2Perturbation(), J3Perturbation()],
)
print(f"Steps: {len(result.steps)}")

# With drag
drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=400.0)
result_drag = propagate_numerical(
    state, timedelta(hours=2), timedelta(seconds=30),
    [TwoBodyGravity(), J2Perturbation(), AtmosphericDragForce(drag)],
)

# With SRP
result_srp = propagate_numerical(
    state, timedelta(hours=2), timedelta(seconds=30),
    [TwoBodyGravity(), SolarRadiationPressureForce(cr=1.5, area_m2=10.0, mass_kg=400.0)],
)
```

#### Configurable atmosphere model

Select between atmosphere density tables:

```python
from constellation_generator import atmospheric_density, AtmosphereModel

rho_high = atmospheric_density(500, AtmosphereModel.HIGH_ACTIVITY)
rho_vallado = atmospheric_density(500, AtmosphereModel.VALLADO_4TH)
print(f"500km density: high={rho_high:.3e}, vallado={rho_vallado:.3e}")
```

#### CZML export (CesiumJS visualization)

Generate CZML for animated 3D visualization in CesiumJS:

```python
from datetime import datetime, timedelta, timezone
from constellation_generator import ShellConfig, generate_walker_shell, derive_orbital_state
from constellation_generator.adapters.czml_exporter import (
    constellation_packets, ground_track_packets, coverage_packets, write_czml,
)

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
shell = ShellConfig(altitude_km=550, inclination_deg=53, num_planes=6,
                    sats_per_plane=10, phase_factor=1, raan_offset_deg=0, shell_name="Test")
sats = generate_walker_shell(shell)
states = [derive_orbital_state(s, epoch) for s in sats]

# Animated satellite orbits
packets = constellation_packets(states, epoch, timedelta(hours=2), timedelta(seconds=60))
write_czml(packets, "constellation.czml")

# Ground track polyline
from constellation_generator import compute_ground_track
track = compute_ground_track(sats[0], epoch, timedelta(minutes=90), timedelta(minutes=1))
gt_packets = ground_track_packets(track)
write_czml(gt_packets, "ground_track.czml")

# Coverage heatmap
from constellation_generator import compute_coverage_snapshot
grid = compute_coverage_snapshot(states, epoch, lat_step_deg=5, lon_step_deg=5)
cov_packets = coverage_packets(grid, lat_step_deg=5, lon_step_deg=5)
write_czml(cov_packets, "coverage.czml")
```

#### Export formats (programmatic)

```python
from constellation_generator.adapters.csv_exporter import CsvSatelliteExporter
from constellation_generator.adapters.geojson_exporter import GeoJsonSatelliteExporter

# CSV
CsvSatelliteExporter().export(sats, "satellites.csv")

# GeoJSON
GeoJsonSatelliteExporter().export(sats, "satellites.geojson")
```

#### Mixing sources

Combine synthetic and live satellites, then serialise for simulation:

```python
from constellation_generator import build_satellite_entity

template = {"Name": "Sat", "Id": 0}
all_sats = satellites + gps_sats
entities = [
    build_satellite_entity(s, template, base_id=100 + i)
    for i, s in enumerate(all_sats)
]
```

## Default Shell Configuration

| Shell | Altitude | Inclination | Planes × Sats | Phase Factor |
|-------|----------|-------------|----------------|--------------|
| LEO-Shell500 | 500 km | 30° | 22 × 72 | 17 |
| LEO-Shell450 | 450 km | 30° | 22 × 72 | 17 |
| LEO-Shell400 | 400 km | 30° | 22 × 72 | 17 |
| SSO Band | 525–2200 km (50 km step) | SSO-computed | 1 × 72 | 0 |

## Architecture

```
src/constellation_generator/
├── domain/                    # Pure logic — only stdlib math/dataclasses/datetime
│   ├── orbital_mechanics.py   # Kepler → Cartesian, SSO inclination, J2 perturbations
│   ├── constellation.py       # Walker shells, SSO bands, ShellConfig, Satellite
│   ├── coordinate_frames.py   # ECI ↔ ECEF ↔ Geodetic (GMST, Bowring, WGS84)
│   ├── propagation.py         # Shared Keplerian + J2 propagation
│   ├── ground_track.py        # Ground track computation (delegates to propagation)
│   ├── observation.py         # Topocentric azimuth/elevation/range
│   ├── access_windows.py      # Satellite rise/set window detection
│   ├── coverage.py            # Grid-based visibility coverage analysis
│   ├── revisit.py             # Time-domain revisit analysis (ECA-optimized)
│   ├── trade_study.py         # Parametric Walker trade studies, Pareto front
│   ├── atmosphere.py          # Exponential density model, drag acceleration
│   ├── lifetime.py            # Orbit lifetime, decay profile (Euler integration)
│   ├── station_keeping.py     # Delta-V budgets, Tsiolkovsky, propellant lifetime
│   ├── conjunction.py         # Screening, TCA, B-plane, collision probability
│   ├── solar.py               # Analytical solar ephemeris (Meeus/Vallado)
│   ├── eclipse.py             # Shadow geometry, beta angle, eclipse windows
│   ├── maneuvers.py           # Hohmann, bi-elliptic, plane change, phasing
│   ├── deorbit.py             # FCC/ESA deorbit compliance assessment
│   ├── orbit_design.py        # SSO/LTAN, frozen orbit, repeat ground track
│   ├── numerical_propagation.py # RK4 integrator + pluggable force models
│   ├── serialization.py       # Simulation format (Y/Z swap, precision)
│   └── omm.py                 # CelesTrak OMM record → OrbitalElements
├── ports/                     # Abstract interfaces (ABC)
│   ├── __init__.py            # SimulationReader, SimulationWriter
│   ├── orbital_data.py        # OrbitalDataSource
│   └── export.py              # SatelliteExporter
├── adapters/                  # Infrastructure (JSON I/O, HTTP, SGP4, export)
│   ├── __init__.py            # JsonSimulationReader/Writer, exporters
│   ├── celestrak.py           # CelesTrakAdapter, SGP4Adapter
│   ├── concurrent_celestrak.py # ConcurrentCelesTrakAdapter (ThreadPoolExecutor)
│   ├── csv_exporter.py        # CsvSatelliteExporter
│   ├── geojson_exporter.py    # GeoJsonSatelliteExporter
│   └── czml_exporter.py       # CZML packets for CesiumJS visualization
└── cli.py                     # CLI entry point (--concurrent, --export-csv, --export-geojson)
```

The domain layer has zero external dependencies. All I/O (file access,
HTTP, SGP4 propagation, export) is confined to the adapter layer behind
port interfaces.

## Tests

```bash
pytest                                    # all 403 tests
pytest tests/test_constellation.py        # 21 synthetic tests (offline)
pytest tests/test_coordinate_frames.py    # 29 coordinate frame tests (offline)
pytest tests/test_j2_perturbations.py     # 12 J2/J3 perturbation tests (offline)
pytest tests/test_propagation.py          # 18 propagation tests (offline)
pytest tests/test_ground_track.py         # 16 ground track tests (offline)
pytest tests/test_observation.py          # 14 observation tests (offline)
pytest tests/test_access_windows.py       # 11 access window tests (offline)
pytest tests/test_coverage.py             # 10 coverage tests (offline)
pytest tests/test_revisit.py             # 20 revisit analysis tests (offline)
pytest tests/test_trade_study.py         # 19 trade study tests (offline)
pytest tests/test_atmosphere.py           # 22 atmospheric drag tests (offline)
pytest tests/test_lifetime.py             # 16 orbit lifetime tests (offline)
pytest tests/test_station_keeping.py      # 17 station-keeping tests (offline)
pytest tests/test_conjunction.py          # 18 conjunction tests (offline)
pytest tests/test_solar.py               # 14 solar ephemeris tests (offline)
pytest tests/test_eclipse.py             # 13 eclipse prediction tests (offline)
pytest tests/test_maneuvers.py           # 21 orbit transfer tests (offline)
pytest tests/test_deorbit.py             # 13 deorbit compliance tests (offline)
pytest tests/test_orbit_design.py        # 17 orbit design tests (offline)
pytest tests/test_export.py               # 18 export tests (offline)
pytest tests/test_czml_exporter.py       # 18 CZML exporter tests (offline)
pytest tests/test_numerical_propagation.py # 22 numerical propagation tests (offline)
pytest tests/test_concurrent_celestrak.py # 12 concurrent adapter tests (offline)
pytest tests/test_live_data.py            # 13 live data tests (network)
```

## Credits

Original code by [Scott Manley](https://www.youtube.com/@scottmanley). Refactored and extended by Jeroen.

## License

This project uses a dual-license model:

**MIT** — the core library (constellation generation, propagation, coordinate
frames, ground track, observation, access windows, coverage, export).
See `LICENSE`.

**Commercial** — the pro modules (atmospheric drag, orbit lifetime,
station-keeping, conjunction assessment). Free for personal, educational,
and academic use. Commercial use requires a one-time EUR 10,000 license.
See `LICENSE-COMMERCIAL.md` or email planet dot jeroen at gmail dot com.
