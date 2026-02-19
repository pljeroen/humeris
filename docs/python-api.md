# Python API Examples

Worked examples for the Humeris Python API. For installation and CLI usage,
see [Getting Started](getting-started.md).

## Walker shell generation

```python
from humeris.domain.constellation import ShellConfig, generate_walker_shell

shell = ShellConfig(
    altitude_km=550, inclination_deg=53,
    num_planes=10, sats_per_plane=20,
    phase_factor=1, raan_offset_deg=0,
    shell_name="Custom-Shell",
)
satellites = generate_walker_shell(shell)
```

## Live data from CelesTrak

```python
from humeris.adapters.celestrak import CelesTrakAdapter

celestrak = CelesTrakAdapter()
gps_sats = celestrak.fetch_satellites(group="GPS-OPS")
iss = celestrak.fetch_satellites(name="ISS (ZARYA)")
```

### Concurrent mode

SGP4 propagation is the bottleneck when fetching large groups (not HTTP).
`ConcurrentCelesTrakAdapter` parallelizes propagation across threads
using `ThreadPoolExecutor`:

```python
from humeris.adapters.concurrent_celestrak import ConcurrentCelesTrakAdapter

concurrent = ConcurrentCelesTrakAdapter(max_workers=16)
starlink = concurrent.fetch_satellites(group="STARLINK")
```

## Coordinate frames

ECI to ECEF conversion applies a Z-axis rotation by the Greenwich Mean
Sidereal Time (GMST) angle. ECEF to Geodetic uses the iterative Bowring
method on the WGS84 ellipsoid:

```python
from datetime import datetime, timezone
from humeris.domain.coordinate_frames import gmst_rad, eci_to_ecef, ecef_to_geodetic

sat = gps_sats[0]
gmst = gmst_rad(sat.epoch)
pos_ecef, vel_ecef = eci_to_ecef(sat.position_eci, sat.velocity_eci, gmst)
lat, lon, alt = ecef_to_geodetic(pos_ecef)
print(f"{sat.name}: {lat:.4f}N, {lon:.4f}E, {alt/1000:.1f} km")
```

## Ground track

Compute the sub-satellite ground track over time using Keplerian two-body
propagation, optionally with J2 secular perturbations. For TLE data, SGP4
propagation via the adapter layer provides SGP4-based tracks.

```python
from datetime import datetime, timedelta, timezone
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.ground_track import compute_ground_track

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

## Topocentric observation

Compute azimuth, elevation, and slant range from a ground station to a
satellite:

```python
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state, propagate_ecef_to
from humeris.domain.observation import GroundStation, compute_observation
from datetime import datetime, timezone

shell = ShellConfig(altitude_km=500, inclination_deg=53, num_planes=1,
                    sats_per_plane=1, phase_factor=0, raan_offset_deg=0, shell_name="Demo")
sat = generate_walker_shell(shell)[0]
epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

station = GroundStation(name="Delft", lat_deg=52.0, lon_deg=4.4, alt_m=0.0)
state = derive_orbital_state(sat, epoch)
sat_ecef = propagate_ecef_to(state, epoch)
obs = compute_observation(station, sat_ecef)
print(f"Az={obs.azimuth_deg:.1f}, El={obs.elevation_deg:.1f}, Range={obs.slant_range_m/1000:.0f} km")
```

## Access windows

Predict satellite visibility windows (rise/set times) from a ground station:

```python
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
from humeris.domain.observation import GroundStation
from humeris.domain.access_windows import compute_access_windows
from datetime import datetime, timedelta, timezone

shell = ShellConfig(altitude_km=420, inclination_deg=51.6, num_planes=1,
                    sats_per_plane=1, phase_factor=0, raan_offset_deg=0, shell_name="ISS-like")
sat = generate_walker_shell(shell)[0]
epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

station = GroundStation(name="Delft", lat_deg=52.0, lon_deg=4.4)
state = derive_orbital_state(sat, epoch)
windows = compute_access_windows(station, state, epoch, timedelta(hours=24), timedelta(seconds=30))
for w in windows:
    print(f"Rise: {w.rise_time}, Set: {w.set_time}, Max el: {w.max_elevation_deg:.1f}")
```

## Coverage analysis

Compute a grid-based coverage snapshot showing how many satellites are
visible from each point:

```python
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
from humeris.domain.coverage import compute_coverage_snapshot
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

## Revisit time analysis

Compute time-domain coverage figures of merit (mean/max revisit, coverage
fraction, mean response time) over an analysis window:

```python
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
from humeris.domain.revisit import compute_revisit
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

## Trade studies

Sweep Walker constellation parameters and compare coverage metrics:

```python
from humeris.domain.trade_study import (
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

## Atmospheric drag and orbit lifetime

Model atmospheric density, compute orbit lifetime under drag decay,
and predict altitude at future times:

```python
from humeris.domain.atmosphere import DragConfig, atmospheric_density
from humeris.domain.lifetime import compute_orbit_lifetime, compute_altitude_at_time
from humeris.domain.orbital_mechanics import OrbitalConstants
from datetime import datetime, timedelta, timezone

drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
print(f"B_c = {drag.ballistic_coefficient:.4f} m2/kg")
print(f"Density at 550 km: {atmospheric_density(550.0):.3e} kg/m3")

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
a = OrbitalConstants.R_EARTH + 550_000
result = compute_orbit_lifetime(a, 0.0, drag, epoch)
print(f"Lifetime at 550 km: {result.lifetime_days:.0f} days ({result.lifetime_days/365.25:.1f} yr)")
print(f"Converged: {result.converged}, profile points: {len(result.decay_profile)}")

alt_1yr = compute_altitude_at_time(a, 0.0, drag, epoch, epoch + timedelta(days=365))
print(f"Altitude after 1 year: {alt_1yr:.1f} km")
```

## Station-keeping delta-V budgets

Compute annual delta-V for drag compensation and plane maintenance,
total propellant budget, and operational lifetime:

```python
from humeris.domain.station_keeping import StationKeepingConfig, compute_station_keeping_budget
from humeris.domain.atmosphere import DragConfig

drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
config = StationKeepingConfig(
    target_altitude_km=550, inclination_deg=53,
    drag_config=drag, isp_s=300,
    dry_mass_kg=250, propellant_mass_kg=10,
)
budget = compute_station_keeping_budget(config)
print(f"Drag dV: {budget.drag_dv_per_year_ms:.2f} m/s/yr")
print(f"Plane dV: {budget.plane_dv_per_year_ms:.2f} m/s/yr")
print(f"Total dV capacity: {budget.total_dv_capacity_ms:.1f} m/s")
print(f"Operational lifetime: {budget.operational_lifetime_years:.1f} yr")
```

## Conjunction screening and collision probability

Screen a constellation for close approaches, refine TCA, compute
B-plane geometry and collision probability.

> **Note**: Collision probability estimates use simplified covariance models.
> Operational conjunction assessment requires authoritative ephemeris data
> (e.g., from 18th Space Defense Squadron) and validated covariance realism.

```python
from humeris.domain.conjunction import screen_conjunctions, assess_conjunction, PositionCovariance
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
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

## Solar ephemeris

Compute Sun position in ECI coordinates at any epoch:

```python
from humeris.domain.solar import sun_position_eci, solar_declination_rad
from datetime import datetime, timezone
import math

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
sun = sun_position_eci(epoch)
print(f"Sun RA={math.degrees(sun.right_ascension_rad):.1f}, Dec={math.degrees(sun.declination_rad):.1f}")
print(f"Distance: {sun.distance_m/1.496e11:.4f} AU")
```

## Eclipse prediction

Determine shadow conditions, beta angle, and eclipse windows:

```python
from humeris.domain.eclipse import eclipse_fraction, compute_beta_angle, compute_eclipse_windows
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
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
print(f"Beta angle: {beta:.1f}")

windows = compute_eclipse_windows(state, epoch, timedelta(hours=3), timedelta(seconds=30))
print(f"Eclipse events in 3h: {len(windows)}")
```

## Orbit transfer maneuvers

Plan Hohmann, bi-elliptic, plane change, and phasing maneuvers:

```python
from humeris.domain.maneuvers import (
    hohmann_transfer, bielliptic_transfer, plane_change_dv, add_propellant_estimate,
)
from humeris.domain.orbital_mechanics import OrbitalConstants
import math

R_E = OrbitalConstants.R_EARTH
r_leo = R_E + 400_000
r_geo = R_E + 35_786_000

plan = hohmann_transfer(r_leo, r_geo)
print(f"LEO to GEO: {plan.total_delta_v_ms:.0f} m/s, {plan.transfer_time_s/3600:.1f} h")

plan_prop = add_propellant_estimate(plan, isp_s=300, dry_mass_kg=500)
print(f"Propellant: {plan_prop.propellant_mass_kg:.1f} kg")

dv_plane = plane_change_dv(7500, math.radians(28.5))
print(f"28.5 plane change: {dv_plane:.0f} m/s")
```

## Deorbit compliance estimation

Estimate orbit decay timelines against FCC 5-year / ESA 25-year guidelines:

> **Note**: These are engineering estimates based on simplified atmospheric
> models, not regulatory compliance determinations. Actual compliance
> assessment requires mission-specific analysis with validated tools.

```python
from humeris.domain.atmosphere import DragConfig
from humeris.domain.deorbit import assess_deorbit_compliance, DeorbitRegulation
from datetime import datetime, timezone

epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
drag = DragConfig(cd=2.2, area_m2=4.0, mass_kg=400.0)

result = assess_deorbit_compliance(800, drag, epoch, isp_s=300, dry_mass_kg=390)
print(f"Compliant: {result.compliant}, lifetime: {result.natural_lifetime_days:.0f} d")
if result.maneuver_required:
    print(f"Deorbit dV: {result.deorbit_delta_v_ms:.1f} m/s")
    print(f"Propellant: {result.propellant_mass_kg:.2f} kg")
```

## Orbit design

Design sun-synchronous, frozen, and repeat ground track orbits:

```python
from humeris.domain.orbit_design import (
    design_sso_orbit, design_frozen_orbit, design_repeat_ground_track,
)
from datetime import datetime, timezone

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

sso = design_sso_orbit(500, 10.5, epoch)
print(f"SSO 500km LTAN 10:30: inc={sso.inclination_deg:.1f}, RAAN={sso.raan_deg:.1f}")

frozen = design_frozen_orbit(800, 98.6)
print(f"Frozen 800km: e={frozen.eccentricity:.6f}, w={frozen.arg_perigee_deg}")

rgt = design_repeat_ground_track(97.0, 1, 15)
print(f"Repeat GT 1d/15rev: alt={rgt.altitude_km:.1f} km")
```

## Numerical propagation

RK4 numerical orbit propagation with composable perturbation forces:

```python
from datetime import datetime, timedelta, timezone
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
from humeris.domain.numerical_propagation import (
    TwoBodyGravity, J2Perturbation, J3Perturbation,
    AtmosphericDragForce, SolarRadiationPressureForce, propagate_numerical,
)
from humeris.domain.atmosphere import DragConfig

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

## Configurable atmosphere model

Select between atmosphere density tables:

```python
from humeris.domain.atmosphere import atmospheric_density, AtmosphereModel

rho_high = atmospheric_density(500, AtmosphereModel.HIGH_ACTIVITY)
rho_vallado = atmospheric_density(500, AtmosphereModel.VALLADO_4TH)
print(f"500km density: high={rho_high:.3e}, vallado={rho_vallado:.3e}")
```

## CZML export

Generate CZML for animated 3D visualization in CesiumJS:

```python
from datetime import datetime, timedelta, timezone
from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
from humeris.adapters.czml_exporter import (
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
from humeris.domain.ground_track import compute_ground_track
track = compute_ground_track(sats[0], epoch, timedelta(minutes=90), timedelta(minutes=1))
gt_packets = ground_track_packets(track)
write_czml(gt_packets, "ground_track.czml")

# Coverage heatmap
from humeris.domain.coverage import compute_coverage_snapshot
grid = compute_coverage_snapshot(states, epoch, lat_step_deg=5, lon_step_deg=5)
cov_packets = coverage_packets(grid, lat_step_deg=5, lon_step_deg=5)
write_czml(cov_packets, "coverage.czml")
```

## Export formats

```python
from humeris.adapters.csv_exporter import CsvSatelliteExporter
from humeris.adapters.geojson_exporter import GeoJsonSatelliteExporter

# CSV
CsvSatelliteExporter().export(sats, "satellites.csv")

# GeoJSON
GeoJsonSatelliteExporter().export(sats, "satellites.geojson")
```

Simulator exporters:

```python
from humeris.adapters.celestia_exporter import CelestiaExporter
from humeris.adapters.kml_exporter import KmlExporter
from humeris.adapters.ubox_exporter import UboxExporter

CelestiaExporter().export(sats, "constellation.ssc")
KmlExporter().export(sats, "constellation.kml")
UboxExporter().export(sats, "constellation.ubox")

# KML with visual layer options
KmlExporter(include_orbits=False, include_planes=True).export(sats, "planes.kml")
```

## Mixing sources

Combine synthetic and live satellites, then serialise for simulation:

```python
from humeris.domain.serialization import build_satellite_entity

template = {"Name": "Sat", "Id": 0}
all_sats = satellites + gps_sats
entities = [
    build_satellite_entity(s, template, base_id=100 + i)
    for i, s in enumerate(all_sats)
]
```
