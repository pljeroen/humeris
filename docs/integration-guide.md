# Integration Guide

## CelesTrak

### Fetching live data

The `CelesTrakAdapter` fetches OMM (Orbit Mean-elements Message) records from
CelesTrak's GP API and converts them to `Satellite` domain objects via SGP4
propagation.

```python
from humeris.adapters.celestrak import CelesTrakAdapter

celestrak = CelesTrakAdapter()

# By group name
gps = celestrak.fetch_satellites(group="GPS-OPS")
starlink = celestrak.fetch_satellites(group="STARLINK")

# By satellite name
iss = celestrak.fetch_satellites(name="ISS (ZARYA)")

# By NORAD catalog number
sat = celestrak.fetch_satellites(catnr=25544)
```

**Requires**: `pip install ".[live]"` (installs `sgp4>=2.22`).

### Available groups

`STATIONS`, `GPS-OPS`, `STARLINK`, `ONEWEB`, `ACTIVE`, `WEATHER`,
`GALILEO`, `BEIDOU`, `IRIDIUM-NEXT`, `PLANET`, `SPIRE`, `GEO`,
`INTELSAT`, `SES`, `TELESAT`, `AMATEUR`, `SCIENCE`, `NOAA`, `GOES`

### Concurrent mode

For large groups (Starlink = 6000+ sats), SGP4 propagation is the bottleneck.
`ConcurrentCelesTrakAdapter` parallelizes propagation across threads:

```python
from humeris.adapters.concurrent_celestrak import ConcurrentCelesTrakAdapter

concurrent = ConcurrentCelesTrakAdapter(max_workers=16)
starlink = concurrent.fetch_satellites(group="STARLINK")
```

### Custom epoch

Pass an epoch to propagate all satellites to a common reference time:

```python
from datetime import datetime, timezone

epoch = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
sats = celestrak.fetch_satellites(group="GPS-OPS", epoch=epoch)
```

### Reproducibility and determinism

**TLE epoch preservation**: Each `Satellite` object stores the TLE epoch
in its `epoch` field. Without an `epoch` override, SGP4 propagates to the
TLE's own epoch — the state vector represents the satellite at the moment
the TLE was generated. With `epoch=<datetime>`, all satellites are
propagated to that common time.

**Determinism**: For a given set of OMM/TLE records, results are fully
deterministic. SGP4 is a deterministic propagator with no random
components. The only source of variation is the TLE data itself, which
CelesTrak updates at most every 2 hours.

**Network failures**: The adapter raises `ConnectionError` with a
descriptive message on HTTP errors or connection timeouts. The default
timeout is 30 seconds (configurable via `CelesTrakAdapter(timeout=60)`).
If CelesTrak returns `"No GP data found"`, an empty list is returned
(not an error). Individual satellite SGP4 failures are logged as warnings
and skipped — the rest of the group is still returned.

**Reproducing results**: To reproduce a run exactly, save the raw OMM
JSON from CelesTrak and replay it through `SGP4Adapter.omm_to_satellite()`
directly. The library does not cache CelesTrak responses, so re-fetching
may yield different TLEs if the upstream data has been updated.

```python
import json
from humeris.adapters.celestrak import CelesTrakAdapter, SGP4Adapter

# Save raw OMM data for reproducibility
celestrak = CelesTrakAdapter()
records = celestrak.fetch_group("GPS-OPS")
with open("gps_omm_snapshot.json", "w") as f:
    json.dump(records, f)

# Later: replay from saved snapshot (no network needed)
sgp4 = SGP4Adapter()
with open("gps_omm_snapshot.json") as f:
    saved_records = json.load(f)
satellites = [sgp4.omm_to_satellite(r) for r in saved_records]
```

## CesiumJS

### Static HTML viewer

Generate a self-contained HTML file with embedded CesiumJS and CZML data:

```python
from humeris.adapters.cesium_viewer import write_cesium_html
from humeris.adapters.czml_exporter import constellation_packets

packets = constellation_packets(states, epoch, duration, step)
write_cesium_html(packets, "viewer.html", title="My Constellation")
```

The HTML file includes CesiumJS loaded from CDN. Open directly in a browser.

### Multiple layers

```python
from humeris.adapters.czml_visualization import (
    eclipse_snapshot_packets,
    ground_station_packets,
)

walker_pkts = snapshot_packets(states, epoch, name="Walker")
eclipse_pkts = eclipse_snapshot_packets(states, epoch, name="Eclipse")

write_cesium_html(
    walker_pkts,
    "multi_layer.html",
    additional_layers=[eclipse_pkts],
)
```

### Cesium Ion token

For terrain and imagery, provide a Cesium Ion access token:

```python
from humeris.adapters.cesium_viewer import generate_interactive_html

html = generate_interactive_html(
    title="Viewer",
    cesium_token="your-cesium-ion-token",
    port=8765,
)
```

Or when creating the server:

```python
server = create_viewer_server(mgr, port=8765, cesium_token="your-token")
```

## Custom data sources

### Implementing OrbitalDataSource

To connect a custom orbital data provider, implement the `OrbitalDataSource`
port:

```python
from humeris.ports.orbital_data import OrbitalDataSource
from humeris.domain.constellation import Satellite

class MyDataSource(OrbitalDataSource):
    def fetch_group(self, group_name):
        # Return list of OMM dicts from your source
        return [{"OBJECT_NAME": "SAT-1", "EPOCH": "...", ...}]

    def fetch_by_name(self, name):
        return [...]

    def fetch_by_catnr(self, catalog_number):
        return [...]

    def fetch_satellites(self, group=None, name=None, catnr=None):
        # Convert OMM records to Satellite domain objects
        from humeris.domain.omm import parse_omm_record
        records = self.fetch_group(group) if group else []
        return [parse_omm_record(r) for r in records]
```

### Implementing SatelliteExporter

To add a custom export format:

```python
from humeris.ports.export import SatelliteExporter

class MyExporter(SatelliteExporter):
    def export(self, satellites, path, epoch=None):
        # Convert satellites to your format, write to path
        count = 0
        with open(path, 'w') as f:
            for sat in satellites:
                f.write(format_satellite(sat))
                count += 1
        return count
```

### Direct domain usage

Use domain modules directly without adapters:

```python
from humeris.domain.propagation import (
    OrbitalState, derive_orbital_state, propagate_to,
)
from humeris.domain.coordinate_frames import (
    gmst_rad, eci_to_ecef, ecef_to_geodetic,
)

# Derive state from a Satellite object
state = derive_orbital_state(satellite, epoch, include_j2=True)

# Propagate to a future time
pos_eci, vel_eci = propagate_to(state, target_time)

# Convert to geodetic
gmst = gmst_rad(target_time)
pos_ecef, _ = eci_to_ecef(pos_eci, vel_eci, gmst)
lat, lon, alt = ecef_to_geodetic(pos_ecef)
```

### SP3 precise ephemeris

Parse IGS SP3 files for validation against precise orbits:

```python
from humeris.domain.sp3_parser import parse_sp3

with open("igs_final.sp3") as f:
    ephemeris = parse_sp3(f.read())

for point in ephemeris.points[:5]:
    print(f"{point.sat_id}: {point.x_km:.3f} {point.y_km:.3f} {point.z_km:.3f}")
```

## Viewer server integration

### Embedding in your application

```python
from humeris.adapters.viewer_server import (
    LayerManager, create_viewer_server,
)

# Create manager with your epoch
mgr = LayerManager(epoch=your_epoch)

# Add layers programmatically
mgr.add_layer(name="My Data", category="Constellation",
              layer_type="walker", states=your_states, params={})

# Create and run server
server = create_viewer_server(mgr, port=8765)
server.serve_forever()  # Blocks. Use threading for non-blocking.
```

### Non-blocking server

```python
import threading

server = create_viewer_server(mgr, port=8765)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()

# Server runs in background, your code continues
# ...

server.shutdown()  # Clean stop
```

### Adding layers at runtime

The `LayerManager` is shared state — add/remove layers while the server runs:

```python
# Add a new analysis layer
new_id = mgr.add_layer(
    name="Analysis:ISL", category="Analysis",
    layer_type="isl", states=constellation_states, params={},
)

# Toggle visibility
mgr.update_layer(new_id, visible=False)

# Remove
mgr.remove_layer(new_id)
```

Changes are reflected immediately when the browser fetches updated CZML.
