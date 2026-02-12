# Interactive Viewer Server

## Overview

The viewer server provides a browser-based 3D visualization of satellite
constellations using CesiumJS. It runs a local HTTP server that serves a
Cesium globe and generates CZML data on demand for 13 analysis layer types.

## Launching

### Server mode (interactive)

```bash
# Default port 8765
python view_constellation.py --serve

# Custom port
python view_constellation.py --serve --port 9000
```

Pre-loads three Walker shells (500/450/400 km, 1584 sats each) and live
ISS data from CelesTrak. Opens `http://localhost:<port>` in your browser.

### Static mode (self-contained HTML)

```bash
# Generate constellation_viewer.html
python view_constellation.py

# Generate and open in browser
python view_constellation.py --open
```

Produces a ~10 MB self-contained HTML file with baked-in CZML. No server
required — open the file directly in any browser.

## Pre-loaded constellations

Server mode starts with these Walker shells:

| Shell | Altitude | Inclination | Planes x Sats | Phase Factor | RAAN Offset |
|-------|----------|-------------|---------------|--------------|-------------|
| Walker-500 | 500 km | 30° | 22 x 72 | 17 | 0.0° |
| Walker-450 | 450 km | 30° | 22 x 72 | 17 | 5.45° |
| Walker-400 | 400 km | 30° | 22 x 72 | 17 | 10.9° |

Plus live ISS data from CelesTrak (animated track).

## Analysis layer types

The viewer dispatches 13 analysis types via `_generate_czml()`. Each uses
sensible defaults that can be overridden via the `params` dict in the API.

### Core layers

| Type | What it shows | Default behaviour |
|------|---------------|-------------------|
| `walker` / `celestrak` | Satellite constellation orbits | Animated if ≤100 sats, snapshot if >100 |
| `eclipse` | Satellites colored by shadow state | Snapshot: green/orange/red points. Animated: interval-based color |
| `coverage` | Ground heatmap of visible satellite count | Snapshot or animated. 10° grid, 10° min elevation |
| `ground_track` | Sub-satellite polyline trace | First satellite, 2h duration, 60s step |
| `ground_station` | Station marker + visibility circle + access windows | 10° min elevation, 6-sat subset for access |

### Topology layers

| Type | What it shows | Default behaviour |
|------|---------------|-------------------|
| `sensor` | Ground-level FOV ellipses following sub-satellite point | 30° circular half-angle |
| `isl` | Satellite points + ISL polylines colored by SNR | Ka-band, 5000 km range, 100-sat cap |
| `network_eclipse` | ISL links colored by endpoint eclipse state | Ka-band, 5000 km range, 100-sat cap |
| `coverage_connectivity` | Ground rectangles colored by coverage × Fiedler value | Ka-band, 10° grid, 100-sat cap |

### Advanced layers

| Type | What it shows | Default behaviour |
|------|---------------|-------------------|
| `fragility` | Satellites colored by spectral fragility index | Ka-band, 1 orbital period control horizon, 100-sat cap |
| `hazard` | Satellites fade green→red over projected orbital lifetime | Cd=2.2, A=0.01 m², m=4 kg. Duration = ½ lifetime capped 1 yr |
| `precession` | J2 RAAN drift over extended timeline | 7-day duration, 15-min step, 24-sat subset |
| `conjunction` | Two-satellite close approach replay with proximity line | states[0] vs states[n/2], ±30 min, 10s step |

## Default configurations

### RF link (Ka-band)

Used by `isl`, `fragility`, `network_eclipse`, and `coverage_connectivity`:

```python
_DEFAULT_LINK_CONFIG = LinkConfig(
    frequency_hz=26e9,          # 26 GHz Ka-band
    transmit_power_w=10.0,      # 10 W
    tx_antenna_gain_dbi=35.0,   # 35 dBi
    rx_antenna_gain_dbi=35.0,   # 35 dBi
    system_noise_temp_k=500.0,  # 500 K
    bandwidth_hz=100e6,         # 100 MHz
    additional_losses_db=2.0,   # 2 dB
    required_snr_db=10.0,       # 10 dB
)
```

### Sensor FOV

Used by `sensor`:

```python
_DEFAULT_SENSOR = SensorConfig(
    sensor_type=SensorType.CIRCULAR,
    half_angle_deg=30.0,
)
```

### Drag model

Used by `hazard`:

```python
_DEFAULT_DRAG = DragConfig(
    cd=2.2,          # Drag coefficient
    area_m2=0.01,    # Cross-sectional area (m²)
    mass_kg=4.0,     # Satellite mass (kg)
)
```

### Performance caps

O(n²) analyses cap satellite count to prevent lockup:

| Constant | Value | Used by |
|----------|-------|---------|
| `_MAX_TOPOLOGY_SATS` | 100 | isl, fragility, network_eclipse, coverage_connectivity |
| `_MAX_PRECESSION_SATS` | 24 | precession |
| `_SNAPSHOT_THRESHOLD` | 100 | auto mode selection (animated vs snapshot) |

## Overriding defaults via params

Pass custom configurations through the `params` dict when adding layers
via the API:

```json
{
    "type": "isl",
    "source_layer": "layer-1",
    "params": {
        "max_range_km": 8000.0
    }
}
```

Internal params (prefixed with `_`) can override default configs:

| Param key | Type | Applies to |
|-----------|------|------------|
| `_link_config` | `LinkConfig` | isl, fragility, network_eclipse, coverage_connectivity |
| `_sensor` | `SensorConfig` | sensor |
| `_drag_config` | `DragConfig` | hazard |
| `_station` | `GroundStation` | ground_station |
| `max_range_km` | `float` | isl, network_eclipse |
| `control_duration_s` | `float` | fragility |
| `lat_step_deg` | `float` | coverage |
| `lon_step_deg` | `float` | coverage |
| `min_elevation_deg` | `float` | coverage |
| `duration` | `timedelta` | all (default 2h) |
| `step` | `timedelta` | all (default 60s) |

## Programmatic usage

```python
from datetime import datetime, timezone
from constellation_generator import ShellConfig, generate_walker_shell, derive_orbital_state
from constellation_generator.adapters.viewer_server import LayerManager, create_viewer_server

epoch = datetime.now(tz=timezone.utc)

# Create layer manager
mgr = LayerManager(epoch=epoch)

# Add a constellation
shell = ShellConfig(altitude_km=550, inclination_deg=53,
                    num_planes=6, sats_per_plane=10,
                    phase_factor=1, raan_offset_deg=0, shell_name="Demo")
sats = generate_walker_shell(shell)
states = [derive_orbital_state(s, epoch, include_j2=True) for s in sats]

layer_id = mgr.add_layer(
    name="Constellation:Demo", category="Constellation",
    layer_type="walker", states=states, params={},
)

# Add analysis layer referencing the constellation's states
eclipse_id = mgr.add_layer(
    name="Analysis:Eclipse", category="Analysis",
    layer_type="eclipse", states=states, params={},
)

# Add a ground station
gs_id = mgr.add_ground_station(
    name="Delft", lat_deg=52.0, lon_deg=4.4, source_states=states[:6],
)

# Start server
server = create_viewer_server(mgr, port=8765)
server.serve_forever()
```
