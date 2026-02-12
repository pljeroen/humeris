# Export Formats

## CSV

Geodetic satellite positions as comma-separated values.

### CLI

```bash
constellation-generator -i sim.json -o out.json --export-csv satellites.csv
```

### Programmatic

```python
from constellation_generator.adapters.csv_exporter import CsvSatelliteExporter

exporter = CsvSatelliteExporter()
count = exporter.export(satellites, "satellites.csv")
```

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Satellite identifier |
| `lat_deg` | float | Geodetic latitude (degrees) |
| `lon_deg` | float | Geodetic longitude (degrees) |
| `alt_km` | float | Altitude above WGS84 ellipsoid (km) |
| `epoch` | str | ISO 8601 timestamp |
| `plane_index` | int | Orbital plane index |
| `sat_index` | int | Satellite index within plane |
| `raan_deg` | float | Right ascension of ascending node (degrees) |
| `true_anomaly_deg` | float | True anomaly (degrees) |

---

## GeoJSON

Standard GeoJSON FeatureCollection with Point geometries.

### CLI

```bash
constellation-generator -i sim.json -o out.json --export-geojson satellites.geojson
```

### Programmatic

```python
from constellation_generator.adapters.geojson_exporter import GeoJsonSatelliteExporter

exporter = GeoJsonSatelliteExporter()
count = exporter.export(satellites, "satellites.geojson")
```

### Structure

```json
{
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [4.4, 52.0, 550.0]
            },
            "properties": {
                "name": "LEO-Shell500-P00-S00",
                "plane_index": 0,
                "sat_index": 0,
                "raan_deg": 0.0,
                "true_anomaly_deg": 0.0
            }
        }
    ]
}
```

Coordinates follow the GeoJSON spec: `[longitude, latitude, altitude_km]`.

---

## CZML

[CZML](https://github.com/AnalyticalGraphicsInc/czml-writer/wiki/CZML-Guide)
is the native format for CesiumJS 3D visualization. The library generates
CZML packets for animated orbits, static snapshots, ground tracks, coverage
heatmaps, and advanced analysis visualizations.

### Writing CZML files

```python
from constellation_generator.adapters.czml_exporter import (
    constellation_packets, snapshot_packets, ground_track_packets,
    coverage_packets, write_czml,
)

# Animated constellation (time-varying positions)
packets = constellation_packets(states, epoch, duration, step)
write_czml(packets, "animated.czml")

# Static snapshot (single-epoch points)
packets = snapshot_packets(states, epoch)
write_czml(packets, "snapshot.czml")

# Ground track polyline
from constellation_generator import compute_ground_track
track = compute_ground_track(satellite, epoch, duration, step)
packets = ground_track_packets(track)
write_czml(packets, "ground_track.czml")

# Coverage heatmap
from constellation_generator import compute_coverage_snapshot
grid = compute_coverage_snapshot(states, epoch, lat_step_deg=5, lon_step_deg=5)
packets = coverage_packets(grid, lat_step_deg=5, lon_step_deg=5)
write_czml(packets, "coverage.czml")
```

### Advanced CZML visualization

```python
from constellation_generator.adapters.czml_visualization import (
    eclipse_constellation_packets,
    eclipse_snapshot_packets,
    sensor_footprint_packets,
    ground_station_packets,
    conjunction_replay_packets,
    isl_topology_packets,
    coverage_evolution_packets,
    fragility_constellation_packets,
    hazard_evolution_packets,
    network_eclipse_packets,
    coverage_connectivity_packets,
)
```

Each function returns a list of CZML packets (document + entity packets).
See [Viewer Server](viewer-server.md) for parameter details per type.

### CZML packet structure

Every CZML document starts with a document packet:

```json
[
    {
        "id": "document",
        "name": "Constellation",
        "version": "1.0",
        "clock": {
            "interval": "2026-03-20T12:00:00Z/2026-03-20T14:00:00Z",
            "currentTime": "2026-03-20T12:00:00Z",
            "multiplier": 60,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER"
        }
    },
    {
        "id": "satellite-0",
        "name": "Sat-0",
        "position": {
            "epoch": "2026-03-20T12:00:00Z",
            "cartographicDegrees": [0, 4.4, 52.0, 550000, 60, 4.5, 52.1, 550000, ...],
            "interpolationAlgorithm": "LAGRANGE",
            "interpolationDegree": 5
        },
        "point": {
            "pixelSize": 5,
            "color": {"rgba": [0, 255, 128, 255]}
        }
    }
]
```

### Plane coloring

Satellites are automatically colored by orbital plane index using a
10-color palette. The plane assignment algorithm groups satellites by
RAAN proximity.

---

## Simulation JSON

The CLI reads/writes a simulation JSON format for integration with external
tools. Satellite entities include ECI position/velocity with Y/Z axis
swap and specified precision.

```bash
constellation-generator -i template.json -o output.json
```

The format preserves all template entities and adds generated satellites
with sequential IDs from `--base-id` (default 0).
