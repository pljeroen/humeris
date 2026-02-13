# API Reference

The viewer server exposes a REST API on `http://localhost:<port>`.
All data endpoints return JSON. CORS is enabled (`Access-Control-Allow-Origin: *`).

## Endpoints

### GET /

Returns the interactive Cesium HTML viewer page.

**Response**: `text/html`, 200

---

### GET /api/state

Returns metadata for all layers (no CZML data).

**Response**: 200
```json
{
    "epoch": "2026-03-20T12:00:00+00:00",
    "layers": [
        {
            "layer_id": "layer-1",
            "name": "Constellation:Walker-500",
            "category": "Constellation",
            "layer_type": "walker",
            "mode": "snapshot",
            "visible": true,
            "num_entities": 1584,
            "params": {
                "altitude_km": 500,
                "inclination_deg": 30
            }
        }
    ]
}
```

Note: params starting with `_` (internal objects) are excluded from the response.

---

### GET /api/czml/{layer_id}

Returns CZML packets for a specific layer.

**Response**: 200
```json
[
    {"id": "document", "name": "Constellation:Walker", "version": "1.0"},
    {"id": "satellite-0", "name": "Sat-0", "position": {...}, "point": {...}},
    ...
]
```

**Errors**: 404 if layer not found.

---

### POST /api/constellation

Add a new constellation layer.

**Request body**:
```json
{
    "type": "walker",
    "params": {
        "altitude_km": 550,
        "inclination_deg": 53,
        "num_planes": 10,
        "sats_per_plane": 20,
        "phase_factor": 1,
        "raan_offset_deg": 0.0,
        "shell_name": "Custom"
    }
}
```

**Constellation types**:

| Type | Required params | Description |
|------|----------------|-------------|
| `walker` | `altitude_km`, `inclination_deg`, `num_planes`, `sats_per_plane` | Synthetic Walker shell |
| `celestrak` | `group` or `name` | Live data from CelesTrak |

Optional walker params: `phase_factor` (default 1), `raan_offset_deg` (default 0), `shell_name`.

CelesTrak params: `group` (e.g. `"GPS-OPS"`, `"STARLINK"`), `name` (e.g. `"ISS (ZARYA)"`).

**Response**: 201
```json
{"layer_id": "layer-1"}
```

**Errors**: 400 for invalid params or unknown type.

---

### POST /api/analysis

Add an analysis layer derived from an existing constellation.

**Request body**:
```json
{
    "type": "eclipse",
    "source_layer": "layer-1",
    "params": {}
}
```

**Analysis types**: `eclipse`, `coverage`, `sensor`, `isl`, `fragility`,
`hazard`, `network_eclipse`, `coverage_connectivity`, `ground_track`,
`conjunction`, `precession`, `kessler_heatmap`, `conjunction_hazard`,
`ground_station`.

See [Viewer Server](viewer-server.md) for default parameters per type.

**Response**: 201
```json
{"layer_id": "layer-2"}
```

**Errors**: 404 if source layer not found, 400 if unknown type, 500 on generation failure.

---

### POST /api/ground-station

Add a ground station layer.

**Request body**:
```json
{
    "name": "Svalbard",
    "lat": 78.23,
    "lon": 15.39
}
```

Uses the first constellation layer found as the source for access window
computation (limited to 6 satellites for performance).

**Response**: 201
```json
{"layer_id": "layer-3"}
```

---

### PUT /api/layer/{layer_id}

Update layer mode or visibility.

**Request body**:
```json
{
    "mode": "snapshot",
    "visible": false
}
```

Both fields are optional. Changing `mode` regenerates the CZML.

**Modes**: `animated` (time-varying positions), `snapshot` (static points).

**Response**: 200
```json
{
    "layer_id": "layer-1",
    "mode": "snapshot",
    "visible": false
}
```

**Errors**: 404 if layer not found.

---

### DELETE /api/layer/{layer_id}

Remove a layer.

**Response**: 200
```json
{"status": "removed"}
```

**Errors**: 404 if layer not found.

---

### OPTIONS (any path)

CORS preflight. Returns 204 with appropriate headers.

## Layer lifecycle

```
POST /api/constellation  →  layer created (auto mode: animated ≤100 sats, snapshot >100)
POST /api/analysis       →  analysis layer created from source constellation
PUT /api/layer/{id}      →  update mode/visibility (mode change regenerates CZML)
GET /api/czml/{id}       →  fetch CZML for rendering
DELETE /api/layer/{id}   →  remove layer
```

## Error format

All errors return JSON:
```json
{"error": "Description of what went wrong"}
```
