# Getting Started

## Requirements

- Python 3.11+
- NumPy >= 1.24 (hard dependency)
- `sgp4>=2.22` for live CelesTrak data (optional)

## Installation

```bash
# Core only (synthetic constellations, no external deps)
pip install .

# With live CelesTrak / SGP4 support
pip install ".[live]"

# Development (includes pytest)
pip install ".[dev]"
```

## Quick start

### Generate a Walker shell

```python
from humeris import ShellConfig, generate_walker_shell

shell = ShellConfig(
    altitude_km=550, inclination_deg=53,
    num_planes=10, sats_per_plane=20,
    phase_factor=1, raan_offset_deg=0,
    shell_name="LEO-550",
)
satellites = generate_walker_shell(shell)
print(f"Generated {len(satellites)} satellites")
```

### Fetch live data

```python
from humeris.adapters.celestrak import CelesTrakAdapter

celestrak = CelesTrakAdapter()
gps = celestrak.fetch_satellites(group="GPS-OPS")
iss = celestrak.fetch_satellites(name="ISS (ZARYA)")
```

### Launch the interactive viewer

```bash
python view_constellation.py --serve
```

Opens a Cesium 3D globe at `http://localhost:8765` with pre-loaded Walker
shells and live ISS data. See [Viewer Server](viewer-server.md) for details.

### Run tests

```bash
pytest                          # 3095 tests, all offline
pytest tests/test_live_data.py  # live CelesTrak (requires network)
```

## CLI usage

```bash
# Default Walker shells → simulation JSON
humeris -i template.json -o output.json

# Live GPS constellation
humeris -i sim.json -o out.json --live-group GPS-OPS

# Starlink with concurrent SGP4 propagation
humeris -i sim.json -o out.json --live-group STARLINK --concurrent

# Export to CSV and GeoJSON
humeris -i sim.json -o out.json --export-csv sats.csv --export-geojson sats.geojson
```

## Next steps

- [Simulation JSON](simulation-json.md) — input/output JSON schema
- [Architecture](architecture.md) — hexagonal design, domain purity
- [Viewer Server](viewer-server.md) — interactive 3D viewer with 15 analysis types
- [API Reference](api-reference.md) — HTTP endpoints
- [Integration Guide](integration-guide.md) — CelesTrak, CesiumJS, custom sources, reproducibility
- [Export Formats](export-formats.md) — CSV, GeoJSON, CZML output
- [Licensing](licensing.md) — MIT core + commercial extensions
