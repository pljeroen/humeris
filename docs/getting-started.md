# Getting Started

## Requirements

- Python 3.11+
- NumPy >= 1.24 (hard dependency)
- `sgp4>=2.22` for live CelesTrak data (optional)

**Platforms**: Linux, macOS, Windows (pure Python, no compiled extensions).

## Installation

### From PyPI

```bash
# Core MIT package (constellation generation, propagation, coverage, export)
pip install humeris-core

# With live CelesTrak / SGP4 support
pip install "humeris-core[live]"

# Full suite (core + 66 commercial analysis modules)
pip install humeris-pro
```

### From GitHub Releases

Download `.whl` files from the [Releases](https://github.com/pljeroen/humeris/releases) page:

```bash
pip install humeris_core-1.27.0-py3-none-any.whl
pip install humeris_pro-1.27.0-py3-none-any.whl
```

### Windows executable

Download `humeris-windows-x64.zip` from [Releases](https://github.com/pljeroen/humeris/releases), extract, and run:

```
humeris.exe -i simulation.json -o output.json
humeris.exe --serve
```

No Python installation required.

### Development (from source)

```bash
git clone https://github.com/pljeroen/humeris.git
cd humeris
pip install -e ./packages/core -e ./packages/pro
```

## Quick start

### Generate a Walker shell

```python
from humeris.domain.constellation import ShellConfig, generate_walker_shell

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
# Via CLI entry point
humeris --serve

# Or via script directly
python scripts/view_constellation.py --serve
```

Opens a Cesium 3D globe at `http://localhost:8765` with pre-loaded Walker
shells and live ISS data. See [Viewer Server](viewer-server.md) for details.

### Run tests

```bash
pytest                          # 3272 tests, all offline
pytest tests/test_live_data.py  # live CelesTrak (requires network)
```

## CLI reference

### Synthetic constellations (Walker shells)

```bash
# Default: 3 Walker shells + SSO band
humeris -i simulation_old.json -o simulation.json

# Custom base ID
humeris -i sim_old.json -o sim.json --base-id 200
```

### Live data from CelesTrak

```bash
# Real GPS constellation (32 satellites)
humeris -i sim.json -o out.json --live-group GPS-OPS

# All Starlink satellites (~6000+) with concurrent SGP4 propagation
humeris -i sim.json -o out.json --live-group STARLINK --concurrent

# Search by name
humeris -i sim.json -o out.json --live-name "ISS (ZARYA)"

# By NORAD catalog number
humeris -i sim.json -o out.json --live-catnr 25544
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
humeris -i sim.json -o out.json --export-csv satellites.csv

# GeoJSON export
humeris -i sim.json -o out.json --export-geojson satellites.geojson

# Both at once
humeris -i sim.json -o out.json --export-csv sats.csv --export-geojson sats.geojson
```

CSV columns: `name`, `lat_deg`, `lon_deg`, `alt_km`, `epoch`, `plane_index`,
`sat_index`, `raan_deg`, `true_anomaly_deg`, `altitude_km`, `inclination_deg`,
`orbital_period_min`, `beta_angle_deg`, `atmospheric_density_kg_m3`, `l_shell`.

GeoJSON produces a FeatureCollection with Point geometries. Coordinates
follow the GeoJSON spec: `[longitude, latitude, altitude_km]`. Properties
include the same orbital analysis fields as CSV.

### Simulator exports

Export directly to 3D space simulators, game engines, and planetarium software:

```bash
humeris -i sim.json -o out.json --export-celestia sats.ssc      # Celestia
humeris -i sim.json -o out.json --export-kml sats.kml            # Google Earth
humeris -i sim.json -o out.json --export-tle sats.tle            # Stellarium / STK / GMAT
humeris -i sim.json -o out.json --export-blender sats.py         # Blender
humeris -i sim.json -o out.json --export-spaceengine sats.sc     # SpaceEngine
humeris -i sim.json -o out.json --export-ksp sats.sfs            # Kerbal Space Program
humeris -i sim.json -o out.json --export-ubox sats.ubox          # Universe Sandbox
```

Optional visual layer flags:

```bash
--no-orbits          # Omit orbit path lines from KML and Blender exports
--kml-planes         # Organize KML by orbital plane folders
--kml-isl            # Include ISL topology lines in KML export
--blender-colors     # Color-code satellites by orbital plane in Blender export
```

All exporters include orbital analysis data (altitude, inclination, period,
beta angle, atmospheric density, L-shell). See
[Simulator Integrations](simulator-integrations.md) for setup instructions
per tool.

## Default shell configuration

| Shell | Altitude | Inclination | Planes x Sats | Phase Factor |
|-------|----------|-------------|---------------|--------------|
| LEO-Shell500 | 500 km | 30 | 22 x 72 | 17 |
| LEO-Shell450 | 450 km | 30 | 22 x 72 | 17 |
| LEO-Shell400 | 400 km | 30 | 22 x 72 | 17 |
| SSO Band | 525-2200 km (50 km step) | SSO-computed | 1 x 72 | 0 |

## Next steps

- [Python API Examples](python-api.md) — worked examples for every module
- [Simulation JSON](simulation-json.md) — input/output JSON schema
- [Architecture](architecture.md) — hexagonal design, domain purity
- [Viewer Server](viewer-server.md) — interactive 3D viewer with 21 analysis types
- [API Reference](api-reference.md) — HTTP endpoints
- [Integration Guide](integration-guide.md) — CelesTrak, CesiumJS, custom sources, reproducibility
- [Export Formats](export-formats.md) — CSV, GeoJSON, CZML output
- [Validation](validation.md) — reference tests, GMAT parity, determinism
- [Licensing](licensing.md) — MIT core + commercial extensions
