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

# All Starlink satellites (~6000+)
constellation-generator -i sim.json -o out.json --live-group STARLINK

# Search by name
constellation-generator -i sim.json -o out.json --live-name "ISS (ZARYA)"

# By NORAD catalog number
constellation-generator -i sim.json -o out.json --live-catnr 25544
```

#### Available CelesTrak groups

`STATIONS`, `GPS-OPS`, `STARLINK`, `ONEWEB`, `ACTIVE`, `WEATHER`,
`GALILEO`, `BEIDOU`, `IRIDIUM-NEXT`, `PLANET`, `SPIRE`, `GEO`,
`INTELSAT`, `SES`, `TELESAT`, `AMATEUR`, `SCIENCE`, `NOAA`, `GOES`

### Programmatic

```python
# Synthetic Walker shell
from constellation_generator import ShellConfig, generate_walker_shell

shell = ShellConfig(
    altitude_km=550, inclination_deg=53,
    num_planes=10, sats_per_plane=20,
    phase_factor=1, raan_offset_deg=0,
    shell_name="Custom-Shell",
)
satellites = generate_walker_shell(shell)

# Live data from CelesTrak
from constellation_generator.adapters.celestrak import CelesTrakAdapter

celestrak = CelesTrakAdapter()
gps_sats = celestrak.fetch_satellites(group="GPS-OPS")
iss = celestrak.fetch_satellites(name="ISS (ZARYA)")

# Mix both and serialise
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
├── domain/                    # Pure logic — only stdlib math/dataclasses
│   ├── orbital_mechanics.py   # Kepler → Cartesian, SSO inclination
│   ├── constellation.py       # Walker shells, SSO bands, ShellConfig
│   ├── serialization.py       # Simulation format (Y/Z swap, precision)
│   └── omm.py                 # CelesTrak OMM record → OrbitalElements
├── ports/                     # Abstract interfaces (ABC)
│   ├── __init__.py            # SimulationReader, SimulationWriter
│   └── orbital_data.py        # OrbitalDataSource
├── adapters/                  # Infrastructure (JSON I/O, HTTP, SGP4)
│   ├── __init__.py            # JsonSimulationReader/Writer
│   └── celestrak.py           # CelesTrakAdapter, SGP4Adapter
└── cli.py                     # CLI entry point
```

The domain layer has zero external dependencies. All I/O (file access,
HTTP, SGP4 propagation) is confined to the adapter layer behind port
interfaces.

## Tests

```bash
pytest                       # all 35 tests
pytest tests/test_constellation.py   # 22 synthetic tests (offline)
pytest tests/test_live_data.py       # 13 live data tests (network)
```

## Credits

Original code by [Scott Manley](https://www.youtube.com/@scottmanley). Refactored and extended by Jeroen.

## License

MIT
