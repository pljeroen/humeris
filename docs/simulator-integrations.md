# Simulator Integrations

Export constellations to 3D space simulators for interactive visualization.

## Universe Sandbox

[Universe Sandbox](https://universesandbox.com/) is a physics-based space
simulator. The `.ubox` format is a ZIP archive containing XML that defines
celestial bodies with Keplerian orbital elements.

### Basic export

```python
from constellation_generator import ShellConfig, generate_walker_shell
from constellation_generator.adapters.ubox_exporter import UboxExporter
from datetime import datetime, timezone

shell = ShellConfig(
    altitude_km=550, inclination_deg=53, num_planes=6, sats_per_plane=10,
    phase_factor=1, raan_offset_deg=0, shell_name="LEO-550",
)
sats = generate_walker_shell(shell)

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
UboxExporter().export(sats, "constellation.ubox", epoch=epoch)
```

Open `constellation.ubox` in Universe Sandbox. Each satellite appears as a
body orbiting Earth with correct Keplerian elements.

### With physical properties

Provide a `DragConfig` to set satellite mass and diameter in the simulation:

```python
from constellation_generator.domain.atmosphere import DragConfig

drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
UboxExporter(drag_config=drag).export(sats, "constellation.ubox", epoch=epoch)
```

This sets:
- **Mass**: from `mass_kg` (converted to Universe Sandbox units: 10^20 kg)
- **Diameter**: derived from `area_m2` assuming a circular cross-section

### With live data

```python
from constellation_generator.adapters.celestrak import CelesTrakAdapter

celestrak = CelesTrakAdapter()
gps = celestrak.fetch_satellites(group="GPS-OPS")
UboxExporter().export(gps, "gps.ubox")
```

### What the export includes

| Property | Source | Notes |
|----------|--------|-------|
| Semi-major axis | ECI position magnitude | km |
| Eccentricity | 0 for Walker shells | Non-zero for TLE-derived sats possible |
| Inclination | Angular momentum vector | Degrees |
| RAAN | `satellite.raan_deg` | Degrees |
| Arg of perigee | 0 for circular orbits | Degrees |
| Mean anomaly | `satellite.true_anomaly_deg` | = true anomaly for circular |
| Epoch date | `epoch` parameter | `YYYY-MM-DD` |
| Mass | `DragConfig.mass_kg` | Optional |
| Diameter | `DragConfig.area_m2` | Optional, derived |

## SpaceEngine

[SpaceEngine](https://spaceengine.org/) is a universe simulator. Custom
objects are added via `.sc` catalog files placed in the `addons/catalogs/planets/`
directory.

### Basic export

```python
from constellation_generator import ShellConfig, generate_walker_shell
from constellation_generator.adapters.spaceengine_exporter import SpaceEngineExporter
from datetime import datetime, timezone

shell = ShellConfig(
    altitude_km=550, inclination_deg=53, num_planes=6, sats_per_plane=10,
    phase_factor=1, raan_offset_deg=0, shell_name="LEO-550",
)
sats = generate_walker_shell(shell)

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
SpaceEngineExporter().export(sats, "constellation.sc", epoch=epoch)
```

### Installation in SpaceEngine

1. Export to `.sc` file
2. Copy to `SpaceEngine/addons/catalogs/planets/`
3. Launch SpaceEngine and navigate to Earth — satellites will be visible

### With physical properties

```python
from constellation_generator.domain.atmosphere import DragConfig

drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
SpaceEngineExporter(drag_config=drag).export(sats, "constellation.sc", epoch=epoch)
```

This sets:
- **Mass**: from `mass_kg` (converted to Earth masses)
- **Radius**: derived from `area_m2` assuming a circular cross-section (km)

### With live data

```python
from constellation_generator.adapters.celestrak import CelesTrakAdapter

celestrak = CelesTrakAdapter()
starlink = celestrak.fetch_satellites(group="STARLINK")
SpaceEngineExporter().export(starlink, "starlink.sc")
```

### What the export includes

| Property | Source | SpaceEngine unit |
|----------|--------|-----------------|
| Semi-major axis | ECI position magnitude | AU |
| Period | Derived from SMA via Kepler's 3rd law | Years |
| Eccentricity | 0 for Walker shells | Dimensionless |
| Inclination | Angular momentum vector | Degrees |
| Ascending node | `satellite.raan_deg` | Degrees |
| Arg of pericenter | 0 for circular orbits | Degrees |
| Mean anomaly | `satellite.true_anomaly_deg` | Degrees |
| Reference plane | Equator | Earth's equatorial plane |
| Mass | `DragConfig.mass_kg` | Earth masses (optional) |
| Radius | `DragConfig.area_m2` | km (optional) |

### SpaceEngine .sc format reference

Each satellite is exported as a `Moon` object (SpaceEngine's type for
objects orbiting planets):

```
Moon "LEO-550-Plane1-Sat1"
{
    ParentBody "Earth"

    Orbit
    {
        RefPlane "Equator"
        SemiMajorAxis 4.6300000000e-05
        Period 1.8300000000e-04
        Eccentricity 0.000000
        Inclination 53.0000
        AscendingNode 0.0000
        ArgOfPericenter 0.0000
        MeanAnomaly 0.0000
    }
}
```

## Combining with analysis data

Both exporters accept any list of `Satellite` objects. Combine synthetic
and live data, or filter by analysis results:

```python
from constellation_generator import generate_walker_shell, ShellConfig
from constellation_generator.adapters.celestrak import CelesTrakAdapter
from constellation_generator.adapters.ubox_exporter import UboxExporter
from constellation_generator.adapters.spaceengine_exporter import SpaceEngineExporter

# Mix synthetic + live
shell = ShellConfig(altitude_km=550, inclination_deg=53, num_planes=6,
                    sats_per_plane=10, phase_factor=1, raan_offset_deg=0,
                    shell_name="MyConstellation")
synthetic = generate_walker_shell(shell)

celestrak = CelesTrakAdapter()
iss = celestrak.fetch_satellites(name="ISS (ZARYA)")

all_sats = synthetic + iss

# Export to both formats
UboxExporter().export(all_sats, "mixed.ubox")
SpaceEngineExporter().export(all_sats, "mixed.sc")
```
