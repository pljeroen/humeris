# Simulator Integrations

Export constellations to 3D space simulators, game engines, and planetarium
software for interactive visualization.

All exporters implement the `SatelliteExporter` protocol and accept any
`list[Satellite]` — synthetic Walker shells, live CelesTrak data, or both.

| Exporter | Format | Target | Physical props |
|----------|--------|--------|---------------|
| `UboxExporter` | `.ubox` (ZIP/JSON) | Universe Sandbox | Mass, radius |
| `SpaceEngineExporter` | `.sc` (text catalog) | SpaceEngine | Mass, radius |
| `KspExporter` | `.sfs` (ConfigNode) | Kerbal Space Program | Mass |
| `CelestiaExporter` | `.ssc` (text catalog) | Celestia | Mass, radius |
| `KmlExporter` | `.kml` (XML) | Google Earth | — |
| `BlenderExporter` | `.py` (Python script) | Blender | — |
| `StellariumExporter` | `.tle` (Two-Line Elements) | Stellarium, STK, GMAT | — |

---

## Universe Sandbox

[Universe Sandbox](https://universesandbox.com/) is a physics-based space
simulator. The `.ubox` format is a ZIP archive containing `simulation.json`
with Earth and satellite body entities using ECI state vectors, plus metadata
files (`version.ini`, `info.json`, `ui-state.json`).

### Basic export

```python
from humeris import ShellConfig, generate_walker_shell
from humeris.adapters.ubox_exporter import UboxExporter
from datetime import datetime, timezone

shell = ShellConfig(
    altitude_km=550, inclination_deg=53, num_planes=6, sats_per_plane=10,
    phase_factor=1, raan_offset_deg=0, shell_name="LEO-550",
)
sats = generate_walker_shell(shell)

epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
UboxExporter().export(sats, "constellation.ubox", epoch=epoch)
```

### Opening in Universe Sandbox

1. Open Universe Sandbox
2. Go to **Open** and browse to your `constellation.ubox` file
3. Earth will load with all satellites orbiting it
4. Use the time controls to watch the constellation evolve
5. Click any satellite to inspect its properties (mass, velocity, orbit)

Satellites are rendered as particles by default. To make them more visible,
increase the **Particle Scale** in View settings, or export with a larger
`area_m2` in `DragConfig` to give them a bigger radius.

### With physical properties

```python
from humeris.domain.atmosphere import DragConfig

drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
UboxExporter(drag_config=drag).export(sats, "constellation.ubox", epoch=epoch)
```

This sets:
- **Mass**: from `mass_kg` (kg, used directly)
- **Radius**: derived from `area_m2` assuming a circular cross-section (metres)

---

## SpaceEngine

[SpaceEngine](https://spaceengine.org/) is a universe simulator. Custom
objects are added via `.sc` catalog files placed in the `addons/catalogs/planets/`
directory.

### Basic export

```python
from humeris.adapters.spaceengine_exporter import SpaceEngineExporter

SpaceEngineExporter().export(sats, "constellation.sc", epoch=epoch)
```

### Installation in SpaceEngine

1. Export to `.sc` file
2. Copy to `SpaceEngine/addons/catalogs/planets/`:
   - **Linux (Steam)**: `~/.local/share/Steam/steamapps/common/SpaceEngine/addons/catalogs/planets/`
   - **Windows (Steam)**: `C:\Program Files (x86)\Steam\steamapps\common\SpaceEngine\addons\catalogs\planets\`
   - **Windows (standalone)**: `C:\SpaceEngine\addons\catalogs\planets\`
3. Launch SpaceEngine (or restart if already running)
4. Press `F3` to open the search dialog and type a satellite name
   (e.g. `LEO-550-Plane1-Sat1`) to fly directly to it
5. Alternatively, use the planetary system browser (`F2` -> Solar System ->
   Earth) and scroll through the moons list — satellites appear as Moon objects

Satellites are tiny (sub-metre radius by default) and won't be visible from
far away. To make them easier to spot, export with a larger cross-section:

```python
drag = DragConfig(cd=2.2, area_m2=1000.0, mass_kg=260.0)
SpaceEngineExporter(drag_config=drag).export(sats, "constellation.sc", epoch=epoch)
```

### With physical properties

```python
drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
SpaceEngineExporter(drag_config=drag).export(sats, "constellation.sc", epoch=epoch)
```

- **Mass**: from `mass_kg` (converted to Earth masses)
- **Radius**: derived from `area_m2` (km)

---

## Kerbal Space Program

[KSP](https://www.kerbalspaceprogram.com/) uses ConfigNode `.sfs` save
files. The exporter generates VESSEL blocks with orbital elements scaled
from Earth to Kerbin so constellations appear at proportionally correct
altitudes.

### Basic export

```python
from humeris.adapters.ksp_exporter import KspExporter

KspExporter().export(sats, "constellation.sfs", epoch=epoch)
```

### Installation in KSP

1. Export to `.sfs` file
2. Open your KSP save's `persistent.sfs` (in `saves/<name>/`)
3. Find the `FLIGHTSTATE { }` block inside `GAME { }`
4. Paste the VESSEL blocks from the exported file before the closing `}`
5. Save and load the game — satellites appear as Probe vessels orbiting Kerbin

### Kerbin scaling

By default, orbital elements are scaled from Earth to Kerbin using the
radius ratio (600 km / 6371 km). Orbits that would fall below Kerbin's
70 km atmosphere are clamped to 80 km altitude.

To disable scaling and use raw Earth-scale values around Kerbin:

```python
KspExporter(scale_to_kerbin=False).export(sats, "constellation.sfs", epoch=epoch)
```

### With physical properties

```python
drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
KspExporter(drag_config=drag).export(sats, "constellation.sfs", epoch=epoch)
```

- **Mass**: from `mass_kg` (converted to metric tons for KSP)
- Each vessel uses a `probeCoreCube` part with `ModuleCommand`

---

## Celestia

[Celestia](https://celestia.space/) is a free, open-source 3D space
simulator. Custom objects are added via `.ssc` catalog files.

### Basic export

```python
from humeris.adapters.celestia_exporter import CelestiaExporter

CelestiaExporter().export(sats, "constellation.ssc", epoch=epoch)
```

### Installation in Celestia

1. Export to `.ssc` file
2. Copy to Celestia's `extras/` directory (or any subdirectory within it)
3. Launch Celestia and navigate to Earth — satellites appear as spacecraft
4. Press `Enter`, type a satellite name, and press `Enter` again to find it
5. Press `G` to go to it, or `F` to follow it

### With physical properties

```python
drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=260.0)
CelestiaExporter(drag_config=drag).export(sats, "constellation.ssc", epoch=epoch)
```

- **Mass**: from `mass_kg` (kg)
- **Radius**: derived from `area_m2` (km); default 0.001 km without DragConfig

### Format details

Each satellite is a `spacecraft` object parented to `"Sol/Earth"` with
`EllipticalOrbit` using Keplerian elements (SMA in km, period in days,
epoch as Julian Date).

---

## Google Earth (KML)

[Google Earth](https://earth.google.com/) supports KML files for geographic
visualization. Each satellite gets a position marker and a full orbit path.

### Basic export

```python
from humeris.adapters.kml_exporter import KmlExporter

KmlExporter().export(sats, "constellation.kml", epoch=epoch)
```

### Opening in Google Earth

1. Export to `.kml` file
2. Open it in Google Earth Pro (desktop) or import in Google Earth Web
3. Each satellite appears as a placemark at its current position
4. Orbit paths are drawn as 3D lines at orbital altitude

You can also open `.kml` files in QGIS, ArcGIS, or any GIS software.

### Custom name

```python
KmlExporter(name="Starlink Shell 1").export(sats, "starlink.kml", epoch=epoch)
```

### What the export includes

- **Position placemarks**: lat/lon/alt from ECI-to-geodetic conversion
- **Orbit paths**: 36-point LineString tracing the full orbit at altitude
- **3D altitude**: `altitudeMode` set to `absolute` (metres above WGS84)
- Satellites grouped in Folders by name for easy toggling

---

## Blender

[Blender](https://www.blender.org/) is a free 3D creation suite. The
exporter generates a Python script that creates the constellation
visualization when run inside Blender.

### Basic export

```python
from humeris.adapters.blender_exporter import BlenderExporter

BlenderExporter().export(sats, "constellation.py", epoch=epoch)
```

### Running in Blender

1. Export to `.py` file
2. Open Blender
3. Go to **Scripting** workspace (or open a Text Editor panel)
4. Open the exported `.py` file
5. Click **Run Script** (or press `Alt+P`)

The script creates:
- **Earth** as a UV sphere (radius 6.371 Blender units = 1 unit per km)
- **Satellites** as small ico spheres at their ECI positions (in km)
- **Orbit curves** as 3D NURBS circles for each satellite

### Custom sizes

```python
BlenderExporter(
    earth_radius_units=6.371,  # default
    sat_radius_units=0.1,      # larger satellites
).export(sats, "constellation.py", epoch=epoch)
```

### Rendering tips

- Add a material to Earth and apply a Blue Marble texture for realism
- Parent all satellites to an Empty and animate rotation for orbit motion
- Use Cycles renderer with emission shaders on satellites for glow effects

---

## Stellarium (TLE)

[Stellarium](https://stellarium.org/) is a free planetarium for your
computer. The exporter generates standard Two-Line Element (TLE) data
that works with Stellarium's satellite plugin, as well as STK, GMAT,
and any other TLE-consuming software.

### Basic export

```python
from humeris.adapters.stellarium_exporter import StellariumExporter

StellariumExporter().export(sats, "constellation.tle", epoch=epoch)
```

### Installation in Stellarium

1. Export to `.tle` file
2. Open Stellarium, go to **Configuration** -> **Plugins** -> **Satellites**
3. Click **Configure**, then **Sources**
4. Add the `.tle` file as a local source (or paste the TLE data)
5. Update sources — satellites appear in the sky view
6. Point your view at the sky to see them pass overhead

### Custom catalog numbers

TLE catalog numbers start at 99001 by default (above NORAD range to avoid
conflicts). To change the starting number:

```python
StellariumExporter(catalog_start=80001).export(sats, "constellation.tle", epoch=epoch)
```

### TLE format

Each satellite produces a standard 3-line TLE entry:

```
LEO-550-Plane1-Sat1
1 99001U 26001A   26079.50000000  .00000000  00000-0  00000-0 0  9990
2 99001  53.0000   0.0000 0000000   0.0000   0.0000 15.53720000    00
```

Mean motion is computed from the orbital radius via Kepler's third law.
Epoch is encoded as 2-digit year + fractional day of year.

### Other TLE consumers

The same `.tle` file works with:
- **STK** (Systems Tool Kit): Import as satellite database
- **GMAT** (NASA): Load as TLE ephemeris source
- **GPredict**: Import for real-time tracking
- **Orbitron**: Import for pass prediction
- **PyEphem / Skyfield**: Load directly in Python

---

## Combining with analysis data

All exporters accept any `list[Satellite]`. Combine synthetic and live
data, or filter by analysis results:

```python
from humeris import generate_walker_shell, ShellConfig
from humeris.adapters.celestrak import CelesTrakAdapter
from humeris.adapters.ubox_exporter import UboxExporter
from humeris.adapters.spaceengine_exporter import SpaceEngineExporter
from humeris.adapters.ksp_exporter import KspExporter
from humeris.adapters.kml_exporter import KmlExporter
from humeris.adapters.blender_exporter import BlenderExporter
from humeris.adapters.stellarium_exporter import StellariumExporter

# Generate constellation
shell = ShellConfig(altitude_km=550, inclination_deg=53, num_planes=6,
                    sats_per_plane=10, phase_factor=1, raan_offset_deg=0,
                    shell_name="MyConstellation")
sats = generate_walker_shell(shell)

# Export to all formats
UboxExporter().export(sats, "constellation.ubox")
SpaceEngineExporter().export(sats, "constellation.sc")
KspExporter().export(sats, "constellation.sfs")
CelestiaExporter().export(sats, "constellation.ssc")
KmlExporter().export(sats, "constellation.kml")
BlenderExporter().export(sats, "constellation.py")
StellariumExporter().export(sats, "constellation.tle")
```

### With live data

```python
celestrak = CelesTrakAdapter()
starlink = celestrak.fetch_satellites(group="STARLINK")

# Visualize real Starlink in any simulator
KmlExporter(name="Starlink").export(starlink, "starlink.kml")
StellariumExporter().export(starlink, "starlink.tle")
```
