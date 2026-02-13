# Simulation JSON Format

The CLI reads an input simulation JSON and writes an output JSON with
generated satellite entities appended.

## Input JSON (minimal)

The input must contain an `Entities` array with at least two entities:

1. An **Earth** entity (matched by `"Name": "Earth"`)
2. A **Satellite template** entity (matched by `"Name": "Satellite"` or
   `--template-name`)

```json
{
  "Entities": [
    {
      "Id": 0,
      "Name": "Earth",
      "Position": "0;0;0",
      "Velocity": "0;0;0"
    },
    {
      "Id": 1,
      "Name": "Satellite",
      "Position": "0;0;0",
      "Velocity": "0;0;0",
      "CustomField": "preserved-in-output"
    }
  ]
}
```

The template entity is deep-copied for each generated satellite. Any extra
fields on the template (e.g. `CustomField`, nested objects) are preserved
in each copy. This lets you carry simulation-tool-specific metadata
through the pipeline.

The input JSON may contain additional top-level keys beyond `Entities`.
They are passed through to the output unchanged.

## Output JSON

The output is the input JSON with generated satellite entities appended
to the `Entities` array. Each satellite entity gets:

| Field | Source | Example |
|-------|--------|---------|
| `Id` | Sequential from `--base-id` (default 100) | `100` |
| `Name` | Generated satellite name | `"LEO-Shell500-P00-S00"` |
| `Position` | ECI position with Y/Z swap (meters) | `"6878137.000;0.000;0.000"` |
| `Velocity` | ECI velocity with Y/Z swap (m/s) | `"0.000000;7612.654321;0.000000"` |

All other fields are copied from the template entity.

### Position and velocity format

Values are semicolon-delimited strings with a **Y/Z axis swap**:

```
ECI [x, y, z]  -->  "{x:.3f};{z:.3f};{y:.3f}"    (position, 3 decimals)
ECI [vx, vy, vz]  -->  "{vx:.6f};{vz:.6f};{vy:.6f}"  (velocity, 6 decimals)
```

This swap matches the coordinate convention of the target simulation tool
(Y-up vs Z-up). The ECI frame used is J2000 (TEME for SGP4-derived
satellites).

### Satellite naming

Synthetic Walker shell satellites are named:

```
{shell_name}-P{plane_index:02d}-S{sat_index:02d}
```

Example: `LEO-Shell500-P03-S17` (shell "LEO-Shell500", plane 3, satellite 17).

CelesTrak satellites keep their catalog name (e.g. `ISS (ZARYA)`,
`GPS BIIR-2 (PRN 13)`).

## Full example

**Input** (`template.json`):

```json
{
  "SimulationName": "My Sim",
  "TimeStep": 0.1,
  "Entities": [
    {"Id": 0, "Name": "Earth", "Position": "0;0;0", "Velocity": "0;0;0"},
    {"Id": 1, "Name": "Satellite", "Position": "0;0;0", "Velocity": "0;0;0", "Mass": 250}
  ]
}
```

**Command**:

```bash
humeris -i template.json -o output.json --base-id 100
```

**Output** (`output.json`):

```json
{
  "SimulationName": "My Sim",
  "TimeStep": 0.1,
  "Entities": [
    {"Id": 0, "Name": "Earth", "Position": "0;0;0", "Velocity": "0;0;0"},
    {"Id": 1, "Name": "Satellite", "Position": "0;0;0", "Velocity": "0;0;0", "Mass": 250},
    {"Id": 100, "Name": "LEO-Shell500-P00-S00", "Position": "6878137.000;0.000;0.000", "Velocity": "0.000000;7612.654321;0.000000", "Mass": 250},
    {"Id": 101, "Name": "LEO-Shell500-P00-S01", "Position": "6431234.567;2345678.901;1234567.890", "Velocity": "-1234.567890;7012.345678;2345.678901", "Mass": 250}
  ]
}
```

Note that `Mass: 250` is carried from the template to each generated entity.

## CLI flags affecting output

| Flag | Effect |
|------|--------|
| `--base-id N` | Starting entity ID (default: 100) |
| `--template-name NAME` | Template entity name to match (default: `Satellite`) |
| `--live-group GROUP` | Use CelesTrak data instead of synthetic shells |
| `--live-name NAME` | Search CelesTrak by satellite name |
| `--live-catnr N` | Search CelesTrak by NORAD catalog number |

## Programmatic access

```python
from humeris.adapters import JsonSimulationReader, JsonSimulationWriter
from humeris.domain.serialization import build_satellite_entity

reader = JsonSimulationReader()
writer = JsonSimulationWriter()

sim = reader.read_simulation("template.json")
template = reader.extract_template_entity(sim, "Satellite")

# Build entity from a Satellite domain object
entity = build_satellite_entity(satellite, template, base_id=100)
sim["Entities"].append(entity)

writer.write_simulation(sim, "output.json")
```
