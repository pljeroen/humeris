# Examples

## trade_study.py

Coverage-optimized constellation design with Pareto front analysis:

```bash
python examples/trade_study.py
```

Produces `optimized.csv` with the Pareto-optimal satellite parameters.

## Simulator files

Pre-generated constellation files — open directly in the target application:

| File | Application | How to use |
|------|-------------|------------|
| `constellation.ubox` | Universe Sandbox | File > Open Simulation > select file |
| `constellation.sc` | SpaceEngine | Copy to `addons/catalogs/` in SpaceEngine install dir, restart |

## Generating your own

```bash
# Generate constellation and export to any format
humeris -i sim_old.json -o sim.json --export-ubox my_constellation.ubox
humeris -i sim_old.json -o sim.json --export-spaceengine my_constellation.sc
humeris -i sim_old.json -o sim.json --export-kml my_constellation.kml
```

See [Simulator Integrations](../docs/simulator-integrations.md) for full setup instructions.
