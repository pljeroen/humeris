# Validation

## Posture

This project treats external references as trusted baselines. The goal is not
to claim superiority, but to learn how far a small, iterative engineering effort
can go under reproducible tests.

We test against textbook values, industry-standard propagators, and archived
GMAT outputs — and publish the comparison artifacts, including cases where our
results are imperfect or still evolving. Any "better" result is treated as
provisional until independently re-validated.

## Reference tests

The library includes 100+ validation tests that cross-check results against
independent references:

- **Vallado** — Circular orbit velocities, orbital periods, Hohmann transfer
  delta-V, J2 RAAN drift rates, SSO inclinations, and atmospheric density
  values from *Fundamentals of Astrodynamics and Applications* (4th ed.)
- **SGP4** — Position, velocity, and orbital properties compared against the
  industry-standard SGP4 propagator using hardcoded ISS, GPS, and GEO OMM
  records
- **SP3 precise ephemeris** — Parser for IGS Standard Product #3 format
  (centimeter-accurate post-processed GNSS orbits), with physical consistency
  checks against known GPS constellation geometry
- **Real-world scenarios** — Six historical spaceflight events validated
  against published data:
  - *'Oumuamua* — Hyperbolic orbit math (e > 1), periapsis distance exact to
    1e-10, positive specific energy cross-checked against vis-viva
  - *ISS* — Orbital period within 0.5 min of 92.68 min, J2 RAAN drift within
    0.15 deg/day of -5.0 deg/day, revolutions/day within 0.1 of 15.54
  - *Tiangong-1* — Orbit lifetime from 340 km converges to weeks-to-months
    (actual reentry ~91 days), monotonic decay trajectory, reentry at 100 km
  - *Starlink* — Hohmann 440->550 km within 5 m/s of ~60 m/s, transfer time
    within 2 min of ~46 min, Walker shell geometry and RAAN separation exact
  - *ENVISAT* — SSO inclination at 770 km within 0.5 deg of 98.55 deg, RAAN
    drift rate within 0.02 deg/day of solar rate (0.9856 deg/day)
  - *Iridium 33 / Cosmos 2251* — Collision relative velocity within 2 km/s of
    11.7 km/s, B-plane conjunction assessment, SIR cascade debris increase,
    orbital periods within 1.5 min of ~100 min
- **Internal cross-checks** — Energy conservation, angular momentum
  invariance, vis-viva identity, coordinate frame round-trips, eclipse
  fraction vs eclipse windows agreement, J2 drift vs SSO condition
  consistency, and propagation element recovery

## GMAT parity

We mirror core GMAT scenarios with native Humeris propagation and compare
against archived GMAT outputs from a separate test-suite repository. This is
a reference check, not a certification claim.

Reference test-suite: [`testsuite_gmat`](https://github.com/pljeroen/testsuite_gmat)

```bash
python scripts/run_gmat_mirror_compare.py \
  --gmat-repo /path/to/gmat
```

This produces commit-linked artifacts under `docs/gmat-parity-runs/` with:

- Humeris mirrored scenario outputs
- Parsed GMAT scenario outputs
- Per-metric comparison report and pass/fail status
- Git references for both repositories

The mirror covers:

- `basic_leo_two_body`
- `advanced_j2_raan_drift`
- `advanced_oumuamua_hyperbolic` (regime parity: hyperbolic behavior checks)
- `advanced_oumuamua_suncentric` (high-fidelity extension with third-body + SRP force stack)

Additional artifacts per run:

- `profile_behavior_annex.json` — conservative/nominal/aggressive screening behavior
- `profile_behavior_history.json` — cross-run profile behavior ledger
- `replay_bundle.json` — deterministic replay package for incident/debug reproduction

Latest parity artifacts:

- [`LATEST`](gmat-parity-runs/LATEST)
- [`LATEST_REPORT`](gmat-parity-runs/LATEST_REPORT)

## Determinism and reproducibility

Understanding what is deterministic matters for research reproducibility,
simulation comparisons, and audits.

| Mode | Deterministic? | Details |
|------|----------------|---------|
| Synthetic Walker shells | Yes | Same `ShellConfig` always produces identical `Satellite` objects. Pure math, no external state. |
| Live CelesTrak (sequential) | For a given TLE set | SGP4 is deterministic. But CelesTrak updates TLEs up to every 2 hours, so re-fetching tomorrow may yield different input data. |
| Live CelesTrak (concurrent) | Same results, different ordering | `--concurrent` uses `as_completed()` — satellite list order depends on thread scheduling. Values are identical. |
| Numerical propagation (RK4) | Yes | Fixed-step RK4 with deterministic force models. Same initial state + same forces = same trajectory. |

**TLE epoch in outputs**: The `Satellite` domain object stores the TLE
epoch in its `epoch` field. However, the simulation JSON output
(`-o output.json`) contains only `Position` and `Velocity` strings — the
TLE epoch is **not** written to the output file. If you need epoch
traceability, use the CSV or GeoJSON exporters (which include the `epoch`
column) or save the raw OMM snapshot (see
[Integration Guide](integration-guide.md#reproducibility-and-determinism)).

**Reproducing a live run**: Save the raw OMM JSON before propagation.
Replay it later with `SGP4Adapter.omm_to_satellite()` for bit-identical
results without network access.
