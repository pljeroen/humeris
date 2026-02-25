# Accuracy Improvement Contracts — Overview

Systematic elimination of all identified accuracy compromises in the Humeris
orbital mechanics library. 24 findings from full codebase audit, organized
into 11 executable contracts in dependency order.

## Execution Order

| Contract | ID | Items | Effort | Dependencies |
|---|---|---|---|---|
| 1 | ACC-01 | Ephemeris unification | Low | None |
| 2 | ACC-02 | Time system fidelity | Low | None |
| 3 | ACC-03 | Frame rotation fix | Medium | ACC-01, ACC-02 |
| 4 | ACC-04 | Gravity model cleanup | Low | None |
| 5 | ACC-05 | Atmosphere drag wiring | Medium | None |
| 6 | ACC-06 | Tidal force fidelity | Low-Med | ACC-01 |
| 7 | ACC-07 | SRP and albedo fidelity | Medium | ACC-01 |
| 8 | ACC-08 | Dense output improvements | Medium | None |
| 9 | ACC-09 | Conjunction fidelity | Medium | None |
| 10 | ACC-10 | Full NRLMSISE-00 | High | ACC-05 |
| 11 | ACC-11 | Coverage and remaining | Low | None |

## Dependency Graph

```
ACC-01 (ephemeris) ──┬──> ACC-03 (frame rotation) ──> ACC-06 (tidal)
ACC-02 (time)     ───┘                             ──> ACC-07 (SRP/albedo)
ACC-04 (gravity)      [independent]
ACC-05 (atm wire) ────> ACC-10 (full NRLMSISE-00)
ACC-08 (dense out)     [independent]
ACC-09 (conjunction)   [independent]
ACC-11 (misc)          [independent]
```

## Parallelizable Groups

**Group A** (no dependencies): ACC-01, ACC-02, ACC-04, ACC-05, ACC-08, ACC-09, ACC-11
**Group B** (after ACC-01 + ACC-02): ACC-03, ACC-06, ACC-07
**Group C** (after ACC-05): ACC-10

## Estimated Total

- Low-effort contracts (1 session each): ACC-01, ACC-02, ACC-04, ACC-11
- Medium-effort contracts (1-2 sessions each): ACC-03, ACC-05, ACC-06, ACC-07, ACC-08, ACC-09
- High-effort contract (multi-session): ACC-10

## Audit Source

Full audit conducted 2026-02-23 with 24 findings across gravity, atmosphere,
time, frames, ephemeris, forces, integration, conjunction, statistics,
constants, coverage, and data format categories.
