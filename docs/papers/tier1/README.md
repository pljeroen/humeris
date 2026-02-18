# Tier 1 — Implemented, Validated (one archived)

Four algorithms are actively implemented in the Humeris domain layer, with
comprehensive test suites. Each paper documents the mathematical foundation,
implementation, and validation approach.

| Paper | Algorithm | Tests | Module |
|-------|-----------|-------|--------|
| 01 | Functorial Force Model Composition | 15 | `functorial_composition.py` |
| 02 | Hodge-CUSUM Topology Change Detector | 20 | `hodge_cusum.py` |
| 03 | Gramian-Guided Reconfiguration (G-RECON) | 20 | `gramian_reconfiguration.py` |
| 04 | Koopman-Spectral Conjunction Screening (KSCS, archived) | 22 (historical) | Removed after falsification |
| 05 | Competing-Risks Population Dynamics | 29 | `competing_risks.py` |

Active Tier-1 set excludes archived KSCS after falsification failure.

All modules follow the library's architecture: domain-pure (stdlib + numpy only),
frozen dataclasses for outputs, absolute imports, commercial license.
