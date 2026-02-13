# Novel Cross-Domain Algorithms — TLDR

Quick reference for the 15 algorithms derived from cross-disciplinary analysis
of the Humeris astrodynamics library.

## Tier 1 — Implemented, Validated

| # | Algorithm | One-liner | Module |
|---|-----------|-----------|--------|
| 1 | Functorial Force Composition | Compose force models with category-theoretic guarantees (associativity, commutativity check, RTN decomposition, frame pullbacks) | `functorial_composition.py` |
| 2 | Hodge-CUSUM Topology Detector | Monitor ISL network topology via Hodge Laplacian spectral features (Betti numbers, spectral gap) with CUSUM change-point detection | `hodge_cusum.py` |
| 3 | G-RECON | Use CW controllability Gramian eigenstructure to find minimum-fuel constellation reconfiguration maneuvers | `gramian_reconfiguration.py` |
| 4 | KSCS | Screen conjunctions in Koopman eigenvalue space (O(N²) spectral distance) before expensive trajectory propagation | `koopman_conjunction.py` |
| 5 | Competing Risks | Multi-risk survival analysis for satellite populations (drag, collision, component failure, deorbit as competing hazards) | `competing_risks.py` |

**Status**: All 5 implemented, 106 tests passing, wired into `__init__.py`.

## Tier 2 — Conceptual, Mathematically Grounded

| # | Algorithm | One-liner |
|---|-----------|-----------|
| 6 | Network SIR Cascade | SIR epidemic dynamics on ISL graph topology — debris propagates through network structure, percolation threshold = 1/λ_max(A) |
| 7 | Hamiltonian Fisher-Rao (HFRCP) | Symplectic covariance propagation preserving phase-space volume (Liouville's theorem), giving tighter uncertainty bounds than standard EKF |
| 8 | Surface Code Coverage (SCCP) | Map coverage grid to topological surface code — isolated gaps are correctable, only spanning failures cause global loss |
| 9 | Percolation-Debris Coupling | Debris density and ISL connectivity as coupled order parameters with phase transitions — "information Kessler syndrome" may precede classical cascade |
| 10 | Bayesian Intent (BIJE) | Joint HMM estimation of orbital state AND operator intent (station-keeping/deorbit/evasive/uncontrolled) from EKF residuals |

**Status**: Mathematically specified. Each paper identifies prerequisites and implementation complexity.

## Tier 3 — Speculative Frontier Ideas

| # | Algorithm | One-liner |
|---|-----------|-----------|
| 11 | Turing Morphogenesis | Reaction-diffusion on sphere produces evenly-spaced satellite patterns — hypothesizes Walker patterns are spherical harmonic Turing modes |
| 12 | Helmholtz Free Energy | Orbit slots as statistical mechanics system — F = E - TS balances collision risk (energy) against configuration flexibility (entropy) |
| 13 | Nash Equilibrium CA (NECA) | Conjunction avoidance as potential game — proves Nash equilibrium exists and best-response dynamics converge, PoA ≤ 2 |
| 14 | Melnikov Separatrix Surfing | Exploit homoclinic connections near unstable manifolds for near-zero-fuel station-keeping along dynamical "highways" |
| 15 | Spectral Gap Coverage | Maximize Fiedler value of coverage Laplacian — spectral gap controls how fast coverage recovers after satellite loss |

**Status**: Speculative. Each paper is honest about what's proven vs conjectured.

## Key Insight

These algorithms aren't independent — they form a web of connections:
- Koopman lifts orbital dynamics to where force composition is natural
- Hodge-CUSUM monitors the topology that Network SIR cascades threaten
- Gramian reconfiguration optimizes what Spectral Gap Coverage measures
- Competing Risks feeds the population dynamics that Percolation-Debris models

The library's breadth (71 domain modules) creates emergent capabilities
that no single module provides alone.
