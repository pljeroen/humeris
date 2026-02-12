# Pre-Release Checklist — Constellation Generator

## DONE (verified)

- [x] **944 tests passing, 36 purity tests, all GREEN**
  - Done: `pytest` exits 0 with 944 passed, 0 failed. 36 purity tests confirm stdlib-only domain.

- [x] **Hexagonal architecture with stdlib-only domain (34 modules)**
  - Done: No domain module imports anything outside stdlib. Purity tests enforce this mechanically.

- [x] **Dual license in place (MIT + commercial EUR 10k)**
  - Done: `LICENSE` (MIT) covers all files except 4 Pro modules. `LICENSE-COMMERCIAL.md` covers `atmosphere.py`, `lifetime.py`, `station_keeping.py`, `conjunction.py` at EUR 10,000 one-time per-org.

- [x] **Feature set complete**
  - Propagation: Keplerian, J2, J3, numerical RK4
  - Perturbations: drag, SRP, solar third-body, lunar third-body
  - Environment: atmosphere model, radiation, eclipse prediction, solar ephemeris, lunar ephemeris
  - Attitude: gravity-gradient torques, aerodynamic torques
  - Relative motion: Clohessy-Wiltshire / Hill equations
  - Communications: ISL topology, link budget
  - Observation: ground stations, access windows, sensor modeling, coverage analysis, revisit analysis, DOP
  - Orbit design: SSO, frozen orbits, repeat ground track
  - Maintenance: station-keeping, deorbit compliance, lifetime estimation, maneuver planning
  - Analysis: conjunction/collision probability, beta angle, eclipse seasons, ground track crossings, pass analysis, constellation metrics
  - Done: Each feature has corresponding domain module(s) and passing tests.

- [x] **Is/Is Not positioning defined**
  - IS: serious simulation toolkit for research, engineering, mission analysis, feeding downstream viewers (CesiumJS/CZML/CSV/GeoJSON)
  - IS NOT: certified flight system, full UI application, turnkey operations platform
  - Done: Positioning is stated here and informs all documentation scope.

- [x] **License scope defined**
  - Per-organization, perpetual, source-available, no SLA, no safety-critical warranty, no liability
  - Done: `LICENSE-COMMERCIAL.md` contains explicit "What you get" / "What you don't get" sections. No ambiguity on scope.

## REMAINING (concrete work)

- [ ] **1. Code coverage measurement**
  - Task: Install `pytest-cov`, run full suite, establish baseline percentage.
  - Done when: `pytest --cov` produces a coverage report, and the baseline percentage is recorded in this checklist or a dedicated file.

- [ ] **2. External validation**
  - Task: Compare at least one propagation output against a published reference (Vallado example problems, IAC test cases, or JPL Horizons data).
  - Done when: A test exists in the test suite that compares computed output against external reference data with a documented tolerance, and the test passes.

- [ ] **3. Accuracy documentation**
  - Task: Document propagation error bounds, ephemeris accuracy limits, atmosphere model validity range, and supported orbit regimes.
  - Done when: `ACCURACY.md` exists at project root with quantified statements (not qualitative hedging). Each claim references the model or algorithm used and its known limitations.

- [ ] **4. CLI test coverage**
  - Task: Write tests for `cli.py` argument parsing, default shell generation, export paths, error handling, and edge cases.
  - Done when: `tests/test_cli.py` exists with at least 10 tests covering argument parsing, valid invocations, invalid input rejection, and help output. All pass.

- [ ] **5. Hypothesis / property-based testing**
  - Task: Implement property-based tests using Hypothesis for at least 3 domain modules (governance requirement from TDDv6).
  - Done when: `hypothesis` is listed in dev dependencies, and at least 15 property-based tests exist across 3+ domain modules. All pass.

- [ ] **6. Domain limitation disclaimers**
  - Task: Add a "Limitations" section to `README.md` and an accuracy caveat to `cli.py --help` output.
  - Done when: `README.md` contains a "Limitations" section listing model boundaries and disclaimers. `constellation-generator --help` prints an accuracy caveat. Both are verifiable by inspection.

## BLOCKED (dependencies)

(none)
