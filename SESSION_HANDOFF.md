# Session Handoff — Humeris v1.19.0+

## Current State

- **Tests**: 1543 passing
- **Last commit**: `29cc305` — Trade study guide + 7 simulator exporters
- **All pushed to remote**

## What Was Done This Session

### Simulator Exporters (all MIT, all committed+pushed)
1. **Universe Sandbox** (.ubox) — REWRITTEN from XML to JSON format (reverse-engineered from real .ubox files). 31 tests.
2. **SpaceEngine** (.sc) — existed before session. 17 tests.
3. **KSP** (.sfs) — NEW. ConfigNode VESSEL blocks, Kerbin-scaled orbits, probeCoreCube parts. 20 tests.
4. **Celestia** (.ssc) — NEW. Spacecraft catalog with EllipticalOrbit, Julian Date epoch. 18 tests.
5. **Google Earth** (.kml) — NEW. Position placemarks + 36-point orbit LineStrings. 26 tests.
6. **Blender** (.py) — NEW. Script generating Earth UV sphere + sat ico spheres + NURBS orbit curves. 27 tests.
7. **Stellarium/TLE** (.tle) — NEW. Standard 69-char Two-Line Elements. 20 tests.

### Docs
- `docs/simulator-integrations.md` — complete rewrite covering all 7 exporters with installation instructions
- `docs/trade-study-guide.md` — NEW. Pareto optimization + conjunction screening tutorial
- `examples/trade_study.py` — NEW. Runnable 25-config sweep producing 14-point Pareto front
- README updated with links

## What To Do Next — Science & Engineering Upgrades

The user wants ALL of the following implemented. This is the gap analysis cross-referenced against what ALREADY EXISTS.

### ALREADY EXISTS in codebase (do NOT re-implement):
- **Third-body perturbations**: `domain/third_body.py` — Sun+Moon gravity
- **SRP with shadow awareness**: `domain/numerical_propagation.py:187-243` — cannonball SRP model, checks eclipse
- **Atmospheric drag**: `domain/atmosphere.py` + `domain/numerical_propagation.py:142-184` — co-rotating atmosphere
- **Torque modeling**: `domain/torques.py` — gravity gradient + aerodynamic torques
- **Maneuvers**: `domain/maneuvers.py` — Hohmann, plane change, phasing
- **Beta angle**: `domain/eclipse.py:105-134` — compute_beta_angle + history + eclipse season prediction
- **J2/J3 perturbations**: `domain/numerical_propagation.py:84-139`
- **Orbital energy/momentum**: `domain/orbit_properties.py:95-141` — vis-viva, specific energy, h magnitude
- **Invariant tests**: 6 test classes (test_invariants_*.py) — A1-A6 two-body, H1-H4 collision, frames, geodetic, J2/SSO

### ACTUALLY MISSING — implement these:

#### 1. Conical Shadow Model (Penumbra)
- **File**: `domain/eclipse.py`
- **Current**: Cylindrical shadow only. `EclipseType.PENUMBRA` enum exists but `is_eclipsed()` never returns it.
- **Need**: Dual-cone model using Sun's finite angular radius. Compute umbra and penumbra half-angles. Return PENUMBRA when satellite is in partial shadow zone.
- **Math**: Sun angular radius α_s = arcsin(R_sun/d_sun), Earth angular radius α_e = arcsin(R_earth/d_earth). Penumbra cone half-angle = α_e + α_s. Umbra cone half-angle = α_e - α_s.
- **Impact**: Power budgeting (penumbra = 30-70% solar power, not 0%)

#### 2. Symplectic Integrator
- **File**: `domain/numerical_propagation.py`
- **Current**: RK4 only (`rk4_step` at line 248)
- **Need**: Add Störmer-Verlet (2nd order symplectic) and optionally Yoshida (4th order). These preserve Hamiltonian energy over long propagations.
- **Architecture**: Add alongside `rk4_step` as alternative step functions. The `propagate_numerical()` function already uses a composable force model — just add integrator selection.
- **Impact**: Month-long propagations without artificial energy drift

#### 3. Beta-Angle Thermal Logic
- **File**: `domain/eclipse.py` or new `domain/thermal.py`
- **Current**: Beta angle computed but not linked to thermal analysis. `environment_analysis.py` has stub data structures (`ThermalMonth`, `SeasonalThermalProfile`) but no actual heat computation.
- **Need**: Link beta angle + eclipse fraction to thermal equilibrium model. Flag "thermal danger zones" (beta ≈ 90° = no eclipse = potential overheating).
- **Math**: Absorbed power Q = α·S·A·cos(θ) + albedo + Earth IR. Equilibrium temp T = (Q/(ε·σ·A_rad))^0.25
- **Impact**: Operational constraint flagging for mission planning

#### 4. Propagation Conservation Tests
- **File**: new tests in `tests/`
- **Current**: Energy/momentum validated at single state points only. NOT tracked across propagation steps.
- **Need**: Propagate forward N steps, check ΔE and Δh bounded. Reversibility test (propagate forward then backward, check recovery). Symplectic integrator should conserve energy to ~10 decimal places.
- **Impact**: Validation that propagation is scientifically correct

#### 5. Event Detection with Root Finding
- **File**: `domain/eclipse.py` and potentially `domain/ground_track.py`
- **Current**: Coarse time-sweep with linear interpolation at boundaries. Accuracy depends entirely on step size.
- **Need**: Bisection or Brent's method to refine event times (eclipse entry/exit, ground track crossings) to machine precision after coarse detection.
- **Impact**: Sub-second event timing instead of step-size-limited

#### 6. Higher-Order Gravity (EGM96/JGM-3)
- **File**: `domain/numerical_propagation.py`
- **Current**: Only J2 + J3 (zonal harmonics)
- **Need**: Spherical harmonic expansion to at least degree/order 8x8 for tesseral harmonics (C22, S22, etc.). Full EGM96 (70x70) is aspirational but 8x8 covers the dominant terms.
- **Impact**: Frozen orbit design, GEO station-keeping, long-term drift prediction

#### 7. Orbit Determination (EKF)
- **File**: new `domain/orbit_determination.py`
- **Current**: Nothing
- **Need**: Extended Kalman Filter that ingests noisy position observations and produces a smoothed state estimate + covariance matrix. Start with simple position-only measurements.
- **Impact**: Ingest real TLE history, smooth data, realistic covariance for conjunction assessment

#### 8. Finite Burns
- **File**: `domain/maneuvers.py`
- **Current**: Impulsive burns only (instant ΔV)
- **Need**: Model engine-on duration, thrust profile, mass depletion (Tsiolkovsky rocket equation integrated over burn arc)
- **Impact**: Realistic maneuver planning for low-thrust electric propulsion

### Implementation Priority (user wants ALL)
1. Conical shadow (penumbra) — small, high-value, fixes existing gap
2. Symplectic integrator — medium, high-value, clean addition
3. Conservation tests — small, validates #2
4. Event detection (root finding) — small, improves eclipse + ground track
5. Beta-angle thermal — medium, links existing modules
6. Higher-order gravity — medium-large, significant new math
7. Finite burns — medium, extends existing maneuvers
8. EKF orbit determination — large, entirely new capability

### Architecture Notes
- ALL domain modules: stdlib only, no external deps
- TDD workflow: RED tests first, then GREEN implementation
- MIT core vs commercial: these are all commercial-tier features
- Hexagonal: domain pure, ports Protocol, adapters external
- Existing ForceModel Protocol in numerical_propagation.py is the extension point for gravity/forces
