# Commercial License — Humeris

## 1. Covered files

The following files are **not** covered by the MIT License:

**Domain modules** (`src/humeris/domain/`):

- `adaptive_integration.py` — Dormand-Prince RK4(5) adaptive integrator
- `atmosphere.py` — Exponential density model, drag acceleration
- `cascade_analysis.py` — Cascade/fragmentation indicators
- `communication_analysis.py` — Network capacity analysis
- `conjunction.py` — Screening, TCA, B-plane, collision probability
- `conjunction_management.py` — Conjunction management workflows
- `kessler_heatmap.py` — Kessler spatial density heatmap, percolation, cascade k_eff
- `constellation_metrics.py` — Coverage/revisit/eclipse statistics, scoring
- `constellation_operability.py` — Operability index
- `control_analysis.py` — CW controllability Gramian
- `coverage_optimization.py` — Coverage optimization
- `decay_analysis.py` — Exponential scale map
- `deorbit.py` — Deorbit lifetime estimation
- `design_optimization.py` — Coverage drift, mass efficiency frontier
- `design_sensitivity.py` — Spectral fragility, altitude sensitivity
- `dilution_of_precision.py` — Fisher information DOP
- `earth_orientation.py` — Earth Orientation Parameters (UT1-UTC, polar motion)
- `eclipse.py` — Shadow geometry, beta angle, eclipse windows
- `environment_analysis.py` — Combined environment assessment
- `graph_analysis.py` — Graph-theoretic ISL (Fiedler, fragmentation)
- `gravity_field.py` — Cunningham V/W spherical harmonic gravity (EGM96 70x70)
- `hazard_reporting.py` — NASA-STD-8719.14 hazard classification, breakup potential, CWI
- `information_theory.py` — BEC channel, coverage spectrum, marginal value
- `inter_satellite_links.py` — ISL topology, link geometry
- `lifetime.py` — Orbit lifetime, decay profile
- `linalg.py` — Linear algebra (Jacobi eigensolver, DFT)
- `link_budget.py` — RF link budget, SNR, data rate
- `maintenance_planning.py` — Maintenance and scheduling
- `maneuver_detection.py` — EKF innovation-based maneuver detection (CUSUM, EWMA, chi-squared)
- `maneuvers.py` — Hohmann, bi-elliptic, plane change, phasing
- `mission_analysis.py` — Cross-domain mission composition
- `mission_economics.py` — Mission economics modeling
- `multi_objective_design.py` — Multi-objective Pareto design
- `nrlmsise00.py` — NRLMSISE-00 atmosphere model with solar activity
- `numerical_propagation.py` — RK4 integrator + pluggable force models
- `orbit_determination.py` — Extended Kalman Filter orbit determination
- `precession_nutation.py` — IAU 2006 precession + IAU 2000B nutation + GCRS↔ITRS
- `operational_prediction.py` — EOL prediction, maneuver feasibility
- `orbit_design.py` — SSO/LTAN, frozen orbit, repeat ground track
- `orbit_properties.py` — Derived properties (velocity, energy, RSW, LTAN)
- `pass_analysis.py` — Doppler, visual magnitude, contact statistics
- `planetary_ephemeris.py` — Chebyshev interpolation for Sun/Moon positions
- `radiation.py` — Radiation environment (L-shell, SAA)
- `relative_motion.py` — CW/Hill relative motion equations
- `revisit.py` — Time-domain revisit analysis
- `sensor.py` — Sensor/payload FOV modeling
- `solar.py` — Analytical solar ephemeris
- `sp3_parser.py` — SP3 precise ephemeris parser
- `spectral_topology.py` — Spectral topology analysis
- `station_keeping.py` — Delta-V budgets, Tsiolkovsky, propellant lifetime
- `statistical_analysis.py` — Survival curves, availability, correlations
- `temporal_correlation.py` — Cross-spectral coherence
- `third_body.py` — Solar/lunar third-body perturbations
- `time_systems.py` — AstroTime value object, UTC/TAI/TT/TDB/GPS conversions
- `torques.py` — Gravity gradient + aerodynamic torques
- `relativistic_forces.py` — Schwarzschild, Lense-Thirring, de Sitter corrections
- `tidal_forces.py` — Solid Earth tides (IERS 2010) + FES2004 ocean tides
- `albedo_srp.py` — Earth albedo + infrared radiation pressure
- `trade_study.py` — Parametric Walker trade studies, Pareto front

**Adapters** (`src/humeris/adapters/`):

- `cesium_viewer.py` — Self-contained HTML viewer with layer selector
- `czml_exporter.py` — CZML packets for CesiumJS visualization
- `czml_visualization.py` — Advanced CZML (ISL, fragility, hazard, coverage)
- `viewer_server.py` — Interactive viewer server with 13 analysis types

**Tests** (`tests/`):

- `test_adaptive_integration.py`, `test_atmosphere.py`, `test_cascade_analysis.py`, `test_cesium_viewer.py`,
  `test_communication_analysis.py`, `test_conjunction.py`,
  `test_conjunction_management.py`, `test_constellation_metrics.py`,
  `test_hazard_reporting.py`, `test_kessler_heatmap.py`,
  `test_maneuver_detection.py`, `test_orbit_determination.py`,
  `test_constellation_operability.py`, `test_control_analysis.py`,
  `test_coverage_optimization.py`, `test_czml_exporter.py`,
  `test_czml_visualization.py`, `test_decay_analysis.py`, `test_deorbit.py`,
  `test_design_optimization.py`, `test_design_sensitivity.py`,
  `test_dilution_of_precision.py`, `test_eclipse.py`,
  `test_environment_analysis.py`, `test_graph_analysis.py`,
  `test_gravity_field.py`, `test_information_theory.py`,
  `test_inter_satellite_links.py`,
  `test_invariants_conjunction.py`, `test_invariants_exporters.py`,
  `test_invariants_frames.py`, `test_invariants_geodetic.py`,
  `test_invariants_j2_sso.py`, `test_invariants_two_body.py`,
  `test_lifetime.py`, `test_linalg.py`, `test_link_budget.py`,
  `test_maintenance_planning.py`, `test_maneuvers.py`,
  `test_mission_analysis.py`, `test_mission_economics.py`,
  `test_multi_objective_design.py`, `test_nrlmsise00.py`, `test_numerical_propagation.py`,
  `test_operational_prediction.py`, `test_solar_aware_eol.py`, `test_orbit_design.py`,
  `test_earth_orientation.py`,
  `test_precession_nutation.py`, `test_time_systems.py`,
  `test_planetary_ephemeris.py`,
  `test_orbit_properties.py`, `test_pass_analysis.py`, `test_radiation.py`,
  `test_relative_motion.py`, `test_revisit.py`, `test_sensor.py`,
  `test_solar.py`, `test_spectral_topology.py`, `test_station_keeping.py`,
  `test_statistical_analysis.py`, `test_temporal_correlation.py`,
  `test_relativistic_forces.py`, `test_tidal_forces.py`, `test_albedo_srp.py`,
  `test_third_body.py`, `test_torques.py`, `test_trade_study.py`,
  `test_validation_vallado.py`, `test_validation_sgp4.py`,
  `test_validation_crosscheck.py`, `test_validation_sp3.py`,
  `test_viewer_server.py`

Copyright (c) 2026 Jeroen Visser. All rights reserved.

## 2. Free use

You may use, copy, and modify these files **at no cost** for:

- Personal projects
- Academic research
- Education and learning
- Evaluation and testing
- Any non-commercial purpose

For the avoidance of doubt, use at a publicly funded academic institution for
research purposes is not commercial use, regardless of funding source.

## 3. Commercial use

Commercial use means using the Software in any activity intended to generate
revenue. This includes, but is not limited to:

- Incorporating the Software into a product or service sold or offered for a fee
- Using the Software to provide paid consulting, analysis, or engineering
  services
- Using the Software as an internal tool at any for-profit entity

This commercial license is intended for business use. If you are a consumer (a
natural person acting outside your trade, business, or profession), you may use
the Software free of charge under the non-commercial terms in Section 2.

## 4. License grant

Subject to the terms of this agreement and payment of the applicable license
fee, the Licensor grants the Licensee a **perpetual, worldwide, non-exclusive,
non-transferable** (except as provided in Section 11) license to use, copy,
modify, and create derivative works of the covered files for the Licensee's
own commercial purposes.

## 5. Pricing

| Tier | Eligibility | One-time license fee |
|------|-------------|---------------------|
| **Startup** | < 50 employees | **EUR 2,000** |
| **Enterprise** | ≥ 50 employees or > EUR 10M consolidated annual revenue | **EUR 7,500** |
| **Government** | Government agencies, public-sector bodies, or entities deriving > 25% of annual revenue from government contracts | **EUR 15,000** |

Employee count and revenue are measured at the purchasing legal entity and its
affiliates, per the most recent audited financial statements or annual accounts
at the time of purchase.

The applicable tier is **locked at the date of purchase**. If the Licensee later
grows beyond their original tier's eligibility criteria, the existing license
remains valid. The Licensee is not required to pay the difference. Future
purchases of additional modules are priced at the then-applicable tier.

## 6. Optional: Support & Updates subscription

**EUR 1,500/year** (all tiers). Includes:

- Priority bug fixes
- Access to future commercial modules as they are released
- Direct communication channel for bug reports and technical usage questions

The subscription requires a prior commercial license purchase. Modules received
during an active subscription period are **perpetually licensed** under the same
terms as the original purchase. If the subscription lapses, the Licensee retains
full use rights to all modules received while the subscription was active, but
does not receive modules released after lapse.

The subscription is optional. The perpetual license stands on its own.

## 7. Warranty disclaimer

THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, WHETHER EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE LICENSOR MAKES NO WARRANTY
THAT THE SOFTWARE WILL BE ERROR-FREE, UNINTERRUPTED, OR FIT FOR ANY PARTICULAR
APPLICATION INCLUDING SAFETY-CRITICAL OR MISSION-CRITICAL SYSTEMS.

THE LICENSEE IS SOLELY RESPONSIBLE FOR DETERMINING THE SUITABILITY OF THE
SOFTWARE FOR ITS INTENDED USE AND FOR ANY RESULTS OBTAINED FROM THE SOFTWARE.

## 8. Limitation of liability

IN NO EVENT SHALL THE LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL,
CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING BUT NOT LIMITED TO LOSS OF
PROFITS, DATA, BUSINESS OPPORTUNITY, OR GOODWILL, REGARDLESS OF THE CAUSE OF
ACTION OR THE THEORY OF LIABILITY.

THE LICENSOR'S TOTAL AGGREGATE LIABILITY UNDER OR IN CONNECTION WITH THIS
AGREEMENT SHALL NOT EXCEED THE LICENSE FEE ACTUALLY PAID BY THE LICENSEE.

Nothing in this agreement excludes or limits liability for intent (*opzet*) or
gross negligence (*bewuste roekeloosheid*) to the extent such exclusion is
prohibited under Dutch law.

## 9. Termination

This license terminates automatically if the Licensee materially breaches any
term of this agreement and fails to cure such breach within 30 days of written
notice from the Licensor.

Upon termination, the Licensee must cease all commercial use of the covered
files and destroy all copies. Sections 7 (Warranty disclaimer), 8 (Limitation
of liability), 10 (Governing law), and 13 (Export control) survive termination.

Termination does not affect the Licensee's right to use the MIT-licensed
portions of the project under the MIT License.

## 10. Governing law and jurisdiction

This agreement is governed by and construed in accordance with the laws of the
Netherlands.

Any dispute arising from or in connection with this agreement shall be submitted
to the exclusive jurisdiction of the competent court in The Hague
(*Rechtbank Den Haag*), the Netherlands.

## 11. Assignment and transfer

**Licensor**: The copyright and licensing authority may be assigned to a
successor entity (such as a company founded by the copyright holder). Any such
assignment will be communicated to existing licensees. License terms remain
unchanged.

**Licensee**: The Licensee may not assign or transfer this license without prior
written consent of the Licensor, except in connection with a merger,
acquisition, or sale of substantially all of the Licensee's assets, provided
the successor entity (a) agrees to be bound by the terms of this agreement and
(b) meets the eligibility criteria for the applicable tier or pays the
difference.

## 12. Verification

The Licensor may request reasonable evidence of tier eligibility (such as annual
accounts or a statement of employee count). The Licensee shall respond within
30 days. Failure to respond constitutes a material breach under Section 9.

## 13. Export control and sanctions

The Licensee represents and warrants that:

- It is not subject to sanctions or export restrictions imposed by the European
  Union, the Netherlands, or the United Nations.
- It shall not re-export or transfer the Software in violation of applicable
  export control laws, including EU Regulation 2021/821 (Dual-Use Regulation).

The Licensor reserves the right to verify the Licensee's identity against the
EU consolidated sanctions list before issuing a license.

## 14. How to purchase

Email **planet.jeroen@gmail.com** with:

- Your company name, registered address, and applicable tier
- Intended use (brief description)
- For Government tier: nature of the contracting entity

You will receive an invoice and a signed license agreement. Payment via bank
transfer. License is effective upon payment.

This document summarizes commercial terms. A binding license agreement is
executed separately upon purchase. Until such agreement is executed, no
commercial use rights are granted.

## 15. Everything else

All other files in this repository are MIT-licensed. See `LICENSE`.
