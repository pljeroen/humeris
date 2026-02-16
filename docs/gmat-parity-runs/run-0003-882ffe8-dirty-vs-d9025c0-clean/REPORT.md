# GMAT Mirror Parity Report

- Status: **PASS**
- Timestamp (UTC): `2026-02-16T22:25:36.018555+00:00`
- Humeris git: `882ffe8-dirty` (commit `882ffe8`, dirty=`dirty`)
- GMAT testsuite git: `d9025c0-clean` (commit `d9025c0`, dirty=`clean`)
- GMAT testsuite repo: `https://github.com/pljeroen/testsuite_gmat`
- GMAT run reference: `run-0008-3b5fc7b-clean`

This is a reference-comparison report. It is intended as a learning and validation artifact, not a certification claim.

## Case: `basic_leo_two_body`
- Case status: **PASS**

| Metric | GMAT | Humeris | Abs delta | Tolerance | Pass |
|---|---:|---:|---:|---:|:---:|
| `startSMA` | 7000 | 7000 | 9.094947e-13 | 5 | yes |
| `startECC` | 0.001 | 0.000999999999999 | 1.116728e-16 | 0.001 | yes |
| `endSMA` | 7000 | 7000.00000083 | 8.259276e-07 | 5 | yes |
| `endECC` | 0.00100000000001 | 0.00099999998181 | 1.819702e-11 | 0.001 | yes |
| `elapsedSecs` | 5400.00000031 | 5400 | 3.143214e-07 | 0.001 | yes |
| `conservation_behavior_match` | true | true |  |  | yes |

## Case: `advanced_j2_raan_drift`
- Case status: **PASS**

| Metric | GMAT | Humeris | Abs delta | Tolerance | Pass |
|---|---:|---:|---:|---:|:---:|
| `startRAAN` | 20 | 20 | 0 | 2 | yes |
| `startINC` | 97.8 | 97.8 | 0 | 0.2 | yes |
| `startECC` | 0.001 | 0.001 | 2.649790e-16 | 0.0005 | yes |
| `elapsedDays` | 7 | 7 | 0 | 1.000000e-06 | yes |
| `raanDriftDeg` | 6.80548074605 | 6.84304855745 | 0.0375678114005 | 2 | yes |
| `j2_regime_match` | true | true |  |  | yes |

## Case: `advanced_oumuamua_hyperbolic`
- Case status: **PASS**

| Metric | GMAT | Humeris | Abs delta | Tolerance | Pass |
|---|---:|---:|---:|---:|:---:|
| `start_ecc_gt_1` | true | true |  |  | yes |
| `end_ecc_gt_1` | true | true |  |  | yes |
| `rmag_changes_materially` | true | true |  |  | yes |
| `elapsed_days_120` | true | true |  |  | yes |

