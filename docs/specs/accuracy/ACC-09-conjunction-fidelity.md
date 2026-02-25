# ACC-09: Conjunction Analysis Fidelity

**Contract**: ACC-09-CONJUNCTION-FIDELITY
**Stage**: 1 (no dependencies)
**Effort**: Medium (1-2 sessions)
**Audit items**: #11, #19, #24

## Problem

### Isotropic covariance in Pc computation (#11)
`assess_conjunction` in `conjunction.py:379-385` computes collision probability
using `sigma_combined = sqrt(combined_trace / 3.0)` — a scalar derived from the
covariance trace. This makes the bivariate normal isotropic, discarding the full
covariance structure. Real encounter covariances are highly elongated along-track
(sigma_along >> sigma_radial, ratio 10:1 to 100:1). The isotropic approximation
can be wrong by 2-3 orders of magnitude in Pc.

### Approximate analytical Pc formula (#19)
`compute_analytical_collision_probability` in `statistical_analysis.py:108-119`
uses a first-order asymptotic expansion valid only when miss distance >> combined
radius. For close approaches (d/r < 5), the formula has 10-50% relative error.
The correct `collision_probability_2d` numerical integral exists in `conjunction.py`.

### Keplerian-only conjunction screening (#24)
`screen_conjunctions` uses J2-only Keplerian `propagate_to()`. For objects at
different altitudes with different drag, position errors grow to 1-10 km per day.
A `screen_conjunctions_numerical` function exists for higher-fidelity screening
but callers must choose it explicitly. This is a documentation/guidance issue.

## Requirements

### Full Covariance Pc

**R1**: `assess_conjunction` SHALL accept full 6x6 position-velocity covariance
matrices (as `PositionCovariance` or a new 6x6 matrix type) and project them
to the B-plane encounter frame at TCA.

**R2**: The B-plane projection SHALL compute the 2x2 projected covariance
in the (radial, cross-track) encounter frame using the relative velocity
direction at TCA.

**R3**: The projected 2x2 covariance SHALL be passed to `collision_probability_2d`
with the correct `sigma_radial` and `sigma_cross` (eigenvalues of the 2x2
projected covariance), not the isotropic trace approximation.

**R4**: When no covariance is provided, the current isotropic approximation
SHALL remain as fallback (backward compatible).

### Statistical Pc Wiring

**R5**: `compute_analytical_collision_probability` SHALL call
`collision_probability_2d` from `conjunction.py` for its `numerical_pc` field
instead of the local asymptotic formula.

**R6**: The `analytical_pc` field SHALL retain the asymptotic formula for
comparison purposes.

**R7**: The `relative_error` field SHALL be computed as
`|analytical_pc - numerical_pc| / numerical_pc` (currently always 0.0).

### Conjunction Screening Documentation

**R8**: `screen_conjunctions` docstring SHALL explicitly state that it uses
J2-only Keplerian propagation and recommend `screen_conjunctions_numerical`
for horizons > 1 day or drag-affected objects.

**R9**: `screen_conjunctions_numerical` docstring SHALL state that it uses
numerical propagation with user-provided force models.

## Test Specifications

### Full Covariance Pc

**T1**: `test_conjunction_full_covariance` — With an elongated covariance
(sigma_along = 1000 m, sigma_radial = 10 m), the Pc from full covariance
projection differs significantly from the isotropic trace approximation.

**T2**: `test_conjunction_isotropic_fallback` — Without covariance, behavior
matches current (backward compat).

**T3**: `test_b_plane_projection` — The 2x2 projected covariance at TCA has
eigenvalues consistent with the input 3D covariance rotated into the encounter
frame.

**T4**: `test_elongated_covariance_pc_lower` — For a typical along-track
dominated covariance with miss distance in the radial direction, full covariance
Pc < isotropic Pc (the isotropic overestimates because it spreads probability
into the wrong dimension).

### Statistical Pc

**T5**: `test_analytical_pc_uses_numerical_integral` — `numerical_pc` field
from `compute_analytical_collision_probability` matches `collision_probability_2d`
output.

**T6**: `test_analytical_vs_numerical_error` — `relative_error` is non-zero
and reflects the actual difference between the asymptotic formula and the
numerical integral.

### Documentation

**T7**: `test_screen_conjunctions_docstring` — Docstring contains "Keplerian"
or "J2-only" and mentions `screen_conjunctions_numerical`.

**T8**: `test_screen_conjunctions_numerical_docstring` — Docstring mentions
"numerical propagation" or "force models".

## Implementation Notes

### B-plane projection
At TCA, compute the relative velocity direction v_rel. The B-plane is
perpendicular to v_rel. Project the combined 3x3 position covariance onto
the B-plane to get a 2x2 covariance. Use eigendecomposition to get sigma_1,
sigma_2 (the B-plane ellipse semi-axes).

```python
# Relative velocity unit vector
v_hat = v_rel / |v_rel|
# B-plane basis vectors (arbitrary orthogonal to v_hat)
e1 = cross(v_hat, [0,0,1]); e1 /= |e1|  # radial-like
e2 = cross(v_hat, e1)                     # cross-track-like
# Project combined 3x3 covariance
P = [[e1^T C e1, e1^T C e2], [e2^T C e1, e2^T C e2]]
# Eigenvalues of P give sigma_1^2, sigma_2^2
```

### PositionCovariance to 3x3 matrix
The existing `PositionCovariance` has sigma_xx, sigma_yy, sigma_zz,
sigma_xy, sigma_xz, sigma_yz. Construct the 3x3 matrix from these.

## Acceptance Criteria

- All T1-T8 pass GREEN
- No regression in existing tests
- Full covariance Pc available when covariance provided
- Isotropic fallback when no covariance
- Statistical Pc relative_error actually computed
- Docstrings updated
