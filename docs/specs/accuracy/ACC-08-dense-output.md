# ACC-08: Dense Output Improvements

**Contract**: ACC-08-DENSE-OUTPUT
**Stage**: 1 (no dependencies)
**Effort**: Medium (1-2 sessions)
**Audit items**: #9, #10

## Problem

### DP45 cubic Hermite instead of continuous extension (#9)
`_hermite_interpolate` in `adaptive_integration.py:514-542` uses cubic Hermite
interpolation (4 conditions: endpoint values + derivatives). This gives O(h^4)
accuracy for a 5th-order method. The Dormand-Prince pair has a natural 4th-order
continuous extension that uses the 7 stage values k1..k7 already computed in
`_dp_full_step`. This would give O(h^5) dense output, matching the method order.

At output_step=60s with h=200s, cubic Hermite error is ~1-10 m vs ~0.01-0.1 m
for the DP continuous extension.

### RK89 cubic Hermite for 8th-order method (#10)
`propagate_rk89_adaptive` uses the same cubic Hermite interpolation between an
8th-order solution and a freshly computed derivative. This is O(h^4) interpolation
for an O(h^9) integration method — a 5-order accuracy mismatch. For h=600s
(typical RK89 step), the interpolation error can be 100x larger than the
integration error.

Verner RK8(9) does not have a published natural continuous extension. Options:
1. Accept the O(h^4) limitation and document it clearly
2. Use a higher-degree polynomial interpolant from multiple step endpoints
3. Subdivide large steps for dense output

## Requirements

### DP45 Continuous Extension

**R1**: `propagate_adaptive` SHALL use the Dormand-Prince 4th-order continuous
extension for dense output instead of cubic Hermite interpolation.

**R2**: The continuous extension SHALL use the 7 stage values k1..k7 from
`_dp_full_step` to construct a 4th-order polynomial in theta (the fraction
of the step).

**R3**: The coefficients SHALL match the published Dormand-Prince continuous
extension (Hairer, Norsett, Wanner, "Solving ODEs I", Section II.6, or
Dormand & Prince 1986).

**R4**: When `output_step_s` is None (natural output), behavior is unchanged.

### RK89 Dense Output

**R5**: `propagate_rk89_adaptive` SHALL document the O(h^4) dense output
limitation in its docstring.

**R6**: `propagate_rk89_adaptive` SHALL accept an optional
`max_dense_output_step_s: float | None` parameter. When set, integration steps
larger than this value are subdivided for dense output purposes (the integrator
still takes the large step, but output interpolation uses shorter sub-intervals).

**R7**: Default `max_dense_output_step_s` SHALL be None (current behavior for
backward compatibility).

## Test Specifications

### DP45 Continuous Extension

**T1**: `test_dp_continuous_extension_exists` — A function
`_dp_continuous_extension(t0, y0, t1, y1, k_stages, theta)` exists and returns
a state vector.

**T2**: `test_dp_continuous_extension_endpoints` — At theta=0 returns y0;
at theta=1 returns y1 (within machine epsilon).

**T3**: `test_dp_continuous_extension_midpoint_accuracy` — For a two-body orbit
with known analytical solution, the continuous extension at the midpoint of a
200s step is within 0.1 m of the analytical position. (Current cubic Hermite
would be ~1-10 m off.)

**T4**: `test_dp_dense_output_vs_hermite` — The continuous extension produces
a different (more accurate) result than cubic Hermite at the same interpolation
point.

**T5**: `test_dp_propagate_adaptive_uses_continuous_extension` — After
propagation with output_step_s, the output points match continuous extension
accuracy (not cubic Hermite accuracy).

### RK89 Dense Output

**T6**: `test_rk89_docstring_mentions_dense_output_limitation` — Docstring
of `propagate_rk89_adaptive` contains "O(h^4)" or "cubic Hermite" or
"3rd-order" interpolation mention.

**T7**: `test_rk89_max_dense_output_step` — With `max_dense_output_step_s=60`,
the output points at 30s intervals have better accuracy than with the default
(which uses full step size for interpolation).

**T8**: `test_rk89_default_unchanged` — Without `max_dense_output_step_s`,
behavior is identical to current.

## Implementation Notes

### DP45 Continuous Extension Coefficients
The 4th-order continuous extension for Dormand-Prince uses 7 bi coefficients:
```
u(theta) = y_n + h * sum(bi(theta) * k_i, i=1..7)
```
where bi(theta) are polynomials in theta. The coefficients are documented in:
- Dormand, J.R.; Prince, P.J. (1986). "Runge-Kutta triples".
  Computers & Mathematics with Applications. 12 (9): 1007-1017.
- Hairer, E.; Norsett, S.P.; Wanner, G. (1993). "Solving ODEs I". Springer.
  Section II.6, pp. 191-193.

### Storing k_stages in the DP propagation loop
Currently `_dp_full_step` returns `(t_new, y_new, k7)`. For the continuous
extension, it needs to return all 7 k-values. Either:
- Return `(k_stages, y_new, k7)` (like RK89 already does)
- Store k_stages in the propagation loop alongside prev_t, prev_state

### RK89 step subdivision for dense output
When `max_dense_output_step_s` is set and h > max_dense_output_step_s, instead
of interpolating across the full step, compute derivative at intermediate points
(using `deriv_fn` directly) and use shorter Hermite intervals. This costs extra
function evaluations but improves dense output accuracy.

## Acceptance Criteria

- All T1-T8 pass GREEN
- No regression in existing tests
- DP45 dense output accuracy improved from O(h^4) to O(h^5)
- RK89 dense output limitation documented
- RK89 optional step subdivision available
