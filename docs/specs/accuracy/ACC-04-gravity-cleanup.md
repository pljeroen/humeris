# ACC-04: Gravity Model Cleanup

**Contract**: ACC-04-GRAVITY-CLEANUP
**Stage**: 1 (no dependencies)
**Effort**: Low (1 session)
**Audit items**: #2, #14

## Problem

### SphericalHarmonicGravity degree-8 cap (#2)
`SphericalHarmonicGravity` in `numerical_propagation.py:244-395` is a second,
independent gravity implementation capped at degree/order 8. The correct
implementation `CunninghamGravity` in `gravity_field.py` supports degrees up
to 70 with bundled EGM96 coefficients. Code that uses `SphericalHarmonicGravity`
silently truncates to degree 8, losing 10-100 m/day accuracy at LEO.

### Central term double-counting risk (#14)
`SphericalHarmonicGravity.acceleration()` at line 337 computes `ar = -mu / (r * r)`
as the central term, then adds harmonic corrections. If a user combines
`TwoBodyGravity() + SphericalHarmonicGravity()`, the central force is doubled
(~9 m/s^2 error — catastrophic). `CunninghamGravity` correctly includes the
central term, with documentation. The risk is a latent user error.

## Requirements

**R1**: `SphericalHarmonicGravity` SHALL be marked deprecated with a clear
docstring directing users to `CunninghamGravity`.

**R2**: `SphericalHarmonicGravity.__init__` SHALL emit a `DeprecationWarning`
stating: "Use CunninghamGravity for production work. SphericalHarmonicGravity
is limited to degree 8 and will be removed in a future version."

**R3**: The docstring of `SphericalHarmonicGravity` SHALL explicitly state
that it includes the central term and MUST NOT be combined with `TwoBodyGravity`.

**R4**: `CunninghamGravity` docstring SHALL explicitly state that it includes
the central term and MUST NOT be combined with `TwoBodyGravity`.

**R5**: All stress mirror scenarios that currently use `SphericalHarmonicGravity(max_degree=8)`
SHALL be evaluated for migration to `CunninghamGravity`. Those that can migrate
without changing test semantics SHALL be migrated. (The GMAT mirror scenarios
already use `CunninghamGravity` for degree > 8.)

## Test Specifications

**T1**: `test_spherical_harmonic_deprecation_warning` — Instantiating
`SphericalHarmonicGravity()` emits `DeprecationWarning`.

**T2**: `test_spherical_harmonic_docstring_central_term` — Docstring contains
"central term" and "TwoBodyGravity" warning text.

**T3**: `test_cunningham_docstring_central_term` — Same for `CunninghamGravity`.

**T4**: `test_spherical_harmonic_still_works` — Existing functionality unchanged
(backward compat). Acceleration values match previous behavior.

**T5**: `test_double_counting_detection` — Document: if both `TwoBodyGravity` and
`SphericalHarmonicGravity` are in force list, total acceleration magnitude is
approximately 2x expected (demonstrates the error for documentation/education).

## Implementation Notes

- Add `warnings.warn(...)` in `SphericalHarmonicGravity.__init__`
- Update docstrings for both classes
- Do NOT remove `SphericalHarmonicGravity` — many tests use it as a lightweight
  gravity model. Removal is a future breaking change.
- Stress mirror scenarios S2 (VLEO drag) already use `SphericalHarmonicGravity(8)`
  and could be migrated to `CunninghamGravity(load_gravity_field(8))` but the
  test expectations may need adjustment due to coefficient differences.

## Acceptance Criteria

- All T1-T5 pass GREEN
- No regression in existing tests
- DeprecationWarning emitted on construction
- Docstrings clearly document central-term inclusion
