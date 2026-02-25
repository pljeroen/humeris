# ACC-03: Frame Rotation Fix

**Contract**: ACC-03-FRAME-ROTATION
**Stage**: 2 (depends on ACC-01, ACC-02)
**Effort**: Medium (1-2 sessions)
**Audit items**: #3, #8, #13

## Problem

### GMST uses UTC, not UT1 (#13)
`gmst_rad` in `coordinate_frames.py:25-58` computes GMST from raw UTC datetime.
The IAU formula requires UT1 for the diurnal rotation and TT for the secular
terms. At current UTC-UT1 difference (~0.1-0.9 s), this introduces up to ~680 m
cross-track position error in ECEF coordinates.

### Polar motion not applied (#8)
`gcrs_to_itrs_matrix` in `precession_nutation.py:506-551` applies frame bias,
precession, nutation, and ERA but does NOT apply the polar motion matrix W.
The function `polar_motion_matrix(xp, yp, s')` exists in `earth_orientation.py`
but is never called. Polar motion amplitude is ~0.3 arcsec = ~9 m at Earth surface.

### Force models use GMST-only rotation (#3)
Every force model needing ECEF coordinates (`CunninghamGravity`, `SphericalHarmonicGravity`,
`AtmosphericDragForce`, `SolidTideForce`, `OceanTideForce`) uses a simple Z-rotation
by GMST. They do not use the full GCRS->ITRS matrix with precession, nutation,
ERA, and polar motion.

## Requirements

**R1**: `gcrs_to_itrs_matrix` SHALL apply the polar motion matrix W when EOP
data is available: `M = W * R3(ERA) * N * P * B` where `W = R3(-s') * R2(xp) * R1(yp)`.

**R2**: `gcrs_to_itrs_matrix` SHALL accept optional `xp_arcsec` and `yp_arcsec`
parameters (defaulting to 0.0) for polar motion.

**R3**: Force models that need body-fixed coordinates SHALL accept an optional
rotation matrix or EOP parameters. When provided, they use the full GCRS->ITRS
matrix instead of GMST-only rotation.

**R4**: `CunninghamGravity.acceleration()` SHALL use the full GCRS->ITRS matrix
when an `AstroTime` and EOP data are available, falling back to GMST rotation
when only a `datetime` is provided. The `ForceModel` protocol signature is
unchanged (`epoch: datetime`), but the gravity model can be constructed with
EOP data: `CunninghamGravity(field, eop_table=...)`.

**R5**: Same pattern for `SolidTideForce`, `OceanTideForce`, `NRLMSISE00DragForce`,
and `AtmosphericDragForce`.

**R6**: The GMST-only path SHALL remain as default (backward compatible). Full
precision is opt-in via constructor parameters.

## Test Specifications

**T1**: `test_gcrs_to_itrs_with_polar_motion` — At a known epoch with known EOP
values (xp=0.1 arcsec, yp=0.2 arcsec), the GCRS->ITRS matrix differs from the
no-polar-motion matrix by a rotation consistent with ~0.1-0.2 arcsec pole offset.

**T2**: `test_gcrs_to_itrs_default_no_polar_motion` — Without EOP params, result
is identical to current behavior (backward compat).

**T3**: `test_cunningham_gravity_with_eop` — `CunninghamGravity` constructed with
`eop_table=...` produces a different (more accurate) acceleration than without.
The difference should be consistent with ~9 m pole offset effect on gravity.

**T4**: `test_cunningham_gravity_default_backward_compat` — Without `eop_table`,
acceleration is identical to current behavior.

**T5**: `test_solid_tide_with_eop` — `SolidTideForce` with EOP produces tidal
acceleration in a different direction than without (consistent with frame correction).

**T6**: `test_frame_rotation_accuracy` — At 5 test epochs, compare GMST-only
vs full GCRS->ITRS rotation. Difference should be consistent with known
precession + nutation + polar motion magnitudes.

**T7**: `test_era_vs_gmst` — Earth Rotation Angle (from UT1) vs GMST (from UTC)
differ by the expected UTC-UT1 offset converted to angle.

## Implementation Notes

### Phase 1: Fix `gcrs_to_itrs_matrix`
- Add `xp_arcsec=0.0`, `yp_arcsec=0.0` parameters
- When non-zero, compute W via existing `polar_motion_matrix()` and apply
- Keep default behavior unchanged

### Phase 2: Add EOP-aware constructors to force models
- `CunninghamGravity.__init__` adds optional `eop_table: EOPTable | None = None`
- In `acceleration()`, if `eop_table` is set:
  1. Convert `epoch` to `AstroTime`
  2. Interpolate EOP at epoch
  3. Compute full GCRS->ITRS matrix
  4. Use instead of GMST rotation
- Same pattern for tidal forces and drag

### GMST fix
The existing `gmst_rad()` in `coordinate_frames.py` is not modified (used by
non-precision paths). Force models that opt in to EOP use ERA from
`precession_nutation.earth_rotation_angle()` instead.

## Acceptance Criteria

- All T1-T7 pass GREEN
- No regression in existing tests
- Default (no EOP) behavior identical to current
- With EOP, frame accuracy improves from ~680 m to ~1 m
