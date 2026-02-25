# ACC-06: Tidal Force Fidelity

**Contract**: ACC-06-TIDAL-FIDELITY
**Stage**: 2 (depends on ACC-01 for Chebyshev ephemeris)
**Effort**: Low-Medium (1 session)
**Audit items**: #4

## Problem

`SolidTideForce` in `tidal_forces.py:216-263` uses a single scalar `k2`
(defaulting to `_K20 = 0.30190`) for all spherical harmonic orders. The correct
IERS 2010 model uses order-dependent Love numbers:
- k20 = 0.30190 (zonal)
- k21 = 0.29830 (tesseral)
- k22 = 0.30102 (sectoral)

These constants ARE defined in the file (lines 30-32) but only `_K20` is used
as the default. The acceleration loop applies the same k2 for all m=0,1,2 terms.

Additionally missing:
- Degree-3 solid tides (k30, k31, k32, k33): ~1e-10 m/s^2 magnitude
- Frequency-dependent corrections (IERS 2010 Table 6.5a): 5 dominant terms,
  up to 2e-11 m/s^2 correction to specific tidal frequencies

Combined position error: ~0.1-1 m/day for precision orbit determination.

## Requirements

**R1**: `SolidTideForce` SHALL apply order-dependent Love numbers: k20 for m=0,
k21 for m=1, k22 for m=2 terms in the degree-2 tidal acceleration.

**R2**: `SolidTideForce` SHALL accept these as constructor parameters with
IERS 2010 defaults:
```python
k20: float = 0.30190
k21: float = 0.29830
k22: float = 0.30102
```

**R3**: `SolidTideForce` SHALL compute degree-3 solid tidal acceleration using
Love numbers k30=0.093, k31=0.093, k32=0.093, k33=0.094 (IERS 2010 Table 6.3).

**R4**: `SolidTideForce` SHALL optionally apply the 5 dominant frequency-dependent
corrections from IERS 2010 Table 6.5a when `include_frequency_dependent=True`
(default False for backward compatibility).

**R5**: Backward compatibility: `SolidTideForce()` with no arguments SHALL
produce results very close to current (the k20/k21/k22 differences are ~0.3%,
so regression thresholds may need minor adjustment).

## Test Specifications

**T1**: `test_solid_tide_order_dependent_k` — At a known epoch, SolidTideForce
with k20=0.30190, k21=0.29830, k22=0.30102 produces a measurably different
acceleration than with uniform k2=0.30190. The difference magnitude should be
consistent with ~0.3% of the tidal acceleration.

**T2**: `test_solid_tide_degree3` — With degree-3 enabled, the total tidal
acceleration includes a degree-3 contribution. Verify by comparing with/without
degree-3: difference should be ~1e-10 m/s^2 magnitude.

**T3**: `test_solid_tide_degree3_direction` — The degree-3 contribution depends
on r^-4 (not r^-3 like degree-2), so it should be proportionally larger at
lower altitudes.

**T4**: `test_frequency_dependent_corrections` — With `include_frequency_dependent=True`,
the acceleration differs from the uncorrected case by ~2e-11 m/s^2 at specific
tidal-frequency epochs.

**T5**: `test_solid_tide_backward_compat` — `SolidTideForce()` with no arguments
produces acceleration within 0.5% of the pre-change value at a reference epoch.

**T6**: `test_love_number_constants` — k20, k21, k22 values match IERS 2010
Table 6.3 within machine precision.

## Implementation Notes

### Order-dependent k2
In the acceleration loop (line 241-263), replace the single `self.k2` with
`k_values = (self.k20, self.k21, self.k22)` indexed by `m`.

### Degree-3 tides
Add a second summation loop for n=3, m=0..3 with the degree-3 formula:
```
a_3m = k3m * (GM_body/GM_earth) * (R_earth/d)^4 * (R_earth/r)^4 * P3m(sin_lat) * ...
```
The Legendre functions P30, P31, P32, P33 are needed.

### Frequency-dependent corrections
The 5 dominant terms from IERS 2010 Table 6.5a are corrections to k20 at
specific tidal frequencies (Ssa, Mm, Mf, etc.). They can be encoded as a
tuple of (frequency, amplitude, phase) and applied when the option is enabled.

## Acceptance Criteria

- All T1-T6 pass GREEN
- No regression in existing tests (tolerance may need minor adjustment)
- Order-dependent Love numbers applied by default
- Degree-3 available via constructor flag
- Frequency-dependent corrections available via constructor flag
