# ACC-07: SRP and Albedo Fidelity

**Contract**: ACC-07-SRP-FIDELITY
**Stage**: 2 (depends on ACC-01 for Chebyshev ephemeris)
**Effort**: Medium (1-2 sessions)
**Audit items**: #7, #17

## Problem

### SRP penumbra model (#7)
`SolarRadiationPressureForce` in `numerical_propagation.py:442-497` uses a
binary shadow function: umbra=0, penumbra=0.5 (hardcoded), full sun=1.0. The
correct penumbra fraction is a smooth function of satellite position within
the penumbra cone, varying continuously from 0 to 1 across ~20-30 km.

The cannonball SRP model itself is acceptable for many applications but should
be documented as such.

### Albedo sector integration (#17)
`AlbedoRadiationPressure` in `albedo_srp.py:89-161` uses a single isotropic
albedo (0.30) and uniform IR (237 W/m^2) across the entire visible Earth disk.
The constructor accepts `n_sectors` but this parameter is stored and never used.
The acceleration is purely radial (no off-axis component from albedo asymmetry).

## Requirements

### SRP Penumbra

**R1**: `SolarRadiationPressureForce` SHALL compute the correct penumbra shadow
fraction using the apparent disk overlap method: the fraction of the solar disk
visible from the satellite, considering the apparent angular radii of the Sun
and Earth as seen from the satellite.

**R2**: The shadow function SHALL return:
- 1.0 when no part of the Sun is blocked by Earth
- 0.0 when the Sun is fully blocked (umbra)
- A value in (0, 1) that is the fraction of the solar disk area visible (penumbra)

**R3**: The implementation SHALL use the standard cylindrical shadow model with
apparent radius comparison (Montenbruck & Gill, Section 3.4).

### Albedo Sector Integration

**R4**: `AlbedoRadiationPressure` SHALL implement sector-based integration of
albedo and IR radiation using `n_sectors` to divide the visible Earth disk
into angular sectors.

**R5**: Each sector SHALL have its own view factor based on geometry (distance,
angle from sub-satellite point). The albedo acceleration SHALL include off-axis
components (not purely radial).

**R6**: With `n_sectors=1`, behavior SHALL match the current isotropic model
(backward compatible).

## Test Specifications

### SRP Penumbra

**T1**: `test_penumbra_full_sun` — Satellite well above the Earth-Sun line,
shadow fraction = 1.0.

**T2**: `test_penumbra_umbra` — Satellite directly behind Earth (umbra),
shadow fraction = 0.0.

**T3**: `test_penumbra_partial` — Satellite at the penumbra boundary, shadow
fraction between 0 and 1 (not hardcoded 0.5).

**T4**: `test_penumbra_smooth_transition` — Shadow fraction varies smoothly
as satellite moves across the penumbra region. Verify monotonicity.

**T5**: `test_penumbra_vs_hardcoded` — In the penumbra region, new shadow
fraction differs from 0.5 (proves the hardcode is replaced).

### Albedo Sector Integration

**T6**: `test_albedo_n_sectors_1_backward_compat` — With `n_sectors=1`,
acceleration matches current isotropic model.

**T7**: `test_albedo_n_sectors_6_off_axis` — With `n_sectors=6`, the
acceleration has a non-zero cross-track component (not purely radial).

**T8**: `test_albedo_magnitude_reasonable` — Total albedo acceleration
magnitude is within 20% of the isotropic estimate (sector integration
shouldn't drastically change the total, just redistribute it).

**T9**: `test_albedo_subsolar_asymmetry` — When the sub-solar point is
offset from the sub-satellite point, the albedo acceleration direction
should tilt toward the brighter (sun-facing) sectors.

## Implementation Notes

### Penumbra shadow function
```python
def _shadow_fraction(r_sat, r_sun, r_earth_radius):
    """Compute fraction of solar disk visible from satellite."""
    # Apparent angular radii
    alpha_sun = math.asin(R_SUN / d_sun)  # ~0.267 deg
    alpha_earth = math.asin(R_EARTH / r_sat_magnitude)
    # Angular separation between Sun and Earth centers as seen from satellite
    cos_theta = dot(r_sat_to_sun, r_sat_to_earth_center) / (...)
    theta = math.acos(cos_theta)
    # Overlap area of two disks
    if theta >= alpha_sun + alpha_earth:
        return 1.0  # no overlap
    if theta + alpha_sun <= alpha_earth:
        return 0.0  # total eclipse (umbra)
    # Partial overlap: compute area ratio
    ...
```

The disk overlap formula uses the intersection area of two circles on the
unit sphere. See Montenbruck & Gill (2000) Eq. 3.79-3.82.

### Sector integration
Divide the visible Earth disk into `n_sectors` angular sectors from the
sub-satellite point. For each sector, compute:
- View factor (solid angle fraction)
- Reflected solar flux (albedo * solar_flux * cos(solar_zenith))
- Emitted IR flux (constant per sector)
- Direction from sector center to satellite
Sum contributions vectorially.

## Acceptance Criteria

- All T1-T9 pass GREEN
- No regression in existing tests
- Penumbra fraction is continuous (not stepped)
- Albedo with n_sectors > 1 has off-axis components
- n_sectors=1 backward compatible
