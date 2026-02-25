# ACC-11: Coverage and Remaining Items

**Contract**: ACC-11-COVERAGE-MISC
**Stage**: 1 (no dependencies)
**Effort**: Low (1 session)
**Audit items**: #18, #23

## Problem

### Coverage: spherical Earth (#23)
`coverage.py` uses spherical Earth for coverage calculations. At high latitudes,
the WGS84 ellipsoid differs from a sphere by up to 21 km in Earth radius. This
shifts elevation angle zero-crossing by ~0.3 degrees and introduces ~few km
error in coverage boundary position.

### Radiation: dipole L-shell (#18)
`radiation.py` uses the McIlwain L-shell from a dipole approximation. In the
South Atlantic Anomaly (SAA), the inner belt dips to ~200 km altitude due to
the eccentric dipole. The pure dipole model has ~20-30% error in dose within
the SAA. NOTE: This affects dose estimates only, not trajectory accuracy.

## Requirements

### Coverage WGS84

**R1**: `compute_coverage_snapshot` SHALL use the WGS84 Earth radius at the
observer latitude for elevation angle computation, not the mean Earth radius.

**R2**: The WGS84 local radius SHALL be computed as:
```
R(lat) = sqrt((a^2*cos(lat))^2 + (b^2*sin(lat))^2) / sqrt((a*cos(lat))^2 + (b*sin(lat))^2)
```
where a = 6378137 m (equatorial), b = 6356752.3142 m (polar).

**R3**: When `lat_range` includes high latitudes (> 70 deg), the ellipsoid
correction SHALL produce measurably different coverage boundaries.

### Radiation Dipole Note

**R4**: `compute_l_shell` docstring SHALL explicitly state that it uses the
centered dipole approximation and note the ~20-30% SAA dose error.

**R5**: No implementation change to the L-shell computation (eccentric dipole
or IGRF would require magnetic field coefficients, which is a large effort
disproportionate to the dose-only impact).

## Test Specifications

### Coverage WGS84

**T1**: `test_coverage_wgs84_vs_spherical` — At 80 deg latitude with a LEO
satellite, the coverage boundary (elevation = 0 deg) differs between WGS84
and spherical by at least 0.1 degrees in elevation angle.

**T2**: `test_coverage_equator_unchanged` — At 0 deg latitude, WGS84 and
spherical produce nearly identical results (< 0.01 deg difference).

**T3**: `test_coverage_wgs84_radius_at_pole` — The WGS84 local radius at 90
deg latitude equals the polar radius (6356752.3142 m), not the mean radius.

**T4**: `test_coverage_wgs84_radius_at_equator` — The WGS84 local radius at
0 deg latitude equals the equatorial radius (6378137 m).

### Radiation Docstring

**T5**: `test_l_shell_docstring_dipole` — Docstring of `compute_l_shell`
contains "dipole" and mentions SAA limitation.

## Implementation Notes

### Coverage WGS84
The key change is in the elevation angle computation. Currently it likely uses:
```python
R_earth = OrbitalConstants.R_EARTH  # mean radius
```
Replace with:
```python
R_earth = wgs84_local_radius(observer_lat_rad)
```

The `wgs84_local_radius` function:
```python
def _wgs84_local_radius(lat_rad: float) -> float:
    a = OrbitalConstants.R_EARTH_EQUATORIAL  # 6378137.0
    b = OrbitalConstants.R_EARTH_POLAR       # 6356752.3142
    cos_lat = math.cos(lat_rad)
    sin_lat = math.sin(lat_rad)
    num = math.sqrt((a**2 * cos_lat)**2 + (b**2 * sin_lat)**2)
    den = math.sqrt((a * cos_lat)**2 + (b * sin_lat)**2)
    return num / den
```

### Radiation docstring
Simple docstring update to `compute_l_shell` in `radiation.py`.

## Acceptance Criteria

- All T1-T5 pass GREEN
- No regression in existing tests
- High-latitude coverage boundaries shift by measurable amount
- Equatorial behavior unchanged
- Radiation docstring updated
