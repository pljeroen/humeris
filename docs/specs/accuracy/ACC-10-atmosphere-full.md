# ACC-10: Full NRLMSISE-00 Implementation

**Contract**: ACC-10-ATMOSPHERE-FULL
**Stage**: 3 (depends on ACC-05 for wiring)
**Effort**: High (multi-session)
**Audit items**: #5, #22

## Problem

### Simplified NRLMSISE-00 parameterization (#5)
`NRLMSISE00Model` in `nrlmsise00.py:302-554` is explicitly documented as a
"simplified parameterization" of NRLMSISE-00. It replaces the actual polynomial
coefficient tables (PT, PD, PS arrays with 150+ terms from Picone 2002) with:
- Linear temperature profile below 120 km
- Single cosine diurnal factor
- Single sinusoidal latitude factor
- Two-parameter F10.7 correction (power law per species)
- Simplified geomagnetic scaling

The full model uses 150+ coefficients per output variable. The simplifications
produce density errors of 20-50% at quiet conditions, up to factors of 2-5
during geomagnetic storms.

### Missing 3-hourly Ap array (#22)
`SpaceWeatherHistory.lookup()` returns `SpaceWeather` with `ap_array=None`.
The full NRLMSISE-00 model uses a 7-element Ap array for time-weighted
geomagnetic activity (Picone 2002 Table 3). Without it, only daily Ap via
`_g0()` is used, losing 30-50% density accuracy during storms.

## Requirements

### Full NRLMSISE-00

**R1**: `NRLMSISE00Model` SHALL implement the complete NRLMSISE-00 evaluation
logic from Picone, M.J., Hedin, A.E., Drob, D.P., and Aikin, A.C. (2002),
"NRLMSISE-00 empirical model of the atmosphere: Statistical comparisons and
scientific issues", J. Geophys. Res., 107(A12), 1468.

**R2**: The implementation SHALL include:
- Full Legendre polynomial expansion in geographic latitude (7 terms)
- Full harmonic expansion in local solar time (diurnal + semidiurnal + terdiurnal)
- Full F10.7 dependence using the solar activity proxy polynomial
- Full Ap dependence using the 7-element weighted array via `_sg0()`
- Species-dependent temperature profiles using the Bates-Walker formulation
- All 7 species: N2, O2, O, He, Ar, H, N
- Correct diffusive equilibrium computation above 120 km
- Correct mixing below 120 km

**R3**: The coefficient arrays (PT, PD, PS, etc.) SHALL be stored as data files
in `humeris/data/` and loaded at model initialization. The total data volume
is approximately 20-50 KB.

**R4**: The existing simplified model SHALL be preserved as
`NRLMSISE00ModelSimplified` (rename) for backward compatibility and for
applications where speed matters more than accuracy.

**R5**: `NRLMSISE00Model` SHALL be the name of the full model (replacing the
current simplified version in the public API).

### 3-Hourly Ap Array

**R6**: `SpaceWeatherHistory.lookup()` SHALL construct the 7-element Ap array
from historical 3-hourly Kp/Ap data when available.

**R7**: The 7-element array SHALL follow Picone 2002 Table 3:
```
ap_array[0] = daily Ap
ap_array[1] = 3-hour Ap for current 3-hour interval
ap_array[2] = 3-hour Ap for 3 hours before
ap_array[3] = 3-hour Ap for 6 hours before
ap_array[4] = 3-hour Ap for 9 hours before
ap_array[5] = weighted average of 8 Ap values, 12-33 hours before
ap_array[6] = weighted average of 8 Ap values, 36-57 hours before
```

**R8**: `space_weather_historical.json` SHALL include 3-hourly Kp values
(converted to Ap via `kp_to_ap()`) for the historical period.

**R9**: When 3-hourly data is not available (e.g., predicted weather), the
array SHALL be filled with the daily Ap value for all elements.

## Test Specifications

### Full NRLMSISE-00

**T1**: `test_nrlmsise00_reference_values` — At 5 standard reference conditions
(from the original NRLMSISE-00 test cases: see nrlmsise-00.c test driver),
total density matches within 5% of the reference implementation output.

**T2**: `test_nrlmsise00_altitude_profile` — Density at 100, 200, 400, 600,
800, 1000 km for quiet conditions (F10.7=150, Ap=4) is within 10% of reference
values.

**T3**: `test_nrlmsise00_solar_activity_response` — Density at 400 km increases
by at least factor 2 between solar min (F10.7=70) and solar max (F10.7=250).

**T4**: `test_nrlmsise00_geomagnetic_response` — With 3-hourly Ap array showing
storm onset (ap_array[1]=200, rest=4), density at 400 km increases significantly
vs quiet conditions.

**T5**: `test_nrlmsise00_diurnal_variation` — Density at 400 km varies by at
least 30% between local noon and local midnight (equatorial, equinox).

**T6**: `test_nrlmsise00_latitude_variation` — Density at 400 km at the poles
differs from equatorial by at least 20% (winter hemisphere has higher density
in thermosphere).

**T7**: `test_nrlmsise00_simplified_still_works` — `NRLMSISE00ModelSimplified`
produces the same output as the current `NRLMSISE00Model` (renamed version).

### 3-Hourly Ap

**T8**: `test_space_weather_lookup_ap_array` — `SpaceWeatherHistory.lookup()`
returns a `SpaceWeather` with non-None `ap_array` of length 7.

**T9**: `test_ap_array_weighting` — The elements ap_array[5] and ap_array[6]
are weighted averages (not simple averages) of the corresponding 3-hourly values.

**T10**: `test_ap_array_fallback` — When 3-hourly data is unavailable, all
elements equal daily Ap.

## Implementation Notes

### NRLMSISE-00 Coefficient Source
The reference implementation is available as:
- C source: `nrlmsise-00.c` by Dominik Brodowski (public domain)
- Fortran source: `nrlmsise00_sub.for` from NRL (public domain)
- Python wrapper: `nrlmsise00` PyPI package (wraps the C code)

The coefficient arrays can be extracted from the C source (they are literal
arrays in the source file). The evaluation logic follows:
1. Compute geographic/geomagnetic parameters
2. Evaluate Legendre polynomials in latitude
3. Evaluate trigonometric series in local solar time
4. Apply F10.7 and Ap modifiers
5. Compute temperature profile (Bates-Walker)
6. Compute number densities from diffusive equilibrium

### Data Volume
- PT (temperature coefficients): ~150 floats
- PD (density coefficients): ~150 floats × 7 species = ~1050 floats
- PS (supplementary): ~50 floats
- Total: ~1250 floats × 8 bytes = ~10 KB as JSON

### 3-Hourly Ap Data
The `space_weather_historical.json` currently stores daily F10.7 and Ap.
It needs to be extended with 3-hourly Kp values. Source: GFZ Potsdam Kp archive
or NOAA SWPC. The data volume is ~8 values/day × 365 days/year × 70 years
× ~4 bytes ≈ 800 KB.

## Acceptance Criteria

- All T1-T10 pass GREEN
- No regression in existing tests
- Full NRLMSISE-00 within 5% of reference implementation at standard test cases
- Simplified model preserved as `NRLMSISE00ModelSimplified`
- 3-hourly Ap array populated from historical data
- Storm response captured in density output
