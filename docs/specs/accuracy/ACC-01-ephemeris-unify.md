# ACC-01: Ephemeris Unification

**Contract**: ACC-01-EPHEMERIS-UNIFY
**Stage**: 1 (foundational — no dependencies)
**Effort**: Low (1 session)
**Audit items**: #6, #15, #16

## Problem

Five force model modules contain independent copies of low-precision Meeus
solar/lunar ephemeris (~1 degree accuracy, ~1700 km Sun error, ~3500 km Moon
error). The full Chebyshev ephemeris with ~100 m accuracy (2000-2050) exists
in `planetary_ephemeris.py` but is not used by any force model.

Duplicated `_sun_position_approx` in:
1. `tidal_forces.py:68-117` (solar position for solid/ocean tides)
2. `relativistic_forces.py:46-94` (solar position for de Sitter)
3. `albedo_srp.py:66-84` (solar position for albedo/IR)

Duplicated `_moon_position_approx` in:
4. `tidal_forces.py:119-158` (lunar position for tides)

Additionally:
5. `third_body.py:45-140` — `moon_position_eci()` uses 6-term Meeus (~0.5 deg)
6. `third_body.py` imports `sun_position_eci` from `solar.py` (2-term Meeus, ~1 arcmin)
7. `relativistic_forces.py:232-238` — `DeSitterForce` computes Earth velocity
   via 1-hour finite difference of low-precision solar position

## Requirements

**R1**: All force models that need Sun position SHALL use
`planetary_ephemeris.evaluate_position(eph["sun"], t)` with ~100 m accuracy.

**R2**: All force models that need Moon position SHALL use
`planetary_ephemeris.evaluate_position(eph["moon"], t)` with ~100 m accuracy.

**R3**: `DeSitterForce` SHALL use `planetary_ephemeris.evaluate_velocity(eph["sun"], t)`
for Earth heliocentric velocity instead of finite difference.

**R4**: All duplicate `_sun_position_approx` and `_moon_position_approx` private
functions SHALL be removed after migration.

**R5**: The `ForceModel.acceleration` protocol takes `epoch: datetime`. Force models
SHALL convert `datetime` to `AstroTime` internally (via `AstroTime.from_utc(epoch)`)
to call the Chebyshev ephemeris. The protocol signature is unchanged.

**R6**: `solar.sun_position_eci()` in `solar.py` SHALL remain as the low-precision
convenience function (used by CLI, visualization, non-precision paths). It is NOT
deprecated — only the force-model usage is migrated.

## Test Specifications

**T1**: `test_tidal_sun_uses_chebyshev` — `SolidTideForce.acceleration()` at a known
epoch produces acceleration consistent with Chebyshev Sun position (not Meeus).
Verify by comparing against a manually computed tidal acceleration using the
Chebyshev position.

**T2**: `test_tidal_moon_uses_chebyshev` — Same for Moon in `SolidTideForce`.

**T3**: `test_ocean_tide_uses_chebyshev` — `OceanTideForce.acceleration()` uses
Chebyshev positions for both Sun and Moon.

**T4**: `test_desitter_uses_chebyshev_velocity` — `DeSitterForce.acceleration()`
produces result consistent with analytical Chebyshev velocity (not finite diff).

**T5**: `test_albedo_uses_chebyshev_sun` — `AlbedoRadiationPressure.acceleration()`
uses Chebyshev Sun position.

**T6**: `test_third_body_solar_uses_chebyshev` — `SolarThirdBodyForce.acceleration()`
uses Chebyshev Sun position.

**T7**: `test_third_body_lunar_uses_chebyshev` — `LunarThirdBodyForce.acceleration()`
uses Chebyshev Moon position.

**T8**: `test_no_meeus_in_force_models` — AST/grep check: no `_sun_position_approx`
or `_moon_position_approx` functions remain in `tidal_forces.py`,
`relativistic_forces.py`, or `albedo_srp.py`.

**T9**: `test_solar_py_still_works` — `solar.sun_position_eci()` still returns valid
results (not broken by refactoring).

## Implementation Notes

- Ephemeris is loaded once via `load_ephemeris()` (module-level cache).
  Force models should call `load_ephemeris()` in `acceleration()` on first use
  (lazy init) or accept an ephemeris dict in `__init__`.
- Preferred: lazy load + module cache. Avoids changing constructor signatures.
- `AstroTime.from_utc(epoch)` conversion is cheap (~microseconds).
- `evaluate_position` returns meters in GCRS frame — same as what force models expect.
- `evaluate_velocity` returns m/s in GCRS frame.

## Acceptance Criteria

- All T1-T9 pass GREEN
- No regression in existing tests
- All `_sun_position_approx` / `_moon_position_approx` removed from force model files
- `solar.py` unchanged
