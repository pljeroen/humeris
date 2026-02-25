# ACC-02: Time System Fidelity

**Contract**: ACC-02-TIME-FIDELITY
**Stage**: 1 (foundational — no dependencies)
**Effort**: Low (1 session)
**Audit items**: #12, #20

## Problem

### TDB-TT series truncation (#12)
`_tdb_minus_tt` in `time_systems.py:196-212` uses only 2 terms of the Fairhead &
Bretagnon (1990) series: the Earth eccentricity term (1.657 ms amplitude) and one
Jupiter longitude term (22 us). The full series has ~60 terms with total accuracy
to ~30 us. Omitted terms contribute up to ~100 us error. At orbital velocity
7500 m/s, 100 us = 0.75 m position error per epoch.

### Constant naming hazard (#20)
`OrbitalConstants.EARTH_OMEGA = 1.99106380e-7 rad/s` is the mean motion of Earth
around the Sun (2pi/tropical year), not the rotation rate. The name is misleading.
The rotation rate is `EARTH_ROTATION_RATE = 7.2921159e-5 rad/s`. Only used in
`sso_inclination_deg()` which is correct, but the naming invites misuse.

## Requirements

**R1**: `_tdb_minus_tt` SHALL implement the 33-term Fairhead & Bretagnon (1990)
series as specified in IERS Conventions 2010, achieving accuracy better than 3 us
(equivalent to <0.02 m at LEO velocity).

**R2**: `OrbitalConstants.EARTH_OMEGA` SHALL be renamed to
`EARTH_MEAN_MOTION_SOLAR` with the same value. The old name SHALL remain as an
alias for backward compatibility, with a comment marking it deprecated.

**R3**: `sso_inclination_deg()` SHALL use the new name `EARTH_MEAN_MOTION_SOLAR`.

## Test Specifications

**T1**: `test_tdb_tt_dominant_term` — At J2000 + 0.5 years (near eccentricity
maximum), `_tdb_minus_tt` returns a value within 3 us of the known Fairhead &
Bretagnon reference value (1.6575 ms * sin(M) where M is known).

**T2**: `test_tdb_tt_accuracy_vs_reference` — Compare against tabulated SOFA
reference values at 5 epochs spanning 2000-2050. All within 3 us.

**T3**: `test_tdb_tt_series_has_33_terms` — The implementation uses >= 30 terms
(verify by counting coefficient entries or checking a regression value that
requires the smaller terms).

**T4**: `test_earth_mean_motion_solar_exists` — `OrbitalConstants.EARTH_MEAN_MOTION_SOLAR`
exists and equals 1.99106380e-7 rad/s.

**T5**: `test_earth_omega_alias_still_works` — `OrbitalConstants.EARTH_OMEGA` still
accessible (backward compat).

**T6**: `test_sso_inclination_uses_new_name` — AST/grep check: `sso_inclination_deg`
references `EARTH_MEAN_MOTION_SOLAR`.

## Implementation Notes

### TDB-TT 33-term series
Source: IERS Conventions 2010, Section 10.2, or Fairhead & Bretagnon 1990
Table 1. The series has the form:

```
TDB - TT = sum_i [ A_i * sin(w_i * T + phi_i) ]
```

where T is Julian centuries of TDB from J2000. The 33 terms can be encoded
as a tuple of (amplitude_s, frequency_rad_per_century, phase_rad) triples.

### EARTH_OMEGA rename
Add `EARTH_MEAN_MOTION_SOLAR` as the primary constant. Keep `EARTH_OMEGA` as
an alias. Update `sso_inclination_deg` to use the new name.

## Acceptance Criteria

- All T1-T6 pass GREEN
- No regression in existing tests
- TDB-TT accuracy < 3 us at all test epochs
