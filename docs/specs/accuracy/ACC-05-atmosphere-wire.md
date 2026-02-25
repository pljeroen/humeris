# ACC-05: Atmosphere Drag Wiring

**Contract**: ACC-05-ATMOSPHERE-WIRE
**Stage**: 1 (no dependencies)
**Effort**: Medium (1 session)
**Audit items**: #1

## Problem

`AtmosphericDragForce` in `numerical_propagation.py:422-425` calls
`atmospheric_density()` from `atmosphere.py` — a hardcoded piecewise-exponential
lookup table (Vallado Table 8-4) with zero solar activity dependence, zero
diurnal variation, zero seasonal variation, and zero geomagnetic response.

The full NRLMSISE-00 model class exists in `nrlmsise00.py` with
`NRLMSISE00DragForce` providing a complete drag force implementation. However,
the basic `AtmosphericDragForce` class (used by many tests and the default
`propagate_numerical` drag path) still uses the table lookup.

At 400 km altitude, density varies by 2-10x between solar min/max. The table
gives a single fixed value. This produces position errors of tens to hundreds
of km per day for multi-day LEO propagations.

## Requirements

**R1**: `AtmosphericDragForce` SHALL accept an optional `space_weather: SpaceWeather | None`
constructor parameter. When provided, it SHALL use `NRLMSISE00Model.evaluate()`
for density instead of `atmospheric_density()`.

**R2**: `AtmosphericDragForce` SHALL accept an optional
`weather_provider: SpaceWeatherProvider | None` constructor parameter. When
provided, it SHALL look up space weather at each epoch dynamically.

**R3**: When neither `space_weather` nor `weather_provider` is provided,
`AtmosphericDragForce` SHALL fall back to the existing `atmospheric_density()`
table lookup (backward compatible).

**R4**: `AtmosphericDragForce` SHALL convert ECI position to geodetic coordinates
(latitude, longitude, altitude) for the NRLMSISE-00 evaluation using the existing
`eci_to_ecef` + `ecef_to_geodetic` path.

**R5**: The conversion from `datetime` to NRLMSISE-00 time parameters (year,
day_of_year, ut_seconds) SHALL be correct for the epoch.

## Test Specifications

**T1**: `test_drag_with_space_weather` — `AtmosphericDragForce` constructed with
`space_weather=SpaceWeather(f107_daily=150, f107_average=150, ap_daily=4)` at
400 km altitude produces a different (and larger, for high F10.7) drag acceleration
than the default table lookup.

**T2**: `test_drag_with_provider` — `AtmosphericDragForce` with a
`SpaceWeatherProvider` produces epoch-dependent density (density changes between
solar max and solar min epochs).

**T3**: `test_drag_default_unchanged` — Without optional params, behavior is
identical to current (backward compat).

**T4**: `test_drag_nrlmsise_vs_table` — At 400 km, solar max (F10.7=200), the
NRLMSISE-00 density is significantly higher than the table density (factor > 2).

**T5**: `test_drag_geodetic_conversion` — The ECI-to-geodetic conversion in the
drag force produces correct latitude/longitude/altitude (verify against known
position).

**T6**: `test_drag_epoch_to_nrlmsise_params` — Year, day_of_year, ut_seconds
extracted correctly from a known datetime.

## Implementation Notes

### In `AtmosphericDragForce.__init__`:
```python
def __init__(self, drag_config: DragConfig,
             space_weather: SpaceWeather | None = None,
             weather_provider: SpaceWeatherProvider | None = None):
```

### In `AtmosphericDragForce.acceleration`:
```python
if self._weather_provider is not None:
    sw = self._weather_provider.lookup(epoch)
    # use NRLMSISE00Model with sw
elif self._space_weather is not None:
    # use NRLMSISE00Model with fixed sw
else:
    # existing table lookup (backward compat)
```

### ECI to geodetic:
```python
gmst = gmst_rad(epoch)
pos_ecef, _ = eci_to_ecef(position, velocity, gmst)
lat_deg, lon_deg, alt_m = ecef_to_geodetic(pos_ecef)
alt_km = alt_m / 1000.0
```

### Epoch to NRLMSISE-00 params:
```python
year = epoch.year
doy = (epoch - datetime(epoch.year, 1, 1, tzinfo=epoch.tzinfo)).days + 1
ut_seconds = epoch.hour * 3600 + epoch.minute * 60 + epoch.second
```

## Acceptance Criteria

- All T1-T6 pass GREEN
- No regression in existing tests
- Default behavior (no space weather) unchanged
- With space weather, drag acceleration varies with solar activity
