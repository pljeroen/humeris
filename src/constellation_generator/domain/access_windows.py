"""
Access window (rise/set) detection.

Sweeps propagated satellite positions over time, detecting elevation
threshold crossings to build visibility windows from a ground station.

No external dependencies — only stdlib dataclasses/datetime.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

from constellation_generator.domain.propagation import (
    OrbitalState,
    propagate_ecef_to,
)
from constellation_generator.domain.observation import (
    GroundStation,
    compute_observation,
)


@dataclass(frozen=True)
class AccessWindow:
    """A single visibility window between a station and a satellite."""
    rise_time: datetime
    set_time: datetime
    max_elevation_deg: float
    duration_seconds: float


def compute_access_windows(
    station: GroundStation,
    orbital_state: OrbitalState,
    start: datetime,
    duration: timedelta,
    step: timedelta,
    min_elevation_deg: float = 10.0,
) -> list[AccessWindow]:
    """
    Compute satellite visibility windows from a ground station.

    Sweeps propagated positions at the given step interval, tracking
    elevation threshold crossings to detect rise and set times.

    Args:
        station: Ground observation station.
        orbital_state: Satellite orbital state for propagation.
        start: Start time (UTC).
        duration: Total analysis duration.
        step: Time step for sweep.
        min_elevation_deg: Minimum elevation for visibility.

    Returns:
        List of AccessWindow objects, chronologically ordered.

    Raises:
        ValueError: If step is zero or negative.
    """
    step_seconds = step.total_seconds()
    if step_seconds <= 0:
        raise ValueError(f"Step must be positive, got {step}")

    duration_seconds = duration.total_seconds()
    windows: list[AccessWindow] = []

    in_window = False
    rise_time = start
    max_el = 0.0
    elapsed = 0.0

    while elapsed <= duration_seconds + 1e-9:
        current_time = start + timedelta(seconds=elapsed)
        sat_ecef = propagate_ecef_to(orbital_state, current_time)
        obs = compute_observation(station, sat_ecef)
        el = obs.elevation_deg

        if el >= min_elevation_deg:
            if not in_window:
                rise_time = current_time
                max_el = el
                in_window = True
            else:
                if el > max_el:
                    max_el = el
        else:
            if in_window:
                set_time = current_time
                dur = (set_time - rise_time).total_seconds()
                windows.append(AccessWindow(
                    rise_time=rise_time,
                    set_time=set_time,
                    max_elevation_deg=max_el,
                    duration_seconds=dur,
                ))
                in_window = False

        elapsed += step_seconds

    if in_window:
        set_time = start + timedelta(seconds=duration_seconds)
        dur = (set_time - rise_time).total_seconds()
        windows.append(AccessWindow(
            rise_time=rise_time,
            set_time=set_time,
            max_elevation_deg=max_el,
            duration_seconds=dur,
        ))

    return windows
