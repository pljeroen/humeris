"""
Grid-based coverage analysis.

Computes a snapshot of how many satellites are visible from each point
on a latitude/longitude grid at a given time.

No external dependencies — only stdlib dataclasses/datetime.
"""
from dataclasses import dataclass
from datetime import datetime

from constellation_generator.domain.propagation import (
    OrbitalState,
    propagate_ecef_to,
)
from constellation_generator.domain.observation import (
    GroundStation,
    compute_observation,
)


@dataclass(frozen=True)
class CoveragePoint:
    """A grid point with its satellite visibility count."""
    lat_deg: float
    lon_deg: float
    visible_count: int


def compute_coverage_snapshot(
    orbital_states: list[OrbitalState],
    analysis_time: datetime,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    min_elevation_deg: float = 10.0,
    lat_range: tuple[float, float] = (-90, 90),
    lon_range: tuple[float, float] = (-180, 180),
) -> list[CoveragePoint]:
    """
    Compute a coverage snapshot: how many satellites are visible per grid point.

    Precomputes all satellite ECEF positions, then for each grid point
    counts how many are above min_elevation.

    Args:
        orbital_states: List of OrbitalState objects to evaluate.
        analysis_time: UTC time for the snapshot.
        lat_step_deg: Latitude grid spacing in degrees.
        lon_step_deg: Longitude grid spacing in degrees.
        min_elevation_deg: Minimum elevation for visibility.
        lat_range: (min_lat, max_lat) in degrees.
        lon_range: (min_lon, max_lon) in degrees.

    Returns:
        List of CoveragePoint objects for the entire grid.
    """
    sat_ecefs = [propagate_ecef_to(state, analysis_time) for state in orbital_states]

    grid: list[CoveragePoint] = []

    lat = lat_range[0]
    while lat <= lat_range[1] + 1e-9:
        lon = lon_range[0]
        while lon <= lon_range[1] - lon_step_deg + 1e-9:
            station = GroundStation(name='grid', lat_deg=lat, lon_deg=lon, alt_m=0.0)
            count = 0
            for sat_ecef in sat_ecefs:
                obs = compute_observation(station, sat_ecef)
                if obs.elevation_deg >= min_elevation_deg:
                    count += 1
            grid.append(CoveragePoint(lat_deg=lat, lon_deg=lon, visible_count=count))
            lon += lon_step_deg
        lat += lat_step_deg

    return grid
