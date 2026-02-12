"""Geometric Dilution of Precision (GDOP/PDOP/HDOP/VDOP/TDOP).

Computes positioning accuracy metrics from satellite geometry as seen
from a ground point, using azimuth/elevation observations.

No external dependencies — only stdlib math/dataclasses.
"""

import math
from dataclasses import dataclass

from constellation_generator.domain.propagation import (
    OrbitalState,
    propagate_ecef_to,
)
from constellation_generator.domain.observation import (
    GroundStation,
    compute_observation,
)


@dataclass(frozen=True)
class DOPResult:
    """Dilution of Precision components for a ground point."""
    gdop: float
    pdop: float
    hdop: float
    vdop: float
    tdop: float
    num_visible: int


@dataclass(frozen=True)
class DOPGridPoint:
    """DOP result at a specific lat/lon grid point."""
    lat_deg: float
    lon_deg: float
    dop: DOPResult


_INF_DOP = DOPResult(
    gdop=float('inf'), pdop=float('inf'), hdop=float('inf'),
    vdop=float('inf'), tdop=float('inf'), num_visible=0,
)


def _invert_4x4(m: list[list[float]]) -> list[list[float]] | None:
    """Invert a 4x4 matrix via Gaussian elimination with partial pivoting.

    Args:
        m: 4x4 matrix as list of lists.

    Returns:
        Inverse matrix, or None if singular.
    """
    n = 4
    # Augmented matrix [M | I]
    aug = [
        [m[i][j] for j in range(n)] + [1.0 if i == k else 0.0 for k in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        # Partial pivoting: find row with largest absolute value in column
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-15:
            return None

        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Eliminate below and above
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    # Extract inverse from augmented matrix
    return [[aug[i][j + n] for j in range(n)] for i in range(n)]


def compute_dop(
    ground_lat_deg: float,
    ground_lon_deg: float,
    sat_positions_ecef: list[tuple[float, float, float]],
    min_elevation_deg: float = 10.0,
) -> DOPResult:
    """Compute Dilution of Precision at a ground point.

    For each satellite above min_elevation, builds a geometry matrix H
    from azimuth/elevation, then computes Q = (H^T H)^-1 to extract
    GDOP, PDOP, HDOP, VDOP, TDOP.

    Args:
        ground_lat_deg: Ground point latitude (degrees).
        ground_lon_deg: Ground point longitude (degrees).
        sat_positions_ecef: List of satellite ECEF positions (m).
        min_elevation_deg: Minimum elevation for visibility (degrees).

    Returns:
        DOPResult with DOP values and visible satellite count.
    """
    station = GroundStation(
        name='dop', lat_deg=ground_lat_deg, lon_deg=ground_lon_deg,
    )

    # Filter visible satellites and build geometry matrix rows
    h_rows: list[list[float]] = []
    for sat_ecef in sat_positions_ecef:
        obs = compute_observation(station, sat_ecef)
        if obs.elevation_deg >= min_elevation_deg:
            el_rad = math.radians(obs.elevation_deg)
            az_rad = math.radians(obs.azimuth_deg)
            # H row: [cos(el)*sin(az), cos(el)*cos(az), sin(el), 1]
            h_rows.append([
                math.cos(el_rad) * math.sin(az_rad),
                math.cos(el_rad) * math.cos(az_rad),
                math.sin(el_rad),
                1.0,
            ])

    num_visible = len(h_rows)
    if num_visible < 4:
        return DOPResult(
            gdop=float('inf'), pdop=float('inf'), hdop=float('inf'),
            vdop=float('inf'), tdop=float('inf'), num_visible=num_visible,
        )

    # Compute H^T H (4x4)
    hth: list[list[float]] = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for row in h_rows:
                hth[i][j] += row[i] * row[j]

    # Q = (H^T H)^-1
    q = _invert_4x4(hth)
    if q is None:
        return DOPResult(
            gdop=float('inf'), pdop=float('inf'), hdop=float('inf'),
            vdop=float('inf'), tdop=float('inf'), num_visible=num_visible,
        )

    gdop = math.sqrt(q[0][0] + q[1][1] + q[2][2] + q[3][3])
    pdop = math.sqrt(q[0][0] + q[1][1] + q[2][2])
    hdop = math.sqrt(q[0][0] + q[1][1])
    vdop = math.sqrt(q[2][2])
    tdop = math.sqrt(q[3][3])

    return DOPResult(
        gdop=gdop, pdop=pdop, hdop=hdop, vdop=vdop, tdop=tdop,
        num_visible=num_visible,
    )


def compute_dop_grid(
    states: list[OrbitalState],
    time,
    lat_step_deg: float = 10.0,
    lon_step_deg: float = 10.0,
    min_elevation_deg: float = 10.0,
    lat_range: tuple[float, float] = (-90, 90),
    lon_range: tuple[float, float] = (-180, 180),
) -> list[DOPGridPoint]:
    """Compute DOP over a latitude/longitude grid.

    Precomputes satellite ECEF positions, then evaluates DOP at each
    grid point.

    Args:
        states: List of OrbitalState objects.
        time: UTC datetime for evaluation.
        lat_step_deg: Latitude grid spacing (degrees).
        lon_step_deg: Longitude grid spacing (degrees).
        min_elevation_deg: Minimum elevation for visibility.
        lat_range: (min_lat, max_lat) in degrees.
        lon_range: (min_lon, max_lon) in degrees.

    Returns:
        List of DOPGridPoint for the entire grid.
    """
    sat_ecefs = [propagate_ecef_to(state, time) for state in states]

    grid: list[DOPGridPoint] = []

    lat = lat_range[0]
    while lat <= lat_range[1] + 1e-9:
        lon = lon_range[0]
        while lon <= lon_range[1] - lon_step_deg + 1e-9:
            dop = compute_dop(lat, lon, sat_ecefs, min_elevation_deg)
            grid.append(DOPGridPoint(lat_deg=lat, lon_deg=lon, dop=dop))
            lon += lon_step_deg
        lat += lat_step_deg

    return grid
