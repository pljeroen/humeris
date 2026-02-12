"""Inter-satellite link geometry and topology.

Line-of-sight (Earth blockage) detection between satellite pairs and
constellation-wide ISL topology analysis with range filtering.

No external dependencies — only stdlib math/dataclasses.
"""

import math
from dataclasses import dataclass

from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.propagation import OrbitalState, propagate_to

_R_EARTH = OrbitalConstants.R_EARTH


@dataclass(frozen=True)
class ISLLink:
    """A single inter-satellite link between two satellites."""
    sat_idx_a: int
    sat_idx_b: int
    distance_m: float
    is_blocked: bool


@dataclass(frozen=True)
class ISLTopology:
    """Constellation-wide ISL topology snapshot."""
    links: tuple[ISLLink, ...]
    num_satellites: int
    num_active_links: int
    mean_distance_m: float
    max_distance_m: float


def is_earth_blocked(
    pos_a_eci: tuple[float, float, float],
    pos_b_eci: tuple[float, float, float],
    earth_radius_m: float | None = None,
) -> bool:
    """Check if the line segment from A to B is blocked by the Earth.

    Finds the closest point on the segment A→B to the origin (Earth center).
    Blocked if that closest distance is less than earth_radius_m.

    Args:
        pos_a_eci: ECI position of satellite A (m).
        pos_b_eci: ECI position of satellite B (m).
        earth_radius_m: Override Earth radius (m).

    Returns:
        True if Earth blocks the line of sight.
    """
    r_earth = earth_radius_m if earth_radius_m is not None else _R_EARTH

    ax, ay, az = pos_a_eci
    bx, by, bz = pos_b_eci

    # Direction vector d = B - A
    dx = bx - ax
    dy = by - ay
    dz = bz - az

    d_dot_d = dx * dx + dy * dy + dz * dz
    if d_dot_d < 1e-20:
        # Same position: not blocked (degenerate case)
        return False

    # t_min = clamp(-dot(A, d) / |d|², 0, 1)
    a_dot_d = ax * dx + ay * dy + az * dz
    t_min = max(0.0, min(1.0, -a_dot_d / d_dot_d))

    # Closest point on segment to origin
    cx = ax + t_min * dx
    cy = ay + t_min * dy
    cz = az + t_min * dz

    closest_dist = math.sqrt(cx * cx + cy * cy + cz * cz)
    return closest_dist < r_earth


def compute_isl_link(
    pos_a_eci: tuple[float, float, float],
    pos_b_eci: tuple[float, float, float],
    idx_a: int,
    idx_b: int,
    earth_radius_m: float | None = None,
) -> ISLLink:
    """Compute ISL link properties for one satellite pair.

    Args:
        pos_a_eci: ECI position of satellite A (m).
        pos_b_eci: ECI position of satellite B (m).
        idx_a: Index of satellite A.
        idx_b: Index of satellite B.
        earth_radius_m: Override Earth radius (m).

    Returns:
        ISLLink with distance and blockage status.
    """
    dx = pos_b_eci[0] - pos_a_eci[0]
    dy = pos_b_eci[1] - pos_a_eci[1]
    dz = pos_b_eci[2] - pos_a_eci[2]
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    blocked = is_earth_blocked(pos_a_eci, pos_b_eci, earth_radius_m)

    return ISLLink(
        sat_idx_a=idx_a,
        sat_idx_b=idx_b,
        distance_m=distance,
        is_blocked=blocked,
    )


def compute_isl_topology(
    states: list[OrbitalState],
    time,
    max_range_km: float = 5000.0,
) -> ISLTopology:
    """Compute ISL topology for all satellite pairs at a given time.

    Propagates each state to the target time, evaluates all pairs for
    distance and Earth blockage, filters by max range.

    Args:
        states: List of OrbitalState objects.
        time: Target datetime for evaluation.
        max_range_km: Maximum ISL range in km (links beyond this are excluded
            from active count).

    Returns:
        ISLTopology with all evaluated links and aggregate statistics.
    """
    n = len(states)
    if n < 2:
        return ISLTopology(
            links=(),
            num_satellites=n,
            num_active_links=0,
            mean_distance_m=0.0,
            max_distance_m=0.0,
        )

    max_range_m = max_range_km * 1000.0

    # Propagate all states
    positions = []
    for state in states:
        pos, _vel = propagate_to(state, time)
        positions.append((pos[0], pos[1], pos[2]))

    # Evaluate all pairs
    links: list[ISLLink] = []
    active_distances: list[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            link = compute_isl_link(positions[i], positions[j], i, j)
            links.append(link)

            if link.distance_m <= max_range_m and not link.is_blocked:
                active_distances.append(link.distance_m)

    num_active = len(active_distances)
    mean_dist = sum(active_distances) / num_active if num_active > 0 else 0.0
    max_dist = max(active_distances) if num_active > 0 else 0.0

    return ISLTopology(
        links=tuple(links),
        num_satellites=n,
        num_active_links=num_active,
        mean_distance_m=mean_dist,
        max_distance_m=max_dist,
    )
