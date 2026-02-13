# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Compact planetary ephemeris via Chebyshev interpolation.

Provides Sun and Moon geocentric positions in GCRS (meters) at
~100m accuracy using pre-computed Chebyshev polynomial coefficients
covering 2000-2050.

References:
    Newhall, X X (1989). "Numerical Representation of Planetary Ephemerides."
    JPL DE440 documentation.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from humeris.domain.time_systems import AstroTime

# --------------------------------------------------------------------------- #
# Chebyshev evaluation (Clenshaw recurrence)
# --------------------------------------------------------------------------- #


def chebyshev_evaluate(coeffs: tuple, t_norm: float) -> float:
    """Evaluate a Chebyshev series at normalized point t_norm ∈ [-1, 1].

    Uses the Clenshaw recurrence: O(n), numerically stable.

    Parameters
    ----------
    coeffs : tuple[float, ...]
        Chebyshev coefficients [c0, c1, ..., cn].
    t_norm : float
        Normalized argument in [-1, 1].

    Returns
    -------
    Value of the Chebyshev series at t_norm.
    """
    n = len(coeffs)
    if n == 0:
        return 0.0
    if n == 1:
        return coeffs[0]

    # Clenshaw recurrence from the last coefficient
    b_kp2 = 0.0
    b_kp1 = 0.0
    for k in range(n - 1, 0, -1):
        b_k = coeffs[k] + 2.0 * t_norm * b_kp1 - b_kp2
        b_kp2 = b_kp1
        b_kp1 = b_k

    return coeffs[0] + t_norm * b_kp1 - b_kp2


def _chebyshev_derivative(coeffs: tuple, t_norm: float, half_span: float) -> float:
    """Evaluate the derivative of a Chebyshev series.

    Uses the recurrence relation for Chebyshev derivative coefficients:
        d'_N = 0, d'_{N-1} = 2N * c_N
        d'_k = d'_{k+2} + 2(k+1) * c_{k+1}  for k = N-2 ... 1
        d'_0 = d'_2 / 2 + c_1

    Then evaluate the derivative series and scale by 1/half_span.
    """
    n = len(coeffs)
    if n <= 1:
        return 0.0

    # Compute derivative coefficients via backward recurrence
    # d'_k gives the Chebyshev coefficients of the derivative polynomial
    dp = [0.0] * n

    # Start from the end
    dp[n - 1] = 0.0
    if n >= 2:
        dp[n - 2] = 2.0 * (n - 1) * coeffs[n - 1]

    for k in range(n - 3, 0, -1):
        dp[k] = dp[k + 2] + 2.0 * (k + 1) * coeffs[k + 1]

    # k=0: special case (halved)
    dp[0] = dp[2] / 2.0 + coeffs[1] if n > 2 else coeffs[1]

    # Evaluate derivative polynomial at t_norm
    result = chebyshev_evaluate(tuple(dp[:n - 1] if n > 1 else dp[:1]), t_norm)

    # Scale: df/dt = (df/dt_norm) / half_span
    return result / half_span


# --------------------------------------------------------------------------- #
# Ephemeris loading
# --------------------------------------------------------------------------- #

_CACHED_EPHEMERIS: Optional[dict] = None


def load_ephemeris(path: Optional[str] = None) -> dict:
    """Load Chebyshev ephemeris data from bundled JSON or custom path.

    Returns
    -------
    dict with keys "sun" and "moon", each containing:
        - gm: gravitational parameter (m³/s²)
        - granule_days: days per granule
        - degree: polynomial degree
        - coefficient_scale: multiplier for stored integer coefficients
        - granules: list of [[cx], [cy], [cz]] coefficient arrays
        - t_start, t_end: epoch range (seconds from J2000 TDB)
    """
    global _CACHED_EPHEMERIS

    if path is None and _CACHED_EPHEMERIS is not None:
        return _CACHED_EPHEMERIS

    if path is None:
        data_path = Path(__file__).parent.parent / "data" / "sun_moon_chebyshev.json"
    else:
        data_path = Path(path)

    with open(data_path) as f:
        data = json.load(f)

    t_start = data.get("t_start", 0)
    t_end = data.get("t_end", 0)

    result = {}
    for body_name, body_data in data["bodies"].items():
        granule_s = body_data["granule_days"] * 86400.0
        scale = body_data.get("coefficient_scale", 1.0)

        # Pre-process granules: convert to tuples of floats, apply scale
        granules = []
        for g in body_data["granules"]:
            # g = [[cx], [cy], [cz]] where coefficients are scaled integers
            cx = tuple(c * scale for c in g[0])
            cy = tuple(c * scale for c in g[1])
            cz = tuple(c * scale for c in g[2])
            granules.append((cx, cy, cz))

        result[body_name] = {
            "gm": body_data["gm"],
            "granule_days": body_data["granule_days"],
            "granule_seconds": granule_s,
            "degree": body_data["degree"],
            "coefficient_scale": scale,
            "granules": granules,
            "t_start": t_start,
            "t_end": t_end,
            "n_granules": len(granules),
        }

    if path is None:
        _CACHED_EPHEMERIS = result

    return result


# --------------------------------------------------------------------------- #
# Position and velocity evaluation
# --------------------------------------------------------------------------- #


def _find_granule(body: dict, t: AstroTime) -> tuple[int, float]:
    """Find the granule index and normalized time for a given epoch.

    Returns
    -------
    (index, t_norm) : granule index and normalized time in [-1, 1].

    Raises
    ------
    ValueError if t is outside the ephemeris range.
    """
    tdb_s = t.tdb_j2000
    t_start = body["t_start"]
    t_end = body["t_end"]
    granule_s = body["granule_seconds"]

    if tdb_s < t_start - granule_s:
        raise ValueError(
            f"Epoch {tdb_s:.0f}s TDB is before ephemeris range "
            f"(starts at {t_start:.0f}s)"
        )
    if tdb_s > t_end + granule_s:
        raise ValueError(
            f"Epoch {tdb_s:.0f}s TDB is after ephemeris range "
            f"(ends at {t_end:.0f}s)"
        )

    # Granule index
    idx = int((tdb_s - t_start) / granule_s)
    idx = max(0, min(idx, body["n_granules"] - 1))

    # Granule time boundaries
    g_t0 = t_start + idx * granule_s
    g_t1 = g_t0 + granule_s

    # Normalize to [-1, 1]
    mid = (g_t0 + g_t1) / 2.0
    half = (g_t1 - g_t0) / 2.0
    t_norm = (tdb_s - mid) / half

    # Clamp to [-1, 1] for safety at boundaries
    t_norm = max(-1.0, min(1.0, t_norm))

    return idx, t_norm


def evaluate_position(
    body: dict,
    t: AstroTime,
) -> tuple[float, float, float]:
    """Evaluate body geocentric position at epoch.

    Parameters
    ----------
    body : dict
        Body ephemeris data from load_ephemeris() (e.g., eph["sun"]).
    t : AstroTime
        Epoch.

    Returns
    -------
    (x, y, z) in meters, GCRS frame.
    """
    idx, t_norm = _find_granule(body, t)
    cx, cy, cz = body["granules"][idx]

    return (
        chebyshev_evaluate(cx, t_norm),
        chebyshev_evaluate(cy, t_norm),
        chebyshev_evaluate(cz, t_norm),
    )


def evaluate_velocity(
    body: dict,
    t: AstroTime,
) -> tuple[float, float, float]:
    """Evaluate body geocentric velocity at epoch via Chebyshev derivative.

    Parameters
    ----------
    body : dict
        Body ephemeris data from load_ephemeris() (e.g., eph["sun"]).
    t : AstroTime
        Epoch.

    Returns
    -------
    (vx, vy, vz) in m/s, GCRS frame.
    """
    idx, t_norm = _find_granule(body, t)
    cx, cy, cz = body["granules"][idx]
    half_span = body["granule_seconds"] / 2.0

    return (
        _chebyshev_derivative(cx, t_norm, half_span),
        _chebyshev_derivative(cy, t_norm, half_span),
        _chebyshev_derivative(cz, t_norm, half_span),
    )
