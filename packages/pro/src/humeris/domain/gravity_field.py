# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Cunningham V/W recursion for spherical harmonic gravity.

Singularity-free algorithm that works directly in ECEF Cartesian
coordinates. No trig, no polar singularity. Industry standard
(GMAT, MONTE, Orekit).

Perturbation-only: does not include the central body term.
Must be used with TwoBodyGravity for total gravitational acceleration.

Reference: Montenbruck & Gill, "Satellite Orbits", Ch. 3.2
           Cunningham (1970), "On the computation of the spherical
           harmonic terms needed during the numerical integration
           of the orbital motion of an artificial satellite"
"""

import json
import math
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _tri(n: int, m: int) -> int:
    """Triangular index: maps (n, m) to flat array position."""
    return n * (n + 1) // 2 + m


def _norm_factor(n: int, m: int) -> float:
    """Fully normalized Legendre normalization factor.

    N_nm = sqrt((2 - delta_{m,0}) * (2n+1) * (n-m)! / (n+m)!)

    Uses math.lgamma for numerical stability at high degree.
    """
    delta = 1.0 if m == 0 else 0.0
    log_ratio = math.lgamma(n - m + 1) - math.lgamma(n + m + 1)
    return math.sqrt((2.0 - delta) * (2 * n + 1) * math.exp(log_ratio))


@dataclass(frozen=True)
class GravityFieldModel:
    """Precomputed gravity field model with Cunningham recursion coefficients.

    Flat triangular storage: index = n*(n+1)//2 + m for (n, m).
    Arrays sized to (max_degree+1)*(max_degree+2)//2.
    """

    name: str
    max_degree: int
    gm: float
    radius: float
    c_bar: tuple[float, ...]
    s_bar: tuple[float, ...]
    _diag_coeff: tuple[float, ...]
    _vert_a: tuple[float, ...]
    _vert_b: tuple[float, ...]
    _acc_p: tuple[float, ...]
    _acc_q: tuple[float, ...]
    _acc_s: tuple[float, ...]


def load_gravity_field(
    max_degree: int = 70,
    path: str | pathlib.Path | None = None,
) -> GravityFieldModel:
    """Load EGM96 coefficients and precompute recursion coefficients.

    Args:
        max_degree: Maximum degree/order (2+). Bundled data supports up to 70.
            For higher degrees (120, 200, 360), provide a custom coefficient
            JSON file via the ``path`` parameter. The JSON format is::

                {"name": "...", "max_degree": N, "gm": ..., "radius": ...,
                 "coefficients": {"n,m": [C_nm, S_nm], ...}}

        path: Path to coefficient JSON file. When None (default), selects the
            bundled egm96_70.json (supports max_degree up to 70).

    Returns:
        Precomputed GravityFieldModel ready for acceleration computation.

    Raises:
        ValueError: If max_degree < 2 or exceeds the data file's max degree.
    """
    if max_degree < 2:
        raise ValueError(f"max_degree must be >= 2, got {max_degree}")

    if path is None:
        path = pathlib.Path(__file__).parent.parent / "data" / "egm96_70.json"
    else:
        path = pathlib.Path(path)

    with open(path) as f:
        raw: dict[str, Any] = json.load(f)

    gm = raw["gm"]
    radius = raw["radius"]
    name = raw["name"]
    coefficients = raw["coefficients"]
    data_max_degree = raw.get("max_degree", max_degree)

    if max_degree > data_max_degree:
        raise ValueError(
            f"max_degree {max_degree} exceeds data file's max degree "
            f"{data_max_degree}. Provide a higher-degree coefficient file "
            f"via the path parameter."
        )

    # N+1 for V/W recursion (acceleration formulas reference one degree higher)
    n_ext = max_degree + 1
    size = (n_ext + 1) * (n_ext + 2) // 2

    c_bar = [0.0] * size
    s_bar = [0.0] * size

    for key, (c, s) in coefficients.items():
        n, m = (int(x) for x in key.split(","))
        if n <= max_degree:
            idx = _tri(n, m)
            c_bar[idx] = c
            s_bar[idx] = s

    # Precompute diagonal recursion coefficients: α_n
    # V̄_nn = α_n * (xr * V̄_{n-1,n-1} - yr * W̄_{n-1,n-1})
    # α_n = sqrt((2n+1) / (2n)) for n >= 1
    diag_coeff = [0.0] * (n_ext + 1)
    for n in range(1, n_ext + 1):
        diag_coeff[n] = math.sqrt((2.0 * n + 1.0) / (2.0 * n))

    # Precompute vertical recursion coefficients: β_nm, γ_nm
    # Standard fully-normalized recursion:
    #   V̄_nm = β_nm * zr * V̄_{n-1,m} - γ_nm * rr * V̄_{n-2,m}
    # where:
    #   β_nm = sqrt((2n-1)(2n+1) / ((n-m)(n+m)))
    #   γ_nm = sqrt((2n+1)(n-m-1)(n+m-1) / ((2n-3)(n-m)(n+m)))
    vert_a = [0.0] * size
    vert_b = [0.0] * size
    for n in range(2, n_ext + 1):
        for m in range(n):  # m < n
            idx = _tri(n, m)
            nm = (n - m) * (n + m)
            if nm > 0:
                vert_a[idx] = math.sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / nm)
                if n >= 2:
                    nm1 = (n - m - 1) * (n + m - 1)
                    if nm1 >= 0:
                        vert_b[idx] = math.sqrt(
                            (2.0 * n + 1.0) * nm1 / ((2.0 * n - 3.0) * nm)
                        )
    # Handle n=1, m=0 (sub-diagonal seed): β = sqrt(3), γ = 0
    if n_ext >= 1:
        vert_a[_tri(1, 0)] = math.sqrt(3.0)

    # Precompute acceleration summation coefficients.
    # These are cross-normalization ratios between degree n and n+1.
    # p_nm = N_nm / N_{n+1,m+1}
    # q_nm = (n-m+2)(n-m+1) * N_nm / N_{n+1,m-1}  [only m >= 1]
    # s_nm = (n-m+1) * N_nm / N_{n+1,m}
    # But it's cleaner to express them directly:
    #   p_nm = sqrt((n+m+2)(n+m+1) / ((2n+1)(2n+3))) * correction_for_delta
    #   etc.
    # We precompute for n=2..max_degree, m=0..n.
    acc_size = (max_degree + 1) * (max_degree + 2) // 2
    acc_p = [0.0] * acc_size
    acc_q = [0.0] * acc_size
    acc_s = [0.0] * acc_size

    for n in range(2, max_degree + 1):
        for m in range(n + 1):
            idx = _tri(n, m)

            # p_nm: relates V̄_{n+1,m+1} to acceleration
            # For m=0: factor of 1; for m>=1: factor of 0.5 applied in loop
            # p_nm = sqrt((n-m+1)(n-m+2) * (2n+1)/(2n+3) * delta_correction)
            # Actually: p = N_nm / N_{n+1,m+1} where N is the normalization factor
            nf_nm = _norm_factor(n, m)
            if m + 1 <= n + 1:
                nf_np1_mp1 = _norm_factor(n + 1, m + 1)
                if nf_np1_mp1 != 0.0:
                    acc_p[idx] = nf_nm / nf_np1_mp1

            # q_nm: for m >= 1
            if m >= 1:
                nf_np1_mm1 = _norm_factor(n + 1, m - 1)
                if nf_np1_mm1 != 0.0:
                    acc_q[idx] = (n - m + 2) * (n - m + 1) * nf_nm / nf_np1_mm1

            # s_nm: z-component
            nf_np1_m = _norm_factor(n + 1, m)
            if nf_np1_m != 0.0:
                acc_s[idx] = (n - m + 1) * nf_nm / nf_np1_m

    return GravityFieldModel(
        name=name,
        max_degree=max_degree,
        gm=gm,
        radius=radius,
        c_bar=tuple(c_bar),
        s_bar=tuple(s_bar),
        _diag_coeff=tuple(diag_coeff),
        _vert_a=tuple(vert_a),
        _vert_b=tuple(vert_b),
        _acc_p=tuple(acc_p),
        _acc_q=tuple(acc_q),
        _acc_s=tuple(acc_s),
    )


def _gmst_rad(epoch: datetime) -> float:
    """Greenwich Mean Sidereal Time in radians."""
    from humeris.domain.coordinate_frames import gmst_rad
    return gmst_rad(epoch)


class CunninghamGravity:
    """Cunningham V/W recursion spherical harmonic gravity.

    Perturbation-only: does not include the central body term (-mu/r² r̂).
    Must be composed with TwoBodyGravity for total gravitational acceleration.
    Implements the ForceModel protocol.
    """

    def __init__(self, model: GravityFieldModel) -> None:
        self._model = model

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Compute perturbation acceleration in ECI frame.

        1. Rotate ECI -> ECEF via GMST
        2. Compute V̄/W̄ via Cunningham recursion (singularity-free)
        3. Sum perturbation acceleration in ECEF
        4. Rotate ECEF -> ECI
        """
        m = self._model
        x_eci, y_eci, z_eci = position
        r2 = x_eci * x_eci + y_eci * y_eci + z_eci * z_eci
        r = math.sqrt(r2)
        if r < 1.0:
            return (0.0, 0.0, 0.0)

        # Rotate ECI -> ECEF
        gmst = _gmst_rad(epoch)
        cos_g = math.cos(gmst)
        sin_g = math.sin(gmst)
        x = cos_g * x_eci + sin_g * y_eci
        y = -sin_g * x_eci + cos_g * y_eci
        z = z_eci

        re = m.radius
        gm = m.gm
        n_max = m.max_degree
        n_ext = n_max + 1  # extended degree for V/W

        # Cunningham auxiliary variables
        re_r2 = re / r2
        xr = x * re_r2
        yr = y * re_r2
        zr = z * re_r2
        rr = re * re / (r2)

        # V̄/W̄ arrays: flat triangular, indices 0..(n_ext+1)*(n_ext+2)//2
        vw_size = (n_ext + 1) * (n_ext + 2) // 2
        v = np.zeros(vw_size)
        w = np.zeros(vw_size)

        # Seed: V̄₀₀ = Rₑ/r, W̄₀₀ = 0
        v[0] = re / r

        # Build V̄/W̄ to degree n_ext
        diag = m._diag_coeff
        va = m._vert_a
        vb = m._vert_b

        for n in range(1, n_ext + 1):
            # Diagonal: V̄_nn, W̄_nn
            idx_nn = _tri(n, n)
            idx_prev = _tri(n - 1, n - 1)
            alpha = diag[n]
            v[idx_nn] = alpha * (xr * v[idx_prev] - yr * w[idx_prev])
            w[idx_nn] = alpha * (xr * w[idx_prev] + yr * v[idx_prev])

            # Sub-diagonal (m = n-1): use vertical recursion with V̄_{n-2,m} = 0
            if n >= 2:
                m_val = n - 1
                idx_nm = _tri(n, m_val)
                idx_n1m = _tri(n - 1, m_val)
                beta = va[idx_nm]
                v[idx_nm] = beta * zr * v[idx_n1m]
                w[idx_nm] = beta * zr * w[idx_n1m]

            # Vertical recursion for m = 0..n-2
            for m_val in range(n - 2, -1, -1):
                idx_nm = _tri(n, m_val)
                idx_n1m = _tri(n - 1, m_val)
                idx_n2m = _tri(n - 2, m_val)
                beta = va[idx_nm]
                gamma = vb[idx_nm]
                v[idx_nm] = beta * zr * v[idx_n1m] - gamma * rr * v[idx_n2m]
                w[idx_nm] = beta * zr * w[idx_n1m] - gamma * rr * w[idx_n2m]

        # Accumulate perturbation acceleration in ECEF
        ax = 0.0
        ay = 0.0
        az = 0.0

        c_bar = m.c_bar
        s_bar = m.s_bar
        acc_p = m._acc_p
        acc_q = m._acc_q
        acc_s = m._acc_s

        for n in range(2, n_max + 1):
            # m = 0 case
            idx_n0 = _tri(n, 0)
            c_n0 = c_bar[idx_n0]
            p_n0 = acc_p[idx_n0]
            s_n0 = acc_s[idx_n0]

            idx_np1_1 = _tri(n + 1, 1)
            idx_np1_0 = _tri(n + 1, 0)

            ax += -c_n0 * p_n0 * v[idx_np1_1]
            ay += -c_n0 * p_n0 * w[idx_np1_1]
            az += -c_n0 * s_n0 * v[idx_np1_0]

            # m >= 1 cases
            for m_val in range(1, n + 1):
                idx_nm = _tri(n, m_val)
                c_nm = c_bar[idx_nm]
                s_nm = s_bar[idx_nm]
                p_nm = acc_p[idx_nm]
                q_nm = acc_q[idx_nm]
                s_nm_coeff = acc_s[idx_nm]

                idx_np1_mp1 = _tri(n + 1, m_val + 1)
                idx_np1_mm1 = _tri(n + 1, m_val - 1)
                idx_np1_m = _tri(n + 1, m_val)

                v_mp1 = v[idx_np1_mp1]
                w_mp1 = w[idx_np1_mp1]
                v_mm1 = v[idx_np1_mm1]
                w_mm1 = w[idx_np1_mm1]
                v_m = v[idx_np1_m]
                w_m = w[idx_np1_m]

                ax += 0.5 * (
                    -c_nm * p_nm * v_mp1
                    - s_nm * p_nm * w_mp1
                    + q_nm * (c_nm * v_mm1 + s_nm * w_mm1)
                )
                ay += 0.5 * (
                    -c_nm * p_nm * w_mp1
                    + s_nm * p_nm * v_mp1
                    + q_nm * (-c_nm * w_mm1 + s_nm * v_mm1)
                )
                az += s_nm_coeff * (-c_nm * v_m - s_nm * w_m)

        # Scale by GM / Rₑ²
        scale = gm / (re * re)
        ax_ecef = ax * scale
        ay_ecef = ay * scale
        az_ecef = az * scale

        # Rotate ECEF -> ECI
        ax_eci = cos_g * ax_ecef - sin_g * ay_ecef
        ay_eci = sin_g * ax_ecef + cos_g * ay_ecef
        az_eci = az_ecef

        return (ax_eci, ay_eci, az_eci)
