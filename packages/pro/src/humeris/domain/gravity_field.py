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

    NumPy-vectorized: V/W vertical recursion uses contiguous slice
    operations; accumulation uses precomputed flat index arrays for
    single-pass gather-multiply-sum.
    """

    def __init__(self, model: GravityFieldModel) -> None:
        self._model = model
        n_max = model.max_degree
        n_ext = n_max + 1

        # Convert tuples to NumPy arrays for vectorized access
        self._c_bar = np.asarray(model.c_bar)
        self._s_bar = np.asarray(model.s_bar)
        self._diag = np.asarray(model._diag_coeff)
        self._va = np.asarray(model._vert_a)
        self._vb = np.asarray(model._vert_b)

        # Precompute V/W recursion indices per degree.
        # For degree n, vertical recursion computes m = 0..n-2
        # using contiguous slices into the flat triangular arrays.
        # Diagonal/sub-diagonal indices precomputed as Python ints.
        self._diag_idx: list[tuple[int, int]] = []     # (idx_nn, idx_prev)
        self._diag_alpha: list[float] = []
        self._subdiag: list[tuple[int, int, float] | None] = []  # (idx_nm, idx_n1m, beta)
        self._vert_slices: list[tuple[int, int, int, int] | None] = []
        for n in range(1, n_ext + 1):
            self._diag_idx.append((
                n * (n + 1) // 2 + n,
                (n - 1) * n // 2 + (n - 1),
            ))
            self._diag_alpha.append(float(self._diag[n]))

            if n >= 2:
                idx_nm = n * (n + 1) // 2 + (n - 1)
                idx_n1m = (n - 1) * n // 2 + (n - 1)
                self._subdiag.append((idx_nm, idx_n1m, float(self._va[idx_nm])))
            else:
                self._subdiag.append(None)

            if n >= 2:
                count = n - 1
                self._vert_slices.append((
                    n * (n + 1) // 2,
                    (n - 1) * n // 2,
                    (n - 2) * (n - 1) // 2,
                    count,
                ))
            else:
                self._vert_slices.append(None)

        # Preallocate V/W buffers (reused across calls)
        vw_size = (n_ext + 1) * (n_ext + 2) // 2
        self._v_buf = np.zeros(vw_size)
        self._w_buf = np.zeros(vw_size)

        # Precompute flat index arrays for accumulation (m=0 terms).
        ns_0 = np.arange(2, n_max + 1)
        self._idx_n0 = (ns_0 * (ns_0 + 1) // 2).astype(np.intp)
        self._idx_np1_1 = ((ns_0 + 1) * (ns_0 + 2) // 2 + 1).astype(np.intp)
        self._idx_np1_0 = ((ns_0 + 1) * (ns_0 + 2) // 2).astype(np.intp)
        # Gather coefficients for m=0 terms
        self._c0 = self._c_bar[self._idx_n0]
        self._p0 = np.asarray(model._acc_p)[self._idx_n0]
        self._s0 = np.asarray(model._acc_s)[self._idx_n0]

        # Precompute flat index arrays for accumulation (m>=1 terms).
        # Enumerate all (n, m) pairs with n=2..N, m=1..n.
        idx_nm_list = []
        idx_np1_mp1_list = []
        idx_np1_mm1_list = []
        idx_np1_m_list = []
        for n in range(2, n_max + 1):
            ms = np.arange(1, n + 1)
            idx_nm_list.append(n * (n + 1) // 2 + ms)
            idx_np1_mp1_list.append((n + 1) * (n + 2) // 2 + ms + 1)
            idx_np1_mm1_list.append((n + 1) * (n + 2) // 2 + ms - 1)
            idx_np1_m_list.append((n + 1) * (n + 2) // 2 + ms)

        self._idx_nm = np.concatenate(idx_nm_list).astype(np.intp)
        self._idx_np1_mp1 = np.concatenate(idx_np1_mp1_list).astype(np.intp)
        self._idx_np1_mm1 = np.concatenate(idx_np1_mm1_list).astype(np.intp)
        self._idx_np1_m = np.concatenate(idx_np1_m_list).astype(np.intp)

        # Pre-gather coefficients for m>=1 terms
        acc_p_arr = np.asarray(model._acc_p)
        acc_q_arr = np.asarray(model._acc_q)
        acc_s_arr = np.asarray(model._acc_s)
        self._c_m1 = self._c_bar[self._idx_nm]
        self._s_m1 = self._s_bar[self._idx_nm]
        self._p_m1 = acc_p_arr[self._idx_nm]
        self._q_m1 = acc_q_arr[self._idx_nm]
        self._s_coeff_m1 = acc_s_arr[self._idx_nm]

    def acceleration(
        self,
        epoch: datetime,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Compute perturbation acceleration in ECI frame.

        1. Rotate ECI -> ECEF via GMST
        2. Compute V̄/W̄ via Cunningham recursion (singularity-free)
        3. Sum perturbation acceleration in ECEF (vectorized)
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
        n_ext = m.max_degree + 1

        # Cunningham auxiliary variables
        re_r2 = re / r2
        xr = x * re_r2
        yr = y * re_r2
        zr = z * re_r2
        rr = re * re / r2

        # V̄/W̄ arrays (preallocated, zeroed each call)
        v = self._v_buf
        w = self._w_buf
        v[:] = 0.0
        w[:] = 0.0
        v[0] = re / r

        va = self._va
        vb = self._vb
        diag_idx = self._diag_idx
        diag_alpha = self._diag_alpha
        subdiag = self._subdiag
        vert_slices = self._vert_slices

        # Build V̄/W̄ to degree n_ext (outer loop sequential, inner vectorized)
        for i in range(n_ext):
            # Diagonal: V̄_nn, W̄_nn (Python floats for speed)
            idx_nn, idx_prev = diag_idx[i]
            alpha = diag_alpha[i]
            vp = float(v[idx_prev])
            wp = float(w[idx_prev])
            v[idx_nn] = alpha * (xr * vp - yr * wp)
            w[idx_nn] = alpha * (xr * wp + yr * vp)

            # Sub-diagonal m=n-1
            sd = subdiag[i]
            if sd is not None:
                idx_nm, idx_n1m, beta = sd
                v[idx_nm] = beta * zr * float(v[idx_n1m])
                w[idx_nm] = beta * zr * float(w[idx_n1m])

            # Vertical recursion m=0..n-2 (vectorized contiguous slice)
            sl = vert_slices[i]
            if sl is not None:
                s0, s1, s2, cnt = sl
                b = va[s0:s0 + cnt]
                g = vb[s0:s0 + cnt]
                v[s0:s0 + cnt] = b * zr * v[s1:s1 + cnt] - g * rr * v[s2:s2 + cnt]
                w[s0:s0 + cnt] = b * zr * w[s1:s1 + cnt] - g * rr * w[s2:s2 + cnt]

        # Accumulate perturbation acceleration (fully vectorized)

        # m=0 terms
        v_p1 = v[self._idx_np1_1]
        w_p1 = w[self._idx_np1_1]
        v_p0 = v[self._idx_np1_0]
        cp = self._c0 * self._p0
        ax = float(np.sum(-cp * v_p1))
        ay = float(np.sum(-cp * w_p1))
        az = float(np.sum(-self._c0 * self._s0 * v_p0))

        # m>=1 terms (single vectorized pass over all (n,m) pairs)
        v_mp1 = v[self._idx_np1_mp1]
        w_mp1 = w[self._idx_np1_mp1]
        v_mm1 = v[self._idx_np1_mm1]
        w_mm1 = w[self._idx_np1_mm1]
        v_m = v[self._idx_np1_m]
        w_m = w[self._idx_np1_m]

        c = self._c_m1
        s = self._s_m1
        p = self._p_m1
        q = self._q_m1

        ax += float(np.sum(0.5 * (
            -c * p * v_mp1 - s * p * w_mp1
            + q * (c * v_mm1 + s * w_mm1)
        )))
        ay += float(np.sum(0.5 * (
            -c * p * w_mp1 + s * p * v_mp1
            + q * (-c * w_mm1 + s * v_mm1)
        )))
        az += float(np.sum(self._s_coeff_m1 * (-c * v_m - s * w_m)))

        # Scale by GM / Rₑ²
        scale = m.gm / (re * re)
        ax_ecef = ax * scale
        ay_ecef = ay * scale
        az_ecef = az * scale

        # Rotate ECEF -> ECI
        ax_eci = cos_g * ax_ecef - sin_g * ay_ecef
        ay_eci = sin_g * ax_ecef + cos_g * ay_ecef
        az_eci = az_ecef

        return (ax_eci, ay_eci, az_eci)
