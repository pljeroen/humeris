# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Linear algebra infrastructure backed by NumPy.

Matrix operations, eigenvalue decomposition for symmetric matrices,
determinant, inverse, and FFT-based DFT.

External dependency: numpy (allowed in domain layer).
"""
import math
from dataclasses import dataclass
from typing import List

import numpy as np

Matrix = List[List[float]]


@dataclass(frozen=True)
class EigenDecomposition:
    """Result of symmetric eigenvalue decomposition."""
    eigenvalues: tuple
    eigenvectors: tuple


@dataclass(frozen=True)
class DFTResult:
    """Result of discrete Fourier transform."""
    frequencies_hz: tuple
    magnitudes: tuple
    phases_rad: tuple


def mat_zeros(n: int, m: int) -> Matrix:
    """Create an NxM zero matrix."""
    return [[0.0] * m for _ in range(n)]


def mat_identity(n: int) -> Matrix:
    """Create an NxN identity matrix."""
    result = mat_zeros(n, n)
    for i in range(n):
        result[i][i] = 1.0
    return result


def mat_multiply(a: Matrix, b: Matrix) -> Matrix:
    """Multiply matrices a (NxM) and b (MxK) → NxK."""
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return []
    result = np.asarray(a) @ np.asarray(b)
    return result.tolist()


def mat_transpose(a: Matrix) -> Matrix:
    """Transpose matrix a."""
    n = len(a)
    if n == 0:
        return []
    return np.asarray(a).T.tolist()


def mat_add(a: Matrix, b: Matrix) -> Matrix:
    """Element-wise addition of matrices a and b."""
    return (np.asarray(a) + np.asarray(b)).tolist()


def mat_scale(a: Matrix, scalar: float) -> Matrix:
    """Scale all elements of matrix a by scalar."""
    return (np.asarray(a) * scalar).tolist()


def mat_trace(a: Matrix) -> float:
    """Trace of a square matrix (sum of diagonal elements)."""
    return float(np.trace(np.asarray(a)))


def mat_determinant(a: Matrix) -> float:
    """Determinant of an NxN matrix."""
    n = len(a)
    if n == 1:
        return a[0][0]
    if n == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return float(np.linalg.det(np.asarray(a)))


def mat_inverse(a: Matrix) -> Matrix:
    """Inverse of an NxN matrix."""
    n = len(a)
    arr = np.asarray(a, dtype=float)
    if abs(np.linalg.det(arr)) < 1e-15:
        return mat_identity(n)  # Singular fallback
    return np.linalg.inv(arr).tolist()


def mat_eigenvalues_symmetric(
    a: Matrix,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> EigenDecomposition:
    """Eigenvalue decomposition for real symmetric matrices.

    Returns eigenvalues sorted ascending and corresponding eigenvectors.
    Uses numpy.linalg.eigh (guaranteed real eigenvalues for symmetric input).
    """
    n = len(a)
    if n == 0:
        return EigenDecomposition(eigenvalues=(), eigenvectors=())
    if n == 1:
        return EigenDecomposition(
            eigenvalues=(a[0][0],),
            eigenvectors=((1.0,),),
        )

    arr = np.asarray(a, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(arr)

    # eigh returns eigenvalues ascending and eigenvectors as columns
    sorted_vals = tuple(float(v) for v in eigenvalues)
    sorted_vecs = tuple(
        tuple(float(eigenvectors[i, k]) for i in range(n))
        for k in range(n)
    )

    return EigenDecomposition(eigenvalues=sorted_vals, eigenvectors=sorted_vecs)


def naive_dft(signal: list, sample_rate_hz: float) -> DFTResult:
    """Discrete Fourier transform via NumPy FFT.

    magnitude[k] = |C[k]| / N
    """
    n = len(signal)
    if n == 0:
        return DFTResult(frequencies_hz=(), magnitudes=(), phases_rad=())

    arr = np.asarray(signal, dtype=float)
    spectrum = np.fft.fft(arr)

    magnitudes = np.abs(spectrum) / n
    phases = np.angle(spectrum)
    # Match original behavior: frequencies = k * sample_rate / N for k=0..N-1
    frequencies = np.arange(n) * (sample_rate_hz / n)

    return DFTResult(
        frequencies_hz=tuple(float(f) for f in frequencies),
        magnitudes=tuple(float(m) for m in magnitudes),
        phases_rad=tuple(float(p) for p in phases),
    )
