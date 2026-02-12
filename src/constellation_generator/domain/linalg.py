# Copyright (c) 2026 Jeroen. All rights reserved.
"""Pure linear algebra infrastructure.

Matrix operations, Jacobi eigenvalue decomposition for symmetric matrices,
LU-based determinant, Gauss-Jordan inverse, and naive DFT.

No external dependencies — only stdlib math/dataclasses.
"""
import math
from dataclasses import dataclass
from typing import List

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
    k = len(b[0])
    result = mat_zeros(n, k)
    for i in range(n):
        for j in range(k):
            s = 0.0
            for p in range(m):
                s += a[i][p] * b[p][j]
            result[i][j] = s
    return result


def mat_transpose(a: Matrix) -> Matrix:
    """Transpose matrix a."""
    n = len(a)
    if n == 0:
        return []
    m = len(a[0])
    return [[a[i][j] for i in range(n)] for j in range(m)]


def mat_add(a: Matrix, b: Matrix) -> Matrix:
    """Element-wise addition of matrices a and b."""
    n = len(a)
    m = len(a[0])
    return [[a[i][j] + b[i][j] for j in range(m)] for i in range(n)]


def mat_scale(a: Matrix, scalar: float) -> Matrix:
    """Scale all elements of matrix a by scalar."""
    n = len(a)
    m = len(a[0])
    return [[a[i][j] * scalar for j in range(m)] for i in range(n)]


def mat_trace(a: Matrix) -> float:
    """Trace of a square matrix (sum of diagonal elements)."""
    return sum(a[i][i] for i in range(len(a)))


def mat_determinant(a: Matrix) -> float:
    """Determinant of an NxN matrix via LU decomposition with partial pivoting."""
    n = len(a)
    if n == 1:
        return a[0][0]
    if n == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]

    # Copy matrix
    lu = [[a[i][j] for j in range(n)] for i in range(n)]
    det = 1.0

    for col in range(n):
        # Partial pivoting
        max_val = abs(lu[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(lu[row][col]) > max_val:
                max_val = abs(lu[row][col])
                max_row = row
        if max_row != col:
            lu[col], lu[max_row] = lu[max_row], lu[col]
            det *= -1.0

        pivot = lu[col][col]
        if abs(pivot) < 1e-15:
            return 0.0
        det *= pivot

        for row in range(col + 1, n):
            factor = lu[row][col] / pivot
            for j in range(col + 1, n):
                lu[row][j] -= factor * lu[col][j]

    return det


def mat_inverse(a: Matrix) -> Matrix:
    """Inverse of an NxN matrix via Gauss-Jordan elimination."""
    n = len(a)
    # Augmented matrix [A | I]
    aug = [[0.0] * (2 * n) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            aug[i][j] = a[i][j]
        aug[i][n + i] = 1.0

    for col in range(n):
        # Partial pivoting
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-15:
            return mat_identity(n)  # Singular fallback

        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    return [[aug[i][n + j] for j in range(n)] for i in range(n)]


def mat_eigenvalues_symmetric(
    a: Matrix,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> EigenDecomposition:
    """Jacobi eigenvalue algorithm for real symmetric matrices.

    Returns eigenvalues sorted ascending and corresponding eigenvectors.
    """
    n = len(a)
    if n == 0:
        return EigenDecomposition(eigenvalues=(), eigenvectors=())
    if n == 1:
        return EigenDecomposition(
            eigenvalues=(a[0][0],),
            eigenvectors=((1.0,),),
        )

    # Work on a copy
    s = [[a[i][j] for j in range(n)] for i in range(n)]
    # Accumulate rotations for eigenvectors
    v = mat_identity(n)

    for _ in range(max_iterations * n):
        # Find largest off-diagonal element
        max_off = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(s[i][j]) > max_off:
                    max_off = abs(s[i][j])
                    p, q = i, j

        if max_off < tolerance:
            break

        # Compute rotation angle
        if abs(s[p][p] - s[q][q]) < 1e-15:
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * s[p][q], s[p][p] - s[q][q])

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Apply Givens rotation: S' = G^T S G
        # Store rows p and q
        row_p = [s[p][j] for j in range(n)]
        row_q = [s[q][j] for j in range(n)]

        for j in range(n):
            s[p][j] = cos_t * row_p[j] + sin_t * row_q[j]
            s[q][j] = -sin_t * row_p[j] + cos_t * row_q[j]

        col_p = [s[i][p] for i in range(n)]
        col_q = [s[i][q] for i in range(n)]

        for i in range(n):
            s[i][p] = cos_t * col_p[i] + sin_t * col_q[i]
            s[i][q] = -sin_t * col_p[i] + cos_t * col_q[i]

        # Accumulate eigenvectors
        for i in range(n):
            vip = v[i][p]
            viq = v[i][q]
            v[i][p] = cos_t * vip + sin_t * viq
            v[i][q] = -sin_t * vip + cos_t * viq

    # Extract eigenvalues and sort
    eigenvalues = [s[i][i] for i in range(n)]
    # Eigenvectors are columns of v
    eigenvectors = [[v[i][k] for i in range(n)] for k in range(n)]

    # Sort by eigenvalue ascending
    indices = sorted(range(n), key=lambda i: eigenvalues[i])
    sorted_vals = tuple(eigenvalues[i] for i in indices)
    sorted_vecs = tuple(tuple(eigenvectors[i]) for i in indices)

    return EigenDecomposition(eigenvalues=sorted_vals, eigenvectors=sorted_vecs)


def naive_dft(signal: list, sample_rate_hz: float) -> DFTResult:
    """Naive O(N^2) discrete Fourier transform.

    C[k] = sum_j x[j] * exp(-2*pi*i*j*k/N)
    magnitude[k] = |C[k]| / N
    """
    n = len(signal)
    if n == 0:
        return DFTResult(frequencies_hz=(), magnitudes=(), phases_rad=())

    magnitudes = []
    phases = []
    frequencies = []

    for k in range(n):
        re = 0.0
        im = 0.0
        for j in range(n):
            angle = 2.0 * math.pi * j * k / n
            re += signal[j] * math.cos(angle)
            im -= signal[j] * math.sin(angle)
        mag = math.sqrt(re * re + im * im) / n
        phase = math.atan2(im, re)
        freq = k * sample_rate_hz / n

        magnitudes.append(mag)
        phases.append(phase)
        frequencies.append(freq)

    return DFTResult(
        frequencies_hz=tuple(frequencies),
        magnitudes=tuple(magnitudes),
        phases_rad=tuple(phases),
    )
