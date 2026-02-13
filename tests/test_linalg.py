# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Tests for domain/linalg.py — pure linear algebra infrastructure."""
import ast
import math

import pytest

from humeris.domain.linalg import (
    EigenDecomposition,
    DFTResult,
    mat_zeros,
    mat_identity,
    mat_multiply,
    mat_transpose,
    mat_add,
    mat_scale,
    mat_eigenvalues_symmetric,
    mat_determinant,
    mat_inverse,
    mat_trace,
    naive_dft,
)


class TestMatrixOperations:
    def test_identity_multiply(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        i = mat_identity(2)
        result = mat_multiply(i, a)
        for r in range(2):
            for c in range(2):
                assert abs(result[r][c] - a[r][c]) < 1e-10

    def test_transpose_involution(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        att = mat_transpose(mat_transpose(a))
        for r in range(2):
            for c in range(3):
                assert abs(att[r][c] - a[r][c]) < 1e-10

    def test_multiply_dimensions(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 2x3
        b = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # 3x2
        result = mat_multiply(a, b)  # 2x2
        assert len(result) == 2
        assert len(result[0]) == 2


class TestDeterminant:
    def test_determinant_identity(self):
        i = mat_identity(3)
        assert abs(mat_determinant(i) - 1.0) < 1e-10

    def test_determinant_singular(self):
        a = [[1.0, 2.0], [2.0, 4.0]]
        assert abs(mat_determinant(a)) < 1e-10

    def test_determinant_2x2(self):
        a = [[3.0, 8.0], [4.0, 6.0]]
        expected = 3.0 * 6.0 - 8.0 * 4.0  # -14
        assert abs(mat_determinant(a) - expected) < 1e-10


class TestInverse:
    def test_inverse_identity(self):
        i = mat_identity(3)
        inv = mat_inverse(i)
        for r in range(3):
            for c in range(3):
                expected = 1.0 if r == c else 0.0
                assert abs(inv[r][c] - expected) < 1e-10

    def test_inverse_roundtrip(self):
        a = [[4.0, 7.0], [2.0, 6.0]]
        inv = mat_inverse(a)
        product = mat_multiply(a, inv)
        for r in range(2):
            for c in range(2):
                expected = 1.0 if r == c else 0.0
                assert abs(product[r][c] - expected) < 1e-8


class TestEigenvalues:
    def test_eigenvalues_diagonal(self):
        a = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        result = mat_eigenvalues_symmetric(a)
        assert isinstance(result, EigenDecomposition)
        vals = sorted(result.eigenvalues)
        assert abs(vals[0] - 1.0) < 1e-8
        assert abs(vals[1] - 2.0) < 1e-8
        assert abs(vals[2] - 3.0) < 1e-8

    def test_eigenvalues_symmetric_2x2(self):
        # [[2, 1], [1, 2]] has eigenvalues 1 and 3
        a = [[2.0, 1.0], [1.0, 2.0]]
        result = mat_eigenvalues_symmetric(a)
        vals = sorted(result.eigenvalues)
        assert abs(vals[0] - 1.0) < 1e-8
        assert abs(vals[1] - 3.0) < 1e-8

    def test_eigenvalues_sorted(self):
        a = [[5.0, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 1.0]]
        result = mat_eigenvalues_symmetric(a)
        for i in range(len(result.eigenvalues) - 1):
            assert result.eigenvalues[i] <= result.eigenvalues[i + 1] + 1e-10

    def test_eigenvectors_orthogonal(self):
        a = [[2.0, 1.0], [1.0, 2.0]]
        result = mat_eigenvalues_symmetric(a)
        v0 = result.eigenvectors[0]
        v1 = result.eigenvectors[1]
        dot = sum(x * y for x, y in zip(v0, v1))
        assert abs(dot) < 1e-8

    def test_eigendecomposition_reconstruct(self):
        a = [[2.0, 1.0], [1.0, 2.0]]
        result = mat_eigenvalues_symmetric(a)
        n = len(a)
        # A ≈ V·Λ·Vᵀ
        for r in range(n):
            for c in range(n):
                val = 0.0
                for k in range(n):
                    val += result.eigenvalues[k] * result.eigenvectors[k][r] * result.eigenvectors[k][c]
                assert abs(val - a[r][c]) < 1e-8


class TestTrace:
    def test_trace_equals_eigenvalue_sum(self):
        a = [[5.0, 2.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 1.0]]
        tr = mat_trace(a)
        result = mat_eigenvalues_symmetric(a)
        eig_sum = sum(result.eigenvalues)
        assert abs(tr - eig_sum) < 1e-8


class TestDFT:
    def test_dft_constant_signal(self):
        signal = [1.0] * 16
        result = naive_dft(signal, 1.0)
        assert isinstance(result, DFTResult)
        # DC component should dominate
        assert result.magnitudes[0] > 0.9

    def test_dft_pure_sine(self):
        n = 64
        f = 5.0  # Hz
        sr = 64.0  # sample rate
        signal = [math.sin(2 * math.pi * f * i / sr) for i in range(n)]
        result = naive_dft(signal, sr)
        # Find peak (excluding DC)
        peak_idx = max(range(1, len(result.magnitudes) // 2), key=lambda k: result.magnitudes[k])
        peak_freq = result.frequencies_hz[peak_idx]
        assert abs(peak_freq - f) < sr / n + 0.1

    def test_dft_parseval(self):
        signal = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        result = naive_dft(signal, 1.0)
        time_energy = sum(x ** 2 for x in signal)
        freq_energy = sum(m ** 2 for m in result.magnitudes) * len(signal)
        assert abs(time_energy - freq_energy) < 1e-6


class TestLinalgPurity:
    def test_module_pure(self):
        import humeris.domain.linalg as mod
        source = ast.parse(open(mod.__file__).read())
        for node in ast.walk(source):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                else:
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                assert top in {"math", "numpy", "dataclasses", "typing", "humeris", "__future__"}, (
                    f"Forbidden import: {top}"
                )
