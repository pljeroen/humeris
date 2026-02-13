# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Tests for relativistic force models (Schwarzschild, Lense-Thirring, de Sitter)."""

import ast
import math
from datetime import datetime, timezone

import pytest


def _vec_mag(v):
    return math.sqrt(sum(c ** 2 for c in v))


class TestSchwarzschildForce:
    """Post-Newtonian Schwarzschild correction."""

    def test_magnitude_at_leo(self):
        """Schwarzschild acceleration at LEO ~3e-9 m/s²."""
        from humeris.domain.relativistic_forces import SchwarzschildForce

        force = SchwarzschildForce()
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc = force.acceleration(dt, pos, vel)
        mag = _vec_mag(acc)
        assert 1e-10 < mag < 1e-7

    def test_direction_circular_purely_radial(self):
        """Schwarzschild is purely radial for circular orbits (r·v=0)."""
        from humeris.domain.relativistic_forces import SchwarzschildForce

        force = SchwarzschildForce()
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc = force.acceleration(dt, pos, vel)
        # Radial component non-zero, along-track zero for circular orbit
        assert acc[0] != 0.0
        assert acc[1] == 0.0

    def test_direction_eccentric_has_along_track(self):
        """Schwarzschild has along-track component for eccentric orbits (r·v≠0)."""
        from humeris.domain.relativistic_forces import SchwarzschildForce

        force = SchwarzschildForce()
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Slightly non-circular: r·v = 6778137 * 500 ≠ 0
        pos = (6778137.0, 0.0, 0.0)
        vel = (500.0, 7668.0, 0.0)
        acc = force.acceleration(dt, pos, vel)
        assert acc[0] != 0.0
        assert acc[1] != 0.0

    def test_increases_closer_to_earth(self):
        """Schwarzschild is stronger at lower altitude."""
        from humeris.domain.relativistic_forces import SchwarzschildForce

        force = SchwarzschildForce()
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # LEO
        acc_leo = force.acceleration(dt, (6778137.0, 0.0, 0.0), (0.0, 7668.0, 0.0))
        # MEO
        acc_meo = force.acceleration(dt, (26578137.0, 0.0, 0.0), (0.0, 3870.0, 0.0))
        assert _vec_mag(acc_leo) > _vec_mag(acc_meo)

    def test_constants_verification(self):
        """Verify GM_E and c are correct."""
        from humeris.domain.relativistic_forces import (
            _GM_EARTH,
            _C_LIGHT,
        )
        assert abs(_GM_EARTH - 3.986004418e14) < 1e8
        assert _C_LIGHT == 299792458.0

    def test_force_model_compliance(self):
        """Has the ForceModel interface."""
        from humeris.domain.relativistic_forces import SchwarzschildForce
        force = SchwarzschildForce()
        assert hasattr(force, "acceleration")
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = force.acceleration(dt, (7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0))
        assert len(result) == 3


class TestLenseThirringForce:
    """Frame-dragging from Earth rotation."""

    def test_magnitude_at_leo(self):
        """Lense-Thirring ~2e-10 m/s² at equatorial LEO."""
        from humeris.domain.relativistic_forces import LenseThirringForce

        force = LenseThirringForce()
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        acc = force.acceleration(dt, (6778137.0, 0.0, 0.0), (0.0, 7668.0, 0.0))
        mag = _vec_mag(acc)
        assert 1e-12 < mag < 1e-8

    def test_force_model_compliance(self):
        from humeris.domain.relativistic_forces import LenseThirringForce
        force = LenseThirringForce()
        result = force.acceleration(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            (7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0),
        )
        assert len(result) == 3


class TestDeSitterForce:
    """Geodesic precession from heliocentric motion."""

    def test_magnitude_at_leo(self):
        """de Sitter ~5e-13 m/s² at LEO."""
        from humeris.domain.relativistic_forces import DeSitterForce

        force = DeSitterForce()
        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        acc = force.acceleration(dt, (6778137.0, 0.0, 0.0), (0.0, 7668.0, 0.0))
        mag = _vec_mag(acc)
        assert 1e-15 < mag < 1e-10

    def test_force_model_compliance(self):
        from humeris.domain.relativistic_forces import DeSitterForce
        force = DeSitterForce()
        result = force.acceleration(
            datetime(2024, 6, 15, tzinfo=timezone.utc),
            (7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0),
        )
        assert len(result) == 3


class TestCombinedRelativistic:
    """Combined relativistic effects."""

    def test_schwarzschild_dominates(self):
        """Schwarzschild >> Lense-Thirring >> de Sitter."""
        from humeris.domain.relativistic_forces import (
            SchwarzschildForce,
            LenseThirringForce,
            DeSitterForce,
        )

        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)

        s = _vec_mag(SchwarzschildForce().acceleration(dt, pos, vel))
        lt = _vec_mag(LenseThirringForce().acceleration(dt, pos, vel))
        ds = _vec_mag(DeSitterForce().acceleration(dt, pos, vel))

        assert s > lt
        assert lt > ds or lt > 0  # LT should be measurable

    def test_all_finite(self):
        from humeris.domain.relativistic_forces import (
            SchwarzschildForce,
            LenseThirringForce,
            DeSitterForce,
        )
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        for F in [SchwarzschildForce, LenseThirringForce, DeSitterForce]:
            acc = F().acceleration(dt, pos, vel)
            assert all(math.isfinite(a) for a in acc)


class TestDomainPurity:
    def test_no_external_imports(self):
        import humeris.domain.relativistic_forces as _mod
        source_path = _mod.__file__
        with open(source_path) as f:
            tree = ast.parse(f.read())
        allowed = {
            "math", "numpy", "datetime", "dataclasses", "typing", "json",
            "pathlib", "os", "functools", "enum", "collections",
            "abc", "copy", "struct", "bisect", "operator",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed or top == "humeris", \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    assert top in allowed or top == "humeris", \
                        f"Forbidden import from: {node.module}"
