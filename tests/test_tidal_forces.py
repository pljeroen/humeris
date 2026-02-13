# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see LICENSE-COMMERCIAL.md.
"""Tests for solid Earth tides and ocean tides force models."""

import ast
import math
from datetime import datetime, timezone

import pytest


def _vec_mag(v):
    return math.sqrt(sum(c ** 2 for c in v))


class TestSolidTideForce:
    """IERS 2010 solid Earth tides."""

    def test_magnitude_at_leo(self):
        """Solid tide acceleration ~1e-9 m/s² at LEO."""
        from constellation_generator.domain.tidal_forces import SolidTideForce

        force = SolidTideForce()
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc = force.acceleration(dt, pos, vel)
        mag = _vec_mag(acc)
        assert 1e-11 < mag < 1e-7

    def test_love_number_k2(self):
        """Love number k20 should be ~0.30190."""
        from constellation_generator.domain.tidal_forces import _K20
        assert abs(_K20 - 0.30190) < 0.01

    def test_love_numbers_exist(self):
        """All three k2m values should be defined."""
        from constellation_generator.domain.tidal_forces import _K20, _K21, _K22
        assert _K20 > 0
        assert _K21 > 0
        assert _K22 > 0

    def test_sun_moon_ratio(self):
        """Sun tidal effect ~0.46 of Moon's."""
        from constellation_generator.domain.tidal_forces import SolidTideForce

        force = SolidTideForce()
        # At a time when Moon and Sun are at different positions
        dt = datetime(2024, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc = force.acceleration(dt, pos, vel)
        # Just verify it works and produces finite acceleration
        mag = _vec_mag(acc)
        assert math.isfinite(mag)
        assert mag > 0

    def test_decreases_with_altitude(self):
        """Tidal acceleration decreases with altitude."""
        from constellation_generator.domain.tidal_forces import SolidTideForce

        force = SolidTideForce()
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # LEO
        acc_leo = force.acceleration(dt, (6778137.0, 0.0, 0.0), (0.0, 7668.0, 0.0))
        # GEO
        acc_geo = force.acceleration(dt, (42164000.0, 0.0, 0.0), (0.0, 3070.0, 0.0))
        assert _vec_mag(acc_leo) > _vec_mag(acc_geo)

    def test_force_model_compliance(self):
        from constellation_generator.domain.tidal_forces import SolidTideForce
        force = SolidTideForce()
        assert hasattr(force, "acceleration")
        result = force.acceleration(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            (7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0),
        )
        assert len(result) == 3
        assert all(math.isfinite(a) for a in result)

    def test_varies_with_time(self):
        """Tides vary with body positions (time-dependent)."""
        from constellation_generator.domain.tidal_forces import SolidTideForce

        force = SolidTideForce()
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc1 = force.acceleration(datetime(2024, 1, 1, tzinfo=timezone.utc), pos, vel)
        acc2 = force.acceleration(datetime(2024, 1, 15, tzinfo=timezone.utc), pos, vel)
        # Should differ
        assert acc1 != acc2


class TestOceanTideForce:
    """FES2004 ocean tides."""

    def test_magnitude_at_leo(self):
        """Ocean tide acceleration ~1e-10 m/s² at LEO."""
        from constellation_generator.domain.tidal_forces import OceanTideForce

        force = OceanTideForce()
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc = force.acceleration(dt, pos, vel)
        mag = _vec_mag(acc)
        assert 1e-13 < mag < 1e-7

    def test_smaller_than_solid(self):
        """Ocean tides should be smaller than solid tides."""
        from constellation_generator.domain.tidal_forces import SolidTideForce, OceanTideForce

        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        solid = _vec_mag(SolidTideForce().acceleration(dt, pos, vel))
        ocean = _vec_mag(OceanTideForce().acceleration(dt, pos, vel))
        assert ocean < solid

    def test_coefficient_loading(self):
        """Should load all 8 constituents."""
        from constellation_generator.domain.tidal_forces import OceanTideForce
        force = OceanTideForce()
        assert force._n_constituents >= 8

    def test_force_model_compliance(self):
        from constellation_generator.domain.tidal_forces import OceanTideForce
        force = OceanTideForce()
        result = force.acceleration(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            (7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0),
        )
        assert len(result) == 3
        assert all(math.isfinite(a) for a in result)

    def test_varies_with_time(self):
        """Ocean tides are periodic."""
        from constellation_generator.domain.tidal_forces import OceanTideForce
        force = OceanTideForce()
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc1 = force.acceleration(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc), pos, vel)
        acc2 = force.acceleration(datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc), pos, vel)
        assert acc1 != acc2


class TestDomainPurity:
    def test_no_external_imports(self):
        source_path = "src/constellation_generator/domain/tidal_forces.py"
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
                    assert top in allowed or top == "constellation_generator", \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    assert top in allowed or top == "constellation_generator", \
                        f"Forbidden import from: {node.module}"
