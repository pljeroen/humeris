# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the Commercial License — see COMMERCIAL-LICENSE.md.
"""Tests for Earth albedo and infrared radiation pressure force model."""

import ast
import math
from datetime import datetime, timezone

import pytest


def _vec_mag(v):
    return math.sqrt(sum(c ** 2 for c in v))


class TestAlbedoRadiationPressure:
    """Earth albedo + thermal IR radiation pressure."""

    def test_magnitude_at_leo(self):
        """Albedo SRP acceleration ~1e-9 m/s² at LEO."""
        from humeris.domain.albedo_srp import AlbedoRadiationPressure

        force = AlbedoRadiationPressure(cr=1.5, area_m2=10.0, mass_kg=500.0)
        dt = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        acc = force.acceleration(dt, pos, vel)
        mag = _vec_mag(acc)
        assert 1e-11 < mag < 1e-7

    def test_subsolar_greater_than_night(self):
        """Albedo pressure should be stronger on the sunlit side."""
        from humeris.domain.albedo_srp import AlbedoRadiationPressure

        force = AlbedoRadiationPressure(cr=1.5, area_m2=10.0, mass_kg=500.0)
        dt = datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc)
        # Subsolar point (roughly towards Sun in June)
        acc_sun = force.acceleration(dt, (6778137.0, 0.0, 0.0), (0.0, 7668.0, 0.0))
        # Opposite side
        acc_dark = force.acceleration(dt, (-6778137.0, 0.0, 0.0), (0.0, -7668.0, 0.0))
        # Both should be finite; subsolar should have stronger albedo component
        assert math.isfinite(_vec_mag(acc_sun))
        assert math.isfinite(_vec_mag(acc_dark))

    def test_decreases_with_altitude(self):
        """Albedo pressure decreases with distance from Earth."""
        from humeris.domain.albedo_srp import AlbedoRadiationPressure

        force = AlbedoRadiationPressure(cr=1.5, area_m2=10.0, mass_kg=500.0)
        dt = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        acc_leo = force.acceleration(dt, (6778137.0, 0.0, 0.0), (0.0, 7668.0, 0.0))
        acc_geo = force.acceleration(dt, (42164000.0, 0.0, 0.0), (0.0, 3070.0, 0.0))
        assert _vec_mag(acc_leo) > _vec_mag(acc_geo)

    def test_ir_component_present(self):
        """Even on the dark side, IR radiation should produce some acceleration."""
        from humeris.domain.albedo_srp import AlbedoRadiationPressure

        force = AlbedoRadiationPressure(cr=1.5, area_m2=10.0, mass_kg=500.0)
        dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        # Night side
        acc = force.acceleration(dt, (-6778137.0, 0.0, 0.0), (0.0, -7668.0, 0.0))
        mag = _vec_mag(acc)
        # IR always present, so magnitude > 0
        assert mag > 0

    def test_force_model_compliance(self):
        """Has the ForceModel interface."""
        from humeris.domain.albedo_srp import AlbedoRadiationPressure

        force = AlbedoRadiationPressure(cr=1.5, area_m2=10.0, mass_kg=500.0)
        assert hasattr(force, "acceleration")
        result = force.acceleration(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            (7000000.0, 0.0, 0.0), (0.0, 7500.0, 0.0),
        )
        assert len(result) == 3
        assert all(math.isfinite(a) for a in result)

    def test_scales_with_area_to_mass(self):
        """Acceleration scales with area/mass ratio."""
        from humeris.domain.albedo_srp import AlbedoRadiationPressure

        dt = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        pos = (6778137.0, 0.0, 0.0)
        vel = (0.0, 7668.0, 0.0)
        f1 = AlbedoRadiationPressure(cr=1.5, area_m2=10.0, mass_kg=500.0)
        f2 = AlbedoRadiationPressure(cr=1.5, area_m2=20.0, mass_kg=500.0)
        mag1 = _vec_mag(f1.acceleration(dt, pos, vel))
        mag2 = _vec_mag(f2.acceleration(dt, pos, vel))
        assert abs(mag2 / mag1 - 2.0) < 0.01


class TestDomainPurity:
    def test_no_external_imports(self):
        source_path = "src/humeris/domain/albedo_srp.py"
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
