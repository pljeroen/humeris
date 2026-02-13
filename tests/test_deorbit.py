# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for deorbit compliance assessment."""
import ast
import math
from datetime import datetime, timezone

import pytest

from constellation_generator.domain.atmosphere import DragConfig
from constellation_generator.domain.orbital_mechanics import OrbitalConstants
from constellation_generator.domain.deorbit import (
    DeorbitAssessment,
    DeorbitRegulation,
    assess_deorbit_compliance,
)


EPOCH = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
# Moderate B_c satellite (B_c = 2.2*4/400 = 0.022 m²/kg)
TYPICAL_DRAG = DragConfig(cd=2.2, area_m2=4.0, mass_kg=400.0)


# ── DeorbitRegulation enum ────────────────────────────────────────

class TestDeorbitRegulation:

    def test_values(self):
        """DeorbitRegulation has FCC_5YEAR and ESA_25YEAR values."""
        assert DeorbitRegulation.FCC_5YEAR.value == "fcc_5year"
        assert DeorbitRegulation.ESA_25YEAR.value == "esa_25year"


# ── DeorbitAssessment dataclass ───────────────────────────────────

class TestDeorbitAssessment:

    def test_frozen(self):
        """DeorbitAssessment is immutable."""
        a = DeorbitAssessment(
            compliant=True, regulation=DeorbitRegulation.FCC_5YEAR,
            natural_lifetime_days=100.0, threshold_days=1826.25,
            maneuver_required=False, deorbit_delta_v_ms=0.0,
            target_perigee_km=None, propellant_mass_kg=None,
        )
        with pytest.raises(AttributeError):
            a.compliant = False

    def test_fields(self):
        """DeorbitAssessment exposes expected fields."""
        a = DeorbitAssessment(
            compliant=True, regulation=DeorbitRegulation.FCC_5YEAR,
            natural_lifetime_days=100.0, threshold_days=1826.25,
            maneuver_required=False, deorbit_delta_v_ms=0.0,
            target_perigee_km=300.0, propellant_mass_kg=5.0,
        )
        assert a.compliant is True
        assert a.natural_lifetime_days == 100.0
        assert a.target_perigee_km == 300.0


# ── Compliance assessment ─────────────────────────────────────────

class TestAssessDeorbitCompliance:

    def test_300km_naturally_compliant(self):
        """300km orbit with typical B_c: naturally compliant (< 5 years)."""
        result = assess_deorbit_compliance(300, TYPICAL_DRAG, EPOCH)
        assert result.compliant is True
        assert result.maneuver_required is False
        assert result.deorbit_delta_v_ms == 0.0

    def test_800km_not_compliant(self):
        """800km orbit: NOT compliant under FCC 5-year rule."""
        result = assess_deorbit_compliance(800, TYPICAL_DRAG, EPOCH)
        assert result.compliant is False
        assert result.maneuver_required is True

    def test_800km_deorbit_dv_positive(self):
        """800km deorbit delta-V is positive and reasonable."""
        result = assess_deorbit_compliance(800, TYPICAL_DRAG, EPOCH)
        assert result.deorbit_delta_v_ms > 0
        assert result.deorbit_delta_v_ms < 500  # reasonable upper bound

    def test_target_perigee_below_operational(self):
        """Target perigee < operational altitude when maneuver required."""
        result = assess_deorbit_compliance(800, TYPICAL_DRAG, EPOCH)
        assert result.target_perigee_km is not None
        assert result.target_perigee_km < 800

    def test_esa_25year_more_lenient(self):
        """ESA 25-year rule is more lenient → higher orbits compliant."""
        fcc = assess_deorbit_compliance(
            550, TYPICAL_DRAG, EPOCH, regulation=DeorbitRegulation.FCC_5YEAR,
        )
        esa = assess_deorbit_compliance(
            550, TYPICAL_DRAG, EPOCH, regulation=DeorbitRegulation.ESA_25YEAR,
        )
        # ESA threshold is 5x higher, so more likely compliant
        assert esa.threshold_days > fcc.threshold_days

    def test_200km_compliant_zero_dv(self):
        """Very low orbit (200km) naturally compliant, zero dV."""
        result = assess_deorbit_compliance(200, TYPICAL_DRAG, EPOCH)
        assert result.compliant is True
        assert result.deorbit_delta_v_ms == 0.0

    def test_propellant_mass_when_provided(self):
        """Propellant mass computed when isp and mass provided."""
        result = assess_deorbit_compliance(
            800, TYPICAL_DRAG, EPOCH, isp_s=300.0, dry_mass_kg=250.0,
        )
        if result.maneuver_required:
            assert result.propellant_mass_kg is not None
            assert result.propellant_mass_kg > 0

    def test_propellant_mass_none_when_not_provided(self):
        """Propellant mass None when isp/mass not provided."""
        result = assess_deorbit_compliance(800, TYPICAL_DRAG, EPOCH)
        assert result.propellant_mass_kg is None

    def test_higher_altitude_more_dv(self):
        """Higher altitude requires more deorbit dV."""
        result_600 = assess_deorbit_compliance(600, TYPICAL_DRAG, EPOCH)
        result_800 = assess_deorbit_compliance(800, TYPICAL_DRAG, EPOCH)
        if result_600.maneuver_required and result_800.maneuver_required:
            assert result_800.deorbit_delta_v_ms > result_600.deorbit_delta_v_ms


# ── Domain purity ─────────────────────────────────────────────────

class TestDeorbitPurity:

    def test_deorbit_module_pure(self):
        """deorbit.py must only import stdlib modules."""
        import constellation_generator.domain.deorbit as mod

        allowed = {'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('constellation_generator'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'constellation_generator':
                        assert False, f"Disallowed import from '{node.module}'"
