# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""
Test package structure consistency for the humeris namespace.

Verifies that both humeris-core and humeris-pro contribute to the
humeris namespace correctly via PEP 420 implicit namespace packages.
"""
import os
from pathlib import Path


def test_core_domain_modules_importable():
    """All MIT core domain modules are importable."""
    core_modules = [
        "humeris.domain.orbital_mechanics",
        "humeris.domain.constellation",
        "humeris.domain.coordinate_frames",
        "humeris.domain.propagation",
        "humeris.domain.coverage",
        "humeris.domain.access_windows",
        "humeris.domain.ground_track",
        "humeris.domain.observation",
        "humeris.domain.omm",
        "humeris.domain.serialization",
    ]
    import importlib
    for mod_name in core_modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import {mod_name}"


def test_pro_domain_modules_importable():
    """Commercial pro domain modules are importable."""
    pro_modules = [
        "humeris.domain.eclipse",
        "humeris.domain.conjunction",
        "humeris.domain.numerical_propagation",
        "humeris.domain.atmosphere",
    ]
    import importlib
    for mod_name in pro_modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Failed to import {mod_name}"


def test_namespace_shared_correctly():
    """Both core and pro contribute to humeris.domain namespace."""
    import humeris.domain.constellation  # core
    import humeris.domain.eclipse  # pro
    # Both accessible under the same namespace
    assert hasattr(humeris.domain.constellation, 'Satellite')
    assert hasattr(humeris.domain.eclipse, 'is_eclipsed')


def test_no_namespace_init_files():
    """Namespace-level directories must not contain __init__.py."""
    project_root = Path(__file__).parent.parent
    for pkg in ("core", "pro"):
        humeris_dir = project_root / "packages" / pkg / "src" / "humeris"
        assert not (humeris_dir / "__init__.py").exists(), \
            f"humeris/__init__.py exists in {pkg} — breaks namespace"
        domain_dir = humeris_dir / "domain"
        if domain_dir.exists():
            assert not (domain_dir / "__init__.py").exists(), \
                f"humeris/domain/__init__.py exists in {pkg} — breaks namespace"
        adapters_dir = humeris_dir / "adapters"
        if adapters_dir.exists():
            assert not (adapters_dir / "__init__.py").exists(), \
                f"humeris/adapters/__init__.py exists in {pkg} — breaks namespace"
