# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""GMAT-mirror parity tests (no GMAT runtime dependency)."""
from __future__ import annotations

from pathlib import Path

import pytest

from humeris.adapters.gmat_mirror import (
    compare_against_gmat,
    find_gmat_run_dir,
    load_gmat_case_values,
    run_humeris_mirror,
)


def test_humeris_mirror_runs_and_is_physically_consistent():
    out = run_humeris_mirror()

    basic = out["basic_leo_two_body"]
    assert abs(basic["elapsedSecs"] - 5400.0) < 1e-9
    assert abs(basic["endSMA"] - basic["startSMA"]) < 0.1
    assert abs(basic["endECC"] - basic["startECC"]) < 1e-6

    j2 = out["advanced_j2_raan_drift"]
    assert abs(j2["elapsedDays"] - 7.0) < 1e-9
    assert 0.01 < j2["raanDriftDeg"] < 30.0
    assert 0.0 <= j2["startECC"] < 1.0
    assert 0.0 <= j2["endECC"] < 1.0

    ou = out["advanced_oumuamua_hyperbolic"]
    assert abs(ou["elapsedDays"] - 120.0) < 1e-9
    assert ou["startECC"] > 1.0
    assert ou["endECC"] > 1.0
    assert abs(ou["endRMAG"] - ou["startRMAG"]) > 1000.0


def test_compare_against_local_gmat_artifacts_if_available():
    gmat_repo = Path("/home/jeroen/gmat")
    if not gmat_repo.exists():
        pytest.skip("Local GMAT testsuite repo not found at /home/jeroen/gmat")

    try:
        run_dir = find_gmat_run_dir(gmat_repo)
    except FileNotFoundError:
        pytest.skip("No GMAT run artifacts found")

    gmat_values = load_gmat_case_values(run_dir)
    humeris_values = run_humeris_mirror()
    comparison = compare_against_gmat(gmat_values, humeris_values)

    assert comparison["status"] in {"pass", "fail"}
    assert len(comparison["cases"]) == 3
    for row in comparison["cases"]:
        assert row["status"] in {"pass", "fail"}
        assert row["metrics"]

