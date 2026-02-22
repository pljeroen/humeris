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


def test_spherical_harmonic_gravity_includes_central_term():
    """SphericalHarmonicGravity includes -mu/r² — never combine with TwoBodyGravity."""
    from datetime import datetime, timezone
    from humeris.domain.numerical_propagation import (
        SphericalHarmonicGravity,
        TwoBodyGravity,
    )

    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # LEO position in ECI (roughly 7000 km radius, equatorial)
    pos = (7_000_000.0, 0.0, 0.0)
    vel = (0.0, 7_546.0, 0.0)

    tb_acc = TwoBodyGravity().acceleration(epoch, pos, vel)
    shg_acc = SphericalHarmonicGravity(max_degree=2).acceleration(epoch, pos, vel)

    # Radial accelerations should be within ~0.2% (J2 perturbation is ~1e-3 of central)
    import math
    tb_r = math.sqrt(tb_acc[0]**2 + tb_acc[1]**2 + tb_acc[2]**2)
    shg_r = math.sqrt(shg_acc[0]**2 + shg_acc[1]**2 + shg_acc[2]**2)
    assert abs(shg_r - tb_r) / tb_r < 0.01, (
        f"SHG ({shg_r:.6f}) should be close to two-body ({tb_r:.6f}) — "
        f"both include central term"
    )


class TestStressMirrorPhysicalRegime:
    """Stress-scenario mirrors — no GMAT golden data, validate physical regime only.

    These mirror four of the eight GMAT stress scenarios.  R3 (high-gravity
    degree-70) and R6 (full-fidelity degree-50) are deferred until
    SphericalHarmonicGravity supports degrees > 8.
    """

    @pytest.fixture(scope="class")
    def stress_results(self):
        from humeris.adapters.gmat_mirror import run_stress_mirrors
        return run_stress_mirrors()

    # R1 — point-mass energy conservation (SMA=8000 km, ECC=0.15, 7 days)
    def test_stress_energy_drift_sma_conserved(self, stress_results):
        r = stress_results["stress_rk4_energy_drift"]
        assert abs(r["endSMA"] - r["startSMA"]) < 0.1, (
            f"SMA drift {abs(r['endSMA'] - r['startSMA']):.6f} km exceeds 0.1 km"
        )

    def test_stress_energy_drift_ecc_conserved(self, stress_results):
        r = stress_results["stress_rk4_energy_drift"]
        assert abs(r["endECC"] - r["startECC"]) < 1e-6, (
            f"ECC drift {abs(r['endECC'] - r['startECC']):.2e} exceeds 1e-6"
        )

    def test_stress_energy_drift_relative_energy(self, stress_results):
        r = stress_results["stress_rk4_energy_drift"]
        assert r["relativeEnergyDrift"] < 1e-7, (
            f"Relative energy drift {r['relativeEnergyDrift']:.2e} exceeds 1e-7"
        )

    def test_stress_energy_drift_elapsed(self, stress_results):
        r = stress_results["stress_rk4_energy_drift"]
        assert abs(r["elapsedDays"] - 7.0) < 1e-6

    # R2 — VLEO drag decay (SMA=6778 km, NRLMSISE-00, 7 days)
    def test_stress_drag_decay_sma_decreases(self, stress_results):
        r = stress_results["stress_drag_decay_vleo"]
        assert r["endSMA"] < r["startSMA"], (
            f"SMA should decrease under drag: start={r['startSMA']:.3f} end={r['endSMA']:.3f}"
        )

    def test_stress_drag_decay_altitude_drops(self, stress_results):
        r = stress_results["stress_drag_decay_vleo"]
        assert r["endAltKm"] < r["startAltKm"], (
            f"Altitude should decrease: start={r['startAltKm']:.1f} end={r['endAltKm']:.1f}"
        )

    def test_stress_drag_decay_orbit_remains_bound(self, stress_results):
        r = stress_results["stress_drag_decay_vleo"]
        assert 0 < r["endECC"] < 1.0, f"Orbit should remain elliptical, got ECC={r['endECC']}"
        assert r["endSMA"] > 6371.0, f"SMA should remain above Earth surface: {r['endSMA']:.1f}"

    def test_stress_drag_decay_elapsed(self, stress_results):
        r = stress_results["stress_drag_decay_vleo"]
        assert abs(r["elapsedDays"] - 7.0) < 1e-6

    # R4 — GEO SRP + third-body (SMA=42164 km, 60 days)
    def test_stress_geo_srp_sma_stable(self, stress_results):
        r = stress_results["stress_srp_geo_long_duration"]
        drift = abs(r["endSMA"] - r["startSMA"])
        assert drift < 200.0, (
            f"GEO SMA drift {drift:.1f} km over 60 days exceeds 200 km tolerance"
        )

    def test_stress_geo_srp_orbit_bound(self, stress_results):
        r = stress_results["stress_srp_geo_long_duration"]
        assert 0 <= r["endECC"] < 0.1, f"GEO ECC should remain small: {r['endECC']}"
        assert r["endSMA"] > 35000.0, f"SMA should remain near GEO: {r['endSMA']:.1f}"

    def test_stress_geo_srp_elapsed(self, stress_results):
        r = stress_results["stress_srp_geo_long_duration"]
        assert abs(r["elapsedDays"] - 60.0) < 1e-6

    def test_stress_geo_srp_force_stack(self, stress_results):
        r = stress_results["stress_srp_geo_long_duration"]
        names = r["forceModels"]
        assert "SphericalHarmonicGravity" in names
        assert "SolarRadiationPressureForce" in names
        assert "SolarThirdBodyForce" in names
        assert "LunarThirdBodyForce" in names

    # R5 — Molniya third-body (SMA=26560 km, ECC=0.74, 30 days)
    def test_stress_molniya_ecc_remains_high(self, stress_results):
        r = stress_results["stress_molniya_thirdbody"]
        assert r["endECC"] > 0.5, (
            f"Molniya ECC should remain high: {r['endECC']:.4f}"
        )

    def test_stress_molniya_aop_drifts(self, stress_results):
        r = stress_results["stress_molniya_thirdbody"]
        aop_drift = abs(r["endAOP"] - r["startAOP"])
        if aop_drift > 180.0:
            aop_drift = 360.0 - aop_drift
        assert aop_drift > 0.01, (
            f"Molniya AOP should drift under J2/third-body: drift={aop_drift:.4f} deg"
        )

    def test_stress_molniya_raan_precesses(self, stress_results):
        r = stress_results["stress_molniya_thirdbody"]
        raan_drift = abs(r["endRAAN"] - r["startRAAN"])
        if raan_drift > 180.0:
            raan_drift = 360.0 - raan_drift
        assert raan_drift > 0.01, (
            f"Molniya RAAN should precess: drift={raan_drift:.4f} deg"
        )

    def test_stress_molniya_orbit_bound(self, stress_results):
        r = stress_results["stress_molniya_thirdbody"]
        assert 0 < r["endECC"] < 1.0, f"Molniya orbit should remain bound: ECC={r['endECC']}"
        assert r["endSMA"] > 10000.0, f"SMA collapsed: {r['endSMA']:.1f}"

    def test_stress_molniya_elapsed(self, stress_results):
        r = stress_results["stress_molniya_thirdbody"]
        assert abs(r["elapsedDays"] - 30.0) < 1e-6

    def test_stress_molniya_force_stack(self, stress_results):
        r = stress_results["stress_molniya_thirdbody"]
        names = r["forceModels"]
        assert "SphericalHarmonicGravity" in names
        assert "SolarRadiationPressureForce" in names
        assert "SolarThirdBodyForce" in names
        assert "LunarThirdBodyForce" in names


def test_find_gmat_run_dir_falls_back_to_latest_complete_snapshot(tmp_path: Path):
    runs_root = tmp_path / "docs" / "test-runs"
    run_tier1 = runs_root / "run-0009-example-clean"
    run_tier2 = runs_root / "run-0010-example-clean"
    run_tier1.mkdir(parents=True)
    run_tier2.mkdir(parents=True)
    (runs_root / "LATEST").write_text("run-0010-example-clean\n", encoding="utf-8")

    # Tier2-only snapshot: no parity-case files.
    (run_tier2 / "cases" / "conjunction_screening_heuristic").mkdir(parents=True)

    # Prior tier1 snapshot with required parity artifacts.
    basic = run_tier1 / "cases" / "basic_leo_two_body" / "basic_leo_two_body_results.txt"
    j2 = run_tier1 / "cases" / "advanced_j2_raan_drift" / "advanced_j2_raan_drift_results.txt"
    hyp = run_tier1 / "cases" / "advanced_oumuamua_hyperbolic" / "advanced_oumuamua_hyperbolic_results.txt"
    basic.parent.mkdir(parents=True)
    j2.parent.mkdir(parents=True)
    hyp.parent.mkdir(parents=True)
    basic.write_text("1 2 3 4 5 6 7\n", encoding="utf-8")
    j2.write_text("1 2 3 4 5 6 7\n", encoding="utf-8")
    hyp.write_text("1 2 3 4 5 6 7\n", encoding="utf-8")

    picked = find_gmat_run_dir(tmp_path)
    assert picked.name == "run-0009-example-clean"
