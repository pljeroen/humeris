# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""GMAT-mirror scenario runner and cross-repo comparison utilities.

This module mirrors key GMAT scenario classes without invoking GMAT itself.
It runs equivalent Humeris propagations and compares outputs against archived
GMAT run artifacts.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.domain.propagation import OrbitalState
from humeris.domain.numerical_propagation import (
    J2Perturbation,
    TwoBodyGravity,
    propagate_numerical,
)
from humeris.domain.orbit_properties import state_vector_to_elements


_UTC = timezone.utc


@dataclass(frozen=True)
class GitInfo:
    state: str
    commit: str
    dirty: str
    label: str


def _capture_git(repo: Path, args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def git_info(repo: Path) -> GitInfo:
    rc, inside = _capture_git(repo, ["rev-parse", "--is-inside-work-tree"])
    if rc != 0 or inside != "true":
        return GitInfo(state="no-repo", commit="none", dirty="unknown", label="norepo")
    rc, commit = _capture_git(repo, ["rev-parse", "--short", "HEAD"])
    commit_label = commit if rc == 0 and commit else "unborn"
    rc, status = _capture_git(repo, ["status", "--porcelain"])
    dirty = "dirty" if rc == 0 and status else "clean"
    return GitInfo(state="repo", commit=commit_label, dirty=dirty, label=f"{commit_label}-{dirty}")


def _angular_delta_deg(start_deg: float, end_deg: float) -> float:
    return abs(((end_deg - start_deg + 180.0) % 360.0) - 180.0)


def _rmag_km(pos_eci: tuple[float, float, float]) -> float:
    return math.sqrt(pos_eci[0] ** 2 + pos_eci[1] ** 2 + pos_eci[2] ** 2) / 1000.0


def _build_state(
    *,
    sma_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    aop_deg: float,
    ta_deg: float,
    epoch: datetime,
) -> OrbitalState:
    a_m = sma_km * 1000.0
    n = math.sqrt(OrbitalConstants.MU_EARTH / abs(a_m) ** 3)
    return OrbitalState(
        semi_major_axis_m=a_m,
        eccentricity=ecc,
        inclination_rad=math.radians(inc_deg),
        raan_rad=math.radians(raan_deg),
        arg_perigee_rad=math.radians(aop_deg),
        true_anomaly_rad=math.radians(ta_deg),
        mean_motion_rad_s=n,
        reference_epoch=epoch,
    )


def _elements(pos: tuple[float, float, float], vel: tuple[float, float, float]) -> dict[str, float]:
    out = state_vector_to_elements(pos, vel)
    return {
        "sma_km": float(out["semi_major_axis_m"]) / 1000.0,
        "ecc": float(out["eccentricity"]),
        "inc_deg": float(out["inclination_deg"]),
        "raan_deg": float(out["raan_deg"]),
    }


def run_humeris_mirror() -> dict[str, dict[str, float]]:
    """Run mirrored scenario set and return per-case metrics."""
    out: dict[str, dict[str, float]] = {}

    # Mirror 1: basic LEO two-body conservation
    basic_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)
    basic_state = _build_state(
        sma_km=7000.0,
        ecc=0.001,
        inc_deg=28.5,
        raan_deg=40.0,
        aop_deg=10.0,
        ta_deg=0.0,
        epoch=basic_epoch,
    )
    basic = propagate_numerical(
        initial_state=basic_state,
        duration=timedelta(seconds=5400.0),
        step=timedelta(seconds=30.0),
        force_models=[TwoBodyGravity()],
        integrator="dormand_prince",
    )
    b_start = basic.steps[0]
    b_end = basic.steps[-1]
    b0 = _elements(b_start.position_eci, b_start.velocity_eci)
    b1 = _elements(b_end.position_eci, b_end.velocity_eci)
    out["basic_leo_two_body"] = {
        "startSMA": b0["sma_km"],
        "startECC": b0["ecc"],
        "startRMAG": _rmag_km(b_start.position_eci),
        "endSMA": b1["sma_km"],
        "endECC": b1["ecc"],
        "endRMAG": _rmag_km(b_end.position_eci),
        "elapsedSecs": 5400.0,
    }

    # Mirror 2: J2 RAAN drift
    j2_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)
    j2_state = _build_state(
        sma_km=7000.0,
        ecc=0.001,
        inc_deg=97.8,
        raan_deg=20.0,
        aop_deg=15.0,
        ta_deg=25.0,
        epoch=j2_epoch,
    )
    j2 = propagate_numerical(
        initial_state=j2_state,
        duration=timedelta(days=7.0),
        step=timedelta(seconds=30.0),
        force_models=[TwoBodyGravity(), J2Perturbation()],
        integrator="dormand_prince",
    )
    j_start = j2.steps[0]
    j_end = j2.steps[-1]
    j0 = _elements(j_start.position_eci, j_start.velocity_eci)
    j1 = _elements(j_end.position_eci, j_end.velocity_eci)
    out["advanced_j2_raan_drift"] = {
        "startRAAN": j0["raan_deg"],
        "startINC": j0["inc_deg"],
        "startECC": j0["ecc"],
        "endRAAN": j1["raan_deg"],
        "endINC": j1["inc_deg"],
        "endECC": j1["ecc"],
        "elapsedDays": 7.0,
        "raanDriftDeg": _angular_delta_deg(j0["raan_deg"], j1["raan_deg"]),
    }

    # Mirror 3: Oumuamua-inspired hyperbolic regime (regime parity, not force parity)
    # Uses Earth mu central-body model due current model architecture.
    ou_epoch = datetime(2018, 1, 1, 0, 0, 0, tzinfo=_UTC)
    ou_state = _build_state(
        sma_km=-130000000.0,
        ecc=1.20,
        inc_deg=122.74,
        raan_deg=24.60,
        aop_deg=241.70,
        ta_deg=40.0,
        epoch=ou_epoch,
    )
    ou = propagate_numerical(
        initial_state=ou_state,
        duration=timedelta(days=120.0),
        step=timedelta(seconds=3600.0),
        force_models=[TwoBodyGravity()],
        integrator="dormand_prince",
    )
    o_start = ou.steps[0]
    o_end = ou.steps[-1]
    o0 = _elements(o_start.position_eci, o_start.velocity_eci)
    o1 = _elements(o_end.position_eci, o_end.velocity_eci)
    out["advanced_oumuamua_hyperbolic"] = {
        "startECC": o0["ecc"],
        "startINC": o0["inc_deg"],
        "startRMAG": _rmag_km(o_start.position_eci),
        "endECC": o1["ecc"],
        "endINC": o1["inc_deg"],
        "endRMAG": _rmag_km(o_end.position_eci),
        "elapsedDays": 120.0,
    }
    return out


def _read_last_numeric_row(path: Path, expected_values: int) -> list[float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in reversed(lines):
        parts = line.split()
        try:
            values = [float(token) for token in parts]
        except ValueError:
            continue
        if len(values) == expected_values:
            return values
    raise ValueError(f"No numeric row with {expected_values} values in {path}")


def load_gmat_case_values(run_dir: Path) -> dict[str, dict[str, float]]:
    """Load mirrored-case numeric values from an archived GMAT run directory."""
    cases = run_dir / "cases"
    basic_vals = _read_last_numeric_row(
        cases / "basic_leo_two_body" / "basic_leo_two_body_results.txt",
        expected_values=7,
    )
    j2_vals = _read_last_numeric_row(
        cases / "advanced_j2_raan_drift" / "advanced_j2_raan_drift_results.txt",
        expected_values=7,
    )
    ou_vals = _read_last_numeric_row(
        cases / "advanced_oumuamua_hyperbolic" / "advanced_oumuamua_hyperbolic_results.txt",
        expected_values=7,
    )
    return {
        "basic_leo_two_body": {
            "startSMA": basic_vals[0],
            "startECC": basic_vals[1],
            "startRMAG": basic_vals[2],
            "endSMA": basic_vals[3],
            "endECC": basic_vals[4],
            "endRMAG": basic_vals[5],
            "elapsedSecs": basic_vals[6],
        },
        "advanced_j2_raan_drift": {
            "startRAAN": j2_vals[0],
            "startINC": j2_vals[1],
            "startECC": j2_vals[2],
            "endRAAN": j2_vals[3],
            "endINC": j2_vals[4],
            "endECC": j2_vals[5],
            "elapsedDays": j2_vals[6],
            "raanDriftDeg": _angular_delta_deg(j2_vals[0], j2_vals[3]),
        },
        "advanced_oumuamua_hyperbolic": {
            "startECC": ou_vals[0],
            "startINC": ou_vals[1],
            "startRMAG": ou_vals[2],
            "endECC": ou_vals[3],
            "endINC": ou_vals[4],
            "endRMAG": ou_vals[5],
            "elapsedDays": ou_vals[6],
        },
    }


def compare_against_gmat(
    gmat_values: dict[str, dict[str, float]],
    humeris_values: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compute per-case comparison report with pass/fail status."""
    cases: list[dict[str, Any]] = []
    overall_ok = True

    for case in ("basic_leo_two_body", "advanced_j2_raan_drift", "advanced_oumuamua_hyperbolic"):
        g = gmat_values[case]
        h = humeris_values[case]
        metric_rows: list[dict[str, Any]] = []

        if case == "basic_leo_two_body":
            tol = {
                "startSMA": 5.0,
                "startECC": 1e-3,
                "endSMA": 5.0,
                "endECC": 1e-3,
                "elapsedSecs": 1e-3,
            }
            metric_names = ["startSMA", "startECC", "endSMA", "endECC", "elapsedSecs"]
            ok = True
            for name in metric_names:
                delta = abs(h[name] - g[name])
                passed = delta <= tol[name]
                ok = ok and passed
                metric_rows.append(
                    {"metric": name, "gmat": g[name], "humeris": h[name], "abs_delta": delta, "tolerance": tol[name], "pass": passed}
                )

            # Physical consistency cross-checks, independent of direct deltas.
            g_conserve = abs(g["endSMA"] - g["startSMA"]) < 1e-2 and abs(g["endECC"] - g["startECC"]) < 1e-8
            h_conserve = abs(h["endSMA"] - h["startSMA"]) < 1e-1 and abs(h["endECC"] - h["startECC"]) < 1e-6
            metric_rows.append({"metric": "conservation_behavior_match", "gmat": g_conserve, "humeris": h_conserve, "pass": g_conserve and h_conserve})
            ok = ok and g_conserve and h_conserve

        elif case == "advanced_j2_raan_drift":
            tol = {
                "startRAAN": 2.0,
                "startINC": 0.2,
                "startECC": 5e-4,
                "elapsedDays": 1e-6,
                "raanDriftDeg": 2.0,
            }
            metric_names = ["startRAAN", "startINC", "startECC", "elapsedDays", "raanDriftDeg"]
            ok = True
            for name in metric_names:
                delta = abs(h[name] - g[name])
                passed = delta <= tol[name]
                ok = ok and passed
                metric_rows.append(
                    {"metric": name, "gmat": g[name], "humeris": h[name], "abs_delta": delta, "tolerance": tol[name], "pass": passed}
                )

            g_j2 = 0.01 < g["raanDriftDeg"] < 30.0 and 0.0 <= g["endECC"] < 1.0
            h_j2 = 0.01 < h["raanDriftDeg"] < 30.0 and 0.0 <= h["endECC"] < 1.0
            metric_rows.append({"metric": "j2_regime_match", "gmat": g_j2, "humeris": h_j2, "pass": g_j2 and h_j2})
            ok = ok and g_j2 and h_j2

        else:
            # Hyperbolic scenario is compared by regime parity, not absolute values.
            checks = {
                "start_ecc_gt_1": (g["startECC"] > 1.0, h["startECC"] > 1.0),
                "end_ecc_gt_1": (g["endECC"] > 1.0, h["endECC"] > 1.0),
                "rmag_changes_materially": (
                    abs(g["endRMAG"] - g["startRMAG"]) > 1000.0,
                    abs(h["endRMAG"] - h["startRMAG"]) > 1000.0,
                ),
                "elapsed_days_120": (abs(g["elapsedDays"] - 120.0) < 1e-3, abs(h["elapsedDays"] - 120.0) < 1e-3),
            }
            ok = True
            for name, (g_ok, h_ok) in checks.items():
                passed = g_ok and h_ok
                ok = ok and passed
                metric_rows.append({"metric": name, "gmat": g_ok, "humeris": h_ok, "pass": passed})

        overall_ok = overall_ok and ok
        cases.append(
            {
                "case": case,
                "status": "pass" if ok else "fail",
                "metrics": metric_rows,
                "gmat": g,
                "humeris": h,
            }
        )

    return {"status": "pass" if overall_ok else "fail", "cases": cases}


def find_gmat_run_dir(gmat_repo: Path, run_id: str | None = None) -> Path:
    runs_root = gmat_repo / "docs" / "test-runs"
    resolved_id = run_id
    if resolved_id is None:
        latest_file = runs_root / "LATEST"
        if not latest_file.exists():
            raise FileNotFoundError(f"GMAT LATEST file missing: {latest_file}")
        resolved_id = latest_file.read_text(encoding="utf-8").strip()
    run_dir = runs_root / resolved_id
    if not run_dir.exists():
        raise FileNotFoundError(f"GMAT run directory not found: {run_dir}")
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

