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
    SolarRadiationPressureForce,
    SphericalHarmonicGravity,
    TwoBodyGravity,
    propagate_numerical,
)
from humeris.domain.orbit_properties import state_vector_to_elements
from humeris.domain.gravity_field import CunninghamGravity, load_gravity_field
from humeris.domain.nrlmsise00 import NRLMSISE00DragForce, SpaceWeather
from humeris.domain.third_body import LunarThirdBodyForce, SolarThirdBodyForce


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
        timeout=30,
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


def _specific_energy_sign(
    pos_eci: tuple[float, float, float],
    vel_eci: tuple[float, float, float],
) -> int:
    r = math.sqrt(pos_eci[0] ** 2 + pos_eci[1] ** 2 + pos_eci[2] ** 2)
    v2 = vel_eci[0] ** 2 + vel_eci[1] ** 2 + vel_eci[2] ** 2
    eps = 0.5 * v2 - OrbitalConstants.MU_EARTH / r
    if abs(eps) < 1e-9:
        return 0
    return 1 if eps > 0.0 else -1


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
        "aop_deg": float(out["arg_perigee_deg"]),
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

    # Mirror 4: Sun-centric high-fidelity extension.
    # Uses an enriched force-model stack (solar/lunar third-body + SRP) while
    # remaining in the current Earth-centered propagator architecture.
    suncentric_forces = [
        TwoBodyGravity(),
        SolarThirdBodyForce(),
        LunarThirdBodyForce(),
        SolarRadiationPressureForce(cr=1.2, area_m2=8.0, mass_kg=1200.0),
    ]
    ou_sun = propagate_numerical(
        initial_state=ou_state,
        duration=timedelta(days=120.0),
        step=timedelta(seconds=3600.0),
        force_models=suncentric_forces,
        integrator="dormand_prince",
    )
    os_start = ou_sun.steps[0]
    os_end = ou_sun.steps[-1]
    os0 = _elements(os_start.position_eci, os_start.velocity_eci)
    os1 = _elements(os_end.position_eci, os_end.velocity_eci)
    out["advanced_oumuamua_suncentric"] = {
        "startECC": os0["ecc"],
        "startINC": os0["inc_deg"],
        "startRMAG": _rmag_km(os_start.position_eci),
        "endECC": os1["ecc"],
        "endINC": os1["inc_deg"],
        "endRMAG": _rmag_km(os_end.position_eci),
        "startEnergySign": _specific_energy_sign(os_start.position_eci, os_start.velocity_eci),
        "endEnergySign": _specific_energy_sign(os_end.position_eci, os_end.velocity_eci),
        "elapsedDays": 120.0,
        "forceModels": [type(force).__name__ for force in suncentric_forces],
    }
    return out


def run_stress_mirrors() -> dict[str, dict[str, Any]]:
    """Run stress-scenario mirrors and return per-case metrics.

    Mirrors four of the eight GMAT stress scenarios:
      - stress_rk4_energy_drift: point-mass energy conservation (7 days)
      - stress_drag_decay_vleo: VLEO with NRLMSISE-00 drag (7 days)
      - stress_srp_geo_long_duration: GEO with SRP + third-body (60 days)
      - stress_molniya_thirdbody: Molniya with J2/SRP/third-body (30 days)

    Deferred (require SphericalHarmonicGravity degree > 8):
      - stress_high_gravity_leo (needs degree 70)
      - stress_sun_synch_full_fidelity (needs degree 50)
    """
    out: dict[str, dict[str, Any]] = {}
    r_earth_km = OrbitalConstants.R_EARTH_EQUATORIAL / 1000.0

    # S1: Point-mass energy conservation (GMAT: RK89, point mass, 7 days)
    s1_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)
    s1_state = _build_state(
        sma_km=8000.0, ecc=0.15, inc_deg=28.5,
        raan_deg=40.0, aop_deg=10.0, ta_deg=0.0,
        epoch=s1_epoch,
    )
    s1_integrator = "rk89"  # GMAT: RungeKutta89
    s1 = propagate_numerical(
        initial_state=s1_state,
        duration=timedelta(days=7.0),
        step=timedelta(seconds=30.0),
        force_models=[TwoBodyGravity()],
        integrator=s1_integrator,
    )
    s1_start, s1_end = s1.steps[0], s1.steps[-1]
    s1e0 = _elements(s1_start.position_eci, s1_start.velocity_eci)
    s1e1 = _elements(s1_end.position_eci, s1_end.velocity_eci)
    out["stress_rk4_energy_drift"] = {
        "startSMA": s1e0["sma_km"],
        "startECC": s1e0["ecc"],
        "endSMA": s1e1["sma_km"],
        "endECC": s1e1["ecc"],
        "relativeEnergyDrift": s1.relative_energy_drift,
        "elapsedDays": 7.0,
        "integrator": s1_integrator,
    }

    # S2: VLEO drag decay (GMAT: PD78, EGM96 8x8 + MSISE90, 7 days)
    s2_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)
    s2_state = _build_state(
        sma_km=6778.137, ecc=0.001, inc_deg=51.6,
        raan_deg=40.0, aop_deg=10.0, ta_deg=0.0,
        epoch=s2_epoch,
    )
    sw = SpaceWeather(f107_daily=175.0, f107_average=175.0, ap_daily=4.0)
    # SphericalHarmonicGravity includes central body term — no TwoBodyGravity
    s2_forces = [
        SphericalHarmonicGravity(max_degree=8),
        NRLMSISE00DragForce(cd=2.2, area_m2=20.0, mass_kg=1000.0, space_weather=sw),
    ]
    s2_integrator = "dormand_prince"  # GMAT: PrinceDormand78
    s2 = propagate_numerical(
        initial_state=s2_state,
        duration=timedelta(days=7.0),
        step=timedelta(seconds=30.0),
        force_models=s2_forces,
        integrator=s2_integrator,
    )
    s2_start, s2_end = s2.steps[0], s2.steps[-1]
    s2e0 = _elements(s2_start.position_eci, s2_start.velocity_eci)
    s2e1 = _elements(s2_end.position_eci, s2_end.velocity_eci)
    out["stress_drag_decay_vleo"] = {
        "startSMA": s2e0["sma_km"],
        "startECC": s2e0["ecc"],
        "startAltKm": s2e0["sma_km"] - r_earth_km,
        "endSMA": s2e1["sma_km"],
        "endECC": s2e1["ecc"],
        "endAltKm": s2e1["sma_km"] - r_earth_km,
        "elapsedDays": 7.0,
        "integrator": s2_integrator,
    }

    # S3: GEO SRP + third-body (GMAT: PD78, JGM2 4x4 + SRP + Sun/Moon, 60 days)
    s3_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)
    s3_state = _build_state(
        sma_km=42164.0, ecc=0.0001, inc_deg=0.05,
        raan_deg=0.0, aop_deg=0.0, ta_deg=0.0,
        epoch=s3_epoch,
    )
    # SphericalHarmonicGravity includes central body term — no TwoBodyGravity
    s3_forces = [
        SphericalHarmonicGravity(max_degree=4),
        SolarRadiationPressureForce(cr=1.4, area_m2=30.0, mass_kg=2000.0),
        SolarThirdBodyForce(),
        LunarThirdBodyForce(),
    ]
    s3_integrator = "dormand_prince"  # GMAT: PrinceDormand78
    s3 = propagate_numerical(
        initial_state=s3_state,
        duration=timedelta(days=60.0),
        step=timedelta(seconds=60.0),
        force_models=s3_forces,
        integrator=s3_integrator,
    )
    s3_start, s3_end = s3.steps[0], s3.steps[-1]
    s3e0 = _elements(s3_start.position_eci, s3_start.velocity_eci)
    s3e1 = _elements(s3_end.position_eci, s3_end.velocity_eci)
    out["stress_srp_geo_long_duration"] = {
        "startSMA": s3e0["sma_km"],
        "startECC": s3e0["ecc"],
        "endSMA": s3e1["sma_km"],
        "endECC": s3e1["ecc"],
        "elapsedDays": 60.0,
        "forceModels": [type(f).__name__ for f in s3_forces],
        "integrator": s3_integrator,
    }

    # S5: Molniya third-body (GMAT: RK89, EGM96 8x8 + SRP + Sun/Moon, 30 days)
    s5_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)
    s5_state = _build_state(
        sma_km=26560.0, ecc=0.74, inc_deg=63.4,
        raan_deg=90.0, aop_deg=270.0, ta_deg=0.0,
        epoch=s5_epoch,
    )
    # SphericalHarmonicGravity includes central body term — no TwoBodyGravity
    s5_forces = [
        SphericalHarmonicGravity(max_degree=8),
        SolarRadiationPressureForce(cr=1.2, area_m2=10.0, mass_kg=1500.0),
        SolarThirdBodyForce(),
        LunarThirdBodyForce(),
    ]
    s5_integrator = "rk89"  # GMAT: RungeKutta89
    s5 = propagate_numerical(
        initial_state=s5_state,
        duration=timedelta(days=30.0),
        step=timedelta(seconds=60.0),
        force_models=s5_forces,
        integrator=s5_integrator,
    )
    s5_start, s5_end = s5.steps[0], s5.steps[-1]
    s5e0 = _elements(s5_start.position_eci, s5_start.velocity_eci)
    s5e1 = _elements(s5_end.position_eci, s5_end.velocity_eci)
    out["stress_molniya_thirdbody"] = {
        "startSMA": s5e0["sma_km"],
        "startECC": s5e0["ecc"],
        "startAOP": s5e0["aop_deg"],
        "startRAAN": s5e0["raan_deg"],
        "endSMA": s5e1["sma_km"],
        "endECC": s5e1["ecc"],
        "endAOP": s5e1["aop_deg"],
        "endRAAN": s5e1["raan_deg"],
        "elapsedDays": 30.0,
        "forceModels": [type(f).__name__ for f in s5_forces],
        "integrator": s5_integrator,
    }

    return out


def run_high_degree_stress_mirrors() -> dict[str, dict[str, Any]]:
    """Run stress scenarios requiring high-degree gravity (CunninghamGravity).

    Mirrors the two GMAT stress scenarios that need spherical harmonic
    degrees above 8:
      - stress_high_gravity_leo: EGM96 degree 70, 14 days
      - stress_sun_synch_full_fidelity: EGM96 degree 50 + drag + SRP + 3body, 30 days

    CunninghamGravity is perturbation-only — must pair with TwoBodyGravity.
    """
    out: dict[str, dict[str, Any]] = {}
    r_earth_km = OrbitalConstants.R_EARTH_EQUATORIAL / 1000.0

    # S3: High-gravity LEO (GMAT: PD78, EGM96 70×70, 14 days)
    s3_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)
    s3_state = _build_state(
        sma_km=6628.137, ecc=0.001, inc_deg=51.6,
        raan_deg=40.0, aop_deg=10.0, ta_deg=0.0,
        epoch=s3_epoch,
    )
    gf70 = load_gravity_field(max_degree=70)
    s3_forces = [
        TwoBodyGravity(),
        CunninghamGravity(gf70),
    ]
    s3_integrator = "dormand_prince"  # GMAT: PrinceDormand78
    s3 = propagate_numerical(
        initial_state=s3_state,
        duration=timedelta(days=14.0),
        step=timedelta(seconds=30.0),
        force_models=s3_forces,
        integrator=s3_integrator,
    )
    s3_start, s3_end = s3.steps[0], s3.steps[-1]
    s3e0 = _elements(s3_start.position_eci, s3_start.velocity_eci)
    s3e1 = _elements(s3_end.position_eci, s3_end.velocity_eci)
    out["stress_high_gravity_leo"] = {
        "startSMA": s3e0["sma_km"],
        "startECC": s3e0["ecc"],
        "startAOP": s3e0["aop_deg"],
        "endSMA": s3e1["sma_km"],
        "endECC": s3e1["ecc"],
        "endAOP": s3e1["aop_deg"],
        "elapsedDays": 14.0,
        "forceModels": [type(f).__name__ for f in s3_forces],
        "integrator": s3_integrator,
    }

    # S6: Sun-synch full fidelity (GMAT: RK89, EGM96 50×50 + MSISE90 + SRP + Sun/Moon, 30 days)
    s6_epoch = datetime(2024, 1, 1, 0, 0, 0, tzinfo=_UTC)
    s6_state = _build_state(
        sma_km=7078.137, ecc=0.001, inc_deg=98.19,
        raan_deg=0.0, aop_deg=0.0, ta_deg=0.0,
        epoch=s6_epoch,
    )
    gf50 = load_gravity_field(max_degree=50)
    sw6 = SpaceWeather(f107_daily=150.0, f107_average=150.0, ap_daily=3.0)
    s6_forces = [
        TwoBodyGravity(),
        CunninghamGravity(gf50),
        NRLMSISE00DragForce(cd=2.2, area_m2=15.0, mass_kg=1200.0, space_weather=sw6),
        SolarRadiationPressureForce(cr=1.2, area_m2=10.0, mass_kg=1200.0),
        SolarThirdBodyForce(),
        LunarThirdBodyForce(),
    ]
    s6_integrator = "rk89"  # GMAT: RungeKutta89
    s6 = propagate_numerical(
        initial_state=s6_state,
        duration=timedelta(days=30.0),
        step=timedelta(seconds=30.0),
        force_models=s6_forces,
        integrator=s6_integrator,
    )
    s6_start, s6_end = s6.steps[0], s6.steps[-1]
    s6e0 = _elements(s6_start.position_eci, s6_start.velocity_eci)
    s6e1 = _elements(s6_end.position_eci, s6_end.velocity_eci)
    out["stress_sun_synch_full_fidelity"] = {
        "startSMA": s6e0["sma_km"],
        "startECC": s6e0["ecc"],
        "startRAAN": s6e0["raan_deg"],
        "endSMA": s6e1["sma_km"],
        "endECC": s6e1["ecc"],
        "endRAAN": s6e1["raan_deg"],
        "endAltKm": s6e1["sma_km"] - r_earth_km,
        "elapsedDays": 30.0,
        "integrator": s6_integrator,
        "forceModels": [type(f).__name__ for f in s6_forces],
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
    required_files = (
        "cases/basic_leo_two_body/basic_leo_two_body_results.txt",
        "cases/advanced_j2_raan_drift/advanced_j2_raan_drift_results.txt",
        "cases/advanced_oumuamua_hyperbolic/advanced_oumuamua_hyperbolic_results.txt",
    )

    def has_required_artifacts(candidate: Path) -> bool:
        return all((candidate / rel).exists() for rel in required_files)

    if run_id is not None:
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"GMAT run directory not found: {run_dir}")
        if not has_required_artifacts(run_dir):
            raise FileNotFoundError(
                f"GMAT run directory missing parity artifacts: {run_dir}"
            )
        return run_dir

    latest_file = runs_root / "LATEST"
    candidates: list[str] = []
    if latest_file.exists():
        latest_id = latest_file.read_text(encoding="utf-8").strip()
        if latest_id:
            candidates.append(latest_id)
    for d in sorted(runs_root.glob("run-*"), reverse=True):
        if d.is_dir() and d.name not in candidates:
            candidates.append(d.name)

    for run_name in candidates:
        run_dir = runs_root / run_name
        if run_dir.exists() and has_required_artifacts(run_dir):
            return run_dir

    raise FileNotFoundError(
        f"No GMAT run with required parity artifacts found under {runs_root}"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
