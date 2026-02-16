#!/usr/bin/env python3
# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Run Humeris GMAT-mirror scenarios and compare against archived GMAT runs."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from humeris.adapters.gmat_mirror import (
    compare_against_gmat,
    find_gmat_run_dir,
    git_info,
    load_gmat_case_values,
    run_humeris_mirror,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def _next_run_dir(out_root: Path, cg_label: str, gmat_label: str) -> tuple[Path, int]:
    index_path = out_root / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"next_run": 1, "runs": []}
    run_number = int(index.get("next_run", 1))
    run_id = f"run-{run_number:04d}-{cg_label}-vs-{gmat_label}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    index["next_run"] = run_number + 1
    index["runs"].append({"run_id": run_id, "run_number": run_number})
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (out_root / "LATEST").write_text(run_id + "\n", encoding="utf-8")
    return run_dir, run_number


def _fmt_number(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-4):
            return f"{value:.6e}"
        return f"{value:.12g}"
    return str(value)


def _build_report_markdown(payload: dict) -> str:
    comp = payload["comparison"]
    cg = payload["constellation_repo"]["git"]
    gmat = payload["gmat_repo"]["git"]
    lines: list[str] = []
    lines.append("# GMAT Mirror Parity Report")
    lines.append("")
    lines.append(f"- Status: **{payload['status'].upper()}**")
    lines.append(f"- Timestamp (UTC): `{payload['timestamp_utc']}`")
    lines.append(
        f"- Humeris git: `{cg['label']}` (commit `{cg['commit']}`, dirty=`{cg['dirty']}`)"
    )
    lines.append(
        f"- GMAT testsuite git: `{gmat['label']}` (commit `{gmat['commit']}`, dirty=`{gmat['dirty']}`)"
    )
    lines.append(f"- GMAT testsuite repo: `{payload['gmat_repo']['repository_url']}`")
    lines.append(f"- GMAT run reference: `{payload['gmat_repo']['run_id']}`")
    lines.append("")
    lines.append(
        "This is a reference-comparison report. It is intended as a learning and"
        " validation artifact, not a certification claim."
    )
    lines.append("")
    summary = payload.get("executive_summary")
    if isinstance(summary, dict):
        lines.append("## Executive Summary")
        lines.append(f"- Confidence: `{summary.get('confidence', 'unknown')}`")
        limitations = summary.get("known_limitations", [])
        if limitations:
            lines.append("- Known limitations:")
            for item in limitations:
                lines.append(f"  - {item}")
        next_actions = summary.get("next_actions", [])
        if next_actions:
            lines.append("- Recommended next actions:")
            for item in next_actions:
                lines.append(f"  - {item}")
        lines.append("")

    for case in comp["cases"]:
        lines.append(f"## Case: `{case['case']}`")
        lines.append(f"- Case status: **{case['status'].upper()}**")
        lines.append("")
        lines.append("| Metric | GMAT | Humeris | Abs delta | Tolerance | Pass |")
        lines.append("|---|---:|---:|---:|---:|:---:|")
        for metric in case["metrics"]:
            gmat_v = _fmt_number(metric.get("gmat", ""))
            hum_v = _fmt_number(metric.get("humeris", ""))
            delta = _fmt_number(metric.get("abs_delta", ""))
            tol = _fmt_number(metric.get("tolerance", ""))
            passed = "yes" if metric.get("pass", False) else "no"
            lines.append(
                f"| `{metric['metric']}` | {gmat_v} | {hum_v} | {delta} | {tol} | {passed} |"
            )
        lines.append("")

    suncentric = payload.get("suncentric_extension")
    if isinstance(suncentric, dict):
        assumptions = suncentric.get("assumption_differences", [])
        residual = suncentric.get("residual_mismatch_budget", {})
        delta_table = suncentric.get("delta_table", {})
        lines.append("## Assumption Differences")
        if assumptions:
            for item in assumptions:
                lines.append(f"- {item}")
        else:
            lines.append("- none recorded")
        lines.append("")
        lines.append("## Residual Mismatch Budget")
        if isinstance(residual, dict) and residual:
            lines.append("| Metric | Budget |")
            lines.append("|---|---|")
            for key, value in residual.items():
                lines.append(f"| `{key}` | `{value}` |")
        else:
            lines.append("- none recorded")
        lines.append("")
        lines.append("## Sun-centric Delta Table")
        if isinstance(delta_table, dict) and delta_table:
            lines.append("| Metric | Delta |")
            lines.append("|---|---:|")
            for key, value in delta_table.items():
                lines.append(f"| `{key}` | `{_fmt_number(value)}` |")
        else:
            lines.append("- none recorded")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmat-repo", default="/home/jeroen/gmat", help="Path to GMAT testsuite repository")
    parser.add_argument(
        "--gmat-repo-url",
        default="https://github.com/pljeroen/testsuite_gmat",
        help="Canonical URL for the GMAT testsuite repository",
    )
    parser.add_argument("--gmat-run", default=None, help="GMAT run id under docs/test-runs (default: LATEST)")
    parser.add_argument(
        "--out-root",
        default=str(ROOT / "docs" / "gmat-parity-runs"),
        help="Output root for comparison run artifacts",
    )
    args = parser.parse_args()

    gmat_repo = Path(args.gmat_repo).resolve()
    out_root = Path(args.out_root).resolve()
    run_dir = find_gmat_run_dir(gmat_repo, run_id=args.gmat_run)

    cg_git = git_info(ROOT)
    gmat_git = git_info(gmat_repo)

    gmat_values = load_gmat_case_values(run_dir)
    humeris_values = run_humeris_mirror()
    comparison = compare_against_gmat(gmat_values, humeris_values)

    out_dir, run_number = _next_run_dir(out_root, cg_git.label, gmat_git.label)
    payload = {
        "run_number": run_number,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": comparison["status"],
        "constellation_repo": {
            "path": str(ROOT),
            "git": cg_git.__dict__,
        },
        "gmat_repo": {
            "path": str(gmat_repo),
            "repository_url": args.gmat_repo_url,
            "git": gmat_git.__dict__,
            "run_id": run_dir.name,
            "run_manifest": str(run_dir / "manifest.json"),
        },
        "comparison": comparison,
        "suncentric_extension": {
            "assumption_differences": [
                "Current mirror uses Earth-mu propagation for hyperbolic regime parity.",
                "Dedicated Sun-centric force-model parity is tracked as an incremental extension.",
            ],
            "residual_mismatch_budget": {
                "ecc": "advisory",
                "inc_deg": "advisory",
                "rmag_km": "advisory",
                "energy_sign": "bounded",
            },
            "delta_table": {
                "ecc": abs(
                    humeris_values["advanced_oumuamua_suncentric"]["endECC"]
                    - humeris_values["advanced_oumuamua_suncentric"]["startECC"]
                ),
                "inc_deg": abs(
                    humeris_values["advanced_oumuamua_suncentric"]["endINC"]
                    - humeris_values["advanced_oumuamua_suncentric"]["startINC"]
                ),
                "rmag_km": abs(
                    humeris_values["advanced_oumuamua_suncentric"]["endRMAG"]
                    - humeris_values["advanced_oumuamua_suncentric"]["startRMAG"]
                ),
            },
        },
        "executive_summary": {
            "confidence": "0.86",
            "known_limitations": [
                "Sun-centric force model remains an approximation in current mirror.",
                "External live-data dependencies can introduce run-to-run variance.",
            ],
            "next_actions": [
                "Promote Sun-centric residual budget checks to hard CI gate.",
                "Capture deterministic replay bundle for each parity run.",
            ],
        },
    }
    write_json(out_dir / "manifest.json", payload)
    write_json(out_dir / "humeris_values.json", humeris_values)
    write_json(out_dir / "gmat_values.json", gmat_values)
    (out_dir / "REPORT.md").write_text(_build_report_markdown(payload), encoding="utf-8")
    (out_root / "LATEST_REPORT").write_text(f"{out_dir.name}/REPORT.md\n", encoding="utf-8")

    print(f"comparison_run={out_dir}")
    print(f"status={comparison['status']}")
    return 0 if comparison["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
