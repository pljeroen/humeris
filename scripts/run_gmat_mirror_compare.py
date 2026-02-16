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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmat-repo", default="/home/jeroen/gmat", help="Path to GMAT testsuite repository")
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
            "git": gmat_git.__dict__,
            "run_id": run_dir.name,
            "run_manifest": str(run_dir / "manifest.json"),
        },
        "comparison": comparison,
    }
    write_json(out_dir / "manifest.json", payload)
    write_json(out_dir / "humeris_values.json", humeris_values)
    write_json(out_dir / "gmat_values.json", gmat_values)

    print(f"comparison_run={out_dir}")
    print(f"status={comparison['status']}")
    return 0 if comparison["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

