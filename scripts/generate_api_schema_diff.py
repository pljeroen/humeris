#!/usr/bin/env python3
"""Generate machine-readable API schema diff artifact."""
from __future__ import annotations

import json
from pathlib import Path

from humeris.domain.api_contracts import schema_diff

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "docs" / "contracts" / "api_schema_previous.json"
CURR = ROOT / "docs" / "contracts" / "api_schema_current.json"
OUT = ROOT / "docs" / "contracts" / "api_schema_diff.json"


def main() -> int:
    prev = json.loads(PREV.read_text(encoding="utf-8"))
    curr = json.loads(CURR.read_text(encoding="utf-8"))

    prev_fields = set(prev.get("fields", []))
    curr_fields = set(curr.get("fields", []))

    diff = schema_diff(previous_fields=prev_fields, current_fields=curr_fields)
    payload = {
        "previous_schema_version": prev.get("schema_version"),
        "current_schema_version": curr.get("schema_version"),
        "diff": diff,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
