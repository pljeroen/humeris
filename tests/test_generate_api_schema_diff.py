# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Release artifact generation test for API schema diff."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_generate_api_schema_diff_script_writes_expected_file():
    repo_root = Path(__file__).resolve().parent.parent
    out = repo_root / "docs" / "contracts" / "api_schema_diff.json"

    proc = subprocess.run(
        [sys.executable, "scripts/generate_api_schema_diff.py"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote=" in proc.stdout
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "previous_schema_version" in payload
    assert "current_schema_version" in payload
    assert "diff" in payload
    assert set(payload["diff"].keys()) == {"added", "removed", "unchanged"}
