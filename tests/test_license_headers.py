# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""Verify all pro package source files have the standard 4-line Commercial license header."""

from pathlib import Path

PRO_ROOT = Path(__file__).resolve().parent.parent / "packages" / "pro" / "src" / "humeris"

EXPECTED_HEADER_LINES = [
    "# Copyright (c) 2026 Jeroen Visser. All rights reserved.",
    "# Licensed under the terms in COMMERCIAL-LICENSE.md.",
    "# Free for personal, educational, and academic use.",
    "# Commercial use requires a paid license \u2014 see COMMERCIAL-LICENSE.md.",
]


def _collect_pro_py_files():
    """Collect all .py files in the pro package."""
    return sorted(PRO_ROOT.rglob("*.py"))


def test_all_pro_files_have_standard_commercial_header():
    """Every .py file in packages/pro/ must start with the standard 4-line header."""
    files = _collect_pro_py_files()
    assert len(files) > 0, "No .py files found in pro package"

    violations = []
    for py_file in files:
        lines = py_file.read_text(encoding="utf-8").splitlines()
        if len(lines) < 4:
            violations.append((py_file, "File has fewer than 4 lines"))
            continue
        for i, expected in enumerate(EXPECTED_HEADER_LINES):
            if lines[i] != expected:
                violations.append((py_file, f"Line {i + 1}: expected {expected!r}, got {lines[i]!r}"))
                break

    if violations:
        msg_parts = [f"\n{len(violations)} file(s) with non-standard headers:"]
        for path, reason in violations:
            msg_parts.append(f"  {path.relative_to(PRO_ROOT.parent.parent.parent)}: {reason}")
        raise AssertionError("\n".join(msg_parts))
