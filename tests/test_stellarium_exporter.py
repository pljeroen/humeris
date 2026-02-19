# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Tests for Stellarium-compatible TLE exporter.

Verifies TLE format compliance, orbital element accuracy, checksum
correctness, and port compliance for the StellariumExporter adapter.
"""
import math
import os
import tempfile
from datetime import datetime, timezone

import pytest

from humeris.domain.constellation import (
    Satellite,
    ShellConfig,
    generate_walker_shell,
)
from humeris.domain.orbital_mechanics import OrbitalConstants
from humeris.ports.export import SatelliteExporter
from humeris.adapters.stellarium_exporter import StellariumExporter


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

SHELL = ShellConfig(
    altitude_km=550,
    inclination_deg=53,
    num_planes=2,
    sats_per_plane=3,
    phase_factor=1,
    raan_offset_deg=0,
    shell_name="Test",
)


def _export_tle_lines(
    satellites: list[Satellite] | None = None,
    epoch: datetime = EPOCH,
    catalog_start: int = 99001,
) -> list[str]:
    """Export satellites to TLE and return all lines."""
    if satellites is None:
        satellites = generate_walker_shell(SHELL)
    exporter = StellariumExporter(catalog_start=catalog_start)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".tle", delete=False
    ) as f:
        path = f.name
    try:
        exporter.export(satellites, path, epoch=epoch)
        with open(path, encoding="utf-8") as f:
            return f.read().splitlines()
    finally:
        os.unlink(path)


def _compute_checksum(line: str) -> int:
    """Recompute TLE checksum for a line (excluding the checksum digit)."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


class TestTleFormat:
    """TLE lines conform to the 69-character fixed-width format."""

    # Each satellite now has 4 lines: comment, name, line1, line2
    _STRIDE = 4

    def test_each_tle_has_four_lines(self):
        lines = _export_tle_lines()
        assert len(lines) % self._STRIDE == 0
        assert len(lines) > 0

    def test_line1_is_69_chars(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            line1 = lines[i + 2]
            assert len(line1) == 69, f"Line 1 length {len(line1)}: '{line1}'"

    def test_line2_is_69_chars(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            line2 = lines[i + 3]
            assert len(line2) == 69, f"Line 2 length {len(line2)}: '{line2}'"

    def test_line1_starts_with_1(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            assert lines[i + 2][0] == "1"

    def test_line2_starts_with_2(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            assert lines[i + 3][0] == "2"

    def test_catalog_numbers_match(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            cat1 = lines[i + 2][2:7].strip()
            cat2 = lines[i + 3][2:7].strip()
            assert cat1 == cat2

    def test_catalog_numbers_sequential(self):
        lines = _export_tle_lines()
        for idx, i in enumerate(range(0, len(lines), self._STRIDE)):
            cat_num = int(lines[i + 2][2:7])
            assert cat_num == 99001 + idx

    def test_satellite_names_preserved(self):
        sats = generate_walker_shell(SHELL)
        lines = _export_tle_lines(satellites=sats)
        for idx, i in enumerate(range(0, len(lines), self._STRIDE)):
            assert lines[i + 1] == sats[idx].name

    def test_comment_line_starts_with_hash(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            assert lines[i].startswith("#")


class TestOrbitalElements:
    """Orbital elements in TLE match the source satellite data."""

    _STRIDE = 4

    def test_inclination_approximately_53(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            inc = float(lines[i + 3][8:16])
            assert abs(inc - 53.0) < 0.01

    def test_mean_motion_approximately_correct(self):
        """Mean motion for 550 km should be ~15.5 rev/day."""
        lines = _export_tle_lines()
        r = OrbitalConstants.R_EARTH_EQUATORIAL + 550_000
        period = 2 * math.pi * math.sqrt(r**3 / OrbitalConstants.MU_EARTH)
        expected_n = 86400.0 / period
        for i in range(0, len(lines), self._STRIDE):
            n = float(lines[i + 3][52:63])
            assert abs(n - expected_n) < 0.1, f"Mean motion {n} vs expected {expected_n}"

    def test_raan_matches_satellite(self):
        sats = generate_walker_shell(SHELL)
        lines = _export_tle_lines(satellites=sats)
        for idx, i in enumerate(range(0, len(lines), self._STRIDE)):
            raan = float(lines[i + 3][17:25])
            expected = sats[idx].raan_deg % 360.0
            assert abs(raan - expected) < 0.01, (
                f"RAAN {raan} vs expected {expected}"
            )

    def test_mean_anomaly_matches_satellite(self):
        sats = generate_walker_shell(SHELL)
        lines = _export_tle_lines(satellites=sats)
        for idx, i in enumerate(range(0, len(lines), self._STRIDE)):
            ma = float(lines[i + 3][43:51])
            expected = sats[idx].true_anomaly_deg % 360.0
            assert abs(ma - expected) < 0.01, (
                f"Mean anomaly {ma} vs expected {expected}"
            )

    def test_epoch_year_correct(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            epoch_year = int(lines[i + 2][18:20])
            assert epoch_year == 26

    def test_epoch_day_correct(self):
        """March 20, 2026 12:00:00 UTC = day 79.5"""
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            epoch_day = float(lines[i + 2][20:32])
            assert abs(epoch_day - 79.5) < 0.001


class TestChecksum:
    """TLE checksum is correct (modulo 10 digit sum)."""

    _STRIDE = 4

    def test_line1_checksum_valid(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            line1 = lines[i + 2]
            expected = _compute_checksum(line1)
            actual = int(line1[68])
            assert actual == expected, (
                f"Line 1 checksum {actual} != {expected}: '{line1}'"
            )

    def test_line2_checksum_valid(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), self._STRIDE):
            line2 = lines[i + 3]
            expected = _compute_checksum(line2)
            actual = int(line2[68])
            assert actual == expected, (
                f"Line 2 checksum {actual} != {expected}: '{line2}'"
            )


class TestReturnCount:
    """Export returns the correct satellite count."""

    def test_returns_satellite_count(self):
        sats = generate_walker_shell(SHELL)
        exporter = StellariumExporter()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tle", delete=False
        ) as f:
            path = f.name
        try:
            count = exporter.export(sats, path, epoch=EPOCH)
            assert count == len(sats)
        finally:
            os.unlink(path)

    def test_empty_list_returns_zero(self):
        exporter = StellariumExporter()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tle", delete=False
        ) as f:
            path = f.name
        try:
            count = exporter.export([], path, epoch=EPOCH)
            assert count == 0
        finally:
            os.unlink(path)

    def test_empty_list_produces_empty_file(self):
        exporter = StellariumExporter()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tle", delete=False
        ) as f:
            path = f.name
        try:
            exporter.export([], path, epoch=EPOCH)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert content == ""
        finally:
            os.unlink(path)

    def test_port_compliance(self):
        assert issubclass(StellariumExporter, SatelliteExporter)


class TestTleEdgeCases:
    """TLE format edge cases and hardening."""

    def test_epoch_microseconds_preserved(self):
        """TLE epoch must include sub-second precision."""
        from humeris.adapters.stellarium_exporter import _epoch_components
        # Epoch with 500000 microseconds (0.5 seconds)
        epoch = datetime(2026, 3, 20, 12, 30, 30, 500000, tzinfo=timezone.utc)
        year, day_frac = _epoch_components(epoch)
        # day 79, 12:30:30.5 = 79 + (12*3600 + 30*60 + 30.5) / 86400
        expected_frac = (12 * 3600 + 30 * 60 + 30.5) / 86400.0
        # The fractional part should reflect sub-second
        actual_frac = day_frac - int(day_frac)
        assert abs(actual_frac - expected_frac) < 1e-8, (
            f"Microseconds lost: expected frac={expected_frac}, got {actual_frac}"
        )

    def test_mean_motion_zero_position_does_not_crash(self):
        """_mean_motion at (0,0,0) must not crash with ZeroDivisionError."""
        from humeris.adapters.stellarium_exporter import _mean_motion
        # Should return 0.0 or raise ValueError, not ZeroDivisionError
        try:
            result = _mean_motion((0.0, 0.0, 0.0))
            assert result == 0.0
        except ValueError:
            pass  # Also acceptable

    def test_tle_catalog_overflow_rejected(self):
        """Generating >999 satellites must not produce malformed TLE lines."""
        # Create 1000 sats — catalog numbers would overflow 5 digits
        shell = ShellConfig(
            altitude_km=550, inclination_deg=53,
            num_planes=10, sats_per_plane=100,
            phase_factor=1, raan_offset_deg=0, shell_name="Overflow",
        )
        sats = generate_walker_shell(shell)
        assert len(sats) == 1000
        exporter = StellariumExporter()
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".tle", delete=False) as f:
            path = f.name
        try:
            # Should either succeed with valid TLE or raise ValueError
            exporter.export(sats, path, epoch=EPOCH)
            with open(path) as f:
                lines = f.readlines()
            # Every TLE line must be exactly 69 chars (+ newline)
            for i, line in enumerate(lines):
                stripped = line.rstrip("\n")
                if stripped and stripped[0] in ("1", "2"):
                    assert len(stripped) == 69, (
                        f"TLE line {i} has {len(stripped)} chars, expected 69: {stripped!r}"
                    )
        except ValueError:
            pass  # Acceptable: reject overflow
        finally:
            os.unlink(path)


class TestStellariumTrailingNewline:
    """Exported TLE file must end with a trailing newline."""

    def test_trailing_newline(self):
        """Exported file must end with a single newline."""
        sats = generate_walker_shell(SHELL)[:1]
        exporter = StellariumExporter()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tle", delete=False
        ) as f:
            path = f.name
        try:
            exporter.export(sats, path, epoch=EPOCH)
            with open(path, "rb") as f:
                raw = f.read()
            assert raw.endswith(b"\n"), "File must end with a newline"
        finally:
            os.unlink(path)
