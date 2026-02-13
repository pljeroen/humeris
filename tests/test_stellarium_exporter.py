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

    def test_each_tle_has_three_lines(self):
        lines = _export_tle_lines()
        assert len(lines) % 3 == 0
        assert len(lines) > 0

    def test_line1_is_69_chars(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            line1 = lines[i + 1]
            assert len(line1) == 69, f"Line 1 length {len(line1)}: '{line1}'"

    def test_line2_is_69_chars(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            line2 = lines[i + 2]
            assert len(line2) == 69, f"Line 2 length {len(line2)}: '{line2}'"

    def test_line1_starts_with_1(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            assert lines[i + 1][0] == "1"

    def test_line2_starts_with_2(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            assert lines[i + 2][0] == "2"

    def test_catalog_numbers_match(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            cat1 = lines[i + 1][2:7].strip()
            cat2 = lines[i + 2][2:7].strip()
            assert cat1 == cat2

    def test_catalog_numbers_sequential(self):
        lines = _export_tle_lines()
        for idx, i in enumerate(range(0, len(lines), 3)):
            cat_num = int(lines[i + 1][2:7])
            assert cat_num == 99001 + idx

    def test_satellite_names_preserved(self):
        sats = generate_walker_shell(SHELL)
        lines = _export_tle_lines(satellites=sats)
        for idx, i in enumerate(range(0, len(lines), 3)):
            assert lines[i] == sats[idx].name


class TestOrbitalElements:
    """Orbital elements in TLE match the source satellite data."""

    def test_inclination_approximately_53(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            inc = float(lines[i + 2][8:16])
            assert abs(inc - 53.0) < 0.01

    def test_mean_motion_approximately_correct(self):
        """Mean motion for 550 km should be ~15.5 rev/day."""
        lines = _export_tle_lines()
        r = OrbitalConstants.R_EARTH + 550_000
        period = 2 * math.pi * math.sqrt(r**3 / OrbitalConstants.MU_EARTH)
        expected_n = 86400.0 / period
        for i in range(0, len(lines), 3):
            n = float(lines[i + 2][52:63])
            assert abs(n - expected_n) < 0.1, f"Mean motion {n} vs expected {expected_n}"

    def test_raan_matches_satellite(self):
        sats = generate_walker_shell(SHELL)
        lines = _export_tle_lines(satellites=sats)
        for idx, i in enumerate(range(0, len(lines), 3)):
            raan = float(lines[i + 2][17:25])
            expected = sats[idx].raan_deg % 360.0
            assert abs(raan - expected) < 0.01, (
                f"RAAN {raan} vs expected {expected}"
            )

    def test_mean_anomaly_matches_satellite(self):
        sats = generate_walker_shell(SHELL)
        lines = _export_tle_lines(satellites=sats)
        for idx, i in enumerate(range(0, len(lines), 3)):
            ma = float(lines[i + 2][43:51])
            expected = sats[idx].true_anomaly_deg % 360.0
            assert abs(ma - expected) < 0.01, (
                f"Mean anomaly {ma} vs expected {expected}"
            )

    def test_epoch_year_correct(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            epoch_year = int(lines[i + 1][18:20])
            assert epoch_year == 26

    def test_epoch_day_correct(self):
        """March 20, 2026 12:00:00 UTC = day 79.5"""
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            epoch_day = float(lines[i + 1][20:32])
            assert abs(epoch_day - 79.5) < 0.001


class TestChecksum:
    """TLE checksum is correct (modulo 10 digit sum)."""

    def test_line1_checksum_valid(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            line1 = lines[i + 1]
            expected = _compute_checksum(line1)
            actual = int(line1[68])
            assert actual == expected, (
                f"Line 1 checksum {actual} != {expected}: '{line1}'"
            )

    def test_line2_checksum_valid(self):
        lines = _export_tle_lines()
        for i in range(0, len(lines), 3):
            line2 = lines[i + 2]
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
