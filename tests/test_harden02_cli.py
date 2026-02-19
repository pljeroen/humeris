# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""H02-R05: CLI test coverage expansion.

Tests for error handling, edge cases, and export dispatch in cli.py.
"""
import json
import os
import sys
import tempfile

import pytest


class TestCliFileNotFound:
    """CLI should produce helpful error on missing input file."""

    def test_missing_input_file_error_message(self, tmp_path, capsys, monkeypatch):
        """File not found produces specific error message."""
        from humeris.cli import main

        nonexistent = str(tmp_path / "nonexistent.json")
        output = str(tmp_path / "out.json")
        monkeypatch.setattr(sys, 'argv', ['humeris', '-i', nonexistent, '-o', output])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "not found" in captured.out.lower(), (
            f"Expected 'not found' in error output, got stderr={captured.err!r}"
        )


class TestCliPortInUse:
    """CLI should handle port-in-use gracefully."""

    def test_port_in_use_message(self, capsys, monkeypatch):
        """Port in use produces helpful error with alternative port."""
        port = 9876

        monkeypatch.setattr(sys, 'argv', ['humeris', '--serve', '--port', str(port)])

        # Mock create_viewer_server at its source module so the lazy import picks it up
        def mock_create_server(*args, **kwargs):
            e = OSError(f"[Errno 98] Address already in use")
            e.errno = 98
            raise e

        monkeypatch.setattr(
            'humeris.adapters.viewer_server.create_viewer_server',
            mock_create_server,
        )

        from humeris.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert str(port) in captured.err, (
            f"Error should mention port {port}, got: {captured.err!r}"
        )
        assert str(port + 1) in captured.err, (
            f"Error should suggest port {port + 1}, got: {captured.err!r}"
        )


class TestCliZeroSatellites:
    """CLI should handle zero satellites gracefully."""

    def test_zero_satellites_export(self, tmp_path, monkeypatch):
        """Zero satellites produces empty CSV output gracefully."""
        from humeris.adapters.csv_exporter import CsvSatelliteExporter

        path = str(tmp_path / "empty.csv")
        count = CsvSatelliteExporter().export([], path)
        assert count == 0
        # File should exist with just the header
        assert os.path.exists(path)


class TestCliExportPaths:
    """CLI export path edge cases."""

    def test_input_parsing_requires_input_and_output(self, monkeypatch):
        """CLI without --input or --output (and not --serve) should error."""
        from humeris.cli import main
        monkeypatch.setattr(sys, 'argv', ['humeris'])

        with pytest.raises(SystemExit):
            main()

    def test_serve_mode_does_not_require_input(self, monkeypatch):
        """--serve mode should not require --input."""
        from humeris.cli import main

        # Mock the serve function so it doesn't actually start a server
        def mock_run_serve(**kwargs):
            pass

        monkeypatch.setattr('humeris.cli._run_serve', mock_run_serve)
        monkeypatch.setattr(sys, 'argv', ['humeris', '--serve'])

        # Should not raise
        main()
