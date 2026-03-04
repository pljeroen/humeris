# Copyright (c) Jeroen Visser. MIT License.
"""Tests for humeris.gui — GUI export tool logic (non-GUI parts)."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestFormatSpec:
    """FormatSpec dataclass and FORMAT_SPECS registry."""

    def test_format_spec_has_required_fields(self):
        from humeris.gui import FormatSpec

        spec = FormatSpec(
            key="csv",
            label="CSV",
            description="Spreadsheet data",
            extension=".csv",
            default_filename="constellation.csv",
            exporter_factory=lambda **kw: MagicMock(),
            options=[],
        )
        assert spec.key == "csv"
        assert spec.extension == ".csv"

    def test_option_spec_has_required_fields(self):
        from humeris.gui import OptionSpec

        opt = OptionSpec(
            key="include_orbits",
            label="Show orbit lines",
            type="bool",
            default=True,
            exporter_kwarg="include_orbits",
        )
        assert opt.key == "include_orbits"
        assert opt.type == "bool"
        assert opt.default is True

    def test_format_specs_has_nine_entries(self):
        from humeris.gui import FORMAT_SPECS

        assert len(FORMAT_SPECS) == 9

    def test_format_specs_keys_unique(self):
        from humeris.gui import FORMAT_SPECS

        keys = [s.key for s in FORMAT_SPECS]
        assert len(keys) == len(set(keys))

    def test_format_specs_all_have_extensions(self):
        from humeris.gui import FORMAT_SPECS

        for spec in FORMAT_SPECS:
            assert spec.extension.startswith("."), f"{spec.key} extension missing dot"

    def test_format_specs_expected_keys(self):
        from humeris.gui import FORMAT_SPECS

        keys = {s.key for s in FORMAT_SPECS}
        expected = {"csv", "geojson", "kml", "celestia", "stellarium",
                    "blender", "spaceengine", "ksp", "ubox"}
        assert keys == expected

    def test_format_specs_expected_extensions(self):
        from humeris.gui import FORMAT_SPECS

        ext_map = {s.key: s.extension for s in FORMAT_SPECS}
        assert ext_map["csv"] == ".csv"
        assert ext_map["geojson"] == ".geojson"
        assert ext_map["kml"] == ".kml"
        assert ext_map["celestia"] == ".ssc"
        assert ext_map["stellarium"] == ".tle"
        assert ext_map["blender"] == ".py"
        assert ext_map["spaceengine"] == ".sc"
        assert ext_map["ksp"] == ".sfs"
        assert ext_map["ubox"] == ".ubox"


class TestExporterFactories:
    """Each FormatSpec factory returns a valid exporter."""

    def test_csv_factory_returns_exporter(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "csv")
        exporter = spec.exporter_factory()
        assert hasattr(exporter, "export")

    def test_geojson_factory_returns_exporter(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "geojson")
        exporter = spec.exporter_factory()
        assert hasattr(exporter, "export")

    def test_kml_factory_returns_exporter(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "kml")
        exporter = spec.exporter_factory()
        assert hasattr(exporter, "export")

    def test_kml_factory_passes_options(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "kml")
        exporter = spec.exporter_factory(include_orbits=True, include_planes=True)
        assert exporter._include_orbits is True
        assert exporter._include_planes is True

    def test_celestia_factory_returns_exporter(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "celestia")
        exporter = spec.exporter_factory()
        assert hasattr(exporter, "export")

    def test_stellarium_factory_returns_exporter(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "stellarium")
        exporter = spec.exporter_factory()
        assert hasattr(exporter, "export")

    def test_blender_factory_returns_exporter(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "blender")
        exporter = spec.exporter_factory()
        assert hasattr(exporter, "export")

    def test_blender_factory_passes_options(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "blender")
        exporter = spec.exporter_factory(include_orbits=False, color_by_plane=True)
        assert exporter._include_orbits is False
        assert exporter._color_by_plane is True

    def test_spaceengine_factory_returns_exporter(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "spaceengine")
        exporter = spec.exporter_factory()
        assert hasattr(exporter, "export")

    def test_ksp_factory_returns_exporter(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "ksp")
        exporter = spec.exporter_factory()
        assert hasattr(exporter, "export")

    def test_ksp_factory_passes_scale_option(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "ksp")
        exporter = spec.exporter_factory(scale_to_kerbin=False)
        assert exporter._scale is False

    def test_ubox_factory_returns_exporter(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "ubox")
        exporter = spec.exporter_factory()
        assert hasattr(exporter, "export")


class TestFormatOptions:
    """Format-specific options are correctly defined."""

    def test_kml_has_orbit_and_plane_options(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "kml")
        opt_keys = {o.key for o in spec.options}
        assert "include_orbits" in opt_keys
        assert "include_planes" in opt_keys
        assert "include_isl" in opt_keys

    def test_blender_has_orbit_and_color_options(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "blender")
        opt_keys = {o.key for o in spec.options}
        assert "include_orbits" in opt_keys
        assert "color_by_plane" in opt_keys

    def test_ksp_has_scale_option(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "ksp")
        opt_keys = {o.key for o in spec.options}
        assert "scale_to_kerbin" in opt_keys

    def test_csv_has_no_options(self):
        from humeris.gui import FORMAT_SPECS

        spec = next(s for s in FORMAT_SPECS if s.key == "csv")
        assert spec.options == []

    def test_all_options_have_labels(self):
        from humeris.gui import FORMAT_SPECS

        for spec in FORMAT_SPECS:
            for opt in spec.options:
                assert opt.label, f"{spec.key}.{opt.key} missing label"


class TestDefaultSatellites:
    """Default satellite loading produces expected count."""

    def test_load_default_satellites_returns_list(self):
        from humeris.gui import load_default_satellites

        sats = load_default_satellites()
        assert isinstance(sats, list)
        assert len(sats) > 0

    def test_load_default_satellites_count(self):
        from humeris.gui import load_default_satellites

        sats = load_default_satellites()
        # Default shells: 3 × (22 planes × 72 sats) + SSO band
        # Exact count depends on SSO config but should be ~4752
        assert len(sats) > 4000


class TestExportOrchestration:
    """Export orchestration passes correct args to exporters."""

    def test_run_export_calls_exporter(self):
        from humeris.gui import run_export, FormatSpec

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = 10

        spec = FormatSpec(
            key="test",
            label="Test",
            description="Test format",
            extension=".test",
            default_filename="test.test",
            exporter_factory=lambda **kw: mock_exporter,
            options=[],
        )

        sats = [MagicMock()]  # fake satellite
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_export(sats, spec, tmpdir, "out.test", {})
            assert result == 10
            mock_exporter.export.assert_called_once()
            call_args = mock_exporter.export.call_args
            assert call_args[0][0] == sats  # satellites
            assert tmpdir in call_args[0][1]  # path contains dir

    def test_run_export_passes_options_to_factory(self):
        from humeris.gui import run_export, FormatSpec, OptionSpec

        factory_kwargs = {}

        def capture_factory(**kwargs):
            factory_kwargs.update(kwargs)
            mock = MagicMock()
            mock.export.return_value = 5
            return mock

        spec = FormatSpec(
            key="test",
            label="Test",
            description="Test format",
            extension=".test",
            default_filename="test.test",
            exporter_factory=capture_factory,
            options=[
                OptionSpec(
                    key="show_lines",
                    label="Show lines",
                    type="bool",
                    default=False,
                    exporter_kwarg="include_lines",
                ),
            ],
        )

        sats = [MagicMock()]
        with tempfile.TemporaryDirectory() as tmpdir:
            run_export(sats, spec, tmpdir, "out.test", {"show_lines": True})
            assert factory_kwargs["include_lines"] is True


class TestCelesTrakGroups:
    """CelesTrak group list for the combobox."""

    def test_celestrak_groups_is_list(self):
        from humeris.gui import CELESTRAK_GROUPS

        assert isinstance(CELESTRAK_GROUPS, list)
        assert len(CELESTRAK_GROUPS) > 5

    def test_celestrak_groups_contains_starlink(self):
        from humeris.gui import CELESTRAK_GROUPS

        assert "STARLINK" in CELESTRAK_GROUPS


class TestGuiSmoke:
    """GUI smoke test — build and destroy window."""

    @pytest.mark.skipif(
        not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
        reason="No display available (headless CI)",
    )
    def test_gui_window_creates_and_destroys(self):
        from humeris.gui import HumerisGui

        gui = HumerisGui()
        # Verify key widgets exist
        assert gui.root is not None
        assert gui.root.title() == "Humeris — Satellite Constellation Export"
        gui.root.destroy()
