# Copyright (c) 2026 Jeroen Michaël Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for Cesium HTML viewer adapter.

Generates a self-contained HTML file embedding CesiumJS (CDN) and CZML data.
Open in any browser to see animated 3D satellite orbits.
"""

import ast
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from constellation_generator import (
    ShellConfig,
    generate_walker_shell,
    derive_orbital_state,
    GroundTrackPoint,
    CoveragePoint,
)
from constellation_generator.adapters.czml_exporter import (
    constellation_packets,
    ground_track_packets,
    coverage_packets,
)


EPOCH = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)


def _make_packets(n_sats=4):
    shell = ShellConfig(
        altitude_km=550, inclination_deg=53,
        num_planes=2, sats_per_plane=n_sats // 2,
        phase_factor=1, raan_offset_deg=0, shell_name="Test",
    )
    sats = generate_walker_shell(shell)
    states = [derive_orbital_state(s, EPOCH) for s in sats]
    return constellation_packets(
        states, EPOCH, timedelta(hours=2), timedelta(seconds=60),
    )


@pytest.fixture
def czml_packets():
    return _make_packets()


@pytest.fixture
def minimal_packets():
    """Minimal document-only CZML."""
    return [{"id": "document", "name": "Test", "version": "1.0"}]


class TestGenerateCesiumHtml:
    """Tests for generate_cesium_html function."""

    def test_returns_string(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        result = generate_cesium_html(czml_packets)
        assert isinstance(result, str)

    def test_contains_doctype(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "<!DOCTYPE html>" in html

    def test_contains_cesium_script(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "cesium.com" in html.lower() or "Cesium.js" in html or "Cesium" in html
        assert "<script" in html

    def test_contains_cesium_css(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "Widgets/widgets.css" in html

    def test_czml_data_embedded(self, czml_packets):
        """The CZML JSON data must be findable in the HTML output."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        # The document packet id should appear
        assert '"id": "document"' in html or '"id":"document"' in html

    def test_czml_data_is_valid_json(self, czml_packets):
        """Extract the embedded CZML and verify it parses as JSON matching input."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        # Find JSON between markers
        match = re.search(r'var\s+czmlData\s*=\s*(\[.*?\]);\s*$', html, re.DOTALL | re.MULTILINE)
        assert match is not None, "Could not find czmlData variable in HTML"
        embedded = json.loads(match.group(1))
        assert len(embedded) == len(czml_packets)
        assert embedded[0]["id"] == "document"

    def test_custom_title(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets, title="My Constellation")
        assert "<title>My Constellation</title>" in html

    def test_default_title(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "<title>" in html

    def test_custom_token_embedded(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets, cesium_token="MY_SECRET_TOKEN_123")
        assert "MY_SECRET_TOKEN_123" in html

    def test_empty_packets(self):
        """Empty CZML list still produces valid HTML."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html([])
        assert "<!DOCTYPE html>" in html
        assert "czmlData" in html

    def test_document_only_packets(self, minimal_packets):
        """Document-only CZML produces valid HTML."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(minimal_packets)
        assert "<!DOCTYPE html>" in html

    def test_ground_track_packets(self):
        """Ground track CZML embeds correctly."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        track = [
            GroundTrackPoint(
                time=EPOCH + timedelta(seconds=i * 60),
                lat_deg=10.0 + i, lon_deg=20.0 + i, alt_km=550.0,
            )
            for i in range(5)
        ]
        pkts = ground_track_packets(track)
        html = generate_cesium_html(pkts)
        assert "ground-track" in html

    def test_coverage_packets(self):
        """Coverage CZML embeds correctly."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        points = [
            CoveragePoint(lat_deg=0.0, lon_deg=0.0, visible_count=3),
            CoveragePoint(lat_deg=10.0, lon_deg=10.0, visible_count=5),
        ]
        pkts = coverage_packets(points, lat_step_deg=10, lon_step_deg=10)
        html = generate_cesium_html(pkts)
        assert "coverage-" in html

    def test_html_closes_all_tags(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "</html>" in html
        assert "</body>" in html
        assert "</head>" in html


class TestMultiLayerHtml:
    """Tests for additional_layers parameter."""

    def test_additional_layers_embedded(self, czml_packets):
        """Additional layers appear in the HTML output."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        track_pkts = ground_track_packets([
            GroundTrackPoint(
                time=EPOCH + timedelta(seconds=i * 60),
                lat_deg=10.0 + i, lon_deg=20.0 + i, alt_km=550.0,
            )
            for i in range(3)
        ])
        html = generate_cesium_html(
            czml_packets, additional_layers=[track_pkts],
        )
        assert "ground-track" in html
        assert "czmlLayers" in html

    def test_multiple_additional_layers(self, czml_packets):
        """Multiple additional layers all embedded."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        track_pkts = ground_track_packets([
            GroundTrackPoint(
                time=EPOCH + timedelta(seconds=i * 60),
                lat_deg=10.0 + i, lon_deg=20.0 + i, alt_km=550.0,
            )
            for i in range(3)
        ])
        cov_pkts = coverage_packets(
            [CoveragePoint(lat_deg=0.0, lon_deg=0.0, visible_count=3)],
            lat_step_deg=10, lon_step_deg=10,
        )
        html = generate_cesium_html(
            czml_packets, additional_layers=[track_pkts, cov_pkts],
        )
        assert "ground-track" in html
        assert "coverage-" in html

    def test_no_additional_layers_backward_compatible(self, czml_packets):
        """No additional_layers → same as before."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "<!DOCTYPE html>" in html
        assert "czmlData" in html


class TestWriteCesiumHtml:
    """Tests for write_cesium_html function."""

    def test_creates_file(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import write_cesium_html
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            write_cesium_html(czml_packets, path)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "<!DOCTYPE html>" in content
        finally:
            os.unlink(path)

    def test_returns_path(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import write_cesium_html
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result = write_cesium_html(czml_packets, path)
            assert result == path
        finally:
            os.unlink(path)

    def test_file_is_valid_html(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import write_cesium_html
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            write_cesium_html(czml_packets, path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert content.startswith("<!DOCTYPE html>")
            assert "</html>" in content
        finally:
            os.unlink(path)

    def test_title_and_token_forwarded(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import write_cesium_html
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            write_cesium_html(
                czml_packets, path,
                title="Custom Title", cesium_token="TOKEN_XYZ",
            )
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "Custom Title" in content
            assert "TOKEN_XYZ" in content
        finally:
            os.unlink(path)

    def test_deterministic_output(self, czml_packets):
        """Same input produces identical output."""
        from constellation_generator.adapters.cesium_viewer import write_cesium_html
        paths = []
        for _ in range(2):
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                path = f.name
            paths.append(path)
            write_cesium_html(czml_packets, path)
        try:
            with open(paths[0], encoding="utf-8") as f1, open(paths[1], encoding="utf-8") as f2:
                assert f1.read() == f2.read()
        finally:
            for p in paths:
                os.unlink(p)


class TestSecurityHardening:
    """Security: input sanitization to prevent XSS/injection in HTML output."""

    def test_title_html_escaped(self):
        """Title containing HTML tags must be escaped in HTML contexts."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(
            [{"id": "document", "version": "1.0"}],
            title='<img src=x onerror=alert(1)>',
        )
        assert '<img src=x onerror=alert(1)>' not in html

    def test_cesium_token_quotes_escaped(self):
        """Cesium token with quotes must be escaped in JS string context."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(
            [{"id": "document", "version": "1.0"}],
            cesium_token='token"inject',
        )
        # Raw unescaped double quote in token would break JS string literal
        assert 'token"inject' not in html

    def test_czml_script_tag_escape(self):
        """CZML data containing </script> must not break HTML structure."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        packets = [
            {"id": "document", "version": "1.0", "note": "</script><script>alert(1)</script>"}
        ]
        html = generate_cesium_html(packets)
        # Exactly 2 closing script tags expected: CDN + inline
        close_count = html.lower().count('</script>')
        assert close_count == 2, f"Expected 2 </script> tags, got {close_count}"

    def test_title_preserves_safe_content(self):
        """Safe title content is preserved after escaping (regression guard)."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(
            [{"id": "document", "version": "1.0"}],
            title='My Safe Title',
        )
        assert 'My Safe Title' in html


class TestLayerSelector:
    """Tests for layer selector UI in generated HTML."""

    def test_layer_selector_div_present(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert 'id="layerSelector"' in html

    def test_layer_selector_css_present(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "#layerSelector" in html

    def test_layer_selector_js_present(self, czml_packets):
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "buildPanel" in html


class TestCategorizedLayerPanel:
    """Tests for categorized layer UI with grouped layers."""

    def test_categories_extracted_from_colon_names(self):
        """Layer names with 'Category:Name' split on ':' for grouping."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        layer1 = [{"id": "document", "name": "Constellation:Walker", "version": "1.0"},
                  {"id": "sat-0", "point": {}}]
        layer2 = [{"id": "document", "name": "Constellation:Starlink", "version": "1.0"},
                  {"id": "sat-1", "point": {}}]
        layer3 = [{"id": "document", "name": "Analysis:Eclipse", "version": "1.0"},
                  {"id": "sat-2", "point": {}}]
        primary = [{"id": "document", "name": "Primary", "version": "1.0"}]
        html = generate_cesium_html(primary, additional_layers=[layer1, layer2, layer3])
        # JS creates details elements grouped by category prefix
        assert 'createElement("details")' in html
        assert '.split(":")' in html
        # Category names appear in embedded layer data
        assert "Constellation:Walker" in html
        assert "Analysis:Eclipse" in html

    def test_show_hide_all_buttons(self):
        """Each category has Show All / Hide All buttons."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        layer1 = [{"id": "document", "name": "Constellation:Walker", "version": "1.0"},
                  {"id": "sat-0", "point": {}}]
        primary = [{"id": "document", "name": "Primary", "version": "1.0"}]
        html = generate_cesium_html(primary, additional_layers=[layer1])
        assert "Show All" in html
        assert "Hide All" in html

    def test_request_render_mode_present(self):
        """requestRenderMode: true should be in viewer options for GPU savings."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html([{"id": "document", "name": "Test", "version": "1.0"}])
        assert "requestRenderMode" in html

    def test_entity_count_shown(self):
        """Layer labels should show entity count."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        layer1 = [{"id": "document", "name": "Constellation:Walker", "version": "1.0"},
                  {"id": "sat-0", "point": {}},
                  {"id": "sat-1", "point": {}}]
        primary = [{"id": "document", "name": "Primary", "version": "1.0"}]
        html = generate_cesium_html(primary, additional_layers=[layer1])
        # Entity count should appear in JS that builds labels
        assert "entities" in html.lower()

    def test_backward_compatible_no_layers(self):
        """Without additional_layers, HTML still works (no categories needed)."""
        from constellation_generator.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html([{"id": "document", "name": "Test", "version": "1.0"}])
        assert "<!DOCTYPE html>" in html
        assert "requestRenderMode" in html


class TestGenerateInteractiveHtml:
    """Tests for generate_interactive_html function (server-mode viewer)."""

    def test_returns_string(self):
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        result = generate_interactive_html()
        assert isinstance(result, str)

    def test_contains_doctype(self):
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "<!DOCTYPE html>" in html

    def test_contains_cesium_script(self):
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "Cesium" in html
        assert "<script" in html

    def test_no_embedded_czml(self):
        """Interactive HTML fetches CZML from server, not embedded inline."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "czmlData" not in html

    def test_fetch_based_api_calls(self):
        """JavaScript uses fetch() to communicate with server API."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "fetch(" in html
        assert "/api/" in html

    def test_add_layer_forms_present(self):
        """Add-layer forms for Walker, CelesTrak, Ground Station."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "Walker" in html
        assert "CelesTrak" in html
        assert "Ground Station" in html

    def test_walker_form_fields(self):
        """Walker form has altitude, inclination, planes, sats_per_plane."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "altitude" in html.lower()
        assert "inclination" in html.lower()

    def test_analysis_layer_types_available(self):
        """Analysis layer types (eclipse, coverage, etc.) accessible from UI."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        for layer_type in ["eclipse", "coverage"]:
            assert layer_type in html.lower(), f"Missing analysis type: {layer_type}"

    def test_mode_toggle_present(self):
        """Snap/Anim toggle for layers."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "snapshot" in html.lower() or "snap" in html.lower()
        assert "animated" in html.lower() or "anim" in html.lower()

    def test_custom_title(self):
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html(title="My Viewer")
        assert "My Viewer" in html

    def test_custom_token(self):
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html(cesium_token="MY_TOKEN_123")
        assert "MY_TOKEN_123" in html

    def test_custom_port(self):
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html(port=9999)
        assert "9999" in html

    def test_no_request_render_mode(self):
        """Interactive viewer should NOT use requestRenderMode (causes blank globe)."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "requestRenderMode" not in html

    def test_should_animate_true(self):
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "shouldAnimate: true" in html

    def test_request_render_calls(self):
        """After data source changes, requestRender() is called."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "requestRender()" in html

    def test_html_closes_all_tags(self):
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "</html>" in html
        assert "</body>" in html
        assert "</head>" in html

    def test_ground_station_presets(self):
        """Ground station form has presets for common stations."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "Svalbard" in html

    def test_layer_remove_button(self):
        """Each layer should have a remove mechanism."""
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        # Remove button or delete action
        assert "removeLayer" in html or "DELETE" in html

    def test_security_title_escaped(self):
        from constellation_generator.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html(title='<script>alert(1)</script>')
        assert '<script>alert(1)</script>' not in html


class TestCesiumViewerPurity:
    """Adapter purity: only stdlib + internal imports allowed."""

    def test_no_external_deps(self):
        import constellation_generator.adapters.cesium_viewer as mod

        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {"json", "math", "datetime", "string", "html"}
        allowed_internal = {"constellation_generator"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    top = node.module.split(".")[0]
                    assert top in allowed_stdlib or top in allowed_internal, \
                        f"Forbidden import from: {node.module}"
