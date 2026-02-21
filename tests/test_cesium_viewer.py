# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
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

from humeris.domain.constellation import ShellConfig, generate_walker_shell
from humeris.domain.propagation import derive_orbital_state
from humeris.domain.ground_track import GroundTrackPoint
from humeris.domain.coverage import CoveragePoint
from humeris.adapters.czml_exporter import (
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
        from humeris.adapters.cesium_viewer import generate_cesium_html
        result = generate_cesium_html(czml_packets)
        assert isinstance(result, str)

    def test_contains_doctype(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "<!DOCTYPE html>" in html

    def test_contains_cesium_script(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "cesium.com" in html.lower() or "Cesium.js" in html or "Cesium" in html
        assert "<script" in html

    def test_contains_cesium_css(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "Widgets/widgets.css" in html

    def test_czml_data_embedded(self, czml_packets):
        """The CZML JSON data must be findable in the HTML output."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        # The document packet id should appear
        assert '"id": "document"' in html or '"id":"document"' in html

    def test_czml_data_is_valid_json(self, czml_packets):
        """Extract the embedded CZML and verify it parses as JSON matching input."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        # Find JSON between markers
        match = re.search(r'var\s+czmlData\s*=\s*(\[.*?\]);\s*$', html, re.DOTALL | re.MULTILINE)
        assert match is not None, "Could not find czmlData variable in HTML"
        embedded = json.loads(match.group(1))
        assert len(embedded) == len(czml_packets)
        assert embedded[0]["id"] == "document"

    def test_custom_title(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets, title="My Constellation")
        assert "<title>My Constellation</title>" in html

    def test_default_title(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "<title>" in html

    def test_custom_token_embedded(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets, cesium_token="MY_SECRET_TOKEN_123")
        assert "MY_SECRET_TOKEN_123" in html

    def test_empty_packets(self):
        """Empty CZML list still produces valid HTML."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html([])
        assert "<!DOCTYPE html>" in html
        assert "czmlData" in html

    def test_document_only_packets(self, minimal_packets):
        """Document-only CZML produces valid HTML."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(minimal_packets)
        assert "<!DOCTYPE html>" in html

    def test_ground_track_packets(self):
        """Ground track CZML embeds correctly."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
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
        from humeris.adapters.cesium_viewer import generate_cesium_html
        points = [
            CoveragePoint(lat_deg=0.0, lon_deg=0.0, visible_count=3),
            CoveragePoint(lat_deg=10.0, lon_deg=10.0, visible_count=5),
        ]
        pkts = coverage_packets(points, lat_step_deg=10, lon_step_deg=10)
        html = generate_cesium_html(pkts)
        assert "coverage-" in html

    def test_html_closes_all_tags(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "</html>" in html
        assert "</body>" in html
        assert "</head>" in html


class TestMultiLayerHtml:
    """Tests for additional_layers parameter."""

    def test_additional_layers_embedded(self, czml_packets):
        """Additional layers appear in the HTML output."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
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
        from humeris.adapters.cesium_viewer import generate_cesium_html
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
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "<!DOCTYPE html>" in html
        assert "czmlData" in html


class TestWriteCesiumHtml:
    """Tests for write_cesium_html function."""

    def test_creates_file(self, czml_packets):
        from humeris.adapters.cesium_viewer import write_cesium_html
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
        from humeris.adapters.cesium_viewer import write_cesium_html
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result = write_cesium_html(czml_packets, path)
            assert result == path
        finally:
            os.unlink(path)

    def test_file_is_valid_html(self, czml_packets):
        from humeris.adapters.cesium_viewer import write_cesium_html
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
        from humeris.adapters.cesium_viewer import write_cesium_html
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
        from humeris.adapters.cesium_viewer import write_cesium_html
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
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(
            [{"id": "document", "version": "1.0"}],
            title='<img src=x onerror=alert(1)>',
        )
        assert '<img src=x onerror=alert(1)>' not in html

    def test_cesium_token_quotes_escaped(self):
        """Cesium token with quotes must be escaped in JS string context."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(
            [{"id": "document", "version": "1.0"}],
            cesium_token='token"inject',
        )
        # Raw unescaped double quote in token would break JS string literal
        assert 'token"inject' not in html

    def test_czml_script_tag_escape(self):
        """CZML data containing </script> must not break HTML structure."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        packets = [
            {"id": "document", "version": "1.0", "note": "</script><script>alert(1)</script>"}
        ]
        html = generate_cesium_html(packets)
        # Exactly 2 closing script tags expected: CDN + inline
        close_count = html.lower().count('</script>')
        assert close_count == 2, f"Expected 2 </script> tags, got {close_count}"

    def test_title_preserves_safe_content(self):
        """Safe title content is preserved after escaping (regression guard)."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(
            [{"id": "document", "version": "1.0"}],
            title='My Safe Title',
        )
        assert 'My Safe Title' in html


class TestLayerSelector:
    """Tests for layer selector UI in generated HTML."""

    def test_layer_selector_div_present(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert 'id="layerSelector"' in html

    def test_layer_selector_css_present(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "#layerSelector" in html

    def test_layer_selector_js_present(self, czml_packets):
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html(czml_packets)
        assert "buildPanel" in html


class TestCategorizedLayerPanel:
    """Tests for categorized layer UI with grouped layers."""

    def test_categories_extracted_from_colon_names(self):
        """Layer names with 'Category:Name' split on ':' for grouping."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
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
        from humeris.adapters.cesium_viewer import generate_cesium_html
        layer1 = [{"id": "document", "name": "Constellation:Walker", "version": "1.0"},
                  {"id": "sat-0", "point": {}}]
        primary = [{"id": "document", "name": "Primary", "version": "1.0"}]
        html = generate_cesium_html(primary, additional_layers=[layer1])
        assert "Show All" in html
        assert "Hide All" in html

    def test_request_render_mode_present(self):
        """requestRenderMode: true should be in viewer options for GPU savings."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html([{"id": "document", "name": "Test", "version": "1.0"}])
        assert "requestRenderMode" in html

    def test_entity_count_shown(self):
        """Layer labels should show entity count."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        layer1 = [{"id": "document", "name": "Constellation:Walker", "version": "1.0"},
                  {"id": "sat-0", "point": {}},
                  {"id": "sat-1", "point": {}}]
        primary = [{"id": "document", "name": "Primary", "version": "1.0"}]
        html = generate_cesium_html(primary, additional_layers=[layer1])
        # Entity count should appear in JS that builds labels
        assert "entities" in html.lower()

    def test_backward_compatible_no_layers(self):
        """Without additional_layers, HTML still works (no categories needed)."""
        from humeris.adapters.cesium_viewer import generate_cesium_html
        html = generate_cesium_html([{"id": "document", "name": "Test", "version": "1.0"}])
        assert "<!DOCTYPE html>" in html
        assert "requestRenderMode" in html


class TestGenerateInteractiveHtml:
    """Tests for generate_interactive_html function (server-mode viewer)."""

    def test_returns_string(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        result = generate_interactive_html()
        assert isinstance(result, str)

    def test_contains_doctype(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "<!DOCTYPE html>" in html

    def test_contains_cesium_script(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "Cesium" in html
        assert "<script" in html

    def test_no_embedded_czml(self):
        """Interactive HTML fetches CZML from server, not embedded inline."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "czmlData" not in html

    def test_fetch_based_api_calls(self):
        """JavaScript uses fetch() to communicate with server API."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "fetch(" in html
        assert "/api/" in html

    def test_add_layer_forms_present(self):
        """Add-layer forms for Walker, CelesTrak, Ground Station."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "Walker" in html
        assert "CelesTrak" in html
        assert "Ground Station" in html

    def test_walker_form_fields(self):
        """Walker form has altitude, inclination, planes, sats_per_plane."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "altitude" in html.lower()
        assert "inclination" in html.lower()

    def test_analysis_layer_types_available(self):
        """Analysis layer types (eclipse, coverage, etc.) accessible from UI."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        for layer_type in ["eclipse", "coverage"]:
            assert layer_type in html.lower(), f"Missing analysis type: {layer_type}"

    def test_mode_toggle_present(self):
        """Snap/Anim toggle for layers."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "snapshot" in html.lower() or "snap" in html.lower()
        assert "animated" in html.lower() or "anim" in html.lower()

    def test_custom_title(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html(title="My Viewer")
        assert "My Viewer" in html

    def test_custom_token(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html(cesium_token="MY_TOKEN_123")
        assert "MY_TOKEN_123" in html

    def test_custom_port(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html(port=9999)
        assert "9999" in html

    def test_no_request_render_mode(self):
        """Interactive viewer should NOT use requestRenderMode (causes blank globe)."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "requestRenderMode" not in html

    def test_should_animate_true(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "shouldAnimate: true" in html

    def test_request_render_calls(self):
        """After data source changes, requestRender() is called."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "requestRender()" in html

    def test_html_closes_all_tags(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "</html>" in html
        assert "</body>" in html
        assert "</head>" in html

    def test_ground_station_presets(self):
        """Ground station form has presets for common stations."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "Svalbard" in html

    def test_layer_remove_button(self):
        """Each layer should have a remove mechanism."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        # Remove button or delete action
        assert "removeLayer" in html or "DELETE" in html

    def test_security_title_escaped(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html(title='<script>alert(1)</script>')
        assert '<script>alert(1)</script>' not in html


class TestInteractiveViewerQol:
    """QOL improvements for the interactive viewer UI."""

    def test_analysis_section_open_by_default(self):
        """Add Analysis section should be open (not collapsed) by default."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        # Find the Analysis details element — it should have 'open'
        # The section contains "Add Analysis" as summary text
        import re
        match = re.search(r'<details([^>]*)>\s*<summary>Add Analysis', html)
        assert match is not None, "Add Analysis section not found"
        assert "open" in match.group(1), "Add Analysis section should be open by default"

    def test_conjunction_button_present(self):
        """Conjunction analysis button should be in the analysis grid."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "addAnalysis('conjunction')" in html, "Conjunction button missing"

    def test_analysis_buttons_have_tooltips(self):
        """All analysis buttons should have title attributes with descriptions."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        import re
        buttons = re.findall(r'<button[^>]*onclick="addAnalysis\([^)]+\)"[^>]*>', html)
        assert len(buttons) >= 21, f"Expected >=21 analysis buttons, found {len(buttons)}"
        for btn in buttons:
            assert 'title="' in btn, f"Button missing tooltip: {btn}"

    def test_toast_notification_system(self):
        """Toast notification system should replace alert() calls."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "showToast" in html, "showToast function not found"
        assert "toast-container" in html or "toastContainer" in html, \
            "Toast container not found"
        # No raw alert() calls (except possibly in Cesium library references)
        import re
        js_alerts = re.findall(r'(?<!\.)alert\(', html)
        assert len(js_alerts) == 0, f"Found {len(js_alerts)} raw alert() calls"

    def test_css_spinner_animation(self):
        """Loading indicator should have a CSS spinner animation."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "@keyframes" in html, "No CSS animation keyframes found"
        assert "spin" in html.lower(), "No spin animation found"

    def test_panel_toggle_button(self):
        """Panel toggle button should exist to show/hide side panel."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "togglePanel" in html, "togglePanel function not found"

    def test_expanded_celestrak_groups(self):
        """CelesTrak dropdown should include additional groups."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        for group in ["ACTIVE", "WEATHER", "GEO", "AMATEUR", "SCIENCE", "NOAA"]:
            assert f'value="{group}"' in html, \
                f"CelesTrak group {group} missing from dropdown"


class TestInteractiveViewerQolPass2:
    """QOL Pass 2: Parameter forms, legends, cap display, controls, export, stats."""

    # --- Tier 1: Analysis parameter forms ---

    def test_coverage_param_fields_present(self):
        """Coverage analysis should have lat_step, lon_step, min_elevation inputs."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert 'id="param-lat-step"' in html, "Missing lat_step parameter field"
        assert 'id="param-lon-step"' in html, "Missing lon_step parameter field"
        assert 'id="param-min-elev"' in html, "Missing min_elevation parameter field"

    def test_isl_param_fields_present(self):
        """ISL analysis should have max_range_km input."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert 'id="param-max-range"' in html, "Missing max_range parameter field"

    def test_drag_param_fields_present(self):
        """Drag-based analyses should have cd, area, mass inputs."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert 'id="param-cd"' in html, "Missing cd parameter field"
        assert 'id="param-area"' in html, "Missing area_m2 parameter field"
        assert 'id="param-mass"' in html, "Missing mass_kg parameter field"

    def test_add_analysis_reads_params(self):
        """addAnalysis() should read parameter form values into params object."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        # JS should read param fields and pass them
        assert "param-lat-step" in html
        assert "param-lon-step" in html
        # The function should build params from form fields, not send empty {}
        # Check that addAnalysis assembles a params object from the form
        assert "gatherAnalysisParams" in html or "buildParams" in html or \
            'getElementById("param-' in html

    # --- Tier 1: Color legends ---

    def test_color_legend_container_present(self):
        """Color legend overlay container should exist in HTML."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert 'id="colorLegend"' in html or 'id="legendOverlay"' in html, \
            "Missing color legend container"

    def test_color_legend_css_present(self):
        """Color legend should have CSS styling."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "colorLegend" in html or "legendOverlay" in html or \
            "legend-overlay" in html

    def test_update_legend_function(self):
        """JS function to update legend when analysis layer selected."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "updateLegend" in html or "showLegend" in html, \
            "Missing legend update function"

    # --- Tier 1: Cap notice display ---

    def test_cap_notice_in_rebuild_panel(self):
        """rebuildPanel should display capped_from info when present."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "capped_from" in html, \
            "rebuildPanel should reference capped_from for cap notice display"

    # --- Tier 2: Global duration/step controls ---

    def test_simulation_duration_input(self):
        """Simulation section should have duration input."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert 'id="sim-duration"' in html, "Missing simulation duration input"

    def test_simulation_step_input(self):
        """Simulation section should have step size input."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert 'id="sim-step"' in html, "Missing simulation step input"

    # --- Tier 2: Export from viewer ---

    def test_export_button_in_layer_panel(self):
        """Each layer in the panel should have an export/download button."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "exportLayer" in html or "downloadLayer" in html, \
            "Missing export/download function for layers"

    # --- Tier 2: Analysis statistics ---

    def test_statistics_display_area(self):
        """Layer panel should have a statistics/metrics display area."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "layer-stats" in html or "layerStats" in html or \
            "statistics" in html.lower(), \
            "Missing statistics display area in layer panel"

    # --- Tier 3: Session save/load ---

    def test_session_save_button(self):
        """UI should have a save session button."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "saveSession" in html, "Missing save session button/function"

    def test_session_load_button(self):
        """UI should have a load session button."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "loadSession" in html, "Missing load session button/function"


class TestInteractiveViewerQolPass3:
    """QOL Pass 3: Bug fixes, validation, UX improvements, power user features."""

    # --- Bug #2: gatherAnalysisParams missing cd/area/mass ---

    def test_gather_params_includes_all_seven_fields(self):
        """gatherAnalysisParams() should read all 7 param fields including cd, area, mass."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        # The JS function should read cd, area, and mass fields
        assert "param-cd" in html  # field exists (already tested)
        # But the gatherAnalysisParams function must actually read them
        import re
        # Find gatherAnalysisParams function body
        match = re.search(r'function gatherAnalysisParams\(\)\s*\{(.*?)\}', html, re.DOTALL)
        assert match is not None, "gatherAnalysisParams function not found"
        fn_body = match.group(1)
        assert "param-cd" in fn_body, \
            "gatherAnalysisParams should read param-cd field"
        assert "param-area" in fn_body, \
            "gatherAnalysisParams should read param-area field"
        assert "param-mass" in fn_body, \
            "gatherAnalysisParams should read param-mass field"

    # --- Bug fix: form validation ---

    def test_walker_altitude_has_min_constraint(self):
        """Walker altitude input should have min attribute to prevent negative values."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        import re
        match = re.search(r'<input[^>]*id="w-alt"[^>]*>', html)
        assert match is not None, "Walker altitude input not found"
        assert 'min=' in match.group(0), \
            "Walker altitude should have min attribute"

    def test_ground_station_lat_has_range(self):
        """Ground station latitude should be constrained to -90..90."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        import re
        match = re.search(r'<input[^>]*id="gs-lat"[^>]*>', html)
        assert match is not None, "Ground station latitude input not found"
        assert 'min=' in match.group(0), \
            "Ground station latitude should have min attribute"
        assert 'max=' in match.group(0), \
            "Ground station latitude should have max attribute"

    def test_ground_station_lon_has_range(self):
        """Ground station longitude should be constrained to -180..180."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        import re
        match = re.search(r'<input[^>]*id="gs-lon"[^>]*>', html)
        assert match is not None, "Ground station longitude input not found"
        assert 'min=' in match.group(0), \
            "Ground station longitude should have min attribute"
        assert 'max=' in match.group(0), \
            "Ground station longitude should have max attribute"

    # --- Promise.all error handling ---

    def test_load_existing_layers_has_catch(self):
        """loadExistingLayers Promise.all should have .catch() handler."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        import re
        # Find the loadExistingLayers section and verify Promise.all has .catch()
        match = re.search(
            r'function loadExistingLayers.*?Promise\.all\(promises\)\.then\(.*?\)(\.[a-z]+\()',
            html, re.DOTALL,
        )
        assert match is not None, "Promise.all in loadExistingLayers not found"
        assert match.group(1) == ".catch(", \
            f"Promise.all should chain .catch(), got: {match.group(1)}"

    # --- Layer renaming ---

    def test_layer_rename_function_exists(self):
        """renameLayer or inline rename mechanism should exist."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "renameLayer" in html or "dblclick" in html or "contenteditable" in html, \
            "Layer rename mechanism not found (renameLayer function or dblclick handler)"

    # --- Timeline sync ---

    def test_apply_settings_syncs_timeline(self):
        """applySettings should update Cesium viewer.clock bounds."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        import re
        match = re.search(r'function applySettings\(\)\s*\{(.*?)\n\s*\}', html, re.DOTALL)
        assert match is not None, "applySettings function not found"
        fn_body = match.group(1)
        assert "viewer.clock" in fn_body or "clock.startTime" in fn_body, \
            "applySettings should sync Cesium viewer.clock timeline"

    # --- Cap toast warning ---

    def test_cap_toast_shown_on_capped_layer(self):
        """rebuildPanel should show toast when a layer has capped_from."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        # rebuildPanel should call showToast when capped_from is detected
        import re
        match = re.search(r'function rebuildPanel\(\)\s*\{(.*)', html, re.DOTALL)
        assert match is not None, "rebuildPanel function not found"
        # Find the content up to the next top-level function
        fn_body = match.group(1)
        # Check that capped_from triggers a toast
        assert "capped_from" in fn_body  # already passes
        # But we need toast notification when capped
        assert "showToast" in fn_body and "capped" in fn_body.lower(), \
            "rebuildPanel should show toast when layers are capped"

    # --- Cancel button / AbortController ---

    def test_abort_controller_present(self):
        """Loading overlay should have cancel/abort mechanism."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "AbortController" in html or "cancelRequest" in html or \
            "cancel" in html.lower().split("loadingindicator")[0] == False, \
            "AbortController or cancel mechanism not found"
        # More specific: loading indicator should have a cancel button
        assert "cancelRequest" in html or \
            ('id="cancelBtn"' in html) or \
            ("AbortController" in html), \
            "Cancel button or AbortController not found in loading overlay"

    # --- Layer opacity ---

    def test_layer_opacity_control(self):
        """Layer panel should have opacity slider/input for layers."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "setOpacity" in html or "opacitySlider" in html or \
            'type="range"' in html, \
            "Opacity slider/control not found in layer panel"

    # --- Keyboard shortcuts ---

    def test_keyboard_shortcuts_handler(self):
        """Keyboard shortcut handler should be registered."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "keydown" in html or "onkeydown" in html, \
            "Keyboard shortcut handler not found"

    # --- Responsive panel ---

    def test_responsive_media_query(self):
        """Panel should have CSS media query for small screens."""
        from humeris.adapters.cesium_viewer import generate_interactive_html
        html = generate_interactive_html()
        assert "@media" in html, \
            "No CSS media query found for responsive panel"


class TestInteractiveViewerBugFixes:
    """Bug fixes for interactive viewer JavaScript."""

    def _html(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        return generate_interactive_html()

    # BUG-006: Session load must clear old data sources
    def test_load_session_clears_old_sources(self):
        """loadSession should clear layerSources before loading new layers."""
        html = self._html()
        # Either clearAllSources is a separate function called by loadSession,
        # or loadSession directly iterates and removes
        assert "clearAllSources" in html or \
            ("loadSession" in html and "dataSources.remove" in html), \
            "loadSession must clear old data sources before loading"
        # Verify clearAllSources function removes tracked sources
        if "clearAllSources" in html:
            import re
            match = re.search(r'function clearAllSources\(\)\s*\{(.*?)\n\s*\}', html, re.DOTALL)
            assert match is not None
            fn_body = match.group(1)
            assert "dataSources.remove" in fn_body

    # BUG-008: Toast spam prevention
    def test_cap_toast_deduplication(self):
        """Cap toast should only show once per layer, not on every rebuild."""
        html = self._html()
        # Must track shown cap toasts in a Set or similar
        assert "cappedToastShown" in html or "shownCapToasts" in html or \
            "capToastIds" in html, \
            "Missing deduplication tracking for cap toasts"

    # BUG-009: Opacity slider remembers value
    def test_opacity_slider_preserves_value(self):
        """Opacity slider should use stored value, not always 100."""
        html = self._html()
        assert "layerOpacities" in html or "opacityState" in html, \
            "Missing opacity state tracking across rebuilds"

    # BUG-013: AbortController signal wired to fetch
    def test_abort_controller_wired_to_fetch(self):
        """fetch() calls should pass AbortController signal."""
        html = self._html()
        # The signal can be wired via a shared helper function (_fetchOpts)
        # or directly in each API function. Check that signal is present
        # in the fetch options infrastructure.
        assert "signal" in html and "currentAbortController" in html, \
            "AbortController signal must be wired to fetch calls"
        # Verify _fetchOpts or direct signal passing exists
        assert "_fetchOpts" in html or "signal: currentAbortController" in html, \
            "Signal must be passed to fetch via _fetchOpts helper or directly"

    # BUG-013: Non-JSON error handling
    def test_fetch_handles_non_json_errors(self):
        """API helpers should handle non-JSON error responses."""
        html = self._html()
        # Should have a catch for json parse failures
        assert "statusText" in html, \
            "API helpers should fall back to statusText for non-JSON errors"

    # BUG-019: Loading counter
    def test_loading_uses_counter(self):
        """Loading indicator should use counter, not boolean."""
        html = self._html()
        assert "loadingCount" in html or "loadCount" in html, \
            "Loading indicator should use a counter, not boolean show/hide"

    # BUG-020: toggleVisible has catch handler
    def test_toggle_visible_has_catch(self):
        """toggleVisible should have .catch() error handler."""
        html = self._html()
        import re
        # Capture entire toggleVisible function including chained methods
        match = re.search(
            r'function toggleVisible\(.*?\)\s*\{(.*?\.catch\(.*?\))',
            html, re.DOTALL,
        )
        assert match is not None, \
            "toggleVisible must have .catch() handler"

    # BUG-021: Toast timeout cleared on dismiss
    def test_toast_timeout_cleared_on_dismiss(self):
        """Toast manual dismiss should clearTimeout."""
        html = self._html()
        assert "clearTimeout" in html, \
            "showToast must clearTimeout on manual dismiss"

    # BUG-024: Blob URL revoked after save
    def test_save_session_revokes_blob_url(self):
        """saveSession should revoke blob URL after download."""
        html = self._html()
        assert "revokeObjectURL" in html, \
            "saveSession must revoke blob URL to prevent leak"

    # BUG-029: Show All/Hide All uses Promise.all
    def test_show_all_uses_promise_all(self):
        """Show All/Hide All should use Promise.all, not setTimeout."""
        html = self._html()
        import re
        # Find the Show All handler in rebuildPanel
        match = re.search(r'showBtn\.onclick.*?function\(\)\s*\{(.*?)\};', html, re.DOTALL)
        if match:
            fn_body = match.group(1)
            assert "Promise.all" in fn_body or "promises" in fn_body, \
                "Show All should use Promise.all instead of setTimeout"

    # BUG-030: dataSources.remove with destroy=true
    def test_remove_data_source_with_destroy(self):
        """dataSources.remove should pass true for destroy parameter."""
        html = self._html()
        import re
        # Find loadLayerCzml function
        match = re.search(r'function loadLayerCzml\(.*?\)\s*\{(.*?)\n\s*\}', html, re.DOTALL)
        assert match is not None
        fn_body = match.group(1)
        # Check that remove calls include true for destroy
        assert "remove(layerSources[layerId], true)" in fn_body or \
            "remove(layerSources[layerId],true)" in fn_body, \
            "dataSources.remove should pass true for destroy"


class TestViewerTableUX:
    """VIEWER-TABLE-UX: table button position and satellite detail selection."""

    def _html(self):
        from humeris.adapters.cesium_viewer import generate_interactive_html
        return generate_interactive_html()

    # UXC1: Table toggle button must clear the Cesium timeline
    def test_table_toggle_button_clears_timeline(self):
        """#satTableToggle bottom >= 36px to avoid overlapping the timeline."""
        html = self._html()
        match = re.search(r'#satTableToggle\s*\{([^}]+)\}', html)
        assert match is not None, "satTableToggle CSS rule not found"
        css = match.group(1)
        bottom_match = re.search(r'bottom:\s*(\d+)px', css)
        assert bottom_match is not None, "bottom property not found in satTableToggle"
        bottom_px = int(bottom_match.group(1))
        assert bottom_px >= 36, (
            f"satTableToggle bottom is {bottom_px}px, must be >= 36px "
            "to clear the Cesium timeline widget"
        )

    # UXC2: flyToSat must set viewer.selectedEntity for InfoBox display
    def test_fly_to_sat_sets_selected_entity(self):
        """flyToSat must set viewer.selectedEntity so Cesium InfoBox shows details."""
        html = self._html()
        # Find the flyToSat function body
        match = re.search(
            r'function flyToSat\([^)]*\)\s*\{(.*?)\n\s{8}\}',
            html, re.DOTALL,
        )
        assert match is not None, "flyToSat function not found"
        body = match.group(1)
        assert "viewer.selectedEntity" in body, (
            "flyToSat must set viewer.selectedEntity to activate "
            "Cesium's InfoBox with satellite details"
        )

    # UXC3: flyToSat must use CZML entity ID formats (satellite-N and snapshot-N)
    def test_fly_to_sat_uses_czml_entity_ids(self):
        """flyToSat must try both satellite-{idx} and snapshot-{idx} CZML ID formats."""
        html = self._html()
        match = re.search(
            r'function flyToSat\([^)]*\)\s*\{(.*?)\n\s{8}\}',
            html, re.DOTALL,
        )
        assert match is not None, "flyToSat function not found"
        body = match.group(1)
        assert '"satellite-"' in body, (
            "flyToSat must try 'satellite-' prefix for animated mode CZML entities"
        )
        assert '"snapshot-"' in body, (
            "flyToSat must try 'snapshot-' prefix for snapshot mode CZML entities"
        )

    # UXC4: Table rows pass original sat index, not sorted position
    def test_table_rows_use_sat_idx_not_position(self):
        """Table row onclick must use row._sat_idx, not the forEach index."""
        html = self._html()
        match = re.search(
            r'function renderSatTable\(\)\s*\{(.*?)\n\s{8}\}',
            html, re.DOTALL,
        )
        assert match is not None, "renderSatTable function not found"
        body = match.group(1)
        assert "_sat_idx" in body, (
            "renderSatTable must use row._sat_idx for flyToSat, "
            "not the forEach index which changes with sorting"
        )

    # UXC5: Table toggle button hides when table is visible
    def test_table_toggle_hides_when_table_open(self):
        """toggleSatTable must hide the button when table becomes visible."""
        html = self._html()
        match = re.search(
            r'function toggleSatTable\(\)\s*\{(.*?)\n\s{8}\}',
            html, re.DOTALL,
        )
        assert match is not None, "toggleSatTable function not found"
        body = match.group(1)
        assert "satTableToggle" in body, (
            "toggleSatTable must reference the toggle button to hide/show it"
        )
        assert 'display' in body or 'none' in body or 'style' in body, (
            "toggleSatTable must change button display when table state changes"
        )

    # UXC6: Table has a close button
    def test_table_has_close_button(self):
        """The satellite table must include a close button in its rendered HTML."""
        html = self._html()
        match = re.search(
            r'function renderSatTable\(\)\s*\{(.*?)\n\s{8}\}',
            html, re.DOTALL,
        )
        assert match is not None, "renderSatTable function not found"
        body = match.group(1)
        assert "toggleSatTable" in body or "closeSatTable" in body, (
            "renderSatTable must include a close button that hides the table"
        )

    # UXC7: Generated JS must have no syntax errors
    def test_generated_js_syntax_valid(self):
        """Generated JavaScript must pass syntax validation."""
        import subprocess
        import tempfile
        html = self._html()
        match = re.search(r'<script>(.*?)</script>', html, re.DOTALL)
        assert match is not None, "No script block found"
        js = match.group(1)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", encoding="utf-8", delete=False,
        ) as f:
            f.write(js)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["node", "--check", tmp_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
        finally:
            import os
            os.unlink(tmp_path)
        assert result.returncode == 0, (
            f"JavaScript syntax error in generated viewer HTML:\n"
            f"{result.stderr.strip()}"
        )


class TestCesiumViewerPurity:
    """Adapter purity: only stdlib + internal imports allowed."""

    def test_no_external_deps(self):
        import humeris.adapters.cesium_viewer as mod

        with open(mod.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        allowed_stdlib = {"json", "math", "numpy", "datetime", "string", "html"}
        allowed_internal = {"humeris"}

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
