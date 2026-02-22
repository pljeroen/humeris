# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in COMMERCIAL-LICENSE.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see COMMERCIAL-LICENSE.md.
"""Self-contained Cesium HTML viewer for CZML data.

Generates a single HTML file that loads CesiumJS from CDN and embeds
CZML data inline. Open in any browser to see animated 3D satellite orbits.

Supports multiple CZML layers (constellation + ground track + coverage)
in a single viewer.

Uses only stdlib json + no external dependencies.
"""

import html
import json


_CESIUM_VERSION = "1.138"


def generate_cesium_html(
    czml_packets: list[dict],
    title: str = "Constellation Viewer",
    cesium_token: str = "",
    additional_layers: list[list[dict]] | None = None,
) -> str:
    """Generate a self-contained HTML string with embedded CesiumJS viewer.

    Args:
        czml_packets: Primary CZML packet list (constellation, ground track, etc.).
        title: HTML page title.
        cesium_token: Cesium Ion access token. Optional — viewer works without
            it but shows a watermark and uses default imagery.
        additional_layers: Extra CZML packet lists to overlay (e.g. ground
            track, coverage). Each is loaded as a separate Cesium data source.

    Returns:
        Complete HTML document as a string.
    """
    safe_title = html.escape(title)

    czml_json = json.dumps(czml_packets, indent=2, ensure_ascii=False)
    # Prevent </script> in JSON from breaking the HTML script block
    czml_json = czml_json.replace("</", "<\\/")

    layers_js = ""
    if additional_layers:
        layer_jsons = [
            json.dumps(layer, indent=2, ensure_ascii=False).replace("</", "<\\/")
            for layer in additional_layers
        ]
        layers_array = ",\n            ".join(layer_jsons)
        layers_js = f"""
        var czmlLayers = [
            {layers_array}
        ];
        czmlLayers.forEach(function(layerData) {{
            var ds = new Cesium.CzmlDataSource();
            ds.load(layerData);
            viewer.dataSources.add(ds);
        }});"""

    token_line = ""
    has_token = bool(cesium_token)
    if cesium_token:
        token_line = f'Cesium.Ion.defaultAccessToken = {json.dumps(cesium_token)};'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <script src="https://cesium.com/downloads/cesiumjs/releases/{_CESIUM_VERSION}/Build/Cesium/Cesium.js"></script>
    <link href="https://cesium.com/downloads/cesiumjs/releases/{_CESIUM_VERSION}/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
    <style>
        html, body, #cesiumContainer {{
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: #000;
        }}
        #infoOverlay {{
            position: absolute;
            top: 8px;
            left: 8px;
            padding: 6px 12px;
            background: rgba(0, 0, 0, 0.6);
            color: #fff;
            font: 13px/1.4 sans-serif;
            border-radius: 4px;
            pointer-events: none;
            z-index: 100;
        }}
        #layerSelector {{
            position: absolute;
            top: 48px;
            left: 8px;
            padding: 8px 12px;
            background: rgba(0, 0, 0, 0.7);
            color: #fff;
            font: 12px/1.6 sans-serif;
            border-radius: 4px;
            z-index: 100;
            max-height: 400px;
            overflow-y: auto;
        }}
        #layerSelector label {{
            display: block;
            cursor: pointer;
            padding: 2px 0;
        }}
        #layerSelector input {{
            margin-right: 6px;
        }}
        #layerSelector details {{
            margin: 4px 0;
        }}
        #layerSelector summary {{
            cursor: pointer;
            font-weight: bold;
            padding: 2px 0;
        }}
        #layerSelector .cat-buttons {{
            margin: 2px 0 4px 0;
        }}
        #layerSelector .cat-buttons button {{
            font-size: 10px;
            margin-right: 4px;
            cursor: pointer;
            background: rgba(255,255,255,0.15);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 3px;
            padding: 1px 6px;
        }}
        .entity-count {{
            color: rgba(255,255,255,0.5);
            font-size: 11px;
        }}
    </style>
</head>
<body>
    <div id="cesiumContainer"></div>
    <div id="infoOverlay">{safe_title}</div>
    <div id="layerSelector"></div>
    <script>
        {token_line}
        var czmlData = {czml_json};
        var viewerOpts = {{
            shouldAnimate: true,
            timeline: true,
            animation: true,
            fullscreenButton: true,
            baseLayerPicker: true,
            sceneModePicker: true,
            navigationHelpButton: true,
            scene3DOnly: false,
            requestRenderMode: true,
            maximumRenderTimeChangePerSecond: Infinity,
        }};
        var hasUserToken = {"true" if has_token else "false"};
        if (!hasUserToken) {{
            var osm = new Cesium.ProviderViewModel({{
                name: "OpenStreetMap",
                iconUrl: Cesium.buildModuleUrl("Widgets/Images/ImageryProviders/openStreetMap.png"),
                tooltip: "OpenStreetMap — free, no token required",
                creationFunction: function() {{
                    return new Cesium.OpenStreetMapImageryProvider({{
                        url: "https://tile.openstreetmap.org/",
                    }});
                }}
            }});
            var naturalEarth = new Cesium.ProviderViewModel({{
                name: "Natural Earth",
                iconUrl: Cesium.buildModuleUrl("Widgets/Images/ImageryProviders/naturalEarthII.png"),
                tooltip: "Natural Earth II — bundled offline fallback",
                creationFunction: function() {{
                    return Cesium.TileMapServiceImageryProvider.fromUrl(
                        Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII")
                    );
                }}
            }});
            viewerOpts.imageryProviderViewModels = [osm, naturalEarth];
            viewerOpts.selectedImageryProviderViewModel = osm;
            viewerOpts.terrainProviderViewModels = [];
        }}
        var viewer = new Cesium.Viewer("cesiumContainer", viewerOpts);
        viewer.scene.globe.enableLighting = true;
        var dataSource = new Cesium.CzmlDataSource();
        dataSource.load(czmlData);
        viewer.dataSources.add(dataSource);
        viewer.zoomTo(dataSource);{layers_js}
        // Categorized layer selector UI
        (function() {{
            var panel = document.getElementById("layerSelector");
            if (!panel) return;
            var ds = viewer.dataSources;
            function buildPanel() {{
                panel.innerHTML = "";
                var categories = {{}};
                for (var i = 0; i < ds.length; i++) {{
                    var source = ds.get(i);
                    var fullName = source.name || ("Layer " + (i + 1));
                    var parts = fullName.split(":");
                    var category = parts.length > 1 ? parts[0] : "Layers";
                    var displayName = parts.length > 1 ? parts.slice(1).join(":") : fullName;
                    var entityCount = source.entities ? source.entities.values.length : 0;
                    if (!categories[category]) categories[category] = [];
                    categories[category].push({{idx: i, name: displayName, entities: entityCount, source: source}});
                }}
                var catKeys = Object.keys(categories);
                for (var c = 0; c < catKeys.length; c++) {{
                    var cat = catKeys[c];
                    var items = categories[cat];
                    var details = document.createElement("details");
                    details.open = true;
                    var summary = document.createElement("summary");
                    summary.textContent = cat;
                    details.appendChild(summary);
                    var btnDiv = document.createElement("div");
                    btnDiv.className = "cat-buttons";
                    var showBtn = document.createElement("button");
                    showBtn.textContent = "Show All";
                    showBtn.dataset.cat = cat;
                    showBtn.addEventListener("click", function() {{
                        var catName = this.dataset.cat;
                        var cbs = this.parentNode.parentNode.querySelectorAll("input[type=checkbox]");
                        for (var j = 0; j < cbs.length; j++) {{
                            cbs[j].checked = true;
                            var idx = parseInt(cbs[j].dataset.idx);
                            ds.get(idx).show = true;
                        }}
                        viewer.scene.requestRender();
                    }});
                    var hideBtn = document.createElement("button");
                    hideBtn.textContent = "Hide All";
                    hideBtn.dataset.cat = cat;
                    hideBtn.addEventListener("click", function() {{
                        var cbs = this.parentNode.parentNode.querySelectorAll("input[type=checkbox]");
                        for (var j = 0; j < cbs.length; j++) {{
                            cbs[j].checked = false;
                            var idx = parseInt(cbs[j].dataset.idx);
                            ds.get(idx).show = false;
                        }}
                        viewer.scene.requestRender();
                    }});
                    btnDiv.appendChild(showBtn);
                    btnDiv.appendChild(hideBtn);
                    details.appendChild(btnDiv);
                    for (var k = 0; k < items.length; k++) {{
                        var item = items[k];
                        var label = document.createElement("label");
                        var cb = document.createElement("input");
                        cb.type = "checkbox";
                        cb.checked = item.source.show;
                        cb.dataset.idx = item.idx;
                        cb.addEventListener("change", function() {{
                            var idx = parseInt(this.dataset.idx);
                            ds.get(idx).show = this.checked;
                            viewer.scene.requestRender();
                        }});
                        label.appendChild(cb);
                        var countText = item.entities > 0 ? " (" + item.entities + " entities)" : "";
                        label.appendChild(document.createTextNode(item.name));
                        var countSpan = document.createElement("span");
                        countSpan.className = "entity-count";
                        countSpan.textContent = countText;
                        label.appendChild(countSpan);
                        details.appendChild(label);
                    }}
                    panel.appendChild(details);
                }}
                if (ds.length === 0) {{
                    panel.style.display = "none";
                }}
            }}
            // Build after data sources are loaded
            setTimeout(buildPanel, 1000);
            ds.dataSourceAdded.addEventListener(function() {{ setTimeout(buildPanel, 500); }});
        }})();
    </script>
</body>
</html>"""


def generate_interactive_html(
    title: str = "Constellation Viewer",
    cesium_token: str = "",
    port: int = 8765,
) -> str:
    """Generate interactive HTML viewer that communicates with a local server.

    Unlike generate_cesium_html() which embeds CZML inline, this generates
    a viewer that fetches CZML from a local HTTP server API, supports
    adding/removing constellations, and toggling analysis layers.

    Args:
        title: HTML page title.
        cesium_token: Cesium Ion access token (optional).
        port: Server port for API calls.

    Returns:
        Complete HTML document as a string.
    """
    safe_title = html.escape(title)

    token_line = ""
    has_token = bool(cesium_token)
    if cesium_token:
        token_line = f'Cesium.Ion.defaultAccessToken = {json.dumps(cesium_token)};'

    api_base = f"http://localhost:{port}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <script src="https://cesium.com/downloads/cesiumjs/releases/{_CESIUM_VERSION}/Build/Cesium/Cesium.js"></script>
    <link href="https://cesium.com/downloads/cesiumjs/releases/{_CESIUM_VERSION}/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
    <style>
        html, body, #cesiumContainer {{
            width: 100%; height: 100%; margin: 0; padding: 0;
            overflow: hidden; background: #000;
        }}
        #cesiumContainer {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; }}
        #infoOverlay {{
            position: absolute; top: 8px; left: 8px;
            padding: 6px 12px; background: rgba(0,0,0,0.6);
            color: #fff; font: 13px/1.4 sans-serif;
            border-radius: 4px; pointer-events: none; z-index: 100;
        }}
        #sidePanel {{
            position: absolute; top: 48px; left: 8px; width: 280px;
            background: rgba(0,0,0,0.8); color: #fff;
            font: 12px/1.5 sans-serif; border-radius: 6px;
            z-index: 100; max-height: calc(100vh - 120px);
            overflow-y: auto; padding: 0;
        }}
        #sidePanel details {{ margin: 0; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        #sidePanel summary {{
            cursor: pointer; font-weight: bold; padding: 8px 12px;
            background: rgba(255,255,255,0.05);
        }}
        #sidePanel summary:hover {{ background: rgba(255,255,255,0.1); }}
        .panel-section {{ padding: 8px 12px; }}
        .form-row {{ margin: 4px 0; display: flex; align-items: center; gap: 6px; }}
        .form-row label {{ min-width: 80px; font-size: 11px; color: rgba(255,255,255,0.7); }}
        .form-row input, .form-row select {{
            flex: 1; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
            color: #fff; padding: 3px 6px; border-radius: 3px; font-size: 11px;
        }}
        .form-row select option {{ background: #222; color: #fff; }}
        .btn {{
            display: inline-block; padding: 4px 10px; margin: 2px;
            background: rgba(80,160,255,0.3); color: #fff;
            border: 1px solid rgba(80,160,255,0.5); border-radius: 3px;
            cursor: pointer; font-size: 11px;
        }}
        .btn:hover {{ background: rgba(80,160,255,0.5); }}
        .btn-sm {{ padding: 2px 6px; font-size: 10px; }}
        .btn-danger {{ background: rgba(255,80,80,0.3); border-color: rgba(255,80,80,0.5); }}
        .btn-danger:hover {{ background: rgba(255,80,80,0.5); }}
        .layer-item {{
            display: flex; align-items: center; gap: 4px;
            padding: 3px 12px; font-size: 11px;
        }}
        .layer-item:hover {{ background: rgba(255,255,255,0.05); }}
        .layer-item input[type=checkbox] {{ margin: 0; }}
        .layer-name {{ flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .entity-count {{ color: rgba(255,255,255,0.4); font-size: 10px; }}
        .mode-toggle {{
            font-size: 9px; padding: 1px 4px; cursor: pointer;
            background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
            color: rgba(255,255,255,0.7); border-radius: 2px;
        }}
        .mode-toggle:hover {{ background: rgba(255,255,255,0.2); }}
        .cat-header {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 4px 12px;
        }}
        .cat-buttons {{ display: flex; gap: 4px; }}
        .preset-row {{ display: flex; flex-wrap: wrap; gap: 2px; margin: 4px 0; }}
        .preset-btn {{
            font-size: 10px; padding: 2px 6px; cursor: pointer;
            background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
            color: rgba(255,255,255,0.7); border-radius: 2px;
        }}
        .preset-btn:hover {{ background: rgba(255,255,255,0.15); }}
        .analysis-grid {{ display: flex; flex-wrap: wrap; gap: 3px; padding: 4px 0; }}
        .analysis-cat {{
            width: 100%; font-size: 9px; color: rgba(255,255,255,0.4);
            text-transform: uppercase; letter-spacing: 0.5px;
            margin-top: 6px; padding-bottom: 2px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        .analysis-cat:first-child {{ margin-top: 0; }}
        #loadingIndicator {{
            display: none; position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%); background: rgba(0,0,0,0.8);
            color: #fff; padding: 16px 24px; border-radius: 8px;
            font: 14px sans-serif; z-index: 1000;
            align-items: center; gap: 10px;
        }}
        #loadingIndicator .spinner {{
            display: inline-block; width: 18px; height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #fff; border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
        #toastContainer {{
            position: fixed; bottom: 16px; right: 16px;
            z-index: 2000; display: flex; flex-direction: column-reverse; gap: 8px;
        }}
        .toast {{
            padding: 10px 16px; border-radius: 6px; color: #fff;
            font: 13px/1.4 sans-serif; max-width: 360px;
            opacity: 0; transform: translateX(40px);
            animation: toastIn 0.3s ease forwards;
            cursor: pointer;
        }}
        .toast-info {{ background: rgba(40,100,200,0.9); }}
        .toast-error {{ background: rgba(200,50,50,0.9); }}
        .toast-success {{ background: rgba(40,160,80,0.9); }}
        @keyframes toastIn {{ to {{ opacity: 1; transform: translateX(0); }} }}
        #panelToggle {{
            position: absolute; top: 48px; left: 8px;
            width: 28px; height: 28px; z-index: 101;
            background: rgba(0,0,0,0.7); color: #fff;
            border: 1px solid rgba(255,255,255,0.2); border-radius: 4px;
            cursor: pointer; font-size: 16px; line-height: 28px;
            text-align: center; display: none;
        }}
        #panelToggle:hover {{ background: rgba(0,0,0,0.9); }}
        #colorLegend {{
            position: absolute; bottom: 48px; right: 8px;
            background: rgba(0,0,0,0.8); color: #fff;
            font: 11px/1.5 sans-serif; border-radius: 6px;
            padding: 8px 12px; z-index: 100;
            display: none; min-width: 120px;
        }}
        #colorLegend .legend-title {{
            font-weight: bold; margin-bottom: 4px; font-size: 12px;
        }}
        .legend-entry {{
            display: flex; align-items: center; gap: 6px; padding: 1px 0;
        }}
        .legend-swatch {{
            width: 14px; height: 14px; border-radius: 2px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .layer-stats {{
            font-size: 10px; color: rgba(255,255,255,0.5);
            padding: 0 12px 2px; font-style: italic;
        }}
        .session-buttons {{ display: flex; gap: 4px; padding: 4px 12px; }}
        .opacity-slider {{
            width: 50px; height: 4px; cursor: pointer;
            accent-color: rgba(80,160,255,0.7);
        }}
        .rename-input {{
            background: rgba(255,255,255,0.15); border: 1px solid rgba(80,160,255,0.5);
            color: #fff; font-size: 11px; padding: 1px 4px; border-radius: 2px;
            width: 100%; outline: none;
        }}
        .edit-params {{
            padding: 2px 12px 6px; font-size: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .edit-params .ep-row {{
            display: flex; align-items: center; gap: 4px; margin: 2px 0;
        }}
        .edit-params .ep-row label {{
            min-width: 60px; color: rgba(255,255,255,0.6); font-size: 10px;
        }}
        .edit-params .ep-row input {{
            flex: 1; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15);
            color: #fff; padding: 2px 4px; border-radius: 2px; font-size: 10px; max-width: 70px;
        }}
        .edit-params .ep-apply {{
            margin-top: 3px; font-size: 9px; padding: 2px 8px;
            background: rgba(80,200,80,0.3); border: 1px solid rgba(80,200,80,0.5);
            color: #fff; border-radius: 2px; cursor: pointer;
        }}
        .edit-params .ep-apply:hover {{ background: rgba(80,200,80,0.5); }}
        .layer-metrics {{
            padding: 2px 12px 4px; font-size: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .layer-metrics .metric-row {{
            display: flex; justify-content: space-between; padding: 1px 0;
            color: rgba(255,255,255,0.7);
        }}
        .layer-metrics .metric-row .metric-key {{
            color: rgba(255,255,255,0.5);
        }}
        .layer-metrics .metric-row .metric-val {{
            font-family: monospace; color: rgba(80,200,255,0.9);
        }}
        #sweepChart {{
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 600px; height: 400px;
            background: rgba(20,25,35,0.97); color: #fff;
            border: 1px solid rgba(80,160,255,0.3); border-radius: 6px;
            display: none; z-index: 200; padding: 12px;
        }}
        #sweepChart.visible {{ display: block; }}
        #sweepChart canvas {{ width: 100%; height: 320px; }}
        #sweepChart .chart-close {{
            position: absolute; top: 6px; right: 10px;
            background: none; border: none; color: rgba(255,100,100,0.8);
            font-size: 16px; cursor: pointer;
        }}
        #sweepChart .chart-title {{
            font-size: 13px; color: rgba(80,160,255,0.9);
            margin-bottom: 6px;
        }}
        #sweepChart .chart-export {{
            margin-top: 6px; font-size: 10px; padding: 2px 8px;
            background: rgba(80,160,255,0.2); border: 1px solid rgba(80,160,255,0.3);
            color: #fff; border-radius: 2px; cursor: pointer;
        }}
        .recent-item {{
            padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .recent-item .recent-name {{
            display: block; color: rgba(80,160,255,0.9); font-size: 11px;
        }}
        .recent-item .recent-meta {{
            display: block; color: rgba(255,255,255,0.4); font-size: 9px;
        }}
        #satTable {{
            position: fixed; bottom: 0; left: 280px; right: 0;
            max-height: 35vh; overflow: auto;
            background: rgba(20,25,35,0.95); color: #fff;
            font-size: 11px; display: none; z-index: 100;
            border-top: 1px solid rgba(80,160,255,0.3);
        }}
        #satTable.visible {{ display: block; }}
        #satTable table {{ width: 100%; border-collapse: collapse; }}
        #satTable th {{
            position: sticky; top: 0; background: rgba(30,35,50,0.98);
            padding: 4px 8px; cursor: pointer; text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.15);
            user-select: none; white-space: nowrap;
        }}
        #satTable th:hover {{ background: rgba(80,160,255,0.2); }}
        #satTable td {{ padding: 3px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        #satTable tr:hover {{ background: rgba(80,160,255,0.15); cursor: pointer; }}
        #satTable .sort-asc::after {{ content: " \\25B2"; font-size: 8px; }}
        #satTable .sort-desc::after {{ content: " \\25BC"; font-size: 8px; }}
        #satTableToggle {{
            position: fixed; bottom: 36px; right: 8px; z-index: 101;
            background: rgba(30,40,60,0.9); color: #fff;
            border: 1px solid rgba(80,160,255,0.3); border-radius: 4px;
            padding: 4px 10px; cursor: pointer; font-size: 11px;
        }}
        #satTableToggle:hover {{ background: rgba(80,160,255,0.3); }}
        @media (max-width: 1024px) {{
            #sidePanel {{ width: 220px; font-size: 11px; }}
            #sidePanel.collapsed {{ display: none; }}
        }}
        @media (max-width: 768px) {{
            #sidePanel {{ display: none; }}
            #panelToggle {{ display: block; }}
        }}
    </style>
</head>
<body>
    <div id="cesiumContainer"></div>
    <div id="infoOverlay">{safe_title}</div>
    <div id="loadingIndicator"><span class="spinner"></span><span id="loadingText">Loading...</span><button id="cancelBtn" class="btn btn-sm btn-danger" style="margin-left:12px" onclick="cancelRequest()">Cancel</button></div>
    <div id="toastContainer"></div>
    <button id="panelToggle" onclick="togglePanel()" title="Toggle panel">&#9776;</button>
    <button id="satTableToggle" onclick="toggleSatTable()">Table</button>
    <div id="satTable"></div>
    <div id="sweepChart"><div class="chart-title" id="sweepTitle"></div><canvas id="sweepCanvas"></canvas><button class="chart-close" onclick="closeSweepChart()">&times;</button><button class="chart-export" onclick="exportSweepCSV()">Export CSV</button></div>
    <div id="colorLegend"></div>
    <div id="sidePanel">
        <!-- Add Layer Section -->
        <details open>
            <summary>Add Layer</summary>
            <div class="panel-section">
                <!-- Walker Form -->
                <details>
                    <summary>Walker Constellation</summary>
                    <div class="panel-section">
                        <div class="form-row"><label>Altitude (km)</label><input type="number" id="w-alt" value="550" min="100" max="100000" step="10"></div>
                        <div class="form-row"><label>Inclination</label><input type="number" id="w-inc" value="53" min="0" max="180" step="1"></div>
                        <div class="form-row"><label>Planes</label><input type="number" id="w-planes" value="6" min="1" max="100"></div>
                        <div class="form-row"><label>Sats/plane</label><input type="number" id="w-spp" value="10" min="1" max="100"></div>
                        <div class="form-row"><label>Phase factor</label><input type="number" id="w-pf" value="1" min="0"></div>
                        <div class="form-row"><label>RAAN offset</label><input type="number" id="w-raan" value="0" min="0" max="360" step="0.1"></div>
                        <div class="form-row"><label>Name</label><input type="text" id="w-name" value="Walker"></div>
                        <button class="btn" onclick="addWalker()">Add Walker</button>
                    </div>
                </details>
                <!-- CelesTrak Form -->
                <details>
                    <summary>CelesTrak</summary>
                    <div class="panel-section">
                        <div class="form-row">
                            <label>Group</label>
                            <select id="ct-group">
                                <option value="">-- select --</option>
                                <option value="STATIONS">ISS / Stations</option>
                                <option value="ACTIVE">Active Satellites</option>
                                <option value="STARLINK">Starlink</option>
                                <option value="GPS-OPS">GPS</option>
                                <option value="GLONASS">GLONASS</option>
                                <option value="GALILEO">Galileo</option>
                                <option value="BEIDOU">BeiDou</option>
                                <option value="IRIDIUM-NEXT">Iridium NEXT</option>
                                <option value="ONEWEB">OneWeb</option>
                                <option value="PLANET">Planet</option>
                                <option value="SPIRE">Spire</option>
                                <option value="WEATHER">Weather</option>
                                <option value="GEO">Geostationary</option>
                                <option value="INTELSAT">Intelsat</option>
                                <option value="SES">SES</option>
                                <option value="TELESAT">Telesat</option>
                                <option value="AMATEUR">Amateur Radio</option>
                                <option value="SCIENCE">Science</option>
                                <option value="NOAA">NOAA</option>
                                <option value="GOES">GOES</option>
                            </select>
                        </div>
                        <div class="form-row"><label>Or name</label><input type="text" id="ct-name" placeholder="e.g. ISS (ZARYA)"></div>
                        <button class="btn" onclick="addCelesTrak()">Fetch &amp; Add</button>
                    </div>
                </details>
                <!-- Ground Station Form -->
                <details>
                    <summary>Ground Station</summary>
                    <div class="panel-section">
                        <div class="form-row"><label>Name</label><input type="text" id="gs-name" value="Station"></div>
                        <div class="form-row"><label>Latitude</label><input type="number" id="gs-lat" value="0" min="-90" max="90" step="0.01"></div>
                        <div class="form-row"><label>Longitude</label><input type="number" id="gs-lon" value="0" min="-180" max="180" step="0.01"></div>
                        <div class="preset-row">
                            <button class="preset-btn" onclick="gsPreset('Svalbard',78.23,15.39)">Svalbard</button>
                            <button class="preset-btn" onclick="gsPreset('Kiruna',67.86,20.96)">Kiruna</button>
                            <button class="preset-btn" onclick="gsPreset('Maspalomas',27.76,-15.59)">Maspalomas</button>
                            <button class="preset-btn" onclick="gsPreset('Santiago',-33.15,-70.67)">Santiago</button>
                            <button class="preset-btn" onclick="gsPreset('Canberra',-35.40,148.98)">Canberra</button>
                        </div>
                        <button class="btn" onclick="addGroundStation()">Add Station</button>
                    </div>
                </details>
            </div>
        </details>
        <!-- Active Layers Section -->
        <details open>
            <summary>Active Layers</summary>
            <div id="layerList" class="panel-section">
                <div style="color:rgba(255,255,255,0.4);font-style:italic">No layers yet</div>
            </div>
        </details>
        <!-- Session Controls -->
        <div class="session-buttons">
            <button class="btn btn-sm" onclick="saveSession()">Save Session</button>
            <button class="btn btn-sm" onclick="loadSession()">Load Session</button>
            <button class="btn btn-sm" onclick="generateReport()">Report</button>
        </div>
        <!-- Recent Scenarios (APP-04) -->
        <details>
            <summary>Recent</summary>
            <div id="recentScenarios" class="panel-section"></div>
        </details>
        <!-- Simulation Settings Section -->
        <details>
            <summary>Simulation</summary>
            <div class="panel-section">
                <div class="form-row"><label>Duration (h)</label><input type="number" id="sim-duration" value="2" min="0.1" step="0.5"></div>
                <div class="form-row"><label>Step (s)</label><input type="number" id="sim-step" value="60" min="1" step="10"></div>
                <button class="btn btn-sm" onclick="applySettings()">Apply</button>
            </div>
        </details>
        <!-- Constraints (APP-07) -->
        <details>
            <summary>Constraints</summary>
            <div class="panel-section">
                <div class="form-row"><label>Metric</label><input type="text" id="cst-metric" placeholder="e.g. beta_angle_avg_beta_deg" style="font-size:9px"></div>
                <div class="form-row"><label>Op</label>
                    <select id="cst-op">
                        <option value=">=">>=</option>
                        <option value="<="><=</option>
                        <option value=">">&gt;</option>
                        <option value="<">&lt;</option>
                    </select>
                </div>
                <div class="form-row"><label>Threshold</label><input type="number" id="cst-threshold" step="0.1"></div>
                <button class="btn btn-sm" onclick="addConstraint()">Add</button>
                <button class="btn btn-sm" onclick="evaluateConstraints()">Evaluate</button>
                <div id="constraintResults" style="margin-top:4px;font-size:10px"></div>
            </div>
        </details>
        <!-- Compare (APP-06) -->
        <details>
            <summary>Compare</summary>
            <div class="panel-section">
                <div class="form-row"><label>Config A</label><select id="cmp-a"></select></div>
                <div class="form-row"><label>Config B</label><select id="cmp-b"></select></div>
                <button class="btn btn-sm" onclick="runCompare()">Compare</button>
                <div id="compareResult" style="margin-top:6px;font-size:10px"></div>
            </div>
        </details>
        <!-- Parameter Sweep (APP-05) -->
        <details>
            <summary>Sweep</summary>
            <div class="panel-section">
                <div class="form-row"><label>Parameter</label>
                    <select id="sw-param">
                        <option value="altitude_km">Altitude (km)</option>
                        <option value="inclination_deg">Inclination (deg)</option>
                        <option value="num_planes">Planes</option>
                        <option value="sats_per_plane">Sats/plane</option>
                    </select>
                </div>
                <div class="form-row"><label>Min</label><input type="number" id="sw-min" value="400" step="10"></div>
                <div class="form-row"><label>Max</label><input type="number" id="sw-max" value="800" step="10"></div>
                <div class="form-row"><label>Step</label><input type="number" id="sw-step" value="50" step="10"></div>
                <div class="form-row"><label>Metric</label>
                    <select id="sw-metric">
                        <option value="coverage">Coverage</option>
                        <option value="eclipse">Eclipse</option>
                        <option value="beta_angle">Beta Angle</option>
                        <option value="deorbit">Deorbit</option>
                        <option value="station_keeping">Station Keeping</option>
                    </select>
                </div>
                <button class="btn btn-sm" onclick="runSweep()">Run Sweep</button>
            </div>
        </details>
        <!-- Analysis Layers Section -->
        <details open>
            <summary>Add Analysis</summary>
            <div class="panel-section">
                <div style="color:rgba(255,255,255,0.5);font-size:11px;margin-bottom:6px">
                    Select a source constellation first, then add analysis layers.
                </div>
                <div class="form-row">
                    <label>Source</label>
                    <select id="analysis-source" onchange="onSourceChanged(this.value)"></select>
                </div>
                <!-- Analysis parameter fields -->
                <details>
                    <summary style="font-size:11px;font-weight:normal;color:rgba(255,255,255,0.6)">Parameters</summary>
                    <div class="panel-section" style="padding-top:2px">
                        <div class="form-row"><label>Lat step</label><input type="number" id="param-lat-step" value="10" min="1" step="1" title="Grid latitude step (deg)"></div>
                        <div class="form-row"><label>Lon step</label><input type="number" id="param-lon-step" value="10" min="1" step="1" title="Grid longitude step (deg)"></div>
                        <div class="form-row"><label>Min elev</label><input type="number" id="param-min-elev" value="10" min="0" step="1" title="Minimum elevation angle (deg)"></div>
                        <div class="form-row"><label>Max range</label><input type="number" id="param-max-range" value="5000" min="100" step="100" title="ISL max range (km)"></div>
                        <div class="form-row"><label>Cd</label><input type="number" id="param-cd" value="2.2" min="0.1" step="0.1" title="Drag coefficient"></div>
                        <div class="form-row"><label>Area (m²)</label><input type="number" id="param-area" value="0.01" min="0.001" step="0.001" title="Cross-section area"></div>
                        <div class="form-row"><label>Mass (kg)</label><input type="number" id="param-mass" value="4.0" min="0.1" step="0.1" title="Satellite mass"></div>
                    </div>
                </details>
                <div class="analysis-grid">
                    <div class="analysis-cat">Orbit &amp; Geometry</div>
                    <button class="btn btn-sm" title="Color satellites by shadow state (sunlit/penumbra/umbra)" onclick="addAnalysis('eclipse')">Eclipse</button>
                    <button class="btn btn-sm" title="Sub-satellite ground trace of first satellite" onclick="addAnalysis('ground_track')">Ground Track</button>
                    <button class="btn btn-sm" title="J2 RAAN drift over 7 days" onclick="addAnalysis('precession')">Precession</button>
                    <button class="btn btn-sm" title="Solar beta angle snapshot" onclick="addAnalysis('beta_angle')">Beta Angle</button>
                    <button class="btn btn-sm" title="Relative trajectory between two satellites" onclick="addAnalysis('relative_motion')">Relative Motion</button>

                    <div class="analysis-cat">Coverage &amp; Navigation</div>
                    <button class="btn btn-sm" title="Grid heatmap of visible satellite count" onclick="addAnalysis('coverage')">Coverage</button>
                    <button class="btn btn-sm" title="Ground-level field of view footprints" onclick="addAnalysis('sensor')">Sensor FOV</button>
                    <button class="btn btn-sm" title="Dilution of precision heatmap" onclick="addAnalysis('dop_grid')">DOP Grid</button>

                    <div class="analysis-cat">Network &amp; Topology</div>
                    <button class="btn btn-sm" title="Inter-satellite link topology colored by SNR" onclick="addAnalysis('isl')">ISL Topology</button>
                    <button class="btn btn-sm" title="Spectral fragility index per satellite" onclick="addAnalysis('fragility')">Fragility</button>
                    <button class="btn btn-sm" title="ISL links colored by endpoint eclipse state" onclick="addAnalysis('network_eclipse')">Net Eclipse</button>
                    <button class="btn btn-sm" title="Coverage weighted by network connectivity" onclick="addAnalysis('coverage_connectivity')">Cov+Connect</button>

                    <div class="analysis-cat">Environment &amp; Risk</div>
                    <button class="btn btn-sm" title="Radiation environment coloring" onclick="addAnalysis('radiation')">Radiation</button>
                    <button class="btn btn-sm" title="Projected lifetime (green=long, red=short)" onclick="addAnalysis('hazard')">Hazard</button>
                    <button class="btn btn-sm" title="Debris density heatmap by altitude and inclination" onclick="addAnalysis('kessler_heatmap')">Kessler</button>
                    <button class="btn btn-sm" title="Conjunction screening with NASA hazard levels" onclick="addAnalysis('conjunction_hazard')">Conj Hazard</button>
                    <button class="btn btn-sm" title="SIR epidemic debris cascade evolution" onclick="addAnalysis('cascade_sir')">Cascade SIR</button>
                    <button class="btn btn-sm" title="Close approach replay between two satellites" onclick="addAnalysis('conjunction')">Conjunction</button>

                    <div class="analysis-cat">Operations &amp; Maintenance</div>
                    <button class="btn btn-sm" title="Deorbit compliance status coloring" onclick="addAnalysis('deorbit')">Deorbit</button>
                    <button class="btn btn-sm" title="Station-keeping delta-V budget coloring" onclick="addAnalysis('station_keeping')">Station-Keep</button>
                    <button class="btn btn-sm" title="Maintenance schedule status coloring" onclick="addAnalysis('maintenance')">Maintenance</button>
                </div>
            </div>
        </details>
    </div>
    <script>
        {token_line}
        var API = "{api_base}";
        var hasUserToken = {"true" if has_token else "false"};
        var viewerOpts = {{
            shouldAnimate: true,
            timeline: true,
            animation: true,
            fullscreenButton: true,
            baseLayerPicker: true,
            sceneModePicker: true,
            navigationHelpButton: true,
            scene3DOnly: false,
        }};
        if (!hasUserToken) {{
            var osm = new Cesium.ProviderViewModel({{
                name: "OpenStreetMap",
                iconUrl: Cesium.buildModuleUrl("Widgets/Images/ImageryProviders/openStreetMap.png"),
                tooltip: "OpenStreetMap — free, no token required",
                creationFunction: function() {{
                    return new Cesium.OpenStreetMapImageryProvider({{
                        url: "https://tile.openstreetmap.org/",
                    }});
                }}
            }});
            var naturalEarth = new Cesium.ProviderViewModel({{
                name: "Natural Earth",
                iconUrl: Cesium.buildModuleUrl("Widgets/Images/ImageryProviders/naturalEarthII.png"),
                tooltip: "Natural Earth II — bundled offline fallback",
                creationFunction: function() {{
                    return Cesium.TileMapServiceImageryProvider.fromUrl(
                        Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII")
                    );
                }}
            }});
            viewerOpts.imageryProviderViewModels = [osm, naturalEarth];
            viewerOpts.selectedImageryProviderViewModel = osm;
            viewerOpts.terrainProviderViewModels = [];
        }}
        var viewer = new Cesium.Viewer("cesiumContainer", viewerOpts);
        viewer.scene.globe.enableLighting = true;

        // Layer tracking: layerId -> CzmlDataSource
        var layerSources = {{}};
        var layerOpacities = {{}};  // layerId -> opacity (0-100)
        var capToastIds = new Set();  // Track which layers showed cap toast

        var currentAbortController = null;
        var loadingCount = 0;
        function showLoading(msg) {{
            currentAbortController = new AbortController();
            loadingCount++;
            var el = document.getElementById("loadingIndicator");
            document.getElementById("loadingText").textContent = msg || "Loading...";
            el.style.display = "flex";
        }}
        function hideLoading() {{
            loadingCount = Math.max(0, loadingCount - 1);
            if (loadingCount === 0) {{
                currentAbortController = null;
                document.getElementById("loadingIndicator").style.display = "none";
            }}
        }}
        function cancelRequest() {{
            if (currentAbortController) {{
                currentAbortController.abort();
                currentAbortController = null;
            }}
            loadingCount = 0;
            document.getElementById("loadingIndicator").style.display = "none";
            showToast("Request cancelled", "info");
        }}
        function showToast(msg, type) {{
            type = type || "info";
            var container = document.getElementById("toastContainer");
            var toast = document.createElement("div");
            toast.className = "toast toast-" + type;
            toast.textContent = msg;
            var delay = type === "error" ? 8000 : type === "success" ? 3000 : 5000;
            var tid = setTimeout(function() {{ toast.remove(); }}, delay);
            toast.onclick = function() {{ clearTimeout(tid); toast.remove(); }};
            container.appendChild(toast);
        }}
        function togglePanel() {{
            var panel = document.getElementById("sidePanel");
            var btn = document.getElementById("panelToggle");
            if (panel.style.display === "none") {{
                panel.style.display = "";
                btn.style.display = "none";
            }} else {{
                panel.style.display = "none";
                btn.style.display = "block";
            }}
        }}

        // --- HTML escaping utility ---
        function escapeHtml(str) {{
            if (str === null || str === undefined) return "";
            return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }}

        // --- Satellite data table (APP-03) ---
        var satTableData = null;
        var satTableSort = {{col: null, asc: true}};
        var satTableLayerId = null;

        function toggleSatTable() {{
            var tbl = document.getElementById("satTable");
            var btn = document.getElementById("satTableToggle");
            tbl.classList.toggle("visible");
            var isVisible = tbl.classList.contains("visible");
            btn.style.display = isVisible ? "none" : "";
            if (isVisible && !satTableData) {{
                // Load table for current source selection
                var sourceEl = document.getElementById("analysis-source");
                if (sourceEl && sourceEl.value) {{
                    loadSatTable(sourceEl.value);
                }}
            }}
        }}

        function onSourceChanged(layerId) {{
            var tbl = document.getElementById("satTable");
            if (tbl.classList.contains("visible") && layerId) {{
                loadSatTable(layerId);
            }}
        }}

        function loadSatTable(layerId) {{
            satTableLayerId = layerId;
            apiGet("/api/table/" + layerId).then(function(data) {{
                satTableData = data;
                satTableSort = {{col: null, asc: true}};
                renderSatTable();
            }}).catch(function(e) {{
                document.getElementById("satTable").innerHTML = '<div style="padding:8px;color:rgba(255,100,100,0.8)">Error loading table</div>';
            }});
        }}

        function renderSatTable() {{
            if (!satTableData) return;
            var cols = satTableData.columns;
            var rows = satTableData.rows.slice();
            if (satTableSort.col !== null) {{
                var sc = satTableSort.col;
                var asc = satTableSort.asc;
                rows.sort(function(a, b) {{
                    var va = a[sc], vb = b[sc];
                    if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
                    return asc ? va - vb : vb - va;
                }});
            }}
            var html = '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 8px;background:rgba(30,35,50,0.98);border-bottom:1px solid rgba(255,255,255,0.15)">';
            html += '<span style="font-weight:bold;font-size:11px;color:rgba(255,255,255,0.7)">Satellite Table (' + rows.length + ')</span>';
            html += '<button onclick="toggleSatTable()" style="background:none;border:none;color:rgba(255,255,255,0.6);cursor:pointer;font-size:16px;padding:0 4px" title="Close table">&times;</button>';
            html += '</div>';
            html += '<table><thead><tr>';
            cols.forEach(function(c) {{
                var cls = "";
                if (satTableSort.col === c) cls = satTableSort.asc ? "sort-asc" : "sort-desc";
                html += '<th class="' + cls + '" onclick="sortSatTable(\\'' + escapeHtml(c) + '\\')">' + escapeHtml(c.replace(/_/g, " ")) + '</th>';
            }});
            html += '</tr></thead><tbody>';
            rows.forEach(function(row) {{
                html += '<tr onclick="flyToSat(' + row._sat_idx + ')">';
                cols.forEach(function(c) {{
                    html += '<td>' + escapeHtml(row[c]) + '</td>';
                }});
                html += '</tr>';
            }});
            html += '</tbody></table>';
            document.getElementById("satTable").innerHTML = html;
        }}

        function sortSatTable(col) {{
            if (satTableSort.col === col) {{
                satTableSort.asc = !satTableSort.asc;
            }} else {{
                satTableSort.col = col;
                satTableSort.asc = true;
            }}
            renderSatTable();
        }}

        function flyToSat(idx) {{
            if (!satTableLayerId) return;
            var ds = layerSources[satTableLayerId];
            if (!ds) return;
            // CZML uses satellite-N (animated) or snapshot-N (snapshot mode)
            var entity = ds.entities.getById("satellite-" + idx)
                      || ds.entities.getById("snapshot-" + idx);
            if (entity) {{
                viewer.selectedEntity = entity;
                viewer.flyTo(entity, {{duration: 1.5}});
            }}
        }}

        // --- Layer rename ---
        function renameLayer(layerId, newName) {{
            if (!newName || !newName.trim()) return;
            apiPut("/api/layer/" + layerId, {{name: newName.trim()}}).then(function() {{
                rebuildPanel();
            }}).catch(function(e) {{
                showToast("Rename error: " + e.message, "error");
            }});
        }}

        // --- Layer opacity ---
        function setOpacity(layerId, value) {{
            var alpha = value / 100;
            if (layerSources[layerId]) {{
                var entities = layerSources[layerId].entities.values;
                for (var i = 0; i < entities.length; i++) {{
                    var e = entities[i];
                    if (e.point) {{
                        var c = e.point.color.getValue(viewer.clock.currentTime);
                        if (c) e.point.color = new Cesium.Color(c.red, c.green, c.blue, alpha);
                    }}
                }}
                viewer.scene.requestRender();
            }}
        }}

        // --- Keyboard shortcuts ---
        document.addEventListener("keydown", function(e) {{
            // Alt+P: toggle panel
            if (e.altKey && e.key === "p") {{ togglePanel(); e.preventDefault(); }}
            // Space: play/pause (only when not in input)
            if (e.key === " " && e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") {{
                viewer.clock.shouldAnimate = !viewer.clock.shouldAnimate;
                e.preventDefault();
            }}
            // Delete: remove focused layer (if any)
            if (e.key === "Delete" && e.target.dataset && e.target.dataset.layerId) {{
                removeLayer(e.target.dataset.layerId);
                e.preventDefault();
            }}
        }});

        // --- API helpers ---
        function _handleResponse(r) {{
            if (!r.ok) {{
                return r.json().catch(function() {{
                    throw new Error(r.statusText || "Request failed");
                }}).then(function(e) {{ throw new Error(e.error || r.statusText); }});
            }}
            return r.json();
        }}
        function _fetchOpts() {{
            var opts = {{}};
            if (currentAbortController) opts.signal = currentAbortController.signal;
            return opts;
        }}
        function apiGet(path) {{
            var opts = _fetchOpts();
            return fetch(API + path, opts).then(_handleResponse);
        }}
        function apiPost(path, body) {{
            var opts = _fetchOpts();
            opts.method = "POST";
            opts.headers = {{"Content-Type": "application/json"}};
            opts.body = JSON.stringify(body);
            return fetch(API + path, opts).then(_handleResponse);
        }}
        function apiPut(path, body) {{
            var opts = _fetchOpts();
            opts.method = "PUT";
            opts.headers = {{"Content-Type": "application/json"}};
            opts.body = JSON.stringify(body);
            return fetch(API + path, opts).then(_handleResponse);
        }}
        function apiDelete(path) {{
            var opts = _fetchOpts();
            opts.method = "DELETE";
            return fetch(API + path, opts).then(_handleResponse);
        }}

        // --- Load CZML for a layer ---
        function loadLayerCzml(layerId) {{
            return apiGet("/api/czml/" + layerId).then(function(czml) {{
                // Remove old data source if exists (destroy=true to free memory)
                if (layerSources[layerId]) {{
                    viewer.dataSources.remove(layerSources[layerId], true);
                }}
                var ds = new Cesium.CzmlDataSource();
                return ds.load(czml).then(function() {{
                    viewer.dataSources.add(ds);
                    layerSources[layerId] = ds;
                    viewer.scene.requestRender();
                    return ds;
                }});
            }});
        }}

        // --- Add Walker ---
        function addWalker() {{
            showLoading("Generating Walker constellation...");
            apiPost("/api/constellation", {{
                type: "walker",
                params: {{
                    altitude_km: parseFloat(document.getElementById("w-alt").value),
                    inclination_deg: parseFloat(document.getElementById("w-inc").value),
                    num_planes: parseInt(document.getElementById("w-planes").value),
                    sats_per_plane: parseInt(document.getElementById("w-spp").value),
                    phase_factor: parseInt(document.getElementById("w-pf").value),
                    raan_offset_deg: parseFloat(document.getElementById("w-raan").value),
                    shell_name: document.getElementById("w-name").value,
                }}
            }}).then(function(resp) {{
                return loadLayerCzml(resp.layer_id);
            }}).then(function() {{
                rebuildPanel();
                hideLoading();
            }}).catch(function(e) {{
                hideLoading();
                showToast("Walker error: " + e.message, "error");
            }});
        }}

        // --- Add CelesTrak ---
        function addCelesTrak() {{
            var group = document.getElementById("ct-group").value;
            var name = document.getElementById("ct-name").value;
            if (!group && !name) {{ showToast("Select a group or enter a name", "error"); return; }}
            showLoading("Fetching from CelesTrak...");
            var params = {{}};
            if (group) params.group = group;
            if (name) params.name = name;
            apiPost("/api/constellation", {{
                type: "celestrak",
                params: params,
            }}).then(function(resp) {{
                return loadLayerCzml(resp.layer_id);
            }}).then(function() {{
                rebuildPanel();
                hideLoading();
            }}).catch(function(e) {{
                hideLoading();
                showToast("CelesTrak error: " + e.message, "error");
            }});
        }}

        // --- Add Ground Station ---
        function gsPreset(name, lat, lon) {{
            document.getElementById("gs-name").value = name;
            document.getElementById("gs-lat").value = lat;
            document.getElementById("gs-lon").value = lon;
        }}
        function addGroundStation() {{
            showLoading("Adding ground station...");
            apiPost("/api/ground-station", {{
                name: document.getElementById("gs-name").value,
                lat: parseFloat(document.getElementById("gs-lat").value),
                lon: parseFloat(document.getElementById("gs-lon").value),
            }}).then(function(resp) {{
                return loadLayerCzml(resp.layer_id);
            }}).then(function() {{
                rebuildPanel();
                hideLoading();
            }}).catch(function(e) {{
                hideLoading();
                showToast("Station error: " + e.message, "error");
            }});
        }}

        // --- Gather analysis params from form ---
        function gatherAnalysisParams() {{
            return {{
                lat_step_deg: parseFloat(document.getElementById("param-lat-step").value),
                lon_step_deg: parseFloat(document.getElementById("param-lon-step").value),
                min_elevation_deg: parseFloat(document.getElementById("param-min-elev").value),
                max_range_km: parseFloat(document.getElementById("param-max-range").value),
                cd: parseFloat(document.getElementById("param-cd").value),
                area_m2: parseFloat(document.getElementById("param-area").value),
                mass_kg: parseFloat(document.getElementById("param-mass").value),
            }};
        }}

        // --- Add Analysis ---
        function addAnalysis(type) {{
            var sourceEl = document.getElementById("analysis-source");
            if (!sourceEl.value) {{ showToast("Select a source constellation first", "error"); return; }}
            showLoading("Computing " + type + " analysis...");
            var params = gatherAnalysisParams();
            apiPost("/api/analysis", {{
                type: type,
                source_layer: sourceEl.value,
                params: params,
            }}).then(function(resp) {{
                return loadLayerCzml(resp.layer_id);
            }}).then(function() {{
                rebuildPanel();
                hideLoading();
            }}).catch(function(e) {{
                hideLoading();
                showToast("Analysis error: " + e.message, "error");
            }});
        }}

        // --- Remove Layer ---
        function removeLayer(layerId) {{
            apiDelete("/api/layer/" + layerId).then(function() {{
                if (layerSources[layerId]) {{
                    viewer.dataSources.remove(layerSources[layerId], true);
                    delete layerSources[layerId];
                    viewer.scene.requestRender();
                }}
                rebuildPanel();
            }}).catch(function(e) {{
                showToast("Remove error: " + e.message, "error");
            }});
        }}

        // --- Toggle Visibility ---
        function toggleVisible(layerId, visible) {{
            return apiPut("/api/layer/" + layerId, {{visible: visible}}).then(function() {{
                if (layerSources[layerId]) {{
                    layerSources[layerId].show = visible;
                    viewer.scene.requestRender();
                }}
            }}).catch(function(e) {{
                showToast("Visibility error: " + e.message, "error");
                rebuildPanel();
            }});
        }}

        // --- Toggle Mode (snapshot/animated) ---
        function toggleMode(layerId, newMode) {{
            showLoading("Switching to " + newMode + "...");
            apiPut("/api/layer/" + layerId, {{mode: newMode}}).then(function() {{
                return loadLayerCzml(layerId);
            }}).then(function() {{
                rebuildPanel();
                hideLoading();
            }}).catch(function(e) {{
                hideLoading();
                showToast("Mode switch error: " + e.message, "error");
            }});
        }}

        // --- Reconfigure walker constellation ---
        function reconfigureConstellation(layerId) {{
            var prefix = "ep-" + layerId + "-";
            var params = {{}};
            var fields = ["altitude_km", "inclination_deg", "num_planes", "sats_per_plane", "phase_factor", "raan_offset_deg"];
            fields.forEach(function(f) {{
                var el = document.getElementById(prefix + f);
                if (el) params[f] = parseFloat(el.value);
            }});
            showLoading("Reconfiguring constellation...");
            apiPut("/api/constellation/" + layerId, {{params: params}}).then(function() {{
                return loadLayerCzml(layerId);
            }}).then(function() {{
                // Reload any analysis layers that depend on this constellation
                return apiGet("/api/state");
            }}).then(function(state) {{
                var reloads = [];
                state.layers.forEach(function(l) {{
                    if (l.category === "Analysis" && layerSources[l.layer_id]) {{
                        reloads.push(loadLayerCzml(l.layer_id));
                    }}
                }});
                return Promise.all(reloads);
            }}).then(function() {{
                rebuildPanel();
                if (satTableLayerId === layerId) loadSatTable(layerId);
                hideLoading();
                showToast("Constellation reconfigured", "success");
            }}).catch(function(e) {{
                hideLoading();
                showToast("Reconfigure error: " + e.message, "error");
            }});
        }}

        // --- Update color legend ---
        function updateLegend(legend, title) {{
            var el = document.getElementById("colorLegend");
            if (!legend || legend.length === 0) {{
                el.style.display = "none";
                return;
            }}
            var html = '<div class="legend-title">' + escapeHtml(title || "Legend") + '</div>';
            legend.forEach(function(entry) {{
                html += '<div class="legend-entry">' +
                    '<span class="legend-swatch" style="background:' + escapeHtml(entry.color) + '"></span>' +
                    '<span>' + escapeHtml(entry.label) + '</span></div>';
            }});
            el.innerHTML = html;
            el.style.display = "block";
        }}

        // --- Export layer CZML ---
        function exportLayer(layerId) {{
            window.open(API + "/api/export/" + layerId, "_blank");
        }}

        // --- Session save/load ---
        function saveSession() {{
            var scenarioName = prompt("Scenario name:", "");
            if (scenarioName === null) return;
            if (!scenarioName.trim()) scenarioName = "Untitled";
            var scenarioDesc = prompt("Description (optional):", "");
            if (scenarioDesc === null) scenarioDesc = "";
            apiPost("/api/session/save", {{name: scenarioName, description: scenarioDesc}}).then(function(resp) {{
                var blob = new Blob([JSON.stringify(resp.session, null, 2)], {{type: "application/json"}});
                var a = document.createElement("a");
                a.href = URL.createObjectURL(blob);
                a.download = scenarioName.replace(/[^a-zA-Z0-9_-]/g, "_") + ".json";
                a.click();
                setTimeout(function() {{ URL.revokeObjectURL(a.href); }}, 1000);
                addToRecentScenarios(resp.session);
                showToast("Session saved: " + scenarioName, "success");
            }}).catch(function(e) {{
                showToast("Save error: " + e.message, "error");
            }});
        }}
        function addToRecentScenarios(session) {{
            var recent = JSON.parse(localStorage.getItem("humeris_recent") || "[]");
            recent.unshift({{
                name: session.name || "Untitled",
                description: session.description || "",
                timestamp: session.timestamp,
                layer_count: session.layer_summary ? session.layer_summary.total : 0,
            }});
            if (recent.length > 5) recent = recent.slice(0, 5);
            localStorage.setItem("humeris_recent", JSON.stringify(recent));
            renderRecentScenarios();
        }}
        function renderRecentScenarios() {{
            var container = document.getElementById("recentScenarios");
            if (!container) return;
            var recent = JSON.parse(localStorage.getItem("humeris_recent") || "[]");
            if (recent.length === 0) {{
                container.innerHTML = '<div style="color:rgba(255,255,255,0.4);font-style:italic;padding:4px 0">No recent scenarios</div>';
                return;
            }}
            var html = "";
            recent.forEach(function(r) {{
                var ts = r.timestamp ? new Date(r.timestamp).toLocaleDateString() : "";
                html += '<div class="recent-item" title="' + escapeHtml(r.description || "") + '">';
                html += '<span class="recent-name">' + escapeHtml(r.name || "Untitled") + '</span>';
                html += '<span class="recent-meta">' + escapeHtml(r.layer_count) + ' layers | ' + escapeHtml(ts) + '</span>';
                html += '</div>';
            }});
            container.innerHTML = html;
        }}
        // --- Report (APP-08) ---
        function generateReport() {{
            window.open(API + "/api/report", "_blank");
        }}

        // --- Constraints (APP-07) ---
        function addConstraint() {{
            var metric = document.getElementById("cst-metric").value.trim();
            var op = document.getElementById("cst-op").value;
            var threshold = parseFloat(document.getElementById("cst-threshold").value);
            if (!metric || isNaN(threshold)) {{ showToast("Fill in metric and threshold", "error"); return; }}
            apiPost("/api/constraints/add", {{metric: metric, operator: op, threshold: threshold}}).then(function(resp) {{
                showToast("Constraint added (" + resp.count + " total)", "success");
                document.getElementById("cst-metric").value = "";
                document.getElementById("cst-threshold").value = "";
            }}).catch(function(e) {{ showToast("Error: " + e.message, "error"); }});
        }}

        function evaluateConstraints() {{
            apiGet("/api/state").then(function(state) {{
                var constLayer = state.layers.find(function(l) {{ return l.category === "Constellation"; }});
                if (!constLayer) {{ showToast("No constellation to evaluate", "error"); return; }}
                return apiPost("/api/constraints/evaluate", {{layer_id: constLayer.layer_id}});
            }}).then(function(resp) {{
                if (!resp) return;
                var html = '<div style="margin-bottom:4px;color:rgba(80,160,255,0.9)">' + resp.summary + '</div>';
                resp.results.forEach(function(r) {{
                    var icon = r.passed ? '<span style="color:#4f4">&#10003;</span>' : '<span style="color:#f44">&#10007;</span>';
                    html += '<div>' + icon + ' ' + r.metric.replace(/_/g, " ") + ' ' + r.operator + ' ' + r.threshold;
                    if (r.actual !== null) html += ' (actual: ' + (typeof r.actual === "number" ? r.actual.toFixed(2) : r.actual) + ')';
                    html += '</div>';
                }});
                document.getElementById("constraintResults").innerHTML = html;
            }}).catch(function(e) {{ showToast("Evaluate error: " + e.message, "error"); }});
        }}

        // --- Compare (APP-06) ---
        function populateCompareSelects() {{
            apiGet("/api/state").then(function(state) {{
                var selA = document.getElementById("cmp-a");
                var selB = document.getElementById("cmp-b");
                if (!selA || !selB) return;
                selA.innerHTML = "";
                selB.innerHTML = "";
                state.layers.forEach(function(l) {{
                    if (l.category === "Constellation") {{
                        var optA = document.createElement("option");
                        optA.value = l.layer_id;
                        optA.textContent = l.name.split(":").pop();
                        selA.appendChild(optA);
                        var optB = optA.cloneNode(true);
                        selB.appendChild(optB);
                    }}
                }});
            }});
        }}

        function runCompare() {{
            var layerA = document.getElementById("cmp-a").value;
            var layerB = document.getElementById("cmp-b").value;
            if (!layerA || !layerB) {{ showToast("Select two constellations", "error"); return; }}
            if (layerA === layerB) {{ showToast("Select different constellations", "error"); return; }}
            apiPost("/api/compare", {{layer_a: layerA, layer_b: layerB}}).then(function(result) {{
                var html = '<table style="width:100%;border-collapse:collapse">';
                html += '<tr><th style="text-align:left;color:rgba(255,255,255,0.5)">Metric</th>';
                html += '<th style="text-align:right;color:rgba(80,160,255,0.8)">' + escapeHtml(result.config_a.name.split(":").pop()) + '</th>';
                html += '<th style="text-align:right;color:rgba(80,200,120,0.8)">' + escapeHtml(result.config_b.name.split(":").pop()) + '</th>';
                html += '<th style="text-align:right;color:rgba(255,200,80,0.8)">Delta</th></tr>';
                var allKeys = Object.keys(result.delta);
                allKeys.forEach(function(k) {{
                    var va = result.config_a.metrics[k];
                    var vb = result.config_b.metrics[k];
                    var d = result.delta[k];
                    var dColor = d > 0 ? "rgba(80,200,120,0.9)" : d < 0 ? "rgba(255,100,100,0.9)" : "rgba(255,255,255,0.5)";
                    var dStr = d > 0 ? "+" + d.toFixed(2) : d.toFixed(2);
                    html += '<tr>';
                    html += '<td style="color:rgba(255,255,255,0.6)">' + escapeHtml(k.replace(/_/g, " ")) + '</td>';
                    html += '<td style="text-align:right;font-family:monospace">' + (va !== undefined ? (typeof va === "number" ? va.toFixed(2) : va) : "-") + '</td>';
                    html += '<td style="text-align:right;font-family:monospace">' + (vb !== undefined ? (typeof vb === "number" ? vb.toFixed(2) : vb) : "-") + '</td>';
                    html += '<td style="text-align:right;font-family:monospace;color:' + dColor + '">' + dStr + '</td>';
                    html += '</tr>';
                }});
                html += '</table>';
                document.getElementById("compareResult").innerHTML = html;
            }}).catch(function(e) {{
                showToast("Compare error: " + e.message, "error");
            }});
        }}

        // --- Parameter sweep (APP-05) ---
        var sweepResults = null;
        var sweepParam = "";
        var sweepMetric = "";

        function runSweep() {{
            var param = document.getElementById("sw-param").value;
            var sMin = parseFloat(document.getElementById("sw-min").value);
            var sMax = parseFloat(document.getElementById("sw-max").value);
            var sStep = parseFloat(document.getElementById("sw-step").value);
            var metric = document.getElementById("sw-metric").value;
            if (sStep <= 0 || sMin >= sMax) {{
                showToast("Invalid sweep range", "error");
                return;
            }}
            // Get base params from first constellation layer
            showLoading("Running sweep...");
            apiGet("/api/state").then(function(state) {{
                var constLayer = state.layers.find(function(l) {{ return l.category === "Constellation"; }});
                var baseParams = constLayer ? constLayer.params : {{altitude_km: 550, inclination_deg: 53, num_planes: 6, sats_per_plane: 10, phase_factor: 0, raan_offset_deg: 0}};
                return apiPost("/api/sweep", {{
                    base_params: baseParams,
                    sweep_param: param,
                    sweep_min: sMin,
                    sweep_max: sMax,
                    sweep_step: sStep,
                    metric_type: metric,
                }});
            }}).then(function(resp) {{
                sweepResults = resp.results;
                sweepParam = param;
                sweepMetric = metric;
                hideLoading();
                renderSweepChart();
                showToast("Sweep complete: " + sweepResults.length + " points", "success");
            }}).catch(function(e) {{
                hideLoading();
                showToast("Sweep error: " + e.message, "error");
            }});
        }}

        function renderSweepChart() {{
            if (!sweepResults || sweepResults.length === 0) return;
            var chart = document.getElementById("sweepChart");
            var canvas = document.getElementById("sweepCanvas");
            var ctx = canvas.getContext("2d");
            chart.classList.add("visible");

            canvas.width = 576;
            canvas.height = 320;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Extract data — pick first numeric metric key
            var metricKeys = Object.keys(sweepResults[0].metrics || {{}});
            var numericKey = metricKeys.find(function(k) {{ return typeof sweepResults[0].metrics[k] === "number"; }});
            if (!numericKey) {{ ctx.fillStyle = "#fff"; ctx.fillText("No numeric metrics", 100, 160); return; }}

            document.getElementById("sweepTitle").textContent = sweepParam.replace(/_/g, " ") + " vs " + numericKey.replace(/_/g, " ");

            var xs = sweepResults.map(function(r) {{ return r.params[sweepParam]; }});
            var ys = sweepResults.map(function(r) {{ return r.metrics[numericKey] || 0; }});

            var xMin = Math.min.apply(null, xs), xMax = Math.max.apply(null, xs);
            var yMin = Math.min.apply(null, ys), yMax = Math.max.apply(null, ys);
            if (yMax === yMin) yMax = yMin + 1;

            var pad = {{left: 60, right: 20, top: 20, bottom: 40}};
            var w = canvas.width - pad.left - pad.right;
            var h = canvas.height - pad.top - pad.bottom;

            function toX(v) {{ return pad.left + (v - xMin) / (xMax - xMin || 1) * w; }}
            function toY(v) {{ return pad.top + h - (v - yMin) / (yMax - yMin) * h; }}

            // Grid
            ctx.strokeStyle = "rgba(255,255,255,0.1)";
            ctx.lineWidth = 1;
            for (var i = 0; i <= 4; i++) {{
                var gy = pad.top + h * i / 4;
                ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(pad.left + w, gy); ctx.stroke();
            }}

            // Axes labels
            ctx.fillStyle = "rgba(255,255,255,0.5)";
            ctx.font = "10px monospace";
            ctx.textAlign = "center";
            ctx.fillText(sweepParam.replace(/_/g, " "), pad.left + w / 2, canvas.height - 5);
            for (var i = 0; i <= 4; i++) {{
                var yv = yMin + (yMax - yMin) * (4 - i) / 4;
                ctx.textAlign = "right";
                ctx.fillText(yv.toFixed(1), pad.left - 5, pad.top + h * i / 4 + 4);
            }}
            xs.forEach(function(x) {{
                ctx.textAlign = "center";
                ctx.fillText(x.toFixed(0), toX(x), canvas.height - 22);
            }});

            // Line
            ctx.strokeStyle = "rgba(80,160,255,0.9)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            xs.forEach(function(x, i) {{
                var px = toX(x), py = toY(ys[i]);
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }});
            ctx.stroke();

            // Points
            ctx.fillStyle = "rgba(80,200,255,1)";
            xs.forEach(function(x, i) {{
                ctx.beginPath();
                ctx.arc(toX(x), toY(ys[i]), 4, 0, 2 * Math.PI);
                ctx.fill();
            }});
        }}

        function closeSweepChart() {{
            document.getElementById("sweepChart").classList.remove("visible");
        }}

        function exportSweepCSV() {{
            if (!sweepResults) return;
            var cols = Object.keys(sweepResults[0].params).concat(Object.keys(sweepResults[0].metrics || {{}}));
            var csv = cols.join(",") + "\\n";
            sweepResults.forEach(function(r) {{
                var row = cols.map(function(c) {{
                    return r.params[c] !== undefined ? r.params[c] : (r.metrics[c] !== undefined ? r.metrics[c] : "");
                }});
                csv += row.join(",") + "\\n";
            }});
            var blob = new Blob([csv], {{type: "text/csv"}});
            var a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = "sweep_" + sweepParam + ".csv";
            a.click();
            setTimeout(function() {{ URL.revokeObjectURL(a.href); }}, 1000);
        }}

        function clearAllSources() {{
            // Remove all tracked data sources from the viewer
            Object.keys(layerSources).forEach(function(lid) {{
                if (layerSources[lid]) {{
                    viewer.dataSources.remove(layerSources[lid], true);
                }}
            }});
            layerSources = {{}};
        }}
        function loadSession() {{
            var input = document.createElement("input");
            input.type = "file";
            input.accept = ".json";
            input.onchange = function(ev) {{
                var file = ev.target.files[0];
                if (!file) return;
                var reader = new FileReader();
                reader.onload = function(e) {{
                    try {{
                        var session = JSON.parse(e.target.result);
                        // Show metadata confirmation
                        var sName = session.name || "Untitled";
                        var sDesc = session.description || "";
                        var sDate = session.timestamp ? new Date(session.timestamp).toLocaleString() : "unknown";
                        var sLayers = session.layer_summary ? session.layer_summary.total : (session.layers ? session.layers.length : 0);
                        var msg = "Load scenario?\\n\\nName: " + sName;
                        if (sDesc) msg += "\\nDescription: " + sDesc;
                        msg += "\\nDate: " + sDate + "\\nLayers: " + sLayers;
                        if (!confirm(msg)) return;
                        clearAllSources();
                        capToastIds = new Set();
                        apiPost("/api/session/load", {{session: session}}).then(function() {{
                            apiGet("/api/state").then(function(state) {{
                                var promises = state.layers.map(function(l) {{
                                    return loadLayerCzml(l.layer_id);
                                }});
                                Promise.all(promises).then(function() {{
                                    rebuildPanel();
                                }});
                            }});
                            showToast("Loaded: " + sName, "success");
                        }}).catch(function(err) {{
                            showToast("Load error: " + err.message, "error");
                        }});
                    }} catch(err) {{
                        showToast("Invalid session file", "error");
                    }}
                }};
                reader.readAsText(file);
            }};
            input.click();
        }}

        // --- Apply simulation settings ---
        function applySettings() {{
            var dur = parseFloat(document.getElementById("sim-duration").value) * 3600;
            var step = parseFloat(document.getElementById("sim-step").value);
            apiPut("/api/settings", {{duration_s: dur, step_s: step}}).then(function() {{
                // Sync Cesium viewer.clock timeline bounds
                var start = viewer.clock.startTime;
                var stop = Cesium.JulianDate.addSeconds(start, dur, new Cesium.JulianDate());
                viewer.clock.stopTime = stop;
                viewer.timeline.zoomTo(start, stop);
                showToast("Settings applied", "success");
            }}).catch(function(e) {{
                showToast("Settings error: " + e.message, "error");
            }});
        }}

        // --- Rebuild Layer Panel ---
        function rebuildPanel() {{
            apiGet("/api/state").then(function(state) {{
                var list = document.getElementById("layerList");
                var sourceSelect = document.getElementById("analysis-source");
                list.innerHTML = "";
                sourceSelect.innerHTML = "";
                var cmpAsel = document.getElementById("cmp-a");
                var cmpBsel = document.getElementById("cmp-b");
                if (cmpAsel) cmpAsel.innerHTML = "";
                if (cmpBsel) cmpBsel.innerHTML = "";

                // Update simulation settings from state
                if (state.duration_s) {{
                    document.getElementById("sim-duration").value = (state.duration_s / 3600).toFixed(1);
                }}
                if (state.step_s) {{
                    document.getElementById("sim-step").value = state.step_s;
                }}

                if (state.layers.length === 0) {{
                    list.innerHTML = '<div style="color:rgba(255,255,255,0.4);font-style:italic">No layers yet</div>';
                    document.getElementById("colorLegend").style.display = "none";
                    return;
                }}

                // Group by category
                var cats = {{}};
                state.layers.forEach(function(layer) {{
                    if (!cats[layer.category]) cats[layer.category] = [];
                    cats[layer.category].push(layer);
                    // Populate source dropdown for constellations
                    if (layer.category === "Constellation") {{
                        var opt = document.createElement("option");
                        opt.value = layer.layer_id;
                        opt.textContent = layer.name.split(":").pop();
                        sourceSelect.appendChild(opt);
                        // Also populate compare selects (APP-06)
                        var cmpA = document.getElementById("cmp-a");
                        var cmpB = document.getElementById("cmp-b");
                        if (cmpA && cmpB) {{
                            cmpA.appendChild(opt.cloneNode(true));
                            cmpB.appendChild(opt.cloneNode(true));
                        }}
                    }}
                }});

                Object.keys(cats).forEach(function(cat) {{
                    var catDiv = document.createElement("div");

                    // Category header with Show All / Hide All
                    var header = document.createElement("div");
                    header.className = "cat-header";
                    var catLabel = document.createElement("strong");
                    catLabel.textContent = cat;
                    header.appendChild(catLabel);
                    var btns = document.createElement("div");
                    btns.className = "cat-buttons";
                    var showBtn = document.createElement("button");
                    showBtn.className = "btn-sm";
                    showBtn.textContent = "Show All";
                    showBtn.style.cssText = "font-size:9px;padding:1px 4px;cursor:pointer;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:#fff;border-radius:2px;";
                    showBtn.onclick = (function(items) {{
                        return function() {{
                            var promises = items.map(function(l) {{ return toggleVisible(l.layer_id, true); }});
                            Promise.all(promises).then(rebuildPanel);
                        }};
                    }})(cats[cat]);
                    var hideBtn = document.createElement("button");
                    hideBtn.className = "btn-sm";
                    hideBtn.textContent = "Hide All";
                    hideBtn.style.cssText = showBtn.style.cssText;
                    hideBtn.onclick = (function(items) {{
                        return function() {{
                            var promises = items.map(function(l) {{ return toggleVisible(l.layer_id, false); }});
                            Promise.all(promises).then(rebuildPanel);
                        }};
                    }})(cats[cat]);
                    btns.appendChild(showBtn);
                    btns.appendChild(hideBtn);
                    header.appendChild(btns);
                    catDiv.appendChild(header);

                    // Layer items
                    cats[cat].forEach(function(layer) {{
                        var item = document.createElement("div");
                        item.className = "layer-item";

                        var cb = document.createElement("input");
                        cb.type = "checkbox";
                        cb.checked = layer.visible;
                        cb.onchange = function() {{ toggleVisible(layer.layer_id, this.checked); }};
                        item.appendChild(cb);

                        var nameSpan = document.createElement("span");
                        nameSpan.className = "layer-name";
                        var displayName = layer.name.split(":").pop();
                        nameSpan.textContent = displayName;
                        nameSpan.title = "Double-click to rename";
                        nameSpan.ondblclick = (function(lid, span) {{
                            return function() {{
                                var inp = document.createElement("input");
                                inp.className = "rename-input";
                                inp.value = span.textContent;
                                inp._cancelled = false;
                                inp.onblur = function() {{
                                    if (!inp._cancelled && inp.parentNode) {{
                                        renameLayer(lid, inp.value);
                                    }}
                                }};
                                inp.onkeydown = function(ev) {{
                                    if (ev.key === "Enter") inp.blur();
                                    if (ev.key === "Escape") {{
                                        inp._cancelled = true;
                                        span.style.display = "";
                                        inp.remove();
                                    }}
                                }};
                                span.style.display = "none";
                                span.parentNode.insertBefore(inp, span.nextSibling);
                                inp.focus();
                                inp.select();
                            }};
                        }})(layer.layer_id, nameSpan);
                        item.appendChild(nameSpan);

                        var countSpan = document.createElement("span");
                        countSpan.className = "entity-count";
                        var countText = "(" + layer.num_entities + ")";
                        if (layer.capped_from) {{
                            countText = "(" + layer.num_entities + " of " + layer.capped_from + ")";
                        }}
                        countSpan.textContent = countText;
                        item.appendChild(countSpan);

                        // Opacity slider (BUG-009: preserve value across rebuilds)
                        var opSlider = document.createElement("input");
                        opSlider.type = "range";
                        opSlider.className = "opacity-slider";
                        opSlider.min = "0";
                        opSlider.max = "100";
                        opSlider.value = layerOpacities[layer.layer_id] != null ? layerOpacities[layer.layer_id] : 100;
                        opSlider.title = "Opacity";
                        opSlider.oninput = (function(lid) {{
                            return function() {{
                                var v = parseInt(this.value);
                                layerOpacities[lid] = v;
                                setOpacity(lid, v);
                            }};
                        }})(layer.layer_id);
                        item.appendChild(opSlider);

                        // Export button
                        var expBtn = document.createElement("button");
                        expBtn.className = "mode-toggle";
                        expBtn.textContent = "\u2913";
                        expBtn.title = "Export CZML";
                        expBtn.onclick = (function(lid) {{
                            return function() {{ exportLayer(lid); }};
                        }})(layer.layer_id);
                        item.appendChild(expBtn);

                        // Mode toggle
                        var modeBtn = document.createElement("button");
                        modeBtn.className = "mode-toggle";
                        var nextMode = layer.mode === "animated" ? "snapshot" : "animated";
                        modeBtn.textContent = layer.mode === "animated" ? "Anim" : "Snap";
                        modeBtn.title = "Switch to " + nextMode;
                        modeBtn.onclick = function() {{ toggleMode(layer.layer_id, nextMode); }};
                        item.appendChild(modeBtn);

                        // Remove button
                        var rmBtn = document.createElement("button");
                        rmBtn.className = "mode-toggle";
                        rmBtn.textContent = "\\u2715";
                        rmBtn.title = "Remove layer";
                        rmBtn.style.color = "rgba(255,100,100,0.8)";
                        rmBtn.onclick = function() {{ removeLayer(layer.layer_id); }};
                        item.appendChild(rmBtn);

                        catDiv.appendChild(item);

                        // Statistics line
                        if (layer.num_entities > 0) {{
                            var statsDiv = document.createElement("div");
                            statsDiv.className = "layer-stats";
                            var statsText = layer.layer_type + " | " + layer.num_entities + " entities";
                            if (layer.capped_from) {{
                                statsText += " (capped from " + layer.capped_from + ")";
                                if (!capToastIds.has(layer.layer_id)) {{
                                    capToastIds.add(layer.layer_id);
                                    showToast("Layer capped: showing " + layer.num_entities + " of " + layer.capped_from + " satellites", "info");
                                }}
                            }}
                            statsDiv.textContent = statsText;
                            catDiv.appendChild(statsDiv);
                        }}

                        // Editable parameter form for walker layers (APP-01)
                        if (layer.editable && layer.params) {{
                            var epDiv = document.createElement("div");
                            epDiv.className = "edit-params";
                            var prefix = "ep-" + layer.layer_id + "-";
                            var epFields = [
                                ["altitude_km", "Alt (km)", layer.params.altitude_km],
                                ["inclination_deg", "Inc (deg)", layer.params.inclination_deg],
                                ["num_planes", "Planes", layer.params.num_planes],
                                ["sats_per_plane", "Sats/plane", layer.params.sats_per_plane],
                                ["phase_factor", "Phase F", layer.params.phase_factor],
                                ["raan_offset_deg", "RAAN off", layer.params.raan_offset_deg],
                            ];
                            epFields.forEach(function(f) {{
                                if (f[2] == null) return;
                                var row = document.createElement("div");
                                row.className = "ep-row";
                                var lbl = document.createElement("label");
                                lbl.textContent = f[1];
                                row.appendChild(lbl);
                                var inp = document.createElement("input");
                                inp.type = "number";
                                inp.id = prefix + f[0];
                                inp.value = f[2];
                                inp.step = f[0] === "num_planes" || f[0] === "sats_per_plane" || f[0] === "phase_factor" ? "1" : "0.1";
                                row.appendChild(inp);
                                epDiv.appendChild(row);
                            }});
                            var applyBtn = document.createElement("button");
                            applyBtn.className = "ep-apply";
                            applyBtn.textContent = "Apply";
                            applyBtn.onclick = (function(lid) {{
                                return function() {{ reconfigureConstellation(lid); }};
                            }})(layer.layer_id);
                            epDiv.appendChild(applyBtn);
                            catDiv.appendChild(epDiv);
                        }}

                        // Metrics display for analysis layers (APP-02)
                        if (layer.metrics) {{
                            var mDiv = document.createElement("div");
                            mDiv.className = "layer-metrics";
                            Object.keys(layer.metrics).forEach(function(key) {{
                                var val = layer.metrics[key];
                                if (typeof val === "object" && val !== null) {{
                                    Object.keys(val).forEach(function(subk) {{
                                        var row = document.createElement("div");
                                        row.className = "metric-row";
                                        row.innerHTML = '<span class="metric-key">' + key + " " + subk + '</span><span class="metric-val">' + val[subk] + '</span>';
                                        mDiv.appendChild(row);
                                    }});
                                }} else {{
                                    var row = document.createElement("div");
                                    row.className = "metric-row";
                                    row.innerHTML = '<span class="metric-key">' + key.replace(/_/g, " ") + '</span><span class="metric-val">' + val + '</span>';
                                    mDiv.appendChild(row);
                                }}
                            }});
                            catDiv.appendChild(mDiv);
                        }}

                        // Show legend for analysis layers
                        if (layer.legend) {{
                            updateLegend(layer.legend, layer.name.split(":").pop());
                        }}
                    }});

                    list.appendChild(catDiv);
                }});
            }});
        }}

        // Load CZML for all layers the server already knows about
        function loadExistingLayers() {{
            apiGet("/api/state").then(function(state) {{
                if (state.layers.length === 0) {{
                    rebuildPanel();
                    return;
                }}
                var promises = state.layers.map(function(layer) {{
                    return loadLayerCzml(layer.layer_id);
                }});
                Promise.all(promises).then(function() {{
                    rebuildPanel();
                    // Set camera to show full Earth + constellation shells
                    viewer.camera.setView({{
                        destination: Cesium.Cartesian3.fromDegrees(10, 20, 15000000),
                    }});
                }}).catch(function(e) {{
                    showToast("Error loading layers: " + e.message, "error");
                    rebuildPanel();
                }});
            }});
        }}
        loadExistingLayers();
        renderRecentScenarios();

        // Zoom to first data source when loaded dynamically
        viewer.dataSources.dataSourceAdded.addEventListener(function(collection, ds) {{
            viewer.scene.requestRender();
        }});
    </script>
</body>
</html>"""


def write_cesium_html(
    czml_packets: list[dict],
    path: str,
    title: str = "Constellation Viewer",
    cesium_token: str = "",
    additional_layers: list[list[dict]] | None = None,
) -> str:
    """Write self-contained Cesium HTML viewer to file.

    Args:
        czml_packets: CZML packet list.
        path: Output file path.
        title: HTML page title.
        cesium_token: Cesium Ion access token (optional).
        additional_layers: Extra CZML packet lists to overlay.

    Returns:
        The output path.
    """
    html = generate_cesium_html(
        czml_packets, title=title, cesium_token=cesium_token,
        additional_layers=additional_layers,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
