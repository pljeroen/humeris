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


_CESIUM_VERSION = "1.124"


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
        #loadingIndicator {{
            display: none; position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%); background: rgba(0,0,0,0.8);
            color: #fff; padding: 16px 24px; border-radius: 8px;
            font: 14px sans-serif; z-index: 1000;
        }}
    </style>
</head>
<body>
    <div id="cesiumContainer"></div>
    <div id="infoOverlay">{safe_title}</div>
    <div id="loadingIndicator">Loading...</div>
    <div id="sidePanel">
        <!-- Add Layer Section -->
        <details open>
            <summary>Add Layer</summary>
            <div class="panel-section">
                <!-- Walker Form -->
                <details>
                    <summary>Walker Constellation</summary>
                    <div class="panel-section">
                        <div class="form-row"><label>Altitude (km)</label><input type="number" id="w-alt" value="550" step="10"></div>
                        <div class="form-row"><label>Inclination</label><input type="number" id="w-inc" value="53" step="1"></div>
                        <div class="form-row"><label>Planes</label><input type="number" id="w-planes" value="6" min="1"></div>
                        <div class="form-row"><label>Sats/plane</label><input type="number" id="w-spp" value="10" min="1"></div>
                        <div class="form-row"><label>Phase factor</label><input type="number" id="w-pf" value="1" min="0"></div>
                        <div class="form-row"><label>RAAN offset</label><input type="number" id="w-raan" value="0" step="0.1"></div>
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
                                <option value="STARLINK">Starlink</option>
                                <option value="GPS-OPS">GPS</option>
                                <option value="GLONASS">GLONASS</option>
                                <option value="GALILEO">Galileo</option>
                                <option value="BEIDOU">BeiDou</option>
                                <option value="IRIDIUM-NEXT">Iridium NEXT</option>
                                <option value="ONEWEB">OneWeb</option>
                                <option value="PLANET">Planet</option>
                                <option value="SPIRE">Spire</option>
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
                        <div class="form-row"><label>Latitude</label><input type="number" id="gs-lat" value="0" step="0.01"></div>
                        <div class="form-row"><label>Longitude</label><input type="number" id="gs-lon" value="0" step="0.01"></div>
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
        <!-- Analysis Layers Section -->
        <details>
            <summary>Add Analysis</summary>
            <div class="panel-section">
                <div style="color:rgba(255,255,255,0.5);font-size:11px;margin-bottom:6px">
                    Select a source constellation first, then add analysis layers.
                </div>
                <div class="form-row">
                    <label>Source</label>
                    <select id="analysis-source"></select>
                </div>
                <div class="analysis-grid">
                    <button class="btn btn-sm" onclick="addAnalysis('eclipse')">Eclipse</button>
                    <button class="btn btn-sm" onclick="addAnalysis('coverage')">Coverage</button>
                    <button class="btn btn-sm" onclick="addAnalysis('ground_track')">Ground Track</button>
                    <button class="btn btn-sm" onclick="addAnalysis('sensor')">Sensor FOV</button>
                    <button class="btn btn-sm" onclick="addAnalysis('isl')">ISL Topology</button>
                    <button class="btn btn-sm" onclick="addAnalysis('fragility')">Fragility</button>
                    <button class="btn btn-sm" onclick="addAnalysis('hazard')">Hazard</button>
                    <button class="btn btn-sm" onclick="addAnalysis('network_eclipse')">Net Eclipse</button>
                    <button class="btn btn-sm" onclick="addAnalysis('coverage_connectivity')">Cov+Connect</button>
                    <button class="btn btn-sm" onclick="addAnalysis('precession')">Precession</button>
                    <button class="btn btn-sm" onclick="addAnalysis('kessler_heatmap')">Kessler</button>
                    <button class="btn btn-sm" onclick="addAnalysis('conjunction_hazard')">Conj Hazard</button>
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

        function showLoading(msg) {{
            var el = document.getElementById("loadingIndicator");
            el.textContent = msg || "Loading...";
            el.style.display = "block";
        }}
        function hideLoading() {{
            document.getElementById("loadingIndicator").style.display = "none";
        }}

        // --- API helpers ---
        function apiGet(path) {{
            return fetch(API + path).then(function(r) {{
                if (!r.ok) return r.json().then(function(e) {{ throw new Error(e.error || r.statusText); }});
                return r.json();
            }});
        }}
        function apiPost(path, body) {{
            return fetch(API + path, {{
                method: "POST",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify(body),
            }}).then(function(r) {{
                if (!r.ok) return r.json().then(function(e) {{ throw new Error(e.error || r.statusText); }});
                return r.json();
            }});
        }}
        function apiPut(path, body) {{
            return fetch(API + path, {{
                method: "PUT",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify(body),
            }}).then(function(r) {{
                if (!r.ok) return r.json().then(function(e) {{ throw new Error(e.error || r.statusText); }});
                return r.json();
            }});
        }}
        function apiDelete(path) {{
            return fetch(API + path, {{method: "DELETE"}}).then(function(r) {{
                if (!r.ok) return r.json().then(function(e) {{ throw new Error(e.error || r.statusText); }});
                return r.json();
            }});
        }}

        // --- Load CZML for a layer ---
        function loadLayerCzml(layerId) {{
            return apiGet("/api/czml/" + layerId).then(function(czml) {{
                // Remove old data source if exists
                if (layerSources[layerId]) {{
                    viewer.dataSources.remove(layerSources[layerId]);
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
                alert("Walker error: " + e.message);
            }});
        }}

        // --- Add CelesTrak ---
        function addCelesTrak() {{
            var group = document.getElementById("ct-group").value;
            var name = document.getElementById("ct-name").value;
            if (!group && !name) {{ alert("Select a group or enter a name"); return; }}
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
                alert("CelesTrak error: " + e.message);
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
                alert("Station error: " + e.message);
            }});
        }}

        // --- Add Analysis ---
        function addAnalysis(type) {{
            var sourceEl = document.getElementById("analysis-source");
            if (!sourceEl.value) {{ alert("Select a source constellation first"); return; }}
            showLoading("Computing " + type + " analysis...");
            apiPost("/api/analysis", {{
                type: type,
                source_layer: sourceEl.value,
                params: {{}},
            }}).then(function(resp) {{
                return loadLayerCzml(resp.layer_id);
            }}).then(function() {{
                rebuildPanel();
                hideLoading();
            }}).catch(function(e) {{
                hideLoading();
                alert("Analysis error: " + e.message);
            }});
        }}

        // --- Remove Layer ---
        function removeLayer(layerId) {{
            apiDelete("/api/layer/" + layerId).then(function() {{
                if (layerSources[layerId]) {{
                    viewer.dataSources.remove(layerSources[layerId]);
                    delete layerSources[layerId];
                    viewer.scene.requestRender();
                }}
                rebuildPanel();
            }}).catch(function(e) {{
                alert("Remove error: " + e.message);
            }});
        }}

        // --- Toggle Visibility ---
        function toggleVisible(layerId, visible) {{
            apiPut("/api/layer/" + layerId, {{visible: visible}}).then(function() {{
                if (layerSources[layerId]) {{
                    layerSources[layerId].show = visible;
                    viewer.scene.requestRender();
                }}
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
                alert("Mode switch error: " + e.message);
            }});
        }}

        // --- Rebuild Layer Panel ---
        function rebuildPanel() {{
            apiGet("/api/state").then(function(state) {{
                var list = document.getElementById("layerList");
                var sourceSelect = document.getElementById("analysis-source");
                list.innerHTML = "";
                sourceSelect.innerHTML = "";

                if (state.layers.length === 0) {{
                    list.innerHTML = '<div style="color:rgba(255,255,255,0.4);font-style:italic">No layers yet</div>';
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
                            items.forEach(function(l) {{ toggleVisible(l.layer_id, true); }});
                            setTimeout(rebuildPanel, 200);
                        }};
                    }})(cats[cat]);
                    var hideBtn = document.createElement("button");
                    hideBtn.className = "btn-sm";
                    hideBtn.textContent = "Hide All";
                    hideBtn.style.cssText = showBtn.style.cssText;
                    hideBtn.onclick = (function(items) {{
                        return function() {{
                            items.forEach(function(l) {{ toggleVisible(l.layer_id, false); }});
                            setTimeout(rebuildPanel, 200);
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
                        item.appendChild(nameSpan);

                        var countSpan = document.createElement("span");
                        countSpan.className = "entity-count";
                        countSpan.textContent = "(" + layer.num_entities + ")";
                        item.appendChild(countSpan);

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
                }});
            }});
        }}
        loadExistingLayers();

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
