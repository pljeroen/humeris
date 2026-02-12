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
    </style>
</head>
<body>
    <div id="cesiumContainer"></div>
    <div id="infoOverlay">{safe_title}</div>
    <script>
        {token_line}
        var czmlData = {czml_json};
        var viewer = new Cesium.Viewer("cesiumContainer", {{
            shouldAnimate: true,
            timeline: true,
            animation: true,
            fullscreenButton: true,
            baseLayerPicker: true,
            sceneModePicker: true,
            navigationHelpButton: true,
            scene3DOnly: false,
        }});
        viewer.scene.globe.enableLighting = true;
        var dataSource = new Cesium.CzmlDataSource();
        dataSource.load(czmlData);
        viewer.dataSources.add(dataSource);
        viewer.zoomTo(dataSource);{layers_js}
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
