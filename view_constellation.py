#!/usr/bin/env python3
# Browser test: Walker shells + live ISS & Starlink from CelesTrak.
#
# Uses static snapshots for large constellations (fast, ~200 bytes/sat)
# and animated tracks only for ISS (1 object where animation is useful).
#
# Usage:
#     python view_constellation.py                    # generate static HTML
#     python view_constellation.py --open             # generate + open in browser
#     python view_constellation.py --serve            # interactive server mode
#     python view_constellation.py --serve --port 9000  # custom port
#
# Static mode output: constellation_viewer.html (self-contained, no server)
# Server mode: opens browser to localhost, add/remove constellations live
# Requires: pip install constellation-generator[live]

import sys
import webbrowser
from datetime import datetime, timedelta, timezone

from constellation_generator.domain.constellation import (
    ShellConfig,
    generate_walker_shell,
)
from constellation_generator.domain.propagation import derive_orbital_state
from constellation_generator.domain.observation import GroundStation
from constellation_generator.adapters.czml_exporter import (
    constellation_packets,
    snapshot_packets,
)
from constellation_generator.adapters.czml_visualization import (
    eclipse_snapshot_packets,
    ground_station_packets,
)
from constellation_generator.adapters.cesium_viewer import write_cesium_html
from constellation_generator.adapters.celestrak import CelesTrakAdapter

DURATION = timedelta(hours=2)
STEP = timedelta(seconds=60)
OUTPUT = "constellation_viewer.html"

# Default Walker shells for pre-loading
DEFAULT_SHELLS = [
    ShellConfig(
        altitude_km=500, inclination_deg=30,
        num_planes=22, sats_per_plane=72,
        phase_factor=17, raan_offset_deg=0.0,
        shell_name="Walker-500",
    ),
    ShellConfig(
        altitude_km=450, inclination_deg=30,
        num_planes=22, sats_per_plane=72,
        phase_factor=17, raan_offset_deg=5.45,
        shell_name="Walker-450",
    ),
    ShellConfig(
        altitude_km=400, inclination_deg=30,
        num_planes=22, sats_per_plane=72,
        phase_factor=17, raan_offset_deg=10.9,
        shell_name="Walker-400",
    ),
]

STATIONS = [
    GroundStation(name="Svalbard", lat_deg=78.23, lon_deg=15.39, alt_m=0.0),
    GroundStation(name="Maspalomas", lat_deg=27.76, lon_deg=-15.59, alt_m=0.0),
    GroundStation(name="Kiruna", lat_deg=67.86, lon_deg=20.96, alt_m=0.0),
    GroundStation(name="Santiago", lat_deg=-33.15, lon_deg=-70.67, alt_m=0.0),
    GroundStation(name="Canberra", lat_deg=-35.40, lon_deg=148.98, alt_m=0.0),
]


def run_serve_mode(port=8765):
    """Start interactive viewer server with pre-loaded constellations."""
    from constellation_generator.adapters.viewer_server import (
        LayerManager,
        create_viewer_server,
    )

    epoch = datetime.now(tz=timezone.utc)
    mgr = LayerManager(epoch=epoch)

    # Pre-load Walker shells
    for shell in DEFAULT_SHELLS:
        print(f"  Generating {shell.shell_name}...")
        sats = generate_walker_shell(shell)
        states = [derive_orbital_state(s, epoch, include_j2=True) for s in sats]
        mgr.add_layer(
            name=f"Constellation:{shell.shell_name}",
            category="Constellation",
            layer_type="walker",
            states=states,
            params={
                "altitude_km": shell.altitude_km,
                "inclination_deg": shell.inclination_deg,
                "num_planes": shell.num_planes,
                "sats_per_plane": shell.sats_per_plane,
                "phase_factor": shell.phase_factor,
                "raan_offset_deg": shell.raan_offset_deg,
                "shell_name": shell.shell_name,
            },
        )
        print(f"    {len(sats)} satellites")

    # Pre-load ISS from CelesTrak
    print("  Fetching ISS from CelesTrak...")
    try:
        celestrak = CelesTrakAdapter()
        iss_sats = celestrak.fetch_satellites(name="ISS (ZARYA)", epoch=epoch)
        if iss_sats:
            iss_states = [derive_orbital_state(s, epoch, include_j2=True) for s in iss_sats]
            mgr.add_layer(
                name="Constellation:ISS",
                category="Constellation",
                layer_type="celestrak",
                states=iss_states,
                params={"name": "ISS (ZARYA)"},
                mode="animated",
            )
            print(f"    ISS: {len(iss_sats)} object(s)")
    except Exception as e:
        print(f"    ISS fetch failed: {e}")

    # Start server
    server = create_viewer_server(mgr, port=port)
    url = f"http://localhost:{port}"
    print(f"\nInteractive viewer server running at {url}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


def run_static_mode():
    """Generate static HTML file (legacy mode)."""
    epoch = datetime.now(tz=timezone.utc)

    # --- Walker shells (synthetic, static snapshots) ---
    walker_sats = []
    for shell in DEFAULT_SHELLS:
        sats = generate_walker_shell(shell)
        walker_sats.extend(sats)
        print(f"  {shell.shell_name}: {len(sats)} satellites")
    print(f"Walker total: {len(walker_sats)} satellites")

    walker_states = [derive_orbital_state(s, epoch, include_j2=True) for s in walker_sats]

    # Static snapshot for Walker (fast, ~200 bytes/sat instead of ~10KB)
    walker_pkts = snapshot_packets(walker_states, epoch, name="Constellation:Walker")

    # Eclipse analysis layer for Walker
    print("Computing eclipse states...")
    eclipse_pkts = eclipse_snapshot_packets(walker_states, epoch, name="Analysis:Eclipse")

    # --- Live data from CelesTrak ---
    celestrak = CelesTrakAdapter()
    layers = []

    # ISS (animated — only 1 object, animation is useful)
    print("Fetching ISS from CelesTrak...")
    try:
        iss_sats = celestrak.fetch_satellites(name="ISS (ZARYA)", epoch=epoch)
        if iss_sats:
            iss_states = [derive_orbital_state(s, epoch, include_j2=True) for s in iss_sats]
            iss_pkts = constellation_packets(iss_states, epoch, DURATION, STEP)
            iss_pkts[0]["name"] = "Constellation:ISS"
            layers.append(iss_pkts)
            print(f"  ISS: {len(iss_sats)} object(s) (animated)")
    except Exception as e:
        print(f"  ISS fetch failed: {e}")

    # Starlink (static snapshot)
    print("Fetching Starlink from CelesTrak...")
    try:
        starlink_sats = celestrak.fetch_satellites(group="STARLINK", epoch=epoch)
        if starlink_sats:
            starlink_states = [derive_orbital_state(s, epoch, include_j2=True) for s in starlink_sats]
            starlink_pkts = snapshot_packets(starlink_states, epoch, name="Constellation:Starlink")
            layers.append(starlink_pkts)
            print(f"  Starlink: {len(starlink_sats)} satellites (snapshot)")
    except Exception as e:
        print(f"  Starlink fetch failed: {e}")

    # Eclipse layer
    layers.append(eclipse_pkts)

    # Ground station layers (one per station, using Walker subset for access)
    print("Adding ground stations...")
    walker_subset = walker_states[:6]  # small subset for access computation
    for station in STATIONS:
        station_pkts = ground_station_packets(
            station, walker_subset, epoch, DURATION, STEP,
            name=f"Analysis:{station.name}",
        )
        layers.append(station_pkts)
        print(f"  {station.name}")

    # --- Write HTML ---
    write_cesium_html(
        walker_pkts,
        OUTPUT,
        title="Constellation Viewer — Walker + ISS + Starlink",
        additional_layers=layers if layers else None,
    )
    print(f"\nWritten to {OUTPUT}")

    if "--open" in sys.argv:
        webbrowser.open(OUTPUT)
        print("Opened in browser")
    else:
        print(f"Open {OUTPUT} in a browser, or re-run with --open")


if __name__ == "__main__":
    if "--serve" in sys.argv:
        port = 8765
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        run_serve_mode(port=port)
    else:
        run_static_mode()
