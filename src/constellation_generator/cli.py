# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Command-line interface for constellation generation.

Usage:
    # Synthetic Walker shells (no extra deps)
    constellation-generator -i sim_old.json -o sim.json

    # Live data from CelesTrak (requires sgp4)
    constellation-generator -i sim_old.json -o sim.json --live-group GPS-OPS
    constellation-generator -i sim_old.json -o sim.json --live-name "ISS (ZARYA)"
    constellation-generator -i sim_old.json -o sim.json --live-catnr 25544

    # Export to CSV, GeoJSON, or interactive HTML viewer
    constellation-generator -i sim.json -o out.json --export-csv sats.csv
    constellation-generator -i sim.json -o out.json --export-geojson sats.geojson
    constellation-generator -i sim.json -o out.json --export-html viewer.html
"""
import argparse
import sys

from constellation_generator.domain.constellation import (
    ShellConfig,
    Satellite,
    generate_walker_shell,
    generate_sso_band_configs,
)
from constellation_generator.domain.serialization import build_satellite_entity
from constellation_generator.adapters import (
    JsonSimulationReader,
    JsonSimulationWriter,
    CsvSatelliteExporter,
    GeoJsonSatelliteExporter,
    write_cesium_html,
)
from constellation_generator.adapters.czml_exporter import constellation_packets
from constellation_generator.domain.propagation import derive_orbital_state


def get_default_shells() -> list[ShellConfig]:
    """
    Default constellation: 3 Walker shells + SSO band.

    - 3 Walker shells at 500/450/400 km, 30° inc, 22 planes × 72 sats
    - SSO band from 525–2200 km in 50 km steps, 1 plane × 72 sats
    """
    walker_shells = [
        ShellConfig(
            altitude_km=500, inclination_deg=30,
            num_planes=22, sats_per_plane=72,
            phase_factor=17, raan_offset_deg=0.0,
            shell_name='LEO-Shell500',
        ),
        ShellConfig(
            altitude_km=450, inclination_deg=30,
            num_planes=22, sats_per_plane=72,
            phase_factor=17, raan_offset_deg=5.45,
            shell_name='LEO-Shell450',
        ),
        ShellConfig(
            altitude_km=400, inclination_deg=30,
            num_planes=22, sats_per_plane=72,
            phase_factor=17, raan_offset_deg=10.9,
            shell_name='LEO-Shell400',
        ),
    ]

    sso_shells = generate_sso_band_configs(
        start_alt_km=525, end_alt_km=2200,
        step_km=50, sats_per_plane=72,
    )

    return walker_shells + sso_shells


def run(
    input_path: str,
    output_path: str,
    shells: list[ShellConfig] | None = None,
    base_id: int = 100,
    template_name: str = "Satellite",
) -> tuple[int, list[Satellite]]:
    """
    Generate synthetic constellation and write to simulation JSON.

    Returns:
        (count, satellites) — number generated and list of Satellite objects.
    """
    reader = JsonSimulationReader()
    writer = JsonSimulationWriter()

    sim = reader.read_simulation(input_path)
    template = reader.extract_template_entity(sim, template_name)
    earth = reader.extract_earth_entity(sim)

    earth['Position'] = "0;0;0"
    earth['Velocity'] = "0;0;0"

    if shells is None:
        shells = get_default_shells()

    all_satellites: list[Satellite] = []
    entities = sim.get('Entities', [])
    next_id = base_id

    for shell in shells:
        satellites = generate_walker_shell(shell)
        all_satellites.extend(satellites)
        for sat in satellites:
            entity = build_satellite_entity(sat, template, base_id=next_id)
            entities.append(entity)
            next_id += 1

    writer.write_simulation(sim, output_path)
    return len(all_satellites), all_satellites


def run_live(
    input_path: str,
    output_path: str,
    group: str | None = None,
    name: str | None = None,
    catnr: int | None = None,
    base_id: int = 100,
    template_name: str = "Satellite",
    concurrent: bool = False,
) -> tuple[int, list[Satellite]]:
    """
    Fetch live satellite data from CelesTrak and write to simulation JSON.

    Requires: pip install constellation-generator[live]

    Returns:
        (count, satellites) — number generated and list of Satellite objects.
    """
    try:
        from constellation_generator.adapters.celestrak import CelesTrakAdapter
        from constellation_generator.adapters.concurrent_celestrak import ConcurrentCelesTrakAdapter
    except ImportError:
        print(
            "Live data requires the sgp4 package.\n"
            "Install with: pip install constellation-generator[live]",
            file=sys.stderr,
        )
        sys.exit(1)

    reader = JsonSimulationReader()
    writer = JsonSimulationWriter()

    sim = reader.read_simulation(input_path)
    template = reader.extract_template_entity(sim, template_name)
    earth = reader.extract_earth_entity(sim)
    earth['Position'] = "0;0;0"
    earth['Velocity'] = "0;0;0"

    celestrak = ConcurrentCelesTrakAdapter() if concurrent else CelesTrakAdapter()
    mode_label = "concurrent" if concurrent else "sequential"
    print(f"Fetching live data from CelesTrak ({mode_label})...")
    satellites = celestrak.fetch_satellites(group=group, name=name, catnr=catnr)
    print(f"Received {len(satellites)} satellites")

    entities = sim.get('Entities', [])
    next_id = base_id
    for sat in satellites:
        entity = build_satellite_entity(sat, template, base_id=next_id)
        entities.append(entity)
        next_id += 1

    writer.write_simulation(sim, output_path)
    return len(satellites), satellites


def main():
    parser = argparse.ArgumentParser(
        description="Generate satellite constellations for simulation (synthetic or live)"
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help="Path to input simulation JSON (with Earth + Satellite template)"
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help="Path to write output simulation JSON"
    )
    parser.add_argument(
        '--base-id', type=int, default=100,
        help="Starting entity ID (default: 100)"
    )
    parser.add_argument(
        '--template-name', default='Satellite',
        help="Name of satellite template entity (default: Satellite)"
    )

    live_group = parser.add_argument_group('live data (CelesTrak)')
    live_group.add_argument(
        '--live-group',
        help="CelesTrak group (e.g. STATIONS, GPS-OPS, STARLINK, ONEWEB, ACTIVE)"
    )
    live_group.add_argument('--live-name', help="Search by satellite name")
    live_group.add_argument('--live-catnr', type=int, help="NORAD catalog number")
    live_group.add_argument(
        '--concurrent', action='store_true', default=False,
        help="Use concurrent SGP4 propagation (faster for large groups)"
    )

    export_group = parser.add_argument_group('export')
    export_group.add_argument(
        '--export-csv',
        help="Export satellite positions to CSV (geodetic coordinates)"
    )
    export_group.add_argument(
        '--export-geojson',
        help="Export satellite positions to GeoJSON (FeatureCollection)"
    )
    export_group.add_argument(
        '--export-html',
        help="Export interactive 3D viewer as self-contained HTML (CesiumJS)"
    )
    export_group.add_argument(
        '--cesium-token', default="",
        help="Cesium Ion access token for imagery (optional, viewer works without)"
    )

    args = parser.parse_args()

    try:
        live_mode = args.live_group or args.live_name or args.live_catnr
        if live_mode:
            count, satellites = run_live(
                input_path=args.input,
                output_path=args.output,
                group=args.live_group,
                name=args.live_name,
                catnr=args.live_catnr,
                base_id=args.base_id,
                template_name=args.template_name,
                concurrent=args.concurrent,
            )
        else:
            count, satellites = run(
                input_path=args.input,
                output_path=args.output,
                base_id=args.base_id,
                template_name=args.template_name,
            )
        print(f"Generated {args.output} with {count} satellites.")

        if args.export_csv:
            csv_count = CsvSatelliteExporter().export(satellites, args.export_csv)
            print(f"Exported {csv_count} satellites to {args.export_csv}")

        if args.export_geojson:
            geojson_count = GeoJsonSatelliteExporter().export(satellites, args.export_geojson)
            print(f"Exported {geojson_count} satellites to {args.export_geojson}")

        if args.export_html:
            from datetime import datetime, timedelta, timezone
            epoch = datetime.now(tz=timezone.utc)
            states = [derive_orbital_state(s, epoch) for s in satellites]
            pkts = constellation_packets(
                states, epoch, timedelta(hours=2), timedelta(seconds=60),
            )
            write_cesium_html(pkts, args.export_html, cesium_token=args.cesium_token)
            print(f"Exported interactive viewer to {args.export_html}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, ConnectionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
