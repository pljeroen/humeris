# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Adapters for simulation I/O and satellite export.

External dependencies (json, csv, file I/O) are confined to this layer.
"""
import json
from typing import Any

from constellation_generator.ports import SimulationReader, SimulationWriter
from constellation_generator.adapters.csv_exporter import CsvSatelliteExporter
from constellation_generator.adapters.geojson_exporter import GeoJsonSatelliteExporter
from constellation_generator.adapters.czml_exporter import (
    constellation_packets,
    constellation_packets_numerical,
    snapshot_packets,
    ground_track_packets,
    coverage_packets,
    write_czml,
)
from constellation_generator.adapters.cesium_viewer import (
    generate_cesium_html,
    generate_interactive_html,
    write_cesium_html,
)
from constellation_generator.adapters.viewer_server import (
    LayerManager,
    LayerState,
    create_viewer_server,
)
from constellation_generator.adapters.czml_visualization import (
    eclipse_constellation_packets,
    eclipse_snapshot_packets,
    sensor_footprint_packets,
    ground_station_packets,
    conjunction_replay_packets,
    coverage_evolution_packets,
    precession_constellation_packets,
    isl_topology_packets,
    fragility_constellation_packets,
    hazard_evolution_packets,
    coverage_connectivity_packets,
    network_eclipse_packets,
)


class JsonSimulationReader(SimulationReader):
    """Reads simulation data from JSON files."""

    def read_simulation(self, path: str) -> dict[str, Any]:
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def extract_template_entity(self, sim_data: dict, entity_name: str) -> dict:
        entities = sim_data.get('Entities', [])
        for entity in entities:
            if entity.get('Name') == entity_name:
                return entity
        raise ValueError(f"Entity '{entity_name}' not found in simulation data")

    def extract_earth_entity(self, sim_data: dict) -> dict:
        return self.extract_template_entity(sim_data, 'Earth')


class JsonSimulationWriter(SimulationWriter):
    """Writes simulation data to JSON files."""

    def write_simulation(self, sim_data: dict, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(sim_data, f, indent=2, ensure_ascii=False)
