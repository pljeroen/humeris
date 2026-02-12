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
    ground_track_packets,
    coverage_packets,
    write_czml,
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
