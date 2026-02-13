# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
JSON simulation file I/O adapter.

Reads and writes simulation data in JSON format.
"""
import json
from typing import Any

from humeris.ports import SimulationReader, SimulationWriter


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
