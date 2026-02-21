# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Port interfaces for simulation file I/O.

Adapters implement these to handle different file formats.
"""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SimulationReader(Protocol):
    """Port for reading simulation template data."""

    def read_simulation(self, path: str) -> dict[str, Any]:
        """Read and parse a simulation file."""
        ...

    def extract_template_entity(self, sim_data: dict, entity_name: str) -> dict:
        """Extract a named entity to use as a template."""
        ...

    def extract_earth_entity(self, sim_data: dict) -> dict:
        """Extract the Earth entity."""
        ...


@runtime_checkable
class SimulationWriter(Protocol):
    """Port for writing simulation output data."""

    def write_simulation(self, sim_data: dict, path: str) -> None:
        """Write simulation data to output file."""
        ...
