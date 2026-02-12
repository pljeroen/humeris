# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
Port interface for external orbital data sources.

Adapters handle the actual HTTP/API calls.
"""
from abc import ABC, abstractmethod
from typing import Any


class OrbitalDataSource(ABC):
    """Port for fetching orbital element data from external sources."""

    @abstractmethod
    def fetch_group(self, group_name: str) -> list[dict[str, Any]]:
        """Fetch OMM records for a named satellite group."""
        ...

    @abstractmethod
    def fetch_by_name(self, name: str) -> list[dict[str, Any]]:
        """Fetch OMM records matching a satellite name."""
        ...

    @abstractmethod
    def fetch_by_catnr(self, catalog_number: int) -> list[dict[str, Any]]:
        """Fetch OMM record for a NORAD catalog number."""
        ...

    @abstractmethod
    def fetch_satellites(self, group: str | None = None,
                        name: str | None = None,
                        catnr: int | None = None) -> list:
        """Fetch and convert to Satellite domain objects."""
        ...
