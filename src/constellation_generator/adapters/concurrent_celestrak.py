"""
Concurrent CelesTrak adapter: parallelizes SGP4 propagation.

Uses ThreadPoolExecutor from stdlib to propagate OMM records concurrently.
HTTP fetching remains sequential (one request per call) to respect
CelesTrak rate limiting. Only SGP4 propagation is parallelized.

External dependencies (urllib, json, concurrent.futures) are confined
to this adapter layer. sgp4 is imported lazily via SGP4Adapter.
"""
import json
import logging
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from urllib.parse import quote

from constellation_generator.adapters.celestrak import SGP4Adapter, BASE_URL
from constellation_generator.ports.orbital_data import OrbitalDataSource
from constellation_generator.domain.constellation import Satellite


_log = logging.getLogger(__name__)


class ConcurrentCelesTrakAdapter(OrbitalDataSource):
    """
    Fetches live orbital data from CelesTrak with concurrent SGP4 propagation.

    HTTP requests are sequential (one per call). SGP4 propagation of
    returned records is parallelized using ThreadPoolExecutor.

    Args:
        max_workers: Thread pool size for SGP4 propagation.
            Default: min(32, os.cpu_count() + 4) — same as Python default.
        base_url: CelesTrak GP API URL.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        base_url: str = BASE_URL,
        timeout: int = 30,
    ):
        self._max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self._base_url = base_url
        self._timeout = timeout
        self._sgp4 = SGP4Adapter()

    def fetch_group(self, group_name: str) -> list[dict[str, Any]]:
        url = f"{self._base_url}?GROUP={group_name}&FORMAT=JSON"
        return self._fetch_json(url)

    def fetch_by_name(self, name: str) -> list[dict[str, Any]]:
        encoded_name = quote(name)
        url = f"{self._base_url}?NAME={encoded_name}&FORMAT=JSON"
        return self._fetch_json(url)

    def fetch_by_catnr(self, catalog_number: int) -> list[dict[str, Any]]:
        url = f"{self._base_url}?CATNR={catalog_number}&FORMAT=JSON"
        return self._fetch_json(url)

    def fetch_satellites(
        self,
        group: str | None = None,
        name: str | None = None,
        catnr: int | None = None,
        epoch: datetime | None = None,
    ) -> list[Satellite]:
        """
        Fetch OMM data and convert to Satellite objects via concurrent SGP4.

        HTTP fetch is a single sequential request. SGP4 propagation of
        the returned records is parallelized across threads.

        Args:
            group: CelesTrak group name.
            name: Satellite name search.
            catnr: NORAD catalog number.
            epoch: Optional datetime to propagate all satellites to.

        Returns:
            List of Satellite domain objects.
        """
        if group:
            records = self.fetch_group(group)
        elif name:
            records = self.fetch_by_name(name)
        elif catnr:
            records = self.fetch_by_catnr(catnr)
        else:
            raise ValueError("Specify one of: group, name, or catnr")

        return self._propagate_concurrent(records, epoch)

    def _propagate_concurrent(
        self,
        records: list[dict[str, Any]],
        epoch: datetime | None,
    ) -> list[Satellite]:
        """Propagate OMM records to Satellite objects using thread pool."""
        satellites: list[Satellite] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    self._sgp4.omm_to_satellite, record, epoch_override=epoch
                ): record
                for record in records
            }

            for future in as_completed(futures):
                record = futures[future]
                try:
                    sat = future.result()
                    satellites.append(sat)
                except (RuntimeError, ValueError, KeyError) as e:
                    _log.warning(
                        "Skipping %s: %s", record.get('OBJECT_NAME', '?'), e
                    )
                    continue

        return satellites

    def _fetch_json(self, url: str) -> list[dict[str, Any]]:
        """Fetch JSON data from CelesTrak API (sequential)."""
        req = urllib.request.Request(
            url, headers={"User-Agent": "ConstellationGenerator/1.4"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                text = response.read().decode("utf-8")
                if text.strip() == "No GP data found":
                    return []
                return json.loads(text)
        except urllib.error.HTTPError as e:
            raise ConnectionError(
                f"CelesTrak API error {e.code}: {e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"CelesTrak connection failed: {e.reason}"
            ) from e
