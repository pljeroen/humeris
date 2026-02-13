# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the MIT License — see LICENSE.
"""
CelesTrak adapter: fetches live orbital data and converts to domain objects.

External dependencies (urllib, json, sgp4) are confined to this layer.

Data sources:
    CelesTrak GP API — https://celestrak.org/NORAD/elements/gp.php
    Groups: STATIONS, GPS-OPS, STARLINK, ONEWEB, ACTIVE, WEATHER, etc.

SGP4 propagation:
    TLE mean elements are SGP4-specific, NOT pure Keplerian.
    Direct Kepler→Cartesian would give wrong answers for real satellites.
    The sgp4 library provides proper TEME state vectors at epoch.
"""
import json
import logging
import math
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from humeris.ports.orbital_data import OrbitalDataSource
from humeris.domain.constellation import Satellite
from humeris.domain.omm import parse_omm_record


_log = logging.getLogger(__name__)

BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"


def _require_sgp4():
    """Import sgp4 lazily; raise clear error if not installed."""
    try:
        from sgp4.api import Satrec, WGS72, jday
    except ImportError:
        raise ImportError(
            "sgp4 is required for live orbital data. "
            "Install with: pip install humeris[live]"
        ) from None
    return Satrec, WGS72, jday


def _normalize_epoch(epoch_str: str) -> str:
    """Normalize ISO epoch string: replace trailing 'Z' with '+00:00'."""
    if epoch_str.endswith("Z"):
        return epoch_str[:-1] + "+00:00"
    return epoch_str


def _as_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (treat naive as UTC)."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class SGP4Adapter:
    """Converts OMM records to Satellite domain objects using SGP4 propagation."""

    def omm_to_satellite(
        self,
        omm_record: dict[str, Any],
        epoch_override: datetime | None = None,
    ) -> Satellite:
        """
        Convert an OMM record to a Satellite using SGP4 propagation.

        Args:
            omm_record: CelesTrak OMM JSON dict.
            epoch_override: Optional datetime to propagate to (default: TLE epoch).

        Returns:
            Satellite with ECI position (m) and velocity (m/s).
        """
        Satrec, WGS72, jday = _require_sgp4()
        elements = parse_omm_record(omm_record)

        sat = Satrec()
        sat.sgp4init(
            WGS72,
            'i',
            elements.norad_cat_id,
            _epoch_to_jd_offset(elements.epoch),
            elements.bstar,
            elements.mean_motion_dot / (2.0 * math.pi / (86400.0**2)),
            elements.mean_motion_ddot,
            elements.eccentricity,
            math.radians(elements.arg_perigee_deg),
            math.radians(elements.inclination_deg),
            math.radians(elements.mean_anomaly_deg),
            elements.mean_motion_rev_per_day * 2.0 * math.pi / 1440.0,
            math.radians(elements.raan_deg),
        )

        if epoch_override:
            jd, fr = _datetime_to_jd(jday, epoch_override)
            propagation_epoch = epoch_override
        else:
            jd, fr = _epoch_str_to_jd(jday, elements.epoch)
            propagation_epoch = datetime.fromisoformat(
                _normalize_epoch(elements.epoch)
            )

        error_code, position_km, velocity_km_s = sat.sgp4(jd, fr)
        if error_code != 0:
            raise RuntimeError(
                f"SGP4 propagation error {error_code} for {elements.object_name}"
            )

        pos_m = (position_km[0] * 1000, position_km[1] * 1000, position_km[2] * 1000)
        vel_ms = (velocity_km_s[0] * 1000, velocity_km_s[1] * 1000, velocity_km_s[2] * 1000)

        return Satellite(
            name=elements.object_name,
            position_eci=pos_m,
            velocity_eci=vel_ms,
            plane_index=0,
            sat_index=elements.norad_cat_id,
            raan_deg=elements.raan_deg,
            true_anomaly_deg=elements.mean_anomaly_deg,
            epoch=propagation_epoch,
        )


class CelesTrakAdapter(OrbitalDataSource):
    """
    Fetches live orbital data from CelesTrak's GP API.

    Available groups (non-exhaustive):
        STATIONS, GPS-OPS, STARLINK, ONEWEB, ACTIVE, WEATHER,
        RESOURCE, SCIENCE, NOAA, GOES, AMATEUR, GALILEO,
        BEIDOU, IRIDIUM, IRIDIUM-NEXT, GLOBALSTAR, ORBCOMM,
        PLANET, SPIRE, GEO, INTELSAT, SES, TELESAT

    Rate limiting: CelesTrak updates at most every 2 hours.
    """

    def __init__(self, base_url: str = BASE_URL, timeout: int = 30):
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
        Fetch OMM data and convert to Satellite objects via SGP4.

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

        satellites = []
        for record in records:
            try:
                sat = self._sgp4.omm_to_satellite(record, epoch_override=epoch)
                satellites.append(sat)
            except (RuntimeError, ValueError, KeyError) as e:
                _log.warning("Skipping %s: %s", record.get('OBJECT_NAME', '?'), e)
                continue

        return satellites

    def _fetch_json(self, url: str) -> list[dict[str, Any]]:
        """Fetch JSON data from CelesTrak API."""
        req = urllib.request.Request(url, headers={"User-Agent": "ConstellationGenerator/1.4"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                text = response.read().decode('utf-8')
                if text.strip() == "No GP data found":
                    return []
                return json.loads(text)
        except urllib.error.HTTPError as e:
            raise ConnectionError(f"CelesTrak API error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"CelesTrak connection failed: {e.reason}") from e


def _epoch_str_to_jd(jday_fn, epoch_str: str) -> tuple[float, float]:
    dt = _as_utc(datetime.fromisoformat(_normalize_epoch(epoch_str)))
    return _datetime_to_jd(jday_fn, dt)


def _datetime_to_jd(jday_fn, dt: datetime) -> tuple[float, float]:
    return jday_fn(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                   dt.second + dt.microsecond / 1e6)


def _epoch_to_jd_offset(epoch_str: str) -> float:
    """SGP4 epoch offset: fractional days since 1949-12-31."""
    dt = _as_utc(datetime.fromisoformat(_normalize_epoch(epoch_str)))
    ref = datetime(1949, 12, 31, tzinfo=timezone.utc)
    delta = dt - ref
    return delta.days + delta.seconds / 86400.0 + delta.microseconds / 86400e6
