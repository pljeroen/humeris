# Copyright (c) 2026 Jeroen Visser. All rights reserved.
# Licensed under the terms in LICENSE-COMMERCIAL.md.
# Free for personal, educational, and academic use.
# Commercial use requires a paid license — see LICENSE-COMMERCIAL.md.
"""Tests for link budget computation."""
import ast
import math
from datetime import datetime, timezone

import pytest

from constellation_generator.domain.link_budget import (
    LinkBudgetResult,
    LinkConfig,
    compute_isl_link_budgets,
    compute_link_budget,
    free_space_path_loss_db,
)


class TestFreeSpacePathLoss:
    """Tests for FSPL formula."""

    def test_fspl_known_value(self):
        """1 km at 1 GHz ≈ 92.45 dB."""
        fspl = free_space_path_loss_db(1000.0, 1e9)
        assert abs(fspl - 92.45) < 0.1

    def test_fspl_double_distance_plus_6db(self):
        """Doubling distance adds ~6.02 dB."""
        f1 = free_space_path_loss_db(1000.0, 1e9)
        f2 = free_space_path_loss_db(2000.0, 1e9)
        assert abs((f2 - f1) - 6.02) < 0.05

    def test_fspl_double_frequency_plus_6db(self):
        """Doubling frequency adds ~6.02 dB."""
        f1 = free_space_path_loss_db(1000.0, 1e9)
        f2 = free_space_path_loss_db(1000.0, 2e9)
        assert abs((f2 - f1) - 6.02) < 0.05

    def test_fspl_positive(self):
        """FSPL is always positive for valid inputs."""
        assert free_space_path_loss_db(100.0, 1e6) > 0


class TestComputeLinkBudget:
    """Tests for full link budget computation."""

    def _isl_config(self):
        return LinkConfig(
            frequency_hz=26e9,          # Ka-band
            transmit_power_w=10.0,      # 10W
            tx_antenna_gain_dbi=35.0,
            rx_antenna_gain_dbi=35.0,
            system_noise_temp_k=500.0,
            bandwidth_hz=100e6,
            additional_losses_db=2.0,
            required_snr_db=10.0,
        )

    def test_link_budget_returns_type(self):
        """Return type is LinkBudgetResult."""
        config = self._isl_config()
        result = compute_link_budget(config, 1000e3)
        assert isinstance(result, LinkBudgetResult)

    def test_link_margin_positive(self):
        """Reasonable ISL config at 1000 km → positive margin."""
        config = self._isl_config()
        result = compute_link_budget(config, 1000e3)
        assert result.link_margin_db > 0

    def test_link_margin_decreases_with_distance(self):
        """Farther distance → worse link margin."""
        config = self._isl_config()
        r1 = compute_link_budget(config, 1000e3)
        r2 = compute_link_budget(config, 3000e3)
        assert r2.link_margin_db < r1.link_margin_db

    def test_max_data_rate_positive(self):
        """Shannon capacity > 0."""
        config = self._isl_config()
        result = compute_link_budget(config, 1000e3)
        assert result.max_data_rate_bps > 0

    def test_noise_floor_kTB(self):
        """Noise floor matches k·T·B."""
        k = 1.380649e-23
        T = 500.0
        B = 100e6
        expected = 10.0 * math.log10(k * T * B)
        config = self._isl_config()
        result = compute_link_budget(config, 1000e3)
        assert abs(result.noise_floor_dbw - expected) < 0.01


class TestISLLinkBudgets:
    """Integration tests for ISL link budget computation."""

    def test_isl_budgets_returns_list(self):
        """Returns list of (ISLLink, LinkBudgetResult) tuples."""
        from constellation_generator.domain.propagation import OrbitalState

        config = LinkConfig(
            frequency_hz=26e9,
            transmit_power_w=2.0,
            tx_antenna_gain_dbi=30.0,
            rx_antenna_gain_dbi=30.0,
            system_noise_temp_k=500.0,
            bandwidth_hz=250e6,
        )
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        mu = 3.986004418e14
        a = 6_371_000.0 + 500_000.0
        n = math.sqrt(mu / a**3)
        states = [
            OrbitalState(
                semi_major_axis_m=a, eccentricity=0.0,
                inclination_rad=math.radians(53.0), raan_rad=0.0,
                arg_perigee_rad=0.0, true_anomaly_rad=math.radians(i * 10.0),
                mean_motion_rad_s=n, reference_epoch=epoch,
            )
            for i in range(3)
        ]
        result = compute_isl_link_budgets(config, states, epoch)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_isl_budgets_blocked_excluded(self):
        """Blocked links are not included in the link budget results."""
        from constellation_generator.domain.propagation import OrbitalState

        config = LinkConfig(
            frequency_hz=26e9,
            transmit_power_w=2.0,
            tx_antenna_gain_dbi=30.0,
            rx_antenna_gain_dbi=30.0,
            system_noise_temp_k=500.0,
            bandwidth_hz=250e6,
        )
        epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        mu = 3.986004418e14
        a = 6_371_000.0 + 500_000.0
        n = math.sqrt(mu / a**3)
        # Two sats on opposite sides of Earth
        states = [
            OrbitalState(
                semi_major_axis_m=a, eccentricity=0.0,
                inclination_rad=math.radians(53.0), raan_rad=0.0,
                arg_perigee_rad=0.0, true_anomaly_rad=0.0,
                mean_motion_rad_s=n, reference_epoch=epoch,
            ),
            OrbitalState(
                semi_major_axis_m=a, eccentricity=0.0,
                inclination_rad=math.radians(53.0), raan_rad=0.0,
                arg_perigee_rad=0.0, true_anomaly_rad=math.pi,
                mean_motion_rad_s=n, reference_epoch=epoch,
            ),
        ]
        result = compute_isl_link_budgets(config, states, epoch)
        # Link is blocked by Earth → should be empty
        assert len(result) == 0


class TestLinkBudgetPurity:
    """Domain purity: link_budget.py must only import stdlib + domain."""

    def test_module_pure(self):
        import constellation_generator.domain.link_budget as mod

        allowed = {'math', 'numpy', 'dataclasses', 'typing', 'abc', 'enum', '__future__', 'datetime'}
        with open(mod.__file__) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in allowed and not root.startswith('constellation_generator'):
                        assert False, f"Disallowed import '{alias.name}'"
            if isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    root = node.module.split('.')[0]
                    if root not in allowed and root != 'constellation_generator':
                        assert False, f"Disallowed import from '{node.module}'"
