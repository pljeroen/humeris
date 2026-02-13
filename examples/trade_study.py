#!/usr/bin/env python3
"""Trade study example: Pareto optimization + conjunction screening.

Sweeps constellation parameters, finds the Pareto-optimal designs,
exports the best one to CSV, and screens for close approaches.

Usage:
    python examples/trade_study.py
"""
from datetime import datetime, timedelta, timezone

from humeris import (
    ShellConfig,
    derive_orbital_state,
    generate_walker_configs,
    generate_walker_shell,
    pareto_front_indices,
    run_walker_trade_study,
    screen_conjunctions,
)
from humeris.adapters.csv_exporter import CsvSatelliteExporter


def main():
    epoch = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

    # --- Step 1: Sweep the design space ---
    print("Generating configurations...")
    configs = generate_walker_configs(
        altitude_range=(550.0,),
        inclination_range=(53.0,),
        planes_range=(3, 4, 6, 8, 10),
        sats_per_plane_range=(4, 6, 8, 10, 12),
    )
    print(f"  {len(configs)} configurations to evaluate")

    # --- Step 2: Evaluate coverage for each ---
    print("Running trade study (this may take a minute)...")
    result = run_walker_trade_study(
        configs, epoch, timedelta(hours=12), timedelta(seconds=60),
        min_elevation_deg=10, lat_step_deg=15, lon_step_deg=15,
        lat_range=(-60.0, 60.0),  # focus on coverage band
    )

    # --- Step 3: Find the Pareto front ---
    # Use mean revisit (more discriminating than max for sparse grids)
    costs = [pt.total_satellites for pt in result.points]
    revisit = [pt.coverage.mean_revisit_s / 60 for pt in result.points]
    front_idx = pareto_front_indices(costs, revisit)

    print(f"\nPareto front ({len(front_idx)} optimal designs):")
    print(f"  {'Config':<20} {'Sats':>5} {'Mean Revisit':>13} {'Coverage':>9}")
    print(f"  {'-'*20} {'-'*5} {'-'*13} {'-'*9}")
    for i in front_idx:
        pt = result.points[i]
        c = pt.config
        label = f"{c.num_planes}x{c.sats_per_plane} @ {c.altitude_km:.0f}km"
        cov = pt.coverage.mean_coverage_fraction * 100
        print(f"  {label:<20} {pt.total_satellites:>5} {revisit[i]:>10.1f} min {cov:>7.1f}%")

    # --- Step 4: Export the best design ---
    best = result.points[front_idx[-1]]  # highest sat count = best revisit
    shell = ShellConfig(
        altitude_km=best.config.altitude_km,
        inclination_deg=best.config.inclination_deg,
        num_planes=best.config.num_planes,
        sats_per_plane=best.config.sats_per_plane,
        phase_factor=best.config.phase_factor,
        raan_offset_deg=0,
        shell_name="Optimized",
    )
    sats = generate_walker_shell(shell)
    CsvSatelliteExporter().export(sats, "examples/optimized.csv")
    print(f"\nExported {len(sats)} satellites to examples/optimized.csv")

    # --- Step 5: Conjunction screening ---
    print("Screening for close approaches...")
    states = [derive_orbital_state(s, epoch) for s in sats]
    names = [s.name for s in sats]
    events = screen_conjunctions(
        states, names, epoch, timedelta(hours=2),
        timedelta(seconds=30), distance_threshold_m=5000
    )
    print(f"  {len(events)} close approaches within 5 km over 2 hours")
    for idx_a, idx_b, t, dist in events[:5]:
        print(f"  {names[idx_a]} - {names[idx_b]}: {dist:.0f} m at {t}")


if __name__ == "__main__":
    main()
