"""Batch training data generation using the simulator with teacher backend.

Usage:
  python -m pet_generator.data_gen --episodes 1000 --output data/train
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from .planner import AutoregressiveMotionBackend, PlannerConfig
from .simulator import DesktopSimulator, ScenarioConfig, TrainingSample


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate PET training data via teacher simulation")
    p.add_argument("--episodes", type=int, default=1000, help="Number of episodes to generate")
    p.add_argument("--duration-ms", type=int, default=30_000, help="Episode duration in ms")
    p.add_argument("--output", type=str, default="data/train", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument("--horizon-steps", type=int, default=12)
    p.add_argument("--dt-ms", type=int, default=33)
    return p


def save_samples(samples: list[TrainingSample], output_dir: Path, episode_id: int) -> int:
    """Save one episode's samples as NDJSON. Returns number of samples written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"episode-{episode_id:05d}.ndjson"
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({
                "condition": s.condition_frames,
                "target": s.target_poses,
                "metadata": s.metadata,
            }, ensure_ascii=False) + "\n")
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ScenarioConfig(duration_ms=args.duration_ms)
    planner_config = PlannerConfig(horizon_steps=args.horizon_steps, dt_ms=args.dt_ms)
    backend = AutoregressiveMotionBackend(planner_config)
    sim = DesktopSimulator()

    total_samples = 0
    start_time = time.monotonic()

    for ep in range(args.episodes):
        episode_seed = args.seed + ep
        samples = sim.generate_episode(backend, config, episode_seed)
        n = save_samples(samples, output_dir, ep)
        total_samples += n

        if (ep + 1) % 100 == 0:
            elapsed = time.monotonic() - start_time
            rate = total_samples / max(elapsed, 0.001)
            print(f"[{ep + 1}/{args.episodes}] {total_samples} samples, {rate:.0f} samples/s")

    elapsed = time.monotonic() - start_time
    print(f"Done: {total_samples} samples in {elapsed:.1f}s ({total_samples / max(elapsed, 0.001):.0f} samples/s)")
    print(f"Output: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
