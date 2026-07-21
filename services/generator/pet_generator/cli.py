"""Command-line entry point for the PET generator."""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import sys
from typing import Sequence

from . import __version__
from .planner import AutoregressiveMotionBackend, PlannerConfig
from .service import GeneratorService


PROCEDURAL_BACKEND = "procedural"
NEURAL_BACKEND = "neural"


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PET realtime motion generator over stdio NDJSON")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=None,
        help="deterministic session seed (also PET_GENERATOR_SEED)",
    )
    parser.add_argument("--horizon-steps", type=int, default=12)
    parser.add_argument("--dt-ms", type=int, default=33)
    parser.add_argument(
        "--backend",
        choices=(PROCEDURAL_BACKEND, NEURAL_BACKEND),
        default=os.environ.get("PET_GENERATOR_BACKEND", PROCEDURAL_BACKEND),
        help="runtime backend (also PET_GENERATOR_BACKEND)",
    )
    parser.add_argument(
        "--metrics-interval-ms",
        type=_non_negative_int,
        default=5_000,
        help="0 disables periodic generator metrics",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=os.environ.get("PET_GENERATOR_LOG_LEVEL", "INFO").upper(),
    )
    return parser


def _resolve_seed(cli_seed: int | None) -> int:
    if cli_seed is not None:
        return cli_seed
    env_seed = os.environ.get("PET_GENERATOR_SEED")
    if env_seed is not None:
        return _non_negative_int(env_seed)
    return secrets.randbits(64)


def _build_backend(kind: str, config: PlannerConfig):
    checkpoint = os.environ.get("PET_GENERATOR_CHECKPOINT")
    if kind == PROCEDURAL_BACKEND:
        if checkpoint:
            raise ValueError(
                "PET_GENERATOR_CHECKPOINT is set but the procedural backend cannot "
                "load checkpoints; refusing to silently ignore it"
            )
        return AutoregressiveMotionBackend(config)
    if kind == NEURAL_BACKEND:
        raise ValueError(
            "the neural backend is not implemented yet; no checkpoint was loaded"
        )
    raise ValueError(f"unsupported generator backend: {kind!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        config = PlannerConfig(horizon_steps=args.horizon_steps, dt_ms=args.dt_ms)
        seed = _resolve_seed(args.seed)
        backend = _build_backend(args.backend, config)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    # Explicit UTF-8 and LF make the pipe identical under Windows and test hosts.
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict", newline="\n")

    service = GeneratorService(
        backend,
        session_seed=seed,
        metrics_interval_ms=args.metrics_interval_ms,
    )
    return service.run(sys.stdin, sys.stdout)
