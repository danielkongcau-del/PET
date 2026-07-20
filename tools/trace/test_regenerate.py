from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.trace.regenerate import (
    canonical_json,
    canonical_sha256,
    main,
    regenerate,
)

from pet_generator.planner import AutoregressiveMotionBackend
from pet_generator.protocol import Envelope
from pet_generator.state import parse_world_state


def world_payload(seq: int, *, seed: int | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": "anonymous-session",
        "coordinate_space": "physical_px",
        "displays": [
            {
                "id": "display-0",
                "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
                "work_area": {"x": 0, "y": 0, "width": 1920, "height": 1040},
                "scale_factor": 1.25,
            }
        ],
        "surfaces": [
            {
                "id": "surface-0",
                "kind": "window_top",
                "display_id": "display-0",
                "window_id": "window-0",
                "x1": 120,
                "x2": 620,
                "y": 420,
                "enabled": True,
                "occluded": False,
            },
            {
                "id": "floor-0",
                "kind": "work_area_floor",
                "display_id": "display-0",
                "window_id": None,
                "x1": 0,
                "x2": 1920,
                "y": 1040,
                "enabled": True,
                "occluded": False,
            },
        ],
        "pet": {
            "x": 192,
            "y": 324,
            "width": 96,
            "height": 96,
            "foot_x": 240,
            "foot_y": 420,
            "vx": 0,
            "vy": 0,
            "facing": 1,
            "behavior": "idle",
            "visible": True,
            "user_dragging": False,
            "surface_id": "surface-0",
        },
        "cursor": None,
        "clicks": [],
        "scene": {"fullscreen_active": False, "pet_allowed": True},
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


def parsed_world(seq: int, timestamp_ms: int, *, seed: int | None = None):
    return parse_world_state(
        Envelope(
            protocol="pet-motion",
            version=1,
            type="world_state",
            seq=seq,
            timestamp_ms=timestamp_ms,
            payload=world_payload(seq, seed=seed),
        )
    )


def trace_event(record_seq: int, kind: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "pet-trace",
        "version": 1,
        "record_seq": record_seq,
        "wall_time_ms": 1_750_000_000_000 + record_seq * 10,
        "elapsed_us": record_seq * 10_000,
        "kind": kind,
        "payload": payload,
    }


def write_episode(
    root: Path,
    records: list[dict[str, object]],
    *,
    manifest: dict[str, object] | None = None,
) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest or {"schema": "pet-trace-manifest", "version": 1}),
        encoding="utf-8",
    )
    with gzip.open(root / "trace-0001.ndjson.gz", "wt", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


class CanonicalJsonTests(unittest.TestCase):
    def test_integer_floats_match_javascript_number_spelling(self) -> None:
        self.assertEqual(canonical_json({"zero": 0.0, "one": 1.0}), '{"one":1,"zero":0}')
        self.assertEqual(canonical_sha256({"x": 0}), canonical_sha256({"x": 0.0}))


class RegenerationTests(unittest.TestCase):
    def test_trace_plans_regenerate_exactly_and_cancel_state_is_replayed(self) -> None:
        timestamp1 = 1_750_000_001_000
        timestamp2 = timestamp1 + 50
        generated1 = 1_750_000_002_000
        generated2 = generated1 + 50
        backend = AutoregressiveMotionBackend()
        first = backend.generate(parsed_world(1, timestamp1), 101, generated1)
        self.assertTrue(backend.cancel(first.plan_id))
        second = backend.generate(parsed_world(3, timestamp2), 102, generated2)
        first_recorded = first.to_payload()
        first_recorded["plan_id"] = "plan-0"
        second_recorded = second.to_payload()
        second_recorded["plan_id"] = "plan-1"
        records = [
            trace_event(
                0,
                "world_state",
                {"seq": 1, "timestamp_ms": timestamp1, "state": world_payload(1)},
            ),
            trace_event(1, "plan_received", {"plan": first_recorded, "received_at_ms": generated1 + 2}),
            trace_event(2, "cancel", {"plan_id": "plan-0", "reason": "surface_changed"}),
            # This unplanned state represents a latest-state inbox drop and must
            # not advance the backend during plan-driven regeneration.
            trace_event(
                3,
                "world_state",
                {"seq": 2, "timestamp_ms": timestamp2 - 1, "state": world_payload(2)},
            ),
            trace_event(
                4,
                "world_state",
                {"seq": 3, "timestamp_ms": timestamp2, "state": world_payload(3)},
            ),
            trace_event(5, "plan_received", {"plan": second_recorded, "received_at_ms": generated2 + 2}),
        ]
        manifest = {
            "schema": "pet-trace-manifest",
            "version": 1,
            "metadata": {
                "determinism": {
                    "backend": "autoregressive-v0",
                    "session_seed": 777,
                    "timing": {"world_state_hz": 20, "plan_horizon_ms": 396, "plan_dt_ms": 33},
                }
            },
        }

        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            write_episode(episode, records, manifest=manifest)
            first_output = io.StringIO()
            first_summary = regenerate(episode, first_output)
            second_output = io.StringIO()
            second_summary = regenerate(episode, second_output)

        self.assertEqual(first_output.getvalue(), second_output.getvalue())
        self.assertEqual(first_summary.result_sha256, second_summary.result_sha256)
        self.assertTrue(first_summary.deterministic)
        self.assertTrue(first_summary.exact_match)
        self.assertEqual(first_summary.plans_compared, 2)
        self.assertEqual(first_summary.cancellations_applied, 1)
        self.assertEqual(first_summary.worlds_without_plan, 1)
        lines = [json.loads(line) for line in first_output.getvalue().splitlines()]
        self.assertEqual([line["type"] for line in lines], ["metadata", "plan", "plan", "summary"])
        self.assertEqual(lines[-1]["summary"]["result_sha256"], first_summary.result_sha256)

    def test_compare_only_detects_difference_without_embedding_plan(self) -> None:
        timestamp = 1_750_000_001_000
        backend = AutoregressiveMotionBackend()
        recorded = backend.generate(parsed_world(1, timestamp), 9, timestamp + 1).to_payload()
        recorded["behavior"] = "intentionally-different"
        records = [
            trace_event(0, "world_state", {"seq": 1, "timestamp_ms": timestamp, "state": world_payload(1)}),
            trace_event(1, "plan_received", {"plan": recorded, "received_at_ms": timestamp + 2}),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            write_episode(episode, records)
            output = io.StringIO()
            summary = regenerate(episode, output, compare_only=True)

        plan_line = json.loads(output.getvalue().splitlines()[1])
        self.assertNotIn("plan", plan_line)
        self.assertFalse(plan_line["comparison"]["exact_match"])
        self.assertFalse(summary.exact_match)
        self.assertEqual(summary.mismatches, 1)

    def test_raw_protocol_world_state_uses_requested_seed_and_virtual_time(self) -> None:
        timestamp = 1_750_000_003_000
        message = {
            "protocol": "pet-motion",
            "version": 1,
            "type": "world_state",
            "seq": 4,
            "timestamp_ms": timestamp,
            "payload": world_payload(4, seed=88),
        }
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "worlds.jsonl"
            source.write_text(json.dumps(message) + "\n", encoding="utf-8")
            output = io.StringIO()
            summary = regenerate(source, output)

        plan_line = json.loads(output.getvalue().splitlines()[1])
        self.assertEqual(plan_line["seed_source"], "world_state")
        self.assertEqual(plan_line["generated_at_source"], "trace_virtual_clock")
        self.assertEqual(plan_line["plan"]["generated_at_ms"], timestamp)
        self.assertEqual(summary.worlds_regenerated, 1)
        self.assertIsNone(summary.exact_match)

    def test_missing_checkpoint_is_explicitly_non_deterministic(self) -> None:
        timestamp = 1_750_000_001_000
        backend = AutoregressiveMotionBackend()
        recorded = backend.generate(parsed_world(1, timestamp), 4, timestamp + 1).to_payload()
        records = [
            trace_event(0, "world_state", {"seq": 1, "timestamp_ms": timestamp, "state": world_payload(1)}),
            trace_event(1, "plan_received", {"plan": recorded, "received_at_ms": timestamp + 2}),
        ]
        manifest = {
            "metadata": {
                "determinism": {
                    "checkpoint": {"path": "private-user-model.ckpt", "sha256": "a" * 64}
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            write_episode(episode, records, manifest=manifest)
            output = io.StringIO()
            summary = regenerate(episode, output)

        metadata = json.loads(output.getvalue().splitlines()[0])
        self.assertFalse(summary.deterministic)
        self.assertIn("checkpoint_missing", summary.nondeterministic_reasons)
        self.assertEqual(metadata["checkpoint"]["status"], "checkpoint_missing")
        self.assertNotIn("private-user-model", output.getvalue())

    def test_checkpoint_hash_mismatch_reports_hashes_but_not_path(self) -> None:
        timestamp = 1_750_000_001_000
        payload = world_payload(1, seed=5)
        records = [
            trace_event(0, "world_state", {"seq": 1, "timestamp_ms": timestamp, "state": payload}),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            checkpoint = episode / "secret-name.ckpt"
            checkpoint.write_bytes(b"model bytes")
            manifest = {
                "metadata": {
                    "determinism": {
                        "checkpoint": {"path": checkpoint.name, "sha256": "0" * 64}
                    }
                }
            }
            write_episode(episode, records, manifest=manifest)
            output = io.StringIO()
            summary = regenerate(episode, output)

        self.assertFalse(summary.deterministic)
        self.assertIn("checkpoint_sha256_mismatch", output.getvalue())
        self.assertIn(hashlib.sha256(b"model bytes").hexdigest(), output.getvalue())
        self.assertNotIn("secret-name", output.getvalue())

    def test_verified_checkpoint_is_not_silently_ignored_by_procedural_backend(self) -> None:
        timestamp = 1_750_000_001_000
        records = [
            trace_event(
                0,
                "world_state",
                {"seq": 1, "timestamp_ms": timestamp, "state": world_payload(1, seed=6)},
            ),
        ]
        checkpoint_bytes = b"future learned model"
        expected = hashlib.sha256(checkpoint_bytes).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            checkpoint = episode / "model.ckpt"
            checkpoint.write_bytes(checkpoint_bytes)
            manifest = {
                "metadata": {
                    "determinism": {
                        "checkpoint": {"relative_path": checkpoint.name, "sha256": expected}
                    }
                }
            }
            write_episode(episode, records, manifest=manifest)
            output = io.StringIO()
            summary = regenerate(episode, output)

        self.assertFalse(summary.deterministic)
        self.assertIn("checkpoint_not_loaded_by_backend", summary.nondeterministic_reasons)
        self.assertIn('"status":"verified"', output.getvalue())

    def test_cli_error_does_not_echo_input_path(self) -> None:
        private_path = "C:/Users/Private Person/not-present/trace.ndjson.gz"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main([private_path])
        self.assertEqual(status, 2)
        self.assertNotIn("Private Person", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
