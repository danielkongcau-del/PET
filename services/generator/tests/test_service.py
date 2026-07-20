from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "packages" / "protocol" / "python"))

from pet_protocol import validate_message  # noqa: E402
from pet_generator.planner import AutoregressiveMotionBackend
from pet_generator.protocol import decode_line
from pet_generator.service import GeneratorService, ServiceMetrics, _LatestStateInbox

from tests.helpers import envelope, hello, line, world_state


class _ExplodingInput:
    def __init__(self, error: BaseException):
        self.error = error

    def __iter__(self):
        raise self.error


class ServiceTests(unittest.TestCase):
    def test_latest_state_queue_replaces_state_and_preserves_click_edges(self) -> None:
        metrics = ServiceMetrics()
        inbox = _LatestStateInbox(metrics)
        first = decode_line(line(world_state(seq=1, click_id="click:old")))
        ping = decode_line(line(envelope("ping", 2, {"nonce": "queued", "sent_at_ms": 1})))
        latest = decode_line(line(world_state(seq=3, click_id="click:new")))
        inbox.put(first)
        inbox.put(ping)
        inbox.put(latest)
        replaced = inbox.get()
        self.assertEqual(replaced.seq, 3)
        self.assertEqual([click["id"] for click in replaced.payload["clicks"]], ["click:old", "click:new"])
        self.assertEqual(inbox.get().type, "ping")
        self.assertEqual(metrics.world_states_dropped, 1)

    def test_latest_state_does_not_cross_hello_or_session_boundary(self) -> None:
        metrics = ServiceMetrics()
        inbox = _LatestStateInbox(metrics)
        state_a = decode_line(line(world_state(seq=1, session_id="session-a")))
        hello_b = decode_line(line(hello(seq=2, session_id="session-b")))
        state_b = decode_line(line(world_state(seq=3, session_id="session-b")))

        inbox.put(state_a)
        inbox.put(hello_b)
        inbox.put(state_b)

        self.assertEqual(inbox.get().payload["session_id"], "session-a")
        self.assertEqual(inbox.get().type, "hello")
        self.assertEqual(inbox.get().payload["session_id"], "session-b")
        self.assertEqual(metrics.world_states_dropped, 0)

    def test_burst_session_handoff_keeps_new_session_state(self) -> None:
        input_stream = io.StringIO(
            line(hello(seq=0, session_id="session-a"))
            + line(world_state(seq=1, session_id="session-a"))
            + line(hello(seq=2, session_id="session-b"))
            + line(world_state(seq=3, session_id="session-b"))
        )
        output_stream = io.StringIO()

        GeneratorService(AutoregressiveMotionBackend(), session_seed=3, metrics_interval_ms=0).run(
            input_stream, output_stream
        )

        messages = [json.loads(item) for item in output_stream.getvalue().splitlines()]
        self.assertEqual(
            [item["type"] for item in messages],
            ["ready", "horizon_plan", "ready", "horizon_plan"],
        )
        self.assertEqual(messages[-1]["payload"]["based_on_seq"], 3)

    def test_in_process_handshake_plan_and_pong(self) -> None:
        ping = envelope("ping", 2, {"nonce": "probe:1", "sent_at_ms": 1_750_000_001_100})
        input_stream = io.StringIO(line(hello()) + line(world_state()) + line(ping))
        output_stream = io.StringIO()
        service = GeneratorService(AutoregressiveMotionBackend(), session_seed=123, metrics_interval_ms=0)
        self.assertEqual(service.run(input_stream, output_stream), 0)
        messages = [json.loads(item) for item in output_stream.getvalue().splitlines()]
        for message in messages:
            validate_message(message)
        self.assertEqual([item["type"] for item in messages], ["ready", "horizon_plan", "pong"])
        self.assertEqual(messages[1]["payload"]["based_on_seq"], 1)
        self.assertEqual(messages[2]["payload"]["nonce"], "probe:1")

    def test_hello_timing_is_applied_to_plans(self) -> None:
        hello_message = hello()
        hello_message["payload"]["config"]["plan_horizon_ms"] = 200
        hello_message["payload"]["config"]["plan_dt_ms"] = 20
        output_stream = io.StringIO()
        GeneratorService(AutoregressiveMotionBackend(), session_seed=5, metrics_interval_ms=0).run(
            io.StringIO(line(hello_message) + line(world_state())), output_stream
        )
        messages = [json.loads(item) for item in output_stream.getvalue().splitlines()]
        plan = messages[1]["payload"]
        self.assertEqual(plan["dt_ms"], 20)
        self.assertEqual(len(plan["points"]), 10)
        self.assertEqual(plan["points"][-1]["t_ms"], 180)

    def test_generator_emits_schema_valid_metrics(self) -> None:
        output_stream = io.StringIO()
        service = GeneratorService(AutoregressiveMotionBackend(), session_seed=5, metrics_interval_ms=1)
        service._last_metrics_monotonic -= 1.0
        service.run(io.StringIO(line(hello()) + line(world_state())), output_stream)
        messages = [json.loads(item) for item in output_stream.getvalue().splitlines()]
        self.assertEqual([item["type"] for item in messages], ["ready", "horizon_plan", "metrics"])
        validate_message(messages[-1])
        self.assertEqual(messages[-1]["payload"]["source"], "generator")

    def test_bad_input_is_recoverable(self) -> None:
        input_stream = io.StringIO("not-json\n" + line(hello()))
        output_stream = io.StringIO()
        GeneratorService(AutoregressiveMotionBackend(), session_seed=1, metrics_interval_ms=0).run(
            input_stream, output_stream
        )
        messages = [json.loads(item) for item in output_stream.getvalue().splitlines()]
        self.assertEqual(messages[0]["type"], "error")
        self.assertTrue(messages[0]["payload"]["recoverable"])
        self.assertEqual(messages[1]["type"], "ready")

    def test_deep_json_is_recoverable(self) -> None:
        deep_json = "[" * 1_200 + "0" + "]" * 1_200 + "\n"
        ping = envelope("ping", 2, {"nonce": "after-deep", "sent_at_ms": 2})
        output_stream = io.StringIO()

        GeneratorService(AutoregressiveMotionBackend(), session_seed=1, metrics_interval_ms=0).run(
            io.StringIO(line(hello()) + deep_json + line(ping)), output_stream
        )

        messages = [json.loads(item) for item in output_stream.getvalue().splitlines()]
        self.assertEqual([item["type"] for item in messages], ["ready", "error", "pong"])
        self.assertEqual(messages[1]["payload"]["code"], "invalid_json")

    def test_reader_fallback_does_not_swallow_fatal_exceptions(self) -> None:
        service = GeneratorService(AutoregressiveMotionBackend(), session_seed=1, metrics_interval_ms=0)
        inbox = _LatestStateInbox(service.metrics)
        with self.assertLogs("pet.generator", level="ERROR"):
            service._read_input(_ExplodingInput(RuntimeError("decoder bug")), inbox)
        recovered = inbox.get()
        self.assertEqual(recovered.error.code, "stdin_error")
        self.assertIsNone(inbox.get())

        for fatal in (MemoryError("out of memory"), SystemExit(7)):
            with self.subTest(error=type(fatal).__name__):
                fatal_inbox = _LatestStateInbox(service.metrics)
                with self.assertRaises(type(fatal)):
                    service._read_input(_ExplodingInput(fatal), fatal_inbox)
                self.assertIsNone(fatal_inbox.get())

    def test_malformed_ping_does_not_poison_following_ping(self) -> None:
        malformed = envelope("ping", 1, {"nonce": "bad", "sent_at_ms": 1, "unexpected": True})
        valid = envelope("ping", 2, {"nonce": "good", "sent_at_ms": 2})
        output_stream = io.StringIO()
        GeneratorService(AutoregressiveMotionBackend(), session_seed=1, metrics_interval_ms=0).run(
            io.StringIO(line(hello()) + line(malformed) + line(valid)), output_stream
        )
        messages = [json.loads(item) for item in output_stream.getvalue().splitlines()]
        self.assertEqual([item["type"] for item in messages], ["ready", "error", "pong"])
        self.assertEqual(messages[-1]["payload"]["nonce"], "good")

    def test_subprocess_stdout_contains_protocol_only(self) -> None:
        generator_root = Path(__file__).resolve().parents[1]
        ping = envelope("ping", 2, {"nonce": "subprocess", "sent_at_ms": 1_750_000_001_100})
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                str(generator_root / "run.py"),
                "--seed",
                "99",
                "--metrics-interval-ms",
                "0",
                "--log-level",
                "INFO",
            ],
            cwd=generator_root,
            input=line(hello()) + line(world_state()) + line(ping),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        messages = [json.loads(item) for item in process.stdout.splitlines()]
        for message in messages:
            validate_message(message)
        self.assertEqual([item["type"] for item in messages], ["ready", "horizon_plan", "pong"])
        self.assertNotIn("generator started", process.stdout)
        self.assertIn("generator started", process.stderr)


if __name__ == "__main__":
    unittest.main()
