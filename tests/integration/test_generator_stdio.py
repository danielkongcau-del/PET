from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_DIR = ROOT / "services" / "generator"
GENERATOR_RUN = GENERATOR_DIR / "run.py"
PROTOCOL_PYTHON = ROOT / "packages" / "protocol" / "python"
if str(PROTOCOL_PYTHON) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_PYTHON))

from pet_protocol import encode_ndjson, make_message, validate_message  # noqa: E402


def now_ms() -> int:
    return int(time.time() * 1000)


def hello_message(seq: int = 0) -> dict[str, Any]:
    return make_message(
        "hello",
        seq,
        {
            "session_id": "e2e-session",
            "host": {"name": "protocol-e2e", "version": "0.1.0", "pid": os.getpid()},
            "requested_version": 1,
            "capabilities": ["world_state_v1", "cancel_v1", "metrics_v1"],
            "config": {
                "world_state_hz": 20,
                "plan_horizon_ms": 400,
                "plan_dt_ms": 33,
                "pet_width": 96,
                "pet_height": 96,
                "privacy": {
                    "screen_capture_enabled": False,
                    "keyboard_enabled": False,
                    "recording_enabled": False,
                },
            },
        },
    )


def world_state_message(seq: int, *, clicked: bool, seed: int) -> dict[str, Any]:
    timestamp = now_ms()
    clicks = []
    if clicked:
        clicks.append(
            {
                "id": f"click-{seq}",
                "button": "left",
                "x": 448,
                "y": 170,
                "target": "pet",
                "timestamp_ms": timestamp,
            }
        )
    return make_message(
        "world_state",
        seq,
        {
            "session_id": "e2e-session",
            "coordinate_space": "physical_px",
            "displays": [
                {
                    "id": "display-1",
                    "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
                    "work_area": {"x": 0, "y": 0, "width": 1920, "height": 1040},
                    "scale_factor": 1.25,
                    "is_primary": True,
                }
            ],
            "windows": [
                {
                    "id": "window-current",
                    "display_id": "display-1",
                    "bounds": {"x": 300, "y": 220, "width": 500, "height": 600},
                    "z_order": 0,
                    "visible": True,
                    "minimized": False,
                    "maximized": False,
                    "fullscreen": False,
                    "active": True,
                    "occluded": False,
                    "eligible": True,
                },
                {
                    "id": "window-nearby",
                    "display_id": "display-1",
                    "bounds": {"x": 900, "y": 330, "width": 480, "height": 420},
                    "z_order": 1,
                    "visible": True,
                    "minimized": False,
                    "maximized": False,
                    "fullscreen": False,
                    "active": False,
                    "occluded": False,
                    "eligible": True,
                },
            ],
            "surfaces": [
                {
                    "id": "surface-current",
                    "kind": "window_top",
                    "display_id": "display-1",
                    "window_id": "window-current",
                    "x1": 300,
                    "x2": 800,
                    "y": 220,
                    "enabled": True,
                    "occluded": False,
                },
                {
                    "id": "surface-nearby",
                    "kind": "window_top",
                    "display_id": "display-1",
                    "window_id": "window-nearby",
                    "x1": 900,
                    "x2": 1380,
                    "y": 330,
                    "enabled": True,
                    "occluded": False,
                },
                {
                    "id": "surface-floor",
                    "kind": "work_area_floor",
                    "display_id": "display-1",
                    "x1": 0,
                    "x2": 1920,
                    "y": 1040,
                    "enabled": True,
                    "occluded": False,
                },
            ],
            "pet": {
                "x": 352,
                "y": 124,
                "width": 96,
                "height": 96,
                "foot_x": 400,
                "foot_y": 220,
                "vx": 0,
                "vy": 0,
                "facing": 1,
                "behavior": "idle",
                "visible": True,
                "user_dragging": False,
                "surface_id": "surface-current",
            },
            "cursor": {
                "x": 448,
                "y": 170,
                "left_down": clicked,
                "right_down": False,
                "middle_down": False,
                "over_pet": clicked,
            },
            "clicks": clicks,
            "scene": {"fullscreen_active": False, "pet_allowed": True},
            "seed": seed,
        },
        timestamp_ms=timestamp,
    )


class GeneratorProcess:
    def __init__(self) -> None:
        python = os.environ.get("PET_GENERATOR_PYTHON", sys.executable)
        env = os.environ.copy()
        python_paths = [str(PROTOCOL_PYTHON), str(GENERATOR_DIR)]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        self.process = subprocess.Popen(
            [python, "-u", str(GENERATOR_RUN)],
            cwd=GENERATOR_DIR,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.messages: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue()
        self.stderr_lines: list[str] = []
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                if line.strip():
                    self.messages.put(validate_message(json.loads(line)))
        except BaseException as exc:  # surfaced on the test thread
            self.messages.put(exc)
        finally:
            self.messages.put(None)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_lines.append(line.rstrip())

    def send(self, message: dict[str, Any]) -> None:
        if self.process.poll() is not None:
            raise AssertionError(f"generator exited {self.process.returncode}: {' | '.join(self.stderr_lines[-10:])}")
        assert self.process.stdin is not None
        self.process.stdin.write(encode_ndjson(message))
        self.process.stdin.flush()

    def send_raw(self, line: str) -> None:
        if self.process.poll() is not None:
            raise AssertionError(f"generator exited {self.process.returncode}: {' | '.join(self.stderr_lines[-10:])}")
        assert self.process.stdin is not None
        self.process.stdin.write(line.rstrip("\r\n") + "\n")
        self.process.stdin.flush()

    def receive(self, expected_type: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        seen: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"timed out waiting for {expected_type}; seen={seen}; stderr={' | '.join(self.stderr_lines[-10:])}"
                )
            try:
                item = self.messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise AssertionError(f"timed out waiting for {expected_type}; seen={seen}") from exc
            if item is None:
                raise AssertionError(
                    f"generator stdout closed while waiting for {expected_type}; stderr={' | '.join(self.stderr_lines[-10:])}"
                )
            if isinstance(item, BaseException):
                raise AssertionError(f"invalid generator stdout: {item}") from item
            seen.append(item["type"])
            if item["type"] == expected_type:
                return item
            if item["type"] == "error":
                raise AssertionError(f"generator returned error: {item['payload']}")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()


@unittest.skipUnless(GENERATOR_RUN.is_file(), "services/generator/run.py is not ready")
class GeneratorStdioEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = GeneratorProcess()

    def tearDown(self) -> None:
        self.generator.close()

    def test_handshake_ping_plan_click_and_cancel(self) -> None:
        started = time.monotonic()
        self.generator.send(hello_message())
        ready = self.generator.receive("ready", timeout=8)
        self.assertEqual("e2e-session", ready["payload"]["session_id"])
        self.assertEqual(1, ready["payload"]["accepted_version"])
        self.assertLess(time.monotonic() - started, 8)

        sent_at = now_ms()
        self.generator.send(make_message("ping", 1, {"nonce": "e2e-ping", "sent_at_ms": sent_at}))
        pong = self.generator.receive("pong", timeout=1)
        self.assertEqual("e2e-ping", pong["payload"]["nonce"])
        self.assertEqual(sent_at, pong["payload"]["ping_sent_at_ms"])

        self.generator.send(world_state_message(2, clicked=False, seed=7001))
        plan = self.generator.receive("horizon_plan", timeout=3)
        self.assertEqual(2, plan["payload"]["based_on_seq"])
        self.assertGreater(plan["payload"]["valid_until_ms"], now_ms())
        self.assertGreater(len(plan["payload"]["points"]), 1)
        self.assertEqual(0, plan["payload"]["points"][0]["t_ms"])

        self.generator.send(world_state_message(3, clicked=True, seed=7002))
        reaction = self.generator.receive("horizon_plan", timeout=3)
        self.assertEqual(3, reaction["payload"]["based_on_seq"])
        self.assertEqual("click_reaction", reaction["payload"]["behavior"])

        self.generator.send(
            make_message(
                "cancel",
                4,
                {
                    "plan_id": reaction["payload"]["plan_id"],
                    "based_on_seq": 3,
                    "reason": "topology_change",
                    "requested_at_ms": now_ms(),
                },
            )
        )
        sent_at = now_ms()
        self.generator.send(make_message("ping", 5, {"nonce": "after-cancel", "sent_at_ms": sent_at}))
        pong = self.generator.receive("pong", timeout=1)
        self.assertEqual("after-cancel", pong["payload"]["nonce"])

    def test_malformed_control_returns_error_and_service_recovers(self) -> None:
        self.generator.send(hello_message())
        self.generator.receive("ready", timeout=8)

        invalid = {
            "protocol": "pet-motion",
            "version": 1,
            "type": "ping",
            "seq": 1,
            "timestamp_ms": now_ms(),
            "payload": {"nonce": 42, "sent_at_ms": now_ms()},
        }
        self.generator.send_raw(json.dumps(invalid, separators=(",", ":")))
        error = self.generator.receive("error", timeout=1)
        self.assertTrue(error["payload"]["recoverable"])
        self.assertEqual(1, error["payload"]["related_seq"])

        sent_at = now_ms()
        self.generator.send(make_message("ping", 2, {"nonce": "recovered", "sent_at_ms": sent_at}))
        pong = self.generator.receive("pong", timeout=1)
        self.assertEqual("recovered", pong["payload"]["nonce"])

    def test_deep_json_returns_error_and_service_recovers(self) -> None:
        self.generator.send(hello_message())
        self.generator.receive("ready", timeout=8)

        self.generator.send_raw("[" * 1_200 + "0" + "]" * 1_200)
        error = self.generator.receive("error", timeout=1)
        self.assertEqual("invalid_json", error["payload"]["code"])
        self.assertTrue(error["payload"]["recoverable"])

        sent_at = now_ms()
        self.generator.send(make_message("ping", 2, {"nonce": "after-deep", "sent_at_ms": sent_at}))
        pong = self.generator.receive("pong", timeout=1)
        self.assertEqual("after-deep", pong["payload"]["nonce"])


if __name__ == "__main__":
    unittest.main()
