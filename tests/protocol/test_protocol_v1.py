from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ModuleNotFoundError:  # pet-core keeps runtime dependencies intentionally small
    Draft202012Validator = None  # type: ignore[assignment,misc]

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PYTHON = ROOT / "packages" / "protocol" / "python"
if str(PROTOCOL_PYTHON) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_PYTHON))

from pet_protocol import (  # noqa: E402
    ProtocolValidationError,
    decode_ndjson_line,
    encode_ndjson,
    validate_message,
)

SCHEMA_PATH = ROOT / "packages" / "protocol" / "schemas" / "v1" / "pet-motion.schema.json"
FIXTURE_PATH = ROOT / "packages" / "protocol" / "fixtures" / "v1" / "session.ndjson"


def fixture_messages() -> list[dict]:
    return [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


class ProtocolV1ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        if Draft202012Validator is not None:
            Draft202012Validator.check_schema(cls.schema)
            cls.schema_validator = Draft202012Validator(cls.schema)
        else:
            cls.schema_validator = None
        cls.messages = fixture_messages()

    def test_fixture_covers_every_message_type(self) -> None:
        expected = {
            "hello",
            "ready",
            "world_state",
            "horizon_plan",
            "cancel",
            "ping",
            "pong",
            "metrics",
            "error",
        }
        self.assertEqual(expected, {message["type"] for message in self.messages})

    def test_fixture_is_valid_against_json_schema_and_python_mirror(self) -> None:
        for message in self.messages:
            with self.subTest(message_type=message["type"]):
                if self.schema_validator is not None:
                    errors = sorted(self.schema_validator.iter_errors(message), key=lambda error: list(error.path))
                    self.assertEqual([], errors, "\n".join(error.message for error in errors))
                self.assertIs(message, validate_message(message))

    def test_python_codec_round_trips_fixture(self) -> None:
        for message in self.messages:
            with self.subTest(message_type=message["type"]):
                encoded = encode_ndjson(message)
                self.assertTrue(encoded.endswith("\n"))
                self.assertEqual(message, decode_ndjson_line(encoded))

    def test_fixture_sequences_are_monotonic_per_sender(self) -> None:
        host_types = {"hello", "ping", "world_state", "cancel"}
        host_seq = [message["seq"] for message in self.messages if message["type"] in host_types]
        generator_seq = [message["seq"] for message in self.messages if message["type"] not in host_types]
        self.assertEqual(host_seq, sorted(set(host_seq)))
        self.assertEqual(generator_seq, sorted(set(generator_seq)))

    def test_rejects_unknown_field_and_version(self) -> None:
        for mutation in ("extra", "version"):
            message = copy.deepcopy(self.messages[0])
            if mutation == "extra":
                message["unexpected"] = True
            else:
                message["version"] = 2
            with self.subTest(mutation=mutation):
                if self.schema_validator is not None:
                    self.assertFalse(self.schema_validator.is_valid(message))
                with self.assertRaises(ProtocolValidationError):
                    validate_message(message)

    def test_schema_declares_strict_v1_discriminator(self) -> None:
        self.assertEqual("https://json-schema.org/draft/2020-12/schema", self.schema["$schema"])
        self.assertFalse(self.schema["additionalProperties"])
        self.assertEqual("pet-motion", self.schema["properties"]["protocol"]["const"])
        self.assertEqual(1, self.schema["properties"]["version"]["const"])
        self.assertEqual(
            {message["type"] for message in self.messages},
            set(self.schema["properties"]["type"]["enum"]),
        )

    def test_rejects_invalid_plan_timing_and_non_finite_values(self) -> None:
        plan = copy.deepcopy(next(message for message in self.messages if message["type"] == "horizon_plan"))
        plan["payload"]["points"][1]["t_ms"] = 34
        with self.assertRaisesRegex(ProtocolValidationError, "spacing"):
            validate_message(plan)

        invalid_json = json.dumps(self.messages[0]).replace('"world_state_hz": 20', '"world_state_hz": NaN')
        with self.assertRaises(ProtocolValidationError):
            decode_ndjson_line(invalid_json)

    def test_deep_json_and_huge_numbers_are_validation_errors(self) -> None:
        with self.assertRaises(ProtocolValidationError):
            decode_ndjson_line("[" * 1_200 + "0" + "]" * 1_200)

        world = copy.deepcopy(next(message for message in self.messages if message["type"] == "world_state"))
        world["payload"]["pet"]["x"] = 10**1_000
        with self.assertRaises(ProtocolValidationError):
            validate_message(world)

    def test_rejects_unsafe_integer_motion_bounds_and_multiple_records(self) -> None:
        hello = copy.deepcopy(self.messages[0])
        hello["seq"] = 2**53
        if self.schema_validator is not None:
            self.assertFalse(self.schema_validator.is_valid(hello))
        with self.assertRaises(ProtocolValidationError):
            validate_message(hello)

        plan = copy.deepcopy(next(message for message in self.messages if message["type"] == "horizon_plan"))
        plan["payload"]["points"][0]["vx"] = 20_001
        if self.schema_validator is not None:
            self.assertFalse(self.schema_validator.is_valid(plan))
        with self.assertRaises(ProtocolValidationError):
            validate_message(plan)

        with self.assertRaisesRegex(ProtocolValidationError, "exactly one"):
            decode_ndjson_line(encode_ndjson(self.messages[0]) + "\n")

    def test_plan_offsets_are_relative_to_world_foot_anchor(self) -> None:
        world = next(message for message in self.messages if message["type"] == "world_state")
        plan = next(message for message in self.messages if message["type"] == "horizon_plan")
        self.assertEqual(world["seq"], plan["payload"]["based_on_seq"])
        self.assertEqual(0, plan["payload"]["points"][0]["t_ms"])
        self.assertEqual(0, plan["payload"]["points"][0]["dx"])
        self.assertEqual(0, plan["payload"]["points"][0]["dy"])


if __name__ == "__main__":
    unittest.main()
