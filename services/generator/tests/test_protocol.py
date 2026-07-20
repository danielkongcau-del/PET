from __future__ import annotations

import io
import json
import unittest

from pet_generator.protocol import ProtocolError, ProtocolWriter, decode_line

from tests.helpers import hello, line, world_state


class ProtocolTests(unittest.TestCase):
    def test_decode_valid_envelope(self) -> None:
        decoded = decode_line(line(hello()))
        self.assertEqual(decoded.type, "hello")
        self.assertEqual(decoded.seq, 0)

    def test_rejects_malformed_json_without_echoing_input(self) -> None:
        with self.assertRaises(ProtocolError) as caught:
            decode_line("{secret contents")
        self.assertEqual(caught.exception.code, "invalid_json")
        self.assertNotIn("secret", str(caught.exception))

    def test_rejects_deep_json_and_huge_numbers_as_protocol_errors(self) -> None:
        with self.assertRaises(ProtocolError) as deep:
            decode_line("[" * 1_200 + "0" + "]" * 1_200)
        self.assertEqual(deep.exception.code, "invalid_json")

        message = world_state()
        message["payload"]["pet"]["x"] = 10**1_000
        with self.assertRaises(ProtocolError) as huge:
            decode_line(line(message))
        self.assertEqual(huge.exception.code, "invalid_message")

    def test_rejects_unknown_message_type(self) -> None:
        message = hello()
        message["type"] = "shutdown"
        with self.assertRaises(ProtocolError) as caught:
            decode_line(line(message))
        self.assertEqual(caught.exception.code, "unsupported_message_type")

    def test_rejects_additional_properties_via_canonical_validator(self) -> None:
        message = hello()
        message["payload"]["unexpected"] = "not echoed"
        with self.assertRaises(ProtocolError) as caught:
            decode_line(line(message))
        self.assertEqual(caught.exception.code, "invalid_message")
        self.assertNotIn("not echoed", str(caught.exception))

    def test_writer_emits_exactly_one_compact_json_line(self) -> None:
        output = io.StringIO()
        writer = ProtocolWriter(output)
        writer.send("pong", {"nonce": "n", "ping_sent_at_ms": 1, "received_at_ms": 2})
        self.assertEqual(output.getvalue().count("\n"), 1)
        parsed = json.loads(output.getvalue())
        self.assertEqual(parsed["protocol"], "pet-motion")
        self.assertEqual(parsed["version"], 1)
        self.assertEqual(parsed["seq"], 0)


if __name__ == "__main__":
    unittest.main()
