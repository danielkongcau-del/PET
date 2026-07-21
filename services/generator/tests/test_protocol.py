from __future__ import annotations

import io
import json
import unittest

from pet_generator.backend import MotionPoint
from pet_generator.protocol import ProtocolError, ProtocolWriter, decode_line

from tests.helpers import hello, line, world_state


class ProtocolTests(unittest.TestCase):
    @staticmethod
    def _motion_point(**pose) -> MotionPoint:
        return MotionPoint(
            t_ms=0,
            dx=0.0,
            dy=0.0,
            vx=0.0,
            vy=0.0,
            facing=1,
            lean=0.0,
            squash=1.0,
            bob=0.0,
            expression="neutral",
            **pose,
        )

    def test_motion_point_producer_requires_one_atomic_pose_branch(self) -> None:
        identity = (0.0, 0.0, 0.0, 1.0)
        valid_3d = self._motion_point(
            root_translation=(0.0, 0.0, 0.0),
            root_rotation=identity,
            local_rotation_deltas=(identity,),
        )
        self.assertEqual(
            set(valid_3d.to_payload()).intersection({
                "bone_rotations", "root_translation", "root_rotation", "local_rotation_deltas",
            }),
            {"root_translation", "root_rotation", "local_rotation_deltas"},
        )
        self.assertIn("bone_rotations", self._motion_point(bone_rotations=(0.0,)).to_payload())

        for partial in (
            {"root_translation": (0.0, 0.0, 0.0)},
            {"root_rotation": identity},
            {"local_rotation_deltas": (identity,)},
            {"root_translation": (0.0, 0.0, 0.0), "root_rotation": identity},
        ):
            with self.subTest(partial=sorted(partial)), self.assertRaisesRegex(
                ValueError, "must be all-or-none",
            ):
                self._motion_point(**partial)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self._motion_point(
                bone_rotations=(0.0,),
                root_translation=(0.0, 0.0, 0.0),
                root_rotation=identity,
                local_rotation_deltas=(identity,),
            )

    def test_motion_point_serialization_rechecks_atomic_pose(self) -> None:
        point = self._motion_point()
        object.__setattr__(point, "root_translation", (0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "must be all-or-none"):
            point.to_payload()

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
