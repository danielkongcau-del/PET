from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from pet_generator.cli import NEURAL_BACKEND, PROCEDURAL_BACKEND, _build_backend
from pet_generator.planner import AutoregressiveMotionBackend, PlannerConfig


class GeneratorBackendSelectionTests(unittest.TestCase):
    def test_procedural_backend_is_explicit_and_checkpoint_free(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            backend = _build_backend(PROCEDURAL_BACKEND, PlannerConfig())
        self.assertIsInstance(backend, AutoregressiveMotionBackend)

    def test_checkpoint_is_never_silently_ignored_by_procedural_backend(self) -> None:
        with patch.dict(
            os.environ,
            {"PET_GENERATOR_CHECKPOINT": "checkpoints/character/motion.pt"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "refusing to silently ignore"):
                _build_backend(PROCEDURAL_BACKEND, PlannerConfig())

    def test_unimplemented_neural_backend_fails_before_ready(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "no checkpoint was loaded"):
                _build_backend(NEURAL_BACKEND, PlannerConfig())


if __name__ == "__main__":
    unittest.main()
