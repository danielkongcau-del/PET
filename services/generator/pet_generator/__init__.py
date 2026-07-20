"""Realtime motion generator for PET.

The package intentionally has no mandatory third-party dependencies.  A learned
PyTorch backend can implement :class:`pet_generator.backend.MotionBackend`
without changing the stdio protocol or the desktop host.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
