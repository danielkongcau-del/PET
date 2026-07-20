"""Repository-friendly entry point for the generator subprocess."""

from __future__ import annotations

from pet_generator.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
