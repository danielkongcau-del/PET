from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_cat_assets import build_sprite


ROOT = Path(__file__).resolve().parents[2]


class CatAssetTests(unittest.TestCase):
    def test_runtime_sprite_contract(self) -> None:
        source = ROOT / "assets" / "pet" / "runtime" / "cat-master-transparent.png"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "cat-48.png"
            metadata = build_sprite(source, output)
            image = Image.open(output).convert("RGBA")

        self.assertEqual((48, 48), image.size)
        self.assertEqual((0, 0, 0, 0), image.getpixel((0, 0)))
        self.assertEqual([24, 46], metadata["footAnchor"])
        self.assertEqual(-1, metadata["sourceFacing"])
        self.assertLessEqual(len(set(image.getdata())), 4)
        self.assertIsNotNone(image.getchannel("A").getbbox())

        covered: set[tuple[int, int]] = set()
        for part in metadata["parts"].values():
            x, y, width, height = part["clip"]
            pixels = {
                (column, row)
                for row in range(y, y + height)
                for column in range(x, x + width)
            }
            self.assertTrue(covered.isdisjoint(pixels))
            covered.update(pixels)
        opaque = {
            (column, row)
            for row in range(image.height)
            for column in range(image.width)
            if image.getpixel((column, row))[3] > 0
        }
        self.assertTrue(opaque.issubset(covered))

    def test_checked_in_metadata_matches_builder(self) -> None:
        source = ROOT / "assets" / "pet" / "runtime" / "cat-master-transparent.png"
        checked_in = json.loads(
            (ROOT / "assets" / "pet" / "runtime" / "cat-parts.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = build_sprite(source, Path(temporary_directory) / "cat-48.png")

        self.assertEqual(checked_in, generated)


if __name__ == "__main__":
    unittest.main()
