from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_cat_assets import build_sprite
from remove_chroma_key import estimate_key_color, remove_chroma_key


ROOT = Path(__file__).resolve().parents[2]


class RemoveChromaKeyTests(unittest.TestCase):
    def test_background_edge_and_foreground_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source.png"
            output = temporary / "output.png"
            image = Image.new("RGBA", (7, 7), (20, 245, 22, 255))
            image.putpixel((3, 3), (252, 250, 244, 255))
            image.putpixel((3, 2), (5, 5, 7, 255))
            image.putpixel((2, 3), (30, 145, 30, 255))
            image.save(source)

            key = remove_chroma_key(source, output)
            result = Image.open(output).convert("RGBA")

        self.assertEqual((20, 245, 22), key)
        self.assertEqual((0, 0, 0, 0), result.getpixel((0, 0)))
        self.assertEqual((252, 250, 244, 255), result.getpixel((3, 3)))
        self.assertEqual((5, 5, 7, 255), result.getpixel((3, 2)))
        edge = result.getpixel((2, 3))
        self.assertGreater(edge[3], 0)
        self.assertLess(edge[3], 255)
        self.assertLessEqual(edge[1], max(edge[0], edge[2]))

    def test_real_master_is_deterministic_and_has_clean_background(self) -> None:
        source = ROOT / "assets" / "pet" / "source" / "cat-master-chroma.png"
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            first_path = temporary / "first.png"
            second_path = temporary / "second.png"
            first_key = remove_chroma_key(source, first_path)
            second_key = remove_chroma_key(source, second_path)
            first_bytes = first_path.read_bytes()
            second_bytes = second_path.read_bytes()
            first = Image.open(first_path).convert("RGBA")
            checked_master = Image.open(
                ROOT / "assets" / "pet" / "runtime" / "cat-master-transparent.png"
            ).convert("RGBA")
            sprite_path = temporary / "cat-48.png"
            metadata = build_sprite(first_path, sprite_path)
            sprite = Image.open(sprite_path).convert("RGBA")
            checked_sprite = Image.open(
                ROOT / "assets" / "pet" / "runtime" / "cat-48.png"
            ).convert("RGBA")

        self.assertEqual(first_key, second_key)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first.tobytes(), checked_master.tobytes())
        self.assertEqual(sprite.tobytes(), checked_sprite.tobytes())
        self.assertEqual([24, 46], metadata["footAnchor"])
        self.assertEqual((1254, 1254), first.size)
        self.assertEqual((0, 0, 0, 0), first.getpixel((0, 0)))
        self.assertEqual((0, 255), first.getchannel("A").getextrema())
        self.assertIsNotNone(first.getchannel("A").getbbox())
        self.assertGreater(first_key[1], first_key[0] + 150)
        self.assertGreater(first_key[1], first_key[2] + 150)

    def test_invalid_feather_range_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.png"
            Image.new("RGB", (1, 1), (0, 255, 0)).save(source)
            with self.assertRaises(ValueError):
                remove_chroma_key(source, source.with_name("output.png"), transparent_distance=40, opaque_distance=40)

    def test_border_estimator_handles_single_pixel_image(self) -> None:
        image = Image.new("RGB", (1, 1), (12, 230, 14))
        self.assertEqual((12, 230, 14), estimate_key_color(image))


if __name__ == "__main__":
    unittest.main()
