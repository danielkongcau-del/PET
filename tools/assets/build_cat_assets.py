"""Build the deterministic 48x48 runtime sprite from a cat master image.

The normal input is the transparent concept master produced by
``remove_chroma_key.py``. Opaque near-white screenshot inputs are also accepted:
a border flood fill removes only their exterior, preserving white pixels
enclosed by the cat's black outline. The result is reduced to a deliberately
tiny palette and resized with nearest-neighbour sampling so Electron can scale
it 2x without blur.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from PIL import Image


CANVAS_SIZE = 48
MAX_SPRITE_WIDTH = 44
MAX_SPRITE_HEIGHT = 40


def _looks_like_exterior(pixel: tuple[int, int, int, int]) -> bool:
    red, green, blue, alpha = pixel
    return alpha > 0 and min(red, green, blue) >= 210 and max(red, green, blue) - min(red, green, blue) <= 12


def _exterior_mask(image: Image.Image) -> set[tuple[int, int]]:
    width, height = image.size
    queue: deque[tuple[int, int]] = deque()
    exterior: set[tuple[int, int]] = set()

    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))

    pixels = image.load()
    while queue:
        x, y = queue.popleft()
        if (x, y) in exterior or not (0 <= x < width and 0 <= y < height):
            continue
        if not _looks_like_exterior(pixels[x, y]):
            continue
        exterior.add((x, y))
        queue.extend(
            (x + offset_x, y + offset_y)
            for offset_y in (-1, 0, 1)
            for offset_x in (-1, 0, 1)
            if offset_x != 0 or offset_y != 0
        )
    return exterior


def build_sprite(source: Path, output: Path) -> dict[str, object]:
    image = Image.open(source).convert("RGBA")
    alpha_min, _ = image.getchannel("A").getextrema()
    exterior = set() if alpha_min < 255 else _exterior_mask(image)
    pixels = image.load()

    for y in range(image.height):
        for x in range(image.width):
            red, green, blue, alpha = pixels[x, y]
            if (x, y) in exterior or alpha < 32:
                pixels[x, y] = (0, 0, 0, 0)
                continue
            luminance = (red * 299 + green * 587 + blue * 114) // 1000
            if red - green > 16 and red > 190 and blue > 120:
                pixels[x, y] = (247, 174, 190, 255)
            else:
                pixels[x, y] = (5, 5, 7, 255) if luminance < 170 else (252, 250, 244, 255)

    alpha_box = image.getchannel("A").getbbox()
    if alpha_box is None:
        raise ValueError(f"No foreground pixels found in {source}")

    cropped = image.crop(alpha_box)
    scale = min(MAX_SPRITE_WIDTH / cropped.width, MAX_SPRITE_HEIGHT / cropped.height)
    sprite_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    sprite = cropped.resize(sprite_size, Image.Resampling.NEAREST)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    offset_x = (CANVAS_SIZE - sprite.width) // 2
    offset_y = CANVAS_SIZE - sprite.height - 2
    canvas.alpha_composite(sprite, (offset_x, offset_y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return {
        "canvas": [CANVAS_SIZE, CANVAS_SIZE],
        "contentBounds": [offset_x, offset_y, sprite.width, sprite.height],
        "footAnchor": [CANVAS_SIZE // 2, offset_y + sprite.height],
        # The checked-in master looks toward negative X (head on the left).
        # Keeping this in metadata prevents renderer-facing semantics from
        # depending on an implicit assumption about the current artwork.
        "sourceFacing": -1,
        "displayScale": 2,
        "parts": {
            # Legacy coarse clips retained for a future semantic-layer asset.
            # They must not be independently transformed from this flat PNG.
            "head": {"clip": [0, 0, 20, 48], "pivot": [16, 29]},
            "body": {"clip": [20, 0, 16, 35], "pivot": [28, 32]},
            "tail": {"clip": [36, 0, 12, 48], "pivot": [37, 30]},
            "frontLeg": {"clip": [28, 35, 8, 13], "pivot": [32, 35]},
            "rearLeg": {"clip": [20, 35, 8, 13], "pivot": [24, 35]},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    metadata = build_sprite(args.source, args.output)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
