"""Deterministically remove a generated green-screen background with Pillow.

The image generator's nominally flat chroma background contains small colour
variations. The key colour is therefore estimated from the median RGB value of
the outer border. Pixels near that colour become transparent; pixels farther
away transition linearly to opaque so anti-aliased outline pixels retain a
soft alpha edge. Green spill is removed from partially transparent pixels.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Iterable

from PIL import Image


RGB = tuple[int, int, int]
DEFAULT_TRANSPARENT_DISTANCE = 32.0
DEFAULT_OPAQUE_DISTANCE = 235.0


def _border_pixels(image: Image.Image) -> Iterable[RGB]:
    pixels = image.load()
    width, height = image.size
    for x in range(width):
        yield pixels[x, 0][:3]
        if height > 1:
            yield pixels[x, height - 1][:3]
    for y in range(1, max(1, height - 1)):
        yield pixels[0, y][:3]
        if width > 1:
            yield pixels[width - 1, y][:3]


def estimate_key_color(image: Image.Image) -> RGB:
    """Return the per-channel median RGB colour around the image border."""

    rgba = image.convert("RGBA")
    border = list(_border_pixels(rgba))
    if not border:
        raise ValueError("Cannot estimate a chroma key from an empty image")
    return tuple(round(statistics.median(pixel[channel] for pixel in border)) for channel in range(3))  # type: ignore[return-value]


def remove_chroma_key(
    source: Path,
    output: Path,
    *,
    key_color: RGB | None = None,
    transparent_distance: float = DEFAULT_TRANSPARENT_DISTANCE,
    opaque_distance: float = DEFAULT_OPAQUE_DISTANCE,
) -> RGB:
    """Write a transparent RGBA image and return the RGB key colour used."""

    if transparent_distance < 0:
        raise ValueError("transparent_distance must be non-negative")
    if opaque_distance <= transparent_distance:
        raise ValueError("opaque_distance must be greater than transparent_distance")

    image = Image.open(source).convert("RGBA")
    resolved_key = key_color or estimate_key_color(image)
    span = opaque_distance - transparent_distance
    output_pixels: list[tuple[int, int, int, int]] = []

    for red, green, blue, source_alpha in image.getdata():
        distance = math.sqrt(
            (red - resolved_key[0]) ** 2 +
            (green - resolved_key[1]) ** 2 +
            (blue - resolved_key[2]) ** 2
        )
        coverage = min(1.0, max(0.0, (distance - transparent_distance) / span))
        alpha = round(source_alpha * coverage)
        if alpha <= 0:
            output_pixels.append((0, 0, 0, 0))
            continue
        if alpha < source_alpha:
            # A neutral edge avoids a green fringe when composited on desktop
            # colours. Keep red/blue intact and cap green at their maximum.
            green = min(green, max(red, blue))
        output_pixels.append((red, green, blue, alpha))

    result = Image.new("RGBA", image.size)
    result.putdata(output_pixels)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output, format="PNG", optimize=True)
    return resolved_key


def _parse_rgb(value: str) -> RGB:
    normalized = value.strip().removeprefix("#")
    if len(normalized) != 6:
        raise argparse.ArgumentTypeError("key colour must use RRGGBB or #RRGGBB")
    try:
        return tuple(int(normalized[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("key colour must use hexadecimal digits") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="opaque chroma-key source PNG")
    parser.add_argument("--output", type=Path, required=True, help="transparent RGBA output PNG")
    parser.add_argument("--key-color", type=_parse_rgb, help="explicit RRGGBB key; default: median border colour")
    parser.add_argument(
        "--transparent-distance",
        type=float,
        default=DEFAULT_TRANSPARENT_DISTANCE,
        help=f"RGB distance at or below which alpha is zero (default: {DEFAULT_TRANSPARENT_DISTANCE:g})",
    )
    parser.add_argument(
        "--opaque-distance",
        type=float,
        default=DEFAULT_OPAQUE_DISTANCE,
        help=f"RGB distance at or above which alpha is unchanged (default: {DEFAULT_OPAQUE_DISTANCE:g})",
    )
    args = parser.parse_args()
    key = remove_chroma_key(
        args.source,
        args.output,
        key_color=args.key_color,
        transparent_distance=args.transparent_distance,
        opaque_distance=args.opaque_distance,
    )
    print(f"wrote {args.output} using key #{key[0]:02x}{key[1]:02x}{key[2]:02x}")


if __name__ == "__main__":
    main()
