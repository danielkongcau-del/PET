"""Generate a character rig manifest and dense training clips from glTF 2.0.

Character-specific choices belong to ``--profile``.  The extractor preserves
the selected skin hierarchy, inverse bind matrices, mesh/skin references and
source animation provenance without assuming a character or joint count.

Example:
  python tools/rig/extract_gltf_skeleton.py \
    --input assets/Cat/scene.gltf \
    --profile tools/rig/profiles/cat-rig-profile.json \
    --manifest-output assets/pet/runtime/cat-character-rig.manifest.json \
    --repository-root .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:  # Support both direct script execution and ``python -m tools.rig...``.
    from .gltf_character import build_rig_manifest, extract_animation_payload, load_gltf
    from .rig_contract import write_json
except ImportError:  # pragma: no cover - exercised by the documented CLI form
    from gltf_character import build_rig_manifest, extract_animation_payload, load_gltf
    from rig_contract import write_json


def _load_profile(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Character rig profile is missing: {path.resolve()}")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("Character rig profile must be a JSON object")
    if profile.get("schema") != "pet-character-rig-profile-v1":
        raise ValueError(f"Unsupported character rig profile schema: {profile.get('schema')!r}")
    return profile


def _path_for_output(path_value: str, repository_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repository_root / path


def generate_character_assets(
    *,
    source_path: Path,
    profile_path: Path,
    manifest_output: Path,
    repository_root: Path,
    skin_index: int | None = None,
    include_animations: bool = True,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    repository_root = repository_root.resolve()
    profile_path = profile_path.resolve()
    profile = _load_profile(profile_path)
    document = load_gltf(source_path)
    try:
        profile_uri = profile_path.relative_to(repository_root).as_posix()
    except ValueError:
        profile_uri = profile_path.as_posix()
    manifest = build_rig_manifest(
        document,
        profile,
        repository_root=repository_root,
        profile_uri=profile_uri,
        skin_index=skin_index,
    )
    generated_clips: list[tuple[Path, dict[str, Any]]] = []
    clip_specs = profile.get("trainingClips", [])
    if not isinstance(clip_specs, list):
        raise ValueError("profile.trainingClips must be an array")
    if include_animations:
        for index, clip_spec in enumerate(clip_specs):
            if not isinstance(clip_spec, Mapping):
                raise ValueError(f"trainingClips[{index}] must be an object")
            output_value = clip_spec.get("path")
            clip_id = clip_spec.get("clipId")
            animation_index = clip_spec.get("animationIndex")
            sample_rate = clip_spec.get("sampleRateHz", profile.get("sampleRateHz", 30.0))
            playback_mode = clip_spec.get("playbackMode")
            if not isinstance(output_value, str) or not output_value:
                raise ValueError(f"trainingClips[{index}].path must be a non-empty string")
            if not isinstance(clip_id, str) or not clip_id:
                raise ValueError(f"trainingClips[{index}].clipId must be a non-empty string")
            if not isinstance(animation_index, int):
                raise ValueError(f"trainingClips[{index}].animationIndex must be an integer")
            if not isinstance(sample_rate, (int, float)) or isinstance(sample_rate, bool):
                raise ValueError(f"trainingClips[{index}].sampleRateHz must be numeric")
            if playback_mode not in {"loop", "once"}:
                raise ValueError(
                    f"trainingClips[{index}].playbackMode must be 'loop' or 'once'"
                )
            clip = extract_animation_payload(
                document,
                manifest,
                animation_index,
                clip_id=clip_id,
                sample_rate_hz=float(sample_rate),
                playback_mode=playback_mode,
            )
            output_path = _path_for_output(output_value, repository_root)
            generated_clips.append((output_path, clip))
            manifest["trainingClips"].append(
                {
                    "clipId": clip_id,
                    "path": Path(output_value).as_posix(),
                    "fingerprint": clip["clipFingerprint"]["value"],
                    "playbackMode": playback_mode,
                }
            )
    for output_path, clip in generated_clips:
        write_json(output_path, clip)
    write_json(manifest_output, manifest)
    return manifest, generated_clips


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Source .gltf or .glb file")
    parser.add_argument("--profile", required=True, type=Path, help="Character rig profile JSON")
    parser.add_argument("--manifest-output", required=True, type=Path, help="Generated runtime manifest")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Root used to make source/profile/training paths portable (default: cwd)",
    )
    parser.add_argument(
        "--skin-index",
        type=int,
        default=None,
        help="Explicit skin override; required when the source has multiple skins and the profile omits one",
    )
    parser.add_argument(
        "--skip-animations",
        action="store_true",
        help="Generate only the manifest (trainingClips remains empty)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest, clips = generate_character_assets(
            source_path=args.input,
            profile_path=args.profile,
            manifest_output=args.manifest_output,
            repository_root=args.repository_root,
            skin_index=args.skin_index,
            include_animations=not args.skip_animations,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(
        f"Generated {args.manifest_output}: {len(manifest['rig']['jointOrder'])} full joints, "
        f"{len(manifest['rig']['drivenJointOrder'])} driven joints, {len(clips)} training clips, "
        f"rig {manifest['rigFingerprint']['value']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
