"""Character-agnostic glTF rig and animation extraction.

This module intentionally has no knowledge of a particular character or joint
count.  Character policy (driven joints, semantic roles and masks) lives in a
small JSON profile; source facts (hierarchy, skinning and animation) are read
from glTF and retained in the generated manifest.
"""

from __future__ import annotations

import base64
import bisect
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote_to_bytes

try:  # Support both direct script execution and ``tools.rig`` imports.
    from .rig_contract import (
        ANIMATION_SCHEMA,
        CHECKPOINT_FORMAT,
        CHECKPOINT_METADATA_SCHEMA,
        RIG_MANIFEST_SCHEMA,
        checkpoint_target_path,
        file_sha256,
        fingerprint_record,
        fingerprint_value,
        nearest_driven_parent_indices,
        validate_character_identifier,
        validate_animation,
        validate_manifest_identifier,
        validate_manifest,
    )
except ImportError:  # pragma: no cover - exercised by the command-line entrypoint
    from rig_contract import (
        ANIMATION_SCHEMA,
        CHECKPOINT_FORMAT,
        CHECKPOINT_METADATA_SCHEMA,
        RIG_MANIFEST_SCHEMA,
        checkpoint_target_path,
        file_sha256,
        fingerprint_record,
        fingerprint_value,
        nearest_driven_parent_indices,
        validate_character_identifier,
        validate_animation,
        validate_manifest_identifier,
        validate_manifest,
    )


_COMPONENT_FORMATS: dict[int, tuple[str, int, bool]] = {
    5120: ("b", 1, True),
    5121: ("B", 1, False),
    5122: ("h", 2, True),
    5123: ("H", 2, False),
    5125: ("I", 4, False),
    5126: ("f", 4, True),
}
_TYPE_WIDTHS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

# This accumulation order is part of the generated Rig ABI.  CPython 3.12
# changed builtin ``sum`` for floats, so transform extraction must not inherit
# interpreter-dependent compensation behavior.
F64_ACCUMULATION = "pet-left-to-right-f64-sum-v1"


def _f64_sum(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return total


@dataclass(frozen=True)
class GltfDocument:
    path: Path
    data: dict[str, Any]
    buffers: tuple[bytes, ...]
    format: str
    buffer_uris: tuple[str | None, ...]


def require_source(path: Path) -> Path:
    """Return a resolved source path or raise an actionable clean-clone error."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Required character source asset is missing: {resolved}. "
            "Raw character exports are intentionally not tracked in this repository. "
            "Restore the asset at that path (including its external buffers), or use "
            "the checked-in runtime manifest and derived animation assets."
        )
    return resolved


def _decode_data_uri(uri: str) -> bytes:
    try:
        header, payload = uri.split(",", 1)
    except ValueError as exc:
        raise ValueError("Malformed glTF data URI") from exc
    if ";base64" in header:
        return base64.b64decode(payload, validate=True)
    return unquote_to_bytes(payload)


def _load_external_buffer(source_path: Path, uri: str) -> bytes:
    if uri.startswith("data:"):
        return _decode_data_uri(uri)
    candidate = (source_path.parent / uri).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"glTF buffer {uri!r} referenced by {source_path} is missing at {candidate}. "
            "Restore all external files belonging to the source export."
        )
    return candidate.read_bytes()


def load_gltf(path: Path) -> GltfDocument:
    source_path = require_source(path)
    suffix = source_path.suffix.lower()
    if suffix == ".gltf":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if data.get("asset", {}).get("version") != "2.0":
            raise ValueError("Only glTF 2.0 sources are supported")
        buffers: list[bytes] = []
        uris: list[str | None] = []
        for index, spec in enumerate(data.get("buffers", [])):
            uri = spec.get("uri")
            if not isinstance(uri, str):
                raise ValueError(f"Text glTF buffer {index} has no URI")
            content = _load_external_buffer(source_path, uri)
            expected = spec.get("byteLength")
            if not isinstance(expected, int) or len(content) < expected:
                raise ValueError(
                    f"glTF buffer {index} has {len(content)} bytes; expected at least {expected}"
                )
            buffers.append(content)
            uris.append(uri if not uri.startswith("data:") else "<embedded:data-uri>")
        return GltfDocument(source_path, data, tuple(buffers), "gltf-2.0", tuple(uris))

    if suffix != ".glb":
        raise ValueError(f"Expected a .gltf or .glb source, got {source_path.name}")
    blob = source_path.read_bytes()
    if len(blob) < 12:
        raise ValueError("GLB header is truncated")
    magic, version, declared_length = struct.unpack_from("<4sII", blob, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(blob):
        raise ValueError("Invalid GLB 2.0 header")
    offset = 12
    json_chunk: bytes | None = None
    binary_chunks: list[bytes] = []
    while offset < len(blob):
        if offset + 8 > len(blob):
            raise ValueError("GLB chunk header is truncated")
        length, kind = struct.unpack_from("<II", blob, offset)
        offset += 8
        chunk = blob[offset : offset + length]
        if len(chunk) != length:
            raise ValueError("GLB chunk is truncated")
        offset += length
        if kind == 0x4E4F534A:
            if json_chunk is not None:
                raise ValueError("GLB contains more than one JSON chunk")
            json_chunk = chunk
        elif kind == 0x004E4942:
            binary_chunks.append(chunk)
    if json_chunk is None:
        raise ValueError("GLB has no JSON chunk")
    data = json.loads(json_chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
    if data.get("asset", {}).get("version") != "2.0":
        raise ValueError("Only glTF 2.0 sources are supported")
    bin_index = 0
    buffers = []
    uris = []
    for index, spec in enumerate(data.get("buffers", [])):
        uri = spec.get("uri")
        if isinstance(uri, str):
            content = _load_external_buffer(source_path, uri)
            display_uri: str | None = uri if not uri.startswith("data:") else "<embedded:data-uri>"
        else:
            if bin_index >= len(binary_chunks):
                raise ValueError(f"GLB buffer {index} has no matching BIN chunk")
            content = binary_chunks[bin_index]
            display_uri = None
            bin_index += 1
        expected = spec.get("byteLength")
        if not isinstance(expected, int) or len(content) < expected:
            raise ValueError(
                f"GLB buffer {index} has {len(content)} bytes; expected at least {expected}"
            )
        buffers.append(content)
        uris.append(display_uri)
    return GltfDocument(source_path, data, tuple(buffers), "glb-2.0", tuple(uris))


def _normalise_integer(value: int, component_type: int) -> float:
    if component_type == 5120:
        return max(float(value) / 127.0, -1.0)
    if component_type == 5121:
        return float(value) / 255.0
    if component_type == 5122:
        return max(float(value) / 32767.0, -1.0)
    if component_type == 5123:
        return float(value) / 65535.0
    if component_type == 5125:
        return float(value) / 4294967295.0
    return float(value)


def decode_accessor(document: GltfDocument, accessor_index: int) -> list[Any]:
    accessors = document.data.get("accessors", [])
    if not isinstance(accessor_index, int) or accessor_index < 0 or accessor_index >= len(accessors):
        raise ValueError(f"Accessor index {accessor_index} is out of range")
    accessor = accessors[accessor_index]
    if "sparse" in accessor:
        raise ValueError(
            f"Accessor {accessor_index} uses sparse storage, which this deterministic "
            "character extractor does not support"
        )
    component_type = accessor.get("componentType")
    accessor_type = accessor.get("type")
    count = accessor.get("count")
    if component_type not in _COMPONENT_FORMATS or accessor_type not in _TYPE_WIDTHS:
        raise ValueError(f"Accessor {accessor_index} has unsupported component/type")
    if not isinstance(count, int) or count < 0:
        raise ValueError(f"Accessor {accessor_index} has invalid count")
    width = _TYPE_WIDTHS[accessor_type]
    fmt, component_size, _ = _COMPONENT_FORMATS[component_type]
    element_size = component_size * width
    if "bufferView" not in accessor:
        zero: Any = 0.0 if width == 1 else [0.0] * width
        return [zero if width == 1 else list(zero) for _ in range(count)]
    view_index = accessor["bufferView"]
    views = document.data.get("bufferViews", [])
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(views):
        raise ValueError(f"Accessor {accessor_index} has invalid bufferView")
    view = views[view_index]
    buffer_index = view.get("buffer")
    if not isinstance(buffer_index, int) or buffer_index < 0 or buffer_index >= len(document.buffers):
        raise ValueError(f"Accessor {accessor_index} has invalid buffer index")
    stride = view.get("byteStride", element_size)
    if not isinstance(stride, int) or stride < element_size:
        raise ValueError(f"Accessor {accessor_index} has invalid byteStride")
    if stride % component_size != 0:
        raise ValueError(f"Accessor {accessor_index} byteStride is not component-aligned")
    view_offset = view.get("byteOffset", 0)
    view_length = view.get("byteLength")
    accessor_offset = accessor.get("byteOffset", 0)
    if (
        not isinstance(view_offset, int)
        or view_offset < 0
        or not isinstance(view_length, int)
        or view_length < 0
        or not isinstance(accessor_offset, int)
        or accessor_offset < 0
    ):
        raise ValueError(f"Accessor {accessor_index} has invalid byte offsets")
    if accessor_offset % component_size != 0:
        raise ValueError(f"Accessor {accessor_index} byteOffset is not component-aligned")
    required_in_view = accessor_offset
    if count:
        required_in_view += (count - 1) * stride + element_size
    if required_in_view > view_length:
        raise ValueError(f"Accessor {accessor_index} reads beyond its bufferView")
    offset = view_offset + accessor_offset
    content = document.buffers[buffer_index]
    if view_offset + view_length > len(content):
        raise ValueError(f"Accessor {accessor_index} references a truncated bufferView")
    values: list[Any] = []
    unpack = struct.Struct("<" + fmt * width)
    normalised = bool(accessor.get("normalized", False)) and component_type != 5126
    for item_index in range(count):
        item_offset = offset + item_index * stride
        if item_offset < 0 or item_offset + element_size > len(content):
            raise ValueError(f"Accessor {accessor_index} reads beyond its buffer")
        components: list[int | float] = list(unpack.unpack_from(content, item_offset))
        if normalised:
            components = [_normalise_integer(int(value), component_type) for value in components]
        result: Any = components[0] if width == 1 else [float(value) for value in components]
        values.append(result)
    return values


def _accessor_spec(document: GltfDocument, accessor_index: Any, label: str) -> Mapping[str, Any]:
    accessors = document.data.get("accessors", [])
    if not isinstance(accessor_index, int) or accessor_index < 0 or accessor_index >= len(accessors):
        raise ValueError(f"{label} references an invalid accessor")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, Mapping):
        raise ValueError(f"{label} accessor must be an object")
    return accessor


def _require_float_accessor(
    document: GltfDocument,
    accessor_index: Any,
    accessor_type: str,
    label: str,
) -> Mapping[str, Any]:
    accessor = _accessor_spec(document, accessor_index, label)
    if (
        accessor.get("componentType") != 5126
        or accessor.get("type") != accessor_type
        or accessor.get("normalized", False) is not False
    ):
        raise ValueError(
            f"{label} must use a non-normalized FLOAT {accessor_type} accessor"
        )
    return accessor


def _identity_matrix() -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _matrix_multiply(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [
            _f64_sum(
                [float(left[row][k]) * float(right[k][column]) for k in range(4)]
            )
            for column in range(4)
        ]
        for row in range(4)
    ]


def _quat_normalise(value: Sequence[float]) -> list[float]:
    if len(value) != 4:
        raise ValueError("Quaternion must have four components")
    length = math.sqrt(_f64_sum([float(component) ** 2 for component in value]))
    if length <= 1e-12:
        raise ValueError("Quaternion has zero length")
    return [float(component) / length for component in value]


def _quat_to_matrix(quaternion: Sequence[float]) -> list[list[float]]:
    x, y, z, w = _quat_normalise(quaternion)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _trs_matrix(translation: Sequence[float], rotation: Sequence[float], scale: Sequence[float]) -> list[list[float]]:
    if len(translation) != 3 or len(scale) != 3:
        raise ValueError("Translation and scale must have three components")
    rot = _quat_to_matrix(rotation)
    matrix = _identity_matrix()
    for row in range(3):
        for column in range(3):
            matrix[row][column] = rot[row][column] * float(scale[column])
        matrix[row][3] = float(translation[row])
    return matrix


def _matrix_to_quat(matrix: Sequence[Sequence[float]]) -> list[float]:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        value = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        value = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        value = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        value = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
    return _quat_normalise(value)


def _decompose_matrix(matrix: Sequence[Sequence[float]]) -> dict[str, list[float]]:
    if any(len(row) != 4 for row in matrix) or len(matrix) != 4:
        raise ValueError("Expected a 4x4 transform matrix")
    if any(abs(float(matrix[3][column]) - (1.0 if column == 3 else 0.0)) > 1e-7 for column in range(4)):
        raise ValueError("Perspective node matrices cannot be represented as rig TRS")
    columns = [[float(matrix[row][column]) for row in range(3)] for column in range(3)]
    scales = [math.sqrt(_f64_sum([value * value for value in column])) for column in columns]
    if any(scale <= 1e-12 for scale in scales):
        raise ValueError("Node matrix contains a zero scale axis")
    unit = [[columns[column][row] / scales[column] for row in range(3)] for column in range(3)]
    for left in range(3):
        for right in range(left + 1, 3):
            dot = _f64_sum(
                [unit[left][row] * unit[right][row] for row in range(3)]
            )
            if abs(dot) > 1e-5:
                raise ValueError("Node matrix contains shear and cannot be represented as TRS")
    determinant = (
        unit[0][0] * (unit[1][1] * unit[2][2] - unit[1][2] * unit[2][1])
        - unit[1][0] * (unit[0][1] * unit[2][2] - unit[0][2] * unit[2][1])
        + unit[2][0] * (unit[0][1] * unit[1][2] - unit[0][2] * unit[1][1])
    )
    if determinant < 0:
        axis = max(range(3), key=lambda index: scales[index])
        scales[axis] = -scales[axis]
        unit[axis] = [-value for value in unit[axis]]
    rotation_matrix = [[unit[column][row] for column in range(3)] for row in range(3)]
    return {
        "translation": [float(matrix[row][3]) for row in range(3)],
        "rotation": _matrix_to_quat(rotation_matrix),
        "scale": scales,
    }


def node_local_matrix(node: Mapping[str, Any]) -> list[list[float]]:
    if "matrix" in node:
        if any(key in node for key in ("translation", "rotation", "scale")):
            raise ValueError("A glTF node cannot declare both matrix and TRS properties")
        values = node["matrix"]
        if not isinstance(values, list) or len(values) != 16:
            raise ValueError("glTF node matrix must contain 16 values")
        return [[float(values[column * 4 + row]) for column in range(4)] for row in range(4)]
    return _trs_matrix(
        node.get("translation", [0.0, 0.0, 0.0]),
        node.get("rotation", [0.0, 0.0, 0.0, 1.0]),
        node.get("scale", [1.0, 1.0, 1.0]),
    )


def node_local_trs(node: Mapping[str, Any]) -> dict[str, list[float]]:
    return _decompose_matrix(node_local_matrix(node))


def _parent_map(nodes: Sequence[Mapping[str, Any]]) -> list[int]:
    parents = [-1] * len(nodes)
    for parent_index, node in enumerate(nodes):
        for child in node.get("children", []):
            if not isinstance(child, int) or child < 0 or child >= len(nodes):
                raise ValueError(f"Node {parent_index} has invalid child index {child!r}")
            if parents[child] != -1:
                raise ValueError(f"glTF node {child} has multiple parents")
            parents[child] = parent_index
    return parents


def _ancestors(index: int, parents: Sequence[int]) -> list[int]:
    result: list[int] = []
    visited: set[int] = set()
    while index >= 0:
        if index in visited:
            raise ValueError("Cycle detected in glTF node hierarchy")
        visited.add(index)
        result.append(index)
        index = parents[index]
    return result


def _lowest_common_ancestor(indices: Sequence[int], parents: Sequence[int]) -> int:
    if not indices:
        raise ValueError("Cannot derive a skeleton root without skin joints")
    common = set(_ancestors(indices[0], parents))
    for index in indices[1:]:
        common.intersection_update(_ancestors(index, parents))
    if not common:
        raise ValueError("Skin joints do not share a node-tree ancestor")
    return max(common, key=lambda index: len(_ancestors(index, parents)))


def _select_skin(document: GltfDocument, skin_index: int | None) -> tuple[int, Mapping[str, Any], list[int], int, list[int]]:
    nodes = document.data.get("nodes", [])
    skins = document.data.get("skins", [])
    if not skins:
        raise ValueError("No skin found in glTF source")
    if skin_index is None:
        if len(skins) != 1:
            raise ValueError(
                f"glTF contains {len(skins)} skins; pass --skin-index or set skinIndex "
                "in the character profile so selection is explicit"
            )
        skin_index = 0
    if skin_index < 0 or skin_index >= len(skins):
        raise ValueError(f"Skin index {skin_index} is out of range for {len(skins)} skins")
    skin = skins[skin_index]
    joints = skin.get("joints", [])
    if not isinstance(joints, list) or not joints:
        raise ValueError(f"Skin {skin_index} has no joints")
    if len(set(joints)) != len(joints):
        raise ValueError(f"Skin {skin_index} contains duplicate joint nodes")
    if any(not isinstance(index, int) or index < 0 or index >= len(nodes) for index in joints):
        raise ValueError(f"Skin {skin_index} contains an invalid joint node index")
    parents = _parent_map(nodes)
    skeleton_root = skin.get("skeleton")
    if skeleton_root is None:
        skeleton_root = _lowest_common_ancestor(joints, parents)
    if not isinstance(skeleton_root, int) or skeleton_root < 0 or skeleton_root >= len(nodes):
        raise ValueError(f"Skin {skin_index} has an invalid skeleton root")
    for joint in joints:
        if skeleton_root not in _ancestors(joint, parents):
            raise ValueError(
                f"Skin {skin_index} joint node {joint} is not below skeleton node {skeleton_root}"
            )
    return skin_index, skin, list(joints), skeleton_root, parents


def _required_skin_nodes(joints: Sequence[int], skeleton_root: int, parents: Sequence[int]) -> set[int]:
    required = {skeleton_root}
    for joint in joints:
        current = joint
        while current != skeleton_root:
            required.add(current)
            current = parents[current]
            if current < 0:
                raise ValueError(f"Joint node {joint} is not below the selected skeleton root")
        required.add(skeleton_root)
    return required


def _ordered_required_nodes(nodes: Sequence[Mapping[str, Any]], root: int, required: set[int]) -> list[int]:
    ordered: list[int] = []

    def visit(index: int) -> None:
        if index not in required:
            return
        ordered.append(index)
        for child in nodes[index].get("children", []):
            visit(child)

    visit(root)
    if set(ordered) != required:
        missing = sorted(required.difference(ordered))
        raise ValueError(f"Could not traverse selected skeleton nodes: {missing}")
    return ordered


def _world_matrix(node_index: int, nodes: Sequence[Mapping[str, Any]], parents: Sequence[int]) -> list[list[float]]:
    chain = list(reversed(_ancestors(node_index, parents)))
    result = _identity_matrix()
    for index in chain:
        result = _matrix_multiply(result, node_local_matrix(nodes[index]))
    return result


def _safe_joint_id(name: str, node_index: int, used: set[str]) -> str:
    safe = re.sub(r"[^a-z0-9_]", "_", name.lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe or not re.match(r"^[a-z_]", safe):
        safe = f"node_{node_index}_{safe}".rstrip("_")
    candidate = safe
    if candidate in used:
        candidate = f"{safe}_node_{node_index}"
    if candidate in used:
        raise ValueError(f"Could not derive a unique joint id for node {node_index}")
    used.add(candidate)
    return candidate


def _as_bool3(value: Any, field: str) -> list[bool]:
    if not isinstance(value, list) or len(value) != 3 or any(type(item) is not bool for item in value):
        raise ValueError(f"{field} must contain exactly three booleans")
    return list(value)


def _dof_for_source(profile: Mapping[str, Any], source_name: str, driven: bool) -> dict[str, list[bool]]:
    defaults = profile.get("drivenDofMask", {}) if driven else profile.get("staticDofMask", {})
    base = {
        "translation": _as_bool3(defaults.get("translation", [False, False, False]), "translation DOF"),
        "rotation": _as_bool3(defaults.get("rotation", [True, True, True] if driven else [False, False, False]), "rotation DOF"),
        "scale": _as_bool3(defaults.get("scale", [False, False, False]), "scale DOF"),
    }
    override = profile.get("dofOverrides", {}).get(source_name)
    if override is not None:
        if not isinstance(override, Mapping):
            raise ValueError(f"DOF override for {source_name!r} must be an object")
        for field in ("translation", "rotation", "scale"):
            if field in override:
                base[field] = _as_bool3(override[field], f"{source_name}.{field}")
    return base


def _source_name(nodes: Sequence[Mapping[str, Any]], index: int) -> str:
    value = nodes[index].get("name")
    return value if isinstance(value, str) and value else f"node_{index}"


def _source_uri(path: Path, repository_root: Path | None) -> str:
    if repository_root is not None:
        try:
            return path.resolve().relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _render_contract(profile: Mapping[str, Any], selected_skin_index: int) -> dict[str, Any]:
    configured = profile.get("render")
    if not isinstance(configured, Mapping):
        raise ValueError("profile.render must be an object")
    # A JSON round trip makes a plain deep copy and rejects accidental custom values.
    render = json.loads(json.dumps(configured, ensure_ascii=False, allow_nan=False))
    for key in (
        "canvas",
        "displayScale",
        "footAnchor",
        "sourceFacing",
        "mode",
        "fallbackModes",
        "sprite",
        "mesh",
    ):
        if key not in render:
            raise ValueError(f"profile.render.{key} is required")
    sprite = render.get("sprite")
    if isinstance(sprite, Mapping):
        for field in ("image", "metadata"):
            value = sprite.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"profile.render.sprite.{field} must be a relative path")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"profile.render.sprite.{field} must be relative to the generated manifest"
                )
    mesh = render.get("mesh")
    if isinstance(mesh, Mapping):
        if mesh.get("source") != "source":
            raise ValueError("profile.render.mesh.source must be 'source'")
        if mesh.get("skinIndex") != selected_skin_index:
            raise ValueError("profile.render.mesh.skinIndex must match the selected glTF skin")
    return render


def _inverse_bind_matrices(document: GltfDocument, skin: Mapping[str, Any], joint_count: int) -> tuple[int | None, list[list[float]]]:
    accessor_index = skin.get("inverseBindMatrices")
    if accessor_index is None:
        identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return None, [list(identity) for _ in range(joint_count)]
    if not isinstance(accessor_index, int):
        raise ValueError("inverseBindMatrices must reference an accessor")
    _require_float_accessor(
        document,
        accessor_index,
        "MAT4",
        "inverseBindMatrices",
    )
    values = decode_accessor(document, accessor_index)
    if len(values) != joint_count or any(not isinstance(value, list) or len(value) != 16 for value in values):
        raise ValueError("inverseBindMatrices accessor must contain one MAT4 per skin joint")
    return accessor_index, [[float(component) for component in value] for value in values]


def _mesh_skin_references(document: GltfDocument, skin_index: int) -> list[dict[str, Any]]:
    nodes = document.data.get("nodes", [])
    meshes = document.data.get("meshes", [])
    references: list[dict[str, Any]] = []
    for node_index, node in enumerate(nodes):
        if node.get("skin") != skin_index:
            continue
        mesh_index = node.get("mesh")
        if not isinstance(mesh_index, int) or mesh_index < 0 or mesh_index >= len(meshes):
            raise ValueError(f"Skinned node {node_index} has invalid mesh index")
        primitives = []
        for primitive_index, primitive in enumerate(meshes[mesh_index].get("primitives", [])):
            attributes = primitive.get("attributes", {})
            if not isinstance(attributes, Mapping):
                raise ValueError(f"Mesh {mesh_index} primitive {primitive_index} has invalid attributes")
            primitives.append(
                {
                    "primitiveIndex": primitive_index,
                    "mode": int(primitive.get("mode", 4)),
                    "indicesAccessor": primitive.get("indices"),
                    "materialIndex": primitive.get("material"),
                    "attributes": {str(key): int(value) for key, value in sorted(attributes.items())},
                }
            )
        references.append(
            {
                "nodeIndex": node_index,
                "nodeName": node.get("name") if isinstance(node.get("name"), str) else None,
                "meshIndex": mesh_index,
                "skinIndex": skin_index,
                "primitives": primitives,
            }
        )
    return references


def _animation_range(document: GltfDocument, animation: Mapping[str, Any]) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for sampler_index, sampler in enumerate(animation.get("samplers", [])):
        input_accessor = sampler.get("input")
        _require_float_accessor(
            document,
            input_accessor,
            "SCALAR",
            f"animation sampler {sampler_index} input",
        )
        times = decode_accessor(document, input_accessor)
        if not times:
            continue
        numeric = [float(value) for value in times]
        if any(right <= left for left, right in zip(numeric, numeric[1:])):
            raise ValueError("Animation sampler input times must be strictly increasing")
        starts.append(numeric[0])
        ends.append(numeric[-1])
    return (min(starts), max(ends)) if starts else (0.0, 0.0)


def _source_animation_summaries(document: GltfDocument) -> list[dict[str, Any]]:
    result = []
    for index, animation in enumerate(document.data.get("animations", [])):
        start, end = _animation_range(document, animation)
        result.append(
            {
                "animationIndex": index,
                "name": animation.get("name") or f"animation_{index}",
                "channelCount": len(animation.get("channels", [])),
                "durationMs": (end - start) * 1000.0,
            }
        )
    return result


def build_rig_manifest(
    document: GltfDocument,
    profile: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
    profile_uri: str | None = None,
    skin_index: int | None = None,
) -> dict[str, Any]:
    nodes = document.data.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("glTF source has no nodes")
    configured_skin = profile.get("skinIndex") if skin_index is None else skin_index
    if configured_skin is not None and not isinstance(configured_skin, int):
        raise ValueError("profile.skinIndex must be an integer")
    selected_index, skin, skin_joints, skeleton_root, parents = _select_skin(document, configured_skin)
    required = _required_skin_nodes(skin_joints, skeleton_root, parents)
    ordered_nodes = _ordered_required_nodes(nodes, skeleton_root, required)
    names = [_source_name(nodes, index) for index in ordered_nodes]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(
            "Selected skeleton contains duplicate source node names, which makes a profile "
            f"ambiguous: {duplicates}. Rename the nodes in the source export."
        )
    driven_sources = profile.get("drivenSourceNames")
    if not isinstance(driven_sources, list) or not driven_sources:
        raise ValueError("profile.drivenSourceNames must be a non-empty list")
    if len(set(driven_sources)) != len(driven_sources):
        raise ValueError("profile.drivenSourceNames contains duplicates")
    missing = [name for name in driven_sources if name not in names]
    if missing:
        raise ValueError(f"Driven source nodes are absent from selected skin: {missing}")
    if len(driven_sources) > 128:
        raise ValueError("At most 128 model-driven joints are supported by the motion protocol")

    motion_root = profile.get("motionRoot", "__motion_root__")
    if not isinstance(motion_root, str) or not re.match(r"^[a-z_][a-z0-9_]*$", motion_root):
        raise ValueError("profile.motionRoot is not a valid joint id")
    used = {motion_root}
    node_to_id = {
        node_index: _safe_joint_id(_source_name(nodes, node_index), node_index, used)
        for node_index in ordered_nodes
    }
    source_to_id = {_source_name(nodes, node_index): node_to_id[node_index] for node_index in ordered_nodes}
    driven_order = [source_to_id[name] for name in driven_sources]
    joint_order = [motion_root] + [node_to_id[index] for index in ordered_nodes]
    skin_joint_positions = {node_index: index for index, node_index in enumerate(skin_joints)}
    semantic_roles = profile.get("semanticRoles", {})
    if not isinstance(semantic_roles, Mapping):
        raise ValueError("profile.semanticRoles must be an object")
    unknown_role_names = sorted(set(semantic_roles).difference(names))
    if unknown_role_names:
        raise ValueError(f"semanticRoles references unknown source nodes: {unknown_role_names}")
    joints: list[dict[str, Any]] = [
        {
            "id": motion_root,
            "parentIndex": -1,
            "restLocal": {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
            },
            "dofMask": {
                "translation": [False, False, False],
                "rotation": [False, False, False],
                "scale": [False, False, False],
            },
            "semanticRole": "motion_root",
            "source": None,
        }
    ]
    output_index = {node_index: index + 1 for index, node_index in enumerate(ordered_nodes)}
    driven_names = set(driven_sources)
    for node_index in ordered_nodes:
        source_name = _source_name(nodes, node_index)
        parent_node = parents[node_index]
        parent_index = 0 if node_index == skeleton_root else output_index[parent_node]
        transform = _decompose_matrix(_world_matrix(node_index, nodes, parents)) if node_index == skeleton_root else node_local_trs(nodes[node_index])
        is_driven = source_name in driven_names
        joints.append(
            {
                "id": node_to_id[node_index],
                "parentIndex": parent_index,
                "restLocal": transform,
                "dofMask": _dof_for_source(profile, source_name, is_driven),
                "semanticRole": semantic_roles.get(source_name, "joint" if node_index in skin_joint_positions else "connector"),
                "source": {
                    "skinIndex": selected_index,
                    "skinJointIndex": skin_joint_positions.get(node_index),
                    "nodeIndex": node_index,
                    "sourceName": source_name,
                },
            }
        )
    parent_indices = [joint["parentIndex"] for joint in joints]
    driven_indices, driven_parents = nearest_driven_parent_indices(joint_order, parent_indices, driven_order)

    configured_masks = profile.get("maskGroups")
    if not isinstance(configured_masks, Mapping) or not configured_masks:
        raise ValueError("profile.maskGroups must be a non-empty object")
    mask_groups: dict[str, list[bool]] = {}
    for mask_name, members in configured_masks.items():
        validate_manifest_identifier(mask_name, "profile.maskGroups key")
        if not isinstance(members, list) or any(member not in driven_sources for member in members):
            raise ValueError(f"Mask group {mask_name!r} must list known driven source names")
        member_set = set(members)
        mask_groups[mask_name] = [name in member_set for name in driven_sources]

    coordinate_system = profile.get(
        "coordinateSystem",
        {"handedness": "right", "up": "+Y", "forward": "+Z", "units": "meter"},
    )
    rig = {
        "coordinateSystem": coordinate_system,
        "motionRoot": motion_root,
        "jointOrder": joint_order,
        "joints": joints,
        "drivenJointOrder": driven_order,
        "drivenJointIndices": driven_indices,
        "drivenParentIndices": driven_parents,
        "maskGroups": mask_groups,
    }
    rig_fingerprint = fingerprint_record(rig)
    inverse_accessor, inverse_matrices = _inverse_bind_matrices(document, skin, len(skin_joints))
    source_uri = _source_uri(document.path, repository_root)
    source = {
        "format": document.format,
        "uri": source_uri,
        "requiredAtBuild": bool(profile.get("sourceRequiredAtBuild", True)),
        "availableInCleanClone": bool(profile.get("sourceAvailableInCleanClone", False)),
        "sha256": file_sha256(document.path),
        "selectedSkinIndex": selected_index,
        "buffers": [
            {
                "index": index,
                "uri": document.buffer_uris[index],
                "byteLength": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for index, content in enumerate(document.buffers)
        ],
        "skins": [
            {
                "skinIndex": selected_index,
                "name": skin.get("name") if isinstance(skin.get("name"), str) else None,
                "skeletonNodeIndex": skeleton_root,
                "jointNodeIndices": skin_joints,
                "inverseBindMatricesAccessor": inverse_accessor,
                "inverseBindMatrices": inverse_matrices,
            }
        ],
        "meshSkinReferences": _mesh_skin_references(document, selected_index),
        "animations": _source_animation_summaries(document),
    }
    character_id = validate_character_identifier(
        profile.get("characterId"), "profile.characterId"
    )
    rig_id = validate_manifest_identifier(profile.get("rigId"), "profile.rigId")
    checkpoint = profile.get("checkpoint", {})
    if not isinstance(checkpoint, Mapping):
        raise ValueError("profile.checkpoint must be an object")
    if set(checkpoint) != {"format", "metadataSchema"}:
        raise ValueError(
            "profile.checkpoint must contain only format and metadataSchema"
        )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("profile.checkpoint.format is unsupported")
    if checkpoint.get("metadataSchema") != CHECKPOINT_METADATA_SCHEMA:
        raise ValueError("profile.checkpoint.metadataSchema is unsupported")
    manifest = {
        "schema": RIG_MANIFEST_SCHEMA,
        "characterId": character_id,
        "rigId": rig_id,
        "rigFingerprint": rig_fingerprint,
        "rig": rig,
        "render": _render_contract(profile, selected_index),
        "checkpoint": {
            "format": CHECKPOINT_FORMAT,
            "metadataSchema": CHECKPOINT_METADATA_SCHEMA,
            "path": checkpoint_target_path(character_id, rig_fingerprint["value"]),
            "characterId": character_id,
            "rigFingerprint": rig_fingerprint["value"],
            "drivenJointOrder": driven_order,
        },
        "source": source,
        "trainingClips": [],
        "provenance": {
            "generator": "tools/rig/extract_gltf_skeleton.py",
            "generatorVersion": 2,
            "profile": profile_uri,
            "f64Accumulation": F64_ACCUMULATION,
        },
    }
    validate_manifest(manifest)
    return manifest


def _lerp(left: Sequence[float], right: Sequence[float], factor: float) -> list[float]:
    return [float(a) + (float(b) - float(a)) * factor for a, b in zip(left, right)]


def _slerp(left: Sequence[float], right: Sequence[float], factor: float) -> list[float]:
    a = _quat_normalise(left)
    b = _quat_normalise(right)
    dot = _f64_sum([x * y for x, y in zip(a, b)])
    if dot < 0:
        b = [-value for value in b]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _quat_normalise(_lerp(a, b, factor))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    left_weight = math.sin((1.0 - factor) * theta) / sin_theta
    right_weight = math.sin(factor * theta) / sin_theta
    return _quat_normalise([left_weight * a[index] + right_weight * b[index] for index in range(4)])


def _sample_channel(times: Sequence[float], values: Sequence[Any], interpolation: str, time: float, path: str) -> list[float]:
    if not times:
        raise ValueError("Animation sampler contains no times")
    if time <= times[0]:
        index = 0
        raw = values[index * 3 + 1] if interpolation == "CUBICSPLINE" else values[index]
        return _quat_normalise(raw) if path == "rotation" else [float(value) for value in raw]
    if time >= times[-1]:
        index = len(times) - 1
        raw = values[index * 3 + 1] if interpolation == "CUBICSPLINE" else values[index]
        return _quat_normalise(raw) if path == "rotation" else [float(value) for value in raw]
    right_index = bisect.bisect_right(times, time)
    left_index = right_index - 1
    span = times[right_index] - times[left_index]
    factor = 0.0 if span <= 0 else (time - times[left_index]) / span
    if interpolation == "STEP":
        raw = values[left_index]
        return _quat_normalise(raw) if path == "rotation" else [float(value) for value in raw]
    if interpolation == "CUBICSPLINE":
        p0 = values[left_index * 3 + 1]
        m0 = values[left_index * 3 + 2]
        p1 = values[right_index * 3 + 1]
        m1 = values[right_index * 3]
        t2 = factor * factor
        t3 = t2 * factor
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + factor
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        raw = [
            h00 * float(p0[index]) + h10 * span * float(m0[index]) + h01 * float(p1[index]) + h11 * span * float(m1[index])
            for index in range(len(p0))
        ]
        return _quat_normalise(raw) if path == "rotation" else raw
    if interpolation != "LINEAR":
        raise ValueError(f"Unsupported glTF animation interpolation: {interpolation}")
    return _slerp(values[left_index], values[right_index], factor) if path == "rotation" else _lerp(values[left_index], values[right_index], factor)


def extract_animation_payload(
    document: GltfDocument,
    manifest: Mapping[str, Any],
    animation_index: int,
    *,
    clip_id: str,
    sample_rate_hz: float,
    playback_mode: str = "once",
) -> dict[str, Any]:
    animations = document.data.get("animations", [])
    if animation_index < 0 or animation_index >= len(animations):
        raise ValueError(f"Animation index {animation_index} is out of range")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if sample_rate_hz > 1000:
        raise ValueError("sample_rate_hz must not exceed 1000")
    if playback_mode not in {"loop", "once"}:
        raise ValueError("playback_mode must be 'loop' or 'once'")
    animation = animations[animation_index]
    start, end = _animation_range(document, animation)
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Animation {animation_index} has no positive duration")
    channels: dict[tuple[int, str], tuple[list[float], list[Any], str]] = {}
    samplers = animation.get("samplers", [])
    for channel_index, channel in enumerate(animation.get("channels", [])):
        target = channel.get("target", {})
        path = target.get("path")
        node_index = target.get("node")
        if path not in ("translation", "rotation", "scale"):
            continue
        if not isinstance(node_index, int) or node_index < 0 or node_index >= len(document.data.get("nodes", [])):
            raise ValueError(f"Animation channel {channel_index} has invalid target node")
        sampler_index = channel.get("sampler")
        if not isinstance(sampler_index, int) or sampler_index < 0 or sampler_index >= len(samplers):
            raise ValueError(f"Animation channel {channel_index} has invalid sampler")
        key = (node_index, path)
        if key in channels:
            raise ValueError(f"Animation has duplicate channels for node {node_index} {path}")
        sampler = samplers[sampler_index]
        input_accessor = sampler.get("input")
        output_accessor = sampler.get("output")
        _require_float_accessor(
            document,
            input_accessor,
            "SCALAR",
            f"animation sampler {sampler_index} input",
        )
        _require_float_accessor(
            document,
            output_accessor,
            "VEC4" if path == "rotation" else "VEC3",
            f"animation sampler {sampler_index} {path} output",
        )
        times = [float(value) for value in decode_accessor(document, input_accessor)]
        values = decode_accessor(document, output_accessor)
        interpolation = sampler.get("interpolation", "LINEAR")
        if interpolation not in ("LINEAR", "STEP", "CUBICSPLINE"):
            raise ValueError(f"Unsupported glTF animation interpolation: {interpolation}")
        expected = len(times) * (3 if interpolation == "CUBICSPLINE" else 1)
        if len(values) != expected:
            raise ValueError(f"Animation sampler {sampler_index} has mismatched input/output counts")
        channels[key] = (times, values, interpolation)

    rig = manifest["rig"]
    source_by_id = {joint["id"]: joint for joint in rig["joints"]}
    driven_order = list(rig["drivenJointOrder"])
    frame_count = max(2, int(math.ceil(duration * sample_rate_hz)) + 1)
    frames: list[dict[str, Any]] = []
    previous_rotations: list[list[float]] | None = None
    for frame_index in range(frame_count):
        relative_time = duration if frame_index == frame_count - 1 else min(frame_index / sample_rate_hz, duration)
        source_time = start + relative_time
        rotations: list[list[float]] = []
        translations: list[list[float]] = []
        for joint_id in driven_order:
            joint = source_by_id[joint_id]
            source = joint["source"]
            if source is None:
                raise ValueError(f"Driven joint {joint_id} has no source mapping")
            node_index = source["nodeIndex"]
            rest = joint["restLocal"]
            rotation_channel = channels.get((node_index, "rotation"))
            translation_channel = channels.get((node_index, "translation"))
            rotation = (
                _sample_channel(*rotation_channel, source_time, "rotation")
                if rotation_channel is not None
                else list(rest["rotation"])
            )
            rotation = _quat_normalise(rotation)
            if previous_rotations is not None:
                previous = previous_rotations[len(rotations)]
                if _f64_sum(
                    [left * right for left, right in zip(previous, rotation)]
                ) < 0.0:
                    rotation = [-component for component in rotation]
            rotations.append(rotation)
            translations.append(
                _sample_channel(*translation_channel, source_time, "translation")
                if translation_channel is not None
                else list(rest["translation"])
            )
        frames.append(
            {
                "timeMs": relative_time * 1000.0,
                "localRotations": rotations,
                "localTranslations": translations,
            }
        )
        previous_rotations = rotations
    if playback_mode == "loop":
        closed_rotations: list[list[float]] = []
        for first, previous in zip(
            frames[0]["localRotations"],
            frames[-2]["localRotations"],
        ):
            closed = list(first)
            if _f64_sum(
                [left * right for left, right in zip(previous, closed)]
            ) < 0.0:
                closed = [-component for component in closed]
            closed_rotations.append(closed)
        frames[-1]["localRotations"] = closed_rotations
        frames[-1]["localTranslations"] = [
            list(translation) for translation in frames[0]["localTranslations"]
        ]
    payload: dict[str, Any] = {
        "schema": ANIMATION_SCHEMA,
        "characterId": manifest["characterId"],
        "rigId": manifest["rigId"],
        "rigFingerprint": manifest["rigFingerprint"]["value"],
        "clipId": clip_id,
        "name": animation.get("name") or f"animation_{animation_index}",
        "durationMs": duration * 1000.0,
        "sampleRateHz": float(sample_rate_hz),
        "playbackMode": playback_mode,
        "jointOrder": driven_order,
        "frames": frames,
        "source": {
            "uri": manifest["source"]["uri"],
            "sha256": manifest["source"]["sha256"],
            "animationIndex": animation_index,
            "animationName": animation.get("name") or f"animation_{animation_index}",
        },
    }
    payload["clipFingerprint"] = fingerprint_record(payload)
    # Keep a stable, readable field order while hashing the payload without its fingerprint.
    payload = {
        "schema": payload["schema"],
        "characterId": payload["characterId"],
        "rigId": payload["rigId"],
        "rigFingerprint": payload["rigFingerprint"],
        "clipFingerprint": payload["clipFingerprint"],
        "clipId": payload["clipId"],
        "name": payload["name"],
        "durationMs": payload["durationMs"],
        "sampleRateHz": payload["sampleRateHz"],
        "playbackMode": payload["playbackMode"],
        "jointOrder": payload["jointOrder"],
        "frames": payload["frames"],
        "source": payload["source"],
    }
    validate_animation(payload)
    return payload


def clip_fingerprint_is_valid(payload: Mapping[str, Any]) -> bool:
    fingerprint = payload.get("clipFingerprint")
    if not isinstance(fingerprint, Mapping):
        return False
    unsigned = {key: value for key, value in payload.items() if key != "clipFingerprint"}
    return fingerprint.get("value") == fingerprint_value(unsigned)
