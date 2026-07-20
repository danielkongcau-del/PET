"""Extract skeleton hierarchy from a glTF 2.0 file and write cat-skeleton-3d.json.

Usage:
  python tools/rig/extract_gltf_skeleton.py \
    --input assets/Cat/scene.gltf \
    --output assets/pet/runtime/cat-skeleton-3d.json
"""

import argparse
import json
from pathlib import Path


def extract_skeleton(gltf: dict) -> list[dict]:
    nodes = gltf.get("nodes", [])
    skins = gltf.get("skins", [])
    if not skins:
        raise ValueError("No skin found in glTF file")

    skin = skins[0]
    joint_indices = set(skin.get("joints", []))
    skeleton_root = skin.get("skeleton")

    if skeleton_root is None:
        raise ValueError("Skin has no skeleton root node")
    if not joint_indices:
        raise ValueError("Skin has no joints")

    print(f"Skin: {len(joint_indices)} joints, skeleton root node {skeleton_root}")

    # Build node index map
    node_index = {}
    for i, node in enumerate(nodes):
        node_index[i] = node

    joints = []
    id_counter = [0]

    def sanitize_id(name):
        if not name:
            return f"bone_{id_counter[0]}"
        safe = "".join(c if c.isalnum() or c == '_' else '_' for c in name.lower())
        if not safe or safe[0].isdigit():
            safe = "bone_" + safe
        id_counter[0] += 1
        return safe

    # Add __motion_root__ first
    joints.append({
        "id": "__motion_root__",
        "name": "Motion Root",
        "parent": None,
        "role": "motion_root",
        "deform": False,
        "restLocal": {
            "translation": [0, 0, 0],
            "rotation": [0, 0, 0, 1],
            "scale": [1, 1, 1],
        },
        "poseDofs": {
            "translation": False,
            "rotation": False,
            "scale": False,
        },
        "limits": {},
        "sprite": None,
    })

    def process_node(node_idx: int, parent_id: str):
        node = nodes[node_idx]
        name = node.get("name", f"node_{node_idx}")
        bid = sanitize_id(name)

        # glTF node transform: T * R * S
        trans = node.get("translation", [0.0, 0.0, 0.0])
        rot = node.get("rotation", [0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
        scale = node.get("scale", [1.0, 1.0, 1.0])

        is_deform = node_idx in joint_indices

        joints.append({
            "id": bid,
            "name": name,
            "parent": parent_id,
            "role": "bone",
            "deform": is_deform,
            "restLocal": {
                "translation": [round(v, 6) for v in trans],
                "rotation": [round(v, 6) for v in rot],
                "scale": [round(v, 6) for v in scale],
            },
            "poseDofs": {
                "translation": False,
                "rotation": is_deform,  # only deform bones are model-driven
                "scale": False,
            },
            "limits": {},
            "sprite": None,
        })

        for child_idx in node.get("children", []):
            process_node(child_idx, bid)

    # Start from skeleton root, attached to __motion_root__
    process_node(skeleton_root, "__motion_root__")

    return joints


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--all-model-driven", action="store_true",
                   help="Mark ALL bones as model-driven (not just skin joints)")
    args = p.parse_args()

    gltf = json.loads(Path(args.input).read_text(encoding="utf-8"))
    joints = extract_skeleton(gltf)

    if args.all_model_driven:
        for j in joints:
            if j["id"] != "__motion_root__":
                j["deform"] = True
                j["poseDofs"]["rotation"] = True
        count = sum(1 for j in joints if j["id"] != "__motion_root__")
        print(f"All {count} bones set to model-driven")

    model_driven = sum(1 for j in joints if j["poseDofs"]["rotation"])
    print(f"Total: {len(joints)} joints, {model_driven} model-driven")

    skeleton = {
        "schema": "pet-rig-v2",
        "coordinateSystem": {
            "handedness": "right",
            "up": "+Y",
            "forward": "+X",
        },
        "motionRoot": "__motion_root__",
        "canvas": [48, 48],
        "displayScale": 2,
        "sourceFacing": -1,
        "joints": joints,
        "drawOrder": [j["id"] for j in joints
                      if j["deform"] and j["id"] != "__motion_root__"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(skeleton, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
