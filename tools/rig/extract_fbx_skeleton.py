"""Extract skeleton hierarchy from FBX and write cat-skeleton-3d.json.

Usage:
  blender --background --python tools/rig/extract_fbx_skeleton.py -- \
    --input assets/Miyabi/17808/03_从原始Max文件中导出的Fbx/Tpose.FBX \
    --output assets/pet/runtime/cat-skeleton-3d.json
"""

import bpy
import json
import math
import sys
import argparse
from pathlib import Path


def parse_args():
    # Blender passes its own args first; ours come after "--"
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="FBX file to read")
    p.add_argument("--output", required=True, help="Output cat-skeleton-3d.json")
    p.add_argument("--armature-name", default=None, help="Armature object name (auto-detect if blank)")
    return p.parse_args(argv)


def quat_to_list(q):
    """Blender quaternion (w, x, y, z) -> our format [x, y, z, w]."""
    return [round(q.x, 6), round(q.y, 6), round(q.z, 6), round(q.w, 6)]


def vec_to_list(v):
    return [round(v.x, 6), round(v.y, 6), round(v.z, 6)]


def find_armature():
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    return None


def extract_skeleton(armature_obj):
    """Walk the armature and extract joint hierarchy."""
    joints = []
    id_counter = [0]

    def bone_id(name):
        # Sanitize name for our ID pattern: lowercase alphanumeric + underscore
        safe = "".join(c if c.isalnum() or c == '_' else '_' for c in name.lower())
        if not safe or safe[0].isdigit():
            safe = "bone_" + safe
        return safe

    def process_bone(edit_bone, parent_id):
        bid = bone_id(edit_bone.name)
        id_counter[0] += 1

        # Blender edit_bone.matrix is the rest pose transform in armature space.
        # For a child bone, we need the LOCAL transform relative to parent.
        if edit_bone.parent:
            parent_mat = edit_bone.parent.matrix.copy()
            local_mat = parent_mat.inverted() @ edit_bone.matrix
        else:
            local_mat = edit_bone.matrix.copy()

        trans = local_mat.to_translation()
        rot = local_mat.to_quaternion()  # returns (w, x, y, z)

        joints.append({
            "id": bid,
            "name": edit_bone.name,
            "parent": parent_id,
            "role": "bone",
            "deform": True,
            "restLocal": {
                "translation": vec_to_list(trans),
                "rotation": quat_to_list(rot),
                "scale": [1, 1, 1],
            },
            "poseDofs": {
                "translation": False,
                "rotation": True,
                "scale": False,
            },
            "limits": {},
            "sprite": None,
        })

        for child in edit_bone.children:
            process_bone(child, bid)

    # Find root bones (bones with no parent)
    roots = [b for b in armature_obj.data.edit_bones if b.parent is None]

    # Insert __motion_root__
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

    # Process actual bones as children of __motion_root__
    for root in roots:
        process_bone(root, "__motion_root__")

    return joints


def compute_draw_order(joints):
    """Simple back-to-front: longer bones later (approximation)."""
    return [j["id"] for j in joints if j["sprite"] is not None or j["role"] == "bone"]


def main():
    args = parse_args()

    # Clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import FBX
    print(f"Importing {args.input}...")
    bpy.ops.import_scene.fbx(filepath=str(Path(args.input).resolve()))

    # Find armature
    armature = find_armature() if not args.armature_name else bpy.data.objects.get(args.armature_name)
    if not armature:
        print("ERROR: No armature found in the FBX file.")
        print("Objects in scene:", [o.name for o in bpy.data.objects])
        return 1

    print(f"Found armature: {armature.name}")

    # Enter edit mode to read bone hierarchy
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')

    joints = extract_skeleton(armature)

    bpy.ops.object.mode_set(mode='OBJECT')

    # Build output
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
        "drawOrder": compute_draw_order(joints),
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(skeleton, indent=2, ensure_ascii=False), encoding="utf-8")

    model_driven = sum(1 for j in joints if j["id"] != "__motion_root__" and j["deform"])
    print(f"Done: {len(joints)} joints ({model_driven} model-driven), written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
