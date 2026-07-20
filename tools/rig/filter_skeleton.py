"""Filter a pet-rig-v2 skeleton to keep only core deform bones for 2.5D use.

Removes:
  - IK controllers and helpers (knee_*, jh_*, *_adv)
  - Twist/fix bones (*_twis*, *_fix*, thigh_fix*)
  - Finger bones (bip001_l_finger*, bip001_r_finger*, bip001_l_hand*_0*)
  - Facial bones, prop bones, weapon bones
  - Non-deforming helper chains

Keeps:
  - pelvis, spine chain, neck, head
  - clavicles, upper arms, forearms, hands
  - thighs, calves, feet, toes
  - Any tail or unique prop (ponytail, weapon)

Usage:
  python tools/rig/filter_skeleton.py --input cat-skeleton-3d.json --output cat-skeleton-3d.json
"""

import argparse
import json
import re
from pathlib import Path


# Bone name patterns to EXCLUDE from model-driven list (still kept in hierarchy)
EXCLUDE_PATTERNS = [
    r"twist",
    r"twis",            # twis (abbreviated twist in Bip001)
    r"fix",
    r"_adv",
    r"^knee_",
    r"^jh_",
    r"^elbow",
    r"^bn_elbow",
    r"^ankle_",
    r"^wrist_",
    r"_finger",
    r"_thumb",
    r"_index",
    r"_middle",
    r"_ring",
    r"_pinky",
    r"_hand.*_[0-9]+$",
    r"_toe[1-9]",
    r"bip001_[lr]_hand$",
    r"_prop",
    r"_weapon",
    r"hair",
    r"^bonename",
    r"^bone\d+$",       # placeholder bones (bone010, bone015)
    r"^skirt",
    r"^coat",
    r"^sleeve",         # sleeve cloth (sleevedown, bn_sleeveup)
    r"^bn_sleeve",
    r"face",
    r"jaw",
    r"eye",
    r"eyebrow",
    r"eyelid",
    r"tongue",
    r"_scarf",
    r"ear\d",
    r"pony",
    r"breast",
    r"^chest_",         # chest physics
    r"^mouth",          # mouth bones
    r"^majia",          # kimono cloth (majia1_01, etc.)
    r"^furisode",       # kimono sleeve cloth
    r"^bn_katana",      # katana weapon
    r"^bn_scabbard",    # scabbard
    r"^pelvis_[lr]",    # pelvis helper bones
    r"^ld_",            # unknown helper chain
    r"^pds_",           # unknown helper chain
]


def should_keep_as_model_driven(bone_id: str, bone_name: str) -> bool:
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, bone_id, re.IGNORECASE):
            return False
        if bone_name and re.search(pattern, bone_name, re.IGNORECASE):
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    sk = json.loads(Path(args.input).read_text(encoding="utf-8"))
    joints = sk["joints"]

    kept = 0
    excluded = 0
    for j in joints:
        if j["id"] == "__motion_root__":
            continue
        name = j.get("name", "")
        if should_keep_as_model_driven(j["id"], name):
            j["poseDofs"]["rotation"] = True
            j["deform"] = True
            kept += 1
        else:
            j["poseDofs"]["rotation"] = False
            j["deform"] = False
            excluded += 1

    print(f"Kept {kept} model-driven bones, excluded {excluded}")

    # Update drawOrder to only include bones with sprites or model-driven
    sk["drawOrder"] = [j["id"] for j in joints
                       if j["deform"] and j["id"] != "__motion_root__"]

    Path(args.output).write_text(
        json.dumps(sk, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
