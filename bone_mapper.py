"""Topology-based standard-skeleton renaming, adopted from ComfyUI-SkinTokens nodes.py.

Detects pelvis / spine / neck / arms / legs / fingers from the bone HIERARCHY
(child counts, chain walks, x-sorted sides) and renames bones + vertex groups to
the UE5 or Mixamo standard conventions. 'Articulated' keeps raw predicted names.
"""
import bpy
from mathutils import Matrix
from typing import Dict, Optional, List


def rename_skeleton_standard(armature_obj: bpy.types.Object, convention: str) -> Dict[str, str]:
    """Rename armature bones + mesh vertex groups to a standard rig convention.

    convention: 'Articulated' (no-op), 'Mixamo', or 'UE5'.
    Returns the applied {old_name: new_name} map.
    """
    if convention not in ("UE5", "Mixamo"):
        return {}

    armature = armature_obj

    class BoneNode:
        def __init__(self, bpy_bone, parent=None):
            self.bpy_bone = bpy_bone
            self.name = bpy_bone.name
            self.parent = parent
            self.children = []
            parent_matrix = bpy_bone.parent.matrix_local if bpy_bone.parent else Matrix.Identity(4)
            local_matrix = parent_matrix.inverted() @ bpy_bone.matrix_local
            self.t = local_matrix.to_translation()
            self.head = bpy_bone.head

    def build_tree(bpy_bone, parent_node=None):
        node = BoneNode(bpy_bone, parent_node)
        for child in bpy_bone.children:
            child_node = build_tree(child, node)
            node.children.append(child_node)
        return node

    roots = [b for b in armature.data.bones if b.parent is None]
    if not roots:
        return {}

    root_nodes = [build_tree(r) for r in roots]
    all_nodes: List[BoneNode] = []

    def get_all_nodes(node):
        all_nodes.append(node)
        for child in node.children:
            get_all_nodes(child)
    for r in root_nodes:
        get_all_nodes(r)

    def count_descendants(node):
        return sum(1 + count_descendants(c) for c in node.children)

    pelvis = None
    for node in all_nodes:
        if len(node.children) >= 3:
            pelvis = node
            break
    if not pelvis:
        pelvis = all_nodes[0]

    rename_map = {}
    if convention == "UE5":
        rename_map[pelvis.name] = "pelvis"
    elif convention == "Mixamo":
        rename_map[pelvis.name] = "Hips"

    pelvis_children = pelvis.children
    pelvis_children_sorted_by_size = sorted(pelvis_children, key=count_descendants, reverse=True)
    spine_root = pelvis_children_sorted_by_size[0]
    remaining_legs = pelvis_children_sorted_by_size[1:]

    legs_sorted_x = sorted(remaining_legs, key=lambda c: c.head.x, reverse=True)
    thigh_l = legs_sorted_x[0] if len(legs_sorted_x) > 0 else None
    thigh_r = legs_sorted_x[1] if len(legs_sorted_x) > 1 else None

    current_spine = spine_root
    spine_idx = 1
    chest_split = None
    while current_spine:
        if convention == "UE5":
            rename_map[current_spine.name] = f"spine_{spine_idx:02d}"
        elif convention == "Mixamo":
            rename_map[current_spine.name] = "Spine" if spine_idx == 1 else f"Spine{spine_idx - 1}"

        if len(current_spine.children) >= 3:
            chest_split = current_spine
            break
        elif len(current_spine.children) == 1:
            current_spine = current_spine.children[0]
            spine_idx += 1
        else:
            break

    if chest_split:
        chest_children = chest_split.children
        chest_children_sorted_by_size = sorted(chest_children, key=count_descendants)
        neck_root = chest_children_sorted_by_size[0]
        remaining_arms = chest_children_sorted_by_size[1:]
        arms_sorted_x = sorted(remaining_arms, key=lambda c: c.head.x, reverse=True)
        clavicle_l = arms_sorted_x[0] if len(arms_sorted_x) > 0 else None
        clavicle_r = arms_sorted_x[1] if len(arms_sorted_x) > 1 else None

        current_neck = neck_root
        neck_idx = 1
        while current_neck:
            if len(current_neck.children) == 0:
                rename_map[current_neck.name] = "head" if convention == "UE5" else "Head"
                break
            else:
                if convention == "UE5":
                    rename_map[current_neck.name] = f"neck_{neck_idx:02d}"
                elif convention == "Mixamo":
                    rename_map[current_neck.name] = "Neck" if neck_idx == 1 else f"Neck{neck_idx - 1}"
                current_neck = current_neck.children[0]
                neck_idx += 1

        def rename_arm_chain(start_node, suffix):
            side_prefix = "Left" if suffix == "l" else "Right"
            if convention == "UE5":
                rename_map[start_node.name] = f"clavicle_{suffix}"
            elif convention == "Mixamo":
                rename_map[start_node.name] = f"{side_prefix}Shoulder"

            curr = start_node.children[0] if start_node.children else None
            if curr:
                if convention == "UE5":
                    rename_map[curr.name] = f"upperarm_{suffix}"
                elif convention == "Mixamo":
                    rename_map[curr.name] = f"{side_prefix}Arm"
                curr = curr.children[0] if curr.children else None
            if curr:
                if convention == "UE5":
                    rename_map[curr.name] = f"lowerarm_{suffix}"
                elif convention == "Mixamo":
                    rename_map[curr.name] = f"{side_prefix}ForeArm"
                curr = curr.children[0] if curr.children else None
            if curr:
                if convention == "UE5":
                    rename_map[curr.name] = f"hand_{suffix}"
                elif convention == "Mixamo":
                    rename_map[curr.name] = f"{side_prefix}Hand"
                fingers = curr.children
                if len(fingers) > 0:
                    thumb_node = min(fingers, key=lambda f: f.head.y)
                    other_fingers = [f for f in fingers if f != thumb_node]
                    if suffix == "l":
                        other_fingers_sorted = sorted(other_fingers, key=lambda f: f.head.x, reverse=True)
                    else:
                        other_fingers_sorted = sorted(other_fingers, key=lambda f: f.head.x, reverse=False)
                    fingers_sorted = [thumb_node] + other_fingers_sorted
                else:
                    fingers_sorted = []

                finger_names_ue5 = ["thumb", "index", "middle", "ring", "pinky"]
                finger_names_mixamo = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                if len(fingers_sorted) == 3:
                    finger_names_ue5 = ["thumb", "index", "pinky"]
                    finger_names_mixamo = ["Thumb", "Index", "Pinky"]

                for f_idx, finger in enumerate(fingers_sorted):
                    curr_finger = finger
                    joint_idx = 1
                    while curr_finger:
                        if convention == "UE5":
                            name_prefix = finger_names_ue5[f_idx] if f_idx < len(finger_names_ue5) else f"finger_{f_idx}"
                            rename_map[curr_finger.name] = f"{name_prefix}_{joint_idx:02d}_{suffix}"
                        elif convention == "Mixamo":
                            name_prefix = finger_names_mixamo[f_idx] if f_idx < len(finger_names_mixamo) else f"Finger{f_idx}"
                            rename_map[curr_finger.name] = f"{side_prefix}Hand{name_prefix}{joint_idx}"
                        curr_finger = curr_finger.children[0] if curr_finger.children else None
                        joint_idx += 1

        if clavicle_l:
            rename_arm_chain(clavicle_l, "l")
        if clavicle_r:
            rename_arm_chain(clavicle_r, "r")

    def rename_leg_chain(start_node, suffix):
        side_prefix = "Left" if suffix == "l" else "Right"
        if convention == "UE5":
            rename_map[start_node.name] = f"thigh_{suffix}"
        elif convention == "Mixamo":
            rename_map[start_node.name] = f"{side_prefix}UpLeg"

        curr = start_node.children[0] if start_node.children else None
        if curr:
            if convention == "UE5":
                rename_map[curr.name] = f"calf_{suffix}"
            elif convention == "Mixamo":
                rename_map[curr.name] = f"{side_prefix}Leg"
            curr = curr.children[0] if curr.children else None
        if curr:
            if convention == "UE5":
                rename_map[curr.name] = f"foot_{suffix}"
            elif convention == "Mixamo":
                rename_map[curr.name] = f"{side_prefix}Foot"
            curr = curr.children[0] if curr.children else None
        if curr:
            if convention == "UE5":
                rename_map[curr.name] = f"ball_{suffix}"
            elif convention == "Mixamo":
                rename_map[curr.name] = f"{side_prefix}ToeBase"

    if thigh_l:
        rename_leg_chain(thigh_l, "l")
    if thigh_r:
        rename_leg_chain(thigh_r, "r")

    # Apply renames: edit bones first, then mesh vertex groups
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    for orig, new in rename_map.items():
        if orig in armature.data.edit_bones:
            armature.data.edit_bones[orig].name = new

    # Inject root bone ONLY for UE5
    if convention == "UE5" and "pelvis" in armature.data.edit_bones:
        root_bone = armature.data.edit_bones.new("root")
        root_bone.head = (0.0, 0.0, 0.0)
        root_bone.tail = (0.0, 0.0, 0.1)
        armature.data.edit_bones["pelvis"].parent = root_bone

    bpy.ops.object.mode_set(mode='OBJECT')

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for orig, new in rename_map.items():
                if orig in obj.vertex_groups:
                    obj.vertex_groups[orig].name = new

    return rename_map
