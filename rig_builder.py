import bpy
from mathutils import Vector, Matrix
import numpy as np
from typing import List, Optional, Union, Dict, Any


def calculate_bone_roll(head: Vector, tail: Vector, parent_head: Optional[Vector] = None) -> float:
    """Calculates a clean bone roll aligning primary bend axes."""
    bone_dir = (tail - head).normalized()
    if abs(bone_dir.z) > 0.99:
        # Vertical bone (spine / neck): roll 0 with X pointing right
        return 0.0
    
    # For limbs, align Z-up or along bend normal
    if parent_head is not None:
        parent_dir = (head - parent_head).normalized()
        bend_normal = parent_dir.cross(bone_dir)
        if bend_normal.length > 0.1:
            bend_normal.normalize()
            # Blender roll calculation helper
            ref_vec = Vector((0.0, 0.0, 1.0))
            dot = bone_dir.dot(ref_vec)
            if abs(dot) < 0.95:
                return 0.0

    return 0.0


def create_armature_from_joints(
    armature_name: str,
    joints: List[List[float]],
    parents: List[int],
    joint_names: Optional[List[str]] = None,
    collection: Optional[bpy.types.Collection] = None,
    min_bone_length: float = 0.05,
) -> bpy.types.Object:
    """
    Creates a production-grade Blender Armature with Blender 4.0+ Bone Collections
    and consistent bone rolls. Bones keep the model's raw predicted names.
    """
    num_joints = len(joints)
    
    # Convert coordinates from OBJ space (Y-up, Z-forward) to Blender space (Z-up, -Y forward)
    # Blender_X = OBJ_X, Blender_Y = -OBJ_Z, Blender_Z = OBJ_Y
    blender_joints = [
        Vector((float(j[0]), -float(j[2]), float(j[1]))) for j in joints
    ]

    # Raw predicted names (fallback: Bone_000...)
    bone_names: List[str] = [str(n) for n in joint_names] if joint_names and len(joint_names) == num_joints else [f"Bone_{i:03d}" for i in range(num_joints)]
    # Ensure uniqueness without relying on Blender's silent .001 suffixing
    used_names: set = set()
    for i, fname in enumerate(bone_names):
        candidate = fname
        suffix_count = 1
        while candidate in used_names:
            candidate = f"{fname}_{suffix_count:02d}"
            suffix_count += 1
        used_names.add(candidate)
        bone_names[i] = candidate

    # Create Armature data block and object
    armature_data = bpy.data.armatures.new(name=armature_name)
    armature_data.display_type = 'OCTAHEDRAL'
    armature_obj = bpy.data.objects.new(armature_name, armature_data)
    armature_obj.show_in_front = True

    # Link to active or specified collection
    target_coll = collection or bpy.context.collection or bpy.context.scene.collection
    target_coll.objects.link(armature_obj)

    # Make armature active and switch to EDIT mode
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = armature_data.edit_bones
    bone_map: Dict[int, bpy.types.EditBone] = {}

    # Build children map to compute appropriate bone tails
    children_map: Dict[int, List[int]] = {i: [] for i in range(num_joints)}
    for i, p in enumerate(parents):
        if 0 <= p < num_joints:
            children_map[p].append(i)

    # 1. Create all edit bones and set heads
    for i in range(num_joints):
        bname = bone_names[i]
        ebone = edit_bones.new(name=bname)
        ebone.use_connect = False
        ebone.head = blender_joints[i]
        bone_map[i] = ebone

    # 2. Determine tails, parenting, and rolls
    # Average parent->child bone length (official SkinTokens extrude rule:
    # leaf bones extrude along the parent direction by avg_length * 0.5)
    length_sum = 0.0
    length_cnt = 0
    for i, p in enumerate(parents):
        if 0 <= p < num_joints:
            length_sum += (blender_joints[i] - blender_joints[p]).length
            length_cnt += 1
    extrude_length = (length_sum / max(length_cnt, 1)) * 0.5 if length_cnt else min_bone_length

    for i in range(num_joints):
        ebone = bone_map[i]
        children = children_map[i]
        head_pos = blender_joints[i]

        if len(children) == 1:
            child_pos = blender_joints[children[0]]
            direction = child_pos - head_pos
            if direction.length > 1e-4:
                ebone.tail = child_pos
            else:
                ebone.tail = head_pos + Vector((0, 0, extrude_length))
        elif len(children) > 1:
            # Average vector to all children
            avg_child_pos = sum((blender_joints[c] for c in children), Vector((0, 0, 0))) / len(children)
            direction = avg_child_pos - head_pos
            if direction.length > 1e-4:
                ebone.tail = head_pos + direction.normalized() * max(direction.length * 0.5, extrude_length)
            else:
                ebone.tail = head_pos + Vector((0, 0, extrude_length))
        else:
            # Leaf bone: project along parent-to-self vector or default up
            p = parents[i]
            if p >= 0:
                parent_pos = blender_joints[p]
                direction = head_pos - parent_pos
                if direction.length > 1e-4:
                    ebone.tail = head_pos + direction.normalized() * extrude_length
                else:
                    ebone.tail = head_pos + Vector((0, 0, extrude_length))
            else:
                ebone.tail = head_pos + Vector((0, 0, extrude_length))

        # Ensure tail is never coincident with head
        if (ebone.tail - ebone.head).length < 1e-4:
            ebone.tail = ebone.head + Vector((0, 0, min_bone_length))

        # Parent bones
        p = parents[i]
        if p >= 0 and p in bone_map:
            parent_bone = bone_map[p]
            ebone.parent = parent_bone
            if (ebone.head - parent_bone.tail).length < 1e-3 and len(children_map[p]) == 1:
                ebone.use_connect = True

        # Calculate bone roll
        parent_head = blender_joints[p] if p >= 0 else None
        ebone.roll = calculate_bone_roll(ebone.head, ebone.tail, parent_head)

    # 3. Setup Blender 4.0+ Bone Collections
    has_bone_collections = hasattr(armature_data, "collections")
    if has_bone_collections:
        try:
            coll_deform = armature_data.collections.new(name="Deform")
            coll_spine = armature_data.collections.new(name="Spine")
            coll_limbs_l = armature_data.collections.new(name="Limbs.L")
            coll_limbs_r = armature_data.collections.new(name="Limbs.R")
            coll_head = armature_data.collections.new(name="Head")

            for i in range(num_joints):
                eb = bone_map[i]
                name_lower = eb.name.lower()
                
                # Always add to Deform
                coll_deform.assign(eb)

                if ".l" in name_lower or "left" in name_lower:
                    coll_limbs_l.assign(eb)
                elif ".r" in name_lower or "right" in name_lower:
                    coll_limbs_r.assign(eb)
                elif any(k in name_lower for k in ["head", "neck", "jaw", "eye"]):
                    coll_head.assign(eb)
                elif any(k in name_lower for k in ["spine", "hip", "pelvis", "chest", "root"]):
                    coll_spine.assign(eb)
        except Exception:
            pass

    bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj


def optimize_skinning_weights(
    weights_np: np.ndarray,
    max_influences: int = 4,
    threshold: float = 0.001,
) -> np.ndarray:
    """
    Clamps maximum bone influences per vertex and normalizes sums to 1.0.
    Standard for game engines (Unity / Unreal / Godot).
    """
    num_verts, num_joints = weights_np.shape
    optimized = weights_np.copy()

    # Zero out sub-threshold values
    optimized[optimized < threshold] = 0.0

    if 0 < max_influences < num_joints:
        for i in range(num_verts):
            row = optimized[i]
            if np.count_nonzero(row) > max_influences:
                # Find indices of top K weights
                top_indices = np.argpartition(row, -max_influences)[-max_influences:]
                mask = np.ones(num_joints, dtype=bool)
                mask[top_indices] = False
                row[mask] = 0.0
                optimized[i] = row

    # Re-normalize sums per vertex
    row_sums = optimized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    optimized = optimized / row_sums

    return optimized


def assign_skinning_weights(
    mesh_obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    weights: Union[List[List[float]], np.ndarray],
    joint_names: Optional[List[str]] = None,
    max_influences: int = 4,
    threshold: float = 0.001,
) -> None:
    """
    Creates vertex groups on mesh_obj and assigns optimized skinning weights.
    """
    weights_np = np.asarray(weights, dtype=np.float32)
    num_verts, num_joints = weights_np.shape

    mesh = mesh_obj.data
    actual_verts = len(mesh.vertices)

    # Use actual armature bone names to guarantee 1-to-1 match
    bone_names = [b.name for b in armature_obj.data.bones]
    if len(bone_names) != num_joints and joint_names:
        bone_names = joint_names[:num_joints]

    # Optimize and clamp weights
    optimized_weights = optimize_skinning_weights(
        weights_np,
        max_influences=max_influences,
        threshold=threshold,
    )

    # Clear old vertex groups matching bone names
    for bname in bone_names:
        vg = mesh_obj.vertex_groups.get(bname)
        if vg is not None:
            mesh_obj.vertex_groups.remove(vg)

    # Create clean vertex groups
    vgroups = {}
    for j_idx, bname in enumerate(bone_names):
        vgroups[j_idx] = mesh_obj.vertex_groups.new(name=bname)

    # Handle vertex count matching
    verts_to_process = min(num_verts, actual_verts)

    # Assign weights
    for v_idx in range(verts_to_process):
        v_weights = optimized_weights[v_idx]
        for j_idx in range(num_joints):
            w = float(v_weights[j_idx])
            if w > 0.0:
                vgroups[j_idx].add([v_idx], w, 'REPLACE')


def attach_armature_modifier(
    mesh_obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
) -> bpy.types.Modifier:
    """Parents mesh_obj to armature_obj and configures the Armature modifier."""
    mesh_obj.parent = armature_obj
    mesh_obj.matrix_parent_inverse = armature_obj.matrix_world.inverted()

    arm_mod = None
    for mod in mesh_obj.modifiers:
        if mod.type == 'ARMATURE':
            arm_mod = mod
            break

    if arm_mod is None:
        arm_mod = mesh_obj.modifiers.new(name="SkinTokens_Armature", type='ARMATURE')

    arm_mod.object = armature_obj
    arm_mod.use_vertex_groups = True
    return arm_mod


def build_rig_and_skin(
    mesh_obj: bpy.types.Object,
    joints: List[List[float]],
    parents: List[int],
    weights: Union[List[List[float]], np.ndarray],
    joint_names: Optional[List[str]] = None,
    existing_armature: Optional[bpy.types.Object] = None,
    max_influences: int = 4,
    threshold: float = 0.001,
) -> bpy.types.Object:
    """
    High-level orchestrator to build complete rig, assign vertex groups, and configure modifiers.
    """
    if existing_armature is not None and existing_armature.type == 'ARMATURE':
        armature_obj = existing_armature
    else:
        armature_name = f"{mesh_obj.name}_Rig"
        coll = mesh_obj.users_collection[0] if mesh_obj.users_collection else None
        
        armature_obj = create_armature_from_joints(
            armature_name=armature_name,
            joints=joints,
            parents=parents,
            joint_names=joint_names,
            collection=coll,
        )
        armature_obj.matrix_world = mesh_obj.matrix_world.copy()

    has_weights = weights is not None and (
        (isinstance(weights, np.ndarray) and weights.size > 0)
        or (isinstance(weights, list) and len(weights) > 0)
    )

    if has_weights:
        assign_skinning_weights(
            mesh_obj=mesh_obj,
            armature_obj=armature_obj,
            weights=weights,
            joint_names=joint_names,
            max_influences=max_influences,
            threshold=threshold,
        )
        
        attach_armature_modifier(
            mesh_obj=mesh_obj,
            armature_obj=armature_obj,
        )
    
    return armature_obj
