import bpy


class SkinTokensSceneSettings(bpy.types.PropertyGroup):
    target_armature: bpy.props.PointerProperty(
        name="Target Armature",
        description="Optional existing armature to predict skinning weights for",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE',
    ) # type: ignore

    naming_style: bpy.props.EnumProperty(
        name="Bone Naming",
        description="Rename bones to a standard skeleton convention (topology-based detection)",
        items=[
            ("ARTICULATED", "Articulated (Raw)", "Keep the model's raw predicted bone names"),
            ("MIXAMO", "Mixamo", "Hips / Spine / LeftArm ... (Mixamo retarget-ready)"),
            ("UE5", "UE5", "pelvis / spine_01 / upperarm_l ... + root bone (Unreal-ready)"),
        ],
        default="ARTICULATED",
    ) # type: ignore

    max_influences: bpy.props.EnumProperty(
        name="Max Influences",
        description="Maximum number of bone influences per vertex (standard for game engines)",
        items=[
            ("4", "4 Bones", "Limit to 4 strongest bone influences per vertex"),
            ("8", "8 Bones", "Limit to 8 strongest bone influences per vertex"),
            ("0", "Unlimited", "No limit on bone influences per vertex"),
        ],
        default="4",
    ) # type: ignore

    weight_threshold: bpy.props.FloatProperty(
        name="Weight Threshold",
        description="Minimum weight threshold (weights below this are pruned)",
        default=0.001,
        min=0.0,
        max=0.1,
        precision=4,
    ) # type: ignore

    use_postprocess: bpy.props.BoolProperty(
        name="Voxel Skin Smoothing",
        description="Optional voxel skin smoothing (official demo default: off; can sharpen or smear weights depending on the mesh)",
        default=False,
    ) # type: ignore

    top_k: bpy.props.IntProperty(
        name="Top K",
        description="Top-K sampling cutoff (official SkinTokens demo default: 5)",
        default=5,
        min=1,
        max=200,
    ) # type: ignore

    top_p: bpy.props.FloatProperty(
        name="Top P",
        description="Nucleus sampling probability cutoff",
        default=0.95,
        min=0.0,
        max=1.0,
    ) # type: ignore

    temperature: bpy.props.FloatProperty(
        name="Temperature",
        description="Sampling temperature (official demo default: 1.0)",
        default=1.0,
        min=0.01,
        max=2.0,
    ) # type: ignore

    repetition_penalty: bpy.props.FloatProperty(
        name="Repetition Penalty",
        description="Penalty for repetitive tokens (official demo default: 2.0)",
        default=2.0,
        min=0.5,
        max=3.0,
    ) # type: ignore

    num_beams: bpy.props.IntProperty(
        name="Num Beams",
        description="Beam search paths (official demo default: 10; higher = slower but cleaner rigs)",
        default=10,
        min=1,
        max=20,
    ) # type: ignore

    is_running: bpy.props.BoolProperty(
        name="Is Running",
        description="Whether a SkinTokens task is currently running in the background",
        default=False,
    ) # type: ignore

    progress_percent: bpy.props.IntProperty(
        name="Progress",
        description="Current task progress percentage",
        default=0,
        min=0,
        max=100,
    ) # type: ignore

    progress_status: bpy.props.StringProperty(
        name="Status",
        description="Current task status description",
        default="",
    ) # type: ignore


class VIEW3D_PT_skintokens(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SkinTokens'
    bl_label = 'SkinTokens Auto-Rig'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.skintokens_settings
        active_obj = context.active_object

        # Target Mesh Box
        box_mesh = layout.box()
        box_mesh.label(text="Target Mesh", icon='MESH_DATA')
        if active_obj and active_obj.type == 'MESH':
            row = box_mesh.row()
            row.label(text=f"Selected: {active_obj.name}")
            row.label(text=f"{len(active_obj.data.vertices):,} verts")
        else:
            box_mesh.label(text="No Mesh Selected (Select a Mesh)", icon='ERROR')

        # Target Existing Armature (Optional)
        box_arm = layout.box()
        box_arm.label(text="Existing Skeleton (Optional)", icon='ARMATURE_DATA')
        box_arm.prop_search(settings, "target_armature", scene, "objects", text="Armature")

        # Live Progress Box when a task is running
        if settings.is_running:
            pbox = layout.box()
            pbox.label(text=settings.progress_status or "Processing...", icon='TIME')
            prow = pbox.row(align=True)
            prow.prop(settings, "progress_percent", text="", slider=True)
            prow.enabled = False

        # Main Action Buttons
        col = layout.column(align=True)
        if settings.is_running:
            col.enabled = False
            col.scale_y = 1.6
            col.operator("skintokens.auto_rig", text=f"Processing... {settings.progress_percent}%", icon='TIME')
        else:
            col.scale_y = 1.4
            if settings.target_armature is not None:
                op_skin_custom = col.operator("skintokens.auto_rig", text="Skin with Selected Armature", icon='MOD_ARMATURE')
                op_skin_custom.use_existing_skeleton = True
                op_skin_custom.skeleton_only = False
            else:
                op_full = col.operator("skintokens.auto_rig", text="Auto-Rig & Skin Mesh", icon='ARMATURE_DATA')
                op_full.use_existing_skeleton = False
                op_full.skeleton_only = False
            
            op_skel = col.operator("skintokens.auto_rig", text="Generate Skeleton Only", icon='BONE_DATA')
            op_skel.use_existing_skeleton = False
            op_skel.skeleton_only = True

        # Production Rigging & Weight Settings
        box_rig = layout.box()
        box_rig.label(text="Rigging & Weights", icon='PREFERENCES')
        box_rig.prop(settings, "naming_style")
        box_rig.prop(settings, "max_influences")
        box_rig.prop(settings, "weight_threshold")
        box_rig.prop(settings, "use_postprocess")

        # AI Sampling Parameters
        box_ai = layout.box()
        box_ai.label(text="Sampling Parameters", icon='MODIFIER')
        box_ai.prop(settings, "temperature")
        box_ai.prop(settings, "top_p")
        box_ai.prop(settings, "top_k")
        box_ai.prop(settings, "repetition_penalty")
        box_ai.prop(settings, "num_beams")

        # Checkpoints & Utilities
        box_util = layout.box()
        box_util.label(text="Environment & Setup", icon='FILE_CACHE')
        row_util = box_util.row(align=True)
        if settings.is_running:
            row_util.enabled = False
        row_util.operator("skintokens.setup_env", text="Setup Venv", icon='COMMUNITY')
        row_util.operator("skintokens.download_ckpt", text="Download Weights", icon='IMPORT')


def register():
    bpy.utils.register_class(SkinTokensSceneSettings)
    bpy.types.Scene.skintokens_settings = bpy.props.PointerProperty(type=SkinTokensSceneSettings)
    bpy.utils.register_class(VIEW3D_PT_skintokens)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_skintokens)
    if hasattr(bpy.types.Scene, "skintokens_settings"):
        del bpy.types.Scene.skintokens_settings
    bpy.utils.unregister_class(SkinTokensSceneSettings)
