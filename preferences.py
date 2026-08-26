import bpy
import os
import shutil
import sys
from pathlib import Path


def get_default_python_path() -> str:
    """Finds a default virtualenv Python or system Python."""
    # Check if a dedicated virtualenv exists in the skintokens cache directory
    venv_py = Path.home() / ".cache" / "skintokens" / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
    if venv_py.exists():
        return str(venv_py)

    # Check current system python3 or sys.executable
    if shutil.which("python3"):
        return shutil.which("python3") or sys.executable
    return sys.executable


def get_default_checkpoint_dir() -> str:
    return str(Path.home() / ".cache" / "skintokens" / "checkpoints")


class SkinTokensAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or "skintokens_blender"

    python_path: bpy.props.StringProperty(
        name="Python Executable",
        description="Path to Python executable with PyTorch, transformers, einops installed",
        default=get_default_python_path(),
        subtype='FILE_PATH',
    ) # type: ignore

    checkpoint_dir: bpy.props.StringProperty(
        name="Checkpoints Directory",
        description="Directory where SkinTokens model checkpoints are stored",
        default=get_default_checkpoint_dir(),
        subtype='DIR_PATH',
    ) # type: ignore

    hf_repo: bpy.props.StringProperty(
        name="HuggingFace Repo ID",
        description="Repository ID for SkinTokens checkpoints",
        default="VAST-AI/SkinTokens",
    ) # type: ignore

    last_env_status: bpy.props.StringProperty(
        name="Environment Status",
        default="Not Verified",
    ) # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Python AI Backend & Dependencies", icon='SETTINGS')
        box.prop(self, "python_path")
        box.prop(self, "checkpoint_dir")
        box.prop(self, "hf_repo")

        # Environment setup & status
        col = box.column(align=True)
        row_status = col.row()
        row_status.label(text=f"Status: {self.last_env_status}", icon='INFO')

        row_ops = col.row(align=True)
        row_ops.operator("skintokens.setup_env", text="Auto-Setup Venv & Dependencies", icon='COMMUNITY')
        row_ops.operator("skintokens.check_env", text="Verify Python Environment", icon='CHECKMARK')

        box_ckpt = layout.box()
        box_ckpt.label(text="Model Checkpoints", icon='FILE_CACHE')
        row_ckpt = box_ckpt.row(align=True)
        row_ckpt.operator("skintokens.download_ckpt", text="Download Weights from Hugging Face", icon='IMPORT')


def register():
    bpy.utils.register_class(SkinTokensAddonPreferences)


def unregister():
    bpy.utils.unregister_class(SkinTokensAddonPreferences)
