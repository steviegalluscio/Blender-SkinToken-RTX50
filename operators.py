import bpy
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, List, Tuple

from . import rig_builder

PROGRESS_PATTERN = re.compile(r"\[PROGRESS:(\d+)\]\s*(.*)")


def get_addon_prefs(context: bpy.types.Context):
    pkg_name = __package__ or "skintokens_blender"
    return context.preferences.addons.get(pkg_name, None)


def export_temp_obj(mesh_obj: bpy.types.Object) -> Tuple[str, str]:
    """Exports active mesh object to a temporary OBJ with exact 1-to-1 vertex indexing."""
    temp_dir = tempfile.mkdtemp(prefix="skintokens_")
    out_path = os.path.join(temp_dir, "input_mesh.obj")

    mesh = mesh_obj.data

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# SkinTokens Native Mesh Export for {mesh_obj.name}\n")
        
        # Write vertices in mesh local coordinates mapped to OBJ space (X=right, Y=up, Z=-forward)
        for v in mesh.vertices:
            f.write(f"v {v.co.x:.6f} {v.co.z:.6f} {-v.co.y:.6f}\n")

        # Write polygon faces
        for poly in mesh.polygons:
            indices = [str(idx + 1) for idx in poly.vertices]
            if len(indices) >= 3:
                f.write(f"f {' '.join(indices)}\n")

    return temp_dir, out_path


def export_armature_json(armature_obj: bpy.types.Object, out_path: str):
    """Exports armature bones and hierarchy to JSON in OBJ coordinate space."""
    bones = armature_obj.data.bones
    bone_names = [b.name for b in bones]
    name_to_idx = {name: i for i, name in enumerate(bone_names)}

    joints = []
    parents = []

    for i, b in enumerate(bones):
        head = b.head_local
        # Convert Blender local coords (X, Y, Z) to OBJ space (X, Z, -Y)
        joints.append([float(head.x), float(head.z), -float(head.y)])

        if b.parent is not None and b.parent.name in name_to_idx:
            parents.append(name_to_idx[b.parent.name])
        else:
            parents.append(-1 if i > 0 else None)

    payload = {
        "joints": joints,
        "parents": parents,
        "joint_names": bone_names,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


class AsyncProcessReader:
    """Reads stdout and stderr from a subprocess asynchronously in background threads.
    Provides non-blocking access to progress updates and stderr output across Windows, Linux, and macOS.
    """

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self._queue: queue.Queue = queue.Queue()
        self._stderr_lines: List[str] = []

        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self):
        try:
            if self.proc.stdout is not None:
                for line in iter(self.proc.stdout.readline, ""):
                    if not line:
                        break
                    self._queue.put(line)
        except Exception:
            pass
        finally:
            try:
                if self.proc.stdout is not None:
                    self.proc.stdout.close()
            except Exception:
                pass

    def _read_stderr(self):
        try:
            if self.proc.stderr is not None:
                for line in iter(self.proc.stderr.readline, ""):
                    if not line:
                        break
                    self._stderr_lines.append(line)
        except Exception:
            pass
        finally:
            try:
                if self.proc.stderr is not None:
                    self.proc.stderr.close()
            except Exception:
                pass

    def get_updates(self) -> List[Tuple[int, str]]:
        updates = []
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                break
            m = PROGRESS_PATTERN.search(line)
            if m:
                pct = int(m.group(1))
                msg = m.group(2).strip()
                updates.append((pct, msg))
        return updates

    def get_stderr(self) -> str:
        return "".join(self._stderr_lines).strip()


class SKINTOKENS_OT_check_env(bpy.types.Operator):
    bl_idname = "skintokens.check_env"
    bl_label = "Check Environment"
    bl_description = "Check if Python executable and required dependencies are available"

    def execute(self, context):
        prefs = get_addon_prefs(context)
        py_path = prefs.preferences.python_path if prefs else sys.executable

        code = """import sys, torch, transformers, einops, huggingface_hub
backends = []
gpu = 'None'
try:
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        backends.append('CUDA/HIP')
except Exception:
    pass
try:
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        gpu = 'Apple Silicon'
        backends.append('MPS')
except Exception:
    pass
try:
    if getattr(torch, 'xpu', None) is not None and torch.xpu.is_available():
        gpu = torch.xpu.get_device_name(0)
        backends.append('XPU')
except Exception:
    pass
backend = '+'.join(backends) or 'CPU only'
print(f'PyTorch {torch.__version__} | {backend} ({gpu}) | transformers {transformers.__version__}')"""
        try:
            res = subprocess.run(
                [py_path, "-c", code],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if res.returncode == 0:
                info = res.stdout.strip()
                if prefs:
                    prefs.preferences.last_env_status = f"Ready ({info})"
                self.report({'INFO'}, f"Environment Valid: {info}")
            else:
                err = res.stderr.strip()
                if prefs:
                    prefs.preferences.last_env_status = "Dependencies Missing"
                self.report({'ERROR'}, f"Verification failed:\n{err}\nTip: Run 'Auto-Setup Venv' in Preferences.")
        except Exception as e:
            if prefs:
                prefs.preferences.last_env_status = f"Execution Failed: {e}"
            self.report({'ERROR'}, f"Failed to execute Python at '{py_path}': {e}")

        return {'FINISHED'}


class SKINTOKENS_OT_setup_env(bpy.types.Operator):
    bl_idname = "skintokens.setup_env"
    bl_label = "Auto-Setup Environment"
    bl_description = "Create isolated virtualenv and install minimal dependencies automatically"

    _proc: Optional[subprocess.Popen] = None
    _reader: Optional[AsyncProcessReader] = None
    _timer = None
    _venv_py: str = ""

    def modal(self, context, event):
        wm = context.window_manager
        settings = context.scene.skintokens_settings

        if event.type == 'ESC':
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            self.cancel(context)
            self.report({'WARNING'}, "Venv setup cancelled.")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self._proc is None:
                return {'PASS_THROUGH'}

            ret = self._proc.poll()
            if ret is not None:
                if self._timer is not None:
                    wm.event_timer_remove(self._timer)
                    self._timer = None
                wm.progress_end()
                context.workspace.status_text_set(None)
                settings.is_running = False
                settings.progress_percent = 100

                for area in context.screen.areas:
                    area.tag_redraw()

                if ret == 0 and os.path.exists(self._venv_py):
                    prefs = get_addon_prefs(context)
                    if prefs:
                        prefs.preferences.python_path = self._venv_py
                        prefs.preferences.last_env_status = "Venv Configured Successfully"
                    self.report({'INFO'}, f"SkinTokens Venv successfully created at: {self._venv_py}")
                    return {'FINISHED'}
                else:
                    err = self._reader.get_stderr() if self._reader else "Unknown venv creation failure."
                    self.report({'ERROR'}, f"Setup failed (code {ret}):\n{err}")
                    return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        settings = context.scene.skintokens_settings
        cache_dir = Path.home() / ".cache" / "skintokens" / "venv"
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        
        self._venv_py = str(cache_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python3"))
        
        # Setup bash/cmd script to create venv and pip install
        sys_py = shutil.which("python3") or sys.executable
        pip_cmd = (
            f'"{sys_py}" -m venv "{cache_dir}" && '
            f'"{self._venv_py}" -m pip install --upgrade pip && '
            f'"{self._venv_py}" -m pip install '
            f'torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && '
            f'"{self._venv_py}" -m pip install '
            f'transformers einops huggingface_hub scipy numpy'
        )

        try:
            self._proc = subprocess.Popen(
                pip_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._reader = AsyncProcessReader(self._proc)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start venv installer: {e}")
            return {'CANCELLED'}

        settings.is_running = True
        settings.progress_percent = 20
        settings.progress_status = "Creating Venv & Installing Dependencies..."

        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        self.report({'INFO'}, "Installing SkinTokens dependencies in background (takes 1-2 mins)...")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        context.workspace.status_text_set(None)
        context.scene.skintokens_settings.is_running = False


class SKINTOKENS_OT_download_ckpt(bpy.types.Operator):
    bl_idname = "skintokens.download_ckpt"
    bl_label = "Download Checkpoints"
    bl_description = "Download SkinTokens model weights from Hugging Face"

    _proc: Optional[subprocess.Popen] = None
    _reader: Optional[AsyncProcessReader] = None
    _timer = None

    def modal(self, context, event):
        wm = context.window_manager
        settings = context.scene.skintokens_settings

        if event.type == 'ESC':
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            self.cancel(context)
            self.report({'WARNING'}, "Checkpoint download cancelled.")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self._proc is None:
                return {'PASS_THROUGH'}

            if self._reader:
                updates = self._reader.get_updates()
                for pct, msg in updates:
                    wm.progress_update(pct)
                    settings.progress_percent = pct
                    settings.progress_status = msg
                    context.workspace.status_text_set(f"SkinTokens: {msg} ({pct}%)")
                    for area in context.screen.areas:
                        area.tag_redraw()

            ret = self._proc.poll()
            if ret is not None:
                if self._timer is not None:
                    wm.event_timer_remove(self._timer)
                    self._timer = None
                wm.progress_end()
                context.workspace.status_text_set(None)
                settings.is_running = False
                settings.progress_percent = 100
                settings.progress_status = "Download Complete"

                for area in context.screen.areas:
                    area.tag_redraw()

                if ret == 0:
                    self.report({'INFO'}, "SkinTokens checkpoints downloaded successfully!")
                else:
                    err = self._reader.get_stderr() if self._reader else "Unknown download failure."
                    self.report({'ERROR'}, f"Download failed: {err}")
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        prefs = get_addon_prefs(context)
        settings = context.scene.skintokens_settings
        py_path = prefs.preferences.python_path if prefs else sys.executable
        ckpt_dir = prefs.preferences.checkpoint_dir if prefs else str(Path.home() / ".cache" / "skintokens" / "checkpoints")
        hf_repo = prefs.preferences.hf_repo if prefs else "VAST-AI/SkinTokens"

        worker_script = str(Path(__file__).parent / "worker.py")
        cmd = [
            py_path,
            worker_script,
            "--download_only",
            "--checkpoint_dir", ckpt_dir,
            "--hf_repo", hf_repo,
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._reader = AsyncProcessReader(self._proc)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to launch download worker: {e}")
            return {'CANCELLED'}

        settings.is_running = True
        settings.progress_percent = 0
        settings.progress_status = "Connecting to Hugging Face..."

        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        self.report({'INFO'}, "Downloading SkinTokens checkpoints in background...")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        context.workspace.status_text_set(None)
        context.scene.skintokens_settings.is_running = False


class SKINTOKENS_OT_auto_rig(bpy.types.Operator):
    bl_idname = "skintokens.auto_rig"
    bl_label = "Auto-Rig Active Mesh"
    bl_description = "Generate AI skeletal rig and skinning weights using SkinTokens"

    use_existing_skeleton: bpy.props.BoolProperty(
        name="Use Target Armature",
        description="Condition skinning on the selected target armature hierarchy",
        default=False,
    ) # type: ignore

    skeleton_only: bpy.props.BoolProperty(
        name="Skeleton Only",
        description="Predict skeleton bones only without generating vertex skinning weights",
        default=False,
    ) # type: ignore

    _proc: Optional[subprocess.Popen] = None
    _reader: Optional[AsyncProcessReader] = None
    _timer = None
    _mesh_name: str = ""
    _armature_name: str = ""
    _temp_dir: str = ""
    _temp_obj_path: str = ""
    _output_json_path: str = ""

    @classmethod
    def poll(cls, context):
        settings = getattr(context.scene, "skintokens_settings", None)
        if settings and settings.is_running:
            return False
        return context.active_object is not None and context.active_object.type == 'MESH'

    def modal(self, context, event):
        wm = context.window_manager
        settings = context.scene.skintokens_settings

        if event.type == 'ESC':
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            self.cleanup(context)
            self.report({'WARNING'}, "SkinTokens process cancelled.")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            if self._proc is None:
                return {'PASS_THROUGH'}

            if self._reader:
                updates = self._reader.get_updates()
                for pct, msg in updates:
                    wm.progress_update(pct)
                    settings.progress_percent = pct
                    settings.progress_status = msg
                    context.workspace.status_text_set(f"SkinTokens: {msg} ({pct}%)")
                    for area in context.screen.areas:
                        area.tag_redraw()

            ret = self._proc.poll()
            if ret is not None:
                if self._timer is not None:
                    wm.event_timer_remove(self._timer)
                    self._timer = None
                wm.progress_end()
                context.workspace.status_text_set(None)
                settings.is_running = False
                settings.progress_percent = 100
                settings.progress_status = "Finished"

                for area in context.screen.areas:
                    area.tag_redraw()

                if ret == 0 and os.path.exists(self._output_json_path):
                    try:
                        with open(self._output_json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        if data.get("status") == "success":
                            mesh_obj = bpy.data.objects.get(self._mesh_name)
                            if mesh_obj is None:
                                raise RuntimeError(f"Original mesh object '{self._mesh_name}' not found.")

                            armature_obj = bpy.data.objects.get(self._armature_name) if self._armature_name else None

                            joints = data["joints"]
                            parents = data["parents"]
                            weights = data.get("weights", [])
                            joint_names = data.get("joint_names")

                            max_influences = int(settings.max_influences)
                            weight_threshold = float(settings.weight_threshold)
                            naming_convention = settings.naming_style

                            built_arm = rig_builder.build_rig_and_skin(
                                mesh_obj=mesh_obj,
                                joints=joints,
                                parents=parents,
                                weights=weights,
                                joint_names=joint_names,
                                existing_armature=armature_obj,
                                max_influences=max_influences,
                                threshold=weight_threshold,
                            )

                            # Rename to standard convention (UE5 / Mixamo) after skinning
                            if naming_convention in ("UE5", "MIXAMO"):
                                from .bone_mapper import rename_skeleton_standard
                                conv = "UE5" if naming_convention == "UE5" else "Mixamo"
                                rename_map = rename_skeleton_standard(built_arm, conv)
                                self.report({'INFO'}, f"SkinTokens: renamed {len(rename_map)} bones to {conv}.")

                            mode_str = "Skeleton generated" if not weights else "Rig & Skin complete"
                            self.report({'INFO'}, f"SkinTokens: {mode_str} ({len(joints)} joints).")
                            self.cleanup(context)
                            return {'FINISHED'}
                        else:
                            err_msg = data.get("error", "Unknown inference error")
                            self.report({'ERROR'}, f"SkinTokens Inference Error: {err_msg}")
                    except Exception as e:
                        self.report({'ERROR'}, f"Failed to build rig: {e}")
                else:
                    err = self._reader.get_stderr() if self._reader else "Unknown worker failure."
                    self.report({'ERROR'}, f"SkinTokens worker error (code {ret}):\n{err}")

                self.cleanup(context)
                return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        mesh_obj = context.active_object
        if not mesh_obj or mesh_obj.type != 'MESH':
            self.report({'ERROR'}, "Please select an active Mesh object.")
            return {'CANCELLED'}

        self._mesh_name = mesh_obj.name
        settings = context.scene.skintokens_settings
        prefs = get_addon_prefs(context)

        py_path = prefs.preferences.python_path if prefs else sys.executable
        ckpt_dir = prefs.preferences.checkpoint_dir if prefs else str(Path.home() / ".cache" / "skintokens" / "checkpoints")
        hf_repo = prefs.preferences.hf_repo if prefs else "VAST-AI/SkinTokens"

        # 1. Export temporary OBJ
        try:
            self._temp_dir, self._temp_obj_path = export_temp_obj(mesh_obj)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to export temporary mesh: {e}")
            return {'CANCELLED'}

        self._output_json_path = os.path.join(self._temp_dir, "skintokens_result.json")

        worker_script = str(Path(__file__).parent / "worker.py")
        cmd = [
            py_path,
            worker_script,
            "--input", self._temp_obj_path,
            "--output", self._output_json_path,
            "--checkpoint_dir", ckpt_dir,
            "--hf_repo", hf_repo,
            "--top_k", str(settings.top_k),
            "--top_p", str(settings.top_p),
            "--temperature", str(settings.temperature),
            "--repetition_penalty", str(settings.repetition_penalty),
            "--num_beams", str(settings.num_beams),
        ]

        # 2. Check if using existing target armature
        target_arm = settings.target_armature if self.use_existing_skeleton else None
        if target_arm is not None:
            self._armature_name = target_arm.name
            skel_json_path = os.path.join(self._temp_dir, "existing_skeleton.json")
            try:
                export_armature_json(target_arm, skel_json_path)
                cmd.extend(["--skeleton_file", skel_json_path, "--use_skeleton"])
            except Exception as e:
                self.report({'ERROR'}, f"Failed to export target armature: {e}")
                self.cleanup(context)
                return {'CANCELLED'}
        else:
            self._armature_name = ""

        if self.skeleton_only:
            cmd.append("--skeleton_only")
        if settings.use_postprocess:
            cmd.append("--use_postprocess")

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._reader = AsyncProcessReader(self._proc)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start SkinTokens worker: {e}\nCheck Python path in Preferences.")
            self.cleanup(context)
            return {'CANCELLED'}

        settings.is_running = True
        settings.progress_percent = 5
        settings.progress_status = "Starting AI Worker..."

        wm = context.window_manager
        wm.progress_begin(0, 100)
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        self.report({'INFO'}, "SkinTokens AI processing in progress...")
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None

        wm.progress_end()
        context.workspace.status_text_set(None)
        context.scene.skintokens_settings.is_running = False

        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass


def register():
    bpy.utils.register_class(SKINTOKENS_OT_check_env)
    bpy.utils.register_class(SKINTOKENS_OT_setup_env)
    bpy.utils.register_class(SKINTOKENS_OT_download_ckpt)
    bpy.utils.register_class(SKINTOKENS_OT_auto_rig)


def unregister():
    bpy.utils.unregister_class(SKINTOKENS_OT_auto_rig)
    bpy.utils.unregister_class(SKINTOKENS_OT_download_ckpt)
    bpy.utils.unregister_class(SKINTOKENS_OT_setup_env)
    bpy.utils.unregister_class(SKINTOKENS_OT_check_env)
