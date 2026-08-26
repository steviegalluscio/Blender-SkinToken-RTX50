#!/usr/bin/env python3
"""
Standalone inference worker for SkinTokens Blender Addon.
Runs in an external Python environment (with PyTorch, Transformers, etc.)
and outputs skeletal hierarchy and skinning weights JSON for Blender.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure addon root and src are in sys.path
ADDON_ROOT = Path(__file__).resolve().parent
if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))

DEFAULT_REPO_ID = "VAST-AI/SkinTokens"
DEFAULT_MODELS = [
    "experiments/skin_vae_2_10_32768/last.ckpt",
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
]
LLM_REPO = "Qwen/Qwen3-0.6B"


def ensure_models(checkpoint_dir: Path, repo_id: str = DEFAULT_REPO_ID) -> Path:
    """Ensures model checkpoints and LLM configs are downloaded locally."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise RuntimeError("huggingface_hub is not installed in the worker Python environment.")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    main_ckpt = checkpoint_dir / DEFAULT_MODELS[1]

    if not main_ckpt.exists():
        print(f"[PROGRESS:10] Connecting to Hugging Face ({repo_id})...", flush=True)
        total_models = len(DEFAULT_MODELS)
        for i, rel_path in enumerate(DEFAULT_MODELS, 1):
            pct = 10 + int((i / total_models) * 50)
            print(f"[PROGRESS:{pct}] Downloading {Path(rel_path).name} ({i}/{total_models})...", flush=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                local_dir=str(checkpoint_dir),
            )
        # Download Qwen config
        print("[PROGRESS:75] Downloading Qwen LLM architecture files...", flush=True)
        llm_dir = checkpoint_dir / "models" / "Qwen3-0.6B"
        snapshot_download(
            repo_id=LLM_REPO,
            local_dir=str(llm_dir),
            ignore_patterns=["*.bin", "*.safetensors"],
        )
        print("[PROGRESS:100] Checkpoint download completed!", flush=True)

    return main_ckpt


def run_inference(args):
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    
    if args.download_only:
        print("[PROGRESS:5] Starting checkpoint download...", flush=True)
        ensure_models(ckpt_dir, repo_id=args.hf_repo)
        print("[PROGRESS:100] Checkpoints downloaded successfully.", flush=True)
        return {"status": "success", "message": "Checkpoints downloaded."}

    import torch
    from torch import Tensor
    import numpy as np

    from src.data.datapath import parse_obj_mesh
    from src.data.dataset import DatasetConfig, RigDatasetModule
    from src.data.transform import Transform
    from src.model.tokenrig import TokenRig, TokenRigResult
    from src.tokenizer.parse import get_tokenizer
    from src.data.vertex_group import voxel_skin
    from src.rig_package.info.asset import Asset

    device = _pick_device()

    print(f"[PROGRESS:10] Using device: {device}", flush=True)

    if args.model_ckpt:
        ckpt_path = Path(args.model_ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = ckpt_dir / ckpt_path
    else:
        ckpt_path = ensure_models(ckpt_dir, repo_id=args.hf_repo)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    print(f"[PROGRESS:25] Loading AI model checkpoint to {device}...", flush=True)
    model = TokenRig.load_from_system_checkpoint(checkpoint_path=str(ckpt_path))
    model = model.to(device)
    model.eval()

    assert model.tokenizer_config is not None
    tokenizer = get_tokenizer(**model.tokenizer_config)
    transform = Transform.parse(**model.transform_config["predict_transform"])

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh file not found: {input_path}")

    print(f"[PROGRESS:40] Loading mesh: {input_path.name}", flush=True)
    
    # Native pure-python OBJ loading
    vertices, faces = parse_obj_mesh(str(input_path))
    num_input_verts = len(vertices)
    if num_input_verts == 0:
        raise ValueError(f"Input mesh {input_path.name} contains 0 vertices.")

    print(f"[Worker] Input mesh has {num_input_verts} vertices, {len(faces)} faces.", flush=True)

    datapath = {
        "data_name": None,
        "loader": "obj",
        "filepaths": {"articulation": [str(input_path)]},
    }

    dataset_config = DatasetConfig.parse(
        shuffle=False,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        datapath=datapath,
    ).split_by_cls()

    module = RigDatasetModule(
        predict_dataset_config=dataset_config,
        predict_transform=transform,
        tokenizer=tokenizer,
        process_fn=model._process_fn,
    )

    dataloader = module.predict_dataloader()["articulation"]
    
    # Check if existing skeleton file is provided
    custom_skeleton = None
    if getattr(args, "skeleton_file", None) and Path(args.skeleton_file).exists():
        with open(args.skeleton_file, "r", encoding="utf-8") as f:
            custom_skeleton = json.load(f)
        print(f"[PROGRESS:50] Loaded existing skeleton: {len(custom_skeleton.get('joints', []))} bones.", flush=True)

    print("[PROGRESS:55] Generating skeletal bone hierarchy with Qwen LLM...", flush=True)
    result_asset = None
    for batch in dataloader:
        batch = {
            k: v.to(device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

        if custom_skeleton is not None:
            bound_min = vertices.min(axis=0)
            bound_max = vertices.max(axis=0)
            mesh_center = (bound_max + bound_min) / 2.0
            mesh_scale = np.max((bound_max - bound_min) / 2.0)

            skel_joints = np.asarray(custom_skeleton["joints"], dtype=np.float32)
            norm_joints = (skel_joints - mesh_center) / mesh_scale
            skel_parents = custom_skeleton["parents"]
            skel_names = custom_skeleton.get("joint_names")

            from src.tokenizer.spec import TokenizeInput
            x = TokenizeInput(
                joints=norm_joints,
                parents=skel_parents,
                cls=None,
                joint_names=skel_names,
            )
            skel_tokens = tokenizer.tokenize(input=x)
            skel_tokens_tensor = torch.tensor([skel_tokens], device=device)
            batch["skeleton_tokens"] = skel_tokens_tensor
            batch["skeleton_mask"] = torch.ones_like(skel_tokens_tensor)
        elif not args.use_skeleton:
            batch.pop("skeleton_tokens", None)
            batch.pop("skeleton_mask", None)

        batch["generate_kwargs"] = dict(
            max_length=2048,
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            temperature=float(args.temperature),
            repetition_penalty=float(args.repetition_penalty),
            num_return_sequences=1,
            num_beams=int(args.num_beams),
            do_sample=True,
        )

        with torch.no_grad():
            print("[PROGRESS:75] Predicting FSQ-CVAE vertex skinning weights...", flush=True)
            preds: List[TokenRigResult] = model.predict_step(
                batch,
                skeleton_tokens=batch.get("skeleton_tokens"),
                make_asset=True,
            )["results"]

        result_asset = preds[0].asset
        break

    if result_asset is None:
        raise RuntimeError("Inference did not return a valid rigged asset.")

    if getattr(args, "skeleton_only", False):
        print("[PROGRESS:95] Skipping skinning weights (Skeleton-Only mode)...", flush=True)
        skin_weights = []
    else:
        # Post-process voxel smoothing if requested
        if args.use_postprocess:
            print("[PROGRESS:90] Applying voxel skin smoothing...", flush=True)
            voxel = result_asset.voxel(resolution=196)
            result_asset.skin *= voxel_skin(
                grid=0,
                grid_coords=voxel.coords,
                joints=result_asset.joints,
                vertices=result_asset.vertices,
                faces=result_asset.faces,
                mode="square",
                voxel_size=voxel.voxel_size,
            )
            result_asset.normalize_skin()

        skin_weights = result_asset.skin

    print("[PROGRESS:95] Transforming joints and hierarchy to mesh space...", flush=True)

    bound_min = vertices.min(axis=0)
    bound_max = vertices.max(axis=0)
    mesh_center = (bound_max + bound_min) / 2.0
    mesh_scale = np.max((bound_max - bound_min) / 2.0)

    # Extract joints, parents, weights
    if result_asset.joints is not None:
        unnorm_joints = np.asarray(result_asset.joints) * mesh_scale + mesh_center
        joints = unnorm_joints.tolist()
    else:
        joints = []
    parents = result_asset.parents.tolist() if isinstance(result_asset.parents, np.ndarray) else list(result_asset.parents)
    joint_names = result_asset.joint_names or [f"Bone_{i:03d}" for i in range(len(joints))]
    
    if isinstance(skin_weights, np.ndarray):
        weights_list = skin_weights.tolist()
    elif isinstance(skin_weights, list):
        weights_list = skin_weights
    else:
        weights_list = np.asarray(skin_weights).tolist()

    # Clear GPU VRAM cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_payload = {
        "status": "success",
        "joints": joints,
        "parents": parents,
        "joint_names": joint_names,
        "weights": weights_list,
        "vertex_count": len(weights_list),
    }
    return output_payload


def _pick_device() -> str:
    """Auto-detect the best available accelerator: CUDA (NVIDIA/ROCm), MPS (Apple), XPU (Intel), else CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return "mps"
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="SkinTokens Standalone Inference Worker")
    parser.add_argument("--input", type=str, required=False, help="Path to input 3D mesh")
    parser.add_argument("--output", type=str, required=False, help="Path to output result JSON")
    parser.add_argument("--checkpoint_dir", type=str, default=str(Path.home() / ".cache" / "skintokens" / "checkpoints"))
    parser.add_argument("--model_ckpt", type=str, default="", help="Specific checkpoint path")
    parser.add_argument("--hf_repo", type=str, default=DEFAULT_REPO_ID, help="HuggingFace repo ID")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=2.0)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--use_skeleton", action="store_true", default=False)
    parser.add_argument("--skeleton_only", action="store_true", default=False, help="Generate skeleton only without skinning")
    parser.add_argument("--skeleton_file", type=str, default="", help="Path to existing skeleton JSON file to condition on")
    parser.add_argument("--use_postprocess", action="store_true", default=False,
                        help="Apply voxel skin smoothing (official demo default: off)")
    parser.add_argument("--no_postprocess", dest="use_postprocess", action="store_false",
                        help="Disable voxel skin smoothing")
    parser.add_argument("--download_only", action="store_true", default=False)

    args = parser.parse_args()

    try:
        payload = run_inference(args)
        if args.output:
            out_file = Path(args.output).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            print(f"[Worker] Successfully saved output to {out_file}", flush=True)
        else:
            print(json.dumps(payload))
        sys.exit(0)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Worker Error] {e}\n{tb}", file=sys.stderr, flush=True)
        err_payload = {
            "status": "error",
            "error": str(e),
            "traceback": tb,
        }
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(err_payload, f)
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
