# This fork makes the add-on install PyTorch 2.7.0 with CUDA 12.8 (`2.7.0+cu128`). The original repo installed a default PyTorch package (`2.14.1+cpu`).

# SkinTokens for Blender

AI skeletal rigging and skinning weight prediction inside Blender. Port of [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens). Select a mesh, press one button, get an armature with vertex group weights.

## Requirements

- Blender 4.2 or newer (tested on 5.x)
- GPU recommended. Any PyTorch backend works: CUDA (NVIDIA), ROCm (AMD), MPS (Apple Silicon), XPU (Intel). Falls back to CPU automatically, which is slow.
  
## Installation

1. Install the addon:
   - Download this as .Zip file, then drag and drop into  Blender or  go to Edit > Preferences > Add-ons > menu (arrow, top right) > Install from Disk, select the zip and enable it.

3. Open Blender preferences, find SkinTokens, set:
   - **Python Executable**: path to the existing venv with all the deps pre-installed or leave it empty.
   - **Checkpoints Directory**: where checkpoints are should be Download.

4. In the 3D viewport sidebar (N key), SkinTokens tab, click **Download Checkpoints** or place them manually:

   ```
   <checkpoint_dir>/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt
   <checkpoint_dir>/experiments/skin_vae_2_10_32768/last.ckpt
   ```

5. Click **Check Environment** to verify everything is found.

## Usage

1. Open the sidebar (N key), go to the **SkinTokens** tab.
2. Select a single mesh object in the viewport.
3. Click **Auto-Rig Active Mesh**. Generation runs in a background process; progress shows in the panel.
4. When finished you get an armature parented to the mesh with an Armature modifier and weighted vertex groups.

### Options

- **Bone Naming**: keep raw predicted names, or rename to Mixamo or UE5 conventions based on skeleton topology (UE5 adds a root bone).
- **Target Armature**: optionally point at an existing armature to only predict skin weights for it.
- **Max Influences**: cap bones per vertex (4 is standard for game engines).
- **Weight Threshold**: discard influences below this weight.
- **Use Postprocess**: voxel-based skin smoothing. Off by default, matching the official demo. On can flip weights near joints.
- **Sampling**: top_k, top_p, temperature, repetition penalty, beams. Defaults match the official demo (top_k=5, temperature=1.0, repetition=2.0, beams=10).


## Credits

Based on [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) by VAST-AI Research.
