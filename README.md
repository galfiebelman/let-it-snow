# Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation

### [Project Page](https://galfiebelman.github.io/let-it-snow/) | [ArXiv](https://arxiv.org/abs/2504.05296)

This is the official implementation of **Let it Snow!**

![Generic badge](https://img.shields.io/badge/conf-CVPR%202026-blue.svg)

> **Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation**<br>
> Gal Fiebelman<sup>1</sup>, Hadar Averbuch-Elor<sup>2</sup>, Sagie Benaim<sup>1</sup><br>
> <sup>1</sup>The Hebrew University of Jerusalem, <sup>2</sup>Cornell University

>**Abstract** <br>
> 3D Gaussian Splatting has recently enabled fast and photorealistic reconstruction of static 3D scenes. However, dynamic editing of such scenes remains a significant challenge. We introduce a novel framework, *Physics-Guided Score Distillation*, to address a fundamental conflict: physics simulation provides a strong motion prior that is insufficient for photorealism, while video-based Score Distillation Sampling (SDS) alone cannot generate coherent motion for complex, multi-particle scenarios. We resolve this through a unified optimization framework where physics simulation guides Score Distillation to jointly refine the motion prior for photorealism while simultaneously optimizing appearance. Specifically, we learn a neural dynamics model that predicts particle motion and appearance, optimized end-to-end via a combined loss integrating Video-SDS for photorealism with our physics-guidance prior. This allows for photorealistic refinements while ensuring the dynamics remain plausible. Our framework enables scene-wide dynamic weather effects, including snowfall, rainfall, fog, and sandstorms, with physically plausible motion. Experiments demonstrate our physics-guided approach significantly outperforms baselines, with ablations confirming this joint refinement is essential for generating coherent, high-fidelity dynamics.

</br>

# Getting Started

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/galfiebelman/let-it-snow.git
cd let-it-snow
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate let-it-snow
```

### 3. Install submodules
```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

</br>

# Usage

We evaluate on scenes from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and [Tanks and Temples](https://www.tanksandtemples.org/). Below we walk through the full pipeline using the **Garden** scene from Mip-NeRF 360 with **snow** as an example. See `scripts/run_garden_snow.sh` for a complete script.

Download and extract the Garden scene:
```bash
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip
mv 360_v2/garden data/garden
```

## Step 0: Train 3DGS Scene and Prepare

Train the static 3D Gaussian Splatting scene:
```bash
python train.py -s data/garden -m output/garden
```

Extract the scene mesh using [2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting):
```bash
# In the 2d-gaussian-splatting repo:
python render.py -m output/garden --render_depth
python scripts/tsdf_fusion.py -m output/garden --voxel_size 0.004 --sdf_trunc 0.02 --mesh_res 1024
```
This produces a mesh at `output/garden/mesh/fuse_post.ply`.

Estimate the ground plane for physics simulation:
```bash
python prepare_scene.py -m output/garden
```
This fits a plane via RANSAC on the lowest 30% of Gaussians and saves the ground alignment transform to `editing_modifier.pkl` alongside the point cloud. Adjust `--distance_threshold` (default 0.02) if the ground estimate looks off.

## Step 1: MPM Simulation

Run the physics simulation with effect-specific parameters:
```bash
python simulate.py \
    -m output/garden \
    --config configs/snow.json \
    --output_dir output/garden/simulation_snow \
    --mesh_path output/garden/mesh/fuse_post.ply
```

This saves per-frame particle positions (`{t}_pos.npz`), IDs (`{t}_ids.npz`), and static particles (`{t}_static.npz`) to the output directory.

## Step 2: Physics-Guided Score Distillation

Optimize the neural dynamics model:
```bash
python optimize.py \
    -m output/garden \
    --simulation_dir output/garden/simulation_snow \
    --mesh_path output/garden/mesh/fuse_post.ply \
    --effect_config_path configs/snow.json \
    --effect_type snow \
    --save_dir_name output/garden/optimize_snow \
    --prompt "Fluffy snowflakes are falling in a table with a vase on it in a garden, accumulating on surfaces. Photorealistic, high detail" \
    --steps 1000 \
    --lr 1e-4 \
    --guidance_scale 100.0
```

Checkpoints and diagnostic videos are saved periodically.

## Step 3: Render

Render the final video from a trained checkpoint:
```bash
python render.py \
    -m output/garden \
    --simulation_dir output/garden/simulation_snow \
    --mesh_path output/garden/mesh/fuse_post.ply \
    --effect_config_path configs/snow.json \
    --effect_type snow \
    --save_dir_name output/garden/optimize_snow \
    --render_fps 25
```

The script automatically finds the latest checkpoint. Use `--ckpt path/to/checkpoint.pth` to specify a particular one.

## Background Snow Enhancement (Optional)

For the snow effect, you can enhance the background with gradual snow accumulation:

1. Generate snowy versions of training views using [ClimateNeRF](https://github.com/y-u-a-n-l-i/Climate_NeRF)
2. Finetune the 3DGS model on these images:
   ```bash
   python train.py -s data/garden_snowy -m output/garden_snowy
   ```
3. Pass the finetuned model as `--bg_path` to both `optimize.py` and `render.py`

</br>

# Configurations

Effect-specific simulation, appearance, and optimization parameters are stored as JSON files in `configs/` (`snow.json`, `rain.json`, `fog.json`, `sand.json`). To run a different effect or tune hyperparameters, edit the corresponding config and change the `--config`, `--effect_config_path`, and `--effect_type` arguments accordingly.

</br>

# BibTeX

If you find our work useful in your research, please consider citing:

```bibtex
@misc{fiebelman2026letsnowanimating3d,
    title={Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation},
    author={Gal Fiebelman and Hadar Averbuch-Elor and Sagie Benaim},
    year={2026},
    eprint={2504.05296},
    archivePrefix={arXiv},
    primaryClass={cs.GR},
    url={https://arxiv.org/abs/2504.05296},
}
```

</br>

# Acknowledgements

Built on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting), [Feature Splatting](https://github.com/vuer-ai/feature-splatting-inria), [Taichi MPM](https://github.com/taichi-dev/taichi_elements), [CogVideoX](https://github.com/THUDM/CogVideo), and [ClimateNeRF](https://github.com/y-u-a-n-l-i/Climate_NeRF). This research was supported by The Israel Science Foundation (grant No. 2416/25).