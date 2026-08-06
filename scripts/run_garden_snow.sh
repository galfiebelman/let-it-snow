#!/bin/bash
# Full pipeline for the Garden scene with Snow effect.
# Adjust paths to match your setup.

set -e

# --- Configuration ---
SCENE_DATA_DIR="data/garden"           # COLMAP dataset (images/, sparse/)
MODEL_PATH="output/garden"             # Where 3DGS model will be saved
MESH_PATH="output/garden/mesh/fuse_post.ply"  # Extracted mesh
SIM_OUTPUT_DIR="output/garden/simulation_snow"
OPTIM_OUTPUT_DIR="output/garden/optimize_snow"
EFFECT_CONFIG="configs/snow.json"
BG_PATH=""  # Set to path of ClimateNeRF-finetuned .ply for background snow enhancement

# --- Step 0: Train 3DGS scene ---
# Uses the upstream gaussian-splatting training script.
# The trained model is saved to MODEL_PATH.
echo "=== Step 0: Training 3DGS scene ==="
python train.py -s "${SCENE_DATA_DIR}" -m "${MODEL_PATH}"

# --- Step 0b: Extract mesh ---
# Use 2d-gaussian-splatting (https://github.com/hbb1/2d-gaussian-splatting)
# to extract a mesh via TSDF fusion.
# Render depth maps from training views, then fuse:
#   python render.py -m $MODEL_PATH --render_depth
#   python scripts/tsdf_fusion.py -m $MODEL_PATH --voxel_size 0.004 --sdf_trunc 0.02 --mesh_res 1024
# The mesh is saved to $MESH_PATH.

# --- Step 0c: Estimate ground plane ---
echo "=== Step 0c: Estimating ground plane ==="
python prepare_scene.py -m "${MODEL_PATH}"

# --- Step 1: MPM Simulation ---
echo "=== Step 1: Running MPM simulation ==="
python simulate.py \
    -m "${MODEL_PATH}" \
    --config "${EFFECT_CONFIG}" \
    --output_dir "${SIM_OUTPUT_DIR}" \
    --mesh_path "${MESH_PATH}"

# --- Step 2: Physics-Guided Score Distillation ---
echo "=== Step 2: Running optimization ==="
python optimize.py \
    -m "${MODEL_PATH}" \
    --simulation_dir "${SIM_OUTPUT_DIR}" \
    --mesh_path "${MESH_PATH}" \
    --effect_config_path "${EFFECT_CONFIG}" \
    --effect_type snow \
    --save_dir_name "${OPTIM_OUTPUT_DIR}" \
    --prompt "Fluffy snowflakes are falling in a table with a vase on it in a garden, accumulating on surfaces. Photorealistic, high detail" \
    --steps 1000 \
    --lr 1e-4 \
    --guidance_scale 100.0 \
    --bg_path "${BG_PATH}"

# --- Step 3: Render final video ---
echo "=== Step 3: Rendering final video ==="
python render.py \
    -m "${MODEL_PATH}" \
    --simulation_dir "${SIM_OUTPUT_DIR}" \
    --mesh_path "${MESH_PATH}" \
    --effect_config_path "${EFFECT_CONFIG}" \
    --effect_type snow \
    --save_dir_name "${OPTIM_OUTPUT_DIR}" \
    --bg_path "${BG_PATH}" \
    --render_fps 25

echo "=== Done! Video saved to ${OPTIM_OUTPUT_DIR}/final_video.mp4 ==="
