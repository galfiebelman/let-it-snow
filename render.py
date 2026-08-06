"""
Render script for trained neural dynamics models.
Loads a checkpoint from optimize.py and renders the final video
with recurrent state passing.
"""

import os
import argparse
import imageio
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
import numpy as np
import glob
import re

from models.gaussian_matrices import GaussianMatrices
from scene import Scene, GaussianModel
from effects import EffectRenderer
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.camera_utils import get_current_view
from torchvision.transforms import functional as TF


def save_frames_as_video(frames_tensor, path, fps=8):
    frames_np = []
    for i in range(frames_tensor.shape[0]):
        frame = frames_tensor[i].detach().cpu().clamp(0, 1)
        frame_pil = TF.to_pil_image(frame)
        frames_np.append(np.array(frame_pil))
    imageio.mimsave(str(path), frames_np, fps=fps)


def precompute_effect_data(args, cam_list, gaussians, gs_pipeline, background,
                           effect_renderer, total_steps, save_init=True):
    output_path = Path(args.save_dir_name)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / "init_video.mp4"
    video_frames = []

    print("Precomputing fallen particle data (Pass 1/2)...")
    with torch.no_grad():
        start_idx = args.render_cam_start_idx
        end_idx = args.render_cam_end_idx

        for i in tqdm(range(total_steps), desc="Caching Fallen Data"):
            t = i / total_steps
            view = get_current_view(cam_list, str(start_idx), str(end_idx), t)
            render_pkg = effect_renderer.renderer.render(
                view, gaussians, gs_pipeline, background, t=i, refine=False
            )
            if save_init:
                frame = render_pkg["render"]
                video_frames.append(frame.cpu())
        if save_init:
            video_tensor = torch.stack(video_frames, dim=0)
            save_frames_as_video(video_tensor, output_path, fps=args.render_fps)

    print("Precomputing moving particle data (Pass 2/2)...")
    if hasattr(effect_renderer.renderer, 'precompute_moving_data'):
        with torch.no_grad():
            effect_renderer.renderer.precompute_moving_data(total_steps)

    print("Precomputation complete.")


def render_full_video(args, cam_list, gaussians, gs_pipeline, background, effect_renderer,
                      gm_moving, gm_fallen, device="cuda"):
    output_path = Path(args.save_dir_name)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / "final_video.mp4"
    video_frames = []
    num_steps = args.num_sim_steps

    print(f"Rendering {num_steps} frames...")

    moving_app_state = gm_moving.get_initial_state()
    fallen_app_state = gm_fallen.get_initial_state()

    init_moving_data = effect_renderer.renderer.get_moving_data(0)
    num_moving_particles = gm_moving.original_opacities_logit.shape[0]
    prev_x_rendered_moving = torch.zeros(num_moving_particles, 3).float().cuda()
    prev_v_corrected_moving = torch.zeros(num_moving_particles, 3).float().cuda()
    init_ids_m = torch.from_numpy(init_moving_data["ids"]).int().cuda()
    init_pos_m = torch.from_numpy(init_moving_data["pos_t_minus_1"]).float().cuda()
    init_vel_m = torch.from_numpy(init_moving_data["velocity"]).float().cuda()
    prev_x_rendered_moving[init_ids_m] = init_pos_m
    prev_v_corrected_moving[init_ids_m] = init_vel_m

    num_fallen_particles = gm_fallen.original_opacities_logit.shape[0]
    prev_x_rendered_fallen = torch.zeros(num_fallen_particles, 3).float().cuda()
    prev_v_corrected_fallen = torch.zeros(num_fallen_particles, 3).float().cuda()

    with torch.no_grad():
        start_idx = args.render_cam_start_idx
        end_idx = args.render_cam_end_idx

        for i in tqdm(range(num_steps), desc="Rendering Video"):
            t = i / num_steps
            view = get_current_view(cam_list, str(start_idx), str(end_idx), t)

            render_pkg = effect_renderer.renderer.render(
                view, gaussians, gs_pipeline, background,
                t=i, refine=True, gm_moving=gm_moving, gm_fallen=gm_fallen,
                gm_moving_app_state=moving_app_state,
                gm_fallen_app_state=fallen_app_state,
                prev_x_rendered_moving=prev_x_rendered_moving,
                prev_v_corrected_moving=prev_v_corrected_moving,
                prev_x_rendered_fallen=prev_x_rendered_fallen,
                prev_v_corrected_fallen=prev_v_corrected_fallen
            )
            frame = render_pkg["render"]
            save_path = os.path.join(args.save_dir_name, f"frame_{i:05d}.png")
            torchvision.utils.save_image(frame, save_path)
            video_frames.append(frame.cpu())

            moving_app_state = render_pkg["new_moving_app_state"]
            fallen_app_state = render_pkg["new_fallen_app_state"]
            prev_x_rendered_moving = render_pkg["new_x_rendered_moving"]
            prev_v_corrected_moving = render_pkg["new_v_corrected_moving"]
            prev_x_rendered_fallen = render_pkg["new_x_rendered_fallen"]
            prev_v_corrected_fallen = render_pkg["new_v_corrected_fallen"]

    if not video_frames:
        print("Error: No frames were rendered.")
        return
    video_tensor = torch.stack(video_frames, dim=0)
    save_frames_as_video(video_tensor, output_path, fps=args.render_fps)
    print(f"Video saved to {output_path}")


def find_latest_checkpoint(save_dir):
    ckpt_files = glob.glob(os.path.join(save_dir, "checkpoint_*.pth"))
    if not ckpt_files:
        return None

    latest_file = None
    latest_step = -1
    for f in ckpt_files:
        match = re.search(r"checkpoint_(\d+).pth", f)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_file = f

    print(f"Found {len(ckpt_files)} checkpoints. Using latest: {latest_file}")
    return latest_file


def main():
    parser = argparse.ArgumentParser(description="Render trained neural dynamics model.")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--save_dir_name", type=str, required=True,
                        help="Directory where checkpoints are saved.")
    parser.add_argument("-o", "--output_path", type=str, default="renders/final_render.mp4")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Specific checkpoint file. If empty, uses latest.")
    parser.add_argument("--save_init", action="store_true",
                        help="Also save video of initial (unrefined) simulation.")
    parser.add_argument("--simulation_dir", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--effect_config_path", type=str, required=True)
    parser.add_argument("--effect_type", type=str, default="snow", choices=["snow", "rain", "fog", "sand"])
    parser.add_argument("--bg_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--moving_hidden_dim", type=int, default=128)
    parser.add_argument("--fallen_hidden_dim", type=int, default=128)
    parser.add_argument("--num_sim_steps", type=int, default=500)
    parser.add_argument("--render_fps", type=int, default=25)
    parser.add_argument("--render_cam_start_idx", type=int, default=0)
    parser.add_argument("--render_cam_end_idx", type=int, default=10)

    args = get_combined_args(parser)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find checkpoint
    load_path = args.ckpt
    if load_path == "":
        load_path = find_latest_checkpoint(args.save_dir_name)
        if load_path is None:
            print(f"Error: No checkpoints found in {args.save_dir_name}")
            return

    if not os.path.exists(load_path):
        print(f"Error: Checkpoint file not found: {load_path}")
        return

    print(f"Loading checkpoint: {load_path}")

    # Load scene
    gaussians = GaussianModel(args.sh_degree, 0)
    gs_pipeline = pipeline.extract(args)
    dataset = model.extract(args)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)

    effect_renderer = EffectRenderer(
        args.effect_type, args.mesh_path,
        args.simulation_dir, args.effect_config_path, args.bg_path
    )

    train_cameras = scene.getTrainCameras()
    if not train_cameras:
        raise ValueError("No training cameras found in the scene.")

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Initialize models
    moving_scales, moving_opacities, moving_colors, moving_rotations = \
        effect_renderer.renderer.get_moving_gaussians()
    fallen_scales, fallen_opacities, fallen_colors, fallen_rotations = \
        effect_renderer.renderer.get_fallen_gaussians()

    gm_moving = GaussianMatrices(
        moving_opacities.cpu().numpy(), moving_scales.cpu().numpy(),
        moving_colors.cpu().numpy(), moving_rotations.cpu().numpy(),
        hidden_dim=args.moving_hidden_dim, device=device, mode='full'
    )
    gm_fallen = GaussianMatrices(
        fallen_opacities.cpu().numpy(), fallen_scales.cpu().numpy(),
        fallen_colors.cpu().numpy(), fallen_rotations.cpu().numpy(),
        hidden_dim=args.fallen_hidden_dim, device=device, mode='appearance'
    )

    # Load checkpoint
    ckpt = torch.load(load_path, map_location=device)
    gm_moving.load_state_dict(ckpt["moving_state"])
    gm_fallen.load_state_dict(ckpt["fallen_state"])
    gm_moving.eval()
    gm_fallen.eval()
    print(f"Loaded model from step {ckpt.get('step', 'N/A')}")

    # Precompute
    precompute_effect_data(
        args, train_cameras, gaussians, gs_pipeline, background,
        effect_renderer, args.num_sim_steps, save_init=args.save_init
    )

    # Render
    render_full_video(
        args, train_cameras, gaussians, gs_pipeline, background,
        effect_renderer, gm_moving, gm_fallen, device
    )


if __name__ == "__main__":
    main()
