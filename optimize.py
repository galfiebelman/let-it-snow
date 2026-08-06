"""
Physics-Guided Score Distillation optimization.
Trains recurrent neural dynamics models for Gaussian appearance and motion
using Video-SDS (CogVideoX-2B) with physics guidance losses.
"""

import os
import argparse
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
import random
import numpy as np

from guidance.cogvideox_guidance import CogVideoXDirectGuidance
from models.gaussian_matrices import GaussianMatrices
from scene import Scene, GaussianModel
from scene.cameras import Camera
from effects import EffectRenderer
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.camera_utils import get_current_view
from torchvision.transforms import functional as TF
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR


def detach_state(state_dict):
    if state_dict is None:
        return None
    return {k: v.detach() for k, v in state_dict.items()}


def save_frames_as_video(frames_tensor, path, fps=8):
    frames_np = []
    for i in range(frames_tensor.shape[0]):
        frame = frames_tensor[i].detach().cpu().clamp(0, 1)
        frame_pil = TF.to_pil_image(frame)
        frames_np.append(np.array(frame_pil))
    imageio.mimsave(str(path), frames_np, fps=fps)


def render_sds_clip(original_cam, gaussians, gs_pipeline, background, effect_renderer,
                    t_start, sds_video_len, downsample_factor,
                    gm_moving, gm_fallen, device="cuda"):
    low_res_cam = Camera(
        colmap_id=original_cam.colmap_id, R=original_cam.R, T=original_cam.T,
        FoVx=original_cam.FoVx, FoVy=original_cam.FoVy,
        image=original_cam.original_image, gt_alpha_mask=None,
        image_name=original_cam.image_name, uid=original_cam.uid,
        data_device=original_cam.data_device
    )
    low_res_cam.image_width = 720 // downsample_factor
    low_res_cam.image_height = 480 // downsample_factor

    total_reg_loss_xyz = 0.0
    total_reg_loss_vel = 0.0
    total_reg_loss_rot = 0.0
    total_op_reg_loss_m = 0.0
    total_sc_reg_loss_m = 0.0
    total_col_reg_loss_m = 0.0
    total_op_reg_loss_f = 0.0
    total_sc_reg_loss_f = 0.0
    total_col_reg_loss_f = 0.0

    video_frames = []
    moving_app_state = gm_moving.get_initial_state()
    fallen_app_state = gm_fallen.get_initial_state()
    t_init = t_start
    if t_init == 0:
        t_init = 1

    init_moving_data = effect_renderer.renderer.get_moving_data(t_init)
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

    for i in range(sds_video_len):
        t_current = t_start + i
        render_pkg = effect_renderer.renderer.render(
            low_res_cam, gaussians, gs_pipeline, background,
            t=t_current, refine=True,
            gm_moving=gm_moving, gm_fallen=gm_fallen,
            gm_moving_app_state=moving_app_state,
            gm_fallen_app_state=fallen_app_state,
            prev_x_rendered_moving=prev_x_rendered_moving,
            prev_v_corrected_moving=prev_v_corrected_moving,
            prev_x_rendered_fallen=prev_x_rendered_fallen,
            prev_v_corrected_fallen=prev_v_corrected_fallen
        )
        frame = render_pkg["render"]

        if "reg_loss_m" in render_pkg:
            total_reg_loss_xyz += render_pkg["reg_loss_m"]
        if "reg_loss_vel" in render_pkg:
            total_reg_loss_vel += render_pkg["reg_loss_vel"]
        if "rot_reg_loss_m" in render_pkg:
            total_reg_loss_rot += render_pkg["rot_reg_loss_m"]
        if "op_reg_loss_m" in render_pkg:
            total_op_reg_loss_m += render_pkg["op_reg_loss_m"]
        if "sc_reg_loss_m" in render_pkg:
            total_sc_reg_loss_m += render_pkg["sc_reg_loss_m"]
        if "col_reg_loss_m" in render_pkg:
            total_col_reg_loss_m += render_pkg["col_reg_loss_m"]
        if "op_reg_loss_f" in render_pkg:
            total_op_reg_loss_f += render_pkg["op_reg_loss_f"]
        if "sc_reg_loss_f" in render_pkg:
            total_sc_reg_loss_f += render_pkg["sc_reg_loss_f"]
        if "col_reg_loss_f" in render_pkg:
            total_col_reg_loss_f += render_pkg["col_reg_loss_f"]

        if frame.dim() == 3 and frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1)
        video_frames.append(frame)

        moving_app_state = detach_state(render_pkg["new_moving_app_state"])
        fallen_app_state = detach_state(render_pkg["new_fallen_app_state"])
        prev_x_rendered_moving = render_pkg["new_x_rendered_moving"].detach()
        prev_v_corrected_moving = render_pkg["new_v_corrected_moving"].detach()
        prev_x_rendered_fallen = render_pkg["new_x_rendered_fallen"].detach()
        prev_v_corrected_fallen = render_pkg["new_v_corrected_fallen"].detach()

    n = sds_video_len if sds_video_len > 0 else 1
    video_tensor = torch.stack(video_frames, dim=0).unsqueeze(0)
    del video_frames
    return (video_tensor,
            total_reg_loss_xyz / n, total_reg_loss_vel / n, total_reg_loss_rot / n,
            total_op_reg_loss_m / n, total_sc_reg_loss_m / n, total_col_reg_loss_m / n,
            total_op_reg_loss_f / n, total_sc_reg_loss_f / n, total_col_reg_loss_f / n)


def precompute_effect_data(args, cam_list, gaussians, gs_pipeline, background, effect_renderer, total_steps,
                           save_init=True):
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

    print(f"Rendering full video for {num_steps} frames...")

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
    print(f"Full video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Physics-Guided Score Distillation optimization.")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--simulation_dir", type=str, required=True,
                        help="Path to precomputed particle simulation data.")
    parser.add_argument("--mesh_path", type=str, required=True,
                        help="Path to the scene mesh file (e.g., fuse_post.ply).")
    parser.add_argument("--effect_config_path", type=str, required=True,
                        help="Path to the effect JSON config (e.g., configs/snow.json).")
    parser.add_argument("--save_dir_name", type=str, required=True, help="Directory to save checkpoints and outputs.")
    parser.add_argument("--effect_type", type=str, default="snow", choices=["snow", "rain", "fog", "sand"],
                        help="Type of weather effect.")
    parser.add_argument("--bg_path", type=str, default="")

    guidance_group = parser.add_argument_group("Guidance")
    guidance_group.add_argument("--pretrained_model_name_or_path", type=str, default="THUDM/CogVideoX-2B")
    guidance_group.add_argument("--half_precision_weights", action="store_true", default=True)
    guidance_group.add_argument("--gpu_size", type=str, default="big", choices=["big", "small"])
    guidance_group.add_argument("--guidance_scale", type=float, default=100.0)
    guidance_group.add_argument("--min_step", type=int, default=20)
    guidance_group.add_argument("--max_step", type=int, default=980)
    guidance_group.add_argument("--sds_variant", type=str, default="sds",
                                choices=['sds', 'nfsd', 'cfgpp', 'nfsd+cfgpp'])
    guidance_group.add_argument("--num_frames", type=int, default=9)
    guidance_group.add_argument("--enable_model_cpu_offload", action="store_true", default=True)
    guidance_group.add_argument("--enable_attention_slicing", action="store_true", default=True)
    guidance_group.add_argument("--enable_vae_slicing", action="store_true", default=True)
    guidance_group.add_argument("--enable_vae_tiling", action="store_true")
    guidance_group.add_argument("--enable_vae_gradient_checkpointing", action="store_true", default=True)
    guidance_group.add_argument("--enable_sequential_cpu_offload", action="store_true")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument("--lr", type=float, default=1e-4)
    train_group.add_argument("--steps", type=int, default=1001)
    train_group.add_argument("--save_interval", type=int, default=50)
    train_group.add_argument("--prompt", type=str, required=True, help="Text prompt for SDS guidance.")
    train_group.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted, artifacts")
    train_group.add_argument("--xyz_reg_weight_moving", type=float, default=0.1)
    train_group.add_argument("--vel_reg_weight_moving", type=float, default=0.1)
    train_group.add_argument("--rot_reg_weight_moving", type=float, default=0.1)
    train_group.add_argument("--op_reg_weight_moving", type=float, default=1.0)
    train_group.add_argument("--sc_reg_weight_moving", type=float, default=1.0)
    train_group.add_argument("--col_reg_weight_moving", type=float, default=35.0)
    train_group.add_argument("--op_reg_weight_fallen", type=float, default=35.0)
    train_group.add_argument("--sc_reg_weight_fallen", type=float, default=35.0)
    train_group.add_argument("--col_reg_weight_fallen", type=float, default=35.0)
    train_group.add_argument("--moving_hidden_dim", type=int, default=128)
    train_group.add_argument("--fallen_hidden_dim", type=int, default=128)
    train_group.add_argument("--sds_video_len", type=int, default=9)
    train_group.add_argument("--sds_downsample_factor", type=float, default=1.0)
    train_group.add_argument("--num_sim_steps", type=int, default=500)
    train_group.add_argument("--only_fallen", action="store_true")
    train_group.add_argument("--load_checkpoint", type=str, default="")

    render_group = parser.add_argument_group("Render")
    render_group.add_argument("--render_only", action="store_true")
    render_group.add_argument("--load_render_checkpoint", type=str, default="")
    render_group.add_argument("--render_output_path", type=str, default="renders/final_video.mp4")
    render_group.add_argument("--render_fps", type=int, default=25)
    render_group.add_argument("--render_cam_start_idx", type=int, default=0)
    render_group.add_argument("--render_cam_end_idx", type=int, default=10)

    args = get_combined_args(parser)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.save_dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load CogVideoX guidance
    mock_cfg = OmegaConf.create({"guidance": {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "half_precision_weights": args.half_precision_weights,
        "gpu_size": args.gpu_size, "guidance_scale": args.guidance_scale,
        "min_step": args.min_step, "max_step": args.max_step,
        "sds_variant": args.sds_variant, "num_frames": args.num_frames, "mode": "t2v",
        "enable_model_cpu_offload": args.enable_model_cpu_offload,
        "enable_attention_slicing": args.enable_attention_slicing,
        "enable_vae_slicing": args.enable_vae_slicing,
        "enable_vae_tiling": args.enable_vae_tiling,
        "enable_vae_gradient_checkpointing": args.enable_vae_gradient_checkpointing,
        "enable_sequential_cpu_offload": args.enable_sequential_cpu_offload,
    }})
    print("Loading CogVideoX guidance...")
    guidance = CogVideoXDirectGuidance(mock_cfg)

    # Load scene and effect renderer
    print("Loading 3D scene...")
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
    fixed_cam = train_cameras[0]

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Initialize neural dynamics models
    print("Initializing neural dynamics models...")
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

    if args.only_fallen:
        gm_moving.return_orig = True
        optim_params = list(gm_fallen.parameters())
    else:
        optim_params = list(gm_moving.parameters()) + list(gm_fallen.parameters())

    optimizer = Adam(optim_params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    scaler = GradScaler()

    # Precompute effect data
    precompute_effect_data(
        args, train_cameras, gaussians, gs_pipeline, background,
        effect_renderer, args.num_sim_steps, save_init=not args.render_only
    )

    # Load checkpoint
    load_path = args.load_render_checkpoint if args.render_only else args.load_checkpoint
    start_step = 1
    if load_path and os.path.exists(load_path):
        print(f"Loading checkpoint from {load_path}")
        ckpt = torch.load(load_path, map_location=device)
        gm_moving.load_state_dict(ckpt["moving_state"])
        gm_fallen.load_state_dict(ckpt["fallen_state"])
        if not args.render_only:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"] + 1
            print(f"Resuming from step {start_step}")

    if args.render_only:
        render_full_video(
            args, train_cameras, gaussians, gs_pipeline, background,
            effect_renderer, gm_moving, gm_fallen, device
        )
        return

    # Training loop
    print(f"Training from step {start_step} to {args.steps}...")
    reg_weight_xyz = args.xyz_reg_weight_moving
    reg_weight_vel = args.vel_reg_weight_moving
    reg_weight_rot = args.rot_reg_weight_moving
    op_reg_weight_m = args.op_reg_weight_moving
    sc_reg_weight_m = args.sc_reg_weight_moving
    col_reg_weight_m = args.col_reg_weight_moving
    op_reg_weight_f = args.op_reg_weight_fallen
    sc_reg_weight_f = args.sc_reg_weight_fallen
    col_reg_weight_f = args.col_reg_weight_fallen

    guidance.calc_text_embeddings(prompt=args.prompt, negative_prompt=args.negative_prompt)

    log_file_path = out_dir / "training_log.txt"

    for step in tqdm(range(start_step, args.steps + 1), desc="Optimizing"):
        progress = (step - start_step) / (args.steps - start_step + 1e-9)

        # Progressive timestep annealing
        start_max_step = args.max_step
        end_max_step = args.min_step + (args.max_step - args.min_step) * 0.75
        current_max_step = int(start_max_step * (1 - progress) + end_max_step * progress)
        if current_max_step <= args.min_step:
            current_max_step = args.min_step + 1
        guidance.max_step = current_max_step
        guidance.min_step = args.min_step

        torch.cuda.empty_cache()
        optimizer.zero_grad()

        cam = random.choice(train_cameras)
        t_start = random.randint(1, args.num_sim_steps - args.sds_video_len)

        rendered_video, reg_loss_xyz, reg_loss_vel, reg_loss_rot, \
            op_reg_loss_m, sc_reg_loss_m, col_reg_loss_m, \
            op_reg_loss_f, sc_reg_loss_f, col_reg_loss_f = render_sds_clip(
            cam, gaussians, gs_pipeline, background, effect_renderer,
            t_start, args.sds_video_len, args.sds_downsample_factor,
            gm_moving, gm_fallen, device
        )

        with autocast():
            result = guidance(
                rgb_BCHW=rendered_video,
                num_frames=args.sds_video_len,
                current_step=step
            )

            loss_sds = result["loss_sds_video"]
            # SDS-adaptive physics guidance: scale regularization by |L_SDS|
            sds_mag = loss_sds.detach()

            loss = (loss_sds +
                    reg_weight_xyz * reg_loss_xyz * sds_mag +
                    reg_weight_vel * reg_loss_vel * sds_mag +
                    reg_weight_rot * reg_loss_rot * sds_mag)

            loss += (op_reg_weight_m * sds_mag * op_reg_loss_m)
            loss += (sc_reg_weight_m * sds_mag * sc_reg_loss_m)
            loss += (col_reg_weight_m * sds_mag * col_reg_loss_m)
            loss += (op_reg_weight_f * sds_mag * op_reg_loss_f)
            loss += (sc_reg_weight_f * sds_mag * sc_reg_loss_f)
            loss += (col_reg_weight_f * sds_mag * col_reg_loss_f)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.save_interval == 0 or step == args.steps:
            loss_item = loss.item()
            loss_sds_item = loss_sds.item()

            log_line = (f"Step {step} | Loss: {loss_item:.6f} | SDS: {loss_sds_item:.6f} | "
                        f"XYZ: {reg_loss_xyz:.6f} | Vel: {reg_loss_vel:.6f} | Rot: {reg_loss_rot:.6f}")
            tqdm.write(log_line)

            try:
                with open(log_file_path, 'a') as f:
                    f.write(log_line + "\n")
            except Exception:
                pass

            with torch.no_grad():
                diag_clip = render_sds_clip(
                    fixed_cam, gaussians, gs_pipeline, background, effect_renderer,
                    t_start, args.sds_video_len, args.sds_downsample_factor,
                    gm_moving, gm_fallen, device
                )[0]
                frames_vis = diag_clip.squeeze(0).cpu()
                video_path = out_dir / f"video_step_{step:06d}.mp4"
                save_frames_as_video(frames_vis, video_path)

                ckpt = {
                    "step": step,
                    "moving_state": gm_moving.state_dict(),
                    "fallen_state": gm_fallen.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(ckpt, os.path.join(out_dir, f"checkpoint_{step:06d}.pth"))

        del rendered_video, result, loss_sds, loss
        del reg_loss_xyz, reg_loss_vel, reg_loss_rot
        del op_reg_loss_m, sc_reg_loss_m, col_reg_loss_m
        del op_reg_loss_f, sc_reg_loss_f, col_reg_loss_f

    print("Training finished.")
    render_full_video(
        args, train_cameras, gaussians, gs_pipeline, background,
        effect_renderer, gm_moving, gm_fallen, device
    )


if __name__ == "__main__":
    main()
