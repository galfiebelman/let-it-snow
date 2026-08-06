#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import open3d as o3d

def quaternion_to_rotation_matrix_batched(q):
    # q: (N, 4) quaternions in w,x,y,z format
    w, x, y, z = q.unbind(-1)
    
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1)
    ], dim=-2) # Shape: (N,3,3)
    
    return R

def get_covariance_batched(scales, quaternions):
    # scales: (N, 3)
    # quaternions: (N, 4)
    
    # Get rotation matrices (N,3,3)
    R = quaternion_to_rotation_matrix_batched(quaternions)
    
    # Create diagonal scale matrices (N,3,3)
    S = torch.diag_embed(scales * scales)
    
    # Batched matrix multiplication
    return R @ S @ R.transpose(-2,-1)

def gaussian_intersections_batched(gaussians_A, gaussians_B, threshold=0.1):
    # Filter upward-facing gaussians in B
    up_vector = torch.tensor([0., 1., 0.], device=gaussians_B['normals'].device)
    upward_mask = (gaussians_B['normals'] @ up_vector) > 0.7
    
    # Filter B gaussians
    B_pos = gaussians_B['positions'][upward_mask]
    B_scales = gaussians_B['scales'][upward_mask]
    B_rotations = gaussians_B['rotations'][upward_mask]
    
    # Compute all pairwise differences (num_A, num_B_filtered, 3)
    pos_diff = gaussians_A['positions'].cpu().unsqueeze(1) - B_pos.cpu().unsqueeze(0)
    
    # Compute covariance matrices (num_A, 3, 3) and (num_B_filtered, 3, 3)
    cov_A = get_covariance_batched(gaussians_A['scales'], gaussians_A['rotations'])
    cov_B = get_covariance_batched(B_scales, B_rotations)
    
    # Broadcast covariance sums (num_A, num_B_filtered, 3, 3)
    cov_sum = cov_A.unsqueeze(1).cpu() + cov_B.unsqueeze(0).cpu()
    
    # Compute Mahalanobis distances
    cov_sum_inv = torch.linalg.inv(cov_sum).cpu()
    mahalanobis = torch.sqrt(torch.einsum('...i,...ij,...j->...', pos_diff, cov_sum_inv, pos_diff))
    
    # Get intersecting pairs
    intersecting = mahalanobis < threshold
    return torch.where(intersecting)

def render(viewpoint_camera,
           pc : GaussianModel,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           render_features = False,
           render_gaussian_idx = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_features=render_features,
        render_gaussian_idx=render_gaussian_idx,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)
        else:
            shs = pc.get_features  # (N, 16 ,3)
    else:
        colors_precomp = override_color
    
    # Get view-independent features (distill features) for each Gaussian for rendering.
    distill_feats = pc.get_distill_features.detach()

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        distill_feats = distill_feats)
    
    # Default synthetic datasets settings
    # rendered_image, radii = rasterizer(
    #     means3D = means3D,                # (N, 3)
    #     means2D = means2D,                # (N, 3)
    #     shs = shs,                        # (N, 16, 3)
    #     colors_precomp = colors_precomp,  # None
    #     opacities = opacity,              # 
    #     scales = scales,                  # (N, 3)
    #     rotations = rotations,            # (N, 4)
    #     cov3D_precomp = cov3D_precomp)    # None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_feat": rendered_feat,
            "render_depth": rendered_depth,
            "render_gaussian_idx": rendered_gaussian_idx,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_snow(viewpoint_camera,
           pc : GaussianModel,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           render_features = False,
           render_gaussian_idx = False, edit_dict=None, t=0,saved=None, scene=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_features=render_features,
        render_gaussian_idx=render_gaussian_idx,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)


    # for obj_dict in edit_dict["objects"]:
    #         for action_dict in obj_dict["actions"]:
    #             particles_trajectory_tn3 = action_dict['particles_trajectory_tn3']
    dir_path = "outputs/same_min"
    xyzs = []
    for dir_name in sorted(os.listdir(dir_path)):
        xyzs.append(torch.from_numpy(np.load(f"{dir_path}/{dir_name}/{t}_pos.npz")["arr_0"]).float())
    # xyzs.append(torch.from_numpy(np.load(f"split_test/{t}.npz")["arr_0"]).float())
    snow_xyz = torch.vstack(xyzs).cuda()
    snow_scales = torch.ones_like(snow_xyz) * 0.005
    qs = []
    for dir_name in sorted(os.listdir(dir_path)):
        qs.append(torch.from_numpy(np.load(f"{dir_path}/{dir_name}/{t}_rot.npz")["arr_0"]).float())
    snow_rotations = torch.zeros((len(snow_xyz), 4), device="cuda")
    snow_rotations[:, 0] = 1.0
    # snow_rotations = torch.vstack(qs).cuda()
    snow_opacities = torch.ones((len(snow_xyz),1), device="cuda") *0.65
    colors = torch.ones_like(snow_xyz)
    colors[:,0] =0.941
    colors[:,1] =0.941
    colors[:,2] =0.961

    statics = []
    per_step = len(snow_xyz)//len(os.listdir(dir_path))
    for idx, dir_name in enumerate(sorted(os.listdir(dir_path))):
        to_add = torch.ones(per_step).bool()
        if saved is not None and dir_name in saved:
            saved_xyz = torch.from_numpy(saved[dir_name]["xyz"]).float().cuda()           
            saved_static = torch.from_numpy(saved[dir_name]["static"]).bool().cuda()            
            snow_xyz[idx * per_step: (idx+1) * per_step][:len(saved_static)][saved_static] = saved_xyz
            # to_add[:len(saved_static)][saved_static] = True
        statics.append(to_add.unsqueeze(-1))
        if os.path.exists(f"{dir_path}/{dir_name}/{t}_static.npz"):
            snow_static = torch.from_numpy(np.load(f"{dir_path}/{dir_name}/{t}_static.npz")["arr_0"]).bool().cuda()
            if saved is not None and dir_name in saved:
                if len(saved_static) != len(snow_static):
                    snow_static = torch.concatenate([snow_static[:len(saved_static)] & ~saved_static, snow_static[len(saved_static):]]) 
                else:
                    snow_static = snow_static & ~saved_static
            if torch.sum(snow_static) == 0:
                continue
            static_xyz = snow_xyz[idx * per_step: (idx+1) * per_step][snow_static].cpu().numpy()
            hit_points, surface_normals, dists = query_surface_snow(static_xyz, scene)
            if np.min(dists)>0.01:
                continue
            hit_points = torch.from_numpy(hit_points).float().cuda()
            surface_normals = torch.from_numpy(surface_normals).float().cuda()
            offset = 0.002
            static = hit_points - surface_normals * offset
            # colors[idx * per_step: (idx+1) * per_step,0][snow_static] = 1
            # colors[idx * per_step: (idx+1) * per_step,1][snow_static] = 0
            # colors[idx * per_step: (idx+1) * per_step,2][snow_static] = 0
            if saved is None or dir_name not in saved:
                if saved is None:
                    saved = {}
                saved[dir_name] = {"xyz": static.cpu().numpy(), "static": snow_static.cpu().numpy()}
            else:
                new_xyz = torch.concat([saved_xyz, static]).cpu().numpy()
                if len(saved_static) != len(snow_static):
                    new_static = torch.concatenate([snow_static[:len(saved_static)] | saved_static, snow_static[len(saved_static):]]) 
                else:
                    new_static = snow_static | saved_static
                saved[dir_name] = {"xyz": new_xyz, "static": new_static.cpu().numpy()}

    statics = torch.vstack(statics).squeeze().cuda()
    tot_means3d = torch.concat([means3D, snow_xyz[statics]])
    tot_scales = torch.concat([scales, snow_scales[statics]])
    tot_rotations = torch.concat([rotations, snow_rotations[statics]])
    tot_opacity = torch.concat([opacity, snow_opacities[statics]])
    tot_colors_precomp = torch.concat([colors_precomp, colors[statics]])

    # Get view-independent features (distill features) for each Gaussian for rendering.
    distill_feats = pc.get_distill_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
        means3D = tot_means3d,
        means2D = means2D,
        shs = shs,
        colors_precomp = tot_colors_precomp,
        opacities = tot_opacity,
        scales = tot_scales,
        rotations = tot_rotations,
        cov3D_precomp = cov3D_precomp,
        distill_feats = distill_feats)
    
    return {"render": rendered_image,
            "render_feat": rendered_feat,
            "render_depth": rendered_depth,
            "render_gaussian_idx": rendered_gaussian_idx,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, "saved":saved}
def query_surface_snow(points, scene):
    points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    batch_results = scene.compute_closest_points(points_tensor)
    
    hit_points = batch_results['points'].numpy()
    normals = batch_results['primitive_normals'].numpy()
    distances = np.linalg.norm(hit_points - points, axis=-1)
    
    return hit_points, normals, distances

def query_surface(points, scene):
    """Query surface point and normal from precomputed mesh"""
    points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    batch_results = scene.compute_closest_points(points_tensor)
    return (
        batch_results['points'].numpy(),  # Hit point
        batch_results['primitive_normals'].numpy()  # Surface normal
    )

def compute_quaternion_from_gradient(gradient):
    """
    Compute quaternion (r,x,y,z format) given SDF gradient (surface normal)
    Args:
        gradient: (N, 3) tensor of surface normals
    Returns:
        quaternion: (N, 4) quaternions in (r,x,y,z) format
    """
    # Get rotation matrix first
    up = gradient / (torch.norm(gradient, dim=-1, keepdim=True) + 1e-8)
    right = torch.ones_like(up)
    right[:,1] = 0
    right = torch.cross(right, up)
    right = right / (torch.norm(right, dim=-1, keepdim=True) + 1e-8)
    forward = torch.cross(up, right)
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-8)
    R = torch.stack([right, up, forward], dim=-1)

    # Convert rotation matrix to quaternion
    trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
    
    # Initialize quaternion storage
    quat = torch.zeros((R.shape[0], 4), device=R.device)
    
    # Case 1: trace > 0
    mask_1 = trace > 0
    if mask_1.any():
        S = 2 * torch.sqrt(1.0 + trace[mask_1])
        quat[mask_1,0] = 0.25 * S
        quat[mask_1,1] = (R[mask_1,2,1] - R[mask_1,1,2]) / S
        quat[mask_1,2] = (R[mask_1,0,2] - R[mask_1,2,0]) / S
        quat[mask_1,3] = (R[mask_1,1,0] - R[mask_1,0,1]) / S
    
    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask_2 = (~mask_1) & (R[:,0,0] > R[:,1,1]) & (R[:,0,0] > R[:,2,2])
    if mask_2.any():
        S = 2 * torch.sqrt(1.0 + R[mask_2,0,0] - R[mask_2,1,1] - R[mask_2,2,2])
        quat[mask_2,0] = (R[mask_2,2,1] - R[mask_2,1,2]) / S
        quat[mask_2,1] = 0.25 * S
        quat[mask_2,2] = (R[mask_2,0,1] + R[mask_2,1,0]) / S
        quat[mask_2,3] = (R[mask_2,0,2] + R[mask_2,2,0]) / S
    
    # Case 3: R[1,1] > R[2,2]
    mask_3 = (~mask_1) & (~mask_2) & (R[:,1,1] > R[:,2,2])
    if mask_3.any():
        S = 2 * torch.sqrt(1.0 + R[mask_3,1,1] - R[mask_3,0,0] - R[mask_3,2,2])
        quat[mask_3,0] = (R[mask_3,0,2] - R[mask_3,2,0]) / S
        quat[mask_3,1] = (R[mask_3,0,1] + R[mask_3,1,0]) / S
        quat[mask_3,2] = 0.25 * S
        quat[mask_3,3] = (R[mask_3,1,2] + R[mask_3,2,1]) / S
    
    # Case 4: remaining cases
    mask_4 = (~mask_1) & (~mask_2) & (~mask_3)
    if mask_4.any():
        S = 2 * torch.sqrt(1.0 + R[mask_4,2,2] - R[mask_4,0,0] - R[mask_4,1,1])
        quat[mask_4,0] = (R[mask_4,1,0] - R[mask_4,0,1]) / S
        quat[mask_4,1] = (R[mask_4,0,2] + R[mask_4,2,0]) / S
        quat[mask_4,2] = (R[mask_4,1,2] + R[mask_4,2,1]) / S
        quat[mask_4,3] = 0.25 * S

    # Normalize quaternions
    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    
    return quat

def render_rain(viewpoint_camera,
           pc : GaussianModel,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           render_features = False,
           render_gaussian_idx = False, edit_dict=None, t=0, saved=None, scene=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_features=render_features,
        render_gaussian_idx=render_gaussian_idx,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)

    xyzs = []
    xyzs.append(torch.from_numpy(np.load(f"split_rain/{t}_pos.npz")["arr_0"]).float())
    rain_xyz = torch.vstack(xyzs).cuda()
    rain_scales = torch.ones_like(rain_xyz) *0.007
    # rain_scales[:,0] *= 0.002
    # rain_scales[:,1] *= 0.006
    # rain_scales[:,2] *= 0.002
    # rain_rotations = torch.from_numpy(np.load(f"split_rain/{t}_rot.npz")["arr_0"]).float().cuda()
    rain_rotations = torch.zeros((len(rain_xyz), 4), device="cuda")
    rain_rotations[:, 0] = 1.0
    rain_opacities = torch.ones((len(rain_xyz),1), device="cuda") *0.65
    colors = torch.ones_like(rain_xyz)
    colors[:,0] =0.941
    colors[:,1] =0.941
    colors[:,2] =0.961
    sdf_values = scene.query_sdf(rain_xyz)
    surface_normals = scene.compute_sdf_gradients(rain_xyz)
    mask =torch.abs(sdf_values) < 0.05
    savedz=False
    if torch.sum(mask)>0:
        # colors[:,0][mask] =1
        # colors[:,1][mask] =0
        # colors[:,2][mask] =0

        savedz=True

    if os.path.exists(f"split_rain/{t}_static.npz"):
        rain_static = torch.from_numpy(np.load(f"split_test_rain/{t}_static.npz")["arr_0"]).bool().cuda()
        static_xyz = rain_xyz[rain_static].cpu().numpy()
        hit_points, surface_normals = query_surface(static_xyz, scene)
        hit_points = torch.from_numpy(hit_points).float().cuda()
        surface_normals = torch.from_numpy(surface_normals).float().cuda()
        offset = 0.00
        static = hit_points + surface_normals * offset
        static_rotations = rain_rotations[rain_static]
        savedz=True  
    a=False
    if saved is not None:
        saved_indices = saved["indices"].cuda()
        saved_mask = torch.zeros(len(rain_xyz)).bool().cuda()
        saved_mask[saved_indices] = True
        rain_xyz[saved_mask] = saved["xyz"].cuda()
        # rain_opacities[saved_mask] = 0.95 * saved["opacity"].cuda()
        rain_rotations[saved_mask] = saved["rots"].cuda()
        # saved_xyz = torch.from_numpy(saved["xyz"]).float().cuda()           
        # saved_rotations = torch.from_numpy(saved["rotations"]).float().cuda() 
        # saved_scales = torch.ones_like(saved_xyz) *0.005
        # # saved_scales[:,0] = 0.002
        # # saved_scales[:,0] = 0.006
        # # saved_scales[:,0] = 0.002
        # saved_opacities = torch.ones((len(saved_xyz),1), device="cuda") *0.5
        # saved_colors = torch.tensor([0.3, 0.32, 0.35], device="cuda").expand(len(saved_xyz),-1)
        # rain_xyz = torch.concat([rain_xyz, saved_xyz])
        # rain_scales = torch.concat([rain_scales, saved_scales])
        # rain_opacities = torch.concat([rain_opacities, saved_opacities])
        # rain_rotations = torch.concatenate([rain_rotations, saved_rotations])
        # colors = torch.concat([colors, saved_colors])        
        # a=True       
    if savedz:
        indices = torch.nonzero(mask)
        if saved is None:
            # saved = {"xyz": static.cpu().numpy(), "rotations": static_rotations.cpu().numpy()}
            end =  rain_xyz[mask] + sdf_values[mask] * surface_normals[mask]
            scale = rain_scales[mask]
            opacitys = rain_opacities[mask]
            count = torch.ones(len(end)).cuda()
            xyz = torch.lerp(rain_xyz[mask], end,1/count)
            sdf_rotations = rain_rotations[mask]
            saved = {"indices":indices.cpu(), "xyz": xyz.cpu(), "scales":scale.cpu(), "opacity":opacitys.cpu(), "count":count.cpu(), "end":end, 
                     "rots":sdf_rotations.cpu()}
        else:
            # new_xyz = torch.concat([saved_xyz, static]).cpu().numpy()
            # new_rotations = torch.concat([saved_rotations, static_rotations]).cpu().numpy()
            # saved = {"xyz": new_xyz, "rotations": new_rotations}
            new_mask = torch.zeros(len(rain_xyz)).bool().cuda()
            new_indices = indices[~torch.isin(indices, saved_indices)]
            new_mask[new_indices] = True
            if torch.sum(new_mask)>0:
                end =  rain_xyz[new_mask] + sdf_values[new_mask].unsqueeze(-1) * surface_normals[new_mask]
                if not torch.allclose(end,rain_xyz[new_mask]):
                    a=1
                scale = rain_scales[new_mask]
                opacitys = rain_opacities[new_mask]
                count = torch.ones(len(end)).cuda()
                new_indices = torch.concatenate([saved_indices, new_indices.unsqueeze(-1)])
                new_end = torch.concatenate([saved["end"].cuda(), end])
                new_scale = torch.concatenate([saved["scales"].cuda(), scale])
                new_opacitys = torch.concatenate([saved["opacity"].cuda(), opacitys])
                new_count = saved["count"].cuda()
                new_count[new_count>0] +=1
                new_count[new_count>3]=-1
                new_count = torch.concatenate([new_count, count])
                new_xyz = torch.concatenate([saved["xyz"].cuda(), rain_xyz[new_mask]])
                count_mask = new_count > 0
                xyz = torch.lerp(new_xyz[count_mask], new_end[count_mask],1/(4-new_count[count_mask].unsqueeze(-1)))
                new_xyz[count_mask] = xyz
                done_falling_mask = new_count == 3
                new_surface_normals = scene.compute_sdf_gradients(new_xyz[done_falling_mask])
                sdf_rotations = compute_quaternion_from_gradient(new_surface_normals)
                new_rots = torch.concatenate([saved["rots"].cuda(), rain_rotations[new_mask]])
                new_rots[done_falling_mask] = sdf_rotations
                new_opacitys[done_falling_mask] = 1.0
                saved = {"indices":new_indices.cpu(), "xyz": new_xyz.cpu(), "end": new_end.cpu(), "scales":new_scale.cpu(), "opacity":new_opacitys.cpu(), 
                         "count":new_count.cpu(), "rots":new_rots.cpu()}
    

    tot_means3d = torch.concat([means3D, rain_xyz]) #if a else means3D
    tot_scales = torch.concat([scales, rain_scales]) #if a else scales
    tot_rotations = torch.concat([rotations, rain_rotations]) #if a else rotations
    tot_opacity = torch.concat([opacity, rain_opacities]) #if a else opacity
    tot_colors_precomp = torch.concat([colors_precomp, colors]) #if a else colors_precomp

    # Get view-independent features (distill features) for each Gaussian for rendering.
    distill_feats = pc.get_distill_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
        means3D = tot_means3d,
        means2D = means2D,
        shs = shs,
        colors_precomp = tot_colors_precomp,
        opacities = tot_opacity,
        scales = tot_scales,
        rotations = tot_rotations,
        cov3D_precomp = cov3D_precomp,
        distill_feats = distill_feats)
    
    return {"render": rendered_image,
            "render_feat": rendered_feat,
            "render_depth": rendered_depth,
            "render_gaussian_idx": rendered_gaussian_idx,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, "saved":saved}

def render_rain2(viewpoint_camera,
           pc : GaussianModel,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           render_features = False,
           render_gaussian_idx = False, edit_dict=None, t=0, saved=None, scene=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_features=render_features,
        render_gaussian_idx=render_gaussian_idx,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)

    xyzs = []
    xyzs.append(torch.from_numpy(np.load(f"split_rain2/{t}_pos.npz")["arr_0"]).float())
    rain_xyz = torch.vstack(xyzs).cuda()
    rain_scales = torch.ones_like(rain_xyz) *0.005
    # rain_rotations = torch.from_numpy(np.load(f"split_rain2/{t}_rot.npz")["arr_0"]).float().cuda()
    rain_rotations = torch.zeros((len(rain_xyz), 4), device="cuda")
    rain_rotations[:, 0] = 1.0
    rain_opacities = torch.ones((len(rain_xyz),1), device="cuda") *0.65
    colors = torch.tensor([0.95, 0.95, 0.96], device="cuda").expand(len(rain_xyz),-1)

    if os.path.exists(f"split_rain2/{t}_static.npz"):
        new_static = np.load(f"split_rain2/{t}_static.npz")["arr_0"]
        # static_xyz = rain_xyz[rain_static].cpu().numpy()
        hit_points, surface_normals = query_surface(new_static, scene)
        hit_points = torch.from_numpy(hit_points).float().cuda()
        surface_normals = torch.from_numpy(surface_normals).float().cuda()
        offset = 0.01
        new_xyz = hit_points + surface_normals * offset
        new_scales =  torch.ones_like(new_xyz) *0.01
        new_rotations = torch.zeros((len(new_xyz), 4), device="cuda")
        new_rotations[:, 0] = 1.0
        new_opacities = torch.ones((len(new_xyz),1), device="cuda")
        new_colors = torch.tensor([0.95, 0.95, 0.96], device="cuda").expand(len(new_xyz),-1)

        if saved is None:
            saved = {"xyz": new_xyz.cpu().numpy(), "rotations": new_rotations.cpu().numpy(), "scales":new_scales.cpu().numpy(), 
                     "opacity":new_opacities.cpu().numpy(), "color":new_colors.cpu().numpy()}

        else:
            new_xyz = torch.concat([torch.from_numpy(saved["xyz"]).float().cuda() , new_xyz]).cpu().numpy()
            new_rotations = torch.concat([torch.from_numpy(saved["rotations"]).float().cuda() , new_rotations]).cpu().numpy()
            new_scales = torch.concat([torch.from_numpy(saved["scales"]).float().cuda() , new_scales]).cpu().numpy()
            new_opacities = torch.concat([torch.from_numpy(saved["opacity"]).float().cuda() , new_opacities]).cpu().numpy()
            new_colors = torch.concat([torch.from_numpy(saved["color"]).float().cuda() , new_colors]).cpu().numpy()
            saved = {"xyz": new_xyz, "rotations": new_rotations, "scales":new_scales, "opacity":new_opacities, "color":new_colors}

    if saved is not None:
        saved_xyz = torch.from_numpy(saved["xyz"]).float().cuda()           
        saved_rotations = torch.from_numpy(saved["rotations"]).float().cuda() 
        saved_scales = torch.from_numpy(saved["scales"]).float().cuda()
        saved_opacities = torch.from_numpy(saved["opacity"]).float().cuda() 
        saved_colors = torch.from_numpy(saved["color"]).float().cuda()
        rain_xyz = torch.concat([rain_xyz, saved_xyz])
        rain_scales = torch.concat([rain_scales, saved_scales])
        rain_opacities = torch.concat([rain_opacities, saved_opacities])
        rain_rotations = torch.concatenate([rain_rotations, saved_rotations])
        colors = torch.concat([colors, saved_colors])        

    tot_means3d = torch.concat([means3D, rain_xyz]) #if savedz else means3D
    tot_scales = torch.concat([scales, rain_scales]) #if savedz else scales
    tot_rotations = torch.concat([rotations, rain_rotations]) #if savedz else rotations
    tot_opacity = torch.concat([opacity, rain_opacities]) #if savedz else opacity
    tot_colors_precomp = torch.concat([colors_precomp, colors]) #if savedz else colors_precomp

    # Get view-independent features (distill features) for each Gaussian for rendering.
    distill_feats = pc.get_distill_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
        means3D = tot_means3d,
        means2D = means2D,
        shs = shs,
        colors_precomp = tot_colors_precomp,
        opacities = tot_opacity,
        scales = tot_scales,
        rotations = tot_rotations,
        cov3D_precomp = cov3D_precomp,
        distill_feats = distill_feats)
    
    return {"render": rendered_image,
            "render_feat": rendered_feat,
            "render_depth": rendered_depth,
            "render_gaussian_idx": rendered_gaussian_idx,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, "saved":saved}


def render_rain3(viewpoint_camera,
           pc : GaussianModel,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           render_features = False,
           render_gaussian_idx = False, edit_dict=None, t=0, saved=None, scene=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_features=render_features,
        render_gaussian_idx=render_gaussian_idx,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)

    xyzs = []
    xyzs.append(torch.from_numpy(np.load(f"outputs/old/split_rain2/{t}_pos.npz")["arr_0"]).float())
    rain_xyz = torch.vstack(xyzs).cuda()
    rain_scales = torch.ones_like(rain_xyz) 
    rain_scales[:,0] *= 0.002
    rain_scales[:,1] *= 0.006
    rain_scales[:,2] *= 0.002   
    # rain_rotations = torch.from_numpy(np.load(f"split_rain2/{t}_rot.npz")["arr_0"]).float().cuda()
    rain_rotations = torch.zeros((len(rain_xyz), 4), device="cuda")
    rain_rotations[:, 0] = 1.0
    rain_opacities = torch.ones((len(rain_xyz),1), device="cuda") *0.25
    colors = torch.tensor([0.85, 0.87, 0.9], device="cuda").expand(len(rain_xyz),-1)
    
    if os.path.exists(f"outputs/old/split_rain2/{t}_static.npz"):
        new_static = torch.from_numpy(np.load(f"outputs/old/split_rain2/{t}_static.npz")["arr_0"]).cuda()
        if t % 2 ==0:
            scene.decay_wetness()
        min_bound = scene.min_bound
        max_bound = scene.max_bound
        within_bbox = ((means3D[:, 0] > min_bound[0]) & (means3D[:, 0] < max_bound[1])) & \
                            ((means3D[:, 1] > min_bound[1]) & (means3D[:, 1] < max_bound[1])) & \
                            ((means3D[:, 2] > min_bound[2]) & (means3D[:, 2] < max_bound[2]))        
        scene.add_wetness(new_static, intensity=0.5, radius=2)
        wetness_values = scene.get_wetness(means3D[within_bbox])
        wet_color = colors_precomp[within_bbox] * (1 - 0.2 * wetness_values[:, None])
        colors_precomp[within_bbox] = wet_color

    tot_means3d = torch.concat([means3D, rain_xyz]) #if savedz else means3D
    tot_scales = torch.concat([scales, rain_scales]) #if savedz else scales
    tot_rotations = torch.concat([rotations, rain_rotations]) #if savedz else rotations
    tot_opacity = torch.concat([opacity, rain_opacities]) #if savedz else opacity
    tot_colors_precomp = torch.concat([colors_precomp, colors]) #if savedz else colors_precomp

    # Get view-independent features (distill features) for each Gaussian for rendering.
    distill_feats = pc.get_distill_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
        means3D = tot_means3d,
        means2D = means2D,
        shs = shs,
        colors_precomp = tot_colors_precomp,
        opacities = tot_opacity,
        scales = tot_scales,
        rotations = tot_rotations,
        cov3D_precomp = cov3D_precomp,
        distill_feats = distill_feats)
    
    return {"render": rendered_image,
            "render_feat": rendered_feat,
            "render_depth": rendered_depth,
            "render_gaussian_idx": rendered_gaussian_idx,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, "saved":saved}

