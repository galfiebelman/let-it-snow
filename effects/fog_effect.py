import json
import torch
import os
import numpy as np
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import open3d as o3d


class FogEffectRenderer:

    def __init__(self,mesh_path, simulation_dir, config_path):
        self.simulation_dir = simulation_dir
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        bounds = self.mesh.get_axis_aligned_bounding_box()
        self.min_bound = torch.tensor(bounds.min_bound, device='cuda')
        self.max_bound = torch.tensor(bounds.max_bound, device='cuda')
        # self.radius = (self.max_bound-self.min_bound).max()
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config = config

    def get_fog_gaussians(self, t):
        xyzs = []
        xyzs.append(torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_pos.npz")["arr_0"]).float())        
        fog_xyz = torch.vstack(xyzs).cuda()
        fog_scales = torch.ones_like(fog_xyz) * self.config["scale"]
        fog_rotations = torch.zeros((len(fog_xyz), 4), device="cuda")
        fog_rotations[:, 0] = 1.0  # Identity rotation
        fog_opacities = torch.ones((len(fog_xyz), 1), device="cuda") * self.config["opacity"]
        fog_colors = torch.tensor(self.config["colors"], device="cuda").expand(len(fog_xyz), -1)        
        return fog_xyz, fog_scales, fog_rotations, fog_opacities, fog_colors
    
    def render(self,
            viewpoint_camera,
            pc: GaussianModel,
            pipe,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            t=0):
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
            render_features=False,
            render_gaussian_idx=False,
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

        # Get fog gaussians
        fog_xyz, fog_scales, fog_rotations, fog_opacities, fog_colors = self.get_fog_gaussians(t)
        
        tot_means3d = torch.concat([means3D, fog_xyz]) 
        tot_scales = torch.concat([scales, fog_scales]) 
        tot_rotations = torch.concat([rotations, fog_rotations]) 
        tot_opacity = torch.concat([opacity, fog_opacities]) 
        tot_colors_precomp = torch.concat([colors_precomp, fog_colors]) 

        # Get view-independent features (distill features) for each Gaussian for rendering.
        distill_feats = pc.get_distill_features


        rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
            means3D=tot_means3d,
            means2D=means2D,
            shs=shs,
            colors_precomp=tot_colors_precomp,
            opacities=tot_opacity,
            scales=tot_scales,
            rotations=tot_rotations,
            cov3D_precomp=cov3D_precomp,
            distill_feats=distill_feats
        )

        return {
            "render": rendered_image,
            "render_feat": rendered_feat,
            "render_depth": rendered_depth,
            "render_gaussian_idx": rendered_gaussian_idx,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii
        }
