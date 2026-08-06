import json
import torch
import os
import numpy as np
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.gaussian_model_dynamic import GaussianModelDynamic
from utils.sh_utils import eval_sh

from scene.gaussian_model import GaussianModel
from wetness_grid import WetnessGrid
import open3d as o3d


# class RainEffectRenderer():
#     def __init__(self,mesh_path, simulation_dir, config_path):
#         self.simulation_dir = simulation_dir
#         with open(config_path, 'r') as f:
#             config = json.load(f)
#         self.config = config
#         self.wetness_grid = WetnessGrid(mesh_path, grid_res=self.config["wetness_grid_res"], decay_rate=self.config["wetness_decay_rate"])
#
#     def render(self,
#             viewpoint_camera,
#             pc : GaussianModel,
#             pipe,
#             bg_color : torch.Tensor,
#             scaling_modifier = 1.0,
#             t=0):
#         """
#         Render the scene.
#
#         Background tensor (bg_color) must be on GPU!
#         """
#
#         # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#         screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#         try:
#             screenspace_points.retain_grad()
#         except:
#             pass
#
#         # Set up rasterization configuration
#         tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#         tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
#
#         raster_settings = GaussianRasterizationSettings(
#             image_height=int(viewpoint_camera.image_height),
#             image_width=int(viewpoint_camera.image_width),
#             tanfovx=tanfovx,
#             tanfovy=tanfovy,
#             bg=bg_color,
#             scale_modifier=scaling_modifier,
#             viewmatrix=viewpoint_camera.world_view_transform,
#             projmatrix=viewpoint_camera.full_proj_transform,
#             sh_degree=pc.active_sh_degree,
#             campos=viewpoint_camera.camera_center,
#             prefiltered=False,
#             render_features=False,
#             render_gaussian_idx=False,
#             debug=pipe.debug
#         )
#
#         rasterizer = GaussianRasterizer(raster_settings=raster_settings)
#
#         means3D = pc.get_xyz
#         means2D = screenspace_points
#         opacity = pc.get_opacity
#
#         scales = None
#         rotations = None
#         cov3D_precomp = None
#         if pipe.compute_cov3D_python:
#             cov3D_precomp = pc.get_covariance(scaling_modifier)
#         else:
#             scales = pc.get_scaling
#             rotations = pc.get_rotation
#
#         shs = None
#         colors_precomp = None
#         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)
#
#         xyzs = []
#         xyzs.append(torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_pos.npz")["arr_0"]).float())
#         rain_xyz = torch.vstack(xyzs).cuda()
#         rain_scales = torch.ones_like(rain_xyz)
#         rain_scales[:,0] *= self.config["scale"][0]
#         rain_scales[:,1] *= self.config["scale"][1]
#         rain_scales[:,2] *= self.config["scale"][2]
#         # rain_rotations = torch.from_numpy(np.load(f"split_rain2/{t}_rot.npz")["arr_0"]).float().cuda()
#         rain_rotations = torch.zeros((len(rain_xyz), 4), device="cuda")
#         rain_rotations[:, 0] = 1.0
#         rain_opacities = torch.ones((len(rain_xyz),1), device="cuda") * self.config["opacity"]
#         colors = torch.tensor(self.config["colors"], device="cuda").expand(len(rain_xyz),-1)
#
#         if os.path.exists(f"{self.simulation_dir}/{t}_static.npz"):
#             if "stump" in self.simulation_dir and t<90:
#                 a=1
#             elif "truck" in self.simulation_dir and t<110:
#                 b=1
#             else:
#                 new_static = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_static.npz")["arr_0"]).cuda()
#                 if t % self.config["decay_steps"] ==0:
#                     self.wetness_grid.decay_wetness()
#                 min_bound = self.wetness_grid.min_bound
#                 max_bound = self.wetness_grid.max_bound
#                 within_bbox = ((means3D[:, 0] > min_bound[0]) & (means3D[:, 0] < max_bound[0])) & \
#                                     ((means3D[:, 1] > min_bound[1]) & (means3D[:, 1] < max_bound[1])) & \
#                                     ((means3D[:, 2] > min_bound[2]) & (means3D[:, 2] < max_bound[2]))
#                 self.wetness_grid.add_wetness(new_static, intensity=0.5, radius=2)
#                 wetness_values = torch.clamp(self.wetness_grid.get_wetness(means3D[within_bbox]),0,1)
#                 wet_color = colors_precomp[within_bbox] * (1 - 0.2 * wetness_values[:, None])
#                 colors_precomp[within_bbox] = wet_color
#
#         tot_means3d = torch.concat([means3D, rain_xyz]) #if savedz else means3D
#         tot_scales = torch.concat([scales, rain_scales]) #if savedz else scales
#         tot_rotations = torch.concat([rotations, rain_rotations]) #if savedz else rotations
#         tot_opacity = torch.concat([opacity, rain_opacities]) #if savedz else opacity
#         tot_colors_precomp = torch.concat([colors_precomp, colors]) #if savedz else colors_precomp
#
#         # Get view-independent features (distill features) for each Gaussian for rendering.
#         distill_feats = pc.get_distill_features
#
#         # Rasterize visible Gaussians to image, obtain their radii (on screen).
#         rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
#             means3D = tot_means3d,
#             means2D = means2D,
#             shs = shs,
#             colors_precomp = tot_colors_precomp,
#             opacities = tot_opacity,
#             scales = tot_scales,
#             rotations = tot_rotations,
#             cov3D_precomp = cov3D_precomp,
#             distill_feats = distill_feats)
#
#         return {"render": rendered_image,
#                 "render_feat": rendered_feat,
#                 "render_depth": rendered_depth,
#                 "render_gaussian_idx": rendered_gaussian_idx,
#                 "viewspace_points": screenspace_points,
#                 "visibility_filter" : radii > 0,
#                 "radii": radii}

class RainEffectRenderer():
    def __init__(self,mesh_path, simulation_dir, config_path):
        self.simulation_dir = simulation_dir
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config = config
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh_o3d_t = o3d.io.read_triangle_mesh(mesh_path)
        self.bounds = mesh_o3d_t.get_axis_aligned_bounding_box()
        self.min_bound = self.bounds.min_bound
        self.max_bound = self.bounds.max_bound
        mesh_o3d_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d_t)
        _ = self.scene.add_triangles(mesh_o3d_t)
        self.fallen_rain_dict = None
        self.moving_rain_cache = {} # --- NEW: Cache for moving particle physics

    def get_moving_data(self, t):
        return self.moving_rain_cache.get(t)

    def query_surface(self, points):
        """Query surface point and normal from precomputed mesh"""
        points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        batch_results = self.scene.compute_closest_points(points_tensor)
        return (
            batch_results['points'].numpy(),  # Hit point
            batch_results['primitive_normals'].numpy()  # Surface normal
        )

    def precompute_moving_data(self, total_steps):
        """
        --- NEW FUNCTION ---
        Pre-loads all moving particle data and calculates physics (velocity).
        This is called ONCE from the main script after the fallen snow pass.
        """
        print("Precomputing moving particle data and velocities...")

        # Load data for t=0 first
        xyz_t_minus_1 = np.array([])
        ids_t_minus_1 = np.array([])

        # Handle t=0 (zero velocity)
        pos_path_0 = f"{self.simulation_dir}/0_pos.npz"
        ids_path_0 = f"{self.simulation_dir}/0_ids.npz"

        xyz_t_minus_1 = np.load(pos_path_0)["arr_0"]
        ids_t_minus_1 = np.load(ids_path_0)["arr_0"]

        xyz_t_0 = np.copy(xyz_t_minus_1)
        self.moving_rain_cache[0] = {
            "ids": ids_t_minus_1,
            "pos_t": xyz_t_0,
            "pos_t_minus_1": xyz_t_0,  # pos(t-1) == pos(t) at t=0
            "velocity": np.zeros_like(xyz_t_0)
        }

        for t in tqdm(range(1, total_steps), desc="Caching Moving Data"):
            pos_path = f"{self.simulation_dir}/{t}_pos.npz"
            ids_path = f"{self.simulation_dir}/{t}_ids.npz"

            xyz_t = np.load(pos_path)["arr_0"]
            ids_t = np.load(ids_path)["arr_0"]

            # --- Alignment Logic ---
            # 1. Create a lookup map for previous positions
            pos_map = {id_val: pos for id_val, pos in zip(ids_t_minus_1, xyz_t_minus_1)}

            # 2. Build aligned pos(t-1) and velocity(t) arrays
            aligned_xyz_t_minus_1_list = []
            velocities_t_list = []

            for id_val, pos_t in zip(ids_t, xyz_t):
                # Find the position of this particle at t-1
                # If it's a new particle, use pos_t as its t-1 pos (zero velocity)
                pos_t_minus_1 = pos_map.get(id_val, pos_t)

                aligned_xyz_t_minus_1_list.append(pos_t_minus_1)
                velocities_t_list.append(pos_t - pos_t_minus_1)

            aligned_xyz_t_minus_1 = np.array(aligned_xyz_t_minus_1_list)
            velocities_t = np.array(velocities_t_list)
            # --- End Alignment ---

            # Store in Cache
            self.moving_rain_cache[t] = {
                "ids": ids_t,
                "pos_t": xyz_t,
                "pos_t_minus_1": aligned_xyz_t_minus_1,
                "velocity": velocities_t
            }

            # Update t-1 for the next iteration
            xyz_t_minus_1 = xyz_t
            ids_t_minus_1 = ids_t

        print("Moving particle cache built.")

    def add_fallen_rain(self,t):
        fallen_rain = np.load(f"{self.simulation_dir}/{t}_static.npz")["arr_0"]
        hit_points, surface_normals = self.query_surface(fallen_rain)
        hit_points = torch.from_numpy(hit_points).float().cuda()
        surface_normals = torch.from_numpy(surface_normals).float().cuda()
        offset = self.config["surface_offset"]
        fallen_xyz = hit_points + surface_normals * offset
        fallen_scales = torch.ones_like(fallen_xyz)
        fallen_scales[:,0] *= self.config["surface_scale"][0]
        fallen_scales[:,1] *= self.config["surface_scale"][1]
        fallen_scales[:,2] *= self.config["surface_scale"][2]
        fallen_rotations = torch.zeros((len(fallen_xyz), 4), device="cuda")
        fallen_rotations[:, 0] = 1.0
        fallen_opacities = torch.zeros((len(fallen_xyz),1), device="cuda")
        fallen_colors = torch.tensor(self.config["fallen_colors"], device="cuda").expand(len(fallen_xyz),-1)

        if self.fallen_rain_dict is None:
            self.fallen_rain_dict = {t:{"xyz": fallen_xyz.cpu().numpy(), "rotations": fallen_rotations.cpu().numpy(), "scales":fallen_scales.cpu().numpy(),
                    "opacity":fallen_opacities.cpu().numpy(), "color": fallen_colors.cpu().numpy()}}

        else:
            # fallen_xyz = torch.concat([torch.from_numpy(self.fallen_rain_dict["xyz"]).float().cuda() , fallen_xyz]).cpu().numpy()
            # fallen_rotations = torch.concat([torch.from_numpy(self.fallen_rain_dict["rotations"]).float().cuda() , fallen_rotations]).cpu().numpy()
            # fallen_scales = torch.concat([torch.from_numpy(self.fallen_rain_dict["scales"]).float().cuda() , fallen_scales]).cpu().numpy()
            # fallen_opacities = torch.concat([torch.from_numpy(self.fallen_rain_dict["opacity"]).float().cuda() , fallen_opacities]).cpu().numpy()
            # fallen_colors = torch.concat([torch.from_numpy(self.fallen_rain_dict["color"]).float().cuda() , fallen_colors]).cpu().numpy()
            # self.fallen_rain_dict = {"xyz": fallen_xyz, "rotations": fallen_rotations, "scales":fallen_scales,
            #                          "opacity":fallen_opacities, "color":fallen_colors}
            self.fallen_rain_dict[t] = {"xyz": fallen_xyz.cpu().numpy(), "rotations": fallen_rotations.cpu().numpy(), "scales":fallen_scales.cpu().numpy(),
                                        "opacity":fallen_opacities.cpu().numpy(), "color": fallen_colors.cpu().numpy()}

    def get_rain_gaussians(self,t):
        xyzs = []
        xyzs.append(torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_pos.npz")["arr_0"]).float())
        rain_xyz = torch.vstack(xyzs).cuda()
        rain_scales = torch.ones_like(rain_xyz)
        rain_scales[:,0] *= self.config["scale"][0]
        rain_scales[:,1] *= self.config["scale"][1]
        rain_scales[:,2] *= self.config["scale"][2]
        rain_rotations = torch.zeros((len(rain_xyz), 4), device="cuda")
        rain_rotations[:, 0] = 1.0
        rain_opacities = torch.ones((len(rain_xyz),1), device="cuda") * self.config["opacity"]
        colors = torch.tensor(self.config["colors"], device="cuda").expand(len(rain_xyz),-1)
        if self.fallen_rain_dict is not None:
            # fallen_xyz = torch.from_numpy(self.fallen_rain_dict["xyz"]).float().cuda()
            # fallen_rotations = torch.from_numpy(self.fallen_rain_dict["rotations"]).float().cuda()
            # fallen_scales = torch.from_numpy(self.fallen_rain_dict["scales"]).float().cuda()
            # fallen_opacities = torch.from_numpy(self.fallen_rain_dict["opacity"]).float().cuda()
            # fallen_colors = torch.from_numpy(self.fallen_rain_dict["color"]).float().cuda()
            fallen_xyz = torch.from_numpy(np.array([])).float().cuda()
            fallen_rotations = torch.from_numpy(np.array([])).float().cuda()
            fallen_scales = torch.from_numpy(np.array([])).float().cuda()
            fallen_opacities = torch.from_numpy(np.array([])).float().cuda()
            fallen_colors = torch.from_numpy(np.array([])).float().cuda()
            for i in range(t):
                if i in self.fallen_rain_dict:
                    fallen_xyz = torch.concat([fallen_xyz, torch.from_numpy(self.fallen_rain_dict[i]["xyz"]).float().cuda()])
                    fallen_rotations = torch.concat([fallen_rotations, torch.from_numpy(self.fallen_rain_dict[i]["rotations"]).float().cuda()])
                    fallen_scales = torch.concat([fallen_scales, torch.from_numpy(self.fallen_rain_dict[i]["scales"]).float().cuda()])
                    fallen_opacities = torch.concat([fallen_opacities, torch.from_numpy(self.fallen_rain_dict[i]["opacity"]).float().cuda()])
                    fallen_colors = torch.concat([fallen_colors, torch.from_numpy(self.fallen_rain_dict[i]["color"]).float().cuda()])
            rain_xyz = torch.concat([rain_xyz, fallen_xyz])
            rain_scales = torch.concat([rain_scales, fallen_scales])
            rain_opacities = torch.concat([rain_opacities, fallen_opacities])
            rain_rotations = torch.concatenate([rain_rotations, fallen_rotations])
            colors = torch.concat([colors, fallen_colors])

        return rain_xyz, rain_scales, rain_rotations, rain_opacities, colors

    def get_rain_gaussians_refine_old(self, t, gm_moving, gm_fallen):
        xyzs = []
        xyzs.append(torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_pos.npz")["arr_0"]).float())
        ids = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_ids.npz")["arr_0"]).int()
        rain_xyz = torch.vstack(xyzs).cuda()
        rain_rotations = torch.zeros((len(rain_xyz), 4), device="cuda")
        rain_rotations[:, 0] = 1.0
        reg_loss_f = 0.0
        reg_loss_m = 0.0

        if gm_moving is not None:
            rain_scales, rain_opacities, colors, rain_xyz, reg_loss_m = gm_moving.produce_from_original(ids, t=t, xyz_positions=rain_xyz)
        else:
            rain_scales = torch.ones_like(rain_xyz)
            rain_scales[:,0] *= self.config["scale"][0]
            rain_scales[:,1] *= self.config["scale"][1]
            rain_scales[:,2] *= self.config["scale"][2]
            rain_opacities = torch.ones((len(rain_xyz),1), device="cuda") * self.config["opacity"]
            colors = torch.tensor(self.config["colors"], device="cuda").expand(len(rain_xyz),-1)
        if self.fallen_rain_dict is not None:
            fallen_xyz = torch.from_numpy(np.array([])).float().cuda()
            fallen_rotations = torch.from_numpy(np.array([])).float().cuda()
            fallen_scales = torch.from_numpy(np.array([])).float().cuda()
            fallen_opacities = torch.from_numpy(np.array([])).float().cuda()
            fallen_colors = torch.from_numpy(np.array([])).float().cuda()
            for i in range(t):
                if i in self.fallen_rain_dict:
                    fallen_xyz = torch.concat([fallen_xyz, torch.from_numpy(self.fallen_rain_dict[i]["xyz"]).float().cuda()])
                    fallen_rotations = torch.concat([fallen_rotations, torch.from_numpy(self.fallen_rain_dict[i]["rotations"]).float().cuda()])
                    fallen_scales = torch.concat([fallen_scales, torch.from_numpy(self.fallen_rain_dict[i]["scales"]).float().cuda()])
                    fallen_opacities = torch.concat([fallen_opacities, torch.from_numpy(self.fallen_rain_dict[i]["opacity"]).float().cuda()])
                    fallen_colors = torch.concat([fallen_colors, torch.from_numpy(self.fallen_rain_dict[i]["color"]).float().cuda()])

            # fallen_xyz = torch.from_numpy(self.fallen_rain_dict["xyz"]).float().cuda()
            # fallen_rotations = torch.from_numpy(self.fallen_rain_dict["rotations"]).float().cuda()
            if gm_fallen is not None:
                if len(fallen_opacities) > 0:
                    fallen_scales, fallen_opacities, fallen_colors, fallen_xyz, reg_loss_f = gm_fallen.produce_from_original(torch.arange(len(fallen_opacities),device="cuda"), t=t, xyz_positions=fallen_xyz)
            else:
                fallen_scales = torch.ones_like(fallen_xyz) * self.config["surface_scale"]
                fallen_opacities = torch.zeros((len(fallen_xyz),1), device="cuda") + 0.001
                fallen_colors = torch.tensor(self.config["colors"], device="cuda").expand(len(fallen_xyz),-1)
            rain_xyz = torch.concat([rain_xyz, fallen_xyz])
            rain_scales = torch.concat([rain_scales, fallen_scales])
            rain_opacities = torch.concat([rain_opacities, fallen_opacities])
            rain_rotations = torch.concatenate([rain_rotations, fallen_rotations])
            colors = torch.concat([colors, fallen_colors])

        return rain_xyz, rain_scales, rain_rotations, rain_opacities, colors, reg_loss_m, reg_loss_f

    def get_rain_gaussians_refine(self, t, gm_moving, gm_fallen):
        moving_data = self.moving_rain_cache.get(t)
        reg_loss_f = 0.0
        reg_loss_m = 0.0
        reg_loss_rot_m = 0.0
        reg_loss_rot_f = 0.0
        if gm_moving is not None:
            rain_scales, rain_opacities, colors, rain_xyz, rain_rotations, reg_loss_m, reg_loss_rot_m = \
                gm_moving.produce_from_original(
                    mask=torch.from_numpy(moving_data["ids"]).int().cuda(),
                    t=t,
                    original_xyz_t_minus_1=torch.from_numpy(moving_data["pos_t_minus_1"]).float().cuda(),
                    original_velocity_t=torch.from_numpy(moving_data["velocity"]).float().cuda(),
                    original_xyz_t=torch.from_numpy(moving_data["pos_t"]).float().cuda(),
                )
        else:
            rain_xyz = torch.from_numpy(moving_data["pos_t"]).float().cuda()
            rain_rotations = torch.zeros((len(rain_xyz), 4), device="cuda")
            rain_rotations[:, 0] = 1.0
            rain_scales = torch.ones_like(rain_xyz) * self.config["scale"]
            rain_opacities = torch.ones((len(rain_xyz), 1), device="cuda") * self.config["opacity"]
            colors = torch.tensor(self.config["colors"], device="cuda").expand(len(rain_xyz), -1)

        if self.fallen_rain_dict is not None:
            fallen_xyz = torch.from_numpy(np.array([])).float().cuda()
            fallen_rotations = torch.from_numpy(np.array([])).float().cuda()
            fallen_scales = torch.from_numpy(np.array([])).float().cuda()
            fallen_opacities = torch.from_numpy(np.array([])).float().cuda()
            fallen_colors = torch.from_numpy(np.array([])).float().cuda()
            for i in range(t):
                if i in self.fallen_rain_dict:
                    fallen_xyz = torch.concat(
                        [fallen_xyz, torch.from_numpy(self.fallen_rain_dict[i]["xyz"]).float().cuda()])
                    fallen_rotations = torch.concat(
                        [fallen_rotations, torch.from_numpy(self.fallen_rain_dict[i]["rotations"]).float().cuda()])
                    fallen_scales = torch.concat(
                        [fallen_scales, torch.from_numpy(self.fallen_rain_dict[i]["scales"]).float().cuda()])
                    fallen_opacities = torch.concat(
                        [fallen_opacities, torch.from_numpy(self.fallen_rain_dict[i]["opacity"]).float().cuda()])
                    fallen_colors = torch.concat(
                        [fallen_colors, torch.from_numpy(self.fallen_rain_dict[i]["color"]).float().cuda()])
            # fallen_xyz = torch.from_numpy(self.fallen_snow_dict["xyz"]).float().cuda()
            # fallen_rotations = torch.from_numpy(self.fallen_snow_dict["rotations"]).float().cuda()
            if gm_fallen is not None:
                if len(fallen_opacities) > 0:
                    fallen_scales, fallen_opacities, fallen_colors, fallen_xyz, fallen_rotations, reg_loss_f, reg_loss_rot_f = \
                        gm_fallen.produce_from_original(
                            mask=torch.arange(len(fallen_opacities), device="cuda"),
                            t=t,
                            original_xyz_t_minus_1=fallen_xyz,
                            original_velocity_t=torch.zeros_like(fallen_xyz).cuda(),
                            original_xyz_t=fallen_xyz)
            # else:
            #     fallen_scales = torch.ones_like(fallen_xyz) * self.config["surface_scale"]
            #     fallen_opacities = torch.ones((len(fallen_xyz),1), device="cuda") - 0.001
            #     fallen_colors = torch.tensor(self.config["colors"], device="cuda").expand(len(fallen_xyz),-1)
            # if t < 150:
            #     num_p, num_f = len(snow_xyz), len(fallen_xyz)
            #     snow_xyz = torch.concat([snow_xyz[:num_p//10], fallen_xyz[:num_f//10]])
            #     snow_scales = torch.concat([snow_scales[:num_p//10], fallen_scales[:num_f//10]])
            #     snow_opacities = torch.concat([snow_opacities[:num_p//10], fallen_opacities[:num_f//10]])
            #     snow_rotations = torch.concatenate([snow_rotations[:num_p//10], fallen_rotations[:num_f//10]])
            #     colors = torch.concat([colors[:num_p//10], fallen_colors])

            rain_xyz = torch.concat([rain_xyz, fallen_xyz])
            rain_scales = torch.concat([rain_scales, fallen_scales])
            rain_opacities = torch.concat([rain_opacities, fallen_opacities])
            rain_rotations = torch.concatenate([rain_rotations, fallen_rotations])
            colors = torch.concat([colors, fallen_colors])

        return rain_xyz, rain_scales, rain_rotations, rain_opacities, colors, reg_loss_m, reg_loss_f, reg_loss_rot_m, reg_loss_rot_f

    def get_rain_gaussians_refine_recurrent(self, t, gm_moving, gm_fallen,
                                  # --- Full t-1 state ---
                                  gm_moving_app_state,
                                  gm_fallen_app_state,
                                  prev_x_rendered_moving,
                                  prev_v_corrected_moving,
                                  prev_x_rendered_fallen,
                                  prev_v_corrected_fallen
                                  ):

        # Get simulation data for *this* timestep (for loss)
        moving_data = self.moving_rain_cache.get(t)
        if moving_data is None:
            print(f"Warning: No moving data in cache for t={t}. Using t=1.")
            moving_data = self.moving_rain_cache.get(1)  # Fallback
            if moving_data is None:
                raise ValueError("Moving rain cache is empty.")

        # --- State and Renderable Buffers ---
        new_moving_app_state, new_fallen_app_state = None, None
        new_x_rendered_moving, new_v_corrected_moving = None, None
        new_x_rendered_fallen, new_v_corrected_fallen = None, None

        reg_loss_xyz = 0.0
        reg_loss_vel = 0.0
        reg_loss_rot = 0.0
        op_reg_loss_m, sc_reg_loss_m, col_reg_loss_m = 0.0, 0.0, 0.0  # *** NEW ***
        op_reg_loss_f, sc_reg_loss_f, col_reg_loss_f = 0.0, 0.0, 0.0  # *** NEW ***

        diag_delta_v_m, diag_angular_v_m = 0.0, 0.0
        diag_delta_op_m, diag_delta_sc_m, diag_delta_col_m = 0.0, 0.0, 0.0
        diag_delta_op_f, diag_delta_sc_f, diag_delta_col_f = 0.0, 0.0, 0.0
        # Buffers for renderable params
        rain_xyz, rain_scales, rain_rotations, rain_opacities, colors = [], [], [], [], []

        if gm_moving is not None:
            # --- Simulation data (for loss) ---
            sim_ids_m = torch.from_numpy(moving_data["ids"]).int().cuda()
            sim_pos_t_m = torch.from_numpy(moving_data["pos_t"]).float().cuda()
            sim_vel_t_m = torch.from_numpy(moving_data["velocity"]).float().cuda()

            sim_pos_t_minus_1_m = torch.from_numpy(moving_data["pos_t_minus_1"]).float().cuda()
            # Create grounded t-1 position buffer
            grounded_prev_x_m = prev_x_rendered_moving.clone()
            # Find particles that are 0 in the buffer (new/uninitialized)
            # and set them to their sim_t-1 position.
            # This is a simple way to find "new" particles.
            uninitialized_mask = (grounded_prev_x_m[sim_ids_m] == 0.0).all(dim=-1)
            particles_to_ground = sim_ids_m[uninitialized_mask]
            positions_to_ground = sim_pos_t_minus_1_m[uninitialized_mask]

            grounded_prev_x_m[particles_to_ground] = positions_to_ground

            # Create full-size sim buffers for gm.step
            full_sim_pos_m = torch.zeros_like(prev_x_rendered_moving)
            full_sim_vel_m = torch.zeros_like(prev_v_corrected_moving)
            full_sim_pos_m[sim_ids_m] = sim_pos_t_m
            full_sim_vel_m[sim_ids_m] = sim_vel_t_m

            (new_moving_app_state,
             new_x_rendered_moving,
             new_v_corrected_moving), \
                (rain_scales_m, rain_opacities_m, colors_m,
                 rain_xyz_m, rain_rotations_m,
                 reg_loss_xyz, reg_loss_rot, reg_loss_vel,
                 op_reg_loss_m, sc_reg_loss_m, col_reg_loss_m,  # <-- ADDED
                 diag_delta_v_m, diag_angular_v_m,
                 diag_delta_op_m, diag_delta_sc_m, diag_delta_col_m) = \
                gm_moving.step(
                    mask=sim_ids_m,
                    t=t,
                    # --- Previous State (t-1) ---
                    prev_app_state=gm_moving_app_state,
                    prev_x_rendered=grounded_prev_x_m,
                    prev_v_corrected=prev_v_corrected_moving,
                    # --- Simulation Target (t) ---
                    original_xyz_t_sim=full_sim_pos_m,
                    original_velocity_t_sim=full_sim_vel_m
                )

            # Append renderable params
            rain_xyz.append(rain_xyz_m)
            rain_scales.append(rain_scales_m)
            rain_rotations.append(rain_rotations_m)
            rain_opacities.append(rain_opacities_m)
            colors.append(colors_m)

        else:
            # Non-refined case
            rain_xyz.append(torch.from_numpy(moving_data["pos_t"]).float().cuda())
            rain_rotations.append(torch.zeros((len(rain_xyz[-1]), 4), device="cuda"))
            rain_rotations[-1][:, 0] = 1.0
            rain_scales.append(torch.ones_like(rain_xyz[-1]) * self.config["scale"])
            rain_opacities.append(torch.ones((len(rain_xyz[-1]), 1), device="cuda") * self.config["opacity"])
            colors.append(torch.tensor(self.config["colors"], device="cuda").expand(len(rain_xyz[-1]), -1))

            # Pass state through
            new_moving_app_state = gm_moving_app_state
            new_x_rendered_moving = prev_x_rendered_moving
            new_v_corrected_moving = prev_v_corrected_moving

        if self.fallen_rain_dict is not None:
            # --- Aggregate all fallen particles UP TO this frame ---
            fallen_xyz_sim = torch.from_numpy(np.array([])).float().cuda()
            fallen_rotations_sim = torch.from_numpy(np.array([])).float().cuda()
            fallen_scales_sim = torch.from_numpy(np.array([])).float().cuda()
            fallen_opacities_sim = torch.from_numpy(np.array([])).float().cuda()
            fallen_colors_sim = torch.from_numpy(np.array([])).float().cuda()

            for i in range(t):
                if i in self.fallen_rain_dict:
                    fallen_xyz_sim = torch.concat(
                        [fallen_xyz_sim, torch.from_numpy(self.fallen_rain_dict[i]["xyz"]).float().cuda()])
                    fallen_rotations_sim = torch.concat(
                        [fallen_rotations_sim, torch.from_numpy(self.fallen_rain_dict[i]["rotations"]).float().cuda()])
                    fallen_scales_sim = torch.concat(
                        [fallen_scales_sim, torch.from_numpy(self.fallen_rain_dict[i]["scales"]).float().cuda()])
                    fallen_opacities_sim = torch.concat(
                        [fallen_opacities_sim, torch.from_numpy(self.fallen_rain_dict[i]["opacity"]).float().cuda()])
                    fallen_colors_sim = torch.concat(
                        [fallen_colors_sim, torch.from_numpy(self.fallen_rain_dict[i]["color"]).float().cuda()])

            # This is the mask of *active* fallen particles for this frame
            fallen_mask = torch.arange(len(fallen_opacities_sim), device="cuda")

            if gm_fallen is not None:
                if len(fallen_mask) > 0:
                    # Create full-size sim buffers for gm.step
                    full_sim_pos_f = torch.zeros_like(prev_x_rendered_fallen)
                    full_sim_vel_f = torch.zeros_like(prev_v_corrected_fallen)

                    full_sim_pos_f[fallen_mask] = fallen_xyz_sim
                    # Velocity is already zeros

                    (new_fallen_app_state,
                     new_x_rendered_fallen,
                     new_v_corrected_fallen), \
                        (rain_scales_f, rain_opacities_f, colors_f,
                         rain_xyz_f, rain_rotations_f,
                         reg_loss_f, reg_loss_rot_f, reg_loss_vel_f,
                         op_reg_loss_f, sc_reg_loss_f, col_reg_loss_f, # <-- ADDED
                         diag_delta_v_f, diag_angular_v_f,
                     diag_delta_op_f, diag_delta_sc_f, diag_delta_col_f) = \
                        gm_fallen.step(
                            mask=fallen_mask,
                            t=t,
                            # --- Previous State (t-1) ---
                            prev_app_state=gm_fallen_app_state,
                            prev_x_rendered=prev_x_rendered_fallen,
                            prev_v_corrected=prev_v_corrected_fallen,
                            # --- Simulation Target (t) ---
                            original_xyz_t_sim=full_sim_pos_f,
                            original_velocity_t_sim=full_sim_vel_f
                        )

                    # Append renderable params
                    rain_xyz.append(rain_xyz_f)
                    rain_scales.append(rain_scales_f)
                    rain_rotations.append(rain_rotations_f)
                    rain_opacities.append(rain_opacities_f)
                    colors.append(colors_f)
                else:
                    # No fallen particles this frame
                    new_fallen_app_state = gm_fallen_app_state
                    new_x_rendered_fallen = prev_x_rendered_fallen
                    new_v_corrected_fallen = prev_v_corrected_fallen
            else:
                # Non-refined case
                if len(fallen_mask) > 0:
                    rain_xyz.append(fallen_xyz_sim)
                    rain_rotations.append(fallen_rotations_sim)
                    rain_scales.append(fallen_scales_sim)
                    rain_opacities.append(fallen_opacities_sim)
                    colors.append(fallen_colors_sim)

                new_fallen_app_state = gm_fallen_app_state
                new_x_rendered_fallen = prev_x_rendered_fallen
                new_v_corrected_fallen = prev_v_corrected_fallen
        else:
            # No fallen rain dict
            new_fallen_app_state = gm_fallen_app_state
            new_x_rendered_fallen = prev_x_rendered_fallen
            new_v_corrected_fallen = prev_v_corrected_fallen

        # Concatenate all renderable particles
        rain_xyz = torch.concat(rain_xyz)
        rain_scales = torch.concat(rain_scales)
        rain_opacities = torch.concat(rain_opacities)
        rain_rotations = torch.concatenate(rain_rotations)
        colors = torch.concat(colors)

        # --- Return ALL new states and renderable params ---
        return (rain_xyz, rain_scales, rain_rotations, rain_opacities, colors,
                reg_loss_xyz, reg_loss_vel, reg_loss_rot,
                op_reg_loss_m, sc_reg_loss_m, col_reg_loss_m,  # *** NEW ***
                op_reg_loss_f, sc_reg_loss_f, col_reg_loss_f,  # *** NEW ***
                diag_delta_v_m, diag_angular_v_m,
                diag_delta_op_m, diag_delta_sc_m, diag_delta_col_m,
                diag_delta_op_f, diag_delta_sc_f, diag_delta_col_f,
                new_moving_app_state, new_fallen_app_state,
                new_x_rendered_moving, new_v_corrected_moving,
                new_x_rendered_fallen, new_v_corrected_fallen)


    def get_moving_gaussians(self):
        """Get initial parameters for moving rain particles"""
        # Get a representative sample (e.g., frame 0)
        array_len = np.load(f"{self.simulation_dir}/total_moving.npz")["arr_0"][0]
        rain_scales = torch.ones(array_len,3)
        rain_scales[:,0] *= self.config["scale"][0]
        rain_scales[:,1] *= self.config["scale"][1]
        rain_scales[:,2] *= self.config["scale"][2]
        rain_opacities = torch.ones((array_len, 1), device="cuda") * self.config["opacity"]
        rain_colors = torch.tensor(self.config["colors"], device="cuda").expand(array_len, -1)
        rain_rotations = torch.zeros((array_len, 4), device="cuda")
        rain_rotations[:, 0] = 1.0
        return rain_scales, rain_opacities, rain_colors, rain_rotations
    
    def get_fallen_gaussians(self):
        """Get initial parameters for fallen rain particles"""
        # Get max number of fallen particles (you might need to adjust this)
        array_len = np.load(f"{self.simulation_dir}/total_static.npz")["arr_0"][0]
        fallen_scales = torch.ones(array_len,3)
        fallen_scales[:,0] *= self.config["surface_scale"][0]
        fallen_scales[:,1] *= self.config["surface_scale"][1]
        fallen_scales[:,2] *= self.config["surface_scale"][2]
        fallen_opacities = torch.zeros((array_len, 1), device="cuda")
        fallen_colors = torch.tensor(self.config["fallen_colors"], device="cuda").expand(array_len, -1)
        fallen_rotations = torch.zeros((array_len, 4), device="cuda")
        fallen_rotations[:, 0] = 1.0
        return fallen_scales, fallen_opacities, fallen_colors, fallen_rotations

    def render(self,
            viewpoint_camera,
            pc : GaussianModel,
            pipe,
            bg_color : torch.Tensor,
            scaling_modifier = 1.0,
            t=0, refine=False, gm_moving=None, gm_fallen=None,
           gm_moving_app_state=None,
           gm_fallen_app_state=None,
           prev_x_rendered_moving=None,
           prev_v_corrected_moving=None,
           prev_x_rendered_fallen=None,
           prev_v_corrected_fallen=None):

        """
        Render the scene.
        
        Background tensor (bg_color) must be on GPU!
        """
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        reg_loss_f = 0.0
        reg_loss_m = 0.0
        reg_loss_rot_m = 0.0
        reg_loss_rot_f = 0.0
        reg_loss_vel = 0.0
        op_reg_loss_m, sc_reg_loss_m, col_reg_loss_m = 0.0, 0.0, 0.0  # *** NEW ***
        op_reg_loss_f, sc_reg_loss_f, col_reg_loss_f = 0.0, 0.0, 0.0  # *** NEW ***
        diag_delta_v_m, diag_angular_v_m, diag_delta_op_m, diag_delta_sc_m, diag_delta_col_m =0.0, 0.0, 0.0, 0.0, 0.0
        diag_delta_op_f, diag_delta_sc_f, diag_delta_col_f = 0.0, 0.0, 0.0
        new_moving_app_state, new_fallen_app_state, new_x_rendered_moving, new_v_corrected_moving, new_x_rendered_fallen, new_v_corrected_fallen = None, None, None, None, None, None

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

        means3D = pc.get_xyz.detach()
        opacity = pc.get_opacity.detach()
           
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling.detach()
            rotations = pc.get_rotation.detach()

        shs = None
        colors_precomp = None
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0).detach()  # (N, 3)

        if os.path.exists(f"{self.simulation_dir}/{t}_static.npz") and not refine:
            if "stump" in self.simulation_dir and t<90:
                a=1
            elif "truck" in self.simulation_dir and t<110:
                b=1
            else:
                self.add_fallen_rain(t)

        if refine:
            if gm_moving_app_state is None:
                rain_xyz, rain_scales, rain_rotations, rain_opacities, colors, reg_loss_m, reg_loss_f, reg_loss_rot_m, reg_loss_rot_f = self.get_rain_gaussians_refine(t, gm_moving, gm_fallen)
            else:
                (rain_xyz, rain_scales, rain_rotations, rain_opacities, colors,
                 reg_loss_m, reg_loss_vel, reg_loss_rot_m,
                 op_reg_loss_m, sc_reg_loss_m, col_reg_loss_m,  # *** NEW ***
                 op_reg_loss_f, sc_reg_loss_f, col_reg_loss_f,  # *** NEW ***
                 diag_delta_v_m, diag_angular_v_m,
                 diag_delta_op_m, diag_delta_sc_m, diag_delta_col_m,
                 diag_delta_op_f, diag_delta_sc_f, diag_delta_col_f,
                 new_moving_app_state, new_fallen_app_state,
                 new_x_rendered_moving, new_v_corrected_moving,
                 new_x_rendered_fallen, new_v_corrected_fallen) = \
                    self.get_rain_gaussians_refine_recurrent(
                        t, gm_moving, gm_fallen,
                        gm_moving_app_state, gm_fallen_app_state,
                        prev_x_rendered_moving, prev_v_corrected_moving,
                        prev_x_rendered_fallen, prev_v_corrected_fallen
                    )

        else:
            rain_xyz, rain_scales, rain_rotations, rain_opacities, colors = self.get_rain_gaussians(t)

        tot_means3d = torch.concat([means3D, rain_xyz])
        tot_rotations = torch.concat([rotations, rain_rotations])
        tot_scales = torch.concat([scales, rain_scales])
        tot_opacity = torch.concat([opacity, rain_opacities])
        tot_colors_precomp = torch.concat([colors_precomp, colors])


        # Get view-independent features (distill features) for each Gaussian for rendering.
        distill_feats = pc.get_distill_features.detach()

        screenspace_points = torch.zeros_like(tot_means3d, dtype=tot_means3d.dtype, requires_grad=False, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points

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
                "radii": radii,
                "reg_loss_m": reg_loss_m,
                "reg_loss_f": reg_loss_f,
                "rot_reg_loss_m": reg_loss_rot_m,
                "rot_reg_loss_f": reg_loss_rot_f,
                "reg_loss_vel": reg_loss_vel,
                "op_reg_loss_m": op_reg_loss_m,  # *** NEW ***
                "sc_reg_loss_m": sc_reg_loss_m,  # *** NEW ***
                "col_reg_loss_m": col_reg_loss_m,  # *** NEW ***
                "op_reg_loss_f": op_reg_loss_f,  # *** NEW ***
                "sc_reg_loss_f": sc_reg_loss_f,  # *** NEW ***
                "col_reg_loss_f": col_reg_loss_f,  # *** NEW ***
                "diag_delta_v_m": diag_delta_v_m,
                "diag_angular_v_m": diag_angular_v_m,
                "diag_delta_op_m": diag_delta_op_m,
                "diag_delta_sc_m": diag_delta_sc_m,
                "diag_delta_col_m": diag_delta_col_m,
                "diag_delta_op_f": diag_delta_op_f,
                "diag_delta_sc_f": diag_delta_sc_f,
                "diag_delta_col_f": diag_delta_col_f,
                "new_moving_app_state": new_moving_app_state,
                "new_fallen_app_state": new_fallen_app_state,
                "new_x_rendered_moving": new_x_rendered_moving,
                "new_v_corrected_moving": new_v_corrected_moving,
                "new_x_rendered_fallen": new_x_rendered_fallen,
                "new_v_corrected_fallen": new_v_corrected_fallen
                }

    def render_new(self,
                viewpoint_camera,
                pc : GaussianModel,
                dynamic_pc : GaussianModelDynamic,
                pipe,
                bg_color : torch.Tensor,
                scaling_modifier = 1.0,
                t=0):
            """
            Render the scene.
            
            Background tensor (bg_color) must be on GPU!
            """
        
            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

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
            opacity = pc.get_opacity
            
            scales = None
            rotations = None
            cov3D_precomp = None
            if pipe.compute_cov3D_python:
                cov3D_precomp = pc.get_covariance(scaling_modifier)
            else:
                scales = pc.get_scaling
                rotations = pc.get_rotation
            
            if os.path.exists(f"{self.simulation_dir}/{t}_static.npz"):
                if "stump" in self.simulation_dir and t<90:
                    a=1
                elif "truck" in self.simulation_dir and t<110:
                    b=1
                else:
                    self.add_fallen_rain(t)
            
            
            dynamic_shs = dynamic_pc.get_features
            dynamic_scales = dynamic_pc.get_scaling
            dynamic_rotations = dynamic_pc.get_rotation
            dynamic_opacity = dynamic_pc.get_opacity

            xyzs = []
            xyzs.append(torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_pos.npz")["arr_0"]).float())
            moving_xyz = torch.vstack(xyzs).cuda()
            fallen_xyz = torch.from_numpy(np.array([])).float().cuda()
            if self.fallen_rain_dict is not None:
                for i in range(t):
                    if i in self.fallen_rain_dict:
                        fallen_xyz = torch.concat([fallen_xyz, torch.from_numpy(self.fallen_rain_dict[i]["xyz"]).float().cuda()])
            fallen_ids = torch.arange(len(fallen_xyz), device="cuda")+ dynamic_pc.num_moving_xyz
            moving_ids = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_ids.npz")["arr_0"]).int().cuda()

            dynamic_xyz = torch.concat([moving_xyz, fallen_xyz]).cuda()
            dynamic_ids = torch.concat([moving_ids, fallen_ids]).cuda()
            dynamic_rotations = dynamic_rotations[dynamic_ids]
            dynamic_scales = dynamic_scales[dynamic_ids]
            dynamic_opacity = dynamic_opacity[dynamic_ids]
            dynamic_shs = dynamic_shs[dynamic_ids]

            # snow_xyz, snow_scales, snow_rotations, snow_opacities, colors = self.get_snow_gaussians(t)
            tot_shs = None
            tot_colors_precomp = None

            shs = pc.get_features
            tot_means3d = torch.concat([means3D, dynamic_xyz])
            tot_rotations = torch.concat([rotations, dynamic_rotations])
            tot_scales = torch.concat([scales, dynamic_scales])
            tot_opacity = torch.concat([opacity, dynamic_opacity])
            tot_shs = torch.concat([shs, dynamic_shs])


            # Get view-independent features (distill features) for each Gaussian for rendering.
            distill_feats = pc.get_distill_features.detach()

            screenspace_points = torch.zeros_like(tot_means3d, dtype=tot_means3d.dtype, requires_grad=False, device="cuda") + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
            means2D = screenspace_points

            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
                means3D = tot_means3d,
                means2D = means2D,
                shs = tot_shs,
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
                    "radii": radii}
    
