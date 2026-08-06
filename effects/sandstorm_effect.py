import json
import torch
import open3d as o3d
import os
import numpy as np
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.gaussian_model_dynamic import GaussianModelDynamic
from utils.sh_utils import eval_sh

class SandstormEffectRenderer:
    def __init__(self, mesh_path, simulation_dir, config_path):
        self.simulation_dir = simulation_dir
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config = config
        
        # Setup surface collision query
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh_o3d_t = o3d.io.read_triangle_mesh(mesh_path)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d_t))
        
        # Sand accumulation data
        self.accumulated_sand_dict = None
        self.moving_sand_cache = {} # --- NEW: Cache for moving particle physics

    def get_moving_data(self, t):
        return self.moving_sand_cache.get(t)

    def _apply_anisotropic_scaling(self, rotations, base_scale, apply=True):
        """Create flattened particles aligned with rotation"""
        scales = torch.ones((len(rotations), 3), device="cuda") * base_scale
        # Flatten based on rotation Z-axis
        if apply:
            scales[:, 2] *= self.config["scale_anisotropy"] 
        return scales

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
        array_len = np.load(f"{self.simulation_dir}/total_moving.npz")["arr_0"][0]

        # Handle t=0 (zero velocity)
        pos_path_0 = f"{self.simulation_dir}/0_pos.npz"
        ids_path_0 = f"{self.simulation_dir}/0_ids.npz"

        xyz_t_minus_1 = np.load(pos_path_0)["arr_0"]
        clone_sand = xyz_t_minus_1 + np.random.uniform(0.001, 0.005, size=(len(xyz_t_minus_1), 3))
        xyz_t_minus_1 = np.concatenate((xyz_t_minus_1, clone_sand))
        ids_t_minus_1 = np.load(ids_path_0)["arr_0"]
        ids_t_minus_1 = np.concatenate((ids_t_minus_1, ids_t_minus_1+array_len))

        xyz_t_0 = np.copy(xyz_t_minus_1)
        self.moving_sand_cache[0] = {
            "ids": ids_t_minus_1,
            "pos_t": xyz_t_0,
            "pos_t_minus_1": xyz_t_0,  # pos(t-1) == pos(t) at t=0
            "velocity": np.zeros_like(xyz_t_0)
        }

        for t in tqdm(range(1, total_steps), desc="Caching Moving Data"):
            pos_path = f"{self.simulation_dir}/{t}_pos.npz"
            ids_path = f"{self.simulation_dir}/{t}_ids.npz"

            xyz_t = np.load(pos_path)["arr_0"]
            clone_sand = xyz_t + np.random.uniform(0.001, 0.005, size=(len(xyz_t), 3))
            xyz_t = np.concatenate((xyz_t, clone_sand))
            ids_t = np.load(ids_path)["arr_0"]
            ids_t = np.concatenate((ids_t, ids_t + array_len))

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
            self.moving_sand_cache[t] = {
                "ids": ids_t,
                "pos_t": xyz_t,
                "pos_t_minus_1": aligned_xyz_t_minus_1,
                "velocity": velocities_t
            }

            # Update t-1 for the next iteration
            xyz_t_minus_1 = xyz_t
            ids_t_minus_1 = ids_t

        print("Moving particle cache built.")

    def _add_accumulated_sand_ablation(self, t):
        dynamic_sand = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_static.npz")["arr_0"]).float().cuda()
        dynamic_rot = torch.zeros((len(dynamic_sand), 4), device="cuda")
        dynamic_rot[:, 0] = 1.0
        dynamic_scales = torch.ones_like(dynamic_sand) * self.config["scale"]
        opacities = torch.ones((len(dynamic_sand), 1), device="cuda") * self.config["dynamic_opacity"]
        colors = torch.tensor(self.config["color"], device="cuda").expand(len(dynamic_sand), -1)
        if self.accumulated_sand_dict is None:
            self.accumulated_sand_dict = {t:{"xyz":dynamic_sand.cpu().numpy(), "rotations":dynamic_rot.cpu().numpy(),"scales":dynamic_scales.cpu().numpy(), 
                                    "opacity":opacities.cpu().numpy(),"color":colors.cpu().numpy()}}
        else:
            self.accumulated_sand_dict[t] = {"xyz":dynamic_sand.cpu().numpy(), "rotations":dynamic_rot.cpu().numpy(),"scales":dynamic_scales.cpu().numpy(), 
                                    "opacity":opacities.cpu().numpy(),"color":colors.cpu().numpy()}

    def _add_accumulated_sand(self, t):
        static_pos = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_static.npz")["arr_0"]).cuda().float()
        
        # Offset from surface to prevent z-fighting
        points_tensor = o3d.core.Tensor(static_pos.cpu().numpy(), dtype=o3d.core.Dtype.Float32)
        results = self.scene.compute_closest_points(points_tensor)
        normals = torch.from_numpy(results['primitive_normals'].numpy()).cuda().float()
        hit_points = torch.from_numpy(results['points'].numpy()).cuda().float()
        static_pos = hit_points + normals * self.config["surface_offset"]
        
        static_rot = torch.zeros((len(static_pos), 4), device="cuda")
        static_rot[:, 0] = 1.0
        static_scales = self._apply_anisotropic_scaling(static_rot, self.config["accumulated_scale"])
        static_opacities = torch.ones((len(static_pos), 1), device="cuda") * self.config["opacity"]
        static_colors = torch.tensor(self.config["color"], device="cuda").expand(len(static_pos), -1)
        if self.accumulated_sand_dict is None:
            self.accumulated_sand_dict = {t:{"xyz": static_pos.cpu().numpy(), "rotations": static_rot.cpu().numpy(), "scales":static_scales.cpu().numpy(), 
                    "opacity":static_opacities.cpu().numpy(), "color": static_colors.cpu().numpy()}}

        else:
            # static_pos = torch.concat([torch.from_numpy(self.accumulated_sand_dict["xyz"]).float().cuda() , static_pos]).cpu().numpy()
            # static_rot = torch.concat([torch.from_numpy(self.accumulated_sand_dict["rotations"]).float().cuda() , static_rot]).cpu().numpy()
            # static_scales = torch.concat([torch.from_numpy(self.accumulated_sand_dict["scales"]).float().cuda() , static_scales]).cpu().numpy()
            # static_opacities = torch.concat([torch.from_numpy(self.accumulated_sand_dict["opacity"]).float().cuda() , static_opacities]).cpu().numpy()
            # static_colors = torch.concat([torch.from_numpy(self.accumulated_sand_dict["color"]).float().cuda() , static_colors]).cpu().numpy()
            # self.accumulated_sand_dict = {"xyz": static_pos, "rotations": static_rot, "scales":static_scales,
            #                          "opacity":static_opacities, "color":static_colors}
            self.accumulated_sand_dict[t] = {"xyz": static_pos.cpu().numpy(), "rotations": static_rot.cpu().numpy(), "scales":static_scales.cpu().numpy(),
                                                "opacity":static_opacities.cpu().numpy(), "color": static_colors.cpu().numpy()}

    def get_sand_gaussians(self, t):
        # Load dynamic sand particles
        dynamic_sand = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_pos.npz")["arr_0"]).float().cuda()
        # dynamic_rot = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_rot.npz")["arr_0"]).float().cuda()
        dynamic_rot = torch.zeros((len(dynamic_sand), 4), device="cuda")
        dynamic_rot[:, 0] = 1.0

        # Apply anisotropic scaling
        dynamic_scales = self._apply_anisotropic_scaling(dynamic_sand, self.config["scale"], False)
        dynamic_opacities = torch.ones((len(dynamic_sand), 1), device="cuda") * self.config["dynamic_opacity"]
        clone_opacities = torch.ones((len(dynamic_sand), 1), device="cuda") * self.config["clone_opacity"]
        clone_sand = dynamic_sand + torch.FloatTensor(len(dynamic_sand), 3).uniform_(0.001, 0.005).cuda()
        clone_rot = dynamic_rot.clone()
        clone_scale = self._apply_anisotropic_scaling(clone_sand, self.config["clone_scale"], False)
        all_pos = torch.concat([dynamic_sand, clone_sand])
        all_rot = torch.concat([dynamic_rot, clone_rot])
        all_scales = torch.concat([dynamic_scales, clone_scale])
        all_opacities = torch.concat([dynamic_opacities, clone_opacities])
        all_colors = torch.tensor(self.config["color"], device="cuda").expand(len(all_pos), -1)
        if self.accumulated_sand_dict is not None:
            # If accumulated sand exists, add it to the moving particles
            # static_pos = torch.from_numpy(self.accumulated_sand_dict["xyz"]).float().cuda()
            # static_rot = torch.from_numpy(self.accumulated_sand_dict["rotations"]).float().cuda()
            # static_scales = torch.from_numpy(self.accumulated_sand_dict["scales"]).float().cuda()
            # static_opacities = torch.from_numpy(self.accumulated_sand_dict["opacity"]).float().cuda()
            # static_colors = torch.from_numpy(self.accumulated_sand_dict["color"]).float().cuda()
            static_pos = torch.from_numpy(np.array([])).float().cuda()
            static_rot = torch.from_numpy(np.array([])).float().cuda()
            static_scales = torch.from_numpy(np.array([])).float().cuda()
            static_opacities = torch.from_numpy(np.array([])).float().cuda()
            static_colors = torch.from_numpy(np.array([])).float().cuda()
            for i in range(t):
                if i in self.accumulated_sand_dict:
                    static_pos = torch.cat([static_pos, torch.from_numpy(self.accumulated_sand_dict[i]["xyz"]).float().cuda()])
                    static_rot = torch.cat([static_rot, torch.from_numpy(self.accumulated_sand_dict[i]["rotations"]).float().cuda()])
                    static_scales = torch.cat([static_scales, torch.from_numpy(self.accumulated_sand_dict[i]["scales"]).float().cuda()])
                    static_opacities = torch.cat([static_opacities, torch.from_numpy(self.accumulated_sand_dict[i]["opacity"]).float().cuda()])
                    static_colors = torch.cat([static_colors, torch.from_numpy(self.accumulated_sand_dict[i]["color"]).float().cuda()])
            all_pos = torch.cat([all_pos, static_pos])
            all_rot = torch.cat([all_rot, static_rot])
            all_scales = torch.cat([all_scales, static_scales])
            all_opacities = torch.cat([all_opacities, static_opacities])
            all_colors = torch.cat([all_colors, static_colors])
        
        return all_pos, all_scales, all_rot, all_opacities, all_colors
    
    def get_sand_gaussians_refine_ablation(self, t, gm_moving, gm_fallen):
        """
        Gets sand particle attributes for refinement, using learnable GaussianMatrices.
        This is the core function for the training loop.
        """
        # Load dynamic particles and their persistent IDs
        sand_xyz = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_pos.npz")["arr_0"]).float().cuda()
        ids_orig = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_ids.npz")["arr_0"]).int()
        array_len = np.load(f"{self.simulation_dir}/total_moving.npz")["arr_0"][0]
        ids = torch.cat([ids_orig, ids_orig + array_len])

        rotations_orig = torch.zeros((len(sand_xyz), 4), device="cuda")
        rotations_orig[:, 0] = 1.0
        sand_rotations = torch.cat([rotations_orig, rotations_orig.clone()])

        clone_sand = sand_xyz + torch.FloatTensor(len(sand_xyz), 3).uniform_(0.001, 0.005).cuda()
        # Get learnable attributes for moving particles using their IDs
        if gm_moving is not None:
            sand_scales, sand_opacities, colors = gm_moving.produce_from_original(ids)
        else:
            sand_scales = self._apply_anisotropic_scaling(torch.ones(len(sand_xyz), 3), self.config["scale"], apply=False)
            sand_opacities = torch.ones((len(sand_xyz), 1), device="cuda") * self.config["dynamic_opacity"]
            clone_scales = self._apply_anisotropic_scaling(torch.ones(len(clone_sand), 3), self.config["clone_scale"], apply=False)
            clone_opacities = torch.ones((len(clone_sand), 1), device="cuda") * self.config["clone_opacity"]
            sand_scales = torch.cat([sand_scales, clone_scales])
            sand_opacities = torch.cat([sand_opacities, clone_opacities])
            colors = torch.tensor(self.config["color"], device="cuda").expand(len(sand_xyz) + len(clone_sand), -1)

        # Combine with accumulated static sand if it exists
        if self.accumulated_sand_dict is not None:
            fallen_xyz = torch.from_numpy(np.array([])).float().cuda()
            fallen_rotations = torch.from_numpy(np.array([])).float().cuda()
            fallen_scales = torch.from_numpy(np.array([])).float().cuda()
            fallen_opacities = torch.from_numpy(np.array([])).float().cuda()
            fallen_colors = torch.from_numpy(np.array([])).float().cuda()
            for i in range(t):
                if i in self.accumulated_sand_dict:
                    fallen_xyz = torch.concat([fallen_xyz, torch.from_numpy(self.accumulated_sand_dict[i]["xyz"]).float().cuda()])
                    fallen_rotations = torch.concat([fallen_rotations, torch.from_numpy(self.accumulated_sand_dict[i]["rotations"]).float().cuda()])
                    fallen_scales = torch.concat([fallen_scales, torch.from_numpy(self.accumulated_sand_dict[i]["scales"]).float().cuda()])
                    fallen_opacities = torch.concat([fallen_opacities, torch.from_numpy(self.accumulated_sand_dict[i]["opacity"]).float().cuda()])
                    fallen_colors = torch.concat([fallen_colors, torch.from_numpy(self.accumulated_sand_dict[i]["color"]).float().cuda()])

            # fallen_xyz = torch.from_numpy(self.accumulated_sand_dict["xyz"]).float().cuda()
            # fallen_rotations = torch.from_numpy(self.accumulated_sand_dict["rotations"]).float().cuda()
            
            # Get learnable attributes for all fallen particles seen so far
            # if gm_fallen is not None:
            #     fallen_scales, fallen_opacities, fallen_colors = gm_fallen.produce_from_original(
            #         torch.arange(len(fallen_opacities), device="cuda")
            #     )
            # else:
            #     fallen_scales = self._apply_anisotropic_scaling(torch.ones(len(fallen_xyz), 3), self.config["accumulated_scale"])
            #     fallen_opacities = torch.ones((len(fallen_xyz), 1), device="cuda") * self.config["opacity"]
            #     fallen_colors = torch.tensor(self.config["color"], device="cuda").expand(len(fallen_xyz), -1)

            # Concatenate all particle attributes
            sand_xyz = torch.cat([sand_xyz,clone_sand, fallen_xyz])
            sand_scales = torch.cat([sand_scales, fallen_scales])
            sand_opacities = torch.cat([sand_opacities, fallen_opacities])
            sand_rotations = torch.cat([sand_rotations, fallen_rotations])
            colors = torch.cat([colors, fallen_colors])

        return sand_xyz, sand_scales, sand_rotations, sand_opacities, colors

    def get_sand_gaussians_refine_old(self, t, gm_moving, gm_fallen):
        """
        Gets sand particle attributes for refinement, using learnable GaussianMatrices.
        This is the core function for the training loop.
        """
        # Load dynamic particles and their persistent IDs
        sand_xyz = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_pos.npz")["arr_0"]).float().cuda()
        ids_orig = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_ids.npz")["arr_0"]).int()
        array_len = np.load(f"{self.simulation_dir}/total_moving.npz")["arr_0"][0]
        ids = torch.cat([ids_orig, ids_orig + array_len])

        rotations_orig = torch.zeros((len(sand_xyz), 4), device="cuda")
        rotations_orig[:, 0] = 1.0
        sand_rotations = torch.cat([rotations_orig, rotations_orig.clone()])

        clone_sand = sand_xyz + torch.FloatTensor(len(sand_xyz), 3).uniform_(0.001, 0.005).cuda()
        # Get learnable attributes for moving particles using their IDs
        reg_loss_f = 0.0
        reg_loss_m = 0.0

        if gm_moving is not None:
            sand_scales, sand_opacities, colors, sand_xyz, reg_loss_m = gm_moving.produce_from_original(ids, t=t, xyz_positions=torch.cat([sand_xyz,clone_sand]))
            # sand_opacities = sand_opacities - 0.05
        else:
            sand_scales = self._apply_anisotropic_scaling(torch.ones(len(sand_xyz), 3), self.config["scale"], apply=False)
            sand_opacities = torch.ones((len(sand_xyz), 1), device="cuda") * self.config["dynamic_opacity"]
            clone_scales = self._apply_anisotropic_scaling(torch.ones(len(clone_sand), 3), self.config["clone_scale"], apply=False)
            clone_opacities = torch.ones((len(clone_sand), 1), device="cuda") * self.config["clone_opacity"]
            sand_scales = torch.cat([sand_scales, clone_scales])
            sand_opacities = torch.cat([sand_opacities, clone_opacities])
            colors = torch.tensor(self.config["color"], device="cuda").expand(len(sand_xyz) + len(clone_sand), -1)

        # Combine with accumulated static sand if it exists
        if self.accumulated_sand_dict is not None:
            fallen_xyz = torch.from_numpy(np.array([])).float().cuda()
            fallen_rotations = torch.from_numpy(np.array([])).float().cuda()
            fallen_scales = torch.from_numpy(np.array([])).float().cuda()
            fallen_opacities = torch.from_numpy(np.array([])).float().cuda()
            fallen_colors = torch.from_numpy(np.array([])).float().cuda()
            for i in range(t):
                if i in self.accumulated_sand_dict:
                    fallen_xyz = torch.concat([fallen_xyz, torch.from_numpy(self.accumulated_sand_dict[i]["xyz"]).float().cuda()])
                    fallen_rotations = torch.concat([fallen_rotations, torch.from_numpy(self.accumulated_sand_dict[i]["rotations"]).float().cuda()])
                    fallen_scales = torch.concat([fallen_scales, torch.from_numpy(self.accumulated_sand_dict[i]["scales"]).float().cuda()])
                    fallen_opacities = torch.concat([fallen_opacities, torch.from_numpy(self.accumulated_sand_dict[i]["opacity"]).float().cuda()])
                    fallen_colors = torch.concat([fallen_colors, torch.from_numpy(self.accumulated_sand_dict[i]["color"]).float().cuda()])

            # fallen_xyz = torch.from_numpy(self.accumulated_sand_dict["xyz"]).float().cuda()
            # fallen_rotations = torch.from_numpy(self.accumulated_sand_dict["rotations"]).float().cuda()
            
            # Get learnable attributes for all fallen particles seen so far
            if gm_fallen is not None:
                if len(fallen_opacities) > 0:
                    fallen_scales, fallen_opacities, fallen_colors, fallen_xyz, reg_loss_f = gm_fallen.produce_from_original(
                        torch.arange(len(fallen_opacities), device="cuda"), t=t, xyz_positions=fallen_xyz
                    )
            else:
                fallen_scales = self._apply_anisotropic_scaling(torch.ones(len(fallen_xyz), 3), self.config["accumulated_scale"])
                fallen_opacities = torch.ones((len(fallen_xyz), 1), device="cuda") * self.config["opacity"]
                fallen_colors = torch.tensor(self.config["color"], device="cuda").expand(len(fallen_xyz), -1)

            # Concatenate all particle attributes
            if gm_moving is not None:
                sand_xyz = torch.cat([sand_xyz, fallen_xyz])
            else:
                sand_xyz = torch.cat([sand_xyz,clone_sand, fallen_xyz])
            sand_scales = torch.cat([sand_scales, fallen_scales])
            sand_opacities = torch.cat([sand_opacities, fallen_opacities])
            sand_rotations = torch.cat([sand_rotations, fallen_rotations])
            colors = torch.cat([colors, fallen_colors])

        return sand_xyz, sand_scales, sand_rotations, sand_opacities, colors, reg_loss_m, reg_loss_f

    def get_sand_gaussians_refine(self, t, gm_moving, gm_fallen):
        moving_data = self.moving_sand_cache.get(t)
        reg_loss_f = 0.0
        reg_loss_m = 0.0
        reg_loss_rot_m = 0.0
        reg_loss_rot_f = 0.0
        if gm_moving is not None:
            sand_scales, sand_opacities, colors, sand_xyz, sand_rotations, reg_loss_m, reg_loss_rot_m = \
                gm_moving.produce_from_original(
                    mask=torch.from_numpy(moving_data["ids"]).int().cuda(),
                    t=t,
                    original_xyz_t_minus_1=torch.from_numpy(moving_data["pos_t_minus_1"]).float().cuda(),
                    original_velocity_t=torch.from_numpy(moving_data["velocity"]).float().cuda(),
                    original_xyz_t=torch.from_numpy(moving_data["pos_t"]).float().cuda(),
                )
        else:
            sand_xyz = torch.from_numpy(moving_data["pos_t"]).float().cuda()
            sand_rotations = torch.zeros((len(sand_xyz), 4), device="cuda")
            sand_rotations[:, 0] = 1.0
            sand_scales = torch.ones_like(sand_xyz) * self.config["scale"]
            sand_opacities = torch.ones((len(sand_xyz), 1), device="cuda") * self.config["opacity"]
            colors = torch.tensor(self.config["colors"], device="cuda").expand(len(sand_xyz), -1)

        if self.accumulated_sand_dict is not None:
            fallen_xyz = torch.from_numpy(np.array([])).float().cuda()
            fallen_rotations = torch.from_numpy(np.array([])).float().cuda()
            fallen_scales = torch.from_numpy(np.array([])).float().cuda()
            fallen_opacities = torch.from_numpy(np.array([])).float().cuda()
            fallen_colors = torch.from_numpy(np.array([])).float().cuda()
            for i in range(t):
                if i in self.accumulated_sand_dict:
                    fallen_xyz = torch.concat(
                        [fallen_xyz, torch.from_numpy(self.accumulated_sand_dict[i]["xyz"]).float().cuda()])
                    fallen_rotations = torch.concat(
                        [fallen_rotations, torch.from_numpy(self.accumulated_sand_dict[i]["rotations"]).float().cuda()])
                    fallen_scales = torch.concat(
                        [fallen_scales, torch.from_numpy(self.accumulated_sand_dict[i]["scales"]).float().cuda()])
                    fallen_opacities = torch.concat(
                        [fallen_opacities, torch.from_numpy(self.accumulated_sand_dict[i]["opacity"]).float().cuda()])
                    fallen_colors = torch.concat(
                        [fallen_colors, torch.from_numpy(self.accumulated_sand_dict[i]["color"]).float().cuda()])
            # fallen_xyz = torch.from_numpy(self.fallen_sand_dict["xyz"]).float().cuda()
            # fallen_rotations = torch.from_numpy(self.fallen_sand_dict["rotations"]).float().cuda()
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
            #     num_p, num_f = len(sand_xyz), len(fallen_xyz)
            #     sand_xyz = torch.concat([sand_xyz[:num_p//10], fallen_xyz[:num_f//10]])
            #     sand_scales = torch.concat([sand_scales[:num_p//10], fallen_scales[:num_f//10]])
            #     sand_opacities = torch.concat([sand_opacities[:num_p//10], fallen_opacities[:num_f//10]])
            #     sand_rotations = torch.concatenate([sand_rotations[:num_p//10], fallen_rotations[:num_f//10]])
            #     colors = torch.concat([colors[:num_p//10], fallen_colors])

            sand_xyz = torch.concat([sand_xyz, fallen_xyz])
            sand_scales = torch.concat([sand_scales, fallen_scales])
            sand_opacities = torch.concat([sand_opacities, fallen_opacities])
            sand_rotations = torch.concatenate([sand_rotations, fallen_rotations])
            colors = torch.concat([colors, fallen_colors])

        return sand_xyz, sand_scales, sand_rotations, sand_opacities, colors, reg_loss_m, reg_loss_f, reg_loss_rot_m, reg_loss_rot_f

    def get_sand_gaussians_refine_recurrent(self, t, gm_moving, gm_fallen,
                                  # --- Full t-1 state ---
                                  gm_moving_app_state,
                                  gm_fallen_app_state,
                                  prev_x_rendered_moving,
                                  prev_v_corrected_moving,
                                  prev_x_rendered_fallen,
                                  prev_v_corrected_fallen
                                  ):

        # Get simulation data for *this* timestep (for loss)
        moving_data = self.moving_sand_cache.get(t)
        if moving_data is None:
            print(f"Warning: No moving data in cache for t={t}. Using t=1.")
            moving_data = self.moving_sand_cache.get(1)  # Fallback
            if moving_data is None:
                raise ValueError("Moving sand cache is empty.")

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
        sand_xyz, sand_scales, sand_rotations, sand_opacities, colors = [], [], [], [], []

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
                (sand_scales_m, sand_opacities_m, colors_m,
                 sand_xyz_m, sand_rotations_m,
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
            sand_xyz.append(sand_xyz_m)
            sand_scales.append(sand_scales_m)
            sand_rotations.append(sand_rotations_m)
            sand_opacities.append(sand_opacities_m)
            colors.append(colors_m)

        else:
            # Non-refined case
            sand_xyz.append(torch.from_numpy(moving_data["pos_t"]).float().cuda())
            sand_rotations.append(torch.zeros((len(sand_xyz[-1]), 4), device="cuda"))
            sand_rotations[-1][:, 0] = 1.0
            sand_scales.append(torch.ones_like(sand_xyz[-1]) * self.config["scale"])
            sand_opacities.append(torch.ones((len(sand_xyz[-1]), 1), device="cuda") * self.config["opacity"])
            colors.append(torch.tensor(self.config["colors"], device="cuda").expand(len(sand_xyz[-1]), -1))

            # Pass state through
            new_moving_app_state = gm_moving_app_state
            new_x_rendered_moving = prev_x_rendered_moving
            new_v_corrected_moving = prev_v_corrected_moving

        if self.accumulated_sand_dict is not None:
            # --- Aggregate all fallen particles UP TO this frame ---
            fallen_xyz_sim = torch.from_numpy(np.array([])).float().cuda()
            fallen_rotations_sim = torch.from_numpy(np.array([])).float().cuda()
            fallen_scales_sim = torch.from_numpy(np.array([])).float().cuda()
            fallen_opacities_sim = torch.from_numpy(np.array([])).float().cuda()
            fallen_colors_sim = torch.from_numpy(np.array([])).float().cuda()

            for i in range(t):
                if i in self.accumulated_sand_dict:
                    fallen_xyz_sim = torch.concat(
                        [fallen_xyz_sim, torch.from_numpy(self.accumulated_sand_dict[i]["xyz"]).float().cuda()])
                    fallen_rotations_sim = torch.concat(
                        [fallen_rotations_sim, torch.from_numpy(self.accumulated_sand_dict[i]["rotations"]).float().cuda()])
                    fallen_scales_sim = torch.concat(
                        [fallen_scales_sim, torch.from_numpy(self.accumulated_sand_dict[i]["scales"]).float().cuda()])
                    fallen_opacities_sim = torch.concat(
                        [fallen_opacities_sim, torch.from_numpy(self.accumulated_sand_dict[i]["opacity"]).float().cuda()])
                    fallen_colors_sim = torch.concat(
                        [fallen_colors_sim, torch.from_numpy(self.accumulated_sand_dict[i]["color"]).float().cuda()])

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
                        (sand_scales_f, sand_opacities_f, colors_f,
                         sand_xyz_f, sand_rotations_f,
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
                    sand_xyz.append(sand_xyz_f)
                    sand_scales.append(sand_scales_f)
                    sand_rotations.append(sand_rotations_f)
                    sand_opacities.append(sand_opacities_f)
                    colors.append(colors_f)
                else:
                    # No fallen particles this frame
                    new_fallen_app_state = gm_fallen_app_state
                    new_x_rendered_fallen = prev_x_rendered_fallen
                    new_v_corrected_fallen = prev_v_corrected_fallen
            else:
                # Non-refined case
                if len(fallen_mask) > 0:
                    sand_xyz.append(fallen_xyz_sim)
                    sand_rotations.append(fallen_rotations_sim)
                    sand_scales.append(fallen_scales_sim)
                    sand_opacities.append(fallen_opacities_sim)
                    colors.append(fallen_colors_sim)

                new_fallen_app_state = gm_fallen_app_state
                new_x_rendered_fallen = prev_x_rendered_fallen
                new_v_corrected_fallen = prev_v_corrected_fallen
        else:
            # No fallen sand dict
            new_fallen_app_state = gm_fallen_app_state
            new_x_rendered_fallen = prev_x_rendered_fallen
            new_v_corrected_fallen = prev_v_corrected_fallen

        # Concatenate all renderable particles
        sand_xyz = torch.concat(sand_xyz)
        sand_scales = torch.concat(sand_scales)
        sand_opacities = torch.concat(sand_opacities)
        sand_rotations = torch.concatenate(sand_rotations)
        colors = torch.concat(colors)

        # --- Return ALL new states and renderable params ---
        return (sand_xyz, sand_scales, sand_rotations, sand_opacities, colors,
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
        """
        Gets initial parameters for all possible moving sand particles.
        This is called once before training to initialize the learnable matrices.
        """
        # Assumes a file exists that specifies the total number of unique moving particles
        array_len = np.load(f"{self.simulation_dir}/total_moving.npz")["arr_0"][0]
        sand_scales = self._apply_anisotropic_scaling(torch.ones(array_len,3), self.config["scale"], apply=False)
        sand_opacities = torch.ones((array_len, 1), device="cuda") * self.config["dynamic_opacity"]
        sand_colors = torch.tensor(self.config["color"], device="cuda").expand(array_len, -1)
        clone_scales = self._apply_anisotropic_scaling(torch.ones(array_len,3), self.config["clone_scale"], apply=False)
        clone_opacities = torch.ones((array_len, 1), device="cuda") * self.config["clone_opacity"]
        clone_colors = torch.tensor(self.config["color"], device="cuda").expand(array_len, -1)
        sand_scales = torch.cat([sand_scales, clone_scales])
        sand_opacities = torch.cat([sand_opacities, clone_opacities])
        sand_colors = torch.cat([sand_colors, clone_colors])
        sand_rotations = torch.zeros((len(sand_opacities), 4), device="cuda")
        sand_rotations[:, 0] = 1.0
        return sand_scales, sand_opacities, sand_colors, sand_rotations
    
    def get_fallen_gaussians(self):
        """
        Gets initial parameters for all possible static (fallen) sand particles.
        This is called once before training to initialize the learnable matrices.
        """
        # Assumes a file exists that specifies the maximum number of static particles
        array_len = np.load(f"{self.simulation_dir}/total_static.npz")["arr_0"][0]
        sand_scales = self._apply_anisotropic_scaling(torch.ones(array_len,3), self.config["accumulated_scale"])
        sand_opacities = torch.ones((array_len, 1), device="cuda") * self.config["opacity"]
        sand_colors = torch.tensor(self.config["color"], device="cuda").expand(array_len, -1)
        sand_rotations = torch.zeros((array_len, 4), device="cuda")
        sand_rotations[:, 0] = 1.0
        return sand_scales, sand_opacities, sand_colors, sand_rotations
    
    def render(self,
            viewpoint_camera,
            pc : GaussianModel,
            pipe,
            bg_color : torch.Tensor,
            scaling_modifier = 1.0,
            t=0, refine=False, gm_moving=None, gm_fallen=None, ablation=False,
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
            if not ablation:
                self._add_accumulated_sand(t)
            else:
                self._add_accumulated_sand_ablation(t)

        if refine:
            if not ablation:
                if gm_moving_app_state is None:
                    sand_pos, sand_scales, sand_rot, sand_opac, sand_colors, reg_loss_m, reg_loss_f, reg_loss_rot_m, reg_loss_rot_f = self.get_sand_gaussians_refine(t, gm_moving, gm_fallen)
                else:
                    (sand_pos, sand_scales, sand_rot, sand_opac, sand_colors,
                     reg_loss_m, reg_loss_vel, reg_loss_rot_m,
                     op_reg_loss_m, sc_reg_loss_m, col_reg_loss_m,  # *** NEW ***
                     op_reg_loss_f, sc_reg_loss_f, col_reg_loss_f,  # *** NEW ***
                     diag_delta_v_m, diag_angular_v_m,
                     diag_delta_op_m, diag_delta_sc_m, diag_delta_col_m,
                     diag_delta_op_f, diag_delta_sc_f, diag_delta_col_f,
                     new_moving_app_state, new_fallen_app_state,
                     new_x_rendered_moving, new_v_corrected_moving,
                     new_x_rendered_fallen, new_v_corrected_fallen) = \
                        self.get_sand_gaussians_refine_recurrent(
                            t, gm_moving, gm_fallen,
                            gm_moving_app_state, gm_fallen_app_state,
                            prev_x_rendered_moving, prev_v_corrected_moving,
                            prev_x_rendered_fallen, prev_v_corrected_fallen
                        )

            else:
                sand_pos, sand_scales, sand_rot, sand_opac, sand_colors = self.get_sand_gaussians_refine_ablation(t, gm_moving, gm_fallen)
            # sand_pos = sand_pos.detach()
            # sand_rot = sand_rot.detach()
        else:
            sand_pos, sand_scales, sand_rot, sand_opac, sand_colors = self.get_sand_gaussians(t)

        tot_means3d = torch.concat([means3D, sand_pos]) #if savedz else means3D
        tot_scales = torch.concat([scales, sand_scales]) #if savedz else scales
        tot_rotations = torch.concat([rotations, sand_rot]) #if savedz else rotations
        tot_opacity = torch.concat([opacity, sand_opac]) #if savedz else opacity
        tot_colors_precomp = torch.concat([colors_precomp, sand_colors]) #if savedz else colors_precomp

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
                "reg_loss_vel":reg_loss_vel,
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
                    self._add_accumulated_sand(t)
            
            
            dynamic_shs = dynamic_pc.get_features
            dynamic_scales = dynamic_pc.get_scaling
            dynamic_rotations = dynamic_pc.get_rotation
            dynamic_opacity = dynamic_pc.get_opacity

            xyzs = []
            xyzs.append(torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_pos.npz")["arr_0"]).float())
            moving_xyz = torch.vstack(xyzs).cuda()
            fallen_xyz = torch.from_numpy(np.array([])).float().cuda()
            clone_xyz = moving_xyz + torch.FloatTensor(len(moving_xyz), 3).uniform_(0.001, 0.005).cuda()
            if self.accumulated_sand_dict is not None:
                for i in range(t):
                    if i in self.accumulated_sand_dict:
                        fallen_xyz = torch.concat([fallen_xyz, torch.from_numpy(self.accumulated_sand_dict[i]["xyz"]).float().cuda()])
            fallen_ids = torch.arange(len(fallen_xyz), device="cuda")+ dynamic_pc.num_moving_xyz
            moving_ids = torch.from_numpy(np.load(f"{self.simulation_dir}/{t}_ids.npz")["arr_0"]).int().cuda()
            moving_ids = torch.cat([moving_ids, moving_ids + (dynamic_pc.num_moving_xyz //2)])            

            dynamic_xyz = torch.concat([moving_xyz, clone_xyz, fallen_xyz]).cuda()
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