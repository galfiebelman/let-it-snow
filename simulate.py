"""
Unified weather simulation script for Let It Snow.

Runs MPM-based particle simulation for various weather effects (snow, rain, fog,
sand) on 3D Gaussian Splatting scenes, driven by JSON config files.

Usage:
    python simulate.py --model_path <path> --config configs/snow.json \
        --mesh_path <mesh.ply> [--output_dir <dir>] [--num_steps <N>]
"""

import os
import json
import pickle
import numpy as np
import taichi as ti
from tqdm import trange
from argparse import ArgumentParser

from arguments import ModelParams, get_combined_args
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from taichi_elements.engine.mpm_solver import MPMSolver
import open3d as o3d

GROUND_Y = 0.05

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        t = np.sqrt(1 + trace)
        w = 0.5 * t
        x = (R[2, 1] - R[1, 2]) / (2 * w)
        y = (R[0, 2] - R[2, 0]) / (2 * w)
        z = (R[1, 0] - R[0, 1]) / (2 * w)
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            t = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            x = 0.5 * t
            w = (R[2, 1] - R[1, 2]) / (2 * x)
            y = (R[0, 1] + R[1, 0]) / (2 * x)
            z = (R[0, 2] + R[2, 0]) / (2 * x)
        elif R[1, 1] > R[2, 2]:
            t = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
            y = 0.5 * t
            w = (R[0, 2] - R[2, 0]) / (2 * y)
            x = (R[0, 1] + R[1, 0]) / (2 * y)
            z = (R[1, 2] + R[2, 1]) / (2 * y)
        else:
            t = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
            z = 0.5 * t
            w = (R[1, 0] - R[0, 1]) / (2 * z)
            x = (R[0, 2] + R[2, 0]) / (2 * z)
            y = (R[1, 2] + R[2, 1]) / (2 * z)

    return np.array([w, x, y, z])

# ---------------------------------------------------------------------------
# Particle emission
# ---------------------------------------------------------------------------

def emit_particles(mpm, material_type, sim_cfg):
    """Emit particles according to the simulation config.

    For most effects particles are spawned in a plane near the top of the
    scene.  Fog (``initial_velocity`` with non-zero x and near-zero y) spawns
    particles throughout the full scene volume instead.
    """
    n = sim_cfg["num_particles_per_emission"]
    init_vel = sim_cfg["initial_velocity"]
    vel_var = sim_cfg["velocity_variation"]

    # Determine emission region: fog fills the volume, others rain from above.
    is_volumetric = abs(init_vel[1]) < 0.01  # near-zero vertical velocity
    x = np.random.uniform(0.05, 0.95, n)
    if is_volumetric:
        y = np.random.uniform(0.05, 0.95, n)
    else:
        y = np.random.uniform(0.80, 0.95, n)
    z = np.random.uniform(0.05, 0.95, n)

    points = np.stack([x, y, z], axis=1)

    # Per-particle velocity with configured variation
    vel_x = init_vel[0] + np.random.uniform(vel_var[0][0], vel_var[0][1], n)
    vel_y = init_vel[1] + np.random.uniform(vel_var[1][0], vel_var[1][1], n)
    vel_z = init_vel[2] + np.random.uniform(vel_var[2][0], vel_var[2][1], n)

    velocity = np.stack([vel_x, vel_y, vel_z], axis=1)

    mpm.add_particles(
        particles=points,
        material=material_type,
        velocity=[velocity[:, 0].mean(), velocity[:, 1].mean(), velocity[:, 2].mean()],
        color=0xEEEEF0,
    )

# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def resolve_material(material_name):
    """Map a material name string to the corresponding MPMSolver constant."""
    mapping = {
        "snow": MPMSolver.material_snow,
        "water": MPMSolver.material_water,
        "sand": MPMSolver.material_sand,
        "elastic": MPMSolver.material_elastic,
    }
    if material_name not in mapping:
        raise ValueError(f"Invalid material type: {material_name}")
    return mapping[material_name]


def simulate(dataset, config, output_dir, mesh_path, num_steps_override=None,
             enable_gui=False, mpm_step_size=4e-3):
    """Run the full MPM simulation driven by *config* (parsed JSON dict)."""

    sim_cfg = config["simulation"]

    # ------------------------------------------------------------------
    # Load 3DGS model
    # ------------------------------------------------------------------
    load_iters = searchForMaxIteration(
        os.path.join(dataset.model_path, "point_cloud")
    )
    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    gaussians.load_ply(
        os.path.join(
            dataset.model_path, "point_cloud",
            f"iteration_{load_iters}", "point_cloud.ply",
        )
    )

    modifier_dict_pt = os.path.join(
        dataset.model_path, "point_cloud",
        f"iteration_{load_iters}", "editing_modifier.pkl",
    )
    with open(modifier_dict_pt, "rb") as f:
        editing_modifier_dict = pickle.load(f)

    # ------------------------------------------------------------------
    # Scene geometry
    # ------------------------------------------------------------------
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    ground_R = editing_modifier_dict["scene"]["ground_R"]
    ground_T = editing_modifier_dict["scene"]["ground_T"]

    mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
    bounds = mesh_o3d.get_axis_aligned_bounding_box()
    min_bound = bounds.min_bound
    max_bound = bounds.max_bound

    within_bbox = (
        (xyz[:, 0] > min_bound[0]) & (xyz[:, 0] < max_bound[0])
        & (xyz[:, 1] > min_bound[1]) & (xyz[:, 1] < max_bound[1])
        & (xyz[:, 2] > min_bound[2]) & (xyz[:, 2] < max_bound[2])
    )
    particles = xyz[within_bbox].copy()
    real_gaussian_particle_size = particles.shape[0]

    # Transform to ground-aligned coordinates
    particles = particles @ ground_R.T + ground_T

    # ------------------------------------------------------------------
    # Normalize to unit cube
    # ------------------------------------------------------------------
    particle_max = particles.max(axis=0)
    particle_min = particles.min(axis=0)
    particle_min[1] = min(particle_min[1], GROUND_Y)
    longest_side = max(particle_max - particle_min)

    particles /= longest_side

    shift_constant = np.array([
        -particles[:, 0].mean() + 0.5,
        -particles[:, 1].min(),
        -particles[:, 2].mean() + 0.5,
    ])
    particles += shift_constant

    y_len = particles[:, 1].max() - particles[:, 1].min()
    particles[:, 1] /= y_len

    # ------------------------------------------------------------------
    # Initialise Taichi & MPM solver
    # ------------------------------------------------------------------
    sim_res = sim_cfg.get("grid_resolution", 64)
    youngs_modulus = sim_cfg.get("youngs_modulus", 0.14)
    poisson_ratio = sim_cfg.get("poisson_ratio", 0.2)

    ti.init(arch=ti.cuda, device_memory_GB=20.0)

    if enable_gui:
        gui = ti.GUI("Let It Snow - Simulation", res=512, background_color=0x112F41)

    mpm = MPMSolver(
        res=(sim_res, sim_res, sim_res),
        size=1,
        max_num_particles=2 ** 21,
        E_scale=youngs_modulus,
        poisson_ratio=poisson_ratio,
        unbounded=True,
    )

    # Add scene Gaussians as stationary material
    mpm.add_particles(
        particles=particles,
        material=MPMSolver.material_stationary,
        color=0xFFFF00,
        motion_override_flag_arr=None,
    )

    # Ground collider
    mpm.add_surface_collider(
        point=(0.0, GROUND_Y - 0.1, 0.0),
        normal=(0, 1, 0),
        surface=mpm.surface_sticky,
    )

    gravity = tuple(sim_cfg.get("gravity", [0.0, -9.8, 0.0]))
    mpm.set_gravity(gravity)

    # ------------------------------------------------------------------
    # Resolve parameters
    # ------------------------------------------------------------------
    material_type = resolve_material(sim_cfg["material_type"])
    num_sim_steps = num_steps_override or sim_cfg.get("num_simulation_steps", 500)
    emission_interval = sim_cfg.get("emission_interval", 2)
    active_threshold = sim_cfg.get("active_threshold", 0.001)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if output_dir is None:
        output_dir = os.path.join(
            dataset.model_path, "simulation", config["effect"]
        )
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    # Save config alongside outputs for reproducibility
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    tot_inactive = 0

    for frame in trange(num_sim_steps, desc=f"Simulating {config['effect']}"):
        # Periodically emit new particles
        if frame % emission_interval == 0:
            emit_particles(mpm, material_type, sim_cfg)

        particles_info = mpm.particle_info()
        prev_active = particles_info["active"][real_gaussian_particle_size:].astype(bool).copy()
        pos_prev = particles_info["position"].copy()

        mpm.step(mpm_step_size, override_velocity=[0, 0, 0])
        mpm.update_active_snow_particles(pos_prev, threshold=active_threshold)

        particles_info = mpm.particle_info()
        real_active = particles_info["active"][real_gaussian_particle_size:].astype(bool).copy()

        # Extract active particle positions and map back to world space
        real_gaussian_pos = particles_info["position"][real_gaussian_particle_size:][real_active].copy()
        ids = np.arange(len(particles_info["active"][real_gaussian_particle_size:]))[real_active]

        real_gaussian_pos[:, 1] *= y_len
        real_gaussian_pos -= shift_constant
        real_gaussian_pos *= longest_side
        real_gaussian_pos = real_gaussian_pos - ground_T
        real_gaussian_pos = real_gaussian_pos @ ground_R

        np.savez(f"{save_dir}/{frame}_pos.npz", real_gaussian_pos)
        np.savez(f"{save_dir}/{frame}_ids.npz", ids)

        # Track newly-static particles
        new_inactive = ~real_active & prev_active
        if np.sum(new_inactive) > 0:
            tot_inactive += np.sum(new_inactive)
            new_inactive_pos = particles_info["position"][real_gaussian_particle_size:][new_inactive].copy()
            new_inactive_pos[:, 1] *= y_len
            new_inactive_pos -= shift_constant
            new_inactive_pos *= longest_side
            new_inactive_pos = new_inactive_pos - ground_T
            new_inactive_pos = new_inactive_pos @ ground_R
            np.savez(f"{save_dir}/{frame}_static.npz", new_inactive_pos)

        if enable_gui:
            np_x = particles_info["position"] / 1.0
            screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2 ** 0.5) - 0.2
            screen_y = np_x[:, 1]
            screen_pos = np.stack([screen_x, screen_y], axis=-1)
            color = particles_info["color"]
            gui.circles(screen_pos, radius=2, color=color)
            gui.show()

    # Save aggregate counts
    np.savez(f"{save_dir}/total_static.npz", np.array([tot_inactive]))
    np.savez(
        f"{save_dir}/total_moving.npz",
        np.array([len(particles_info["active"][real_gaussian_particle_size:])]),
    )
    print(f"Simulation complete. {num_sim_steps} frames saved to {save_dir}")


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="Unified weather simulation for 3DGS scenes")
    model = ModelParams(parser, sentinel=True)

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to a JSON config file (e.g. configs/snow.json)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save simulation outputs (default: <model_path>/simulation/<effect>)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=None,
        help="Override the number of simulation steps from the config",
    )
    parser.add_argument(
        "--mesh_path", type=str, required=True,
        help="Path to the scene mesh (.ply) used for bounding-box filtering",
    )
    parser.add_argument(
        "--enable_gui", action="store_true", default=False,
        help="Show a Taichi GUI window during simulation",
    )
    parser.add_argument(
        "--mpm_step_size", type=float, default=4e-3,
        help="MPM solver time-step size",
    )

    args = get_combined_args(parser)

    # Load effect config
    with open(args.config, "r") as f:
        config = json.load(f)

    print(f"Simulating '{config['effect']}' effect for model: {args.model_path}")

    simulate(
        dataset=model.extract(args),
        config=config,
        output_dir=args.output_dir,
        mesh_path=args.mesh_path,
        num_steps_override=args.num_steps,
        enable_gui=args.enable_gui,
        mpm_step_size=args.mpm_step_size,
    )
