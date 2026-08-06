"""
Prepare a trained 3DGS scene for weather simulation.

Estimates the ground plane via RANSAC and saves the ground alignment transform
(editing_modifier.pkl) needed by simulate.py.

Usage:
    python prepare_scene.py -m output/garden [--ground_label "ground,grass,floor"]
"""

import os
import pickle
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R

from arguments import ModelParams, get_combined_args
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration


def point_to_plane_distance(point, plane):
    x, y, z = point
    A, B, C, D = plane
    return abs(A * x + B * y + C * z + D) / np.sqrt(A**2 + B**2 + C**2)


def estimate_ground_plane(points, distance_threshold=0.02, ransac_n=3, num_iterations=2000):
    """Estimate ground plane from point cloud using RANSAC.

    Assumes the ground plane is the dominant horizontal plane.  We select the
    lowest 30% of points (by y-coordinate in the original COLMAP frame) as
    candidates, then fit a plane with RANSAC.

    Returns (ground_R, ground_T) that rotate the scene so y-axis is up and the
    ground sits near y=0.
    """
    y_vals = points[:, 1]
    y_thresh = np.percentile(y_vals, 30)
    low_points = points[y_vals <= y_thresh]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(low_points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    origin_plane_distance = point_to_plane_distance((0, 0, 0), plane_model)

    plane_normal = np.array(plane_model[:3])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    y_axis = np.array([0.0, -1.0, 0.0])

    rotation_angle = np.arccos(np.clip(np.dot(plane_normal, y_axis), -1, 1))
    rotation_axis = np.cross(plane_normal, y_axis)
    norm = np.linalg.norm(rotation_axis)
    if norm < 1e-8:
        rotation_matrix = np.eye(3)
    else:
        rotation_axis = rotation_axis / norm
        axis_angle = rotation_axis * rotation_angle
        rotation_matrix = R.from_rotvec(axis_angle).as_matrix()

    ground_T = np.array([0.0, origin_plane_distance, 0.0])

    print(f"Plane equation: {plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0")
    print(f"Plane normal: {plane_normal}")
    print(f"Inliers: {len(inliers)} / {len(low_points)} candidate points")

    return rotation_matrix, ground_T, inliers


def main():
    parser = ArgumentParser(description="Prepare scene for weather simulation")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--distance_threshold", type=float, default=0.02,
                        help="RANSAC distance threshold for ground plane estimation")
    args = get_combined_args(parser)
    dataset = model.extract(args)

    load_iters = searchForMaxIteration(
        os.path.join(dataset.model_path, "point_cloud")
    )
    print(f"Loading model from iteration {load_iters}")

    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    gaussians.load_ply(
        os.path.join(
            dataset.model_path, "point_cloud",
            f"iteration_{load_iters}", "point_cloud.ply",
        )
    )

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    print(f"Loaded {xyz.shape[0]} Gaussians")

    ground_R, ground_T, inliers = estimate_ground_plane(
        xyz, distance_threshold=args.distance_threshold
    )

    editing_modifier_dict = {
        "scene": {
            "ground_R": ground_R,
            "ground_T": ground_T,
        },
        "objects": [],
    }

    save_path = os.path.join(
        dataset.model_path, "point_cloud",
        f"iteration_{load_iters}", "editing_modifier.pkl",
    )
    with open(save_path, "wb") as f:
        pickle.dump(editing_modifier_dict, f)
    print(f"Saved ground plane alignment to {save_path}")


if __name__ == "__main__":
    main()
