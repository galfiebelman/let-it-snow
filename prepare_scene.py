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
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.system_utils import searchForMaxIteration


def up_from_cameras(train_cameras):
    """World up direction, estimated as the mean camera image-up vector.

    The COLMAP frame is not axis-aligned, so we cannot assume any coordinate
    axis is vertical. Each camera's image-up direction (-y in camera space)
    points roughly toward the sky; averaging over all views is robust.
    """
    ups = [ci.R @ np.array([0.0, -1.0, 0.0]) for ci in train_cameras]
    up = np.mean(ups, axis=0)
    return up / np.linalg.norm(up)


def estimate_ground_plane(points, up, distance_threshold=0.02, low_percent=30,
                          ransac_n=3, num_iterations=2000):
    """Estimate the ground plane via RANSAC using a data-derived up direction.

    Candidate ground points are the lowest `low_percent` of points measured
    *along `up`* (not along a fixed axis). RANSAC fits a plane to them; the
    normal is oriented toward `up`, and a rotation is built that maps the
    ground normal to +Y with the ground at Y=0 (the frame simulate.py expects,
    gravity = [0, -9.8, 0]).

    Returns (ground_R, ground_T, inliers).
    """
    height = points @ up
    thresh = np.percentile(height, low_percent)
    low_points = points[height <= thresh]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(low_points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    n = np.array(plane_model[:3], dtype=np.float64)
    D = float(plane_model[3])
    norm = np.linalg.norm(n)
    n /= norm
    D /= norm
    if np.dot(n, up) < 0:          # orient normal toward up
        n, D = -n, -D

    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(n, target)
    s = np.linalg.norm(v)
    c = float(np.dot(n, target))
    if s < 1e-8:
        ground_R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        ground_R = R.from_rotvec((v / s) * np.arccos(np.clip(c, -1, 1))).as_matrix()

    p0 = -D * n                    # point on plane closest to origin
    ground_T = np.array([0.0, -(ground_R @ p0)[1], 0.0])

    print(f"Up direction: {np.round(up, 4)}")
    print(f"Plane normal (oriented up): {np.round(n, 4)}   offset D={D:.4f}")
    print(f"Angle(plane normal, up): {np.degrees(np.arccos(np.clip(np.dot(n, up), -1, 1))):.2f} deg")
    print(f"Inliers: {len(inliers)} / {len(low_points)} candidate points")

    return ground_R, ground_T, inliers


def main():
    parser = ArgumentParser(description="Prepare scene for weather simulation")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--distance_threshold", type=float, default=0.02,
                        help="RANSAC distance threshold for ground plane estimation")
    parser.add_argument("--low_percent", type=float, default=30,
                        help="Percentage of lowest points (along the up direction) used as ground candidates")
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

    scene_info = sceneLoadTypeCallbacks["Colmap"](dataset.source_path, dataset.images, dataset.eval)
    up = up_from_cameras(scene_info.train_cameras)

    ground_R, ground_T, inliers = estimate_ground_plane(
        xyz, up, distance_threshold=args.distance_threshold, low_percent=args.low_percent
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
