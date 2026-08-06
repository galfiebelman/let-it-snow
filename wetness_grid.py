import torch
import open3d as o3d
import torch.nn.functional as F

class WetnessGrid:
    def __init__(self, mesh_path, grid_res=64, decay_rate=0.95):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.decay_rate = decay_rate
        
        # Setup 3D wetness grid
        bounds = self.mesh.get_axis_aligned_bounding_box()
        self.min_bound = torch.tensor(bounds.min_bound, device='cuda')
        self.max_bound = torch.tensor(bounds.max_bound, device='cuda')
        self.grid_res = torch.tensor([grid_res]*3, device='cuda')
        
        # Initialize wetness grid
        self.wetness_grid = torch.zeros((grid_res, grid_res, grid_res), device='cuda')
        
        # Create raycasting scene
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))

    def _world_to_grid(self, points):
        """Convert world coordinates to grid indices"""
        return (points - self.min_bound) / (self.max_bound - self.min_bound) * (self.grid_res-1)

    def _gaussian_kernel(self, radius=3):
        """Create 3D Gaussian kernel"""
        axis = torch.linspace(-radius, radius, 2*radius+1, device='cuda')
        x, y, z = torch.meshgrid(axis, axis, axis, indexing='ij')
        d = torch.stack([x, y, z], dim=-1)
        weights = torch.exp(-torch.sum(d**2, dim=-1)/(2*(radius/2)**2))
        return weights / weights.sum()

    def add_wetness(self, points, intensity=1.0, radius=3):
        """Add wetness to grid around collision points"""
        # Find closest surface points
        o3d_points = o3d.core.Tensor(points.cpu().numpy(), dtype=o3d.core.Dtype.Float32)
        result = self.scene.compute_closest_points(o3d_points)
        surface_points = torch.from_numpy(result['points'].numpy()).cuda()
        
        # Convert to grid coordinates
        grid_coords = self._world_to_grid(surface_points).long()
        
        # Apply Gaussian kernel
        kernel = self._gaussian_kernel(radius)
        k_center = kernel.shape[0] // 2
        
        for coord in grid_coords:
            x_slice = slice(coord[0]-k_center, coord[0]+k_center+1)
            y_slice = slice(coord[1]-k_center, coord[1]+k_center+1)
            z_slice = slice(coord[2]-k_center, coord[2]+k_center+1)
            
            # Apply bounds checking
            valid_x = (x_slice.start >= 0) & (x_slice.stop <= self.grid_res[0])
            valid_y = (y_slice.start >= 0) & (y_slice.stop <= self.grid_res[1])
            valid_z = (z_slice.start >= 0) & (z_slice.stop <= self.grid_res[2])
            
            if valid_x and valid_y and valid_z:
                self.wetness_grid[x_slice, y_slice, z_slice] += kernel * intensity

    def decay_wetness(self):
        """Apply exponential decay to wetness grid"""
        self.wetness_grid *= self.decay_rate

    def get_wetness(self, points):
        """Get wetness values for 3D points using trilinear interpolation"""
        indices = self._world_to_grid(points).float()
        indices = torch.clamp(indices, 0, self.grid_res[0]-1).long()
        return self.wetness_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
    