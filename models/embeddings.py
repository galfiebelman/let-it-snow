import math
import torch
import torch.nn as nn


class XYZEmbedding(nn.Module):
    """Fourier feature embedding for 3D coordinates."""

    def __init__(self, embed_dim=12, num_freqs=8, sigma=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_freqs = num_freqs
        self.sigma = sigma
        if embed_dim % (3 * 2) != 0:
            raise ValueError(f"XYZ embed_dim ({embed_dim}) must be divisible by 6.")
        self.dims_per_coord = embed_dim // 3
        freqs = self.sigma * torch.randn(self.dims_per_coord // 2) * (
                2 ** torch.arange(self.dims_per_coord // 2) / (self.dims_per_coord // 2))
        self.register_buffer('freqs', freqs)

    def forward(self, xyz):
        xyz_unsqueezed = xyz.unsqueeze(-1)
        args = xyz_unsqueezed * self.freqs.view(1, 1, -1)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding.view(xyz.shape[0], -1)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        max_period = 10000.0
        freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim) / half_dim)
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        args = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding
