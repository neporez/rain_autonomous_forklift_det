import time

import torch
import torch.nn as nn
from .builder import serializer

from .serialization import encode

@serializer.register_module('serializer')
class Serialization(nn.Module):
    def __init__(self, serializer_cfg):
        super(Serialization, self).__init__()
        self.cfg = serializer_cfg
        self.grid_size = self.cfg.get('grid_size', None)
        self.depth = self.cfg.get('depth', None)
        self.order = self.cfg.get('order', None)
    
    @torch.no_grad()
    def forward(self, xyz: torch.Tensor, bid: torch.Tensor, **kwargs) -> torch.Tensor:
        xyz = xyz.contiguous()
        bid = bid.contiguous()
        B,N, _ = xyz.shape
        xyz_flat = xyz.view(B * N, 3)
        bid_flat = bid.view(B * N)

        min_coord = xyz_flat.min(dim=0, keepdim=True).values
        shifted = xyz_flat - min_coord
        grid_coord = torch.floor(shifted / self.grid_size).to(torch.int32)

        code = encode(grid_coord, batch=bid_flat, depth=self.depth, order=self.order)  # (M,)

        sorted_idx = torch.argsort(code)  # (M,)

        sorted_idx_local = sorted_idx - bid_flat * N  # (B×N,)

        sorted_idx_per_batch = sorted_idx_local.view(B, N)

        return sorted_idx_per_batch




