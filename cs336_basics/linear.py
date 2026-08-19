import torch
import math
from torch import nn
from torch import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        sigma = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(
            w, 
            mean=0.0,
            std=sigma,
            a=-3.0*sigma,
            b=3.0*sigma
        )
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        trans = einsum("... i, j i -> ... j", x, self.weight )
        return trans
    

