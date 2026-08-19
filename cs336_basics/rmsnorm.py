import torch
from torch import nn
import math


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        g = torch.ones((d_model), device=device, dtype=dtype)

        self.weight = nn.Parameter(g) # Q: 何时使用torch. 何时使用nn.


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)

        rms = ((x*x).sum(dim=-1, keepdim=True)/self.d_model + self.eps).sqrt()
        # 很重要这里，需要整理

        result = x * self.weight / rms

        return result.to(dtype)

