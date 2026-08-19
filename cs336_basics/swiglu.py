import torch
from torch import nn
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def silu(self, x: torch.Tensor)->torch.Tensor:
        return x / (1+(-x).exp())

    def forward(self, x: torch.Tensor)->torch.Tensor:
        silu_result = self.silu(self.w1(x))
        ffn_result = self.w2(silu_result * self.w3(x))
        return ffn_result
    
