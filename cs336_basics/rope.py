import torch
from torch import nn
import math

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.half_dim = d_k // 2
        angels = [[0] * (d_k // 2) for i in range(max_seq_len)]
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                angels[i][k] = i / math.pow(theta, (2*k)/d_k)

        cos_cache = torch.tensor(angels)
        cos_cache = cos_cache.cos()
        sin_cache = torch.tensor(angels)
        sin_cache = sin_cache.sin()

        self.register_buffer(
            "cos_cache",
            cos_cache, 
            persistent=False
        )
        self.register_buffer(
            "sin_cache",
            sin_cache,
            persistent=False
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)-> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        selected_cos = self.cos_cache[token_positions]
        selected_sin = self.sin_cache[token_positions]

        x_even_result = x_even * selected_cos - x_odd * selected_sin
        x_odd_result = x_even * selected_sin + x_odd * selected_cos

        paired = torch.stack(
            [x_even_result, x_odd_result], 
            dim=-1,
        )

        result = paired.flatten(start_dim=-2)

        return result



