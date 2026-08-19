import torch
from torch import nn
from cs336_basics.swiglu import SwiGLU
from cs336_basics.attention import Attention
from cs336_basics.rmsnorm import RMSNorm

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attention = Attention(d_model=d_model, num_heads=num_heads, device=device)
        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.rmsnorm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, theta, token_positions, max_seq_len, rope)->torch.Tensor:
        middle_result = x + self.attention(self.rmsnorm1(x), theta, token_positions, max_seq_len, rope)
        final_result = middle_result + self.swiglu(self.rmsnorm2(middle_result))

        return final_result
