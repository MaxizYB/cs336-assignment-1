import torch
from torch import nn
from torch import einsum
from cs336_basics.softmax import softmax
import math
from cs336_basics.rope import RoPE
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None):
        super().__init__()
        self.q_proj_weight = nn.Parameter(torch.zeros((d_model, d_model), device=device))
        self.k_proj_weight = nn.Parameter(torch.zeros((d_model, d_model), device=device))
        self.v_proj_weight = nn.Parameter(torch.zeros((d_model, d_model), device=device))
        self.o_proj_weight = nn.Parameter(torch.zeros((d_model, d_model), device=device))

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads




    @classmethod
    def scaled_dot_product_attention(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None)-> torch.Tensor:
        qk_product = einsum("... i k, ... j k -> ... i j", q, k)

        if mask is not None:
            qk_product = qk_product.masked_fill(~mask, -torch.inf)

        attention_result = softmax(qk_product / math.sqrt(q.shape[-1]), dim=-1) @ v

        return attention_result

    def forward(self, x: torch.Tensor, theta=None, token_position=None, max_seq_len=None, rope=None, device=None)-> torch.Tensor:
        Q = einsum("... i k, ... j k -> ... i j", x, self.q_proj_weight)
        K = einsum("... i k, ... j k -> ... i j", x, self.k_proj_weight)
        V = einsum("... i k, ... j k -> ... i j", x, self.v_proj_weight)

        Q = Q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2) # 所有权重约定 输出维度，输入维度
        K = K.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
        V = V.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)


        if rope is not None:
            # rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len)
            Q = rope(Q, token_position)
            K = rope(K, token_position)

        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2])).bool()
        multi_result = self.scaled_dot_product_attention(Q, K, V, mask)

        return multi_result.transpose(-3, -2).flatten(-2) @ self.o_proj_weight.transpose(-2, -1)

        


        



        
