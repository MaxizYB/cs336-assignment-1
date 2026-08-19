import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        emb = torch.zeros((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(
            emb, 
            mean=0,
            std=1,
            a=-3.0,
            b=3.0
        )
        self.weight = nn.Parameter(emb)

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        selected = self.weight[token_ids]
        return selected