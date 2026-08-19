import torch
from torch import nn

def softmax(x: torch.Tensor, dim: int)-> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    result = x.exp() / (x.exp()).sum(dim=dim, keepdim=True)
    return result
