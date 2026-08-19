import torch
from torch import nn
import math

def lr_schedule(
        it: int, 
        max_learning_rate: float,
        min_learnint_rate: float, 
        warmup_iters: int,
        cosine_cycle_iters: int,
)-> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    elif it < cosine_cycle_iters:
        return min_learnint_rate + 0.5*(max_learning_rate - min_learnint_rate) * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))

    else:
        return min_learnint_rate
        