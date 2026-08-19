from collections.abc import Iterable
import torch


def gradient_clipping(parameters, max_l2_norm):
    eps = 1e-6
    total_norm = 0.0
    grads = []
    
    for p in parameters:
        if p.grad is not None:
            grad = p.grad
            total_norm += grad.pow(2).sum()
            grads.append(grad)
    
    total_norm = total_norm.sqrt()
    
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(clip_coef)