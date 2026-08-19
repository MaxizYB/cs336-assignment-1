import numpy as np
import torch

def data_loading(dataset, batch_size, context_length, device):
    x = torch.from_numpy(dataset).to(device)
    max_start = len(x) - context_length - 1
    starts = torch.randint(0, max_start+1, (batch_size,), device=device)
    offsets = torch.arange(context_length, device=device)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
    inputs = x[idx]          # 取 x[i : i+m]
    targets = x[idx + 1]     # 取 x[i+1 : i+m+1] (即预测下一个token)
    
    return inputs, targets
