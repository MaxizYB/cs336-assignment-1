import torch
from torch import nn
from cs336_basics.softmax import softmax

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # input.shape: (..., vocab_size)
        # targets.shape: (...)

        max_values = input.max(
            dim=-1,
            keepdim=True,
        ).values

        # (..., vocab_size)
        shifted_logits = input - max_values

        # (...)
        log_sum_exp = torch.log(
            torch.exp(shifted_logits).sum(dim=-1)
        )

        # (..., 1)
        target_logits = torch.gather(
            shifted_logits,
            dim=-1,
            index=targets.unsqueeze(-1),
        )

        # (..., 1) -> (...)
        target_logits = target_logits.squeeze(-1)

        # (...)
        losses = log_sum_exp - target_logits

        # 标量
        return losses.mean()