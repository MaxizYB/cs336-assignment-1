import torch
from torch import optim
import math
from collections.abc import Callable, Iterable


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        # 1. 检查超参数
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if weight_decay < 0:
            raise ValueError(
                f"Invalid weight_decay: {weight_decay}"
            )

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")

        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")

        # 2. 声明默认超参数
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        # 3. 交给 Optimizer 基类管理参数和参数组
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """
        执行一次参数更新。

        调用 step() 前，用户应当已经执行 loss.backward()，
        因此梯度保存在 parameter.grad 中。
        """

        loss = None

        # 1. 可选 closure
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 2. 遍历参数组
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 3. 遍历这一组中的所有参数
            for parameter in group["params"]:
                # 没有梯度，不更新该参数
                if parameter.grad is None:
                    continue

                grad = parameter.grad

                # 最小实现可以直接拒绝稀疏梯度
                if grad.is_sparse:
                    raise RuntimeError(
                        "MyAdamW does not support sparse gradients"
                    )

                # 4. 取得该参数自己的状态
                state = self.state[parameter]

                # 5. 惰性初始化状态
                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                # 6. 当前参数的迭代次数
                state["step"] += 1
                t = state["step"]

                # 7. 取出一阶矩和二阶矩
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # ==========================================
                # 从这里开始填写 AdamW 数学公式
                # ==========================================

                # 校正学习率
                corrected_lr = (
                    lr
                    * math.sqrt(1 - beta2 ** t)
                    / (1 - beta1 ** t)
                )

                # 解耦权重衰减
                parameter.mul_(1 - lr * weight_decay)

                # 更新一阶矩
                exp_avg.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1,
                )

                # 更新二阶矩
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # 构造分母
                denominator = exp_avg_sq.sqrt().add_(eps)

                # 更新参数
                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-corrected_lr,
                )
        return loss