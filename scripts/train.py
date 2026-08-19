import argparse
from pathlib import Path
import numpy as np
import torch
from cs336_basics.transformer import Transformer
from cs336_basics.adamw import AdamW
from cs336_basics.lr_schedule import lr_schedule



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer"
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("encoded/TinyStoriesV2-GPT4-train.bin")
    )
    parser.add_argument(
        "--valid-data",
        type=Path,
        default=Path("encoded/TinyStoriesV2-GPT4-valid.bin")
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--config",
        type=Path,
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/output")
    )

    


def set_seed(seed):
    """设置 Python、NumPy、PyTorch 随机种子。"""
    ...


def load_datasets(args)->tuple[np.typing.NDArray, np.typing.NDArray]:
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    valid_data = np.memmap(args.valid_data, dtype=np.uint16, mode="r")

    return train_data, valid_data
    


def build_model(args)->torch.nn.Module:
    transformer = Transformer(d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff, device=args.device)

    return transformer


def build_optimizer(model: torch.nn.Module, args):
    optimizer = AdamW(model.parameters)

    return optimizer


def evaluate(model, validation_data, args):
    """在验证集上采样多个 batch，返回平均验证损失。"""
    ...


def train(args):

    train_data, valid_data = load_datasets(args)

    model = build_model(args)

    optimizer = build_optimizer(args)

    max_iters = args.max_iters

    for it in range(max_iters):
        lr = lr_schedule(it, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)


def main():
    """程序入口。"""
    


if __name__ == "__main__":
    main()