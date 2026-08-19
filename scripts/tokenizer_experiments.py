"""
运行示例：

1. 压缩率实验：
uv run python scripts/tokenizer_experiments.py ratios \
    --tiny-data data/TinyStoriesV2-GPT4-train.txt \
    --owt-data data/owt_train.txt \
    --tiny-vocab artifacts/vocab.pkl \
    --tiny-merges artifacts/merges.pkl \
    --owt-vocab artifacts/vocab.pkl \
    --owt-merges artifacts/merges.pkl

2. 吞吐量实验：
uv run python scripts/tokenizer_experiments.py throughput \
    --data data/owt_train.txt \
    --vocab artifacts/owt_vocab.pkl \
    --merges artifacts/owt_merges.pkl \
    --sample-mib 10

3. 编码数据集：
uv run python scripts/tokenizer_experiments.py encode \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output encoded/TinyStoriesV2-GPT4-train.bin \
    --vocab artifacts/vocab.pkl \
    --merges artifacts/merges.pkl
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Iterator
from pathlib import Path
from statistics import median

import numpy as np

# 如果你的 Tokenizer 不在这里，请修改这一行
from cs336_basics.Tokenizer import Tokenizer


SPECIAL_TOKEN = "<|endoftext|>"

# 1 MiB = 1024 * 1024 bytes
MIB = 1024 * 1024

# Pile 的大小，按照题目给出的 825 GB 计算
PILE_SIZE_BYTES = 825_000_000_000


# ============================================================
# 文档读取
# ============================================================

def iter_documents(
    file_path: Path,
    separator: str = SPECIAL_TOKEN,
    chunk_size: int = MIB,
) -> Iterator[str]:
    """
    从文件中逐块读取文本，并按照特殊 token 拆分成文档。

    这样不会一次性把整个文件加载进内存。

    注意：
    返回的文档中不包含 <|endoftext|>。
    """

    buffer = ""

    with file_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        while True:
            chunk = file.read(chunk_size)

            if chunk == "":
                break

            buffer += chunk

            # 一个 chunk 可能包含多个文档
            while True:
                separator_index = buffer.find(separator)

                if separator_index == -1:
                    break

                document = buffer[:separator_index]

                # 删除已经处理的文档和特殊 token
                buffer = buffer[
                    separator_index + len(separator):
                ]

                yield document

    # 文件末尾可能还有最后一个文档
    if buffer:
        yield buffer


def iter_corpus_segments(
    file_path: Path,
    separator: str = SPECIAL_TOKEN,
    chunk_size: int = MIB,
) -> Iterator[str]:
    """
    逐段读取语料库，但保留特殊 token。

    例如原始语料：

        doc1<|endoftext|>doc2<|endoftext|>

    依次产生：

        doc1<|endoftext|>
        doc2<|endoftext|>

    这个函数主要用于最终的数据集编码，因为训练数据中需要
    保留 <|endoftext|> 的 token ID。
    """

    buffer = ""

    with file_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        while True:
            chunk = file.read(chunk_size)

            if chunk == "":
                break

            buffer += chunk

            while True:
                separator_index = buffer.find(separator)

                if separator_index == -1:
                    break

                segment_end = separator_index + len(separator)

                # 产生包含特殊 token 的文本段
                yield buffer[:segment_end]

                buffer = buffer[segment_end:]

    if buffer:
        yield buffer


# ============================================================
# 抽样文档
# ============================================================

def sample_documents(
    file_path: Path,
    number_of_documents: int = 10,
    seed: int = 42,
) -> list[str]:
    """
    使用 reservoir sampling 从整个数据集中均匀抽样文档。

    优点：
    不需要预先知道一共有多少篇文档；
    不需要把所有文档加载进内存。
    """

    random_generator = random.Random(seed)

    samples: list[str] = []
    documents_seen = 0

    for document in iter_documents(file_path):
        # 跳过空文档
        if not document.strip():
            continue

        documents_seen += 1

        if len(samples) < number_of_documents:
            samples.append(document)
            continue

        # 从 [0, documents_seen) 中随机选择一个位置
        replacement_index = random_generator.randrange(documents_seen)

        # 只有选中的位置在样本范围内，才执行替换
        if replacement_index < number_of_documents:
            samples[replacement_index] = document

    if len(samples) < number_of_documents:
        raise ValueError(
            f"{file_path} 中只有 {len(samples)} 篇非空文档，"
            f"无法抽样 {number_of_documents} 篇"
        )

    return samples


# ============================================================
# 加载 Tokenizer
# ============================================================

def load_tokenizer(
    vocab_path: Path,
    merges_path: Path,
) -> Tokenizer:
    """
    从 pickle 文件加载分词器。

    特殊 token 必须传给 Tokenizer，否则 <|endoftext|>
    可能会被当成普通文本拆开。
    """

    return Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=[SPECIAL_TOKEN],
    )


# ============================================================
# 压缩率
# ============================================================

def compression_statistics(
    tokenizer: Tokenizer,
    documents: list[str],
) -> tuple[int, int, float]:
    """
    返回：

        UTF-8 字节总数
        token 总数
        bytes/token

    bytes/token 越大，说明平均一个 token 表示的字节越多，
    因此压缩效果越好。
    """

    total_bytes = 0
    total_tokens = 0

    for document in documents:
        document_bytes = len(document.encode("utf-8"))
        token_ids = tokenizer.encode(document)

        total_bytes += document_bytes
        total_tokens += len(token_ids)

    if total_tokens == 0:
        raise ValueError("没有产生任何 token")

    bytes_per_token = total_bytes / total_tokens

    return total_bytes, total_tokens, bytes_per_token


def print_compression_result(
    name: str,
    tokenizer: Tokenizer,
    documents: list[str],
) -> None:
    total_bytes, total_tokens, ratio = compression_statistics(
        tokenizer=tokenizer,
        documents=documents,
    )

    print(f"\n{name}")
    print(f"  UTF-8 bytes : {total_bytes:,}")
    print(f"  tokens      : {total_tokens:,}")
    print(f"  bytes/token : {ratio:.4f}")


def run_ratio_experiments(arguments: argparse.Namespace) -> None:
    """
    完成题目的 (a) 和 (b)。
    """

    print("正在加载分词器……")

    tiny_tokenizer = load_tokenizer(
        vocab_path=arguments.tiny_vocab,
        merges_path=arguments.tiny_merges,
    )

    owt_tokenizer = load_tokenizer(
        vocab_path=arguments.owt_vocab,
        merges_path=arguments.owt_merges,
    )

    print("正在抽样文档……")

    tiny_documents = sample_documents(
        file_path=arguments.tiny_data,
        number_of_documents=arguments.num_documents,
        seed=arguments.seed,
    )

    owt_documents = sample_documents(
        file_path=arguments.owt_data,
        number_of_documents=arguments.num_documents,
        seed=arguments.seed,
    )

    # 题目 (a)
    print_compression_result(
        name="TinyStories tokenizer -> TinyStories documents",
        tokenizer=tiny_tokenizer,
        documents=tiny_documents,
    )

    print_compression_result(
        name="OpenWebText tokenizer -> OpenWebText documents",
        tokenizer=owt_tokenizer,
        documents=owt_documents,
    )

    # 题目 (b)
    print_compression_result(
        name="TinyStories tokenizer -> OpenWebText documents",
        tokenizer=tiny_tokenizer,
        documents=owt_documents,
    )


# ============================================================
# 吞吐量测试
# ============================================================

def load_benchmark_text(
    file_path: Path,
    target_bytes: int,
) -> str:
    """
    从数据集中读取若干完整文档，直到大约达到 target_bytes。

    返回值会保留文档之间的 <|endoftext|>。
    """

    documents: list[str] = []
    current_bytes = 0
    separator_bytes = len(SPECIAL_TOKEN.encode("utf-8"))

    for document in iter_documents(file_path):
        if documents:
            current_bytes += separator_bytes

        documents.append(document)
        current_bytes += len(document.encode("utf-8"))

        if current_bytes >= target_bytes:
            break

    if not documents:
        raise ValueError(f"{file_path} 中没有读取到文档")

    return SPECIAL_TOKEN.join(documents)


def benchmark_tokenizer(
    tokenizer: Tokenizer,
    text: str,
    repeats: int = 3,
) -> tuple[float, int]:
    """
    测量 tokenizer.encode() 的吞吐量。

    返回：

        median bytes/second
        token 数量
    """

    input_bytes = len(text.encode("utf-8"))

    if input_bytes == 0:
        raise ValueError("吞吐量测试文本不能为空")

    # 预热，减少第一次执行带来的额外影响
    warmup_text = text[: min(len(text), 100_000)]
    tokenizer.encode(warmup_text)

    throughputs: list[float] = []
    token_count = 0

    for run_index in range(repeats):
        start_time = time.perf_counter()

        token_ids = tokenizer.encode(text)

        elapsed_time = time.perf_counter() - start_time
        token_count = len(token_ids)

        bytes_per_second = input_bytes / elapsed_time
        throughputs.append(bytes_per_second)

        print(
            f"run {run_index + 1}: "
            f"{elapsed_time:.3f} seconds, "
            f"{bytes_per_second / MIB:.2f} MiB/s"
        )

        # 不需要保存这些 token
        del token_ids

    return median(throughputs), token_count


def run_throughput_experiment(
    arguments: argparse.Namespace,
) -> None:
    """
    完成题目的 (c)。
    """

    tokenizer = load_tokenizer(
        vocab_path=arguments.vocab,
        merges_path=arguments.merges,
    )

    target_bytes = arguments.sample_mib * MIB

    print(
        f"正在加载大约 {arguments.sample_mib} MiB 的测试文本……"
    )

    benchmark_text = load_benchmark_text(
        file_path=arguments.data,
        target_bytes=target_bytes,
    )

    actual_bytes = len(benchmark_text.encode("utf-8"))

    print(f"实际测试文本大小：{actual_bytes / MIB:.2f} MiB")
    print("开始测试吞吐量……")

    bytes_per_second, token_count = benchmark_tokenizer(
        tokenizer=tokenizer,
        text=benchmark_text,
        repeats=arguments.repeats,
    )

    estimated_seconds = PILE_SIZE_BYTES / bytes_per_second
    estimated_hours = estimated_seconds / 3600
    estimated_days = estimated_hours / 24

    print("\n吞吐量测试结果")
    print(f"  token count : {token_count:,}")
    print(f"  bytes/second: {bytes_per_second:,.2f}")
    print(f"  MiB/second  : {bytes_per_second / MIB:.2f}")
    print(f"  Pile hours  : {estimated_hours:.2f}")
    print(f"  Pile days   : {estimated_days:.2f}")


# ============================================================
# uint16 数据集编码
# ============================================================

def encode_dataset_to_uint16(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    buffer_size: int = 1_000_000,
    overwrite: bool = False,
) -> int:
    """
    将整个文本数据集编码为连续的 uint16 token ID。

    采用流式处理：

    1. 逐篇文档读取；
    2. 使用 encode_iterable() 惰性产生 token ID；
    3. 积累一定数量后写入磁盘；
    4. 不把全部 token 放进内存。

    返回写入的 token 总数。
    """

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"输出文件已经存在：{output_path}\n"
            "如果确定要覆盖，请加上 --overwrite"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    token_buffer: list[int] = []
    total_tokens = 0
    next_progress_report = 10_000_000

    text_segments = iter_corpus_segments(input_path)

    # encode_iterable 返回 Iterator[int]
    token_iterator = tokenizer.encode_iterable(text_segments)

    with output_path.open("wb") as output_file:
        for token_id in token_iterator:
            token_id = int(token_id)

            # uint16 能保存的范围是 0 到 65535
            if not 0 <= token_id <= np.iinfo(np.uint16).max:
                raise ValueError(
                    f"token ID {token_id} 超出了 uint16 的范围"
                )

            token_buffer.append(token_id)
            total_tokens += 1

            if len(token_buffer) >= buffer_size:
                token_array = np.asarray(
                    token_buffer,
                    dtype=np.uint16,
                )

                token_array.tofile(output_file)
                token_buffer.clear()

            if total_tokens >= next_progress_report:
                print(f"已编码 {total_tokens:,} 个 token")

                while total_tokens >= next_progress_report:
                    next_progress_report += 10_000_000

        # 写入最后不足 buffer_size 的 token
        if token_buffer:
            token_array = np.asarray(
                token_buffer,
                dtype=np.uint16,
            )

            token_array.tofile(output_file)

    expected_file_size = total_tokens * np.dtype(np.uint16).itemsize
    actual_file_size = output_path.stat().st_size

    if actual_file_size != expected_file_size:
        raise RuntimeError(
            "输出文件大小不正确："
            f"expected={expected_file_size}, "
            f"actual={actual_file_size}"
        )

    return total_tokens


def run_dataset_encoding(
    arguments: argparse.Namespace,
) -> None:
    """
    完成题目的 (d)。
    """

    tokenizer = load_tokenizer(
        vocab_path=arguments.vocab,
        merges_path=arguments.merges,
    )

    print(f"输入文件：{arguments.input}")
    print(f"输出文件：{arguments.output}")
    print("开始编码……")

    start_time = time.perf_counter()

    total_tokens = encode_dataset_to_uint16(
        tokenizer=tokenizer,
        input_path=arguments.input,
        output_path=arguments.output,
        buffer_size=arguments.buffer_size,
        overwrite=arguments.overwrite,
    )

    elapsed_time = time.perf_counter() - start_time
    output_size = arguments.output.stat().st_size

    print("\n编码完成")
    print(f"  tokens     : {total_tokens:,}")
    print(f"  time       : {elapsed_time:.2f} seconds")
    print(f"  output size: {output_size / MIB:.2f} MiB")

    print("\n可以使用下面的代码读取：")
    print(
        f'data = np.memmap("{arguments.output}", '
        'dtype=np.uint16, mode="r")'
    )


# ============================================================
# 命令行参数
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CS336 tokenizer experiments"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------
    # ratios
    # --------------------------------------------------------

    ratios_parser = subparsers.add_parser(
        "ratios",
        help="运行压缩率与跨数据集实验",
    )

    ratios_parser.add_argument(
        "--tiny-data",
        type=Path,
        required=True,
    )

    ratios_parser.add_argument(
        "--owt-data",
        type=Path,
        required=True,
    )

    ratios_parser.add_argument(
        "--tiny-vocab",
        type=Path,
        required=True,
    )

    ratios_parser.add_argument(
        "--tiny-merges",
        type=Path,
        required=True,
    )

    ratios_parser.add_argument(
        "--owt-vocab",
        type=Path,
        required=True,
    )

    ratios_parser.add_argument(
        "--owt-merges",
        type=Path,
        required=True,
    )

    ratios_parser.add_argument(
        "--num-documents",
        type=int,
        default=10,
    )

    ratios_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ratios_parser.set_defaults(
        function=run_ratio_experiments
    )

    # --------------------------------------------------------
    # throughput
    # --------------------------------------------------------

    throughput_parser = subparsers.add_parser(
        "throughput",
        help="测试分词器吞吐量",
    )

    throughput_parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )

    throughput_parser.add_argument(
        "--vocab",
        type=Path,
        required=True,
    )

    throughput_parser.add_argument(
        "--merges",
        type=Path,
        required=True,
    )

    throughput_parser.add_argument(
        "--sample-mib",
        type=int,
        default=10,
    )

    throughput_parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    throughput_parser.set_defaults(
        function=run_throughput_experiment
    )

    # --------------------------------------------------------
    # encode
    # --------------------------------------------------------

    encode_parser = subparsers.add_parser(
        "encode",
        help="将数据集编码为 uint16 文件",
    )

    encode_parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    encode_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    encode_parser.add_argument(
        "--vocab",
        type=Path,
        required=True,
    )

    encode_parser.add_argument(
        "--merges",
        type=Path,
        required=True,
    )

    encode_parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
    )

    encode_parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    encode_parser.set_defaults(
        function=run_dataset_encoding
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    # 每一个子命令都通过 set_defaults 设置了对应函数
    arguments.function(arguments)


if __name__ == "__main__":
    main()