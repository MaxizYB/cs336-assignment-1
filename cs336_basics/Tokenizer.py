import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re


PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # 复制一份，避免外部修改原对象影响Tokenizer
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        # bytes -> token ID
        # 编码时需要反向查询
        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        # 如果特殊token不在词表中，将其加入词表
        next_token_id = max(
            self.vocab.keys(),
            default=-1,
        ) + 1

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")

            if special_bytes not in self.token_to_id:
                self.vocab[next_token_id] = special_bytes
                self.token_to_id[special_bytes] = next_token_id
                next_token_id += 1

        # 特殊token字符串 -> token ID
        self.special_token_to_id = {
            special_token: self.token_to_id[
                special_token.encode("utf-8")
            ]
            for special_token in self.special_tokens
        }

        # merge pair -> 优先级
        # merges越靠前，优先级越高
        self.merge_ranks = {
            merge: rank
            for rank, merge in enumerate(self.merges)
        }

        # 重叠特殊token必须优先匹配更长的
        #
        # 例如同时存在：
        # <|endoftext|>
        # <|endoftext|><|endoftext|>
        #
        # 必须先尝试匹配第二个
        sorted_special_tokens = sorted(
            dict.fromkeys(self.special_tokens),
            key=len,
            reverse=True,
        )

        if sorted_special_tokens:
            special_pattern = (
                "("
                + "|".join(
                    re.escape(token)
                    for token in sorted_special_tokens
                )
                + ")"
            )

            self.special_token_pattern = re.compile(
                special_pattern
            )
        else:
            self.special_token_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ):
        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)

        with vocab_path.open("rb") as file:
            vocab = pickle.load(file)

        with merges_path.open("rb") as file:
            merges = pickle.load(file)

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def encode(self, text: str) -> list[int]:
        # 第一步：按特殊token切分，同时保留特殊token
        if self.special_token_pattern is None:
            parts = [text]
        else:
            parts = self.special_token_pattern.split(text)

        integer_tokens: list[int] = []

        for part in parts:
            if part == "":
                continue

            # 特殊token直接作为一个完整token处理
            if part in self.special_token_to_id:
                integer_tokens.append(
                    self.special_token_to_id[part]
                )
                continue

            # 普通文本执行正则预分词
            for match in PAT.finditer(part):
                pre_token = match.group()

                current_tokens = [
                    bytes([byte_value])
                    for byte_value
                    in pre_token.encode("utf-8")
                ]

                # 不断寻找当前相邻pair中优先级最高的merge
                while len(current_tokens) >= 2:
                    possible_merges = (
                        (
                            current_tokens[i],
                            current_tokens[i + 1],
                        )
                        for i in range(
                            len(current_tokens) - 1
                        )
                        if (
                            current_tokens[i],
                            current_tokens[i + 1],
                        )
                        in self.merge_ranks
                    )

                    best_merge = min(
                        possible_merges,
                        key=self.merge_ranks.__getitem__,
                        default=None,
                    )

                    # 当前已经不存在可执行的merge
                    if best_merge is None:
                        break

                    current_tokens_new: list[bytes] = []
                    i = 0

                    while i < len(current_tokens):
                        if (
                            i + 1 < len(current_tokens)
                            and current_tokens[i]
                            == best_merge[0]
                            and current_tokens[i + 1]
                            == best_merge[1]
                        ):
                            current_tokens_new.append(
                                current_tokens[i]
                                + current_tokens[i + 1]
                            )
                            i += 2
                        else:
                            current_tokens_new.append(
                                current_tokens[i]
                            )
                            i += 1

                    current_tokens = current_tokens_new

                # 直接通过反向词表查询ID
                integer_tokens.extend(
                    self.token_to_id[token]
                    for token in current_tokens
                )

        return integer_tokens

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return decoded_bytes.decode(
            "utf-8",
            errors="replace",
        )