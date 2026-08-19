import regex as re
import os
from pathlib import Path
import pickle


def pre_tokenizer(
    input_str: str,
)-> list[str]:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    split_str = re.split(PAT, input_str)

    return split_str


def pre_inside_merge(
    pre_token: str, 
    merges_path: str | os.PathLike
) -> list[str]:
    current_tokens = [bytes([b]) for b in pre_token.encode("utf-8")]
    merges_path = Path(merges_path)

    # 这里的 pickle 小知识
    with merges_path.open("rb") as f:
        merges = pickle.load(f)

    for merge in merges:
        current_tokens_new = []
        i = 0
        while i < len(current_tokens) - 1:
            if current_tokens[i] == merge[0] and current_tokens[i+1] == merge[1]:
                current_tokens_new.append(current_tokens[i]+current_tokens[i+1])
                i+=1
            else:
                current_tokens_new.append(current_tokens[i])
        current_tokens = current_tokens_new

    

    

    


