from .bpe import bpe
from pathlib import Path
import pickle

if __name__ == "__main__":
    root_path = Path("./")
    data_path = root_path / "data" 
    input_path = data_path / "TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    output_dir = root_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = 10000

    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"

    vocab, merges = bpe(input_path, vocab_size, special_tokens)

    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    with merges_path.open("wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"vocab saved to: {vocab_path.resolve()}")
    print(f"merges saved to: {merges_path.resolve()}")


