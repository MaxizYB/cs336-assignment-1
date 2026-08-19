import os
import regex as re
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from . import pretokenization_example as pe
from collections import Counter, defaultdict

def worker(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str]
) -> dict[str, int]:
    with open(input_path, "rb") as f:
        f.seek(start)

        chunk = f.read(end - start).decode("utf-8")

    pattern = "|".join(
            re.escape(token)
            for token in special_tokens
        )
    
    parts = re.split(pattern, chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_token_counts: dict[str, int] = {}
    for part in parts:
        # words = re.finditer(PAT, words)
        for match in re.finditer(PAT, part):
            pre_token = match.group()
            if pre_token not in pre_token_counts:
                pre_token_counts[pre_token] = 1
            else:
                pre_token_counts[pre_token] += 1
    return pre_token_counts

def pair_freq_in_word(
    words: list[bytes]
) -> Counter[tuple[bytes, bytes]]:
    ans = Counter()
    for by1, by2 in zip(words[:-1], words[1:]):
        if (by1, by2) not in ans:
            ans[(by1, by2)] = 1
        else :
            ans[(by1, by2)] += 1
    return ans

def bpe(
    input_path: str | os.PathLike,
    vocab_size: int, 
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab_map = {}
    merges: list[tuple[bytes, bytes]] = []
    for i in range(256):
        vocab_map[i] = bytes([i])
    # print(vocab_map)
    for i in range(256, 256+len(special_tokens)):
        vocab_map[i] = special_tokens[i-256].encode("utf-8")

    merge_times = vocab_size - 256 - len(special_tokens)

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = pe.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = []
    with mp.Pool(processes=4) as pool:
        # 首先，上面的这个是什么意思
        for i in range(len(boundaries)-1):
            start = boundaries[i]
            end = boundaries[i+1]

            tasks.append((input_path, start, end, special_tokens))
        results = pool.starmap(worker, tasks)

        pre_token_count = Counter({})

        for result in results:
            pre_token_count.update(Counter(result))

    words = []
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for idx, key in enumerate(pre_token_count.keys()):
        key_bytes = [bytes([byte_val]) for byte_val in key.encode("utf-8")]
        words.append(key_bytes)
        for by1, by2 in zip(key_bytes[:-1], key_bytes[1:]):
            pair_counts[(by1, by2)] += pre_token_count[key]
            if (by1, by2) not in pair_to_words:
                pair_to_words[(by1, by2)] = set([idx])
            else:
                pair_to_words[(by1, by2)].add(idx)

    words_count = [val for val in pre_token_count.values()]

    for i in range(merge_times):
        k, v = max(pair_counts.items(), key= lambda item: (item[1], item[0]))
        merges.append((k[0], k[1]))
        vocab_map[len(vocab_map)] =  k[0] + k[1]

        # pair_to_words_snapshot = pair_to_words.copy()
        for idx in set(pair_to_words[(k[0], k[1])]):
            raw_count = pair_freq_in_word(words[idx])
            new_word = []
            j = 0
            while j < len(words[idx]):
                
                if j < len(words[idx]) - 1 and words[idx][j] == k[0] and words[idx][j+1] == k[1]:
                    j+=2
                    new_word.append(k[0]+k[1])
                    continue
                new_word.append(words[idx][j])
                j+=1

            # for j in range(len(words[idx])-1):
            #     if words[idx][j] == k[0] and words[idx][j+1] == k[1]:
            #         j+=1
            #         new_word.append(k[0]+k[1])
            #         continue
            #     new_word.append(words[idx][j])
            words[idx] = new_word
            new_count = pair_freq_in_word(words[idx])

            
            for key, value in new_count.items():
                need = 0
                if key in raw_count.keys():
                    need = new_count[key] - raw_count[key]
                else:
                    pair_to_words[key].add(idx)
                    need = new_count[key]

                pair_counts[key] += need * words_count[idx]

            for key, value in raw_count.items():
                if key not in new_count.keys():
                    pair_counts[key] -= value * words_count[idx]
                    pair_to_words[key].discard(idx)


            


        # pair_to_words_copy = pair_to_words
        # for idx in pair_to_words_copy[(k[0], k[1])]:
        #     for j in range(len(words[idx]) - 1):
        #         if words[idx][j] == k[0] and words[idx][j+1] == k[1]:
        #             if j > 0:
        #                 pair_counts[(words[idx][j-1], words[idx][j])] -= words_count[idx]
        #                 pair_to_words[(words[idx][j-1], words[idx][j])].discard(idx)
        #                 pair_counts[(words[idx][j-1], k[0]+k[1])] += words_count[idx]
        #                 pair_to_words[(words[idx][j-1], k[0]+k[1])].add(idx)
        #             pair_counts[(words[idx][j], words[idx][j+1])] -= words_count[idx]
        #             pair_to_words[(words[idx][j], words[idx][j+1])].discard(idx)
        #             if j + 2 < len(words[idx]):
        #                 pair_counts[(words[idx][j+1], words[idx][j+2])] -= words_count[idx]
        #                 pair_to_words[(words[idx][j+1], words[idx][j+2])].discard(idx)
        #                 pair_counts[(k[0]+k[1], words[idx][j+2])] += words_count[idx]
        #                 pair_to_words[(k[0]+k[1], words[idx][j+2])].add(idx)
        #             words[idx][j] = k[0]+k[1]
        #             del words[idx][j+1]

    return vocab_map, merges
                        


    
    
    

    
