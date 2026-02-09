from struct import pack
from math import inf
from numpy import std, mean
from typing import List, Sequence, Tuple

EPS = 1e-9

def norm(to_norm: Sequence[int]) -> Sequence[float]:
    n_std: float = std(to_norm)
    n_mean: float = mean(to_norm)

    return [(x * n_mean) / (n_std + EPS) for x in to_norm]

def index_max(iter: Sequence[int] | Sequence[float]) -> Tuple[int, int | float]:
    idx_max: int = 0
    maxi: int | float = -inf

    for i in range(len(iter)):
        if iter[i] > maxi:
            maxi = iter[i]
            idx_max = i
    return (idx_max, maxi)

def hash_list(iter: Sequence[float]) -> str:
    format: str = '>' + str(len(iter)) + 'f'
    return pack(format, *iter)