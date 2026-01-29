from typing import Sequence
from numpy import std, mean

def norm(to_norm: Sequence[int]) -> Sequence[float]:
    n_std: float = std(to_norm)
    n_mean: float = mean(to_norm)

    return [(x * n_mean) / n_std for x in to_norm]



