from struct import pack
from math import inf
from numpy import std, mean
from typing import Dict, List, Sequence, Tuple
from utils.exception import CriticalException, FolderCreationError
from utils.logger import logger
import os

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
    form: str = '>' + str(len(iter)) + 'f'
    my_hash: str = str(pack(form, *iter))
    return my_hash

def create_folder(path_to_folder: str, on_error: bool = False):
    try:
        os.mkdir(path_to_folder)
    except FileExistsError as err:
        if on_error:
            raise err
    except Exception as exc:
        logger.error(f'An unexpected error happened on creating folder: {exc}')
        raise FolderCreationError(path_to_folder)

def prep_to_dict(
    list_1: List,
    list_2: List,
    name_1: str,
    name_2: str,
) -> Dict[str, List]:
    res: Dict[str, List] = {
        "epoch": [],
        "discriminant": [],
        "data": [],
    }
    size: int = len(list_1)
    if size != len(list_2):
        logger.error("Mismatch between to list before ploting")
        raise CriticalException("Mismatch len")
    for i in range(size):
        res["epoch"].append(i)
        res["epoch"].append(i)
        res["discriminant"].append(name_1)
        res["discriminant"].append(name_2)
        res["data"].append(list_1[i])
        res["data"].append(list_2[i])
    return res