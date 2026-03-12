from typing import List, Tuple
from utils.types import ArrayF, FloatT
import math
import numpy as np

SEED = 42

def nothing(
    shape: Tuple[int, int] = (1,1)
) -> ArrayF | List[ArrayF]:
    if SEED: np.random.seed(SEED)
    return np.random(size=shape).astype(FloatT)

def random_normal(
        shape: Tuple[int, int] = (1,1),     
        mean: FloatT = 0.0,
        stddev: FloatT = 0.05
    ) -> ArrayF | List[ArrayF]:
    if SEED: np.random.seed(SEED)
    return np.random.normal(loc=mean, scale=stddev, size=shape).astype(FloatT)
    
def random_uniform(
        shape: Tuple[int, int] = (1,1),
        minval: FloatT = 0.,
        maxval: FloatT = 1.
    ) -> ArrayF | List[ArrayF]:
    if SEED: np.random.seed(SEED)
    return np.random.uniform(low=minval, high=maxval, size=shape).astype(FloatT)
        
def zeros(shape: Tuple[int, int] = (1,1)) -> ArrayF | List[ArrayF]:
    if SEED: np.random.seed(SEED)
    return np.zeros(shape).astype(FloatT)

def ones(shape: Tuple[int, int] = (1,1)) -> ArrayF | List[ArrayF]:
    if SEED: np.random.seed(SEED)
    return np.ones(shape).astype(FloatT)
    
def xavier_normal(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1,
        fan_out: int = 1
    ) -> ArrayF | List[ArrayF]:
    if SEED: np.random.seed(SEED)
    stddev = math.sqrt(2/(fan_in+fan_out))
    return np.random.normal(loc=0, scale=stddev, size=shape).astype(FloatT)

def xavier_uniform(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1,
        fan_out: int = 1
    ) -> ArrayF | List[ArrayF]:
    if SEED: np.random.seed(SEED)
    limit = math.sqrt(6/(fan_in+fan_out))
    return np.random.uniform(low=-limit, high=limit, size=shape).astype(FloatT)
    
def he_normal(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1
    ) -> ArrayF | List[ArrayF]:
    if SEED: np.random.seed(SEED)
    stddev = math.sqrt(2/fan_in)
    return np.random.normal(loc=0, scale=stddev, size=shape).astype(FloatT)

def he_uniform(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1
    ) -> ArrayF | List[ArrayF]:
    if SEED: np.random.seed(SEED)
    limit = math.sqrt(6/fan_in)
    return np.random.uniform(low=-limit, high=limit, size=shape).astype(FloatT)

