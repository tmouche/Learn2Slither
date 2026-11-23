from mlp.utils.types import ArrayF, FloatT
from typing import List, Tuple
import math
import numpy as np

SEED = 42
generator = np.random.Generator(np.random.PCG64(SEED))

def nothing(
    shape: Tuple[int, int] = (1,1)
) -> ArrayF | List[ArrayF]:
    return generator.random(size=shape).astype(FloatT)

def random_normal(
        shape: Tuple[int, int] = (1,1),     
        mean: FloatT = 0.0,
        stddev: FloatT = 0.05
    ) -> ArrayF | List[ArrayF]:
    return generator.normal(loc=mean, scale=stddev, size=shape).astype(FloatT)
    
def random_uniform(
        shape: Tuple[int, int] = (1,1),
        minval: FloatT = 0.,
        maxval: FloatT = 1.
    ) -> ArrayF | List[ArrayF]:
    return generator.uniform(low=minval, high=maxval, size=shape).astype(FloatT)
        
def zeros(shape: Tuple[int, int] = (1,1)) -> ArrayF | List[ArrayF]:
    return np.zeros(shape).astype(FloatT)

def ones(shape: Tuple[int, int] = (1,1)) -> ArrayF | List[ArrayF]:
    return np.ones(shape).astype(FloatT)
    
def xavier_normal(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1,
        fan_out: int = 1
    ) -> ArrayF | List[ArrayF]:
    stddev = math.sqrt(2/(fan_in+fan_out))
    return generator.normal(loc=0, scale=stddev, size=shape).astype(FloatT)

def xavier_uniform(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1,
        fan_out: int = 1
    ) -> ArrayF | List[ArrayF]:
    limit = math.sqrt(6/(fan_in+fan_out))
    return generator.uniform(low=-limit, high=limit, size=shape).astype(FloatT)
    
def he_normal(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1
    ) -> ArrayF | List[ArrayF]:
    stddev = math.sqrt(2/fan_in)
    return generator.normal(loc=0, scale=stddev, size=shape).astype(FloatT)

def he_uniform(
        shape: Tuple[int, int] = (1,1),
        fan_in: int = 1
    ) -> ArrayF | List[ArrayF]:
    limit = math.sqrt(6/fan_in)
    return generator.uniform(low=-limit, high=limit, size=shape).astype(FloatT)

