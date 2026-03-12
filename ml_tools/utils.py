from utils.types import ArrayF, FloatT
import numpy as np

def step(z: FloatT | ArrayF, center: FloatT) -> FloatT | ArrayF:
    if isinstance(z, FloatT):
        return 1. if z > center else 0.
    return np.array([1. if x > center else 0. for x in z], dtype=FloatT)

def accuracy(tp: int, tn: int, fp: int, fn: int) -> FloatT:
    denominator: int = tp + tn + fp + fn
    if not denominator:
        return 0.
    return (tp + tn) / denominator

def precision(tp: int, fp: int) -> FloatT:
    denominator: int = tp + fp
    if not denominator:
        return 0.
    return tp / denominator

def recall(tp: int, fn: int) -> FloatT:
    denominator: int = tp + fn
    if not denominator:
        return 0.
    return tp / denominator

def f1(precision: FloatT, recall: FloatT) -> FloatT:
    denominator: int = precision + recall
    if not denominator:
        return 0.
    return 2 * ((precision * recall) / denominator)

