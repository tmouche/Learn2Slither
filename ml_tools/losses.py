from math import pow, log
from utils.types import ArrayF, FloatT
import numpy as np

EPS = 1e-8

def mean_square_error(
        y: FloatT | ArrayF,
        e: FloatT | ArrayF
    ) -> FloatT:
    if isinstance(y, FloatT):
        return pow(e-y, 2)
    return (1/len(y))*sum([pow(e[i]-y[i], 2) for i in range(len(y))])
    
def mean_absolute_error(
        y: FloatT | ArrayF,
        e: FloatT | ArrayF
    ) -> FloatT:
    if isinstance(y, FloatT):
        return abs(e-y)
    return (1/len(y))*sum([abs(e[i]-y[i]) for i in range(len(y))])
    
def binary_cross_entropy(
        y: FloatT | ArrayF,
        e: FloatT | ArrayF
    )-> FloatT:
    if isinstance(y, FloatT):
        return -(e*log(y+EPS)+(1-e)*log(1-y+EPS))
    return -1/len(y)*sum([e[i]*log(y[i]+EPS)+(1-e[i])*log(1-y[i]+EPS) for i in range(len(y))])
    
def spare_cross_entropy(
        y: FloatT | ArrayF,
        e: FloatT | ArrayF
    ) -> FloatT:
    if isinstance(y, FloatT):
        raise Exception("Error log: spare cross entropy doesnot handle scalar")
    return -log(y[np.argmax(e)])

def categorical_cross_entropy(
        y: FloatT | ArrayF,
        e: FloatT | ArrayF
    ) -> FloatT:
    if isinstance(y, FloatT):
        raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
    return -sum([e[i]*log(y[i]+EPS) for i in range(len(y))])
    
def kullback_leibler_divergence(
        y: FloatT | ArrayF,
        e: FloatT | ArrayF
    ) -> FloatT:
    if isinstance(y, FloatT):
        return e*log(e/y+EPS)
    return sum([e[i]*log(e[i]/y[i]+EPS) for i in range(len(y))])

