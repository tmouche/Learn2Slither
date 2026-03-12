from abc import ABC, abstractmethod
from math import exp, pow
from utils.logger import Logger
from utils.types import ArrayF, FloatT
import ml_tools.losses as Losses
import numpy as np

logger = Logger()

class Activation(ABC):

    delta = None
    loss = None

    def __init__(self, activation_function_name: str, loss_function_name: str):
        try:
            self.loss = getattr(Losses, loss_function_name)
            self.delta = getattr(self, ("_" + loss_function_name + "_grad"))
        except AttributeError:
            logger.error(f"{loss_function_name} can not be associated with {activation_function_name}")
            raise Exception()

    @staticmethod
    @abstractmethod
    def activation(z: FloatT | ArrayF) -> FloatT | ArrayF:
        pass

    @staticmethod
    @abstractmethod
    def prime(y: FloatT | ArrayF) -> FloatT | ArrayF:
        pass

class Nothing(Activation):
    def __init__(self, loss_function_name: str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z: FloatT | ArrayF) -> FloatT | ArrayF:
        return z

    @staticmethod
    def prime(y: FloatT | ArrayF) -> FloatT | ArrayF:
        return y
    
    def _mean_square_error_grad(
            self,
            y: FloatT | ArrayF, 
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss = y - e
        prime = self.prime(y)
        return loss * prime
    
    def _mean_absolute_error_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss = abs(y - e)
        prime = self.prime(y)
        return loss * prime
    
    def _binary_cross_entropy_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss = y - e
        return loss
    
    def _categorical_cross_entropy_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss = y - e
        return loss

    def _spare_cross_entropy_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss = y - e
        return loss


class Sigmoid(Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z: FloatT | ArrayF) -> FloatT | ArrayF:
        if isinstance(z, FloatT):
            return 1./(1.+exp(-z))
        z: ArrayF = np.clip(z, -500, 500, dtype=FloatT)
        return np.array([1./(1.+exp(-x)) for x in z], dtype=FloatT)

    @staticmethod
    def prime(y: FloatT | ArrayF) -> FloatT | ArrayF:
        return y*(1.-y)
    
    def _mean_square_error_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = y - e
        prime: FloatT | ArrayF = self.prime(y)
        return loss * prime
    
    def _mean_absolute_error_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = abs(y - e)
        prime: FloatT | ArrayF = self.prime(y)
        return loss * prime
    
    def _binary_cross_entropy_grad(self, y, e):
        loss = y - e
        return loss
    
    def _categorical_cross_entropy_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = y - e
        return loss

    def _spare_cross_entropy_grad(
            self: FloatT | ArrayF,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss = y - e
        return loss


class Tanh(Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z: FloatT | ArrayF) -> FloatT | ArrayF:
        if isinstance(z, FloatT):
            return (exp(z)-exp(-z))/(exp(z)+exp(-z))
        return np.array([(exp(x)-exp(-x))/(exp(x)+exp(-x)) for x in z], dtype=FloatT)

    @staticmethod
    def prime(y: FloatT | ArrayF) -> FloatT | ArrayF:
        if isinstance(y, FloatT):
            return 1-pow(y, 2)
        return np.array([1-pow(x, 2) for x in y], dtype=FloatT)
    
    def _mean_square_error_grad(
            self,
            y: FloatT | ArrayF, 
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = y - e
        prime: FloatT | ArrayF = self.prime(y)
        return loss * prime
    
    def _mean_absolute_error_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = abs(y - e)
        prime: FloatT | ArrayF = self.prime(y)
        return loss * prime 
    


class ReLu(Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z: FloatT | ArrayF) -> FloatT | ArrayF:
        if isinstance(z, FloatT):
            return z if z > 0. else 0.
        return np.array([x if x > 0. else 0. for x in z], dtype=FloatT)

    @staticmethod
    def prime(y: FloatT | ArrayF) -> FloatT | ArrayF:
        if isinstance(y, FloatT):
            return 1. if y > 0. else 0.
        return np.array([1. if x > 0. else 0. for x in y], dtype=FloatT)
    
    def _mean_square_error_grad(
        self,
        y: FloatT | ArrayF,
        e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = y - e
        prime: FloatT | ArrayF = self.prime(y)
        return loss * prime
    
    def _mean_absolute_error_grad(
        self,
        y: FloatT | ArrayF,
        e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = abs(y - e)
        prime: FloatT | ArrayF = self.prime(y)
        return loss * prime



class Leaky_ReLu(Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z: FloatT | ArrayF) -> FloatT | ArrayF:
        if isinstance(z, FloatT):    
            return z if z > 0. else 0.01*z
        return np.array([x if x > 0. else 0.01*x for x in z], dtype=FloatT)

    @staticmethod
    def prime(y: FloatT | ArrayF) -> FloatT | ArrayF:
        if isinstance(y, FloatT):
            return 1. if y > 0 else 0.01
        return np.array([1. if x > 0. else 0.01 for x in y], dtype=FloatT)
    
    def _mean_square_error_grad(
        self,
        y: FloatT | ArrayF,
        e: FloatT | ArrayF) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = y - e
        prime: FloatT | ArrayF = self.prime(y)
        return loss * prime
    
    def _mean_absolute_error_grad(
        self,
        y: FloatT | ArrayF,
        e: FloatT | ArrayF) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = abs(y - e)
        prime: FloatT | ArrayF = self.prime(y)
        return loss * prime
    
    def _spare_cross_entropy_grad(
        self,
        y: FloatT | ArrayF,
        e: FloatT | ArrayF) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = y - e
        return loss


class Softmax(Activation):
    def __init__(self, loss_function_name:str):
        super().__init__(self.__class__.__qualname__, loss_function_name)

    @staticmethod
    def activation(z: FloatT | ArrayF) -> FloatT | ArrayF:
        if isinstance(z, FloatT):
            raise Exception("Error log: Softmax is in/out vector function and doesnot handle scalar")
        sum_exp_z = sum(np.exp(z))
        return np.array([exp(x)/sum_exp_z for x in z], dtype=FloatT)

    @staticmethod
    def prime(y: FloatT | ArrayF) -> FloatT:
        if isinstance(y, FloatT):
            return 1. if y > 0 else 0.01
        return np.array([1. if x > 0. else 0.01 for x in y], dtype=FloatT  )
    
    def _categorical_cross_entropy_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ) -> FloatT | ArrayF:
        loss: FloatT | ArrayF = y - e
        return loss
    
    def _kullback_leibler_divergence_grad(
            self,
            y: FloatT | ArrayF,
            e: FloatT | ArrayF
        ):
        loss: FloatT | ArrayF = abs(y - e)
        return loss
