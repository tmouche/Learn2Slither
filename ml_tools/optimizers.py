from abc import ABC, abstractmethod
from core.network import Network
from math import floor
from ml_tools.fire import Fire
from typing import Dict, List
from utils.logger import Logger
from utils.types import ArrayF, FloatT
import numpy as np

logger = Logger()

EPS: FloatT = 1e-8

class Optimizer(ABC):

    in_use_weights: List[List[ArrayF]]
    in_use_biaises: List[ArrayF]

    fire: Fire
    net: Network

    accuracies: ArrayF
    losses: ArrayF

    def __init__(
            self,
            fire: Fire,
            network: Network
        ):
        self.accuracy = 0.
        self.loss = 0.

        self.fire = fire
        self.net = network
        self.in_use_weights = network.weights
        self.in_use_biaises = network.biaises

    def full(self, dataset: List[Dict[str, ArrayF]]):
        """
        Train and evaluate the network on the entire dataset at once.
        Args:
            dataset (List[Dict[str, ArrayF]]): Complete dataset as a list of input-output dictionaries.
        Returns:
            Dict: Computed performance metrics after training.
        Logs:
            - None explicitly; metrics are computed internally.
        Notes:
            - Resets network gradients before training.
            - Calls `_learn` on the full dataset as a single batch.
            - Returns metrics via `_metric`.
        """
        self._reset()
        self._learn([dataset], len(dataset))
        return self._metric()

    def deterministic(self, dataset: List[Dict[str, ArrayF]]) -> Dict[str, FloatT]:
        """
        Train and evaluate the network using deterministic batching (sequential order).
        Args:
            dataset (List[Dict[str, ArrayF]]): Dataset as a list of input-output dictionaries.
        Returns:
            Dict: Computed performance metrics after training.
        Logs:
            - None explicitly; metrics are computed internally.
        Notes:
            - Resets network gradients before training.
            - Splits the dataset into sequential batches using `_prepare_batch`.
            - Calls `_learn` on each batch sequentially.
            - Returns metrics via `_metric`.
        """
        self._reset()
        batch: List[List[Dict[str, ArrayF]]] = self._prepare_batch(dataset, self.net.batch_size)
        self._learn(batch, self.net.batch_size)
        return self._metric()

    def stochastic(self, dataset: List[Dict[str, ArrayF]]) -> Dict:
        """
        Train and evaluate the network using stochastic batching (randomized order).
        Args:
            dataset (List[Dict[str, ArrayF]]): Dataset as a list of input-output dictionaries.
        Returns:
            Dict: Computed performance metrics after training.
        Logs:
            - None explicitly; metrics are computed internally.
        Notes:
            - Resets network gradients before training.
            - Randomizes dataset order and splits into batches using `_prepare_batch`.
            - Calls `_learn` on each batch sequentially.
            - Returns metrics via `_metric`.
        """
        self._reset()
        random_dataset: List[Dict[str, ArrayF]] = dataset.copy() 
        np.random.shuffle(random_dataset)
        batch: List[List[Dict[str, ArrayF]]] = self._prepare_batch(random_dataset, self.net.batch_size)
        self._learn(batch, self.net.batch_size)
        return self._metric()

    def _learn(self, batch: List[List[Dict[str, ArrayF]]], batch_size: int):
        e_accuracies: FloatT = 0.
        e_losses: FloatT = 0.
        for i in range(len(batch)):
            t_acc, t_loss = self.fire.backward(list(batch[i]) if self.net.batch_size == 1 else batch[i], self.in_use_weights, self.in_use_biaises)
            e_accuracies += t_acc
            e_losses += t_loss
            self._update(batch_size)
            self.fire._reset()
        self.accuracy = e_accuracies / len(batch)
        self.loss = e_losses / len(batch)
    
    def _prepare_batch(
            self,
            dataset: List[Dict[str, ArrayF]],
            batch_size: int
        ) -> List[Dict[str, ArrayF]]:
        new_ds_len: int = floor(len(dataset) / batch_size) * batch_size
        return [[dataset[i] for i in range(j, j + batch_size)] for j in range(0, new_ds_len, batch_size)]

    def _metric(self) -> Dict[str, List[FloatT]]:
        return dict(accuracy=self.accuracy, loss=self.loss)

    @abstractmethod
    def _update(self, batch_size: int):
        pass

    @abstractmethod
    def _reset(self):
        pass


class Gradient_Descent(Optimizer):
    """
    Gradient Descent optimizer for updating network weights and biases.

    Args:
        fire (Fire): Object providing gradients (nabla_w, nabla_b) for the network.
        network (Network): Neural network instance to optimize.
    """
    def __init__(
            self,
            fire: Fire,
            network: Network
        ):
            Optimizer.__init__(self, fire=fire, network=network)

    def _update(self, batch_size: int):
        for i in range(len(self.net.weights)):
            self.net.weights[i] -= (self.net.learning_rate * (self.fire.nabla_w[i] / batch_size))
            self.net.biaises[i] -= (self.net.learning_rate * (self.fire.nabla_b[i] / batch_size))

    def _reset(self):
        pass


class RMS_Propagation(Optimizer):
    """
    RMSProp optimizer: updates network parameters using adaptive learning rates 
    based on a moving average of squared gradients.

    Args:
        fire (Fire): Object providing gradients (nabla_w, nabla_b) for the network.
        network (Network): Neural network instance to optimize.
        velocity_rate (FloatT): Decay rate for velocity computation.
    """
    velocity_rate: FloatT

    velocity_w: List[List[ArrayF]]
    velocity_b: List[ArrayF]

    def __init__(
            self,
            fire: Fire,
            network: Network,
            velocity_rate: FloatT
        ):
            Optimizer.__init__(self, fire=fire, network=network)
            self.velocity_rate = velocity_rate
            self.velocity_w = list(np.full((len(w),len(w[0])) , 0., dtype=FloatT) for w in self.net.weights)
            self.velocity_b = list(np.full(len(w), 0., dtype=FloatT) for w in self.net.weights)

    def _update(self, batch_size:int):
        for i in range(len(self.net.weights)):
            self._update_velocity(i, batch_size)
            self._update_parameters(i, batch_size)

    def _update_velocity(self, index: int, batch_size:int):
        self.velocity_w[index] = self.velocity_rate * self.velocity_w[index] + (1 - self.velocity_rate)*(np.power(self.fire.nabla_w[index]/batch_size, 2))
        self.velocity_b[index] = self.velocity_rate * self.velocity_b[index] + (1 - self.velocity_rate)*(np.power(self.fire.nabla_b[index]/batch_size, 2))

    def _update_parameters(self, index: int, batch_size:int):
        self.net.weights[index] -= (self.net.learning_rate / (np.sqrt(self.velocity_w[index]) + EPS)) * (self.fire.nabla_w[index] / batch_size)
        self.net.biaises[index] -= (self.net.learning_rate / (np.sqrt(self.velocity_b[index]) + EPS)) * (self.fire.nabla_b[index] / batch_size)

    def _reset(self):
        pass


class Nesterov_Accelerated_Gradient(Optimizer):
    """
    Nesterov Accelerated Gradient optimizer: applies momentum with lookahead updates
    to accelerate convergence.

    Args:
        fire (Fire): Object providing gradients (nabla_w, nabla_b) for the network.
        network (Network): Neural network instance to optimize.
        momentum_rate (FloatT): Momentum decay rate for velocity updates.
    """
    momentum_rate: FloatT

    momentum_w: List[List[ArrayF]]
    momentum_b: List[ArrayF]

    ahead_w: List[List[ArrayF]]
    ahead_b: List[ArrayF]

    __r_momentum_w: List[List[ArrayF]]
    __r_momentum_b: List[ArrayF]

    def __init__(
            self,
            fire: Fire,
            network: Network,
            momentum_rate: FloatT
    ):
        Optimizer.__init__(self, fire=fire, network=network)
        self.momentum_rate = momentum_rate
        self.__r_momentum_w = list(np.full((len(w),len(w[0])) , 0., dtype=FloatT) for w in self.net.weights)
        self.__r_momentum_b = list(np.full(len(w), 0., dtype=FloatT) for w in self.net.weights)
        self._reset()
        self.ahead_w = self.net.weights.copy()
        self.ahead_b = self.net.biaises.copy()
        self.in_use_weights = self.ahead_w
        self.in_use_biaises = self.ahead_b

    def _reset(self):
        self.momentum_w = self.__r_momentum_w.copy()
        self.momentum_b = self.__r_momentum_b.copy()

    def _update(self, batch_size: int):
        for i in range(len(self.net.weights)):
            self._update_momentum(i, batch_size)
            self._update_parameters(i)
            self._update_ahead_parameters(i)
    
    def _update_momentum(self, index: int, batch_size:int):
        self.momentum_w[index] = self.momentum_rate * self.momentum_w[index] + (self.fire.nabla_w[index] / batch_size)
        self.momentum_b[index] = self.momentum_rate * self.momentum_b[index] + (self.fire.nabla_b[index] / batch_size) 

    def _update_parameters(self, index: int):
        self.net.weights[index] = self.net.weights[index] - (self.net.learning_rate * self.momentum_w[index])
        self.net.biaises[index] = self.net.biaises[index] - (self.net.learning_rate * self.momentum_b[index])

    def _update_ahead_parameters(self, index: int):
        self.ahead_w[index] = self.net.weights[index] - (self.momentum_rate * self.net.learning_rate * self.momentum_w[index])
        self.ahead_b[index] = self.net.biaises[index] - (self.momentum_rate * self.net.learning_rate * self.momentum_b[index])


class ADAM(Optimizer):
    """
    Adam optimizer: combines momentum and RMSProp to adapt learning rates
    for each parameter individually.

    Args:
        fire (Fire): Object providing gradients (nabla_w, nabla_b) for the network.
        network (Network): Neural network instance to optimize.
        momentum_rate (FloatT): Momentum decay rate.
        velocity_rate (FloatT): Decay rate for RMSProp-style velocity.
    """
    velocity_rate: FloatT
    momentum_rate: FloatT

    velocity_w: List[List[ArrayF]]
    velocity_b: List[ArrayF]
    momentum_w: List[List[ArrayF]]
    momentum_b: List[ArrayF]

    __r_momentum_w: List[List[ArrayF]]
    __r_momentum_b: List[ArrayF]

    def __init__(
        self,
        fire: Fire,
        network: Network,
        momentum_rate: FloatT,
        velocity_rate: FloatT
    ):
        Optimizer.__init__(self, fire=fire, network=network)
        self.momentum_rate = momentum_rate
        self.velocity_rate = velocity_rate
        self.velocity_w = list(np.full((len(w),len(w[0])) , 0., dtype=FloatT) for w in self.net.weights)
        self.velocity_b = list(np.full(len(w), 0., dtype=FloatT) for w in self.net.weights)
        self.__r_momentum_w = list(np.full((len(w),len(w[0])) , 0., dtype=FloatT) for w in self.net.weights)
        self.__r_momentum_b = list(np.full(len(w), 0., dtype=FloatT) for w in self.net.weights)
        self._reset()

    def _reset(self):
        self.momentum_w = self.__r_momentum_w.copy()
        self.momentum_b = self.__r_momentum_b.copy()

    def _update(self, batch_size: int):
        for i in range(len(self.net.weights)):
            self._update_momentum(i, batch_size)
            self._update_velocity(i, batch_size)
            self._update_parameters(i, batch_size)

    def _update_momentum(self, index: int, batch_size: int):
        self.momentum_w[index] = self.momentum_rate * self.momentum_w[index] + (1 - self.momentum_rate) * (self.fire.nabla_w[index] / batch_size)
        self.momentum_b[index] = self.momentum_rate * self.momentum_b[index] + (1 - self.momentum_rate) * (self.fire.nabla_b[index] / batch_size) 

    def _update_velocity(self, index: int, batch_size:int):
        self.velocity_w[index] = self.velocity_rate * self.velocity_w[index] + (1 - self.velocity_rate)*(np.power(self.fire.nabla_w[index]/batch_size, 2))
        self.velocity_b[index] = self.velocity_rate * self.velocity_b[index] + (1 - self.velocity_rate)*(np.power(self.fire.nabla_b[index]/batch_size, 2))

    def _update_parameters(self, index: int, batch_size: int):
        momentum_corr_w: List[ArrayF] = self.momentum_w[index] / (1 - self.momentum_rate ** batch_size)
        momentum_corr_b: List[ArrayF] = self.momentum_b[index] / (1 - self.momentum_rate ** batch_size)
        velocity_corr_w: List[ArrayF] = self.velocity_w[index] / (1 - self.velocity_rate ** batch_size)
        velocity_corr_b: List[ArrayF] = self.velocity_b[index] / (1 - self.velocity_rate ** batch_size)

        self.net.weights[index] -= momentum_corr_w/(np.sqrt(velocity_corr_w + EPS)) * self.net.learning_rate
        self.net.biaises[index] -= momentum_corr_b/(np.sqrt(velocity_corr_b + EPS)) * self.net.learning_rate


class Adapative_Ahead_Momentum(RMS_Propagation, Nesterov_Accelerated_Gradient):
    """
    Hybrid optimizer combining RMSProp and Nesterov Accelerated Gradient:
    applies adaptive learning rates with momentum and lookahead updates.

    Args:
        fire (Fire): Object providing gradients (nabla_w, nabla_b) for the network.
        network (Network): Neural network instance to optimize.
        momentum_rate (FloatT): Momentum decay rate.
        velocity_rate (FloatT): Decay rate for RMSProp-style velocity.
    """
    def __init__(
            self,
            fire: Fire,
            network: Network,
            momentum_rate: FloatT,
            velocity_rate: FloatT
    ):
        RMS_Propagation.__init__(
            self,
            fire=fire,
            network=network,
            velocity_rate=velocity_rate
        )
        Nesterov_Accelerated_Gradient.__init__(
            self,
            fire=fire,
            network=network,
            momentum_rate=momentum_rate
        )

    def _reset(self):
        RMS_Propagation._reset(self)
        Nesterov_Accelerated_Gradient._reset(self)

    def _update(self, batch_size: int):
        for i in range(len(self.net.weights)):
            self._update_momentum(i, batch_size)
            self._update_velocity(i, batch_size)
            self._update_parameters(i)
            self._update_ahead_parameters(i)

    def _update_parameters(self, index: int):
        self.net.weights[index] -= self.momentum_w[index]/(np.sqrt(self.velocity_w[index] + EPS)) * self.net.learning_rate
        self.net.biaises[index] -= self.momentum_b[index]/(np.sqrt(self.velocity_b[index] + EPS)) * self.net.learning_rate
    