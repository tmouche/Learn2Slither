from core.layer import Layer
from ml_tools.utils import step
from typing import Dict, List
from utils.logger import Logger
from utils.types import ArrayF, FloatT
import numpy as np

from time import perf_counter

logger = Logger()

class Fire:

    accuracies: ArrayF
    losses: ArrayF
    
    layers: List[Layer]

    nabla_w: List[List[ArrayF]]
    nabla_b: List[ArrayF]

    __r_nabla_w: List[List[ArrayF]]
    __r_nabla_b: List[ArrayF]

    def __init__(
            self,
            layers: List[Layer]
        ):
        self.accuracies = np.ndarray(0)
        self.losses = np.ndarray(0)

        self.layers = layers
        
        self.__r_nabla_w = list()
        self.__r_nabla_b = list()

        for i in range(1, len(self.layers)):
            previous_size: int = self.layers[i-1].shape
            size: int = self.layers[i].shape
            self.__r_nabla_w.append(list(np.full((size,previous_size) , 0., dtype=FloatT)))
            self.__r_nabla_b.append(list(np.full(size, 0., dtype=FloatT)))

        self._reset()


    def backward(
            self,
            dataset: List[Dict[str, ArrayF]],
            weights: List[List[ArrayF]],
            biaises: List[ArrayF]
        ) -> tuple[FloatT, FloatT]:
        """
        Perform a backward pass (backpropagation) to compute gradients for weights and biases.
        Args:
            dataset (List[Dict[str, ArrayF]]): Dataset as a list of input-output dictionaries.
            weights (List[List[ArrayF]]): Current weights of the network layers.
            biaises (List[ArrayF]): Current biases of the network layers.
        Updates:
            - self.nabla_w: Accumulates gradients of weights over the dataset.
            - self.nabla_b: Accumulates gradients of biases over the dataset.
            - self.accuracies: Appends mean accuracy over the dataset.
            - self.losses: Appends mean loss over the dataset.
        Logs:
            - None explicitly, but tracks per-sample loss and accuracy for statistics.
        Notes:
            - Computes the delta for the output layer using the activation's `delta` function.
            - Propagates the delta backward through each layer using the chain rule.
            - Uses `np.outer` to compute weight gradients.
            - Step function with threshold 0.5 is used for accuracy calculation.
            - Mean loss and accuracy over the dataset are appended to `self.losses` and `self.accuracies`.
        """
        ds_size: int = len(dataset)
        e_accuracies: FloatT = 0.
        e_losses: FloatT = 0.

        for i in range(ds_size):
            out: List[ArrayF] = self.forward(dataset[i].get("data"), weights, biaises)
            e_losses += self.layers[-1].activation.loss(out[-1], dataset[i].get("label"))
            e_accuracies += 1 if (step(out[-1], 0.5) == dataset[i].get("label")).all() else 0
            delta: ArrayF = self.layers[-1].activation.delta(out[-1], dataset[i].get("label"))
            self.nabla_w[-1] += np.outer(delta, out[-2])
            self.nabla_b[-1] += delta
            idx: int = len(weights)-2
            while idx >= 0:
                prime: ArrayF = self.layers[idx+1].activation.prime(out[idx+1])
                delta: ArrayF = np.dot(np.transpose(weights[idx+1]), delta) * prime
                self.nabla_w[idx] += np.outer(np.array(delta), np.array(out[idx]))
                self.nabla_b[idx] += delta
                idx-=1

        return e_accuracies / ds_size, e_losses / ds_size
    

    def forward(
            self,
            input: ArrayF,
            weights: List[List[ArrayF]],
            biaises: List[ArrayF]
        ) -> List[ArrayF]:
        """
        Perform a forward pass through the network and compute outputs for all layers.
        Args:
            input (ArrayF): Input data array for the network.
            weights (List[List[ArrayF]]): Current weights of the network layers.
            biaises (List[ArrayF]): Current biases of the network layers.
        Returns:
            List[ArrayF]: List of outputs for each layer, including the input as the first element.
        Logs:
            - None explicitly, purely computes forward activations.
        Notes:
            - Applies each layer's activation function after linear combination of inputs and weights.
            - The final layer's activation is applied in the same manner.
            - The output list preserves the sequential outputs for use in backpropagation.
        """
        out: List[ArrayF] = [input]
        for i in range(len(weights) - 1):
            out.append(self.layers[i + 1].activation.activation(self.layer(out[-1], weights[i], biaises[i])))
        out.append(self.layers[-1].activation.activation(self.layer(out[-1], weights[-1], biaises[-1])))
        return out


    def layer(
            self,
            input: ArrayF,
            weights: List[ArrayF],
            biaises: ArrayF
        ) -> ArrayF:
        """
        Compute the linear transformation of the input for a single network layer.
        Args:
            input (ArrayF): Input array to the layer.
            weights (List[ArrayF]): Weight vectors for the layer neurons.
            biaises (ArrayF): Bias values for the layer neurons.
        Returns:
            ArrayF: Resulting array after applying the linear transformation (dot product plus bias) for each neuron.
        Logs:
            - None explicitly, purely performs computation.
        Notes:
            - Each neuron's output is computed as `dot(weight, input) + bias`.
            - The function returns an array containing all neuron outputs for the layer.
        """
        res: ArrayF = np.ndarray(0)
        for w, b in zip(weights, biaises):
            res = np.append(res, np.dot(w, input) + b)
        return res


    def full(
            self,
            input: ArrayF,
            weights: List[List[ArrayF]],
            biaises: List[ArrayF]
        ) -> ArrayF:
        """
        Compute the network's output for a given input using a full forward pass.      
        Args:
            input (ArrayF): Input array for the network.
            weights (List[List[ArrayF]]): Current weights of the network layers.
            biaises (List[ArrayF]): Current biases of the network layers.      
        Returns:
            ArrayF: Output of the final layer after the forward pass.      
        Logs:
            - None explicitly; purely returns the final output of the network.     
        Notes:
            - Internally calls `self.forward` and returns only the last layer's output.
        """
        return self.forward(input, weights, biaises)[-1]
    
    def _reset(self):
        self.nabla_w = self.__r_nabla_w.copy()
        self.nabla_b = self.__r_nabla_b.copy()