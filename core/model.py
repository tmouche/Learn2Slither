from core.layer import Layer
from core.network import Network
from ml_tools.evaluations import binary_classification
from ml_tools.fire import Fire
from time import perf_counter
from typing import Callable, Dict, List
from utils.constant import POSITIV
from utils.exception import (
    ModelActivation,
    ModelLoss,
    ModelNetCreate,
    NetworkException,
    UnexpectedException,
)
from utils.history import save_to_history
from utils.logger import Logger
from utils.types import ArrayF, FloatT
import json
import ml_tools.activations as Activation
import ml_tools.losses as Loss
import numpy as np
import pickle


logger = Logger()

class Model:

    accuracies: Dict[str, ArrayF]
    losses: Dict[str, ArrayF]

    fire: Fire
    network: Network

    def __init__(self):
        
        self.accuracies = dict(testing=np.ndarray(0), training=np.ndarray(0))
        self.losses = dict(testing=np.ndarray(0), training=np.ndarray(0))

        self.fire = None
        self.network = None


    def create_network(
        self,
        layers: List[Layer],
        learning_rate: FloatT,
        batch_size: int
    ):
        """
        Create and initialize a new neural network with specified layers, learning rate, and batch size.

        Args:
            layers (List[Layer]): List of layer objects defining the network architecture.
            learning_rate (FloatT): Learning rate for the network optimizer.
            batch_size (int): Number of samples per training batch.

        Raises:
            ModelNetCreate: If network creation fails due to a `NetworkException`.

        Logs:
            - Error if `NetworkException` is raised during network initialization.
        """
        try:
            new_net: Network = Network(
                layers=layers,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
        except NetworkException as netErr:
            logger.error(netErr)
            raise ModelNetCreate(netErr)
        self.network = new_net
        self.fire = Fire(self.network.layers) 


    def load_network(self, save_folder: str) -> Network:
        with open(f"{save_folder}/settings.json", 'r') as fs:
            settings: Dict[str, str | List[Dict[str, str]]] = json.loads(fs.read())
            layers: List[Layer] = []
            for l in settings.get("layers"):
                layer = Layer(
                    l.get("shape"),
                    l.get("output_activation"),
                    l.get("parameters_initializer")
                )
                layers.append(layer)
            learning_rate: FloatT = FloatT(settings.get("learning_rate"))
            batch_size: int = int(settings.get("batch_size"))
            self.create_network(layers, learning_rate, batch_size)
            self.load_layers(settings.get("loss_function"))
        with open(f"{save_folder}/weights.pkl", 'rb') as fw:
            self.network.weights = pickle.load(fw)
        with open(f"{save_folder}/biaises.pkl", 'rb') as fb:
            self.network.biaises = pickle.load(fb)


    def fit(
        self,
        optimizer: Callable,
        ds_train: List[Dict[str, ArrayF]],
        ds_test: List[Dict[str, ArrayF]],
        loss: str,
        epochs: int,
        early_stoper: FloatT = 0.,
        print_training_state: bool = True,
        history_save: bool = False,
    ):
        """
        Train the neural network on the provided training dataset and evaluate on the test dataset.

        Args:
            optimizer (Callable): Function performing optimization for one training step.
            ds_train (List[Dict[str, ArrayF]]): Training dataset as a list of input-output dictionaries.
            ds_test (List[Dict[str, ArrayF]]): Testing dataset for validation.
            loss (str): Name of the loss function to use.
            epochs (int): Maximum number of training epochs.
            early_stoper (FloatT, optional): Threshold for early stopping based on test loss. Defaults to 0.
            print_training_state (bool, optional): If True, prints progress at each epoch. Defaults to True.
            history_save (bool, optional): If True, saves training history after completion. Defaults to False.

        Raises:
            UnexpectedException: If an unexpected exception occurs during training.
            KeyboardInterrupt: If training is interrupted manually.

        Logs:
            - Info when training starts.
            - Info if training is stopped manually.
            - Error if an unexpected exception occurs during training.

        Notes:
            - Tracks training and testing accuracy and loss over epochs.
            - Supports early stopping based on `early_stoper`.
            - Optionally saves detailed training history including metrics, network configuration, and timing.
        """

        self.load_layers(loss)
        
        logger.info("Starting training...")
        start_time: FloatT = perf_counter()
        max_epoch: int = 0
        for i_epoch in range(epochs):
            try:
                training: Dict[str, FloatT] = optimizer(ds_train)
                testing: Dict[str, FloatT] = binary_classification(self.network, self.network.layers[-1].activation.loss, ds_test, POSITIV)

                self.accuracies["testing"] = np.append(self.accuracies.get("testing"), testing.get("accuracy"))
                self.losses["testing"] = np.append(self.losses.get("testing"), testing.get("loss"))
                self.accuracies["training"] = np.append(self.accuracies.get("training"), training.get("accuracy"))
                self.losses["training"] = np.append(self.losses.get("training"), training.get("loss"))

                if print_training_state:
                    Model._print_epoch_state(
                        self,
                        act_epoch=i_epoch,
                        epoch=epochs,
                        training_accuracy=training.get("accuracy"),
                        training_loss=training.get("loss"),
                        testing_accuracy= testing.get("accuracy"),
                        testing_loss= testing.get("loss"),
                    )
                if testing.get("loss") < early_stoper:
                    max_epoch = i_epoch
                    break
            except KeyboardInterrupt:
                max_epoch = i_epoch
                logger.info("Trainning Stopped")
                break
            except Exception as e:
                logger.error(f"Unexpected exception: {e}")
                raise UnexpectedException()
        time_stamp: FloatT = perf_counter() - start_time
        if not max_epoch: max_epoch = epochs
        if history_save:
            save_to_history(
                optimizer=f"{optimizer.__self__.__class__.__qualname__}.{optimizer.__name__}",
                activation_function=self.network.layers[-1].output_activation,
                loss_function=loss,
                epoch=max_epoch,
                learning_rate=self.network.learning_rate,
                network_shape=[self.network.layers[i].shape for i in range(len(self.network.layers))],
                batch_size=self.network.batch_size,
                min_training_loss=min(self.losses.get('training')),
                min_testing_loss=min(self.losses.get('testing')),
                accuracy=testing.get('accuracy'),
                precision=testing.get('precision'),
                recall=testing.get('recall'),
                f1=testing.get('f1'),
                time=time_stamp
            )
        
    def get_loss(func_name: str) -> Callable:
        try:
            return getattr(Loss, func_name)
        except AttributeError as attrErr:
            logger.error(f"Loss {func_name} not found")
            raise ModelLoss(attrErr)
        except Exception as e:
            logger.error(f"Unexpected exception: {e}")
            raise UnexpectedException()

    def load_layers(self, loss: str):
        self.network.loss_function = loss
        for i in range(1, len(self.network.layers)):
            try:
                self.network.layers[i].activation = getattr(Activation, self.network.layers[i].output_activation)(loss)
            except AttributeError:
                logger.error(f"Activation {self.network.layers[i].output_activation} not found")
                raise ModelActivation(self.network.layers[i].output_activation)
            except Exception as e:
                logger.error(f"Unexpected exception: {e}")
                raise UnexpectedException()

    def _print_epoch_state(
            self,
            act_epoch:int,
            epoch:int,
            training_accuracy:float,
            training_loss:float,
            testing_accuracy:float,
            testing_loss:float,
    ):
        message = (
            "\n===============================\n"
            f"At epoch {act_epoch}/{epoch}\n"
            "=== Training ===\n"
            f"Accuracy: {training_accuracy},\n"
            f"Loss: {training_loss},\n"
            "=== Testing ===\n"
            f"Accuracy: {testing_accuracy},\n"
            f"Loss: {testing_loss},\n"
            "==============================\n"
        )
        logger.info(message)