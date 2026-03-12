from core.network import Network
from ml_tools.utils import (
    step,
    accuracy,
    precision,
    recall,
    f1
)
from typing import (
    Callable, 
    Dict,
    List,
)
from utils.logger import Logger
from utils.types import ArrayF, FloatT

logger = Logger()

def binary_classification(
    network: Network,
    loss_fnc: Callable,
    ds_test: List[Dict[str, ArrayF]],
    positiv: List[int]
) -> Dict[str, FloatT]:
    """
    Evaluate a network's performance on a binary classification test dataset.

    Args:
        network (Network): Neural network object containing layers, weights, and biases.
        loss_fnc (Callable): Loss function to compute per-sample loss.
        ds_test (List[Dict[str, ArrayF]]): Test dataset as a list of input-output dictionaries.
        positiv (List[int]): Labels considered as the positive class.

    Returns:
        Dict[str, FloatT]: Dictionary containing the following metrics:
            - accuracy: Overall classification accuracy.
            - loss: Average loss over the test dataset.
            - precision: Precision score for the positive class.
            - recall: Recall score for the positive class.
            - f1: F1 score for the positive class.

    Logs:
        - None explicitly, but metrics are computed for evaluation purposes.

    Notes:
        - Uses a threshold of 0.5 on the network output for binary decisions.
        - Counts true positives (tp), true negatives (tn), false positives (fp), and false negatives (fn)
          to compute classification metrics.
        - Average loss is computed over all samples in `ds_test`.
    """
    losses:List[FloatT] = list()
    tp = tn = fp = fn = 0

    for t in ds_test:
        output: ArrayF = network.fire.full(t.get("data"), network.weights, network.biaises)
        label: List[int] = list(t.get("label"))
        if (step(output, 0.5) == positiv).all():
            if label == positiv:
                tp += 1 
            else:
                fp += 1
        else:
            if label == positiv:
                fn += 1
            else:
                tn += 1
        losses.append(loss_fnc(output, label))
    value_precision: FloatT = precision(tp, fp)
    value_recall: FloatT = recall(tp, fn)
    value_f1: FloatT = f1(value_precision, value_recall)
    average_loss: FloatT = sum(losses) / len(losses)
    return {
        "accuracy": accuracy(tp, tn, fp, fn),
        "loss": average_loss,
        "precision": value_precision,
        "recall": value_recall,
        "f1": value_f1
    }


