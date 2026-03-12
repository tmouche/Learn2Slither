from typing import List, Dict
from utils.logger import Logger
import pandas
import os

logger = Logger()

COLUMNS = [
    "optimizer",
    "activation function",
    "loss function",
    "epoch",
    "learning rate",
    "network shape",
    "min training loss",
    "min testing loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "time"
]

FILE = "results/history.csv"

def save_to_history(
        optimizer:str=None,
        activation_function:str=None,
        loss_function:str=None,
        epoch:int=None,
        learning_rate:float=None,
        network_shape:List=None,
        batch_size:int=None,
        min_training_loss:float=None,
        min_testing_loss:float=None,
        accuracy:float=None,
        precision:float=None,
        recall:float=None,
        f1:float=None,
        time:float=None
):
    
    try:
        if os.path.exists(FILE):
            df:pandas.DataFrame = pandas.read_csv(FILE)
        else:
            df:pandas.DataFrame = pandas.DataFrame(columns=COLUMNS)

        data:Dict = {
                "optimizer": optimizer,
                "activation function": activation_function,
                "loss function": loss_function,
                "epoch": epoch,
                "learning rate": learning_rate,
                "network shape": network_shape,
                "batch size": batch_size,
                "min training loss": min_training_loss,
                "min testing loss": min_testing_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "time": time
        }

        row:pandas.Series = pandas.Series(data).reindex(df.columns)
        df.loc[len(df)] = row

        df.to_csv(FILE, index=False)
    except Exception as e:
        logger.error(f"An error occured on the history updating: {e}")
        raise Exception()
    
