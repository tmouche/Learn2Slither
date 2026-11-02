from utils.constant import COLUMNS, DROP_COLUMNS
from typing import List, Dict
from utils.types import ArrayF, FloatT
import numpy
import pandas

def	normalize(
        df:pandas.DataFrame,
        starting_column:int
    ) -> pandas.DataFrame:
    """
    Norm a pandas dataFrame by columns.

    Args:
        df (pandas.DataFrame): The DataFrame who contains the data.
        starting_column (int): The first column who is going to be normed.

    Returns:
        pandas.DataFrame: a normed DataFrame by columns.
    """
    for i in range(starting_column, len(df.columns)):
        act_column = df.iloc[:,i] 
        c_min = min(act_column)
        c_max = max(act_column)
        for x in range(len(act_column)):
            df.iloc[x,i] = (df.iloc[x,i] - c_min) / (c_max - c_min)
    return df


def create_normalized_data(
        file_path:str,
    ) -> pandas.DataFrame:
    try:
        df = pandas.read_csv(file_path, header=None)
        df.columns = COLUMNS
    except:
        print(f"Error log: can not process {file_path}")
        raise FileNotFoundError(file_path)
    df.drop(columns=DROP_COLUMNS, inplace=True)
    normalized_df: pandas.DataFrame = normalize(df, 1) 
    return normalized_df


def process_df_1_output(
        df: pandas.DataFrame
    ) -> List[Dict[str, ArrayF]]:
    """
    Convert DataFrame into structured dataset
    for a single binary output model.

    Each row of the input DataFrames is transformed into a dictionary
    containing:
        - "label": Binary target encoded as [1] if the first column is 'M',
                   otherwise [0].
        - "data": Feature vector extracted from remaining columns and
                  converted into a NumPy array.

    Args:
        df (pandas.DataFrame):
            dataset where the first column contains class labels
            and the remaining columns contain features.

    Returns:
        List[Dict[str, ArrayF]]:
            - Processed dataset.
    """
    data: List[Dict[str, ArrayF]] = []
    for i in range(len(df)):
        data.append({})
        data[-1]["label"] = numpy.array([1] if df.iloc[i, 0] == 'M' else [0], dtype=FloatT)
        data[-1]["data"] = numpy.array(df.iloc[i, 1:], dtype=FloatT)
    return data


def process_df_2_output(
        df:pandas.DataFrame,
    ) -> List[Dict[str, ArrayF]]:
    """
    Convert training and testing DataFrames into structured datasets
    for a two-output (one-hot encoded) classification model.

    Each row of the input DataFrames is transformed into a dictionary
    containing:
        - "label": One-hot encoded target vector:
            - [1, 0] if the first column is 'M'
            - [0, 1] otherwise
        - "data": Feature vector extracted from remaining columns and
                  converted into a NumPy array.

    Args:
        df (pandas.DataFrame):
            dataset where the first column contains class labels
            and the remaining columns contain features.

    Returns:
        List[Dict[str, ArrayF]]:
            - Processed dataset.
    """
    data: List[Dict[str, ArrayF]] = []
    for i in range(len(df)):
        data.append({})
        data[-1]["label"] = numpy.array([1, 0], dtype=FloatT) if df.iloc[i, 0] == 'M' else numpy.array([0, 1], dtype=FloatT)
        data[-1]["data"] = numpy.array(df.iloc[i, 1:], dtype=FloatT)
    return data
