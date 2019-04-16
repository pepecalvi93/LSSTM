import pandas as pd
import numpy as np

def make_data_svm(df , start_index, L):
    """

    Parameters
    ----------
    df: pd.DataFrame
    start_index: int
    L: int (lookback window)

    Returns
    -------
    train_data
    train_labels
    test_data
    test_labels
    """
    df = df[start_index:start_index + L]

    df_data = df.drop('Label', axis=1)
    df_label = df['Label']

    train_data = np.array(df_data[0:L-1])
    train_labels = np.array(df_label[0:L-1])

    test_data = np.array(df_data.iloc[L-1])
    test_label = df_label[L-1]

    """
    This is the time-index of how the S&P actually performed on that day. Multiply the predicted labels with the #
    differences at these indices to get the generated profits"
    """
    associated_index = df_data.iloc[-2].name


    return train_data, train_labels, test_data, test_label, associated_index



def make_data_stm()