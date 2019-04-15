import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from temp.read_data.eth80 import get_meta, load_as_factorised, load_as_original
from hottbox.core import Tensor


import random

def split_train_test(df, train_size):
    test_size = 1 - train_size
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)
    return df_train, df_test


def load_data(labels, train_size=0.5):
    """

    Parameters
    ----------
    labels: list of 2 elements (1 to 9)

    Returns: X_train, X_test, y_train, y_test
    -------

    """
    df_meta = get_meta()
    selected_A1 = ['066']
    selected_A2 = ['027']
    selected_label = labels

    case = 2

    if case == 1:
        df = df_meta[(df_meta.Angle_1.isin(selected_A1)) &
                     (df_meta.Angle_2.isin(selected_A2)) &
                     (df_meta.Label.isin(selected_label))
                    ]
    elif case == 2:
        df = df_meta[(df_meta.Label.isin(selected_label))
                    ]
    elif case == 3:
        df = df_meta[(df_meta.Angle_1.isin(selected_A1)) &
                     (df_meta.Angle_2.isin(selected_A2))
                    ]
    else:
        df = df_meta

    df_train, df_test = split_train_test(df, train_size)

    to_gray = False
    if to_gray:
        orig_shape = (128, 128)
    else:
        orig_shape = (128, 128, 3)

    X_train_flatten, y_train = load_as_original(df=df_train, to_gray=to_gray)
    X_test_flatten, y_test = load_as_original(df=df_test, to_gray=to_gray)
    X_train = np.apply_along_axis(lambda x: Tensor(x.reshape(orig_shape)),
                                  1,
                                  X_train_flatten
                                  ).tolist()
    X_test = np.apply_along_axis(lambda x: Tensor(x.reshape(orig_shape)),
                                 1,
                                 X_test_flatten
                                 ).tolist()

    return X_train, X_test, y_train, y_test