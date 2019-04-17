import pandas as pd
import numpy as np
import copy
from hottbox.core import Tensor

def make_data_svm(df , start_index, L):
    """

    Parameters
    ----------
    df: pd.DataFrame, the data organized for SVM
    start_index: int
    L: int (lookback window)

    Returns
    -------
    train_data
    train_labels
    test_data
    test_labels
    """

    df_svm = df.copy(deep=True)

    df_svm = df_svm[start_index:start_index + L]

    df_data = df_svm.drop('Label', axis=1)
    df_label = df_svm['Label']

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



def make_data_stm(d, start_index, L, tensor_size):
    """

    Notes: Assumes 3rd order tensors for now

    Parameters
    ----------
    data: dict, the data organized for STM. Each entry of the dict is a dataframe.
    The labels are in the first slice

    start_index: int
    L: int
    tensor_size:list,  the size of your tensor data.

    Returns
    -------
    train_data
    train_labels
    test_data
    test_labels

    """

    dict_stm = copy.deepcopy(d)
    n_slices = len(dict_stm.keys())
    for key in dict_stm.keys():
        dict_stm[key] = dict_stm[key][start_index:start_index + L]

    data_np = np.zeros([L] + tensor_size[1:])
    for i, key in enumerate(dict_stm.keys()):
        data_np[:,:,i] = np.array(dict_stm[key].drop('Label', axis=1))

    train_data = []
    n_tensors = L - tensor_size[0]+1
    train_labels = np.zeros(n_tensors-1)

    for i in range(n_tensors):
        # if training data
        if i<n_tensors-1:
            train_data.append(Tensor(data_np[i:i+tensor_size[0],:,:]))
            train_labels[i] = dict_stm['Price']['Label'][i+tensor_size[0]-1]

        #otherwise it's testing data
        else:
            test_data = Tensor(data_np[i:i + tensor_size[0], :, :])
            test_label = dict_stm['Price']['Label'][i + tensor_size[0] - 1]


    associated_index = dict_stm['Price'].iloc[-2].name


    return train_data, train_labels, test_data, test_label, associated_index