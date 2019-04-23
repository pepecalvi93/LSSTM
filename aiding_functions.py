import pickle
from hottbox.core import Tensor
import numpy as np


def save_obj(obj, name ):
    with open('./finance_data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('./finance_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def contractor(x, w, modes):
    """

    Parameters
    ----------
    x: Tensor object
    w: weights for STM to be contracted against
    modes: modes for STM to be contracted against

    Returns
    -------
    x_vec = contracted tensor along all modes except for one

    """

    temp = x.copy()
    for w, mode in zip(w, modes):
        temp.mode_n_product(np.expand_dims(w, axis=0), mode, inplace=True)
    x_vec = np.expand_dims(temp.data.squeeze(), axis=0)

    return x_vec