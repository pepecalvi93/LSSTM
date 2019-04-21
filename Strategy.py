import pandas as pd
from sklearn.svm import SVC
from data_makers import make_data_svm, make_data_stm
import numpy as np
from LS_STM import LSSTM
from aiding_functions import load_obj
import copy

class Strategy:
    def __init__(self, raw_data):
        """

        Parameters
        ----------
        raw_data: pd.DataFrame, having the prices of the time series
        """
        self.raw_data = raw_data

    def svm(self, data_svm, lookBack=250, kernel='linear', C=10, gamma='auto'):
        raw_data = copy.deepcopy(self.raw_data)
        raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m-%d')
        raw_data = raw_data.set_index('Date')

        data_svm = copy.deepcopy(data_svm)
        data_svm['Date'] = pd.to_datetime(data_svm['Date'], format='%Y-%m-%d')
        data_svm = data_svm.set_index('Date')

        M = len(data_svm)
        lookBack = lookBack
        y_pred_svm = np.zeros(M - 1 - lookBack)
        success = 0

        clf = SVC(kernel=kernel, C=C, gamma=gamma)

        indices = []
        for i in range(M - 1 - lookBack):
            train_data, train_label, test_data, test_label, associated_index = make_data_svm(data_svm, start_index=i,
                                                                                             L=lookBack)
            clf.fit(train_data, train_label)
            y_pred_svm[i] = clf.predict(test_data.reshape(1, -1))

            indices.append(associated_index)
            if test_label == y_pred_svm[i]:
                success += 1

        raw_data_diff = raw_data.diff().dropna().loc[indices]
        raw_data_diff['Strategy'] = raw_data_diff['SPX Close'] * y_pred_svm
        raw_data_diff = raw_data_diff[['SPX Close', 'Strategy']]


        results = {'Accuracy': success / (M - 1 - lookBack),
                   'Performance': raw_data_diff.cumsum()}


        return results


    def stm(self, data_stm, tensor_size, lookBack, C=10, kernel='linear', sig2=1, max_iter=100, lag=False, verbose=False):
        raw_data = copy.deepcopy(self.raw_data)
        raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m-%d')
        raw_data = raw_data.set_index('Date')

        data_stm = copy.deepcopy(data_stm)

        M = len(data_stm['Price'])
        tensor_size = tensor_size
        lookBack = lookBack

        n_tens = M - lookBack - tensor_size[0] + 1

        y_pred_stm = np.zeros(n_tens - 1)

        stm = LSSTM(C=C, kernel=kernel, sig2=sig2, max_iter=max_iter)
        success = 0
        indices = []
        for i in range(n_tens - 1):
            if verbose:
                print("\r{0}".format((float(i) / (n_tens - 1)) * 100))
            train_data, train_labels, test_data, test_label, associated_index = make_data_stm(d=data_stm,start_index=i, L=lookBack, tensor_size=tensor_size, lag=lag)
            stm.fit(train_data, train_labels)
            y_tmp, _ = stm.predict(test_data)
            y_pred_stm[i] = y_tmp[0]

            indices.append(associated_index)
            if test_label == y_pred_stm[i]:
                success += 1

        raw_data_diff = raw_data.diff().dropna().loc[indices]
        raw_data_diff['Strategy'] = raw_data_diff['SPX Close'] * y_pred_stm
        raw_data_diff = raw_data_diff[['SPX Close', 'Strategy']]


        #raw_data_diff.cumsum().plot(figsize=(10, 5))

        results = {'Accuracy': success / (n_tens - 1),
                   'Performance': raw_data_diff.cumsum()}

        return results