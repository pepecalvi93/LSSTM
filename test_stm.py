import pandas as pd
from LS_STM import LSSTM
from data_makers import make_data_stm
from aiding_functions import load_obj
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


raw_data = pd.read_csv('./finance_data/raw_data.csv')
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m-%d')
raw_data = raw_data.set_index('Date')

data_stm = load_obj('data_stm')

M = len(data_stm['Price'])
tensor_size = [2,3,2]
lookBack = 250

n_tens = M - lookBack - tensor_size[0] + 1

y_pred_stm = np.zeros(n_tens - 1)

stm = LSSTM(C=10, max_iter=100)
success = 0
indices = []
for i in range(n_tens - 1):
    print("\r{0}".format((float(i) / (n_tens-1)) * 100))
    train_data, train_labels, test_data, test_label, associated_index = make_data_stm(d=data_stm, start_index=i,
                                                                                      L=lookBack,
                                                                                      tensor_size=tensor_size)
    stm.fit(train_data, train_labels)
    y_tmp, _ = stm.predict(test_data)
    y_pred_stm[i] = y_tmp[0]

    indices.append(associated_index)
    if test_label == y_pred_stm[i]:
        success += 1

raw_data_diff = raw_data.diff().dropna().loc[indices]
raw_data_diff['Strategy'] = raw_data_diff['SPX Close'] * y_pred_stm
raw_data_diff = raw_data_diff[['SPX Close', 'Strategy']]

print(success/(n_tens - 1))

raw_data_diff.cumsum().plot(figsize=(10,5))
plt.show()