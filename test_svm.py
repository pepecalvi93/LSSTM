import pandas as pd
from data_makers import make_data_svm
from aiding_functions import load_obj
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from LS_STM import LSSTM


#---------------------------------------SVM--------------------------------------------------
raw_data = pd.read_csv('./finance_data/raw_data.csv')
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m-%d')
raw_data = raw_data.set_index('Date')

data_svm = pd.read_csv('./finance_data/raw_data_svm.csv')
data_svm['Date'] = pd.to_datetime(data_svm['Date'], format='%Y-%m-%d')
data_svm = data_svm.set_index('Date')

M = len(data_svm)
lookBack = 250
y_pred_svm = np.zeros(M-1-lookBack)
success = 0

clf = SVC(kernel='linear', C=100, gamma='auto')

indices = []
for i in range(M-1-lookBack):
    train_data, train_label, test_data, test_label, associated_index = make_data_svm(data_svm, start_index=i, L=lookBack)
    clf.fit(train_data, train_label)
    y_pred_svm[i] = clf.predict(test_data.reshape(1,-1))

    indices.append(associated_index)
    if test_label == y_pred_svm[i]:
        success += 1

raw_data_diff = raw_data.diff().dropna().loc[indices]
raw_data_diff['Strategy'] = raw_data_diff['SPX Close'] * y_pred_svm
raw_data_diff = raw_data_diff[['SPX Close', 'Strategy']]

print(success/(M-1-lookBack))

raw_data_diff.cumsum().plot(figsize=(10,5))
plt.show()




#---------------------------------------STM--------------------------------------------------
dict_data = load_obj(dict_data)



























