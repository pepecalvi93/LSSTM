import pandas as pd
from aiding_functions import make_data_svm
from sklearn.svm import SVC
import numpy as np
from LS_STM import LSSTM



raw_data = pd.read_csv('./finance_data/raw_data.csv')
data_svm = pd.read_csv('./finance_data/raw_data_svm.csv')
data_svm['Date'] = pd.to_datetime(data_svm['Date'], format='%Y-%m-%d')
data_svm = data_svm.set_index('Date')

M = len(data_svm)
lookBack = 250
y_pred_svm = np.zeros(M-1-lookBack)
success = 0

clf = SVC(kernel='rbf', C=100, gamma='auto')

for i in range(M-1-lookBack):
    train_data, train_label, test_data, test_label = make_data_svm(data_svm, start_index=i, L=lookBack)
    clf.fit(train_data, train_label)
    y_pred_svm[i] = clf.predict(test_data.reshape(1,-1))
    if test_label == y_pred_svm[i]:
        success += 1

print(success/(M-1-lookBack))
print('CIAO')