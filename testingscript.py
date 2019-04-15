import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from temp.read_data.eth80 import get_meta, load_as_factorised, load_as_original
from hottbox.core import Tensor
from sklearn.svm import SVC

import random
from STM.LS_STM import LSSTM

def split_train_test(df, train_size):
    test_size = 1 - train_size
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)
    return df_train, df_test


df_meta = get_meta()
selected_A1 = ['066']
selected_A2 = ['027']
selected_label = [5,6]

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

train_size = 0.5
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

print(type(X_test))
print(X_test[0])
plt.imshow(X_test[0].data)

meanTrain = np.zeros(X_train[0].shape)
for i in range(len(X_train)):
    meanTrain += X_train[i].data
meanTrain = meanTrain / len(X_train)


y_train = y_train.tolist()
y_test = y_test.tolist()



XXtrain = [Tensor((X_train[i].data - meanTrain) / np.linalg.norm(X_train[i].data - meanTrain)) for i in range(len(X_train))]
#XXtrain = [Tensor(XXtrain[i].data / (XXtrain[i].frob_norm)) for i in range(len(XXtrain))]

XXtest = [Tensor((X_test[i].data - meanTrain) / np.linalg.norm(X_test[i].data - meanTrain)) for i in range(len(X_test))]
#XXtest = [Tensor(XXtest[i].data / (XXtest[i].frob_norm)) for i in range(len(XXtest))]


stm = LSSTM(C=10, max_iter=200)
stm.fit(XXtrain, y_train, kernel='RBF', sig2=0.01)

y_pred = stm.predict(XXtest)

y_pred = np.array(y_pred)
y_test = np.array(y_test)
acc = (y_test == y_pred).sum() / len(y_test)
n_pos = (y_pred == selected_label[0]).sum()
n_neg = (y_pred == selected_label[1]).sum()

print('\n=========STM Results==========')
print('\nAccuracy rate: {}'.format(acc))
print('Number of positive predictions: {}\nNumber of negative predictions: {}'.format(n_pos, n_neg))
print('Iteration number reached: {}'.format(stm.model['nIter']))


X_svm_train = [tmp.vectorise().data for tmp in XXtrain]
X_svm_test = [tmp.vectorise().data for tmp in XXtest]

print('\n=========SVM Results==========')

clf = SVC(kernel='rbf', C=10, gamma='auto')
clf.fit(X_svm_train, y_train)
y_pred_svm = clf.predict(X_svm_test)
acc_svm = (y_test == y_pred_svm).sum() / len(y_test)
n_pos_svm = (y_pred_svm == selected_label[0]).sum()
n_neg_svm = (y_pred_svm == selected_label[1]).sum()

print('\nAccuracy rate: {}'.format(acc_svm))
print('Number of positive predictions: {}\nNumber of negative predictions: {}'.format(n_pos_svm, n_neg_svm))



























































































