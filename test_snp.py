import pandas as pd
from aiding_functions import load_obj
from Strategy import Strategy
import matplotlib.pyplot as plt


raw_data = pd.read_csv('./finance_data/raw_data.csv')
data_svm = pd.read_csv('./finance_data/raw_data_svm.csv')
data_stm = load_obj('data_stm')

C = 10
sig2 = 100
strat = Strategy(raw_data = raw_data)
results = strat.stm(data_stm=data_stm, tensor_size=[2,3,2], lookBack=250, lag=True, C=C, kernel='linear', sig2=sig2, verbose=True)
results_stm = results['Performance']
results_stm.plot(figsize=(15,5))
plt.title('Cost = {}, sigma = {}'.format(C, sig2))
plt.show()

print(results['Accuracy'])
