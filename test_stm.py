import pandas as pd
from data_makers import make_data_svm
from aiding_functions import load_obj
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


raw_data = pd.read_csv('./finance_data/raw_data.csv')
raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m-%d')
raw_data = raw_data.set_index('Date')

data_stm = load_obj('data_stm')