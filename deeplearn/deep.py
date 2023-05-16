import os
import math
import datetime
import random as rn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

from PyEMD import EMD, CEEMDAN, Visualisation

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import GRU, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, Dropout, Flatten, SimpleRNN, \
    LSTM
# from keras.callbacks import EarlyStopping
# from tensorflow.keras import regularizers
# from keras.utils.np_utils import to_categorical
# 在使用tensflow.keras可能会遇到如下错误:问题出现原因是在tensorflow与keras之间多了一层python
from tensorflow.python.keras import optimizers

data0=pd.read_csv('NASA电容量.csv',usecols=['B0006'])
S1 = data0.values
S = S1[:,0]
t = np.arange(0,len(S),1)
ceemdan=CEEMDAN()
ceemdan.ceemdan(S)
imfs, res = ceemdan.get_imfs_and_residue()
print(len(imfs))
vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=t , include_residue=False)

