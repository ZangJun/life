import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
#plt.style.use('fivethirtyeight')
plt.style.use('bmh')
#plt.style.use('classic')

