import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import L2
from tensorflow.python.keras.callbacks import EarlyStopping

# 数据加载读取
data = pd.read_csv('D:/15-创新规划云平台/电芯智能预测/learn-coding/SVM/C104循环测试数据-副本1.csv')
# data = pd.read_csv("D:/15-创新规划云平台/电芯智能预测/learn-coding/SVM/C104循环测试数据-副本1.csv",
#                    usecols=["cycle_count", "discharge_capacity", "discharge_energy", "charge_capacity", "charge_energy",
#                             "voltage"])
#
# target = pd.read_csv("D:/15-创新规划云平台/电芯智能预测/learn-coding/SVM/C104循环测试数据-副本1.csv")[
#     "capacity_rentention"]
data = data[['cycle_count', 'discharge_capacity', 'discharge_energy', 'charge_capacity', 'charge_energy',
             'voltage']].values
target = data["capacity_rentention"].values

print(data, target)
# print(X,y)

# 划分测试
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
train_size = int(len(data)*0.8)
X_train,X_test = data[:train_size], data[train_size:]
y_train,y_test = target[train_size:], target[train_size:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print(X_train, X_train.shape)
# 模型定义

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)])

model.compile(loss=',mse', optimizer='adam', metrics=['accuracy'])

# 定义回调函数
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# 训练
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test),
                    callbacks=[early_stop])

accuracy = model.evaluate(X_test, y_test)[1]

cycle_range = np.arange(1, data['cycle_count'].max())
features = np.zeros((len(cycle_range), X_train.shape[1]))
features[:, 0] = cycle_range
features = (features - mean) / std
predictions = model.predict(features)

retention_80 = cycle_range[np.argmax(predictions > 0.8)]

print("准确率", accuracy)
print("80%循环次数", retention_80)
