import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.regularizers import L2

# 数据加载读取
# data = pd.DataFrame(np.random.rand(10000,6),
#                     columns=['cycle_count','discharge_capacity','discharge_energy','current','voltage','temperature'])
data = pd.read_csv("D:/15-创新规划云平台/电芯智能预测/learn-coding/SVM/C104循环测试数据-副本1.csv",
                   usecols=["cycle_count", "discharge_capacity", "discharge_energy", "charge_capacity", "charge_energy",
                            "voltage"])

# target = pd.DataFrame(np.random.rand(10000,1),
#                     columns=['capacity_rentention'])

target = pd.read_csv("D:/15-创新规划云平台/电芯智能预测/learn-coding/SVM/C104循环测试数据-副本1.csv")[
    "capacity_rentention"]
# print(data, target)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

model = Sequential()
model.add(LSTM(64, input_shape=(None, 6), return_sequences=True, kernel_regularizer=L2(0.01)))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=True, kernel_regularizer=L2(0.01)))
model.add(Dropout(0.2))
model.add(LSTM(16, kernel_regularizer=L2(0.01)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
#定义回调函数
early_stop = EarlyStopping(monitor='val_loss',patience=5)

history = model.fit(X_train.values.reshape(-1, 1, 6), y_train.values, epochs=1050, batch_size=32,
                    validation_data=(X_test.values.reshape(-1, 1, 6), y_test.values),callbacks=[early_stop])
print(X_train.values.reshape(-1, 1, 6))
print(X_train.values.shape,X_train.values.reshape(-1, 1, 6).shape)

loss, accuracy = model.evaluate(X_test.values.reshape(-1, 1, 6), y_test.values)
print("accuracy：", accuracy)

predictions = model.predict(X_test.values.reshape(-1, 1, 6))
cycles_at_80_pct = np.argmax(predictions > 0.8) + 1
print("Number of cycles at 80 capacity retention:", cycles_at_80_pct)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['accuracy'], label='test')
plt.title('Model loss')
plt.xlabel('Loss')
plt.ylabel('epoch')
plt.legend()
plt.show()

x_values = range(len(predictions))
y_values = [prediction[0] for prediction in predictions]
plt.plot(x_values, y_values, label='capacity retention')
plt.title('capacity retention vs cycle count')
plt.xlabel('capacity retention')
plt.ylabel('cycle count')
plt.legend()
plt.show()
