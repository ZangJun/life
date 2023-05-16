import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, Callback

# 数据载入
data = pd.read_csv('D:/15-创新规划云平台/电芯智能预测/learn-coding/SVM/C104循环测试数据-副本1.csv')

X = data[
    ['charge_capacity', 'charge_energy', 'discharge_capacity', 'discharge_energy', 'voltage', 'cycle_count']].values
y = data['capacity_rentention1'].values

# X = data[['charge_capacity', 'charge_energy', 'discharge_capacity', 'discharge_energy', 'voltage', 'cycle_count']]
# y = data['capacity_rentention1']

# 划分特征变量和目标变量
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RNN模型输入是三维的，需要将输入转化为三维
# def reshape_dataset(dataset):
#     reshape_dataset = []
#     for i in range(len(dataset)):
#         reshape_dataset.append(dataset[i:i + 100])
#     return np.array(reshape_dataset)


#   将numpy转化为张量
# X_train = reshape_dataset(X_train)
# X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
# X_test = reshape_dataset(X_test)
# X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

# 目标变量进行转化
# y_train = np.expand_dims(y_train.values[99:], axis=1)
# y_test = np.expand_dims(y_test.values[99:], axis=1)

# 创建RNN模型
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))

# 编译模型
model.compile(loss='mse', optimizer='adam')
model.summary()
model.save('battery-life-model')

# 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stop])

# 评估模型
loss = model.evaluate(X_test, y_test)
rmse = np.sqrt(loss)
accuracy = 1 - rmse / np.mean(y_test)

# 预测
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_pred = model.predict(X_test)

print('loss', loss)
print("准确率：%.2f%%" % (accuracy * 100))

# print('accuracy', accuracy)

# 图形
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val_loss')
# plt.axhline(y=80,color='red',linestyle='-')
# plt.axhline(x = data['cycle_count'],color='red',linestyle='-')
plt.title('Model loss&accuracy')
plt.xlabel('Loss')
plt.ylabel('epoch')
plt.legend()
plt.show()



# 输出目标谁循环次数变化
plt.plot(data['cycle_count'], data['capacity_rentention1'], label='capacity retention')
plt.xlabel('cycle count')
plt.ylabel('capacity rentention(%)')
plt.legend()
plt.show()



#
# index = np.min(np.where(y_pred > 0.8))
# print("cycle count at 80%% capacity rentention : %d" % (index + 1))
model.save("battery-life-model")