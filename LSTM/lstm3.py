import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

data = pd.read_csv('D:/15-创新规划云平台/电芯智能预测/learn-coding/SVM/C104循环测试数据-副本1.csv')

X = data[
    ['charge_capacity', 'charge_energy', 'discharge_capacity', 'discharge_energy', 'voltage', 'cycle_count']].values
y = data['capacity_rentention1'].values

print(X, y)

# 划分特征变量和目标变量
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建RNN模型
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001))
model.summary()
# 定义回调函数
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# 开始训练
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
history = model.fit(X_train, y_train, epochs=100, batch_size=64, callbacks=[early_stop])

# 预测
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_pred = model.predict(X_test)

# 准确率
mse = model.evaluate(X_test, y_test)
rmse = np.sqrt(mse)
accuracy = 1 - rmse / np.mean(y_test)
print("准确率：%.2f%%" % (accuracy * 100))

# 输出目标谁循环次数变化
plt.plot(data['cycle_count'], data['capacity_rentention1'], label='capacity retention')
plt.xlabel('cycle count')
plt.ylabel('capacity rentention')
plt.legend()
plt.show()

index = np.min(np.where(y_pred > 0.8))
print("cycle count at 80%% capacity rentention : %d" % (index + 1))
