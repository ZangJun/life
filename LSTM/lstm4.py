import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, Callback

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

# 定义回调函数
early_stop = EarlyStopping(monitor='val_loss', patience=5)


# 定义损失函数和准确率计算的回调函数
class LossHistory(Callback):
    # def on_train_begin(self, logs={}):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.accuracies = []

    def on_epoch_end(self, batch, logs={}):
        # 预测
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_pred = self.model.predict(X_test)

        # 准确率
        mse = self.model.evaluate(X_test, y_test)
        rmse = np.sqrt(mse)
        accuracy = 1 - rmse / np.mean(y_test)
        self.accuracies.append(accuracy)


history_loss = LossHistory()
history_accuracy = AccuracyHistory()

# 开始训练
for i in range(100):
    loss = model.train_on_batch(X_train, y_train)
    history_loss.on_epoch_end(i)
    history_accuracy.on_epoch_end(i)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# history = model.fit(X_train, y_train, epochs=100, batch_size=64, callbacks=[early_stop])

# 输出目标谁循环次数变化
# plt.plot(data['cycle_count'], data['capacity_rentention1'], label='capacity retention')
# plt.xlabel('cycle count')
# plt.ylabel('capacity rentention')
# plt.legend()
# plt.show()

plt.plot(history_loss.losses)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(history_accuracy.accuracies)
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_pred = model.predict(X_test_reshaped)
mse = model.evaluate(X_test_reshaped, y_test)
rmse = np.sqrt(mse)
accuracy = accuracy = 1 - rmse / np.mean(y_test)
print("准确率：%.2f%%" % (accuracy * 100))
index = np.min(np.where(y_pred > 0.8))
print("cycle count at 80%% capacity rentention : %d" % (index + 1))
