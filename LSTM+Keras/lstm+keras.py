import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import glob
import time

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from tqdm import tqdm_notebook as tqdm
from ipywidgets import IntProgress

# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# from livelossplot import PlotLossesKeras

# 以下数据来自马里兰夫大学
# Python中缺失值删除 pd.dropna()函数
# # index_col=0,将第一列作为索引列，dataframe打印就会失去第一列
df_35 = pd.read_csv('CS2_35.csv', index_col=0).dropna()
df_36 = pd.read_csv('CS2_36.csv', index_col=0).dropna()
df_37 = pd.read_csv('CS2_37.csv', index_col=0).dropna()
df_38 = pd.read_csv('CS2_38.csv', index_col=0).dropna()
# df_39 = pd.read_csv('C104循环测试数据-副本1.csv', index_col=0).dropna()


# plt.rcParams['figure.figsize'] = (4.48, 4.48) #2.24, 2.24 设置figure_size尺寸
# plt.rcParams['savefig.dpi'] = 50 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率

fig = plt.figure(figsize=(9, 8), dpi=150)
names = ['capacity', 'resistance', 'CCCT', 'CVCT']
titles = ['Discharge Capacity (mAh)', 'Internal Resistance (Ohm)',
          'Constant Current Charging Time (s)', 'Constant Voltage Charging Time (s)']
plt.subplots_adjust(hspace=0.25)
plt.show()

# fig = plt.figure(figsize=(9, 14), dpi=350)
# # fig = plt.savefig(dpi=50)
# # plt.rcParams['savefig.dpi'] = 50  # 图片像素
# names = ['charge_capacity', 'charge_energy', 'discharge_capacity', 'discharge_energy', 'voltage', 'cycle_count',
#          'capacity_rentention1']
# titles = ['Charge capacity (Ah)', 'Charge energy (Wh)', 'Discharge capacity (Ah)', 'Discharge energy (Wh)',
#           'Voltage (V)', 'Cycle count (圈)', 'Capacity rentention (%)']

for i in range(4):
    # plt.subplot(2,2,1)表示将整个图像窗口分为2行2列, 当前位置为1.
    plt.subplot(2, 2, i + 1)
    plt.plot(df_35[names[i]], 'o', ms=2, label='#35')
    plt.plot(df_36[names[i]], 'o', ms=2, label='#36')
    plt.plot(df_37[names[i]], 'o', ms=2, label='#37')
    plt.plot(df_38[names[i]], 'o', ms=2, label='#38')
    plt.title(titles[i], fontsize=14)
    plt.legend(loc='upper right')
    if i == 3:
        plt.ylim(1000, 5000)
    plt.xlim(-20, 1100)
plt.show()

# for i in range(7):
#     plt.subplot(4, 2, i + 1)
#     plt.plot(df_39[names[i]], 'o', ms=2, label='#C104F')
#     plt.title(titles[i], fontsize=14)
#     plt.legend(loc='upper right')
#     # if i == 3:
#     #     plt.ylim(1000, 5000)
#     plt.xlim(-20, 2000)

def ConvertData(dataset, t_width):
    X_trains = []
    y_trains = []

    for df in dataset:
        # len()函数，可以用来获取字符串长数，以及字节数；当中的encode()
        # >> > stu2 = '信春哥，得永生'
        # >> > len(stu2.encode("gbk"))
        # 14
        # 方法，是用来将字符串进行编码；如果我们想用其他的编码方式去获取，我们只需要再encode()
        # 方法里添加你要使用的编码。
        t_length = len(df)
        # # 1：参数值为终点，起点值默认为0，步长值默认为1
        # a = np.arange(6)
        # # [0 1 2 3 4 5]
        train_x = np.arange(t_length)
        # np.array() 把列表转化为数组,列表不存在维度问题，数组是有维度的
        capacity = np.array(df['capacity'])
        train_y = capacity

        for i in range(t_length - t_width):
            # L.append(obj)
            # 参数:obj - - 追加到列表末尾的对象。
            # 返回值:该方法无返回值，但会在原来的列表末尾追加新的对象。
            X_trains.append(train_y[i:i + t_width])
            y_trains.append(train_y[i + t_width])

    X_trains = np.array(X_trains)
    y_trains = np.array(y_trains)

    return X_trains, y_trains


X_train, y_train = ConvertData([df_35, df_37, df_38], 50)
X_test, y_test = ConvertData([df_36], 50)
# 打印输出值理解ConvertData
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(df_35.shape)
# print(X_train)

# 打印数据到csv文件
# record = pd.DataFrame(X_train)
# record.to_csv('X_train.csv', index=False)

# print("这是y_train", y_train)  # y_train 是列表
print("这是df_35长度,进行了降噪剩余长度", len(df_35.dropna(axis=0)))
print("这是df_35的容量\n", df_35.dropna(axis=0)['capacity'])
print("这是df_35转化后为数组的情况\n", np.array(df_35.dropna(axis=0)['capacity']))
print("这是df_35转化后为数组的情况的构型\n", np.array(df_35.dropna(axis=0)['capacity']).shape)
print("这是df_35的 ConverData里面的 i 的取值范围 \n", len(df_35.dropna(axis=0)) - 50)

a = np.array(df_35.dropna(axis=0)['capacity'][1:1 + 50])
print("train_y这是X_trains a：\n", a, a.shape)
# record = pd.DataFrame(a)
# record.to_csv('X_train_a.csv', index=False)

b = np.array(df_35.dropna(axis=0)['capacity'][1 + 50])
print("train_y这是y_trains b：\n", b, b.shape)

c = np.array(df_35.dropna(axis=0)['capacity'][2:2 + 50])
print("train_y这是X_trains c：\n", c, c.shape)
# record = pd.DataFrame(c)
# record.to_csv('X_train_c.csv', index=False)

d = np.array(df_35.dropna(axis=0)['capacity'][2 + 50])
print("train_y这是y_trains d：\n", c, c.shape)

print("数据长度分别是\n", len(df_38), len(df_35), len(df_37), len(df_38) + len(df_35) + len(df_37))

idx = np.arange(0, X_train.shape[0], 1)
idx = np.random.permutation(idx)
idx_lim = idx[:500]

X_train = X_train[idx_lim]
y_train = y_train[idx_lim]
X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], 1])
X_test = X_test.reshape([X_test.shape[0], X_test.shape[1], 1])
y_train = y_train.reshape([y_train.shape[0], 1])
y_test = y_test.reshape([y_test.shape[0], 1])
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

length_of_sequence = X_train.shape[1]
in_out_neurons = 1
n_hidden = 3

model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons),
               return_sequences=False, dropout=0))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

model.summary()
model.save('battery-life-model')
#回调函数
early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
history = model.fit(X_train, y_train,
                    batch_size=50,
                    epochs=100,
                    validation_split=0.01,
                    # callbacks=[early_stopping, PlotLossesKeras()]
                    callbacks=[early_stopping]
                    )

# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='val_loss')
# # plt.axhline(y=80,color='red',linestyle='-')
# # plt.axhline(x = data['cycle_count'],color='red',linestyle='-')
# plt.title('Model loss&accuracy')
# plt.xlabel('Loss')
# plt.ylabel('epoch')
# plt.legend()
# plt.show()



predicted = model.predict(X_train)
predicted = predicted.reshape(predicted.shape[0])
fig1 = plt.figure(figsize=(12, 4), dpi=150)
plt.plot(predicted, alpha=0.7, label='Prediction')
plt.plot(y_train, alpha=0.7, label='True')
plt.legend(loc='upper right', fontsize=12)
plt.show()

predicted = model.predict(X_test[300:800])
predicted = predicted.reshape(predicted.shape[0])

x_range = np.linspace(301, 800, 500)
fig = plt.figure(figsize=(12, 4), dpi=150)
plt.plot(x_range, predicted, label='predict')
plt.plot(x_range, y_test[300:800], label='true')
plt.xlabel('Number of Cycle', fontsize=13)
plt.ylabel('DIscharge Capacity (Ah)', fontsize=13)
plt.title('LSTM Prediction of Discharge Capacity of Test Data (CS2-36)', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.show()

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test[300:800], predicted)

initial = X_test[500]
results = []
for i in tqdm(range(50)):
    if i == 0:
        initial = initial.reshape(1, 50, 1)
        res = model.predict(initial)
        results.append(res[0][0])
    else:
        initial = initial.reshape(50, 1)
        initial = np.vstack((initial[1:], np.array(res)))
        initial = initial.reshape(1, 50, 1)
        res = model.predict([initial])
        results.append(res[0][0])

fig = plt.figure(figsize=(12, 4), dpi=150)
plt.plot(np.linspace(501, 550, 50), results, 'o-', ms=4, lw=1, label='predict')
plt.plot(np.linspace(401, 550, 150), y_test[400:550], 'o-', lw=1, ms=4, label='true')
plt.legend(loc='upper right', fontsize=12)
plt.xlabel('Number of Cycle', fontsize=13)
plt.ylabel('Discharge Capacity (Ah)', fontsize=13)
plt.show()
