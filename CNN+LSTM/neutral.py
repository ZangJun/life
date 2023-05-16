import joblib
import matplotlib.pyplot as plt
import torch

import pickle

plt.rcParams['font.sans-serif']=['SimHei']
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

data = pd.read_csv("D:/15-创新规划云平台/电芯智能预测/learn-coding/SVM/C104循环测试数据-副本.csv")
from sklearn.preprocessing  import MinMaxScaler

X= data[['voltage','cc','ce','dc','de']].values
y= data[['battery_life']]

# 'voltage''current''temperature''charge_cycle''discharge_cycle''battery_type''battery_life'

# 归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection  import train_test_split

X_train,X_test,y_train,y_tets = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# 构建深度学习训练模型
# 参数 activation 表示激活函数，以字符串的形式给出，包括relu、softmax、sigmoid、tanh 等。
model = keras.Sequential([
    # keras.layers.Dense(164,activation='relu',input_shape=(5,)),
    # keras.layers.Dense(32,activation='relu'),
    # keras.layers.Dense(64,activation='softmax',input_shape=(5,)),
    # keras.layers.Dense(32,activation='softmax'),

    keras.layers.Dense(64,activation='tanh',input_shape=(5,)),
    keras.layers.Dense(32,activation='tanh'),

    keras.layers.Dense(1)
                          ])
model.summary()

# 编译模型
model.compile(optimizer='adam',loss='mse',metrics=['mae'])


# 定义回调函数
early_stop = EarlyStopping(monitor='val_loss',patience=10)

# 训练模型
history = model.fit(X_train,y_train,epochs=11000,batch_size=16,validation_split=0.2,callbacks=[early_stop])

# 损失函数变化曲线
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.title("训练及验证损失函数")
plt.xlabel("训练次数")
plt.ylabel("损失函数MAE")
epochs = range(1,len(train_loss) + 1)
plt.plot(epochs,train_loss,label="训练损失")
plt.plot(epochs,val_loss,label="验证损失")
plt.title("训练及验证损失")
plt.xlabel("训练次数")
plt.ylabel("Loss")
plt.legend()
plt.show()

#MSE 均方差损失（ Mean Squared Error Loss）MSE是深度学习任务中最常用的一种损失函数，也称为 L2 LossMSE是真实值与预测值的差值的平方然后求和平均
#RMSE 均方根误差（Root Mean Square Error） RMSE就是对MSE开方之后的结果
#MAE 平均绝对误差损失 （Mean Absolute Error Loss）MAE也是常用的损失函数之一，也称为 L1 LossMAE是真实值与预测值的差值的绝对值然后求和平均


#准确率/误差率变化曲线
train_mae = history.history['mae']
val_mae = history.history['val_mae']
plt.title("训练及验证平均绝对误差损失")
plt.xlabel("训练次数")
plt.ylabel("平均绝对误差损失")
# plt.plot(epochs,train_mae,'bo',label = "训练损失函数")
plt.plot(epochs,train_mae,label = "训练MAE")
plt.plot(epochs,val_mae,label = "验证MAE")
# plt.plot(epochs,val_mae,'b',label = "验证损失函数")
plt.legend()
plt.show()


#评估
test_loss,test_mae = model.evaluate(X_test,y_tets)
print('Test mae:',test_mae)
print(history.history)
# mse,r_squared = model.evaluate(X_test,y_tets)
# print("均方误差",mse)
# print("决定系数",r_squared)
test_data = np.array([[3.1384,143.5,498.9,142.7,440.7],[3.1437,143,495.9,143,443.2]])
test_lables = np.array([[1],[2]])
mse,r_squared = model.evaluate(test_data,test_lables)
print("均方误差",mse)
print("决定系数",r_squared)
model.save('battery_model.h5')
with open('model.pkl', 'wb') as f:
    pickle.dump(model,f)

#加载模型
model = tf.keras.models.load_model("battery_model.h5")
new_data = np.array([[3.1384,143.5,498.9,142.7,440.7]])
predcictions = model.predict(new_data)

print("预测值："+str(predcictions))

