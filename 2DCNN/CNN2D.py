import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, LeakyReLU
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

B0005 = loadmat('C:/Users/zang.jun/Desktop/NASAbattery/B0006.mat')


def extract_discharge(Battery):
    cap = []
    i = 1
    for Bat in Battery.values():
        if Bat['cycle'] == 'discharge':
            cap.append((Bat['data']['Capacity'][0]))
            i += 1
            return cap


cap6 = extract_discharge(B0005)

print(cap6)
