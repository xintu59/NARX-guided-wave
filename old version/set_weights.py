import os
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io

n_u = 20
n_l = 4
hidden_unit = 20

model = tf.keras.models.load_model('NARX_trained6_SGD')
weight = model.get_weights()
print(weight)

for i in range(len(weight)):

    print(weight[i].shape)

f1 = scipy.io.loadmat('IW.mat')
IW = np.array(f1['IW'])
print(IW.shape)
f2 = scipy.io.loadmat('b1.mat')
b1 = np.array(f2['b1'])
print(b1.shape)
f3 = scipy.io.loadmat('LW.mat')
LW = np.array(f3['LW2_1'])
print(LW.shape)
f4 = scipy.io.loadmat('b2.mat')
b2 = np.array(f4['b2'])
print(b2.shape)

weight [0] = np.transpose(IW)
weight [1] = b1[:,0]
weight [2] = np.transpose(LW)
print(weight[2].shape)
weight [3] = b2 [0]

model.layers[1].set_weights([weight[0],weight[1]])
model.layers[3].set_weights([weight[2],weight[3]])

model.save('NARX_from_matlab')