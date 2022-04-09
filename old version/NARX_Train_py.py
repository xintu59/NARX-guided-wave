
import os
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io
from collections import deque

f1 = scipy.io.loadmat('train_normalised2.mat')
training_signals = np.array(f1['train_normalised'])
training_signals = np.transpose(training_signals)

f2 = scipy.io.loadmat('test_normalised.mat')
test_signals = np.array(f2['test_normalised']) # (220,3880)

training_samples = training_signals.flatten()
training_samples = training_samples[:, np.newaxis]

n_u = 20
n_l = 4
n_h = n_u + n_l
hidden_unit = 20
batch_size = 77576
test_size = 10
test_st = 160
sample_length = len(training_samples)
x_train_len = sample_length - n_h

test_samples = test_signals[test_st : (test_st+20), :]
test_samples = test_samples.flatten()
test_samples = test_samples[:, np.newaxis]

x_train = np.zeros ((x_train_len, n_h))
y_train = np.zeros ((x_train_len, 1))
x_test = np.zeros ((x_train_len, n_h))
y_test = np.zeros ((x_train_len,1))

for i in range(x_train_len):
    x_train[i,:] = training_samples[i:(i+n_h),0]
    y_train[i,0] = training_samples[(i+n_h),0]
    x_test[i,:] = test_samples[i:(i+n_h),0]
    y_test[i,0] = test_samples[(i+n_h),0]

x_train_in1 = deque()
x_train_in2 = deque()
x_test_in1 = deque()
x_test_in2 = deque()
for i in range(n_l):
   split_train = x_train[:,i:i+n_u]
   x_train_in1.appendleft(split_train)
   extra_train_value = x_train[:,i+n_u]
   x_train_in2.appendleft(extra_train_value)

   split_test = x_test[:,i:i+n_u]
   x_test_in1.appendleft(split_test)
   extra_test_value = x_test[:,i+n_u]
   x_test_in2.appendleft(extra_test_value)

x_train_in1 = np.asarray(x_train_in1) #(4,77576,20)
x_train_in2 = np.asarray(x_train_in2) #(4,77576)
x_test_in1 = np.asarray(x_test_in1)
x_test_in2 = np.asarray(x_test_in2)

x_train_in1 = np.concatenate(x_train_in1, axis = 1)
x_train_in2 = np.transpose(x_train_in2)
x_train_in = np.concatenate((x_train_in1,x_train_in2), axis=1)

x_test_in1 = np.concatenate(x_test_in1, axis = 1)
x_test_in2 = np.transpose(x_test_in2)
x_test_in = np.concatenate((x_test_in1,x_test_in2), axis=1)

tfkl = tf.keras.layers
Shape = (n_u+1)*n_l
inp = tfkl.Input(shape = Shape)
hidden = tfkl.Dense(hidden_unit, activation=None)(inp)
hidden = tf.keras.activations.tanh(hidden) 
output = tfkl.Dense(1, activation=None) (hidden)
model = tf.keras.Model(inputs=inp, outputs=output)

model.compile(loss=[tf.losses.MeanSquaredError()], 
              metrics=[tf.metrics.MeanAbsoluteError()], 
              optimizer=tf.optimizers.Adam(learning_rate=3e-4))

def get_callbacks():
    early_stopping =  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500) #, mode='min'
    #learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.9 , patience = 5, verbose=2, min_delta=5e-8)
    return (early_stopping) #, learning_rate_reduction

early_stopping = get_callbacks()

history = model.fit(x_train_in, y_train, epochs=1000000, batch_size=batch_size, validation_split=0.1, 
                    callbacks=[early_stopping], verbose=2)
model.summary()

plt.plot(history.history['loss'], label='MSE (training data)')
plt.plot(history.history['val_loss'], label='MSE (validation data)')
plt.yscale('log')
plt.title('Training performance')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.show()

performance = model.evaluate(x_test_in,  y_test, verbose=2)
predictions = model.predict(x_test_in)
residual_raw = y_test - predictions
plt.plot(residual_raw[0:3850])
plt.show()

model.save('NARX43')


