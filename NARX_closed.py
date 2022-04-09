import os
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io
import math

n_u = 20 #input signal length
n_l = 20 #number of time delayed input signals
n_h = n_u + n_l
hidden_unit = math.ceil(math.sqrt((n_u+1)*n_l+1)+10) # empirical formula used in the paper #20
batch_size = 256 # tested to be the optimum for current optimiser
test_size = 20
test_st = 160
OUT_STEPS = 5 #number of points predicted into the future
MAX_EPOCHS = 5001
tail = n_h+OUT_STEPS #prediction error substracted at the end of each signal
initial_learning_rate = 1e-3
decay_rate = 0.99

f1 = scipy.io.loadmat('train_normalised2.mat')
training_signals = np.array(f1['train_normalised'])
training_signals = np.transpose(training_signals)
training_samples = training_signals.flatten()

f2 = scipy.io.loadmat('test_normalised.mat')
test_signals = np.array(f2['test_normalised'])
test_samples = test_signals[test_st : (test_st+test_size), :]
test_samples = test_samples.flatten()

train_len = len(training_samples)-n_h-OUT_STEPS+1

x_train = np.zeros((train_len, n_h))
y_train = np.zeros((train_len,1))
for i in range(train_len):
    x_train[i,:] = training_samples[i:(i+n_h)]
    y_train[i,:] = training_samples[(i+n_h+OUT_STEPS-1)]

x_test = np.zeros((train_len, n_h))
y_test = np.zeros((train_len,1))
for i in range(train_len):
    x_test[i,:] = test_samples[i:(i+n_h)]
    y_test[i,:] = test_samples[(i+n_h+OUT_STEPS-1)]

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

class FeedBack(tf.keras.Model):
    def __init__(self, hidden_units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.hidden_units = hidden_units

        self.hidden_layer = tf.keras.layers.Dense(hidden_units, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

feedback_model = FeedBack(hidden_units=hidden_unit, out_steps=OUT_STEPS)

def split(self, inputs):
    x = []
    for b in range(n_l):
        split_layer = tf.keras.layers.Lambda(lambda x: x[:,b:b+n_u+1])(inputs)
        x.append(split_layer)
    return x

FeedBack.split = split

def call(self, inputs, training=None):
    
    for n in range(self.out_steps):
        if n_l == 1:
            inp = inputs
        else:
            x = self.split(inputs)
            inp = tf.keras.layers.Concatenate(axis=-1)(x)
        hidden = self.hidden_layer(inp)
        output = self.output_layer(hidden)
        inputs = tf.concat((inputs[:,-n_h+1:],output),axis=1) #feed the output back to input
    return output

FeedBack.call = call

feedback_model.compile(loss=tf.losses.MeanSquaredError(),
                       optimizer=tf.optimizers.Adam(learning_rate=initial_learning_rate),
                       metrics=[tf.metrics.MeanAbsoluteError()])

#def sched(epoch,lr): #lr here is current lr, required as an input even if not used
    #return initial_learning_rate * np.power(decay_rate,epoch)
#lrsched = tf.keras.callbacks.LearningRateScheduler(sched, verbose=0)

def get_callbacks():
    early_stopping =  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True) #, mode='min'
    #learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.9 , patience = 5, verbose=2, min_delta=5e-8)
    return (early_stopping) #, learning_rate_reduction

early_stopping = get_callbacks()

history = feedback_model.fit(x_train, y_train, epochs=MAX_EPOCHS, batch_size=batch_size,
                    validation_split=0.1,callbacks=[early_stopping], verbose=2)

feedback_model.summary()

plt.plot(history.history['loss'], label='MSE (training data)')
plt.plot(history.history['val_loss'], label='MSE (validation data)')
plt.yscale('log')
plt.title('Training performance')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.show()

performance = feedback_model.evaluate(x_test,  y_test, verbose=2)
#predictions = feedback_model.predict(x_test)
#residual_raw = y_test - predictions
#plt.plot(residual_raw[0:(3880-tail)])
#plt.show()

feedback_model.save('NARX_nu'+f'{n_u}'+'_nl'+f'{n_l}'+'_out'+f'{OUT_STEPS}')
