import os
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io
import statistics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import math

n_u = 20
n_l = 20
n_h = n_u + n_l
hidden_unit = math.ceil(math.sqrt((n_u+1)*n_l+1)+10)
test_size = 220
batch_size = 256
OUT_STEPS = 10
MAX_EPOCHS = 500
tail = n_h+OUT_STEPS

model = tf.keras.models.load_model('NARX_nu'+f'{n_u}'+'_nl'+f'{n_l}'+'_out'+f'{OUT_STEPS}') 
model.summary()

f1 = scipy.io.loadmat('train_normalised2.mat')
training_signals = np.array(f1['train_normalised']) # (20,3880)
training_signals = np.transpose(training_signals)

f2 = scipy.io.loadmat('test_normalised.mat')
test_signals = np.array(f2['test_normalised']) # (220,3880)

training_samples = training_signals.flatten()
training_samples = training_samples[:, np.newaxis] # (77600,1)

f3 = scipy.io.loadmat('test_defect_5db.mat')
test_defect_5db = np.array(f3['test_defect']) # (220,3880)
f5 = scipy.io.loadmat('test_defect_10db.mat')
test_defect_10db = np.array(f5['test_defect'])
f6 = scipy.io.loadmat('test_defect_20db.mat')
test_defect_20db = np.array(f6['test_defect'])
f7 = scipy.io.loadmat('test_defect_25db.mat')
test_defect_25db = np.array(f7['test_defect'])
f8 = scipy.io.loadmat('test_defect_30db.mat')
test_defect_30db = np.array(f8['test_defect'])

f4 = scipy.io.loadmat('test_max.mat')
scale = np.array(f4['test_max'])
scale = scale[0,0:test_size] # (1,220)

scale_defect_5db = np.amax(abs(test_defect_5db))
scale_defect_10db = np.amax(abs(test_defect_10db))
scale_defect_20db = np.amax(abs(test_defect_20db))
scale_defect_25db = np.amax(abs(test_defect_25db))
scale_defect_30db = np.amax(abs(test_defect_30db))

test_defect_5db = test_defect_5db/scale_defect_5db
test_defect_10db = test_defect_10db/scale_defect_10db
test_defect_20db = test_defect_20db/scale_defect_20db
test_defect_25db = test_defect_25db/scale_defect_25db
test_defect_30db = test_defect_30db/scale_defect_30db

error_len = test_defect_5db.shape[1]-tail 

error_absolute_5db = np.zeros ((test_defect_5db.shape[1], test_size)) 
error_absolute_10db = np.zeros ((test_defect_5db.shape[1], test_size)) 
error_absolute_20db = np.zeros ((test_defect_5db.shape[1], test_size)) 
error_absolute_25db = np.zeros ((test_defect_5db.shape[1], test_size)) 
error_absolute_30db = np.zeros ((test_defect_5db.shape[1], test_size)) 
residual_absolute = np.zeros ((test_defect_5db.shape[1], test_size))

defect_signals_5db = test_defect_5db.flatten()
defect_signals_10db = test_defect_10db.flatten()
defect_signals_20db = test_defect_20db.flatten()
defect_signals_25db = test_defect_25db.flatten()
defect_signals_30db = test_defect_30db.flatten()
df_signals = test_signals.flatten()

test_len =len(defect_signals_5db)- n_h -OUT_STEPS+1

x_defect_5db = np.zeros ((test_len, n_h))
y_defect_5db = np.zeros ((test_len, 1))

x_defect_10db = np.zeros ((test_len, n_h))
y_defect_10db = np.zeros ((test_len, 1))

x_defect_20db = np.zeros ((test_len, n_h))
y_defect_20db = np.zeros ((test_len, 1))

x_defect_25db = np.zeros ((test_len, n_h))
y_defect_25db = np.zeros ((test_len, 1))

x_defect_30db = np.zeros ((test_len, n_h))
y_defect_30db = np.zeros ((test_len, 1))

x_df = np.zeros ((test_len, n_h))
y_df = np.zeros ((test_len, 1))

for j in range(n_h):
    
    x_defect_5db[:,j] = defect_signals_5db[j:j+test_len]
    x_defect_10db[:,j] = defect_signals_10db[j:j+test_len]
    x_defect_20db[:,j] = defect_signals_20db[j:j+test_len]
    x_defect_25db[:,j] = defect_signals_25db[j:j+test_len]
    x_defect_30db[:,j] = defect_signals_30db[j:j+test_len]
    x_df[:,j] = df_signals[j:j+test_len]

y_defect_5db[:,0] = defect_signals_5db[-test_len::]
y_defect_10db[:,0] = defect_signals_10db[-test_len::]
y_defect_20db[:,0] = defect_signals_20db[-test_len::]
y_defect_25db[:,0] = defect_signals_25db[-test_len::]
y_defect_30db[:,0] = defect_signals_30db[-test_len::]
y_df[:,0] = df_signals[-test_len::]

x_defect_5db = tf.convert_to_tensor(x_defect_5db, dtype=tf.float32)
y_defect_5db = tf.convert_to_tensor(y_defect_5db, dtype=tf.float32)
x_defect_10db = tf.convert_to_tensor(x_defect_10db, dtype=tf.float32)
y_defect_10db = tf.convert_to_tensor(y_defect_10db, dtype=tf.float32)
x_defect_20db = tf.convert_to_tensor(x_defect_20db, dtype=tf.float32)
y_defect_20db = tf.convert_to_tensor(y_defect_20db, dtype=tf.float32)
x_defect_25db = tf.convert_to_tensor(x_defect_25db, dtype=tf.float32)
y_defect_25db = tf.convert_to_tensor(y_defect_25db, dtype=tf.float32)
x_defect_30db = tf.convert_to_tensor(x_defect_30db, dtype=tf.float32)
y_defect_30db = tf.convert_to_tensor(y_defect_30db, dtype=tf.float32)
x_df = tf.convert_to_tensor(x_df, dtype=tf.float32)
y_df = tf.convert_to_tensor(y_df, dtype=tf.float32)

predicted_baseline_5db = model.predict(x_defect_5db)
predicted_baseline_10db = model.predict(x_defect_10db)
predicted_baseline_20db = model.predict(x_defect_20db)
predicted_baseline_25db = model.predict(x_defect_25db)
predicted_baseline_30db = model.predict(x_defect_30db)
baseline_df = model.predict(x_df)

df_residual = y_df - baseline_df
df_residual = np.asarray(tf.squeeze(df_residual))
df_residual = df_residual[-(test_size-1)*test_defect_5db.shape[1]:]
df_residual = df_residual - statistics.mean(df_residual)

defect_error_5db = np.asarray(y_defect_5db) - predicted_baseline_5db
defect_error_5db = np.asarray(tf.squeeze(defect_error_5db))
defect_error_5db = defect_error_5db[-(test_size-1)*test_defect_5db.shape[1]:]
defect_error_5db = defect_error_5db - statistics.mean(defect_error_5db)

defect_error_10db = np.asarray(y_defect_10db) - predicted_baseline_10db
defect_error_10db = np.asarray(tf.squeeze(defect_error_10db))
defect_error_10db = defect_error_10db[-(test_size-1)*test_defect_5db.shape[1]:]
defect_error_10db = defect_error_10db - statistics.mean(defect_error_10db)

defect_error_20db = np.asarray(y_defect_20db) - predicted_baseline_20db
defect_error_20db = np.asarray(tf.squeeze(defect_error_20db))
defect_error_20db = defect_error_20db[-(test_size-1)*test_defect_5db.shape[1]:]
defect_error_20db = defect_error_20db - statistics.mean(defect_error_20db)

defect_error_25db = np.asarray(y_defect_25db) - predicted_baseline_25db
defect_error_25db = np.asarray(tf.squeeze(defect_error_25db))
defect_error_25db = defect_error_25db[-(test_size-1)*test_defect_5db.shape[1]:]
defect_error_25db = defect_error_25db - statistics.mean(defect_error_25db)

defect_error_30db = np.asarray(y_defect_30db) - predicted_baseline_30db
defect_error_30db = np.asarray(tf.squeeze(defect_error_30db))
defect_error_30db = defect_error_30db[-(test_size-1)*test_defect_5db.shape[1]:]
defect_error_30db = defect_error_30db - statistics.mean(defect_error_30db)

for k in range(test_size-1):
    error_absolute_5db [:,k] = defect_error_5db[k*test_defect_5db.shape[1]: (k+1)*test_defect_5db.shape[1]] *scale_defect_5db
    error_absolute_5db [:,k] = np.multiply(error_absolute_5db [:,k], scale[k])
    error_absolute_5db [:,k] = abs(error_absolute_5db [:,k])

    error_absolute_10db [:,k] = defect_error_10db[k*test_defect_10db.shape[1]: (k+1)*test_defect_5db.shape[1]] *scale_defect_10db
    error_absolute_10db [:,k] = np.multiply(error_absolute_10db [:,k], scale[k])
    error_absolute_10db [:,k] = abs(error_absolute_10db [:,k])

    error_absolute_20db [:,k] = defect_error_20db[k*test_defect_20db.shape[1]: (k+1)*test_defect_5db.shape[1]] *scale_defect_20db
    error_absolute_20db [:,k] = np.multiply(error_absolute_20db [:,k], scale[k])
    error_absolute_20db [:,k] = abs(error_absolute_20db [:,k])

    error_absolute_25db [:,k] = defect_error_25db[k*test_defect_25db.shape[1]: (k+1)*test_defect_5db.shape[1]] *scale_defect_25db
    error_absolute_25db [:,k] = np.multiply(error_absolute_25db [:,k], scale[k])
    error_absolute_25db [:,k] = abs(error_absolute_25db [:,k])

    error_absolute_30db [:,k] = defect_error_30db[k*test_defect_30db.shape[1]: (k+1)*test_defect_5db.shape[1]] *scale_defect_30db
    error_absolute_30db [:,k] = np.multiply(error_absolute_30db [:,k], scale[k])
    error_absolute_30db [:,k] = abs(error_absolute_30db [:,k])
    
    residual_absolute [:,k] = df_residual[k*test_defect_5db.shape[1]: (k+1)*test_defect_5db.shape[1]] 
    residual_absolute [:,k] = np.multiply(residual_absolute [:,k], scale[k])
    residual_absolute [:,k] = abs(residual_absolute [:,k])

error_absolute_5db = error_absolute_5db [-error_len:-5,:]
error_absolute_10db = error_absolute_10db [-error_len:-5,:]
error_absolute_20db = error_absolute_20db [-error_len:-5,:]
error_absolute_25db = error_absolute_25db [-error_len:-5,:]
error_absolute_30db = error_absolute_30db [-error_len:-5,:]
residual_absolute = residual_absolute [-error_len:-5,:]

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Residual without defect (left) and with defect (right)')
ax1.plot(residual_absolute[:,0])
ax1.set_ylim([0,0.05])
ax2.plot(error_absolute_30db[:,0])
ax2.set_ylim([0,0.05])
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Residual without defect (left) and with defect (right)')
ax1.plot(residual_absolute[:,10])
ax1.set_ylim([0,0.05])
ax2.plot(error_absolute_30db[:,10])
ax2.set_ylim([0,0.05])
plt.show()

max_residual = np.amax(residual_absolute, axis = 0)
max_error_5db = np.amax(error_absolute_5db, axis = 0)
max_error_10db = np.amax(error_absolute_10db, axis = 0)
max_error_20db = np.amax(error_absolute_20db, axis = 0)
max_error_25db = np.amax(error_absolute_25db, axis = 0)
max_error_30db = np.amax(error_absolute_30db, axis = 0)

scores_5db = np.concatenate((max_residual,max_error_5db))
scores_10db = np.concatenate((max_residual,max_error_10db))
scores_20db = np.concatenate((max_residual,max_error_20db))
scores_25db = np.concatenate((max_residual,max_error_25db))
scores_30db = np.concatenate((max_residual,max_error_30db))
label1 = np.zeros((len(max_residual)))
label2 = np.ones((len(max_error_5db)))
label = np.concatenate((label1, label2))
PFA_5db, POD_5db, thresholds_5db = roc_curve(label, scores_5db)
PFA_10db, POD_10db, thresholds_10db = roc_curve(label, scores_10db)
PFA_20db, POD_20db, thresholds_20db = roc_curve(label, scores_20db)
PFA_25db, POD_25db, thresholds_25db = roc_curve(label, scores_25db)
PFA_30db, POD_30db, thresholds_30db = roc_curve(label, scores_30db)

roc_auc_5db = auc(PFA_5db, POD_5db)
roc_auc_10db = auc(PFA_10db, POD_10db)
roc_auc_20db = auc(PFA_20db, POD_20db)
roc_auc_25db = auc(PFA_25db, POD_25db)
roc_auc_30db = auc(PFA_30db, POD_30db)
plt.figure()
plt.plot(PFA_5db, POD_5db, label = '-5dB (area = %0.2f)'%roc_auc_5db)
plt.plot(PFA_10db, POD_10db, label = '-10dB (area = %0.2f)'%roc_auc_10db)
plt.plot(PFA_20db, POD_20db, label = '-20dB (area = %0.2f)'%roc_auc_20db)
plt.plot(PFA_25db, POD_25db, label = '-25dB (area = %0.2f)'%roc_auc_25db)
plt.plot(PFA_30db, POD_30db, label = '-30dB (area = %0.2f)'%roc_auc_30db)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Probability of False Alarm')
plt.ylabel('Probability of Detection')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

