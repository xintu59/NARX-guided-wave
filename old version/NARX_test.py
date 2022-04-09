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
from collections import deque
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

n_u = 20
n_l = 4
n_h = n_u + n_l
hidden_unit = 20
test_size = 220

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

model = tf.keras.models.load_model('NARX36')
#model.summary()
error_len = test_defect_5db.shape[1]-30 # 3850

error_absolute_5db = np.zeros ((error_len, test_size)) # (3850,220)
error_absolute_10db = np.zeros ((error_len, test_size)) 
error_absolute_20db = np.zeros ((error_len, test_size)) 
error_absolute_25db = np.zeros ((error_len, test_size)) 
error_absolute_30db = np.zeros ((error_len, test_size)) 
residual_absolute = np.zeros ((error_len, test_size))

defect_signals_5db = test_defect_5db.flatten()
defect_signals_5db = defect_signals_5db[:, np.newaxis] # (853600, 1)

defect_signals_10db = test_defect_10db.flatten()
defect_signals_10db = defect_signals_10db[:, np.newaxis] 

defect_signals_20db = test_defect_20db.flatten()
defect_signals_20db = defect_signals_20db[:, np.newaxis] 

defect_signals_25db = test_defect_25db.flatten()
defect_signals_25db = defect_signals_25db[:, np.newaxis]

defect_signals_30db = test_defect_30db.flatten()
defect_signals_30db = defect_signals_30db[:, np.newaxis]

df_signals = test_signals.flatten()
df_signals = df_signals[:, np.newaxis]

x_defect_5db = np.zeros ((len(defect_signals_5db)-n_h, n_h))
y_defect_5db = np.zeros ((len(defect_signals_5db)-n_h, 1))

x_defect_10db = np.zeros ((len(defect_signals_5db)-n_h, n_h))
y_defect_10db = np.zeros ((len(defect_signals_5db)-n_h, 1))

x_defect_20db = np.zeros ((len(defect_signals_5db)-n_h, n_h))
y_defect_20db = np.zeros ((len(defect_signals_5db)-n_h, 1))

x_defect_25db = np.zeros ((len(defect_signals_5db)-n_h, n_h))
y_defect_25db = np.zeros ((len(defect_signals_5db)-n_h, 1))

x_defect_30db = np.zeros ((len(defect_signals_5db)-n_h, n_h))
y_defect_30db = np.zeros ((len(defect_signals_5db)-n_h, 1))

x_df = np.zeros ((len(defect_signals_5db)-n_h, n_h))
y_df = np.zeros ((len(defect_signals_5db)-n_h, 1))

for j in range(len(defect_signals_5db)-n_h):

    x_defect_5db[j,:] = defect_signals_5db[j:(j+n_h),0]
    y_defect_5db[j,0] = defect_signals_5db[(j+n_h),0]
  

    x_defect_10db[j,:] = defect_signals_10db[j:(j+n_h),0]
    y_defect_10db[j,0] = defect_signals_10db[(j+n_h),0]

    x_defect_20db[j,:] = defect_signals_20db[j:(j+n_h),0]
    y_defect_20db[j,0] = defect_signals_20db[(j+n_h),0]

    x_defect_25db[j,:] = defect_signals_25db[j:(j+n_h),0]
    y_defect_25db[j,0] = defect_signals_25db[(j+n_h),0]

    x_defect_30db[j,:] = defect_signals_30db[j:(j+n_h),0]
    y_defect_30db[j,0] = defect_signals_30db[(j+n_h),0]

    x_df[j,:] = df_signals[j:(j+n_h),0]
    y_df[j,0] = df_signals[(j+n_h),0]

x_defect_in1_5db = deque()
x_defect_in2_5db = deque()

x_defect_in1_10db = deque()
x_defect_in2_10db = deque()

x_defect_in1_20db = deque()
x_defect_in2_20db = deque()

x_defect_in1_25db = deque()
x_defect_in2_25db = deque()

x_defect_in1_30db = deque()
x_defect_in2_30db = deque()
x_df_in1 = deque()
x_df_in2 = deque()

for j in range(n_l):
    split_defect_5db = x_defect_5db[:,j:j+n_u]
    x_defect_in1_5db.appendleft(split_defect_5db)
    extra_defect_value_5db = x_defect_5db[:,j+n_u]
    x_defect_in2_5db.appendleft(extra_defect_value_5db)

    split_defect_10db = x_defect_10db[:,j:j+n_u]
    x_defect_in1_10db.appendleft(split_defect_10db)
    extra_defect_value_10db = x_defect_10db[:,j+n_u]
    x_defect_in2_10db.appendleft(extra_defect_value_10db)

    split_defect_20db = x_defect_20db[:,j:j+n_u]
    x_defect_in1_20db.appendleft(split_defect_20db)
    extra_defect_value_20db = x_defect_20db[:,j+n_u]
    x_defect_in2_20db.appendleft(extra_defect_value_20db)

    split_defect_25db = x_defect_25db[:,j:j+n_u]
    x_defect_in1_25db.appendleft(split_defect_25db)
    extra_defect_value_25db = x_defect_25db[:,j+n_u]
    x_defect_in2_25db.appendleft(extra_defect_value_25db)

    split_defect_30db = x_defect_30db[:,j:j+n_u]
    x_defect_in1_30db.appendleft(split_defect_30db)
    extra_defect_value_30db = x_defect_30db[:,j+n_u]
    x_defect_in2_30db.appendleft(extra_defect_value_30db)

    split_df = x_df[:,j:j+n_u]
    x_df_in1.appendleft(split_df)
    extra_df_value = x_df[:,j+n_u]
    x_df_in2.appendleft(extra_df_value)

x_defect_in1_5db = np.asarray(x_defect_in1_5db)
x_defect_in2_5db = np.asarray(x_defect_in2_5db)

x_defect_in1_10db = np.asarray(x_defect_in1_10db)
x_defect_in2_10db = np.asarray(x_defect_in2_10db)

x_defect_in1_20db = np.asarray(x_defect_in1_20db)
x_defect_in2_20db = np.asarray(x_defect_in2_20db)

x_defect_in1_25db = np.asarray(x_defect_in1_25db)
x_defect_in2_25db = np.asarray(x_defect_in2_25db)

x_defect_in1_30db = np.asarray(x_defect_in1_30db)
x_defect_in2_30db = np.asarray(x_defect_in2_30db)
x_df_in1 = np.asarray(x_df_in1)
x_df_in2 = np.asarray(x_df_in2)

x_defect_in1_5db = np.concatenate(x_defect_in1_5db, axis = 1)
x_defect_in2_5db = np.transpose(x_defect_in2_5db)
x_defect_in_5db = np.concatenate((x_defect_in1_5db,x_defect_in2_5db), axis=1)

x_defect_in1_10db = np.concatenate(x_defect_in1_10db, axis = 1)
x_defect_in2_10db = np.transpose(x_defect_in2_10db)
x_defect_in_10db = np.concatenate((x_defect_in1_10db,x_defect_in2_10db), axis=1)

x_defect_in1_20db = np.concatenate(x_defect_in1_20db, axis = 1)
x_defect_in2_20db = np.transpose(x_defect_in2_20db)
x_defect_in_20db = np.concatenate((x_defect_in1_20db,x_defect_in2_20db), axis=1)

x_defect_in1_25db = np.concatenate(x_defect_in1_25db, axis = 1)
x_defect_in2_25db = np.transpose(x_defect_in2_25db)
x_defect_in_25db = np.concatenate((x_defect_in1_25db,x_defect_in2_25db), axis=1)

x_defect_in1_30db = np.concatenate(x_defect_in1_30db, axis = 1)
x_defect_in2_30db = np.transpose(x_defect_in2_30db)
x_defect_in_30db = np.concatenate((x_defect_in1_30db,x_defect_in2_30db), axis=1)

x_df_in1 = np.concatenate(x_df_in1, axis = 1)
x_df_in2 = np.transpose(x_df_in2)
x_df_in = np.concatenate((x_df_in1,x_df_in2), axis=1)

predicted_baseline_5db = model.predict(x_defect_in_5db)
predicted_baseline_10db = model.predict(x_defect_in_10db)
predicted_baseline_20db = model.predict(x_defect_in_20db)
predicted_baseline_25db = model.predict(x_defect_in_25db)
predicted_baseline_30db = model.predict(x_defect_in_30db)
baseline_df = model.predict(x_df_in)
df_residual = y_df[:,0] - baseline_df[:,0]
df_residual = df_residual - statistics.mean(df_residual)
 
defect_error_5db = y_defect_5db[:,0] - predicted_baseline_5db[:,0]
defect_error_5db = defect_error_5db - statistics.mean(defect_error_5db)

defect_error_10db = y_defect_10db[:,0] - predicted_baseline_10db[:,0]
defect_error_10db = defect_error_10db - statistics.mean(defect_error_10db)

defect_error_20db = y_defect_20db[:,0] - predicted_baseline_20db[:,0]
defect_error_20db = defect_error_20db - statistics.mean(defect_error_20db)

defect_error_25db = y_defect_25db[:,0] - predicted_baseline_25db[:,0]
defect_error_25db = defect_error_25db - statistics.mean(defect_error_25db)

defect_error_30db = y_defect_30db[:,0] - predicted_baseline_30db[:,0]
defect_error_30db = defect_error_30db - statistics.mean(defect_error_30db)

for k in range(test_size):
    error_absolute_5db [:,k] = defect_error_5db[k*test_defect_5db.shape[1]: (k+1)*test_defect_5db.shape[1]-30] *scale_defect_5db
    error_absolute_5db [:,k] = np.multiply(error_absolute_5db [:,k], scale[k])
    error_absolute_5db [:,k] = abs(error_absolute_5db [:,k])

    error_absolute_10db [:,k] = defect_error_10db[k*test_defect_10db.shape[1]: (k+1)*test_defect_10db.shape[1]-30] *scale_defect_10db
    error_absolute_10db [:,k] = np.multiply(error_absolute_10db [:,k], scale[k])
    error_absolute_10db [:,k] = abs(error_absolute_10db [:,k])

    error_absolute_20db [:,k] = defect_error_20db[k*test_defect_20db.shape[1]: (k+1)*test_defect_20db.shape[1]-30] *scale_defect_20db
    error_absolute_20db [:,k] = np.multiply(error_absolute_20db [:,k], scale[k])
    error_absolute_20db [:,k] = abs(error_absolute_20db [:,k])

    error_absolute_25db [:,k] = defect_error_25db[k*test_defect_25db.shape[1]: (k+1)*test_defect_25db.shape[1]-30] *scale_defect_25db
    error_absolute_25db [:,k] = np.multiply(error_absolute_25db [:,k], scale[k])
    error_absolute_25db [:,k] = abs(error_absolute_25db [:,k])

    error_absolute_30db [:,k] = defect_error_30db[k*test_defect_30db.shape[1]: (k+1)*test_defect_30db.shape[1]-30] *scale_defect_30db
    error_absolute_30db [:,k] = np.multiply(error_absolute_30db [:,k], scale[k])
    error_absolute_30db [:,k] = abs(error_absolute_30db [:,k])
    
    residual_absolute [:,k] = df_residual[k*test_defect_5db.shape[1]: (k+1)*test_defect_5db.shape[1]-30] 
    residual_absolute [:,k] = np.multiply(residual_absolute [:,k], scale[k])
    residual_absolute [:,k] = abs(residual_absolute [:,k])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Residual without defect (left) and with defect (right)')
ax1.plot(residual_absolute[:,0])
ax1.set_ylim([0,0.005])
ax2.plot(error_absolute_20db[:,0])
ax2.set_ylim([0,0.005])
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Residual without defect (left) and with defect (right)')
ax1.plot(residual_absolute[:,10])
ax1.set_ylim([0,0.005])
ax2.plot(error_absolute_20db[:,10])
ax2.set_ylim([0,0.005])
plt.show()

max_residual = np.amax(residual_absolute, axis = 0)
max_error_5db = np.amax(error_absolute_5db, axis = 0)
max_error_10db = np.amax(error_absolute_10db, axis = 0)
max_error_20db = np.amax(error_absolute_20db, axis = 0)
max_error_25db = np.amax(error_absolute_25db, axis = 0)
max_error_30db = np.amax(error_absolute_30db, axis = 0)
#print(max_residual.shape)
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
#print(thresholds)
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
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Probability of False Alarm')
plt.ylabel('Probability of Detection')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
