Data from Matlab:
'test_defect_*.mat' are test signals with different amplitudes of defects added.
'test_normalised.mat' is the set of defect-free test signals normalised to range [-1 1] by each signal's own maximum.
'test_max.mat' contains the scale factor for each signal in 'test_normalised.mat'.
'train_normalised2.mat' is the normalised training data with range [-1 1].
Code:
'NARXc_closed.py' trains a NARX model. When OUT_STEPS=1, it is a feedforward network; when OUT_STEPS>=2, it is a closed loop network.
'NARXc_test.py' tests the trained models.
'NARXc_TrainAndTest.py' combines the above two to track AUC and mse during training.
Note: this version of code is not compatible with Matlab. 
The code in folder 'old version' trains NARX models that have weights interchangeable with Matlab MARX, but they are feedforward networks only.
