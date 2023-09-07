# NARX-guided-wave
NARX network in Tensorflow for guided wave defect detection

Data from Matlab:
* 'test_defect_*.mat' are test signals with different amplitudes of defects added.
* 'test_normalised.mat' is the set of defect-free test signals normalised to range [-1 1] by each signal's own maximum.
* 'test_max.mat' contains the scale factor for each signal in 'test_normalised.mat'.
* 'train_normalised2.mat' is the normalised training data with range [-1 1].

Code:
* 'NARX_closed.py' trains a NARX model. When OUT_STEPS=1, it is a feedforward network; when OUT_STEPS>=2, it is a closed loop network.
* 'NARXc_test.py' tests the trained models.
* 'NARXc_TrainAndTest.py' combines the above two to track AUC and mse during training.
* Note: this version of NARX is not compatible with Matlab due to a different order of the weights for simpler codes. 
The code in folder 'old version' trains NARX models that have weights interchangeable with Matlab NARX, but they are used as feedforward networks only.

Notebook:
'down_sample.ipynb' contains most up-to-date features such as down-sampling, restore best weights.

Please cite the following paper if you've used this code for your research:

Tu, Xin L., Richard J. Pyle, Anthony J. Croxford, and Paul D. Wilcox. ["Potential and limitations of NARX for defect detection in guided wave signals."](https://journals.sagepub.com/doi/full/10.1177/14759217221113240) Structural Health Monitoring 22, no. 3 (2023): 1863-1875.
