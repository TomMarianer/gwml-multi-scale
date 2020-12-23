#!/usr/bin/python
"""
parameters file for local
"""

from pathlib import Path

scales = [0.5, 1.0, 2.0]
input_shape = (299, 299)

frange=(10, 2048)
qrange=(4, 100)

### model parameters

condition_method = None #'gs_wrap_no_ty'
model = None #'new/resnet152v2/adadelta/gpu_test_15'

### kde parameters

# map grid ranges
# x_range = (-16, 22) 
# y_range = (-25, 21.5)

x_range = (-11, 24)
y_range = (-18, 21)

# estimator parameters
kernel = 'gaussian'
bandwidth = 0.3

### thresholds

kde_th = -11.5
gram_th = 5100

### paths

features_path = Path('/Users/tommarianer/LOSC Data/multi_scale/features')
conditioned_path = Path('/Users/tommarianer/LOSC Data/multi_scale/conditioned_data')