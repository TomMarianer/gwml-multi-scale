#!/usr/bin/python
"""
script for splitting the dataset into training test and validation sets
as well as augmenting the training set
"""

import psutil
from os import listdir
from os.path import isfile, join
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
import time

start_time = time.time()

input_shape = (299, 299)
scales = [0.5, 1.0, 2.0]
shifts = np.linspace(-min(scales)/2, min(scales)/2, 21)
no_shift_idx = int(len(shifts) / 2)

data_path = Path('/storage/fast/users/tommaria/data/multi_scale/training/combined')
with h5py.File(join(data_path, 'trainingset.hdf5'), 'r') as f:
	x = np.asarray(f['x'])
	y = [item.decode('ascii') for item in list(f['y'])]
	times = np.asarray(f['times'])

print(np.asarray(x).shape)
print(type(x[0][0][0]))
print(np.asarray(y).shape)
print(np.asarray(times).shape)

temp = []

for jdx, val in enumerate(y):
	temp.append([val, times[jdx]])

y = np.asarray(temp)

x_train_aug = []
y_train = []
x_test = []
y_test = []
x_val = []
y_val = []

for label in np.unique(y[:,0]):
	vmem = psutil.virtual_memory()
	avail_mem = vmem.available >> 20
	if avail_mem < 3e5:
		print(dfidx, vmem.total >> 20, vmem.available >> 20, vmem.used >> 20, vmem.free >> 20, vmem.percent)
		print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
		sys.exit('not enough available memory')

	idx = y[:,0] == label
	x_temp, x_test_temp, y_temp, y_test_temp = train_test_split(x[idx], y[idx], test_size=0.2, random_state=0)
	if np.asarray(x_temp).shape[0] == 1:
		x_train_temp = x_temp
		y_train_temp = y_temp
		x_val_temp = []
		y_val_temp = []

	else:
		x_train_temp, x_val_temp, y_train_temp, y_val_temp = train_test_split(x_temp, y_temp, test_size=0.1, random_state=0)

	x_train_aug.extend(x_train_temp)
	y_train.extend(y_train_temp)
	x_test.extend([sample[no_shift_idx] for sample in x_test_temp])
	y_test.extend(y_test_temp)
	x_val.extend([sample[no_shift_idx] for sample in x_val_temp])
	y_val.extend(y_val_temp)
	
del x, y, times, temp

y_train = np.asarray(y_train)
times_train = y_train[:,1].astype('float64')
y_train = y_train[:,0]

y_test = np.asarray(y_test)
times_test = y_test[:,1].astype('float64')
y_test = y_test[:,0]

y_val = np.asarray(y_val)
times_val = y_val[:,1].astype('float64')
y_val = y_val[:,0]

print(np.asarray(x_test).shape)
print(type(x_test[0][0]))
print(len(y_test))
print(times_test.shape)

print(np.asarray(x_val).shape)

data_path = Path('/storage/fast/users/tommaria/data/multi_scale/training/combined')
with h5py.File(join(data_path, 'dataset_test_val.hdf5'), 'w') as f:
	f.create_dataset('x_test', data=x_test)
	f.create_dataset('x_val', data=x_val)
	f.create_dataset('y_test', data=[item.encode('ascii') for item in y_test])
	f.create_dataset('y_val', data=[item.encode('ascii') for item in y_val])
	f.create_dataset('times_test', data=times_test)
	f.create_dataset('times_val', data=times_val)

del x_test, y_test, times_test, x_val, y_val, times_val

x_train = []
x_aug = []
y_aug = []

for sample, label in zip(x_train_aug, y_train):
	x_train.append(sample[no_shift_idx])
	x_aug.extend(sample[:no_shift_idx])
	y_aug.extend([label] * len(sample[:no_shift_idx]))
	x_aug.extend(sample[no_shift_idx:])
	y_aug.extend([label] * len(sample[no_shift_idx:]))

del x_train_aug

print(np.asarray(x_train).shape)
print(type(x_train[0][0]))
print(len(y_train))
print(times_train.shape)

print(np.asarray(x_aug).shape)
print(type(x_aug[0][0]))
print(len(y_aug))

data_path = Path('/storage/fast/users/tommaria/data/multi_scale/training/combined')
with h5py.File(join(data_path, 'dataset_train.hdf5'), 'w') as f:
	f.create_dataset('x_train', data=x_train)
	f.create_dataset('y_train', data=[item.encode('ascii') for item in y_train])
	f.create_dataset('times_train', data=times_train)

data_path = Path('/storage/fast/users/tommaria/data/multi_scale/training/combined')
with h5py.File(join(data_path, 'dataset_aug.hdf5'), 'w') as f:
	f.create_dataset('x_aug', data=x_aug)
	f.create_dataset('y_aug', data=[item.encode('ascii') for item in y_aug])

print('Done')
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
