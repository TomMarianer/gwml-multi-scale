#!/usr/bin/python
"""
script for training a cnn on the spectrograms generated of the gravityspy glitch dataset 
using the transfer learning method
"""

import git
from os import listdir
from os.path import isfile, join, dirname, realpath

def get_git_root(path):
	"""Get git root path
	"""
	git_repo = git.Repo(path, search_parent_directories=True)
	git_root = git_repo.git.rev_parse("--show-toplevel")
	return git_root

file_path = dirname(realpath(__file__))
git_path = get_git_root(file_path)

import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from skimage import io
from skimage.transform import resize
import numpy as np
import h5py
from pathlib import Path
import time
import keras
from keras.layers import Flatten, Dense, Dropout, Input, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(git_path + '/power/tools')
from gsparams import *

data_path = Path('/dovilabfs/work/tommaria/gw/data/multi_scale/training')

with h5py.File(join(data_path, 'dataset_train.hdf5'), 'r') as f:
	x_train = np.asarray(f['x_train'])
	y_train = [item.decode('ascii') for item in list(f['y_train'])]

with h5py.File(join(data_path, 'dataset_aug.hdf5'), 'r') as f:
	x_aug = np.asarray(f['x_aug'])
	y_aug = [item.decode('ascii') for item in list(f['y_aug'])]

with h5py.File(join(data_path, 'dataset_test_val.hdf5'), 'r') as f:
	x_val = np.asarray(f['x_val'])
	y_val = [item.decode('ascii') for item in list(f['y_val'])]
	# x_test = np.asarray(f['x_test'])
	# y_test = [item.decode('ascii') for item in list(f['y_test'])]

# encode labels to integers
encoder = LabelEncoder().fit(y_train)
y_train = encoder.transform(y_train)
# y_test = encoder.transform(y_test)
y_val = encoder.transform(y_val)
y_aug = encoder.transform(y_aug)

# one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_aug = keras.utils.to_categorical(y_aug, num_classes)

del encoder

if x_train.shape[1] < 4:
	x_train = np.transpose(np.asarray(x_train), [0, 3, 2, 1])
	# x_test = np.transpose(np.asarray(x_test), [0, 3, 2, 1])
	x_val = np.transpose(np.asarray(x_val), [0, 3, 2, 1])
	x_aug = np.transpose(np.asarray(x_aug), [0, 3, 2, 1])

# # resize images if required
# if x_train.shape[1:3] != input_size:
# 	x_train = np.asarray([resize(img, input_size, mode='reflect') for img in x_train])
# 	# x_test = np.asarray([resize(img, input_size, mode='reflect') for img in x_test])
# 	x_val = np.asarray([resize(img, input_size, mode='reflect') for img in x_val])
# 	x_aug = np.asarray([resize(img, input_size, mode='reflect') for img in x_aug])

x_min = x_train.min()
# x_train -= x_min
# x_test -= x_min
# x_val -= x_min
# x_aug -= x_min

x_max = x_train.max()
# x_train /= x_max

# # x_test = (x_test - x_min) / x_max
# x_val = (x_val - x_min) / x_max
# # x_aug = (x_aug - x_min) / x_max

# x_temp = x_train

vmax = 60 #1000 #255
vmin = -10
# vmax = np.percentile(np.amax(x_train, axis=(1, 2, 3)), 95)
# vmin = np.percentile(np.amin(x_train, axis=(1, 2, 3)), 5)

x_train[x_train > vmax] = vmax
x_train[x_train < vmin] = vmin
x_train = (x_train - vmin) / (vmax - vmin)

# x_test[x_test > vmax] = vmax
# x_test[x_test < vmin] = vmin
# x_test = (x_test - vmin) / (vmax - vmin)

x_val[x_val > vmax] = vmax
x_val[x_val < vmin] = vmin
x_val = (x_val - vmin) / (vmax - vmin)

x_aug[x_aug > vmax] = vmax
x_aug[x_aug < vmin] = vmin
x_aug = (x_aug - vmin) / (vmax - vmin)

x_ta = np.append(x_train, x_aug, axis=0)
y_ta = np.append(y_train, y_aug, axis=0)
# x_ta = x_train
# y_ta = y_train

input_shape = x_train[0].shape

del x_train, y_train, x_aug, y_aug

# Define network

if extract_model == 'vgg16':
	from keras.applications.vgg16 import VGG16
	trained_model = VGG16(include_top=False, input_shape=input_shape)
elif extract_model == 'inceptionv3':
	from keras.applications.inception_v3 import InceptionV3
	trained_model = InceptionV3(include_top=False, input_shape=input_shape)
elif extract_model == 'xception':
	from keras.applications.xception import Xception
	trained_model = Xception(include_top=False, input_shape=input_shape)
elif extract_model == 'resnet152v2':
	from keras.applications.resnet_v2 import ResNet152V2
	trained_model = ResNet152V2(include_top=False, input_shape=input_shape)
elif extract_model == 'inception_resnetv2':
	from keras.applications.inception_resnet_v2 import InceptionResNetV2
	trained_model = InceptionResNetV2(include_top=False, input_shape=input_shape)

output = trained_model.layers[-1].output
output = GlobalMaxPooling2D()(output)
output = Dense(100, activation='relu')(output)
output = Dense(num_classes, activation='softmax')(output)
model = Model(input=trained_model.input, output=output)
# model.summary()

# model_name = 'no_aug_truncated_alt2'

model_path = Path('/dovilabfs/work/tommaria/gw/data/multi_scale/models/' + extract_model + '/' + model_name)
weights_file = join(model_path, model_name + '.weights.best.hdf5')
if not os.path.exists(model_path):
    os.makedirs(model_path)

print('Model: ' + opt_method + '/' + extract_model + '/' + model_name + '\n')

# Compile the model

optimizer = Adagrad(lr=1e-4, epsilon=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_method, metrics=['accuracy'])

# Train the model

checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, 
                               save_best_only=True, monitor='val_accuracy')#'val_loss')
progbar = TQDMCallback()

batch_size = 16 #32
epochs = 20

start_time = time.time()

oldStdout = sys.stdout
file = open(join(model_path, 'logFile'), 'w')
sys.stdout = file

# print(x_min, x_max)
# print(vmax, vmin)
# print(x_ta.min(), x_ta.max())
# print(np.amax(x_temp, axis=(1, 2, 3)).shape)
# print(np.percentile(np.amax(x_temp, axis=(1, 2, 3)), [95, 90, 85, 80, 75]))
# print(np.percentile(np.amin(x_temp, axis=(1, 2, 3)), [5, 10, 15, 20, 25]))

model.fit(x_ta, y_ta, batch_size=batch_size, epochs=epochs, 
          callbacks=[checkpointer, progbar], validation_data=(x_val, y_val), verbose=2, shuffle=True)

print("--- Network training time is %.7s minutes ---\n" % ((time.time() - start_time) / 60.0))

model.save(join(model_path, model_name + '.h5'))

print(x_ta.shape)
# print(x_test.shape)
print(x_val.shape)

file.close()
sys.stdout = oldStdout