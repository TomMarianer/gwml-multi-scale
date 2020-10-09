#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:39:36 2019

@author: tommarianer

script for generating spectrograms of the gravityspy glitch dataset from raw strain data
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

import matplotlib
matplotlib.use('TkAgg')

import sys
# sys.path.append('/storage/home/tommaria/thesis/tools')
sys.path.append(git_path + '/astrophys/tools')
from tools_gs import *
from params import *
import pandas as pd

metadata = pd.read_csv('trainingset_v1d1_metadata.csv', 
					   usecols=['event_time', 'ifo', 'label', 'gravityspy_id'])

dfidxs = metadata.index

scales = [0.5, 1.0, 2.0]
shifts = np.linspace(-min(scales)/2, min(scales)/2, 21)

x = []
y = []
times = []
idx_start = 0
idx_stop  = np.min([len(dfidxs), idx_start + 10])

start_time = time.time()
for dfidx in tqdm(dfidxs[idx_start:idx_stop]):
	t_i = (metadata['event_time'][dfidx]) - Tc/2
	t_f = (metadata['event_time'][dfidx]) + Tc/2
	detector = metadata['ifo'][dfidx][0]
	data = condition_chunks(t_i, t_f, Tc=Tc, To=To, local=True, fw=fw, qtrans=False, 
							qsplit=False, dT=dT, detector=detector, save=False)[0]
	qt = data.q_transform(frange=(10, 2048), qrange=(4, 100), whiten=True, tres=min(scales)/input_shape[0], 
						  logf=True, fres=input_shape[1])
	x_shifts = []
	for shift in shifts:
		x_scales = []
		for scale in scales:
			x_scales.append(qt.crop(metadata['event_time'][dfidx] - scale / 2 + shifts[shift_idx], 
			metadata['event_time'][dfidx] + scale / 2 + shifts[shift_idx])[::int(scale/min(scales))])
		x_shifts.append(x_scales)

	x.append(x_shifts)
	y.append(metadata['label'][dfidx])
	times.append(metadata['event_time'][dfidx])

	# x = np.asarray(x)
	print(np.asarray(x).shape)
	print(type(x[0][0][0]))

import h5py

y = [item.encode('ascii') for item in y]

data_path = Path('/storage/fast/users/tommaria/data/new_search/training')

with h5py.File(join(data_path, 'trainingset_' + str(idx_start).zfill(4) + '_' + str(idx_stop).zfill(4) + '.hdf5'), 'w') as f:
	f.create_dataset('x', data=x)
	f.create_dataset('y', data=y)
	f.create_dataset('times', data=times)

print('Done')
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
