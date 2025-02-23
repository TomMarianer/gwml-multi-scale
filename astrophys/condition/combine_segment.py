#!/usr/bin/env python3
"""
script for combining the conditioned spectrogram files
used after conditioning using the condition_raw.py (or the condition_raw_par.py) script
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

segment_list = get_segment_list('BOTH')
detector = 'L'
print(detector)

# H1 segments - 
# L1 segments - 

start_time = time.time()

# for segment in segment_list[40:80]:
for seg_num in np.asarray([326, 548, 587, 811, 877, 921, 1117, 1265, 1369, 1580, 1587, 1633, 1650, 1659, 1669, 1687]):
	segment = segment_list[seg_num]

	print(seg_num, segment)

	t_i = segment[0]
	t_f = segment[1]

	data_path = Path('/storage/fast/users/tommaria/data/multi_scale/conditioned_data/16KHZ/' + detector + 
	                 '1/segment-' + str(t_i) + '-' + str(t_f))

	files = [join(data_path, f) for f in sorted(listdir(data_path)) if isfile(join(data_path, f))]

	x = []
	times = []

	for file in files:
		with h5py.File(join(data_path,file), 'r') as f:
			x.extend(list(f['x']))
			times.extend(list(f['times']))

	# x = np.asarray(x)
	# times = np.asarray(times)

	data_path = Path('/arch/tommaria/data/multi_scale/conditioned_data/16KHZ/' + detector + '1/combined')
	if not exists(data_path):
		makedirs(data_path)

	with h5py.File(join(data_path, 'segment-' + str(t_i) + '-' + str(t_f) + '.hdf5'), 'w') as f:
		f.create_dataset('x', data=x)
		f.create_dataset('times', data=times)

	print(np.asarray(x).shape)
	# print(type(x[0][0][0]))
	print(np.asarray(times).shape)

print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
