#!/usr/bin/env python3
"""
script for conditioning raw strain data and generating spectrograms
parallel version
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
from tools_gs_par import *
from params import *

segment_list = get_segment_list('BOTH')
detector = 'H'

files = get_files(detector)
scales = [0.5, 1.0, 2.0]
input_shape = (299, 299)
local = True

frange=(10, 2048)
qrange=(4, 100)

start_time = time.time()
# H1 segments - 
# L1 segments - 
for seg_num in np.arange(257, 300):
	segment = segment_list[seg_num]

	t_i = segment[0]
	t_f = segment[1]

	data_path = Path('/storage/fast/users/tommaria/data/multi_scale/conditioned_data/16KHZ/' + detector + 
	                 '1/segment-' + str(t_i) + '-' + str(t_f))

	print(seg_num)
	print(data_path)

	chunks = get_chunks(t_i, t_f, Tc, To + (max(scales) - min(scales)))

	pool = mp.Pool(mp.cpu_count() - 1)
	results = pool.starmap(load_condition_multi_scale, [(chunk[0], chunk[1], local, Tc, To, fw, window, detector, \
														 input_shape, scales, frange, qrange, data_path) for chunk in chunks])
	
	pool.close()
	pool.join()

	for result in results:
		if result == 'not enough available memory':
			sys.exit('not enough available memory')


print('Done')
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))
