#!/usr/bin/python3

import psutil
import gc

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

import sys
sys.path.append(git_path + '/astrophys/tools')
from tools_gs_par import *
from params import *

sys.path.append(git_path + '/shared')
from inject_tools import *

local = True
# qtrans = True
# qsplit = True
# save = True

### set injection type

# inj_type = 'sg'
# inj_type = 'rd'
# inj_type = 'ga'
# inj_type = 'cg'
# inj_type = 'cg_inc'
# inj_type = 'cg_double'
inj_type = 'wn'

segment_list = get_segment_list('BOTH')
detector = 'L'
files = get_files(detector)

params_path = Path(git_path + '/shared/injection_params')

### choose injection times (either with or without the constraint that the times have no glitch in them)

inj_df = pd.read_csv(join(params_path, 'soft_inj_time.csv'), usecols=['H'])
# inj_df = pd.read_csv(join(params_path, 'soft_inj_time_no_constraint.csv'), usecols=[detector])

sky_loc = pd.read_csv(join(params_path, 'sky_loc_csv.csv'), usecols=['ra', 'dec', 'pol', 'alpha'])

start_time = time.time()

### get chunks that contain the injection times

times_par = []
inj_times = []
chunks = []

# for t_inj in inj_df[detector][:15]: # only take the first 15 for the initial constrained injections
for i, t_inj in enumerate(inj_df['H'][:15]): # only use H injection times, calculate actual delay according to sky_loc (in inject_tools)
	segment = find_segment(t_inj, segment_list)
	# chunk_list = get_chunks(segment[0], segment[1], Tc=Tc, To=To)
	chunk_list = get_chunks(segment[0], segment[1], Tc, To + (max(scales) - min(scales)))
	chunk = find_segment(t_inj, chunk_list)
	chunks.append(find_segment(t_inj, chunk_list))
	inj_times.append(t_inj)
	times_par.append((chunk[0], chunk[1], t_inj, sky_loc['ra'][i], sky_loc['dec'][i], sky_loc['pol'][i], sky_loc['alpha'][i]))

vmem = psutil.virtual_memory()
print('pre-mp', vmem.total >> 20, vmem.available >> 20, vmem.used >> 20, vmem.free >> 20, vmem.percent)

num_cnt = 0

params_dict = {'sg': ['f0', 'Q', 'A'], 
			   'rd': ['f0', 'tau', 'A'], 
			   'ga': ['tau', 'A'], 
			   'cg': ['f0', 'Q', 'A'], 
			   'cg_inc': ['f0', 'Q', 'A'], 
			   'cg_double': ['f0', 'Q', 'A'], 
			   'wn': ['f_low', 'f_high', 'tau', 'A']}

param_keys = params_dict[inj_type]
params = pd.read_csv(join(params_path, inj_type + '_params_csv.csv'), usecols=param_keys)

inj_params = {key: [] for key in param_keys}

for index, row in params.iterrows():
	for key in param_keys:
		inj_params[key] = row[key]
# 
### ugly code right here, should fix sometime. uncomment the injection type you want to use and comment out the rest
# 
# if inj_type == 'sg':
# 	params = pd.read_csv(join(params_path, inj_type + '_params_csv.csv'), usecols=['f0', 'Q', 'A'])
# 	for f_inj, Q, A in zip(params['f0'], params['Q'], params['A']):
# 	# for f_inj, Q, A in zip(params['f0'][num_s:num_e], params['Q'][num_s:num_e], params['A'][num_s:num_e]):
# 		inj_params = {'f0': f_inj, 'Q': Q, 'A': A}
# 
# if inj_type == 'rd':
# 	params = pd.read_csv(join(params_path, inj_type + '_params_csv.csv'), usecols=['f0', 'tau', 'A'])
# 	for f_inj, tau, A in zip(params['f0'], params['tau'], params['A']):
# 	# for f_inj, tau, A in zip(params['f0'][num_s:num_e], params['tau'][num_s:num_e], params['A'][num_s:num_e]):
# 		inj_params = {'f0': f_inj, 'tau': tau, 'A': A}
# 
# if inj_type == 'ga':
# 	params = pd.read_csv(join(params_path, inj_type + '_params_csv.csv'), usecols=['tau', 'A'])
# 	for tau, A in zip(params['tau'], params['A']):
# 	# for tau, A in zip(params['tau'][num_s:num_e], params['A'][num_s:num_e]):
# 		inj_params = {'tau': tau, 'A': A}
# 
# if inj_type == 'cg':
# 	params = pd.read_csv(join(params_path, inj_type + '_params_csv.csv'), usecols=['f0', 'Q', 'A'])
# 	for f_inj, Q, A in zip(params['f0'], params['Q'], params['A']):
# 	# for f_inj, Q, A in zip(params['f0'][num_s:num_e], params['Q'][num_s:num_e], params['A'][num_s:num_e]):
# 		inj_params = {'f0': f_inj, 'Q': Q, 'A': A}
# 
# if inj_type == 'cg_inc':
# 	params = pd.read_csv(join(params_path, inj_type + '_params_csv.csv'), usecols=['f0', 'Q', 'A'])
# 	# for f_inj, Q, A in zip(params['f0'], params['Q'], params['A']):
# 	for f_inj, Q, A in zip(params['f0'][num_s:num_e], params['Q'][num_s:num_e], params['A'][num_s:num_e]):
# 		inj_params = {'f0': f_inj, 'Q': Q, 'A': A}
# 
# if inj_type == 'cg_double':
# 	params = pd.read_csv(join(params_path, inj_type + '_params_csv.csv'), usecols=['f0', 'Q', 'A'])
# 	for f_inj, Q, A in zip(params['f0'], params['Q'], params['A']):
# 	# for f_inj, Q, A in zip(params['f0'][num_s:num_e], params['Q'][num_s:num_e], params['A'][num_s:num_e]):
# 		inj_params = {'f0': f_inj, 'Q': Q, 'A': A}
# 
# if inj_type == 'wn':
# 	params = pd.read_csv(join(params_path, inj_type + '_params_csv.csv'), usecols=['f_low', 'f_high', 'tau', 'A'])
# 	for f_low, f_high, tau, A in zip(params['f_low'], params['f_high'], params['tau'], params['A']):
# 	# for f_low, f_high, tau, A in zip(params['f_low'][num_s:num_e], params['f_high'][num_s:num_e], params['tau'][num_s:num_e], \
# 	# 	params['A'][num_s:num_e]):
# 		inj_params = {'f_low': f_low, 'f_high': f_high, 'tau': tau, 'A': A}
# 
	pool = mp.Pool(15)
	results = pool.starmap(load_inject_condition_multi_scale, 
						   [(t[0], t[1], t[2], t[3], t[4], t[5], t[6], inj_type, inj_params, \
						   	local, Tc, To, fw, window, detector, input_shape, scales, frange, qrange) for t in times_par])

	vmem = psutil.virtual_memory()
	print(str(num_cnt) + ' pre-pool.close', vmem.total >> 20, vmem.available >> 20, vmem.used >> 20, vmem.free >> 20, vmem.percent)
	
	pool.close()
	pool.join()

	x = []
	times = []
	for result in results:
		if result is None:
			sys.exit('not enough available memory')

		x.append(result[0])
		times.append(result[1])

	vmem = psutil.virtual_memory()
	print(str(num_cnt) + ' post-pool.close', vmem.total >> 20, vmem.available >> 20, vmem.used >> 20, vmem.free >> 20, vmem.percent)

	del results
	gc.collect()

	x = np.asarray(x)
	times = np.asarray(times)

	data_path = Path('/arch/tommaria/data/multi_scale/conditioned_data/16KHZ/' + detector + '1/injected/' + inj_type)
	if not exists(data_path):
		makedirs(data_path)

	if inj_type == 'sg':
		fname = 'injected-' + inj_type + '-A-' + str(int(round(row['A'] / 1e-22))).zfill(4) + 'e-22-f0-'  + \
				str(row['f0']).zfill(4) + '-Q-' + str(row['Q']).zfill(4) + '.hdf5'

	elif inj_type == 'ga':
		fname = 'injected-' + inj_type + '-A-' + str(int(round(row['A'] / 1e-22))).zfill(4) + 'e-22-tau-' + \
				str(int(round(row['tau'] * 1e4))).zfill(5) + 'e-4.hdf5'

	elif inj_type == 'rd':
		fname = 'injected-' + inj_type + '-A-' + str(int(round(row['A'] / 1e-22))).zfill(4) + 'e-22-f0-'  + \
				str(row['f0']).zfill(4) + '-tau-' + str(int(round(row['tau'] * 1e3))).zfill(4) + 'e-3.hdf5'

	elif inj_type == 'cg' or inj_type == 'cg_inc':
		fname = 'injected-' + inj_type + '-A-' + str(int(round(row['A'] / 1e-22))).zfill(4) + 'e-22-f0-'  + \
				str(row['f0']).zfill(4) + '-Q-' + str(row['Q']).zfill(4) + '.hdf5'

	elif inj_type == 'cg_double':
		fname = 'injected-' + inj_type + '-A-' + str(int(round(row['A'] / 1e-23))).zfill(4) + 'e-23-f0-'  + \
				str(row['f0']).zfill(4) + '-Q-' + str(row['Q']).zfill(4) + '.hdf5'

	elif inj_type == 'wn':
		fname = 'injected-' + inj_type + '-A-' + str(int(round(row['A'] / 1e-22))).zfill(4) + 'e-22-f_low-'  + \
				str(row['f_low']).zfill(4) + '-f_high-' + str(row['f_high']).zfill(4) + '-tau-' + \
				str(int(round(row['tau'] * 1e3))).zfill(4) + 'e-3.hdf5'

	with h5py.File(join(data_path, fname), 'w') as f:
		f.create_dataset('x', data=x)
		f.create_dataset('times', data=times)

	print(fname)
	print(x.shape)
	print(times.shape)

	del x, times
	gc.collect()

	num_cnt += 1

print(detector)
print('Done')
print("--- Execution time is %.7s minutes ---\n" % ((time.time() - start_time) / 60))

