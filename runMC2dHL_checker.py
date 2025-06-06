import numpy as np
from time import time
from storage import npz_file_finder
import json
from MCclasses import HopfieldMC as hop
import sys

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

idx_s = 0
idx_x = 12
idx_y = 0

x_arg = 'lmb'
x_values = np.linspace(start = 0, stop = 0.5, num = 20)

y_arg = 'K'
# y_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
y_values = np.arange(start = 0, stop = 200, step = 10, dtype = int)

disable = False

print(f'Values: lmb = {x_values[idx_x]}, K = {y_values[idx_y]}')

array_dict = {'beta': 10,
              'H': 0,
              'max_it': 30,
              'error': 0.002
              }

sys_kwargs = {'neurons': 3000,
              'M': 1,
              'rho': 0,
              'mixM': 0,
              'quality': [1, 1, 1],
              'sigma_type': 'mix',
              'noise_dif': False
              }

dynamic = 'parallel'
save_n = True
save_int = True
av_counter = 3

len_x = len(x_values)
len_y = len(y_values)

directory = 'MC2d'

npz_files = npz_file_finder(directory = directory, prints = False, dynamic = dynamic, K = y_values,
                            lmb = x_values, av_counter = av_counter, save_n = save_n, save_int = save_int,
                            **array_dict, **sys_kwargs)

if len(npz_files) > 1:
    print('Warning: more than 1 experiments found for given inputs.')

try:
    file_npz = npz_files[0]
    file_npy_m = file_npz[:-4] + f'_sample{idx_s}_m.npy'
    file_npy_ints = file_npz[:-4] + f'_sample{idx_s}_ints.npy'
    file_json = file_npz[:-3] + 'json'
    with open(file_json, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        entropy_from_os = int(data['entropy'])
    entropy = (entropy_from_os, idx_s)
    mattis_from_file = np.load(file_npy_m)[idx_x*len_y + idx_y]
    ints_from_file = np.load(file_npy_ints)[idx_x * len_y + idx_y]
except IndexError or FileNotFoundError:
    print('No file saved for these inputs')
    entropy = np.random.SeedSequence().entropy
    mattis_from_file = np.zeros((3, 3))
    ints_from_file = None

if ints_from_file is not None:
    print('From file:')
    print(mattis_from_file)
    print(f'Ran to {ints_from_file} iteration(s).')

rng_seeds1 = np.random.SeedSequence(entropy).spawn(len_x * len_y)
rng_seeds2 = np.random.SeedSequence(entropy).spawn(len_x * len_y)

t = time()
# To use to compare
system1 = hop(lmb = x_values[idx_x], K = y_values[idx_y], rngSS = rng_seeds1[idx_x * len_y + idx_y], **sys_kwargs)

print(f'Initialized system 1 in {round(time() - t, 3)} s.')
t = time()
alt_sys_kwargs = dict(sys_kwargs)
sys_kwargs['sigma_type'] = 'dis'
# To use for new inputs
system2 = hop(lmb = x_values[idx_x], K = y_values[idx_y], **alt_sys_kwargs)
print(f'Initialized system 2 in {round(time() - t, 3)} s.')

print(f'J matrices check: {np.array_equal(system1.J, system2.J)}')

compare_simulations = True

if compare_simulations:

    alt_array_dict = dict(array_dict)
    alt_array_dict['error'] = 0
    alt_array_dict['max_it'] = 500

    time0 = time()
    print('System 1 running...')
    output1 = system1.simulate(dynamic = dynamic, sim_rngSS = rng_seeds1[idx_x * len_y + idx_y].spawn(1)[0], av_counter = av_counter,
                               prints = False, cut = False, **array_dict)[0]
    if not np.array_equal(np.mean(output1[-av_counter:], axis=0), mattis_from_file):
        sys.exit('Sanity check failed.')
    else:
        print('Sanity check cleared.')
    time1 = time()-time0
    time0 = time()
    system2.add_load(8, prints=True)
    print('\n System 2 running...')
    output2 = system2.simulate(dynamic = 'parallel', sim_rngSS = rng_seeds2[idx_x * len_y + idx_y].spawn(1)[0], disable = True, prints = True,
                               av_counter = 3, cut = False, **alt_array_dict)[0]
    time2 = time()-time0
    # print(f'Check 2: {np.array_equal(np.mean(output2[-av_counter:], axis = 0), mattis_from_file)}\n')
    # print(f'Fast noise checker: {np.array_equal(np.random.default_rng(rng_seeds1[idx_x * len_y + idx_y]).random(10),
    #                                                np.random.default_rng(rng_seeds2[idx_x * len_y + idx_y]).random(10))}')
    # print(f'Time per iteration (system 1): {time1/(len(output1)-1)}')
    # print(f'Time per iteration (system 2): {time2 / (len(output2) - 1)}')


