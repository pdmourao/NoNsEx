import numpy as np
from time import time
from storage import npz_file_finder
import json
from MCclasses import HopfieldMC as hop
from MCfuncsCopy import disentanglement, disentanglement_2d
import sys

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

idx_s = 0


x_arg = 'rho'
x_values = np.linspace(start = 0, stop = 0.3, num = 50, endpoint = False)

y_arg = 'lmb'
# y_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
y_values = np.linspace(start = 0, stop = 0.5, num = 50)

idx_x = 45
idx_y = 45

print(f'Values: rho = {x_values[idx_x]}, lmb = {y_values[idx_y]}')

array_dict = {'beta': 5,
              'H': 0,
              'max_it': 30,
              'error': 0.002
              }

sys_kwargs = {'neurons': 3000,
              'K': 3,
              'M': 50,
              'mixM': 0,
              'quality': [1, 1, 1],
              'sigma_type': 'mix',
              'noise_dif': False
              }

new_dict = {**array_dict, 'layers': 3, 'h_norm': 0, 'neurons': 3000, 'k': 3, 'm': 50, 'split': False, 'supervised': True, 'av_counter': 3, 'dynamic': 'parallel'}
new_dict.pop('H')

r_values = np.sqrt(1 / (x_values * sys_kwargs['M'] + 1))

dynamic = 'parallel'
save_n = True
av_counter = 3
save_int = True

len_x = len(x_values)
len_y = len(y_values)

directory = 'MC2d'

npz_files = npz_file_finder(directory = directory, prints = True, dynamic = dynamic, rho = x_values, save_int = save_int,
                            lmb = y_values, av_counter = av_counter, save_n = save_n, **array_dict, **sys_kwargs)

if len(npz_files) > 1:
    print('Warning: more than 1 experiments found for given inputs.')
    print(f'Using the first: {npz_files[0]}')

try:
    file_npz = npz_files[0]
    file_npy = file_npz[:-4] + f'_sample{idx_s}_m.npy'
    file_npy_ints = file_npz[:-4] + f'_sample{idx_s}_ints.npy'
    file_json = file_npz[:-3] + 'json'
    with open(file_json, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        entropy_from_os = int(data['entropy'])
    entropy = (entropy_from_os, idx_s)
    mattis_from_file = np.load(file_npy).reshape((len_x, len_y, 3, 3))
    ints_from_file = np.load(file_npy_ints)[idx_x * len_y + idx_y]
except (IndexError,FileNotFoundError) as e:
    print('No file saved for these inputs')
    entropy = np.random.SeedSequence().entropy
    mattis_from_file = np.zeros((3, 3))
    ints_from_file = None

if ints_from_file is not None:
    print('From file:')
    print(mattis_from_file)
    print(f'Ran to {ints_from_file} iteration(s).')


# print(mattis_from_file)

rng_seeds1 = np.random.SeedSequence(entropy).spawn(len_x * len_y)
rng_seeds2 = np.random.SeedSequence(entropy).spawn(len_x * len_y)

t = time()
# To use to compare
system1 = hop(rho = x_values[idx_x], lmb = y_values[idx_y], rngSS = rng_seeds1[idx_x * len_y + idx_y], **sys_kwargs)

print(f'Initialized system 1 in {round(time() - t, 3)} s.')
t = time()
alt_sys_kwargs = dict(sys_kwargs)
# To use for new inputs
rng1 = rng_seeds1[idx_x * len_y + idx_y]
rng2 = rng_seeds2[idx_x * len_y + idx_y]



print('System 1 running...')


mattis, ex_mags, its = disentanglement_2d(x_arg = 'r', y_arg = 'lmb', x_values = r_values, y_values = y_values, entropy = entropy, checker = mattis_from_file, **new_dict)

mattis1, ex_mags1, its1 = system1.simulate(dynamic = dynamic, sim_rngSS = rng1.spawn(1)[0], av_counter = av_counter,
                            av = True, **array_dict)

if not np.array_equal(mattis1, mattis_from_file):
    print('1st sanity check failed.')
else:
    print('1st Sanity check cleared.')

if not np.array_equal(mattis1, mattis):
    print('2nd sanity check failed.')
else:
    print('2nd Sanity check cleared.')

if not np.array_equal(ex_mags, ex_mags1):
    print('3rd sanity check failed.')
else:
    print('3rd Sanity check cleared.')

if not np.array_equal(its, its1):
    print('4th sanity check failed.')
else:
    print('4th Sanity check cleared.')



