import numpy as np
from time import time
from storage import npz_file_finder
import json
from MCclasses import HopfieldMC as hop, TAM as tam
import sys

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

idx_s = 0
x = 0.1
y = 20.4


l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)
y_values = np.linspace(start = 25, stop = 1, num = 50, endpoint = False)[::-1]
# y_values = np.linspace(start = 0, stop = 0.2, num = 50)

idx_l = (np.abs(l_values - x)).argmin()
idx_y = (np.abs(y_values - y)).argmin()

print(f'Values: lmb = {l_values[idx_l]}, beta = {y_values[idx_y]}')

array_dict = {'beta': y_values,
              'H': 0,
              'max_it': 30,
              'error': 0.002,
              }

sys_kwargs = {'neurons': 5000,
              'K': 5,
              'rho': 0.05,
              'M': 150,
              'mixM': 0,
              'quality': [1, 1, 1],
              'sigma_type': 'mix',
              'noise_dif': False
              }

dynamic = 'parallel'
save_n = False
save_int = False
av_counter = 3

len_l = len(l_values)
len_y = len(y_values)

directory = 'MC2d_Lb'

# Identify the y_axis array
y_arg = None
for item, value in array_dict.items():
    if not np.isscalar(value):
        assert y_arg is None, 'Too many non-scalar arguments given to MC2d_Lb'
        y_arg = item
        y_values = value

npz_files = npz_file_finder(directory = directory, prints = False, dynamic = dynamic, lmb = l_values,
                            av_counter = av_counter, save_n = save_n, **array_dict, **sys_kwargs)

if len(npz_files) > 1:
    print('Warning: more than 1 experiments found for given inputs.')

try:
    file_npz = npz_files[0]
    file_npy = file_npz[:-4] + f'_sample{idx_s}_m.npy'
    file_json = file_npz[:-3] + 'json'
    with open(file_json, mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        entropy_from_os = int(data['entropy'])
    entropy = (entropy_from_os, idx_s)
    mattis_from_file = np.load(file_npy)[idx_l*len_y + idx_y]
except IndexError or FileNotFoundError:
    print('No file saved for these inputs')
    entropy = np.random.SeedSequence().entropy
    mattis_from_file = np.zeros((3, 3))

print('Saved values:')
print(mattis_from_file)

system1 = hop(lmb = l_values[idx_l], rngSS = np.random.SeedSequence(entropy), **sys_kwargs)

t = time()
# To use to compare
r = np.sqrt(1 / (sys_kwargs['rho'] * sys_kwargs['M'] + 1))
system = tam(lmb = l_values[idx_l], neurons=sys_kwargs['neurons'], layers=3, r=r, m=sys_kwargs['M'], split = sys_kwargs['noise_dif'], supervised = True)

system.noise_patterns = np.random.default_rng(np.random.SeedSequence(entropy))
system.noise_examples = system.noise_patterns

system.fast_noise = np.random.default_rng(system.noise_patterns.spawn(len_l * len_y)[idx_l * len_y + idx_y])

system.add_patterns(sys_kwargs['K'])
system.initial_state = system.mix()
system.external_field = system.mix(0)

print('Initial states')
print(system.mattis(system.initial_state))
print(system1.mattis(system1.sigma))

print(np.array_equal(system1.J, np.transpose(system.J, [0, 2, 1, 3])))
print(f'Initialized systems in {round(time() - t, 3)} s.')

# print(f'J matrices check: {np.array_equal(system1.J, system2.J)}')

# print(system1.ex_mags(system1.sigma))
# print(system2.ex_mags(system2.sigma))

rng_seeds = np.random.SeedSequence(entropy=entropy).spawn(len_l * len_y)
# rng_seeds2 = np.random.SeedSequence(entropy=entropy).spawn(len_l * len_y)
print(f'Generated seeds for simulate in {round(time() - t0, 3)} s.')

# system = tam(neurons=neurons, layers=3, r=r, m=m, split = sys_kwargs['noise_dif'], supervised = supervised)

compare_simulations = True

if compare_simulations:
    print('System 1 running...')
    array_dict['beta'] = y_values[idx_y]
    mattis1 = system1.simulate(dynamic=dynamic, sim_rngSS=rng_seeds[idx_l * len_y + idx_y], av_counter=av_counter,
                               prints=False, av = True, **array_dict)[0]
    time0 = time()
    print('System running...')
    mattis, ex_mags, max_its = system.simulate(beta = y_values[idx_y], max_it = array_dict['max_it'], dynamic = dynamic, error = array_dict['error'], av_counter = av_counter, h_norm = array_dict['H'], av = True)
    time1 = time()-time0
    if not np.array_equal(mattis, mattis_from_file):
        print('Sanity check failed.')
    else:
        print('Sanity check cleared.')


