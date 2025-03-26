import numpy as np
from time import time
from storage import npz_file_finder
import json
from MCclasses import HopfieldMC as hop

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

idx_s = 0
idx_l = 14
idx_y = 7

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)

y_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
# y_values = np.linspace(start = 0, stop = 0.2, num = 50)

print(f'Values: lmb = {l_values[idx_l]}, beta = {y_values[idx_y]}')

array_dict = {'beta': y_values,
              'H': 0,
              'max_it': 20,
              'error': 0.01,
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

dynamic = 'sequential'
save_n = False
av_counter = 5

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


print(mattis_from_file)


t = time()
# To use to compare
system1 = hop(lmb = l_values[idx_l], rngSS = np.random.SeedSequence(entropy), **sys_kwargs)

alt_sys_kwargs = dict(sys_kwargs)
# To use for new inputs
system2 = hop(lmb = l_values[idx_l], rngSS = np.random.SeedSequence(entropy), **alt_sys_kwargs)
t0 = time()
print(f'Initialized system in {round(t0 - t, 3)} s.')
# print(system1.ex_mags(system1.sigma))
# print(system2.ex_mags(system2.sigma))
R = system2.r**2 + (1 - system2.r**2)/system2.M
# print(system1.ex_mags(np.sign(system1.ex_av[0,:3]))*np.sqrt(1+sys_kwargs['rho']))
# print(system1.ex_mags(np.sign(system1.pat[:3]))*np.sqrt(1+sys_kwargs['rho']))
# print(system2.ex_mags(system2.ex_av[0,:3])*np.sqrt((1+sys_kwargs['rho'])/R))
print(np.array_equal(system1.J, system2.J))

# print(system1.ex_mags(system1.sigma))
# print(system2.ex_mags(system2.sigma))

rng_seeds1 = np.random.SeedSequence(entropy=entropy).spawn(len_l * len_y)
rng_seeds2 = np.random.SeedSequence(entropy=entropy).spawn(len_l * len_y)
print(f'Generated seeds for simulate in {round(time() - t0, 3)} s.')

compare_simulations = True

if compare_simulations:
    array_dict[y_arg] = y_values[idx_y]
    array_dict['error'] = 0
    array_dict['max_it'] = 50
    alt_array_dict = dict(array_dict)
    alt_array_dict['error'] = 0
    alt_array_dict['max_it'] = 100
    print('System 1 running...')
    output1 = system1.simulate(dynamic = dynamic, sim_rngSS = rng_seeds1[idx_l * len_y + idx_y], av_counter = av_counter,
                               prints = True, **array_dict)[0]
    print('\n System 2 running...')
    output2 = system2.simulate(dynamic = 'parallel', sim_rngSS = rng_seeds2[idx_l * len_y + idx_y], disable = True, prints = True,
                               av_counter = 5, **alt_array_dict)[0]

    print(f'\nCheck 1: {np.array_equal(np.mean(output1[-av_counter:], axis = 0), mattis_from_file)}')
    print(f'Check 2: {np.array_equal(np.mean(output2[-av_counter:], axis = 0), mattis_from_file)}\n')




