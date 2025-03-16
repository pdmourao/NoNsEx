import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import MC2d_Lb, mags_id
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
idx_l = 10
idx_y = 10

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)

y_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
# y_values = np.linspace(start = 0, stop = 0.2, num = 50)

array_dict = {'beta': y_values,
              'H': 0,
              'max_it': 20,
              'error': 0.01,
              }

dynamic = 'parallel'
neurons = 5000
K = 5
rho = 0.05
M = 100
av_counter = 5
quality = [1, 1, 1]
sigma_type = 'mix'
noise_dif = False

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

npz_files = npz_file_finder(directory = directory, prints = False, neurons = neurons, K = K, dynamic = dynamic,
                            rho = rho, M = M, quality = quality, sigma_type = sigma_type, noise_dif = noise_dif,
                            lmb = l_values, av_counter = av_counter, **array_dict)

if len(npz_files) > 1:
    print('Warning: more than 1 experiments found for given inputs.')

file_npz = npz_files[0]
file_npy = file_npz[:-4] + f'_sample{idx_s}.npy'
file_json = file_npz[:-3] + 'json'

with open(file_json, mode="r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    entropy_from_os = int(data['entropy'])


entropy = (entropy_from_os, idx_s)

mattis_from_file = np.load(file_npy)[idx_l*len_y + idx_y]



t = time()
# To use to compare
system1 = hop(neurons= neurons, K= K, rho = rho, M = M, lmb = l_values[idx_l], quality= quality,
              sigma_type = sigma_type, noise_dif = noise_dif, entropy = entropy)

# To use for new inputs
system2 = hop(neurons= neurons, K= K, rho = rho, M = M, lmb = l_values[idx_l], quality= quality,
              sigma_type = sigma_type, noise_dif = noise_dif, entropy = entropy)

t0 = time()
print(f'Initialized system in {round(t0 - t, 3)} s.')
rng_seeds1 = np.random.SeedSequence(entropy=entropy).spawn(len_l * len_y)
rng_seeds2 = np.random.SeedSequence(entropy=entropy).spawn(len_l * len_y)
print(f'Generated seeds for simulate in {round(time() - t0, 3)} s.')

array_dict[y_arg] = y_values[idx_y]
output1 = system1.simulate(dynamic = dynamic, sim_rngSS = rng_seeds1[idx_l * len_y + idx_y], av_counter = av_counter,
                           **array_dict)[0]
output2 = system2.simulate(dynamic = dynamic, sim_rngSS = rng_seeds2[idx_l * len_y + idx_y], av_counter = av_counter,
                           **array_dict)[0]

print(f'\nCheck: {np.array_equal(np.mean(output1[-av_counter:], axis = 0), mattis_from_file)}\n')


print(np.mean(output2[-av_counter:], axis = 0))




