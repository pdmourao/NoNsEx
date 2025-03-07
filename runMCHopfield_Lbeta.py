import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import gJprod, recovered_pats, MC2d_Lb
from time import time
import os
from storage import file_finder
from npy_append_array import NpyAppendArray

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

samples = 0
sample_graph = 30

kwargs = {'neurons': 3000, 'K': 3, 'rho': 0.05, 'H': 0, 'M': 10000, 'max_it': 10, 'error': 0, 'av_counter': 1}

parallel = False
use_tf = False

noise_dif = False

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)
beta_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]

len_l = len(l_values)
len_b = len(beta_values)
n_pixels = len_l * len_b

if parallel and use_tf:
    dl = 'PDtf'
elif parallel and not use_tf:
    dl = 'PDnp'
else:
    dl = 'SD'

if noise_dif:
    noise_string = 'in'
else:
    noise_string = 'dn'

files = file_finder('MC2d', file_spec = f'_{noise_string}{dl}_', **kwargs)

try:
    filename = files[0]
except IndexError:
    print('Creating new.')
    filename = os.path.join('MC2d', f'MC2d_{noise_string}{dl}_Lb{n_pixels}_{int(time())}.npz')

for idx in range(samples):
    t = time()
    print(f'\nSolving system {idx + 1}/{samples}...')

    mattisses = MC2d_Lb(l_values = l_values, beta_values = beta_values, parallel = parallel, use_tf = use_tf,
                        disable = False, noise_dif=noise_dif, **kwargs)

    with NpyAppendArray(filename[:-1] + 'y', delete_if_exists = False) as file:
        file.append(mattisses.reshape((1, 9 * n_pixels)))

    if len(files) == 0 and idx == 0:
        np.savez(filename, **kwargs)

    print('File appended.')

    t = time() - t
    print(f'System ran in {round(t/60)} minutes.')


final_time = time() - t0

try:
    mattis_trials = np.load(filename[:-1] + 'y')
except FileNotFoundError:
    mattis_trials = np.zeros((1, 9 * n_pixels))

samples = min(len(mattis_trials), sample_graph)

m_array_trials = mattis_trials[:samples].reshape((samples, len_l, len_b, 3, 3))
success_array = np.zeros((4, samples, len_l, len_b))

print('\nCalculating success rates...')
cutoff = 0.8

t = time()

recovery_array = [[[recovered_pats(m_array_trials[idx_s, idx_l, idx_b], cutoff) for idx_b in range(len_b)] for idx_l in range(len_l)] for idx_s in range(samples)]

for idx_s in range(samples):
    for idx_l in range(len_l):
        for idx_b in range(len_b):
            if idx_l == 12 and idx_b == 24:
                print(m_array_trials[idx_s, idx_l, idx_b])
                print(recovery_array[idx_s][idx_l][idx_b])
                pass
            how_many_pats = len(set([abs(index) for index in recovery_array[idx_s][idx_l][idx_b] if index is not None]))
            for missing in range(4):
                if how_many_pats == 3 - missing:
                    success_array[missing, idx_s, idx_l, idx_b] = 1

print(f'Calculated success rates in {time() - t} seconds.')
for missing in range(4):
    n_pats = 3 - missing
    c = plt.imshow(np.transpose(np.average(success_array[missing], axis = 0)), cmap = 'Greens', vmin = 0, vmax = 1,
                   extent=[l_values[0], l_values[-1], beta_values[-1], beta_values[0]], aspect='auto',
                   interpolation='nearest')

    plt.colorbar(c)

    plt.xlabel('$λ$')
    plt.ylabel(f'$β$')
    plt.title(f'N = {kwargs['neurons']}, K = {kwargs['K']}, ρ = {kwargs['rho']}, M = {kwargs['M']}, H = {kwargs['H']}\n{samples} sample(s), {cutoff} cutoff, {n_pats} pattern(s)')

    plt.show()
