from MCfuncs import MC1d_beta
import numpy as np
from time import time
from storage import file_finder
import os
from npy_append_array import NpyAppendArray
from matplotlib import pyplot as plt

samples = 8
sample_graph = 30
disable = False

kwargs = {'beta': 1 / np.linspace(0.01, 1, 100, endpoint=True),
          'neurons': 5000,
          'K': 5,
          'rho': 0.1,
          'M': 1000,
          'lmb': 0.2,
          'H': 0,
          'max_it': 20,
          'error': 0.02,
          'av_counter': 5,
          'quality': [0.9, 0.9, 0.9]
          }

x_values = 1/kwargs['beta']

parallel = False
use_tf = False
noise_dif = False
random_systems = False

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

if random_systems:
    rand_string = 'rs'
else:
    rand_string = 'ds'

len_b = len(x_values)

full_string = f'_{noise_string}{dl}{rand_string}_'

files = file_finder('MC1d', file_spec=full_string, **kwargs)

try:
    filename = files[0]
except IndexError:
    print('Creating new.')
    filename = os.path.join('MC1d', f'MC1d{full_string}beta{len_b}_{int(time())}.npz')

for sample in range(samples):

    t = time()
    print(f'\nSolving system {sample + 1}/{samples}...')
    mattisses = MC1d_beta(parallel=parallel, use_tf=use_tf, noise_dif=noise_dif, random_systems=random_systems,
                          disable=True, **kwargs)

    with NpyAppendArray(filename[:-1] + 'y', delete_if_exists=False) as file:
        file.append(mattisses.reshape(1, np.size(mattisses)))

    if len(files) == 0 and sample == 0:
        np.savez(filename, **kwargs)

    print('File appended.')

    t = time() - t
    print(f'System ran in {round(t / 60)} minutes.')

# CODE WHAT TO GRAPH
try:
    m_arrays_flat = np.load(filename[:-1] + 'y')
except FileNotFoundError:
    m_arrays_flat = np.zeros((sample_graph, len_b, 3, 3))
samples = min(len(m_arrays_flat), sample_graph)
m_arrays = m_arrays_flat[:samples].reshape((samples, len_b, 3, 3))
m_array_av = np.mean(m_arrays, axis = 0)

print(m_arrays[:, 49, :, :])

for i in range(3):
    plt.scatter(x_values, m_array_av[:, i, i], label=f'm[{i},{i}]')
    for j in range(3):
        if j == 1 and i != 1:
            # plt.scatter(x_values, m[:, i, j]-0.02*i, label=f'm[{i},{j}]')
            pass

plt.show()