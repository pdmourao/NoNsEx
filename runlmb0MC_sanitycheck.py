from MCfuncs import MCrho_s
import numpy as np
from tqdm import tqdm
from time import time
import os
from storage import file_finder
from npy_append_array import NpyAppendArray

samples = 1
rho_values = np.arange(0, 0.6, 0.01)

parallel = True
directory = 'MCrho_s'
save_file = False

if parallel:
    dl = 'PD'
else:
    dl = 'SD'


kwargs = {'rho_values': rho_values, 'neurons': 2000, 'K': 3, 'M': 1000, 'error': 1e-3, 'beta': 10000, 'max_it': 10}

files = file_finder(directory, **kwargs)

try:
    filename = files[0]
except IndexError:
    print('Creating new.')
    filename = os.path.join(directory, f'MC1d_{dl}_rho_s_{len(rho_values)}_{int(time())}.npz')


for idx in range(samples):
    t = time()
    print(f'\nSolving system {idx + 1}/{samples}...')
    m_array_initial, m_array = MCrho_s(**kwargs, disable = True)

    if save_file:
        with NpyAppendArray(filename[:-4] + '_m.npy', delete_if_exists = False) as file:
            file.append(m_array.reshape((1, kwargs['K'] * len(rho_values))))
        with NpyAppendArray(filename[:-4] + '_i.npy', delete_if_exists = False) as file:
            file.append(m_array_initial.reshape((1, kwargs['K'] * len(rho_values))))

        if len(files) == 0 and idx == 0:
            np.savez(filename, **kwargs)

        print('File appended.')

    t = time() - t
    print(f'System ran in {round(t/60)} minutes.')

if save_file:
    try:
        m_array = np.load(filename[:-4] + '_m.npy')
        m_array_initial = np.load(filename[:-4] + '_i.npy')
    except FileNotFoundError:
        m_array = np.zeros((1, kwargs['K'] * len(rho_values)))
        m_array_initial = np.zeros((1, kwargs['K'] * len(rho_values)))

samples = len(m_array)

m_array_trials = m_array.reshape((samples, len(rho_values), kwargs['K']))
m_array_trials_initial = m_array_initial.reshape((samples, len(rho_values), kwargs['K']))

one_sample = samples-1
for idx_rho, rho in enumerate(rho_values):
    print(f'\n rho = {rho}')
    print('initial')
    print(m_array_trials_initial[one_sample,idx_rho])
    print('final')
    print(m_array_trials[one_sample, idx_rho])


