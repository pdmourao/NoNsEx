from MCfuncs import MC1d_beta
import numpy as np
from time import time
from storage import file_finder
import os
from npy_append_array import NpyAppendArray
from matplotlib import pyplot as plt
from FPfields import NoNsEx, m_in, initial_q
import FPfuncs as fp

samples = 0
disable = False
colors = ['red', 'orange', 'blue', 'green']

kwargs = {'beta': 1 / np.linspace(0.01, 1, 100, endpoint=True),
          'rho': 0.1,
          'lmb': 0.1 ,
          'H': 0,
}

kwargs_MC = {'neurons': 5000,
             'K': 5,
             'M': 1000,
             'max_it': 20,
             'error': 0.01,
             'av_counter': 5,
             'quality': [1, 1, 1]
          }

x_values = 1/kwargs['beta']

parallel = True
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

files = file_finder('MC1d', file_spec=full_string, **kwargs_MC, **kwargs)

try:
    filename = files[0]
except IndexError:
    print('Creating new.')
    filename = os.path.join('MC1d', f'MC1d{full_string}beta{len_b}_{int(time())}.npz')

for sample in range(samples):

    t = time()
    print(f'\nSolving system {sample + 1}/{samples}...')
    mattisses = MC1d_beta(parallel=parallel, use_tf=use_tf, noise_dif=noise_dif, random_systems=random_systems,
                          disable=disable, **kwargs_MC, **kwargs)

    with NpyAppendArray(filename[:-1] + 'y', delete_if_exists=False) as file:
        file.append(mattisses.reshape(1, np.size(mattisses)))

    if len(files) == 0 and sample == 0:
        np.savez(filename, **kwargs_MC, **kwargs)

    print('File appended.')

    t = time() - t
    print(f'System ran in {round(t / 60)} minutes.')

# CODE WHAT TO GRAPH

try:
    m_arrays_flat = np.load(filename[:-1] + 'y')
except FileNotFoundError:
    m_arrays_flat = np.zeros((samples, len_b, 3, 3))
samples = len(m_arrays_flat)

print(f'Found {samples} samples.')

m_arrays = m_arrays_flat[:samples].reshape((samples, len_b, 3, 3))

cutoff = 0.7

m_MC = np.zeros((len_b, 3))
rate_success_MC = np.zeros(len_b)

x_values_1 = []
x_values_2 = []
m_dis = []
m_notdis = []

for idx_x, x in enumerate(x_values):
    mags_dis = []
    mags_notdis = []
    successes = 0
    for idx_s in range(samples):
        this_diag = np.sort(np.diagonal(m_arrays[idx_s, idx_x]))[::-1]
        if all([entry > cutoff for entry in this_diag]):
            mags_dis.append(this_diag)
            successes += 1
        else:
            mags_notdis.append(this_diag)
    if successes > 0:
        m_dis.append(np.mean(np.array(mags_dis)))
        x_values_1.append(x)
    if successes < samples:
        m_notdis.append(np.mean(np.array(mags_notdis), axis=0))
        x_values_2.append(x)
    rate_success_MC[idx_x] = successes / samples


[plt.scatter(x_values_2, np.array(m_notdis)[:, i], label=f'm[{i},{i}]', color = colors[i], s = 1) for i in range(3)]
plt.scatter(x_values_1, m_dis, color = colors[-1], s = 1)
plt.plot(x_values, rate_success_MC, linestyle = 'dashed', color = 'black')

field = NoNsEx

kwargs_FP = {'alpha': 0, 'max_it': 1000, 'ibound': 1e-20, 'error': 1e-16}

pert_matrix = np.array([[1,  0,  0],
                        [0,  0,  0],
                        [0,  0, -1]])

pert = 1e-8*pert_matrix
args = m_in(4/10)+pert, initial_q

m, q, n = fp.solve(field, *args, use_files = True, disable = False, **kwargs_FP, **kwargs)

idx_tr = fp.FindTransition(vec_m = m, tr_det = fp.tr_det_NoNsEx)

draw_plots = True
if draw_plots:
    plt.plot(x_values[:idx_tr], m[:idx_tr, 0, 0], color=colors[-1], linestyle = 'dashed')
    [plt.plot(x_values[idx_tr:], m[idx_tr:, i, i], color = colors[i], linestyle = 'dashed') for i in range(3)]

plt.vlines(x = (x_values[idx_tr-1]+x_values[idx_tr-1])/2, ymin = 0, ymax = 1, color = 'grey', linestyle = 'dashed')

plt.title(f'{kwargs_MC['neurons']} neurons, K = {kwargs_MC['K']}\nrho = {kwargs['rho']}, lmb = {kwargs['lmb']}, {samples} sample(s)')

plt.show()


