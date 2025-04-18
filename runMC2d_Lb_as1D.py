from MCfuncs import MC2d_Lb
import numpy as np
from time import time
from storage import npz_file_finder
import os
from npy_append_array import NpyAppendArray
from matplotlib import pyplot as plt
from FPfields import NoNsEx, m_in, initial_q
import FPfuncs as fp

samples = 10
disable = False
colors = ['red', 'orange', 'blue', 'green']


x_arg = 'beta'
kwargs = {'beta': 1/np.linspace(0.01, 1, 100, endpoint=True)[::-1],
          'rho': 0.05,
          'lmb': 0.1 ,
          'H': 0,
}

kwargs_MC = {'neurons': 5000,
             'K': 5,
             'M': 150,
             'mixM': 0,
             'max_it': 30,
             'error': 0.002,
             'av_counter': 3,
             'quality': [1, 1, 1],
             'dynamic': 'parallel',
             'sigma_type': 'mix',
             'noise_dif': False,
             'save_n': False,
             **kwargs
             }

if x_arg == 'lmb':
    kwargs_MC['beta'] = [kwargs_MC['beta']]
else:
    kwargs_MC['lmb'] = [kwargs_MC['lmb']]

sigma_type = 'mix'
x_values = kwargs_MC[x_arg]
len_x = len(x_values)

m_arrays, m_arrays_ex = MC2d_Lb(directory = 'MC1d_Lb', disable = disable, n_samples = samples, **kwargs_MC)


cutoff = 0.7

m_MC = np.zeros((len_x, 3))
rate_success_MC = np.zeros(len_x)

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

draw_FP = False
if draw_FP:
    field = NoNsEx

    kwargs_FP = {'alpha': 0, 'max_it': 1000, 'ibound': 1e-20, 'error': 1e-16}

    pert_matrix = np.array([[1,  0,  0],
                            [0,  0,  0],
                            [0,  0, -1]])

    pert = 1e-8*pert_matrix
    if sigma_type == 'dis':
        pert_matrix = np.array([[1, 0, 0],
                                [0, 0, 0],
                                [0, 0, -1]])

        pert = 1e-8 * pert_matrix
        initial_m = m_in(kwargs_MC['quality'][0] - 1/2)
    elif sigma_type == 'mix':
        pert_matrix = np.array([[ 1,-1,-1],
                                [-1, 1,-1],
                                [-1,-1, 1]])

        pert = 1e-8 * pert_matrix
        initial_m = m_in()

    args = initial_m+pert, initial_q

    m, q, n = fp.solve(field, *args, use_files = True, disable = False, **kwargs_FP, **kwargs)

    idx_tr = fp.FindTransitionFromVec(vec_m = m, tr_det = fp.tr_det_NoNsEx)

    plt.plot(x_values[:idx_tr], m[:idx_tr, 0, 0], color=colors[-1], linestyle = 'dashed')
    [plt.plot(x_values[idx_tr:], m[idx_tr:, i, i], color = colors[i], linestyle = 'dashed') for i in range(3)]

    plt.vlines(x = (x_values[idx_tr-1]+x_values[idx_tr-1])/2, ymin = 0, ymax = 1, color = 'grey', linestyle = 'dashed')

plt.title(f'{kwargs_MC['neurons']} neurons, K = {kwargs_MC['K']}\nrho = {kwargs['rho']}, lmb = {kwargs['lmb']}, {samples} sample(s)')

plt.show()


