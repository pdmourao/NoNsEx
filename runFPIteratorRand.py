import FPfuncs as fp
import numpy as np
from FPfields import NoNsEx, m_in, initial_q
from matplotlib import pyplot as plt

samples = 6
use_files = True
field = NoNsEx
eps = 1e-10

kwargs = {'lmb': 0.2, 'rho': 0.1, 'beta': 1/np.linspace(0.01, 1, 100, endpoint = True), 'alpha': 0, 'H': 0, 'max_it': 1000, 'ibound': 1e-20, 'error': 1e-15}

args = m_in(4/10), initial_q
print('Initial arguments:')
print(args)

x_arg = None
x_values = 0

for key, value in kwargs.items():
    if not np.isscalar(value):
        if x_arg is not None:
            print('Warning: multiple arrays given as inputs.')
        x_arg = key
        x_values = value

len_x = len(x_values)

m_array = np.zeros((samples, len_x, 3, 3))
n_array = np.zeros((samples, len_x, 3, 3))
q_array = np.zeros((samples, len_x, 3))

cutoff = 0.9

for idx in range(samples):
    print(f'\nSolving sample {idx+1}/{samples}...')
    m_array[idx], q_array[idx], n_array[idx] = fp.solve(field, *args, use_files = True, rand = (idx, eps), **kwargs)

for idx in range(samples):
    m = m_array[idx]

    print('New sample')
    for idx_x, value in enumerate(x_values):
        print(f'{x_arg} = {round(value, 2)}')
        print(m[idx_x])

    fig, ax = plt.subplots()

    if x_arg is None:
        x_arg = 'it'
        x_values = np.arange(len(m))

    for i in range(3):
        plt.scatter(1/x_values, m[:, i, i] - 0.01 + 0.01 * i, label=f'm[{i},{i}]')

    plt.ylabel('$m$')
    plt.xlabel(f'${fp.arg_to_label[x_arg]}$')
    plt.ylim(-1,1)
    # ax.spines['bottom'].set_position('center')
    # plt.axhline(y= - 1, color='black', linewidth = 2)
    plt.legend()

    exempt_from_title = [x_arg]
    title_strings = []
    for key, value in kwargs.items():
        if key not in exempt_from_title:
            try:
                title_strings.append(f'{fp.arg_to_label[key]} = {value}')
            except KeyError:
                title_strings.append(f'{key} = {value}')

    plt.title(", ".join(title_strings))


    # Testing FindTransitions

    det_list = []
    det_list = [lambda x: fp.disentangle_det(x, threshold =cutoff)]

    for func in det_list:
        tr_idx = fp.FindTransition(tr_det = func, vec_m = m)
        if tr_idx > 0:
            idx = tr_idx - 1
            plt.vlines(x=x_values[idx], ymin=0, ymax=m[idx, 1, 1], linestyle='dashed', color='black')
    plt.show()