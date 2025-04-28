import FPfuncs as fp
import numpy as np
from FPfields import NoNsEx, m_in, initial_q
from matplotlib import pyplot as plt
from FPfuncs import recovered_pats

use_files = True
field = NoNsEx

# rho_values = np.arange(start = 0, stop = 0.31, step = 0.01)
rho_values = [0.05]
beta_values1 = np.arange(start = 1.5, stop = 5, step = 0.5)
beta_values2 = np.arange(start = 5, stop = 21)
beta_values = np.concatenate([beta_values1, beta_values2])
print(beta_values)

pert_1 = np.array([[ 1, 0, -1],
                       [ 1, 0, -1],
                       [ 1, 0, -1]])
pert_2 = np.array([[ 1, -1, -1],
                       [ 1, -1, -1],
                       [ 1, -1, -1]])
pert_3 = np.array([[ 1, -1, -1],
                       [-1,  1, -1],
                       [-1,  1, -1]])
pert_4 = np.array([[ 1, -1, -1],
                       [-1,  1, -1],
                       [-1, -1,  1]])

pert_5 = np.array([[ 0, 0,  0],
                       [-1, 1,  0],
                       [-1, 0,  1]])

pert_6 = np.array([[3,  -2, -1],
                    [3,  -2, -1],
                    [3,  -2, -1]])

eps = 1e-8
other_initial = np.array([[1-eps,   eps, eps],
                              [1-eps,   eps, eps],
                              [1-eps,   eps, eps]])

    # pert = np.array([[0.01, -0.06, 0], [0.01, -0.06, 0], [0.01, -0.06, 0]])

for rho in rho_values:
    print(f'rho = {rho}')
    kwargs = {'lmb': np.linspace(0, 0.5, 100, endpoint = False),
              'rho': rho,
              'beta': 5,
              'alpha': 0,
              'H': 0,
              'max_it': 1000,
              'ibound': 1e-12,
              'error': 1e-10}

    pert = 1e-8 * pert_2
    args = m_in(4/10)+pert, initial_q

    m_out, q_out, n_out = fp.solve(field, *args, use_files = use_files, disable = False, **kwargs)

    pert = 1e-8 * pert_4
    args = m_in() + pert, initial_q

    m_ini, q_ini, n_ini = fp.solve(field, *args, use_files=use_files, disable=False, **kwargs)


graph = False

if graph:
    fig, ax = plt.subplots()

    x_arg = None
    for key, value in kwargs.items():
        if not np.isscalar(value):
            if x_arg is not None:
                print('Warning: multiple arrays given as inputs.')
            x_arg = key
            x_values = value

    exempt_from_title = [x_arg, 'max_it', 'ibound', 'error', 'alpha', 'H']

    if x_arg is None:
        x_arg = 'it'
        x_values = np.arange(len(m))
    elif x_arg == 'beta':
        x_arg = 'T'
        x_values = 1/x_values

    for i in range(3):
        plt.scatter(x_values, m[:, i, i] - 0.01 + 0.01 * i, label=f'm[{i},{i}]')
        for j in range(3):
            if j == 1 and i != 1:
                # plt.scatter(x_values, m[:, i, j]-0.02*i, label=f'm[{i},{j}]')
                pass


    plt.ylabel('$m$')
    plt.xlabel(f'${fp.arg_to_label[x_arg]}$')
    plt.ylim(0,1)
    # ax.spines['bottom'].set_position('center')
    # plt.axhline(y= - 1, color='black', linewidth = 2)
    plt.legend()

    title_strings = []
    for key, value in kwargs.items():
        if key not in exempt_from_title:
            try:
                title_strings.append(f'{fp.arg_to_label[key]} = {value}')
            except KeyError:
                title_strings.append(f'{key} = {value}')

    plt.title(", ".join(title_strings))


    # Testing FindTransitions
    cutoff = 0.9
    cutoff_mix = 0.1


    det_list = [lambda x: fp.disentangle_det(x, threshold =cutoff), lambda x: fp.tr_notdis_NoNsEx(x, threshold1 = cutoff)]
    det_list = []
    for func in det_list:
        tr_idx = fp.FindTransitionFromVec(tr_det = func, vec_m = m)
        if tr_idx > 0:
            idx = tr_idx - 1
            plt.vlines(x=x_values[idx], ymin=0, ymax=m[idx, 1, 1], linestyle='dashed', color='black')

    plt.show()

calc_pats = False
if calc_pats:
    how_many_pats = np.zeros(len(m))

    for idx_m, m_entry in enumerate(m):
        print(f'{x_arg} = {x_values[idx_m]}')
        print(m_entry)
        pats = recovered_pats(m_entry, cutoff, cutoff_mix)
        how_many_pats[idx_m] = len(set([abs(pat) for pat in pats if pat is not None and pat != 4]))
        print(pats)
        print(how_many_pats[idx_m])

    plt.scatter(x_values, how_many_pats)
    plt.show()

