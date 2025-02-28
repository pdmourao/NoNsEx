import FPfuncs as fp
import numpy as np
from FPfields import NoNsEx, m_in, initial_q
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from time import time
from storage import file_finder

use_files = True
field = NoNsEx

kwargs = {'lmb': np.arange(0, 0.5, 0.005), 'rho': 0.2, 'beta': 10, 'alpha': 0, 'H': 0, 'max_it': 1000, 'ibound': 1e-20, 'error': 1e-15}

pert_1pat = np.array([[ 1, -1, -1],
                      [ 1, -1, -1],
                      [ 1, -1, -1]])
pert_2pat = np.array([[ 1, -1, -1],
                      [-1,  1, -1],
                      [-1,  1, -1]])
pert_3pat = np.array([[ 1, -1, -1],
                      [-1,  1, -1],
                      [-1, -1,  1]])

pert = pert_2pat

eps = 1e-10

args = m_in() + pert * eps, initial_q
print('Perturbation matrix used:')
print(pert*eps)

m, q, n = fp.solve(field, *args, use_files = True, **kwargs)

print(m)

fig, ax = plt.subplots()

x_arg = None
for key, value in kwargs.items():
    if not np.isscalar(value):
        if x_arg is not None:
            print('Warning: multiple arrays given as inputs.')
        x_arg = key
        x_values = value

if x_arg is None:
    x_arg = 'it'
    x_values = np.arange(len(m))

try:
    for i in range(3):
        plt.scatter(x_values, m[:,i,i]-0.01+0.01*i, label = f'm[{i},{i}]')
        for j in range(3):
            if j == 1 and i != 1:
                plt.scatter(x_values, m[:, i, j]-0.02*i, label=f'm[{i},{j}]')
                pass
except IndexError:
    try:
        for i in range(3):
            pass
            plt.scatter(x_values, m[:, i] - 0.01 + 0.01 * i, label=f'm[{i}]')
    except IndexError:
        plt.scatter(x_values, m, label='m')


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
cutoff = 0.9

det_list = []
det_list = [lambda x: fp.disentangle_det(x, threshold =cutoff), lambda x: fp.tr_notdis_NoNsEx(x, threshold1 = cutoff)]

for func in det_list:
    tr_idx = fp.FindTransition(tr_det = func, vec_m = m)
    if tr_idx > 0:
        idx = tr_idx - 1
        plt.vlines(x=x_values[idx], ymin=0, ymax=m[idx, 1, 1], linestyle='dashed', color='black')

plt.show()