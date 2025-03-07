import FPfuncs as fp
import numpy as np
from FPfields import NoNsEx, HopEx
from time import time
from matplotlib import pyplot as plt
import os
from storage import file_finder

use_files = False

t = time()

m_in, m_out, initial_q = fp.NoNs_initial(epsilon = 0, pert_in = 0, pert_out = 0)
initial_m = m_in

# kwargs = {'lmb': 0, 'rho': 0.2, 'T': 0.1, 'alpha': 0, 'H': 0, 'max_it': 100, 'use_previous': 0, 'error': 0}
kwargs = {'rho': 0.2, 'T': 0.1, 'alpha': 0, 'H': 0, 'max_it': 100, 'use_previous': 0, 'error': 0}

# m_array, q_array = fp.solver(initial_m = initial_m, initial_q = initial_q, use_files = use_files, field = NoNsEx, **kwargs)
m_array, q_array = fp.solver(initial_m = 1/2, initial_q = 1, field = HopEx, **kwargs)
# plot part

x_arg = 'it'
x_values = np.arange(len(m_array))+1

title_dict = {var: kwargs[var] for var in ['T', 'rho', 'lmb', 'H', 'rho']}

for key, item in kwargs.items():
    if not np.isscalar(item):
        x_arg = key
        x_values = item
        title_dict.pop(key)


fig, ax = plt.subplots()

for i in range(3):
    plt.scatter(x_values, m_array[:,i,i]-0.01+0.01*i, label = f'm[{i},{i}]')
    for j in range(3):
        if j == 1 and i == 2:
            # plt.scatter(x_values, m_array[:, i, j], label=f'm[{i},{j}]')
            pass


plt.ylabel('$m$')
plt.xlabel(f'${fp.greek[x_arg]}$')
plt.ylim(0,1)
# ax.spines['bottom'].set_position('center')
# plt.axhline(y= - 1, color='black', linewidth = 2)
plt.legend()

plt.title(", ".join([f'{fp.greek[key]} = {value}' for key, value in title_dict.items()]))
t = time() - t
# print(f'Magnetization post transition is {m_post}.')


# Testing FindTransitions
cutoff = 0.95

det_list = []
# det_list = [fp.tr_det_NoNsEx]

for func in det_list:
    tr_idx = fp.FindTransition(tr_det = func, vec_m = m_array)
    if tr_idx > 0:
        idx = tr_idx - 1
        plt.vlines(x=x_values[idx], ymin=0, ymax=m_array[idx, 1, 1], linestyle='dashed', color='black')

plt.show()

# print(m_array)