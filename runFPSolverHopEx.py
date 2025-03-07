import FPfuncs as fp
import numpy as np
from FPfields import NoNsEx, HopEx
from time import time
from matplotlib import pyplot as plt
import os
from storage import file_finder

use_files = False

t = time()
var_tag = None

m_in, m_out, initial_q = fp.NoNs_initial(epsilon = 0, pert_in = 0, pert_out = 0)
initial_m = np.array([[1-0.001, 1, 1],[0, 0, 0],[0, 0, 0.001]])

pert3d_arrays = np.zeros(shape = (3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            pert3d_arrays[i, j, k] = np.array([i-1, j-1, k-1])
eprange = np.arange(0, 0.2, 0.01)
initial_m_array = np.zeros(shape = (len(eprange), 3))
pert=np.array([-1, -1, 1])
for idx_e, epsilon in enumerate(eprange):
    initial_m_array[idx_e] = np.array([1/2, 1/2, 1/2]) + epsilon * pert



kwargs = {'rho': np.arange(0, 0.5, 0.01), 'T': 0.1, 'alpha': 0, 'H': 0, 'max_it': 200, 'use_previous': 0, 'error': 1e-32}
pert9d = np.random.uniform(low = -1, high = 1, size = (3, 3))
pert3d = np.random.uniform(low = -1, high = 1, size = 3)
# print(pert9d)
print(pert3d)
# initial_m_array = [m_in + epsilon * pert9d for epsilon in np.arange(0, 0.5, 0.01)]
epsilon = 0.5
initial_m = np.array([1/2, 1/2, 1/2]) + epsilon * pert3d

# m_array, q_array = fp.solver(initial_m = initial_m, initial_q = np.array([1, 1, 1]), use_files = use_files, field = NoNsEx, var_tag = var_tag, **kwargs, not_all_neg=False)
m_array, q_array = fp.solver(initial_m = initial_m, initial_q = 0, use_files = use_files, field = HopEx, var_tag = None, **kwargs, not_all_neg=False)
# plot part
# print(m_array)
if var_tag is None:
    x_arg = 'it'
else:
    x_arg = var_tag
x_values = np.arange(len(m_array))+1

title_dict = {var: kwargs[var] for var in kwargs.keys() if var in ['T', 'rho', 'lmb', 'H', 'rho']}

for key, item in kwargs.items():
    if not np.isscalar(item):
        x_arg = key
        x_values = item
        title_dict.pop(key)

print(len(x_values))
print(len(m_array))

fig, ax = plt.subplots()

try:
    for i in range(3):
        plt.scatter(x_values, m_array[:,i,i]-0.01+0.01*i, label = f'm[{i},{i}]')
        for j in range(3):
            if j == 1 and i == 2:
                # plt.scatter(x_values, m_array[:, i, j], label=f'm[{i},{j}]')
                pass
except IndexError:
    try:
        for i in range(3):
            pass
            plt.scatter(x_values, m_array[:, i] - 0.01 + 0.01 * i, label=f'm[{i}]')
    except IndexError:
        plt.scatter(x_values, m_array, label='m')

print(m_array)

plt.ylabel('$m$')
plt.xlabel(f'${fp.greek[x_arg]}$')
plt.ylim(-1,1)
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