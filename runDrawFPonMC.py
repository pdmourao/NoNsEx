from scipy.interpolate import make_interp_spline
import numpy as np
from time import time
from matplotlib import pyplot as plt
import os
import csv
from storage import file_finder
from MCfuncs import recovered_pats
import FPfuncs as fp


t0 = time()

interpolate_bool = False

kwargs = {'neurons': 3000, 'K': 3, 'rho': 0.2, 'H': 0, 'M': 10000, 'max_it': 10, 'error': 1e-3}

parallel = True

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)
beta_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
len_l = len(l_values)
len_b = len(beta_values)
n_pixels = len_l * len_b

if parallel:
    dl = 'P'
else:
    dl = 'S'

files = file_finder('MC2d', file_spec = f'_{dl}D_', **kwargs)

try:
    mattis_trials = np.load(files[0][:-1] + 'y')
except IndexError:
    mattis_trials = np.zeros((1, 9 * n_pixels))

samples = len(mattis_trials)

m_array_trials = mattis_trials.reshape((samples, len_l, len_b, 3, 3))
success_array = np.zeros((samples, len_l, len_b))

# print('\nCalculating success rates...')
cutoff = 0.85

t = time()

recovery_array = [[[recovered_pats(m_array_trials[idx_s, idx_l, idx_b], cutoff) for idx_b in range(len_b)] for idx_l in range(len_l)] for idx_s in range(samples)]

for idx_s in range(samples):
    for idx_l in range(len_l):
        for idx_b in range(len_b):
            if idx_s == 0 and idx_b == 24:
                print(m_array_trials[idx_s, idx_l, idx_b])
                print(recovery_array[idx_s][idx_l][idx_b])
                # pass
            how_many_pats = len(set([abs(index) for index in recovery_array[idx_s][idx_l][idx_b] if index is not None]))
            if how_many_pats == 3:
                success_array[idx_s, idx_l, idx_b] = 1

# print(m_array_trials[0,:,10,:,:])
# print(recovery_array[0][:][10][:])
# print(f'Calculated success rates in {round(time() - t, 2)} seconds.')

c = plt.imshow(np.transpose(np.average(success_array, axis = 0)), cmap='Greens', vmin = 0, vmax = 1, extent=[l_values[0], l_values[-1], beta_values[-1], beta_values[0]],
               aspect='auto', interpolation='nearest')
plt.colorbar(c)

plt.xlabel('$λ$')
plt.ylabel(f'$β$')
plt.title(f'N = {kwargs['neurons']}, K = {kwargs['K']}, ρ = {kwargs['rho']}, M = {kwargs['M']}, H = {kwargs['H']}\n{samples} sample(s), {cutoff} cutoff')


# tr_dets = []
alpha = 0
tr_dets = [lambda m: fp.disentangle_det(m, cutoff), lambda m: fp.tr_notdis_NoNsEx(m, cutoff)]
tr_arrays = [[] for det in tr_dets]

m_in, m_out, initial_q = fp.NoNs_initial(epsilon=0, pert_in=1e-10, pert_out=0)
initial_m = m_in

for file in file_finder('FP1d', file_spec='NoNsEx_lmb', initial_m = m_in, initial_q = initial_q,
                            rho = kwargs['rho'], H = kwargs['H'], alpha = alpha):
    with np.load(file) as data:
        for idx_det, tr_det in enumerate(tr_dets):
            idx_tr = fp.FindTransition(data['m'], tr_det=tr_det)
            if idx_tr > 0:
                tr_arrays[idx_det].append([data['lmb'][idx_tr], 1 / data['T']])

for tr_array in tr_arrays:
    try:
        trans_sorted = sorted(tr_array, key=lambda x: x[0])
        final_trans = list(map(list, zip(*trans_sorted)))
        l_array, b_array = final_trans[0], final_trans[1]

        if interpolate_bool:
            interpolator = make_interp_spline(l_array, b_array)
            l_values_smooth = np.linspace(start=l_array[0], stop=l_array[-1], num=500, endpoint=True)
            plt.plot(l_values_smooth, interpolator(l_values_smooth))
        else:
            plt.scatter(l_array, b_array)
    except IndexError:
        pass

print(f'Program ran in {round(time() - t0, 2)} seconds.')
plt.show()