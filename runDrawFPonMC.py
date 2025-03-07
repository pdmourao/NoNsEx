from scipy.interpolate import make_interp_spline
import numpy as np
from time import time
from matplotlib import pyplot as plt
import os
import csv
from storage import file_finder
from MCfuncs import recovered_pats
import FPfuncs as fp
from FPfields import m_in, initial_q


t0 = time()

directory = 'MC2d_old'
filename = 'MC2d_dnSD_Lb2500_1740393456.npz'

interpolate_bool = False

kwargs = {'neurons': 2000, 'K': 3, 'rho': 0.2, 'H': 0, 'M': 10000, 'max_it': 50, 'error': 0, 'av_counter': 5}

parallel = False
noise_dif = False

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)
beta_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
len_l = len(l_values)
len_b = len(beta_values)
n_pixels = len_l * len_b

if parallel:
    dl = 'P'
else:
    dl = 'S'

if noise_dif is True:
    nd = 'in'
else:
    nd = 'dn'

if filename != '':
    file = os.path.join(directory, filename)
else:
    file = file_finder(directory, file_spec = f'_{nd}{dl}Dnp_', **kwargs)

try:
    mattis_trials = np.load(file[:-1] + 'y')
except IndexError:
    mattis_trials = np.zeros((1, 9 * n_pixels))

samples = len(mattis_trials)

m_array_trials = mattis_trials.reshape((samples, len_l, len_b, 3, 3))
success_array = np.zeros((4, samples, len_l, len_b))

# print('\nCalculating success rates...')
cutoff = 0.8

t = time()

recovery_array = [[[recovered_pats(m_array_trials[idx_s, idx_l, idx_b], cutoff) for idx_b in range(len_b)] for idx_l in range(len_l)] for idx_s in range(samples)]

for idx_s in range(samples):
    for idx_l in range(len_l):
        for idx_b in range(len_b):
            how_many_pats = len(set([abs(index) for index in recovery_array[idx_s][idx_l][idx_b] if index is not None]))
            for missing in range(4):
                if how_many_pats == 3 - missing:
                    success_array[missing, idx_s, idx_l, idx_b] = 1

print(f'Calculated success rates in {time() - t} seconds.')
for missing in range(4):
    n_pats = 3 - missing
    c = plt.imshow(np.transpose(np.average(success_array[missing], axis = 0)), cmap = 'Greens', vmin = 0, vmax = 1,
                   extent=[l_values[0], l_values[-1], beta_values[-1], beta_values[0]], aspect='auto',
                   interpolation='nearest')

    plt.colorbar(c)

    plt.xlabel('$λ$')
    plt.ylabel(f'$β$')
    plt.title(f'N = {kwargs['neurons']}, K = {kwargs['K']}, ρ = {kwargs['rho']}, M = {kwargs['M']}, H = {kwargs['H']}\n{samples} sample(s), {cutoff} cutoff, {n_pats} pattern(s)')

    plt.show()

# print(m_array_trials[0,:,10,:,:])
# print(recovery_array[0][:][10][:])
# print(f'Calculated success rates in {round(time() - t, 2)} seconds.')

c = plt.imshow(np.transpose(np.average(success_array[0], axis = 0)), cmap='Greens', vmin = 0, vmax = 1, extent=[l_values[0], l_values[-1], beta_values[-1], beta_values[0]],
               aspect='auto', interpolation='nearest')
plt.colorbar(c)

plt.xlabel('$λ$')
plt.ylabel(f'$β$')
plt.title(f'N = {kwargs['neurons']}, K = {kwargs['K']}, ρ = {kwargs['rho']}, M = {kwargs['M']}, H = {kwargs['H']}\n{samples} sample(s), {cutoff} cutoff')



alpha = 0
tr_dets = [lambda m: fp.tr_notdis_NoNsEx(m, cutoff)]
tr_arrays = [[] for det in tr_dets]

pert_4 = np.array([[ 1, -1, -1],
                   [-1,  1, -1],
                   [-1, -1,  1]])

pert = 1e-8*pert_4

initial_m = m_in() + pert

for file in file_finder('FP1d', file_spec='NoNsEx', arr_0 = initial_m, arr_1 = initial_q,
                            rho = kwargs['rho'], H = kwargs['H'], alpha = alpha):
    with np.load(file) as data:
        for idx_det, tr_det in enumerate(tr_dets):
            idx_tr = fp.FindTransition(data['m'], tr_det=tr_det)
            if idx_tr > 0:
                tr_arrays[idx_det].append([data['lmb'][idx_tr], data['beta']])

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