import numpy as np
from time import time
from storage import npz_file_finder, mathToPython
from scipy.interpolate import make_interp_spline
import json
from MCclasses import HopfieldMC as hop

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

idx_s = 3


rho_values = np.linspace(start = 0.2, stop = 0, num = 200, endpoint = False)[::-1]
len_rho= len(rho_values)

idx_rho = 4
rho_split = rho_values[idx_rho]
rho_notsplit = rho_values[idx_rho]/3

array_dict = {'H': 0,
              'max_it': 30,
              'error': 0.001,
              'av_counter': 3,
              'dynamic': 'sequential'
              }

sys_kwargs_split = {'neurons': 3000,
              'K': 3,
              'M': 50,
              'mixM': 0,
              'quality': [1, 1, 1],
              'sigma_type': 'mix',
              }

others = {'minlmb': 0,
          'minT': 1e-3,
          'suf': '_Tmax300_R100'
          }


interpolatorL = make_interp_spline(*mathToPython('maxL'+others['suf'],'optpar'))
interpolatorT = make_interp_spline(*mathToPython('maxT'+others['suf'],'optpar'))

beta_split = 1/interpolatorT(rho_split) if interpolatorT(rho_split) > others['minT'] else np.inf
beta_notsplit = 1/interpolatorT(rho_notsplit) if interpolatorT(rho_notsplit) > others['minT'] else np.inf

lmb_split = max(interpolatorL(rho_split),others['minlmb'])
lmb_notsplit = max(interpolatorL(rho_notsplit),others['minlmb'])

print(f'Split system inputs: rho = {rho_split}, beta = {beta_split}, lmb = {lmb_split}')
print(f'Not split system inputs: rho = {rho_notsplit}, beta = {beta_notsplit}, lmb = {lmb_notsplit}')

sys_kwargs_notsplit = dict(sys_kwargs_split)
sys_kwargs_notsplit['M'] = 3 * sys_kwargs_split['M']

directory = 'ToSplit_Or_NotToSplit'

npz_files = npz_file_finder(directory = directory, prints = True, rho = rho_values, **array_dict, **sys_kwargs_split)

file_npz = npz_files[0]
file_m_split = file_npz[:-4] + f'_sample{idx_s}_m_split.npy'
file_n_split = file_npz[:-4] + f'_sample{idx_s}_n_split.npy'
file_ints_split = file_npz[:-4] + f'_sample{idx_s}_ints_split.npy'
file_m_notsplit = file_npz[:-4] + f'_sample{idx_s}_m_notsplit.npy'
file_n_notsplit = file_npz[:-4] + f'_sample{idx_s}_n_notsplit.npy'
file_ints_notsplit = file_npz[:-4] + f'_sample{idx_s}_ints_notsplit.npy'
file_json = file_npz[:-3] + 'json'
with open(file_json, mode="r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    entropy_from_os = int(data['entropy'])
entropy = (entropy_from_os, idx_s)
m_fromfile_split = np.load(file_m_split)[idx_rho]
m_fromfile_notsplit = np.load(file_m_notsplit)[idx_rho]
n_fromfile_split = np.load(file_n_split)[idx_rho]
n_fromfile_notsplit = np.load(file_n_notsplit)[idx_rho]
ints_fromfile_split = np.load(file_ints_split)[idx_rho]
ints_fromfile_notsplit = np.load(file_ints_notsplit)[idx_rho]

print(f'Split in the files ran to {ints_fromfile_split} iteration(s) and got magnetization:')
print(m_fromfile_split)
print(f'Not split in the files ran to {ints_fromfile_notsplit} iteration(s) and got magnetization:')
print(m_fromfile_notsplit)

rng_seeds1 = np.random.SeedSequence(entropy).spawn(2*len_rho)
rng_seeds2 = np.random.SeedSequence(entropy).spawn(2*len_rho)
rng1_split = rng_seeds1[2*idx_rho]
rng1_notsplit = rng_seeds1[2*idx_rho+1]
rng2_split = rng_seeds2[2*idx_rho]
rng2_notsplit = rng_seeds2[2*idx_rho+1]

t = time()

split = hop(rho = rho_split, lmb = lmb_split, rngSS = rng1_split, noise_dif = True, **sys_kwargs_split)

jointblur=np.concatenate(tuple(split.blur))
fullblur = np.full(shape = (split.L, sys_kwargs_notsplit['M'], split.K, split.N),
                    fill_value = jointblur)

sys_kwargs_notsplit['K'] = split.pat
notsplit = hop(rho = rho_notsplit, lmb = lmb_notsplit, rngSS=rng1_notsplit, noise_dif = False, blur=fullblur, **sys_kwargs_notsplit)

checker = True
if checker:
    m_checker_split, n_checker_split, ints_checker_split = split.simulate(beta = beta_split, cut = True, sim_rngSS = rng1_split.spawn(1)[0], **array_dict)
    m_checker_notsplit, n_checker_notsplit, ints_checker_notsplit = notsplit.simulate(beta = beta_notsplit, cut = True, sim_rngSS = rng1_notsplit.spawn(1)[0], **array_dict)

    checks = [np.array_equal(m_fromfile_split, np.average(m_checker_split, axis = 0)),
              np.array_equal(n_fromfile_split, np.average(n_checker_split, axis = 0)),
              ints_fromfile_split==ints_checker_split,
              np.array_equal(m_fromfile_notsplit,np.average(m_checker_notsplit, axis = 0)),
              np.array_equal(n_fromfile_notsplit, np.average(n_checker_notsplit, axis =0)),
              ints_fromfile_notsplit==ints_checker_notsplit]
    print(f'Checks: {all(checks)}')

array_dict['max_it'] = 100
array_dict['error'] = 0

m_new_split, n_new_split, ints_new_split = split.simulate(prints = True, beta = beta_split, cut = True, sim_rngSS = rng2_split.spawn(1)[0], **array_dict)
m_new_notsplit, n_new_notsplit, ints_new_notsplit = notsplit.simulate(prints = True, beta = beta_notsplit, cut = True, sim_rngSS = rng2_notsplit.spawn(1)[0], **array_dict)