import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import SplittingExperiment as SE, gridvec_toplot
from time import time
from MCfuncs import mags_id

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

samples = 0
interpolate_bool = True

rho_values = np.linspace(start = 0.5, stop = 0, num = 200, endpoint = False)[::-1]
len_rho= len(rho_values)

disable = False

kwargs = {'neurons': 3000,
          'K': 3,
          'rho_values': rho_values,
          'H': 0,
          'M': 50,
          'mixM': 0,
          'max_it': 30,
          'error': 0.002,
          'av_counter': 3,
          'quality': [1, 1, 1],
          'minlmb': 0.07,
          'minT': 1e-3,
          'dynamic': 'sequential',
          'sigma_type': 'mix',
          'suf': '_Tmax300_R100'
          }

m_array_trials_split, n_array_trials_split, int_array_trials_split, m_array_trials_notsplit, n_array_trials_notsplit, int_array_trials_notsplit = SE(disable = disable, n_samples = samples, **kwargs)

cutoff = 0.6
all_samples = len(m_array_trials_split)
success_array_split = np.zeros((all_samples, len_rho))
success_array_notsplit = np.zeros((all_samples, len_rho))

for idx_s in range(all_samples):
    for idx_rho in range(len_rho):
        if mags_id('dis', m_array_trials_split[idx_s, idx_rho], cutoff):
            success_array_split[idx_s, idx_rho] = 1
        if mags_id('dis', m_array_trials_notsplit[idx_s, idx_rho], cutoff):
            success_array_notsplit[idx_s, idx_rho] = 1

success_av_split = np.average(success_array_split, axis=0)
success_av_notsplit = np.average(success_array_notsplit, axis=0)

int_split_av = np.average(int_array_trials_split, axis=0)
int_notsplit_av = np.average(int_array_trials_notsplit, axis=0)

plt.plot(rho_values, success_av_split, color = 'yellow')
plt.plot(rho_values, success_av_notsplit, color = 'green')
plt.show()

plt.plot(rho_values, int_split_av, color = 'yellow')
plt.plot(rho_values, int_notsplit_av, color = 'green')
plt.show()

if False:
    for i in range(len(rho_values)):
        print(rho_values[i])
        print('split')
        print(m_array_trials_split[0,i])
        print('notsplit')
        print(m_array_trials_notsplit[0, i])