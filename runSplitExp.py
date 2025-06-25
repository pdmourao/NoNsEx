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

samples = 100
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

cutoff = 0.75
all_samples = len(m_array_trials_split)
success_array_split = np.zeros((all_samples, len_rho))
success_array_notsplit = np.zeros((all_samples, len_rho))

x_axes_split = []
x_axes_notsplit = []

m_ps_split = []
m_ps_notsplit = []

m_stds_split = []
m_stds_notsplit = []

axis_rho = rho_values/3

for idx_rho, rho in enumerate(axis_rho):
    mags_split = []
    mags_notsplit = []
    for idx_s in range(all_samples):
        if mags_id('dis', m_array_trials_split[idx_s, idx_rho], cutoff):
            success_array_split[idx_s, idx_rho] = 1
            mags_split.append(np.sort(m_array_trials_split[idx_s, idx_rho], axis=None)[-3:])
        if mags_id('dis', m_array_trials_notsplit[idx_s, idx_rho], cutoff):
            success_array_notsplit[idx_s, idx_rho] = 1
            mags_notsplit.append(np.sort(m_array_trials_notsplit[idx_s, idx_rho], axis=None)[-3:])

    if len(mags_split) > 0:
        m_ps_split.append(np.mean(mags_split))
        m_stds_split.append(np.std(mags_split))
        x_axes_split.append(rho)

    if len(mags_notsplit) > 0:
        m_ps_notsplit.append(np.mean(mags_notsplit))
        m_stds_notsplit.append(np.std(mags_notsplit))
        x_axes_notsplit.append(rho)

success_av_split = np.average(success_array_split, axis=0)
success_av_notsplit = np.average(success_array_notsplit, axis=0)

int_split_av = np.average(int_array_trials_split, axis=0)
int_notsplit_av = np.average(int_array_trials_notsplit, axis=0)

colors = ['blue', 'green']

fig_success, ax_success = plt.subplots(1)
ax_success.plot(axis_rho, success_av_split, color = colors[0])
ax_success.plot(axis_rho, success_av_notsplit, color = colors[1])
ax_success.set_xlim(axis_rho[0], axis_rho[-1])
ax_success.set_ylim(0,1)

ax_success.set_xlabel(r'$\rho$')

ax_success.set_title('Rates of disentanglement')

plt.show()

fig_mags, ax_mags = plt.subplots(1)
ax_mags.errorbar(x = x_axes_split, y = m_ps_split, yerr = m_stds_split, label=f'split', color = colors[0], fmt = 'none')
ax_mags.errorbar(x = x_axes_notsplit, y = m_ps_notsplit, yerr = m_stds_notsplit, label=f'not split', color = colors[1], fmt = 'none')
ax_mags.set_xlim(axis_rho[0], axis_rho[-1])
ax_mags.set_ylim(0,1)

ax_mags.set_xlabel(r'$\rho$')
ax_mags.set_ylabel('$m$')

ax_mags.set_title('Recovered magnetizations')

plt.show()


# plt.plot(rho_values, int_split_av, color = 'yellow')
# plt.plot(rho_values, int_notsplit_av, color = 'green')
# plt.show()
