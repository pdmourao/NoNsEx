import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import splitting_beta
import UsainBolt as Ub
from time import time
from MCfuncs import mags_id

t0 = time()


# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

T_values = np.linspace(start = 0, stop = 0.2, num = 101, endpoint = True)
with np.errstate(divide='ignore'):
    beta_values = 1/T_values

len_beta= len(beta_values)

disable = False

kwargs = {'neurons': 5000,
          'K': 5,
          'beta_values': beta_values,
          'H': 0,
          'M': 50,
          'mixM': 0,
          'max_it': 30,
          'error': 0.005,
          'av_counter': 3,
          'quality': [1, 1, 1],
          'lmb': 0.07,
          'rho': 0.45,
          'dynamic': 'sequential',
          'sigma_type': 'mix'
          }

system = Ub.Experiment(splitting_beta, directory = 'MCData', **kwargs) # initialize experiment
m_split, n_split, its_split, m_notsplit, n_notsplit, its_notsplit = system.read() # read samples

# Do the graphs
cutoff = 0.6
all_samples = len(m_split)
success_array_split = np.zeros((all_samples, len_beta))
success_array_notsplit = np.zeros((all_samples, len_beta))

x_axes_split = []
x_axes_notsplit = []

m_ps_split = []
m_ps_notsplit = []

m_stds_split = []
m_stds_notsplit = []

for idx_T, T in enumerate(T_values):
    mags_split = []
    mags_notsplit = []
    for idx_s in range(all_samples):
        if mags_id('dis', m_split[idx_s, idx_T], cutoff):
            success_array_split[idx_s, idx_T] = 1
            mags_split.append(np.sort(m_split[idx_s, idx_T], axis=None)[-3:])
        if mags_id('dis', m_notsplit[idx_s, idx_T], cutoff):
            success_array_notsplit[idx_s, idx_T] = 1
            mags_notsplit.append(np.sort(m_notsplit[idx_s, idx_T], axis=None)[-3:])

    if len(mags_split) > 0:
        m_ps_split.append(np.mean(mags_split))
        m_stds_split.append(np.std(mags_split))
        x_axes_split.append(T)

    if len(mags_notsplit) > 0:
        m_ps_notsplit.append(np.mean(mags_notsplit))
        m_stds_notsplit.append(np.std(mags_notsplit))
        x_axes_notsplit.append(T)

success_av_split = np.average(success_array_split, axis=0)
success_av_notsplit = np.average(success_array_notsplit, axis=0)

its_split_av = np.average(its_split, axis=0)
its_notsplit_av = np.average(its_notsplit, axis=0)

colors = ['blue', 'green']

fig_success, ax_success = plt.subplots(1)
ax_success.plot(T_values, success_av_split, color = colors[0])
ax_success.plot(T_values, success_av_notsplit, color = colors[1])
ax_success.set_xlim(T_values[0], T_values[-1])
ax_success.set_ylim(0,1)

ax_success.set_xlabel(r'$T$')

ax_success.set_title('Rates of disentanglement')

plt.show()

fig_mags, ax_mags = plt.subplots(1)
ax_mags.errorbar(x = x_axes_split, y = m_ps_split, yerr = m_stds_split, label=f'split', color = colors[0], fmt = 'none')
ax_mags.errorbar(x = x_axes_notsplit, y = m_ps_notsplit, yerr = m_stds_notsplit, label=f'not split', color = colors[1], fmt = 'none')
ax_mags.set_xlim(T_values[0], T_values[-1])
ax_mags.set_ylim(0,1)

ax_mags.set_xlabel(r'$T$')
ax_mags.set_ylabel('$m$')

ax_mags.set_title('Recovered magnetizations')

plt.show()


plt.plot(T_values, its_split_av, color = 'yellow')
plt.plot(T_values, its_notsplit_av, color = 'green')
plt.show()
