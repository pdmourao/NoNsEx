import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import MC2d_Lb, mags_id_old, gridvec_toplot
from time import time
import FPfuncs as fp
import os

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

samples = 1

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)
y_values = np.linspace(start = 25, stop = 1, num = 50, endpoint = False)[::-1]

disable = False

kwargs = {'neurons': 4000,
          'K': 15,
          'lmb': l_values,
          'beta': y_values,
          'rho': 0,
          'H': 0,
          'M': 1,
          'mixM': 0,
          'max_it': 10,
          'error': 0.002,
          'av_counter': 3,
          'quality': [1, 1, 1],
          'dynamic': 'sequential',
          'sigma_type': 'mix',
          'noise_dif': False,
          'save_n': True,
          'save_int': True
          }

len_l = len(l_values)
len_y = len(y_values)

m_array_trials, n_array_trials, int_array_trials = MC2d_Lb(directory = 'MC2d_Lb', disable = disable, n_samples = samples, **kwargs)

cutoff = 0.8

fig, ax = plt.subplots(1)
gridvec_toplot(ax, 'dis', m_array_trials, limx0 = l_values[0], limx1 = l_values[-1],
               limy0 = y_values[0], limy1 = y_values[-1], cutoff = cutoff, interpolate = 'y')
ax.invert_yaxis()
ax.set_xlabel(fp.arg_to_label['lmb'])
ax.set_ylabel(fp.arg_to_label['beta'])
ax.set_title(f'{fp.arg_to_label['rho']} = {kwargs['rho']}')
plt.show()


fig_int, ax_int = plt.subplots(1)
vec_for_imshow = np.transpose(np.flip(np.max(int_array_trials, axis=0), axis=-1))

ax_int.imshow(vec_for_imshow, cmap='Reds', vmin=0, vmax=10, aspect='auto', interpolation='nearest',
              extent=[l_values[0], l_values[-1], y_values[0], y_values[-1]])

ax_int.set_xlim(l_values[0], l_values[-1])
ax_int.set_ylim(y_values[0], y_values[-1])
ax_int.invert_yaxis()
plt.show()
