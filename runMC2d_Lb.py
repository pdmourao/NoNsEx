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

samples = 10

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)
y_values = np.linspace(start = 25, stop = 1, num = 50, endpoint = False)[::-1]
# y_values = np.linspace(start = 0, stop = 0.2, num = 50)

flip_yaxis = True
disable = False

kwargs = {'neurons': 5000,
          'K': 5,
          'lmb': l_values,
          'beta': y_values,
          'rho': 0.05,
          'H': 0,
          'M': 150,
          'mixM': 0,
          'max_it': 30,
          'error': 0.002,
          'av_counter': 3,
          'quality': [1, 1, 1],
          'dynamic': 'parallel',
          'sigma_type': 'mix',
          'noise_dif': False,
          'save_n': False
          }

len_l = len(l_values)
len_y = len(y_values)

m_array_trials, n_array_trials = MC2d_Lb(directory = 'MC2d_Lb', disable = disable, n_samples = samples, **kwargs)

cutoff = 0.9

fig, ax = plt.subplots(1)
gridvec_toplot(ax, 'dis', m_array_trials, 'lmb', 'beta', l_values, y_values, cutoff = cutoff, rho = 50, H = 0, interpolate = 'y')
ax.yaxis.set_inverted(True)
ax.set_xlabel(fp.arg_to_label['lmb'])
ax.set_ylabel(fp.arg_to_label['beta'])
ax.set_title(f'{fp.arg_to_label['rho']} = {kwargs['rho']}')
plt.show()