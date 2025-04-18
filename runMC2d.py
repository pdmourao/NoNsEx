import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import MC2d, gridvec_toplot
from time import time

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

samples = 0
interpolate_bool = True

x_arg = 'rho'
x_values = np.linspace(start = 0, stop = 0.3, num = 50, endpoint = False)

y_arg = 'lmb'
# y_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
y_values = np.linspace(start = 0, stop = 0.5, num = 50)

disable = False

kwargs = {'neurons': 3000,
          'K': 3,
          'beta': 5,
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

len_x = len(x_values)
len_y = len(y_values)

m_array_trials, n_array_trials = MC2d(directory = 'MC2d', disable = disable, n_samples = samples, x_arg = x_arg,
                                      y_arg = y_arg, x_values = x_values, y_values = y_values, **kwargs)

cutoff = 0.95

fig, ax = plt.subplots(1)
gridvec_toplot(ax, 'dis', m_array_trials, 'rho', 'lmb', x_values, y_values, cutoff = cutoff, beta = kwargs['beta'], H = kwargs['H'])
plt.show()
