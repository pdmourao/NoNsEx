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

samples = 50
interpolate_bool = True

x_arg = 'rho'
x_values = np.linspace(start = 0, stop = 0.3, num = 50, endpoint = False)

y_arg = 'lmb'
# y_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
y_values = np.linspace(start = 0, stop = 0.5, num = 50)

disable = False

kwargs = {'neurons': 5000,
          'K': 5,
          'beta': np.inf,
          'H': 0,
          'M': 50,
          'mixM': 0,
          'max_it': 100,
          'error': 1,
          'av_counter': 1,
          'quality': [1, 1, 1],
          'dynamic': 'sequential',
          'sigma_type': 'mix',
          'noise_dif': True,
          'save_n': True,
          'save_int': True
          }

len_x = len(x_values)
len_y = len(y_values)

m_array_trials, n_array_trials, int_array_trials = MC2d(directory = 'MC2d', disable = disable, n_samples = samples, x_arg = x_arg,
                                      y_arg = y_arg, x_values = x_values, y_values = y_values, **kwargs)

graph = False
if graph:
    cutoff = 0.8
    # print(m_array_trials)

    fig, ax = plt.subplots(1)
    gridvec_toplot(ax, 'dis', m_array_trials, limx0 = x_values[0], limx1 = x_values[-1],
                   limy0 = y_values[0], limy1 = y_values[-1], cutoff = cutoff, beta = kwargs['beta'])

    if x_arg == 'lmb':
        x_arg = 'lambda'
    if y_arg == 'lmb':
        y_arg = 'lambda'

    ax.set_xlabel(rf'$\{x_arg}$')
    ax.set_ylabel(rf'$\{y_arg}$')

    beta_title = rf'\infty' if np.isinf(kwargs['beta']) else kwargs['beta']

    ax.set_title(rf'$\beta = {beta_title}$')
    plt.show()

    # print(np.max(int_array_trials, axis=0))
    fig, ax = plt.subplots(1)
    vec_for_imshow = np.transpose(np.flip(np.max(int_array_trials, axis=0), axis=-1))

    ax.imshow(vec_for_imshow, cmap='Reds', vmin=0, vmax=kwargs['max_it'], aspect='auto', interpolation='nearest',
                  extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]])

    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_ylim(y_values[0], y_values[-1])
    plt.show()

