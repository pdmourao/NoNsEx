import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import MC2d, gridvec_toplot
from time import time
import FPfuncs as fp

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

x_arg = 'rho'
x_values = np.linspace(start = 0, stop = 0.3, num = 50, endpoint = False)

y_arg = 'lmb'
# y_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
y_values = np.linspace(start = 0, stop = 0.5, num = 50)

beta_values = [5,10]
M_values = [150,100]
c_values = [0.8, 0.9, 0.95]

disable = False

kwargs = {'neurons': 3000,
          'K': 3,
          'H': 0,
          'mixM': 0,
          'max_it': 30,
          'error': 0.002,
          'av_counter': 3,
          'quality': [1, 1, 1],
          'dynamic': 'parallel',
          'sigma_type': 'mix',
          'noise_dif': False,
          'save_n': False,
          'save_int': False
          }

len_x = len(x_values)
len_y = len(y_values)


fig, axs = plt.subplots(len(c_values), len(beta_values), squeeze = False)
for idx_c, cutoff in enumerate(c_values):
    for idx_b, beta_v in enumerate(beta_values):
        ax = axs[idx_c, idx_b]
        m_array_trials, n_array_trials, int_array_trials = MC2d(directory='MC2d', disable=disable, n_samples=0, x_arg=x_arg,
                                              y_arg=y_arg, x_values=x_values, y_values=y_values, beta = beta_v,
                                              M = M_values[idx_b], **kwargs)

        c = gridvec_toplot(ax, 'dis', m_array_trials, 0, 0.3, 0, 0.5, cutoff = cutoff,
                       beta = beta_v)

        fig.colorbar(c, ax = ax)

        if idx_b > 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel(f'${fp.arg_to_label[y_arg]}$')
        if idx_c < len(c_values) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel(f'${fp.arg_to_label[x_arg]}$')
        if idx_c == 0:
            ax.set_title(f'${fp.arg_to_label['beta']} = {beta_v}$')


plt.show()
