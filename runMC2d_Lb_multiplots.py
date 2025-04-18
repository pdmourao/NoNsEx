import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import MC2d_Lb, gridvec_toplot
from time import time
import FPfuncs as fp

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

x_arg = 'lmb'
x_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)

y_arg = 'beta'
y_values = np.linspace(start = 25, stop = 1, num = 50, endpoint = False)[::-1]


rho_values = [0.05]
c_values = [0.95, 0.99]

disable = False

kwargs = {'neurons': 5000,
          'K': 5,
          'beta': y_values,
          'lmb': x_values,
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


fig, axs = plt.subplots(len(rho_values), len(c_values), squeeze = False)
for idx_r, rho_v in enumerate(rho_values):
    for idx_c, cutoff in enumerate(c_values):
        ax = axs[idx_r,idx_c]
        m_array_trials, n_array_trials = MC2d_Lb(directory='MC2d_Lb', disable=disable, n_samples=0, rho = rho_v,
                                              **kwargs)

        c = gridvec_toplot(ax, 'dis', m_array_trials, x_arg, y_arg, 0, 0.5, 1, 25, cutoff = cutoff,
                           interpolate = 'y', rho = int(1000*rho_v), H = kwargs['H'])

        ax.set_ylim(1.5, 20)
        ax.yaxis.set_inverted(True)

        if idx_c > 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel(f'${fp.arg_to_label[y_arg]}$')
        if True:
            fig.colorbar(c, ax = ax)
        if idx_r < len(rho_values) - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel(f'${fp.arg_to_label[x_arg]}$')
        if idx_r == 0:
            ax.set_title(f'c = {cutoff}')


plt.show()
