import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import SplittingExperiment as SE, gridvec_toplot
from time import time
from storage import npz_file_finder
import os

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

samples = 50
interpolate_bool = True

rho_values = np.linspace(start = 0.2, stop = 0, num = 200, endpoint = False)[::-1]

disable = False

kwargs = {'neurons': 3000,
          'K': 3,
          'rho_values': rho_values,
          'H': 0,
          'M': 50,
          'mixM': 0,
          'max_it': 30,
          'error': 0.001,
          'av_counter': 3,
          'quality': [1, 1, 1],
          'minlmb': 0,
          'minT': 1e-3,
          'dynamic': 'sequential',
          'sigma_type': 'mix',
          'suf': '_Tmax300_R100'
          }

m_array_trials_split, n_array_trials_split, int_array_trials_split, m_array_trials_notsplit, n_array_trials_notsplit, int_array_trials_notsplit = SE(disable = disable, n_samples = samples, **kwargs)

cutoff = 0.8
