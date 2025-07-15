import numpy as np
import UsainBolt as Ub
from MCfuncs import splitting_beta
from time import time
from multiprocessing import Pool

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

samples = 50
parallel = False

T_values = np.linspace(start = 0, stop = 0.2, num = 101, endpoint = True)
with np.errstate(divide='ignore'):
    beta_values = 1/T_values

len_beta= len(beta_values)

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
          'lmb': 0.08,
          'rho': 0.45,
          'dynamic': 'sequential',
          'sigma_type': 'mix'
          }

system = Ub.Experiment(splitting_beta, directory = 'MCData', **kwargs) # initialize experiment
system()
samples_array = system.samples_missing(samples)

if parallel:
    if __name__ == "__main__":
        with Pool() as pool:
            pool.map(system.run,samples_array)
elif not parallel:
    [system.run(s) for s in samples_array]

