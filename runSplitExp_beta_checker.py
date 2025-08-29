import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import splitting_beta
from MCfuncsCopy import splitting_beta as sb
import UsainBolt as Ub
from time import time
from MCclasses import HopfieldMC as hop
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

kwargs_new = {'neurons': 5000,
          'k': 5,
          'beta_values': beta_values,
          'h_norm': 0,
          'm': 150,
          'supervised': True,
          'max_it': 30,
          'error': 0.005,
          'av_counter': 3,
          'lmb': 0.07,
          'r': np.sqrt(1 / (0.15 * 150 + 1)),
          'dynamic': 'sequential',
              'layers': 3
          }

system = Ub.Experiment(splitting_beta, directory = 'MCData', **kwargs) # initialize experiment
print(f'There are {len(system.samples_present())} samples present.')
m_split, n_split, its_split, m_notsplit, n_notsplit, its_notsplit = system.read() # read samples

sample = 15
entropy = (system.entropy, sample)

len_beta = len(beta_values)

inputs_sys = dict(kwargs)
inputs_sys.pop('beta_values')
inputs_sys.pop('max_it')
inputs_sys.pop('av_counter')
inputs_sys.pop('error')
inputs_sys.pop('H')
inputs_sys.pop('dynamic')
inputs_sys_notsplit = dict(inputs_sys)
inputs_sys_notsplit['M'] = 3*50
inputs_sys_notsplit['rho'] = 0.45/3
inputs_sim = {'max_it': 30, 'error': 0.005, 'av_counter': 3, 'H': 0, 'dynamic': 'sequential'}



rng_seeds = np.random.SeedSequence(entropy).spawn(2)
rng_seeds_split = rng_seeds[0].spawn(len_beta)
rng_seeds_notsplit = rng_seeds[1].spawn(len_beta)


split = hop(rngSS=rng_seeds[0], noise_dif=True, **inputs_sys)
inputs_sys_notsplit['K'] = split.pat
jointex = np.full(shape=(split.L, inputs_sys_notsplit['M'], split.K, split.N), fill_value=np.concatenate(tuple(layer for layer in split.ex)))
notsplit = hop(rngSS=rng_seeds[1], ex=jointex, noise_dif=False, **inputs_sys_notsplit)

beta = beta_values[0]
beta_idx = 0
m1_split, n1_split, its1_split = split.simulate(beta=beta,sim_rngSS=rng_seeds_split[beta_idx], cut=True, av=True, **inputs_sim)
m1_notsplit, n1_notsplit, its1_notsplit = notsplit.simulate(beta=beta, sim_rngSS=rng_seeds_notsplit[beta_idx], cut=True, av=True, **inputs_sim)

print(m1_split)
print(m_split[sample, 0])
print(np.array_equal(m1_split, m_split[sample,0]))
print(np.array_equal(n1_split, m_split[sample,0]))
print(np.array_equal(its_split, m_split[sample,0]))
print(np.array_equal(m1_notsplit, m_notsplit[sample,0]))
print(np.array_equal(m1_notsplit, m_notsplit[sample,0]))
print(np.array_equal(m1_notsplit, m_notsplit[sample,0]))


checker0 = (np.array([m_notsplit[sample], m_split[sample]]),
            np.array([n_notsplit[sample], n_split[sample]]),
            np.array([its_notsplit[sample], its_split[sample]]))

output = sb(entropy = entropy, checkers = checker0, disable = True, **kwargs_new)
