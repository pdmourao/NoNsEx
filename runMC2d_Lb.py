import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import MC2d_Lb
from FPfuncs import recovered_pats
from time import time

t0 = time()

# Here we run the freqs function for varying temperatures and betas
# Then plot the corresponding color graph
# And save values to a file, as well as lambda and beta values used
# File name has all the inputs below
# Also includes number of pixels

# The pixels are the values of beta and l given in the arrays below l_values and beta_values

samples = 0

l_values = np.linspace(start = 0, stop = 0.5, num = 50, endpoint = False)
len_l = len(l_values)
beta_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
len_b = len(beta_values)

kwargs = {'neurons': 5000,
          'K': 5,
          'lmb': l_values,
          'beta': beta_values,
          'rho': 0.05,
          'H': 0,
          'M': 100,
          'max_it': 20,
          'error': 0.01,
          'av_counter': 5,
          'quality': [1, 1, 1],
          'dynamic': 'sequential',
          'sigma_type': 'mix',
          'noise_dif': False
          }


states = ['3 pats', '2 pats', '1 pat', 'positive mixes','with Nones or mixes']
m_array_trials = MC2d_Lb(n_samples = samples, **kwargs)
all_samples = len(m_array_trials)
success_array = np.zeros((len(states), all_samples, len_l, len_b))

print('\nCalculating success rates...')
cutoff = 0.7
cutoff_mix = 0.1

t = time()

recovery_array = [[[recovered_pats(m_array_trials[idx_s, idx_l, idx_b], cutoff, cutoff_mix) for idx_b in range(len_b)] for idx_l in range(len_l)] for idx_s in range(all_samples)]

for idx_s in range(all_samples):
    for idx_l in range(len_l):
        for idx_b in range(len_b):
            if idx_l == 1:
                print(f'Sample {idx_s+1}, beta = {beta_values[idx_b]}, lmb = {l_values[idx_b]}')
                print(m_array_trials[idx_s, idx_l, idx_b])
                print(recovery_array[idx_s][idx_l][idx_b])
            how_many_pats = len(set([abs(index) for index in recovery_array[idx_s][idx_l][idx_b] if index is not None and index != 4]))
            for missing in range(3):
                if how_many_pats == 3 - missing:
                    success_array[missing, idx_s, idx_l, idx_b] = 1
                    # print(missing)
            if all([index == 4 for index in recovery_array[idx_s][idx_l][idx_b]]):
                success_array[3, idx_s, idx_l, idx_b] = 1
                # print('3')
            elif all([index == 4 or index is None for index in recovery_array[idx_s][idx_l][idx_b]]):
                success_array[4, idx_s, idx_l, idx_b] = 1
                # print('4')
            if sum(success_array[:, idx_s, idx_l, idx_b]) < 1:
                # print(success_array[:, idx_s, idx_l, idx_b])
                pass


print(f'Calculated success rates in {time() - t} seconds.')
for idx, state in enumerate(states):
    c = plt.imshow(np.transpose(np.average(success_array[idx], axis = 0)), cmap = 'Greens', vmin = 0, vmax = 1,
                   extent=[l_values[0], l_values[-1], beta_values[-1], beta_values[0]], aspect='auto',
                   interpolation='nearest')

    plt.colorbar(c)

    plt.xlabel('$λ$')
    plt.ylabel(f'$β$')
    plt.title(f'N = {kwargs['neurons']}, K = {kwargs['K']}, ρ = {kwargs['rho']}, M = {kwargs['M']}, H = {kwargs['H']}\n{all_samples} sample(s), {cutoff} cutoff, {state}')

    plt.show()
