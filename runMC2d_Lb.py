import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import MC2d_Lb, mags_id
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
y_values = np.linspace(start = 20, stop = 0, num = 50, endpoint = False)[::-1]
# y_values = np.linspace(start = 0, stop = 0.2, num = 50)
len_y = len(y_values)

flip_yaxis = False


kwargs = {'neurons': 5000,
          'K': 5,
          'lmb': l_values,
          'beta': y_values,
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


states = ['3pats',
          '3pats_signed',
          '2pats',
          '2pats_signed',
          '1pats',
          '1pats_signed',
          'mix',
          'mix_signed',
          'other']

m_array_trials = MC2d_Lb(n_samples = samples, **kwargs)
all_samples = len(m_array_trials)
success_array = np.zeros((len(states), all_samples, len_l, len_y))

print('\nCalculating success rates...')
cutoff = 0.8
cutoff_mix = 0.1

t = time()


for idx_s in range(all_samples):
    for idx_l in range(len_l):
        for idx_y in range(len_y):
            state_rec = mags_id(m_array_trials[idx_s, idx_l, idx_y], cutoff, cutoff_mix)
            idx_state = states.index(state_rec)
            success_array[idx_state, idx_s, idx_l, idx_y] = 1

success_av = np.average(success_array, axis = 1)

if flip_yaxis:
    vec_for_imshow = np.transpose(success_av, [0, 2, 1])
    y_min, y_max = y_values[-1], y_values[0]
else:
    vec_for_imshow = np.transpose(np.flip(success_av, axis = -1), [0, 2, 1])
    y_min, y_max = y_values[0], y_values[-1]


print(f'Calculated success rates in {time() - t} seconds.')
for idx, state in enumerate(states):
    vec_to_plot = vec_for_imshow[idx]
    if np.sum(vec_to_plot) > 0:
        c = plt.imshow(vec_for_imshow[idx], cmap = 'Greens', vmin = 0, vmax = 1,
                       extent=[l_values[0], l_values[-1], y_min, y_max], aspect='auto',
                       interpolation='nearest')

        plt.colorbar(c)

        plt.xlabel('$λ$')
        plt.ylabel(f'$β$')
        plt.title(f'N = {kwargs['neurons']}, K = {kwargs['K']}, ρ = {kwargs['rho']}, M = {kwargs['M']}, H = {kwargs['H']}\n{all_samples} sample(s), {cutoff} cutoff, {state}')

        plt.show()
