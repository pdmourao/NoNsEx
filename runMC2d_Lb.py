import numpy as np
from matplotlib import pyplot as plt
from MCfuncs import MC2d_Lb, mags_id
from time import time
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

states = [f'{num+1}pats{signed}{inc}'
          for num in range(3)
          for signed in ['', '_signed']
          for inc in ['', '_inc']
          ] + ['mix', 'mix_signed', 'other']

len_l = len(l_values)
len_y = len(y_values)

m_array_trials, n_array_trials = MC2d_Lb(directory = 'MC2d_Lb', disable = disable, n_samples = samples, **kwargs)

print(m_array_trials)
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

try:
    tr_hessian_mix = np.fromfile(os.path.join('TransitionData',f'HessianMix_L_rho{int(1000*kwargs['rho'])}_H{int(1000*kwargs['H'])}'),np.float64)
    tr_hessian_mix = tr_hessian_mix.reshape((int(np.size(tr_hessian_mix)/2),2))

    tr_hessian_dis = np.fromfile(os.path.join('TransitionData', f'HessianDis_L_rho{int(1000*kwargs['rho'])}_H{int(1000*kwargs['H'])}'),np.float64)
    tr_hessian_dis = tr_hessian_dis.reshape((int(np.size(tr_hessian_dis)/2),2))

    print(tr_hessian_dis)
except FileNotFoundError:
    print('No Hessian transitions given.')
    pass



print(f'Calculated success rates in {time() - t} seconds.')
for idx, state in enumerate(states):
    vec_to_plot = vec_for_imshow[idx]
    if np.sum(vec_to_plot) > 0 and state in ['3pats', 'mix']:
        c = plt.imshow(vec_for_imshow[idx], cmap = 'Greens', vmin = 0, vmax = 1, aspect='auto', interpolation='nearest',
                       extent = [l_values[0], l_values[-1], y_min, y_max])

        plt.colorbar(c)

        plt.xlabel('$λ$')
        plt.ylabel(f'$β$')
        plt.title(f'N = {kwargs['neurons']}, K = {kwargs['K']}, ρ = {kwargs['rho']}, M = {kwargs['M']}, H = {kwargs['H']}\n{all_samples} sample(s), {cutoff} cutoff, {state}')

        try:
            plt.scatter(tr_hessian_mix[:,1],tr_hessian_mix[:,0], color = 'black')
            plt.scatter(tr_hessian_dis[:, 1], tr_hessian_dis[:, 0], color='black')
        except NameError:
            pass

        plt.show()

colordict = {'mix': 'Reds',
             '3pats': 'Greens'}

for idx, state in enumerate(states):
    vec_to_plot = vec_for_imshow[idx]

    if np.sum(vec_to_plot) > 0 and state in ['3pats', 'mix']:
        this_vec = vec_for_imshow[idx]
        this_vec[this_vec == 0] = None
        c = plt.imshow(this_vec, cmap = colordict[state], vmin = 0, vmax = 1, aspect='auto', interpolation='nearest',
                       extent = [l_values[0], l_values[-1], y_min, y_max])

        plt.colorbar(c)

plt.xlabel('$λ$')
plt.ylabel(f'$β$')
plt.title(f'N = {kwargs['neurons']}, K = {kwargs['K']}, ρ = {kwargs['rho']}, M = {kwargs['M']}, H = {kwargs['H']}\n{all_samples} sample(s), {cutoff} cutoff')

try:
    plt.scatter(tr_hessian_mix[:,1],tr_hessian_mix[:,0], color = 'black')
    plt.scatter(tr_hessian_dis[:, 1], tr_hessian_dis[:, 0], color='black')
except NameError:
    pass

plt.show()