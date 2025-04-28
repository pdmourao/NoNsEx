from MCfuncs import MC2d_Lb, mags_id
import numpy as np
from matplotlib import pyplot as plt
from FPfields import NoNsEx, m_in, initial_q
import FPfuncs as fp

samples = 30
disable = False
colors = ['green', 'orange', 'red']


x_arg = 'beta'
kwargs = {'beta': 1/np.linspace(0.01, 1, 100, endpoint=True)[::-1],
          'rho': 0.05,
          'lmb': 0.1,
          'H': 0,
}

kwargs_MC = {'neurons': 5000,
             'K': 5,
             'M': 150,
             'mixM': 0,
             'max_it': 30,
             'error': 0.002,
             'av_counter': 3,
             'quality': [1, 1, 1],
             'dynamic': 'parallel',
             'sigma_type': 'mix',
             'noise_dif': False,
             'save_n': False,
             **kwargs
             }

if x_arg == 'lmb':
    kwargs_MC['beta'] = [kwargs_MC['beta']]
else:
    kwargs_MC['lmb'] = [kwargs_MC['lmb']]


m_arrays, n_arrays = MC2d_Lb(directory = 'MC1d_Lb', disable = disable, n_samples = samples, **kwargs_MC)

if x_arg == 'lmb':
    kwargs_MC['beta'] = [kwargs_MC['beta']]
    x_values = kwargs_MC[x_arg]
    m_arrays = m_arrays[:, :, 0]

elif x_arg == 'beta':
    kwargs_MC['lmb'] = [kwargs_MC['lmb']]
    x_values = 1 / kwargs_MC[x_arg][::-1]
    m_arrays = np.flip(m_arrays[:, 0], 1)

else:
    kwargs_MC['lmb'] = [kwargs_MC['lmb']]
    x_values = kwargs_MC[x_arg]
    m_arrays = m_arrays[:, 0]

len_x = len(x_values)

cutoff_dis = 0.9
cutoff_mix = 0.1

m_MC = np.zeros((len_x, 3))
rate_success_MC = np.zeros(len_x)

x_axes = [[] for color in colors]
m_ps = [[] for color in colors]

for idx_x, x in enumerate(x_values):
    mags = [[] for color in colors]

    successes = 0
    for idx_s in range(samples):
        this_m = m_arrays[idx_s, idx_x]
        this_diag = np.sort(np.diagonal(this_m))[::-1]
        if mags_id('dis', this_m, cutoff_dis):
            mags[0].append(np.sort(this_m, axis = None)[-3:])
            successes += 1
        elif mags_id('mix', this_m, cutoff_mix):
            mags[1].append(this_m)
        else:
            mags[2].append(this_m)

    for idx_m, mag in enumerate(mags):
        if len(mag) > 0:
            m_ps[idx_m].append(np.mean(mag))
            x_axes[idx_m].append(x)
    rate_success_MC[idx_x] = successes / samples


[plt.scatter(x_axes[idx], m_av, label=f'{idx}', color = colors[idx], s = 1) for idx, m_av in enumerate(m_ps)]
plt.plot(x_values, rate_success_MC, linestyle = 'dashed', color = 'black')


draw_FP = False
if draw_FP:
    field = NoNsEx

    kwargs_FP = {'alpha': 0, 'max_it': 1000, 'ibound': 1e-20, 'error': 1e-16}

    pert_matrix = np.array([[1,  0,  0],
                            [0,  0,  0],
                            [0,  0, -1]])

    pert = 1e-8*pert_matrix
    if sigma_type == 'dis':
        pert_matrix = np.array([[1, 0, 0],
                                [0, 0, 0],
                                [0, 0, -1]])

        pert = 1e-8 * pert_matrix
        initial_m = m_in(kwargs_MC['quality'][0] - 1/2)
    elif sigma_type == 'mix':
        pert_matrix = np.array([[ 1,-1,-1],
                                [-1, 1,-1],
                                [-1,-1, 1]])

        pert = 1e-8 * pert_matrix
        initial_m = m_in()

    args = initial_m+pert, initial_q

    m, q, n = fp.solve(field, *args, use_files = True, disable = False, **kwargs_FP, **kwargs)

    idx_tr = fp.FindTransitionFromVec(vec_m = m, tr_det = fp.tr_det_NoNsEx)

    plt.plot(x_values[:idx_tr], m[:idx_tr, 0, 0], color=colors[-1], linestyle = 'dashed')
    [plt.plot(x_values[idx_tr:], m[idx_tr:, i, i], color = colors[i], linestyle = 'dashed') for i in range(3)]

    plt.vlines(x = (x_values[idx_tr-1]+x_values[idx_tr-1])/2, ymin = 0, ymax = 1, color = 'grey', linestyle = 'dashed')

plt.title(f'{kwargs_MC['neurons']} neurons, K = {kwargs_MC['K']}\nrho = {kwargs['rho']}, lmb = {kwargs['lmb']}, {samples} sample(s)')

plt.show()

