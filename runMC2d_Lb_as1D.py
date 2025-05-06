from MCfuncs import MC2d_Lb, mags_id
import numpy as np
from matplotlib import pyplot as plt
from FPfields import NoNsEx, m_in, initial_q
import FPfuncs as fp

samples = 90
disable = False
colors = ['orange', 'green', 'red']


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

cutoff_dis = 0.9
cutoff_mix = 0.3
gap_mix = 0.1

if x_arg == 'lmb':
    kwargs_MC['beta'] = [kwargs_MC['beta']]
else:
    kwargs_MC['lmb'] = [kwargs_MC['lmb']]

x_values = kwargs_MC[x_arg]
len_x = len(x_values)

m_arrays, n_arrays = MC2d_Lb(directory = 'MC1d_Lb', disable = disable, n_samples = samples, **kwargs_MC)

field = NoNsEx

kwargs_FP = {'alpha': 0, 'max_it': 1000, 'ibound': 1e-16, 'error': 1e-12}

pert_matrix = np.array([[1,  0,  0],
                        [0,  0,  0],
                        [0,  0, -1]])

pert = 1e-8*pert_matrix
if kwargs_MC['sigma_type'] == 'dis':
    pert_matrix = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, -1]])

    pert = 1e-8 * pert_matrix
    initial_m = m_in(kwargs_MC['quality'][0] - 1/2)

    f_tr = [fp.tr_det_NoNsEx]

elif kwargs_MC['sigma_type'] == 'mix':
    pert_matrix = np.array([[ 1,-1,-1],
                            [-1, 1,-1],
                            [-1,-1, 1]])

    pert = 1e-8 * pert_matrix
    initial_m = m_in()

    f_tr = [lambda m_entry: not fp.thresh_NoNsEx(m_entry, cutoff_dis), fp.tr_det_NoNsEx]

args = initial_m+pert, initial_q
m_FP, q, n = fp.solve(field, *args, use_files = True, disable = False, **kwargs_FP, **kwargs)


if x_arg == 'lmb':
    kwargs_MC['beta'] = [kwargs_MC['beta']]
    m_arrays = m_arrays[:, :, 0]

elif x_arg == 'beta':
    kwargs_MC['lmb'] = [kwargs_MC['lmb']]
    x_values = 1 / x_values[::-1]
    m_arrays = np.flip(m_arrays[:, 0], 1)
    m_FP = np.flip(m_FP, 0)

else:
    kwargs_MC['lmb'] = [kwargs_MC['lmb']]
    m_arrays = m_arrays[:, 0]

idx_trs = [0] + [fp.FindTransitionFromVec(vec_m = m_FP, tr_det = func) for func in f_tr] + [len_x]

print(m_arrays[0])

rate_success_MC = np.zeros(len_x)

x_axes = [[] for color in colors]
m_ps = [[] for color in colors]
m_stds = [[] for color in colors]

for idx_x, x in enumerate(x_values):
    mags = [[] for color in colors]

    successes = 0
    for idx_s in range(samples):
        this_m = m_arrays[idx_s, idx_x]
        this_diag = np.sort(np.diagonal(this_m))[::-1]
        if mags_id('dis', this_m, cutoff_dis):
            mags[1].append(np.sort(this_m, axis = None)[-3:])
            successes += 1
        elif mags_id('mix', this_m, cutoff_mix):
            mags[0].append(this_m)
        else:
            mags[2].append(np.max(np.abs(this_m)))

    for idx_m, mag in enumerate(mags):
        if len(mag) > 0:
            m_ps[idx_m].append(np.mean(mag))
            m_stds[idx_m].append(np.std(mag))
            x_axes[idx_m].append(x)
    rate_success_MC[idx_x] = successes / samples

print(m_FP)

for tr, idx_tr in enumerate(idx_trs[:-1]):
    if tr < 2:
        plt.plot(x_values[idx_tr:idx_trs[tr+1]], np.max(m_FP[idx_tr:idx_trs[tr+1]], axis = (1, 2)),
                 color=colors[tr], linestyle = 'dashed', linewidth = 0.5)

    if tr == 1:
        plt.vlines(x=(x_values[idx_tr - 1] + x_values[idx_tr - 1]) / 2, ymin=0, ymax=1, color='grey',
                   linestyle='dashed')
    plt.xlim(0,1)
    plt.ylabel('$m$')
    plt.xlabel('$T$')

if False:
    [plt.plot(x_values[idx_tr:], m[idx_tr:, i, i], color = colors[i], linestyle = 'dashed') for i in range(3)]


plt.title(rf'Disentangled magnetizations ($\rho = {kwargs['rho']}$, $\lambda = {kwargs['lmb']}$)')

plt.errorbar(x = x_axes[0], y = m_ps[0], yerr = m_stds[0], label=f'mix', color = colors[0], fmt = 'none')
plt.errorbar(x = x_axes[1], y = m_ps[1], yerr = m_stds[1], label=f'dis', color = colors[1], fmt = 'none')
# plt.scatter(x_axes[2], m_ps[2], label=f'others', color = colors[2], s = 1)

plt.show()

plt.plot(x_values, rate_success_MC, linestyle = 'dashed', color = 'black')
plt.xlabel('$T$')
plt.title(rf'Disentanglement frequency ($\rho = {kwargs['rho']}$, $\lambda = {kwargs['lmb']}$)')

plt.show()