import FPfuncs as fp
from FPfields import NoNsEx
import numpy as np
from matplotlib import pyplot as plt
from time import time

t = time()

m_in, m_out, initial_q = fp.NoNs_initial(epsilon = 0.1, pert_in = 0, pert_out = 0.01)

x_arg = 'T'
x_values = np.arange(1/100, 101/100, 1/100)

label_arg = 'lmb'
label_values = np.arange(0, 0.5, 0.1)

other_args = {'rho': 0.5, 'alpha': 0, 'H': 0}
args = dict(other_args)

if label_arg == 'beta':
    label_arg_old = 'beta'
    label_arg = 'T'
    label_values_old = label_values
    label_values = 1 / label_values
else:
    label_arg_old = label_arg
    label_values_old = label_values

for idx_label, label_value in enumerate(label_values):

    t0 = time()
    print(f'\nSolving {fp.greek[label_arg]} = {label_value}...')
    args[label_arg] = label_value

    m_array, q_array = fp.solver(initial_m = m_out, initial_q = initial_q, x_var = (x_arg, x_values),
                                 tr_det = lambda x, y: fp.tr_det_NoNsEx(y, 1e-5), field = NoNsEx,
                                 max_it = 1000, error = 1e-12, use_previous = 1, disable = True, **args)
    if len(np.shape(m_array)) < 3:
        print(f'No stable solutions found for {fp.greek[label_arg_old]} = {round(label_values_old[idx_label], 2)}.')

    else:
        m_len = len(m_array)
        x_tr = x_values[m_len - 1]
        line = plt.plot(x_values[:m_len], m_array[:, 0, 0],
                        label = f'{fp.greek[label_arg_old]} = {round(label_values_old[idx_label], 2)}')
        # Draws the vertical transition line (if there was one)
        if m_len < len(x_values):
            print(f'Found transition for {fp.greek[label_arg_old]} = {round(label_values_old[idx_label], 2)} at {x_arg} = '
                  f'{round(x_tr, 2)}.')
            plt.vlines(x = x_tr, ymin=0, ymax = m_array[-1, 0, 0], color=line[-1].get_color(),
                       linestyle = 'dashed')
        else:
            print(f'No transition found for {fp.greek[label_arg_old]} = {round(label_values_old[idx_label], 2)}.')

plt.ylabel('$m$')
plt.ylim(0, 1)
plt.xlabel(f'${fp.greek[x_arg]}$')
plt.xlim(0, max(x_values[-1], 1))
plt.legend()
# plt.title(", ".join([f'{fp.greek[key]} = {round(value, 2)}' for key, value in other_args.items()]))
plt.title(f'{fp.greek['rho']} = {round(other_args['rho'], 2)}')
t = time() - t
mts = np.floor(t / 60).astype(int)
print(f'Program ran in {round(t, 2)} seconds ({mts}m{t-60*mts}s).')
plt.show()