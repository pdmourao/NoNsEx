import numpy as np
import FPfuncs as fp
from FPfields import NoNsEx, m_in, initial_q_LL
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

interpolate_bool = True

x_arg = 'lmb'
y_arg = 'beta'
label_arg = 'rho'

x_values = np.linspace(0, 0.5, 100, endpoint = False)
# label_values = np.arange(start = 0, stop = 0.25, step = 0.05)
label_values = [0.05, 0.1]
y_values = np.array([1/0.8, 1.5, 2, 2.5, 3, 4, 5, 10, 20])

others = {'alpha': 0,
          'H': 0,
          'max_it': 1000,
          'ibound': 1e-12,
          'error': 1e-10}

pert_out = np.array([[ 1, -1, -1],
                     [ 1, -1, -1],
                     [ 1, -1, -1]])

pert_in = np.array([[ 1, -1, -1],
                    [-1,  1, -1],
                    [-1, -1,  1]])

args_in = m_in() + 1e-8*pert_in, initial_q_LL
args_out = m_in(4/10) + 1e-8*pert_out, initial_q_LL

plt.figure(figsize=(5.15, 5.15))
plt.clf()
plt.subplot(111)

ax = plt.gca()

inputs = {**others, x_arg: x_values}
for value_l in label_values:
	tr_x = [0]
	tr_y = [1]
	for value_y in y_values:

		inputs[label_arg] = value_l
		inputs[y_arg] = value_y
		m, q, n = fp.solve_old(NoNsEx, *args_out, use_files=True, disable=False, **inputs)

		idx_tr = 0
		for idx_m, m_value in enumerate(m):
			if np.std(np.diag(m_value)) > 1e-5:
				idx_tr = idx_m
				break
		if idx_tr > 0:
			tr_x.append((x_values[idx_tr]+x_values[idx_tr-1]/2))
			if y_arg == 'beta':
				tr_y.append(1/value_y)
			else:
				tr_y.append(value_y)
	if interpolate_bool:
		interpolator = make_interp_spline(tr_x, tr_y)
		x_values_smooth = np.linspace(start=tr_x[0], stop=tr_x[-1], num=500, endpoint=True)
		plt.plot(x_values_smooth, interpolator(x_values_smooth))
	else:
		plt.scatter(tr_x, tr_y, label=f'${fp.arg_to_label[label_arg]}$ = {round(value_l,2)}')


if y_arg == 'beta':
	y_arg = 'T'
plt.legend()
plt.ylabel(f'${fp.arg_to_label[y_arg]}$')
plt.xlabel(f'${fp.arg_to_label[x_arg]}$')
plt.xlim(x_values[0],x_values[-1]+x_values[1]-x_values[0])
plt.ylim(0,1)

plt.title(f'Input state phase diagram')
plt.show()
