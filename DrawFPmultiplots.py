import numpy as np
import FPfuncs as fp
from FPfields import NoNsEx, m_in, initial_q
from matplotlib import pyplot as plt

pick_in = False
pert_eps =1e-8

plot_arg = 'rho'
plot_values = [0.05,0.1]

x_arg = 'beta'
label_arg = 'lmb'

x_values = 1/np.linspace(0.01, 1, 100, endpoint = True)
label_values = np.arange(start = 0, stop = 0.3, step = 0.05)

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

if pick_in:
	args = m_in() + pert_eps*pert_in, initial_q
	title_str = 'Input'
else:
	args = m_in(4/10) + pert_eps*pert_out, initial_q
	title_str = 'Output'

fig, axs = plt.subplots(len(plot_values), squeeze = False)

for ax, value_p in zip(axs.flat, plot_values):

	for value in label_values:

		inputs = {**others, x_arg: x_values, label_arg: value, plot_arg: value_p}
		m, q, n = fp.solve(NoNsEx, *args, use_files=True, disable=False, **inputs)

		idx_tr = len(m)
		for idx_m, m_value in enumerate(m):
			if np.std(np.diag(m_value)) > 1e-5 or (np.mean(np.diag(m_value)) > 0.5 and pick_in):
				idx_tr = idx_m
				break
		# color = next(ax._get_lines.prop_cycler)['color']
		line = ax.plot(1/x_values[:idx_tr], m[:idx_tr, 0, 0], label=rf'$\lambda$ = {round(value,3)}')
		color = line[0].get_color()
		if len(m) > idx_tr > 0:
			ax.vlines(x = 1/((x_values[idx_tr]+x_values[idx_tr - 1])/2), ymin = 0, ymax = m[idx_tr-1,0,0], color = color,
					   linestyle = 'dashed')
	title_arg = None
	for arg in ['beta', 'lmb', 'rho']:
		if arg not in [x_arg, label_arg]:
			title_arg = arg

	ax.legend()
	if x_arg == 'beta':
		ax.set_xlim(1/x_values[0], 1/x_values[-1])
		x_arg = 'T'
	else:
		ax.set_xlim(x_values[0],x_values[-1]+x_values[1]-x_values[0])

	ax.set_ylabel('$m$')
	ax.set_xlabel(rf'${x_arg}$')
	ax.label_outer()

	ax.set_title(rf'$\{plot_arg} = {value_p}$')

	if x_arg == 'T':
		x_arg = 'beta'

plt.show()

