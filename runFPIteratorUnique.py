import FPfuncs as fp
import numpy as np
from FPfields import HLH, NoNsEx_HL, m_in, initial_q_i, initial_q_o, initial_p_i, initial_p_o, pert_dis, pert_spur
from matplotlib import pyplot as plt
from FPfuncs import recovered_pats

field = NoNsEx_HL
directory = 'FP1d'
disable = True

eps = 1e-8

kwargs = {'lmb': 0,
          'beta': 10,
          'alpha': np.linspace(0, 0.5, 100, endpoint = False),
          'h': np.zeros(shape = 3),
          'max_it': 1000,
          'errorbound': 1e-12,
          'error': 1e-10
          }

kwargs_hop = {'beta': 10,
          'alpha': np.linspace(0, 0.5, 100, endpoint = False),
          'h': 0,
          'max_it': 1000,
          'errorbound': 0,
          'error': 1e-10
          }

# get which argument is an array (if any)
x_arg = None
for key, value in kwargs.items():
    if key != 'h' and not np.isscalar(value):
        assert x_arg is None, 'Warning: multiple arrays given as inputs.'
        x_arg = key
        x_values = [value, value]

args_o = m_in(4/10)+eps*pert_spur, initial_q_o, initial_p_o

m_hop, q_hop, p_hop = fp.solve(HLH, 9/10, 1, 1, directory = directory, disable = disable, x_arg = x_arg, **kwargs_hop)

m_out, q_out, p_out = fp.solve(field, *args_o, directory = directory, disable = disable, x_arg = x_arg, **kwargs)


args_i = m_in() + eps*pert_dis, initial_q_i, initial_p_i

m_in, q_in, p_in = fp.solve(field, *args_i, directory = directory, disable=disable, x_arg = x_arg, **kwargs)

if x_arg is None:
    x_values = [np.arange(m_in), np.arange(m_out)]
elif x_arg == 'beta':
    x_arg = 'T'
    x_values = [1/x_val for x_val in x_values]

m_values = [m_in, m_out]

graph = False

if graph:
    fig, axes = plt.subplots(2)

    exempt_from_title = [x_arg, 'max_it', 'ibound', 'error', 'alpha', 'H']

    for ax_idx, ax in enumerate(axes):
        m = m_values[ax_idx]
        x_array = x_values[ax_idx]
        for i in range(3):
            ax.scatter(x_array, m[:, i, i], label=f'm[{i},{i}]')


        ax.ylabel('$m$')
        ax.xlabel(f'${fp.arg_to_label[x_arg]}$')
        ax.ylim(0,1)
        # ax.spines['bottom'].set_position('center')
        # plt.axhline(y= - 1, color='black', linewidth = 2)
        ax.legend()

        title_strings = []
        for key, value in kwargs.items():
            if key not in exempt_from_title:
                try:
                    title_strings.append(f'{fp.arg_to_label[key]} = {value}')
                except KeyError:
                    title_strings.append(f'{key} = {value}')

        ax.title(", ".join(title_strings))
    plt.show()

