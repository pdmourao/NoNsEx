import FPfuncs as fp
import numpy as np
from FPfields import HLH, NoNsEx_HL, m_in, initial_q_i, initial_q_o, initial_p_i, initial_p_o, pert_dis, pert_spur
from matplotlib import pyplot as plt
from FPfuncs import recovered_pats

directory = None
disable = True
parallel = False

eps = 0

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
          'errorbound': 1e-12,
          'error': 1e-10
          }

# get which argument is an array (if any)
x_arg = None
for key, value in kwargs.items():
    if key != 'h' and not np.isscalar(value):
        assert x_arg is None, 'Warning: multiple arrays given as inputs.'
        x_arg = key

args_o = m_in(4/10)+eps*pert_spur, initial_q_o, initial_p_o
args_i = m_in() + eps*pert_dis, initial_q_i, initial_p_i
kwargs['alpha'] = 0.1
kwargs_hop['alpha'] = 0.1

outputs=[
    fp.solve(HLH, 9/10, 1, 1, directory = directory, disable = disable, **kwargs_hop),
    fp.solve(NoNsEx_HL, *args_o, directory = directory, disable = disable, **kwargs)
]
print(outputs[0])
print(output for output in outputs[1])

# fp.solve(field, *args_i, directory = directory, disable=disable, x_arg = x_arg, **kwargs)

# x_arg = None

if x_arg is not None:

    if x_arg == 'beta':
        x_values = 1 / kwargs[x_arg]
        x_arg = 'T'
    else:
        x_values = kwargs[x_arg]

    m_values = [output[1] for output in outputs]

    fig, axes = plt.subplots(len(outputs))

    exempt_from_title = [x_arg, 'max_it', 'ibound', 'error', 'alpha', 'H']

    for ax_idx, ax in enumerate(axes):
        m = m_values[ax_idx]
        x_array = x_values[ax_idx]
        for i in range(3):
            ax.scatter(x_array, m[:, i, i], label=f'm[{i},{i}]')


        ax.set_ylabel('$m$')
        ax.set_xlabel(f'${fp.arg_to_label[x_arg]}$')
        ax.set_ylim(0,1)
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

        ax.set_title(", ".join(title_strings))
    plt.show()

