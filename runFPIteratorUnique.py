import FPfuncs as fp
import numpy as np
from FPfields import HLH, NoNsEx_HL, m_in, initial_q_i, initial_q_o, initial_p_i, initial_p_o, pert_dis, pert_spur, HLH_onlyq
from matplotlib import pyplot as plt
from FPfuncs import recovered_pats

directory = None
disable = True
parallel = False

kwargs = {'beta': 10,
          'alpha': np.arange(0, 0.2, 0.01),
          'h': 0,
          'max_it': 2000,
          'errorbound': 1e-15,
          'error': 1e-10
          }

# get which argument is an array (if any)
x_arg = None
for key, value in kwargs.items():
    if key != 'h' and not np.isscalar(value):
        assert x_arg is None, 'Warning: multiple arrays given as inputs.'
        x_arg = key

kwargs_single = dict(kwargs)
kwargs_single['alpha'] = 0.1
output_single = fp.solve(HLH, 9/10, 1, 1, directory = directory, disable = disable, **kwargs_single)


if False:
    outputs=[fp.solve(HLH, 9/10, 1, 1, directory = directory, disable = disable, x_arg = x_arg, parallel_CPUs = parallel, **kwargs)]

    # fp.solve(field, *args_i, directory = directory, disable=disable, x_arg = x_arg, **kwargs)

    # x_arg = None

    if x_arg is not None:

        if x_arg == 'beta':
            x_values = 1 / kwargs[x_arg]
            x_arg = 'T'
        else:
            x_values = kwargs[x_arg]

        m_values = [output[1] for output in outputs]

        fig, ax = plt.subplots(1)

        exempt_from_title = [x_arg, 'max_it', 'ibound', 'error', 'alpha', 'H']

        m = m_values[0]
        x_array = x_values
        ax.scatter(x_array, m)


        ax.set_ylabel('$m$')
        ax.set_xlabel(f'${fp.arg_to_label[x_arg]}$')
        ax.set_ylim(0,1)
        # ax.spines['bottom'].set_position('center')
        # plt.axhline(y= - 1, color='black', linewidth = 2)
        # ax.legend()

        title_strings = []
        for key, value in kwargs.items():
            if key not in exempt_from_title:
                try:
                    title_strings.append(f'{fp.arg_to_label[key]} = {value}')
                except KeyError:
                        title_strings.append(f'{key} = {value}')

        ax.set_title(", ".join(title_strings))
    plt.show()

