import numpy as np
from storage import npz_file_finder
from FPfields import m_in, initial_q_LL

pick_in = False
pert_eps = 1e-8
pert_out = np.array([[ 1, -1, -1],
                     [ 1, -1, -1],
                     [ 1, -1, -1]])

pert_in = np.array([[ 1, -1, -1],
                    [-1,  1, -1],
                    [-1, -1,  1]])

if pick_in:
	args = m_in() + pert_eps*pert_in, initial_q_LL
	title_str = 'Input'
else:
	args = m_in(4/10) + pert_eps*pert_out, initial_q_LL
	title_str = 'Output'

directory = 'FP1d_old'
excluded = ['m', 'q', 'n', 'arr_1', 'lmb', 'alpha', 'H']

start_min_inputs = [0, 4/10, 1/2]
label_values = np.arange(start = 0, stop = 0.25, step = 0.05)
for file in npz_file_finder(directory, *args, lmb = np.linspace(0, 0.5, 100, endpoint = False), beta = 5, rho = label_values[0]):
    print('\n')
    with np.load(file) as data:
        print(file)
        for key in data:
            if key not in excluded:
                if key == 'arr_0':
                    pass
                else:
                    print(f'{key} = {data[key]}')


