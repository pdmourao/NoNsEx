import numpy as np
from storage import file_finder
from FPfields import m_in

directory = 'FP1d'
excluded = ['m', 'q', 'n', 'arr_1', 'lmb', 'max_it', 'alpha', 'H']

start_min_inputs = [0, 4/10, 1/2]

for file in file_finder(directory, rho = 0.05):
    with np.load(file) as data:
        print(file)
        for key in data:
            if key not in excluded:
                if key == 'arr_0':
                    for epsilon in start_min_inputs:
                        if np.linalg.norm(data[key] - m_in(epsilon)) < 1e-3:
                            print('Perturbation of:')
                            print(data[key])
                            print('with')
                            print(data[key] - m_in(epsilon))
                else:
                    print(key)
                    print(data[key])


