import numpy as np
from storage import file_finder
from FPfields import m_in

directory = 'MC2d'
excluded = ['H', 'rho']


for file in file_finder(directory):
    new_inputs = {}
    with np.load(file) as data:
        print(file)
        for key in data:
            new_inputs[key] = data[key]
            if key not in excluded:
                print(key)
                print(data[key])
    new_inputs['quality'] = np.array([1, 1, 1])
    # np.savez(file[:-4] + 'A.npz', **new_inputs)


